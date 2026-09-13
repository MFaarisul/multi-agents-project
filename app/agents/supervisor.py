from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.state import SystemState
from app.core.config import settings
from app.core.structured_output import parse_json_model

llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.llm_api_key, base_url=settings.llm_base_url)


class SupervisorDecision(BaseModel):
    """Structured routing decision produced by the supervisor LLM."""
    requires_rag_context: bool = Field(
        description=(
            "True if unstructured document/glossary retrieval (RAG) is needed "
            "to answer the query."
        )
    )
    requires_sql_data: bool = Field(
        description=(
            "True if structured SQL data from the database is needed to "
            "answer the query."
        )
    )
    output_mode: Literal["concise_text", "markdown_table", "pdf_report"] = Field(
        description=(
            "concise_text for quick factual or conversational answers, "
            "markdown_table when the user wants data in a table or breakdown, "
            "pdf_report when the user explicitly asks for a report or detailed document."
        )
    )


SYSTEM_PROMPT = """You are the Supervisor Agent in an Enterprise AI Analytics engine.
Analyze the user's intent based on the query and conversation history.

Available capabilities:
- Document retrieval (RAG): corporate glossary, refund policy, marketing
  guidelines, governance policy, org structure, revenue report figures/charts
- SQL database: customers, subscriptions, orders, refunds, marketing campaigns

Decide:
1. Does this query need unstructured document/glossary context? (requires_rag_context)
2. Does this query need structured SQL data from the database? (requires_sql_data)
3. What output format suits the request?
   - concise_text: quick factual or conversational answers
   - markdown_table: when the user wants data in a table or breakdown
   - pdf_report: when the user explicitly asks for a "report" or detailed document

Note: SQL queries also benefit from RAG context (domain rules, glossary).
If neither is needed, it's a general Q&A question.
Greetings, small talk, tests ("test", "hello"), thanks, and general light
conversation are general Q&A: set both requires_rag_context and
requires_sql_data to False.

Respond with ONLY this JSON object, no other text, fields in this exact order:
{"requires_rag_context": false, "requires_sql_data": false, "output_mode": "concise_text"}
"""


async def supervisor_node(state: SystemState) -> dict:
    """Intent analysis via prompt-declared JSON output — no hardcoded keywords."""
    # token_counter=len counts MESSAGES, not tokens — this caps the prompt
    # at the last 20 conversation turns.
    trimmed_history = trim_messages(
        state["messages"],
        max_tokens=20,
        strategy="last",
        token_counter=len,
        start_on="human",
    )

    prompt = [SystemMessage(content=SYSTEM_PROMPT)] + trimmed_history
    response = await llm.ainvoke(prompt)
    decision = parse_json_model(response.content, SupervisorDecision)

    return {
        "requires_sql_data": decision.requires_sql_data,
        "requires_rag_context": decision.requires_rag_context,
        "output_mode": decision.output_mode,
    }


def route(state: SystemState) -> str:
    """Conditional-edge router after the supervisor.

    RAG or SQL queries enter the Context Phase (qa_rag) first.
    General Q&A / small talk goes to smalltalk (terminal).
    The supervisor never sends anything directly to text_to_sql; that
    decision belongs to the supervisor_evaluator after RAG retrieval.
    """
    if state.get("requires_rag_context") or state.get("requires_sql_data"):
        return "qa_rag"
    return "smalltalk"
