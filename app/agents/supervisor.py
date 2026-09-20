from __future__ import annotations

from langchain_core.messages import SystemMessage, trim_messages
from langchain_openai import ChatOpenAI

from app.core.state import SystemState
from app.core.config import settings
from app.schemas.decisions import SupervisorDecision


llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)
structured_llm = llm.with_structured_output(SupervisorDecision, method="function_calling")


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
"""


async def supervisor_node(state: SystemState) -> dict:
    messages = trim_messages(
        state.get("messages", []),
        max_tokens=50,
        strategy="last",
        token_counter=len,
        start_on="human",
    )

    prompt = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    decision = await structured_llm.ainvoke(prompt)

    return {
        "requires_sql_data": decision.requires_sql_data,
        "requires_rag_context": decision.requires_rag_context,
        "output_mode": decision.output_mode,
    }


def route(state: SystemState) -> str:
    if state.get("requires_rag_context") or state.get("requires_sql_data"):
        return "qa_rag"
    return "smalltalk"
