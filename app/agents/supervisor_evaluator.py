from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.state import SystemState
from app.core.structured_output import parse_json_model

llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.llm_api_key, base_url=settings.llm_base_url)


class EvaluatorDecision(BaseModel):
    """Structured decision produced by the supervisor evaluator LLM."""
    requires_sql_data: bool = Field(
        description=(
            "True if structured SQL data from the database is still needed to "
            "answer the user's query after reviewing the retrieved RAG context. "
            "False if the RAG context alone is sufficient."
        )
    )


EVALUATOR_PROMPT = """You are the Supervisor Evaluator in an Enterprise AI Analytics engine.

The Q&A / RAG agent has already retrieved unstructured corporate documents for the user's query.
Your job: decide whether the retrieved RAG context is SUFFICIENT to answer the user's query,
or whether structured SQL data from the database is still NEEDED.

Guidelines:
- If the query asks for numeric facts, aggregations, counts, revenue, lists of records,
  or anything that lives in the relational tables, set requires_sql_data=True.
- If the query is about definitions, policies, rules, glossary terms, or concepts
  that the retrieved documents already cover, set requires_sql_data=False.
- If the query asks to SEE a document figure, chart, or diagram (e.g. "show me
  the org chart"), set requires_sql_data=False — figures are delivered as
  retrieved images, not database rows.

User query:
{query}

Retrieved RAG context:
{rag_context}

Respond with ONLY this JSON object, no other text, fields in this exact order:
{{"requires_sql_data": false}}
"""


async def node(state: SystemState) -> dict:
    """Re-evaluate after RAG whether SQL is still needed (LLM-driven)."""
    query = state["user_query"]
    rag_context = state.get("business_rules") or "None"

    prompt = EVALUATOR_PROMPT.format(query=query, rag_context=rag_context)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    decision = parse_json_model(response.content, EvaluatorDecision)

    return {
        "requires_sql_data": decision.requires_sql_data,
    }


def route(state: SystemState) -> str:
    """Conditional-edge router used by the graph after this node runs."""
    return "text_to_sql" if state.get("requires_sql_data") else "synthesis"
