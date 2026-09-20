from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import SystemState
from app.schemas.decisions import EvaluatorDecision

llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.llm_api_key, base_url=settings.llm_base_url)


# Function-calling mode: the decision arrives on the tool_calls channel,
# immune to gateway-injected notice lines / reasoning text in content.
# (The gateway ignores response_format — langchain's default method.)
structured_llm = llm.with_structured_output(EvaluatorDecision, method="function_calling")


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
"""


async def node(state: SystemState) -> dict:
    """Re-evaluate after RAG whether SQL is still needed (LLM-driven)."""
    query = state["user_query"]
    rag_context = state.get("business_rules") or "None"

    prompt = EVALUATOR_PROMPT.format(query=query, rag_context=rag_context)
    decision = await structured_llm.ainvoke([SystemMessage(content=prompt)])

    return {
        "requires_sql_data": decision.requires_sql_data,
    }


def route(state: SystemState) -> str:
    """Conditional-edge router used by the graph after this node runs."""
    return "text_to_sql" if state.get("requires_sql_data") else "synthesis"
