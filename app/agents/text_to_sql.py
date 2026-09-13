from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import SystemState
from app.tools.sql_tools import execute_read_only_sql

llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.llm_api_key, base_url=settings.llm_base_url)

SCHEMA_PROMPT = """You are a Text-to-SQL agent. Given the user's question and the
PostgreSQL schema below, write a single read-only SQL query (SELECT or WITH only)
that answers it. Return ONLY the SQL, no markdown fences, no explanation.

Schema:
- customers(customer_id, full_name, email, country, segment, created_at)
- subscriptions(subscription_id, customer_id, plan_tier, monthly_fee, status, start_date, end_date)
- orders(order_id, customer_id, amount, order_status, created_at)
- refunds(refund_id, order_id, amount, reason, refunded_at)
- marketing_campaigns(campaign_id, campaign_name, channel, budget, spent, quarter, campaign_year)
- customer_events(event_id, customer_id, order_id, event_type, event_date, amount, details)
  -- Historical activity log. event_type IN ('signup', 'subscription_started',
  -- 'order_placed', 'refund_issued'). Use it for trend / time-series
  -- questions (e.g. revenue or signups by month) via date_trunc on event_date.
"""

CORRECTION_PROMPT = """The previous SQL query failed with this error:
{error}

Previous query:
{sql}

Rewrite the query to fix the error. Return ONLY the corrected SQL.
"""

MAX_RETRIES = 3

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_sql(text: str) -> str:
    match = _SQL_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip().rstrip(";")


async def node(state: SystemState) -> dict:
    """Generate SQL, execute read-only, and self-correct on error (up to MAX_RETRIES)."""
    user_query = state["user_query"]
    error_count = state.get("sql_error_count", 0)

    # 1. Generate initial SQL.
    messages = [{"role": "system", "content": SCHEMA_PROMPT},
                {"role": "user", "content": user_query}]
    ai_response = await llm.ainvoke(messages)
    sql = _extract_sql(ai_response.content)

    # 2. Execute + self-correct loop.
    for attempt in range(MAX_RETRIES):
        result_str = execute_read_only_sql.invoke({"query": sql})
        result = json.loads(result_str)

        if result.get("status") == "success":
            return {
                "executed_sql": sql,
                "retrieved_data": result.get("data", []),
                "sql_error_count": error_count,
            }

        # Error path
        error_count += 1
        error_msg = result.get("error") or result.get("message", "Unknown error")

        if attempt == MAX_RETRIES - 1:
            return {
                "executed_sql": sql,
                "retrieved_data": [],
                "sql_error_count": error_count,
            }

        # Ask the LLM to correct the query.
        correction_messages = [
            {"role": "system", "content": CORRECTION_PROMPT.format(error=error_msg, sql=sql)},
            {"role": "user", "content": user_query},
        ]
        ai_response = await llm.ainvoke(correction_messages)
        sql = _extract_sql(ai_response.content)

    return {
        "executed_sql": sql,
        "retrieved_data": [],
        "sql_error_count": error_count,
    }
