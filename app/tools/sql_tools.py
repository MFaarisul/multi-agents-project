import json
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from app.core.config import settings

engine = create_engine(
    settings.readonly_db_uri,
    isolation_level="AUTOCOMMIT",
    connect_args={
        "options": "-c default_transaction_read_only=on -c statement_timeout=5000"
    },
)

# Statements that must never reach the database. These are checked before the
# role-level guardrail so that tool-call errors are descriptive to the agent.
# Word-boundary matching avoids false positives on column names like
# `updated_at` (which contains `update` as a substring).
_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "vacuum",
)
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE
)


class SQLExecutionInput(BaseModel):
    query: str = Field(description="Strict read-only SQL query to run.")


@tool("execute_read_only_sql", args_schema=SQLExecutionInput)
def execute_read_only_sql(query: str) -> str:
    """Executes SELECT queries against PostgreSQL database using read-only permissions."""
    clean_sql = query.strip().rstrip(";")

    lowered = clean_sql.lower()

    # Guardrail 1: only SELECT / WITH statements permitted.
    if not lowered.startswith(("select", "with")):
        return json.dumps(
            {"error": "Security Restriction: Only SELECT or WITH queries permitted."}
        )

    # Guardrail 2: reject mutating keywords as standalone words in the statement.
    match = _FORBIDDEN_RE.search(clean_sql)
    if match:
        kw = match.group(1)
        return json.dumps(
            {"error": f"Security Restriction: '{kw}' statements are not permitted."}
        )

    try:
        with engine.connect() as conn:
            result = conn.execute(text(clean_sql))
            rows = [dict(row._mapping) for row in result.fetchall()]
            return json.dumps({"status": "success", "data": rows[:100]}, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
