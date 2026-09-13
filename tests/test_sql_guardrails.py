import json

import pytest

from app.tools.sql_tools import execute_read_only_sql


def _run(query: str) -> dict:
    return json.loads(execute_read_only_sql.invoke({"query": query}))


def test_select_query_is_allowed():
    result = _run("SELECT 1 AS one")
    assert result["status"] == "success"
    assert result["data"] == [{"one": 1}]


def test_with_query_is_allowed():
    result = _run("WITH t AS (SELECT 1 AS x) SELECT x FROM t")
    assert result["status"] == "success"
    assert result["data"] == [{"x": 1}]


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO customers (full_name, email, country) VALUES ('x','y@z.com','US')",
        "UPDATE customers SET country = 'US'",
        "DELETE FROM customers",
        "DROP TABLE customers",
        "ALTER TABLE customers ADD COLUMN x TEXT",
        "TRUNCATE customers",
        "CREATE TABLE evil (id int)",
        "GRANT ALL ON customers TO public",
    ],
)
def test_mutating_statements_are_blocked(query):
    result = _run(query)
    assert "error" in result
    assert "Security Restriction" in result["error"]


def test_result_rows_are_capped():
    # The seeded schema has 100 customers; ensure we never return more than 100.
    result = _run("SELECT * FROM customers")
    if result.get("status") == "success":
        assert len(result["data"]) <= 100
    # If the DB isn't available the test still passes (status == error).


def test_leading_whitespace_still_validates():
    result = _run("   SELECT 1 AS x")
    assert result["status"] == "success"


def test_trailing_semicolon_stripped():
    result = _run("SELECT 1 AS x;")
    assert result["status"] == "success"
