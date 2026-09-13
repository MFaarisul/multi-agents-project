import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set; multi-turn checkpoint test needs live Postgres",
)


def test_second_turn_remembers_first_turn():
    import uuid

    from langchain_core.messages import HumanMessage

    from app.graph import get_app_graph

    graph = get_app_graph()
    thread_id = f"test-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1
    graph.ainvoke(
        {
            "messages": [HumanMessage(content="Hello")],
            "user_query": "Hello",
            "thread_id": thread_id,
            "output_mode": "concise_text",
            "requires_sql_data": False,
            "requires_rag_context": False,
            "sql_error_count": 0,
        },
        config=config,
    )

    # Turn 2 — should be able to reference turn 1 via checkpoint history
    result = graph.ainvoke(
        {
            "messages": [HumanMessage(content="What did I just ask?")],
            "user_query": "What did I just ask?",
            "thread_id": thread_id,
            "output_mode": "concise_text",
            "requires_sql_data": False,
            "requires_rag_context": False,
            "sql_error_count": 0,
        },
        config=config,
    )

    # The checkpoint should retain at least the messages from both turns.
    state = graph.get_state(config)
    messages = state.values.get("messages", [])
    assert len(messages) >= 2
    human_contents = [m.content for m in messages if m.type == "human"]
    assert "Hello" in human_contents
    assert "What did I just ask?" in human_contents
