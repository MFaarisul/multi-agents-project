from __future__ import annotations

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph

from app.agents import qa_rag, smalltalk, supervisor, supervisor_evaluator, synthesis, text_to_sql
from app.core.state import SystemState
from app.db.session import get_checkpoint_pool


async def build_agent_graph():
    builder = StateGraph(SystemState)

    # Register Nodes — 3 agents (smalltalk + qa_rag, text_to_sql, synthesis)
    # + 2 routers (supervisor, supervisor_evaluator)
    builder.add_node("supervisor", supervisor.supervisor_node)
    builder.add_node("smalltalk", smalltalk.node)
    builder.add_node("qa_rag", qa_rag.node)
    builder.add_node("supervisor_evaluator", supervisor_evaluator.node)
    builder.add_node("text_to_sql", text_to_sql.node)
    builder.add_node("synthesis", synthesis.node)

    # Entry Point
    builder.set_entry_point("supervisor")

    # supervisor ─► qa_rag (Context Phase)       if rag OR sql
    # supervisor ─► smalltalk (Direct Response)   if neither
    builder.add_conditional_edges("supervisor", supervisor.route)

    # qa_rag ─► supervisor_evaluator
    builder.add_edge("qa_rag", "supervisor_evaluator")

    # supervisor_evaluator ─► text_to_sql   if SQL_NEEDED
    # supervisor_evaluator ─► synthesis     if RAG_SUFFICIENT
    builder.add_conditional_edges("supervisor_evaluator", supervisor_evaluator.route)

    # text_to_sql ─► synthesis ─► END
    builder.add_edge("text_to_sql", "synthesis")
    builder.add_edge("synthesis", END)

    # smalltalk ─► END  (terminal — bypasses synthesis)
    builder.add_edge("smalltalk", END)

    # Compile with async Postgres Checkpointer for memory persistence
    # (AsyncPostgresSaver is required: the FastAPI endpoints drive the graph
    # via ainvoke / astream_events, which need async checkpoint methods)
    pool = get_checkpoint_pool()
    # Pool is created unopened (open=False, see app.db.session) — open it
    # explicitly; wait=True fails fast if the DB is unreachable.
    await pool.open(wait=True, timeout=30.0)
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    return builder.compile(checkpointer=checkpointer)


# Lazily built so importing the module doesn't require a live database.
app_graph = None


async def get_app_graph():
    global app_graph
    if app_graph is None:
        app_graph = await build_agent_graph()
    return app_graph
