from __future__ import annotations

from functools import lru_cache

from psycopg_pool import AsyncConnectionPool

from app.core.config import settings


@lru_cache
def get_checkpoint_pool() -> AsyncConnectionPool:
    """Lazy singleton async connection pool used by the PostgresSaver checkpointer.

    Async is required because the FastAPI endpoints drive the graph via
    ainvoke / astream_events. autocommit=True is required because
    PostgresSaver.setup() runs CREATE INDEX CONCURRENTLY, which cannot
    execute inside a transaction block. The pool is created unopened
    (open=False) — it is opened explicitly in build_agent_graph(), which
    avoids the AsyncConnectionPool open-in-constructor deprecation.
    """
    pool = AsyncConnectionPool(
        conninfo=settings.checkpoint_pool_uri,
        max_size=10,
        kwargs={"autocommit": True},
        open=False,
    )
    return pool


async def close_checkpoint_pool() -> None:
    pool = get_checkpoint_pool()
    if not pool.closed:
        await pool.close()
    get_checkpoint_pool.cache_clear()
