"""Per-request execution tracing for the agent graph.

Emits one combined log line per graph node (total duration + chronological
breakdown of the LLM calls and tool runs inside it), plus a final arrow
summary:  input -> supervisor (2.31s) -> … -> output | total 17.59s

Usage: create one handler per request and pass it in the LangGraph config —
LangGraph propagates it to every node and nested LLM/tool run automatically.
"""

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("trace")

KNOWN_NODES = {
    "supervisor",
    "qa_rag",
    "supervisor_evaluator",
    "text_to_sql",
    "smalltalk",
    "synthesis",
}


def _model_name(serialized: dict[str, Any] | None, metadata: dict[str, Any] | None) -> str:
    model = ((serialized or {}).get("kwargs") or {}).get("model_name")
    if not model:
        model = ((serialized or {}).get("kwargs") or {}).get("model")
    if not model:
        model = (metadata or {}).get("ls_model_name")
    return str(model) if model else "llm"


class LLMTraceHandler(BaseCallbackHandler):
    """Collects node/LLM/tool timings for a single request and logs them.

    One instance per request keeps concurrent requests isolated.
    """

    def __init__(self) -> None:
        # run_id -> node run_id (nested runs inherit their node's run id)
        self._run_node: dict[str, str] = {}
        # node run_id -> {"name", "start", "ingredients": [str]}
        self._node_buf: dict[str, dict[str, Any]] = {}
        self._llm_start: dict[str, dict[str, Any]] = {}
        self._tool_start: dict[str, dict[str, Any]] = {}
        self.node_timings: list[tuple[str, float]] = []

    # -- helpers ----------------------------------------------------------

    def _node_rid_for(self, parent_run_id: UUID | str | None) -> str | None:
        if parent_run_id is None:
            return None
        return self._run_node.get(str(parent_run_id))

    # -- chains (nodes) ----------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | None,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        name = name or ((serialized or {}).get("name") or "")
        if name in KNOWN_NODES:
            self._node_buf[rid] = {"name": name, "start": time.perf_counter(), "ingredients": []}
            self._run_node[rid] = rid
        else:
            # Nested run (e.g. structured-output wrapper): inherit its node.
            node_rid = self._node_rid_for(parent_run_id)
            if node_rid:
                self._run_node[rid] = node_rid

    def on_chain_end(
        self,
        outputs: dict[str, Any] | None,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        buf = self._node_buf.pop(rid, None)
        if buf is None:
            self._run_node.pop(rid, None)
            return
        dur = time.perf_counter() - buf["start"]
        self.node_timings.append((buf["name"], dur))
        line = f"{buf['name']}: {dur:.2f}s"
        if buf["ingredients"]:
            line += " [" + " | ".join(buf["ingredients"]) + "]"
        logger.info("%s", line)
        self._run_node.pop(rid, None)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        rid = str(run_id)
        buf = self._node_buf.pop(rid, None)
        if buf is not None:
            logger.error("%s FAILED: %s", buf["name"], error)
        self._run_node.pop(rid, None)

    # -- LLM calls ----------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node_rid = self._node_rid_for(parent_run_id)
        if not node_rid or node_rid not in self._node_buf:
            return
        self._llm_start[str(run_id)] = {
            "start": time.perf_counter(),
            "model": _model_name(serialized, metadata),
            "node_rid": node_rid,
        }

    def on_llm_start(self, serialized: dict[str, Any] | None, prompts: list[str], **kwargs: Any) -> None:
        # Chat models route through on_chat_model_start; kept for coverage.
        self.on_chat_model_start(
            serialized,
            prompts,
            run_id=kwargs.get("run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            metadata=kwargs.get("metadata"),
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        info = self._llm_start.pop(str(run_id), None)
        if info is None:
            return
        dur = time.perf_counter() - info["start"]
        buf = self._node_buf.get(info["node_rid"])
        if buf is not None:
            buf["ingredients"].append(f"LLM {info['model']}: {dur:.2f}s")

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        info = self._llm_start.pop(str(run_id), None)
        if info is not None:
            buf = self._node_buf.get(info["node_rid"])
            if buf is not None:
                buf["ingredients"].append(f"LLM {info['model']} FAILED")

    # -- tools ---------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        node_rid = self._node_rid_for(parent_run_id)
        if not node_rid or node_rid not in self._node_buf:
            return
        name = run_name or ((serialized or {}).get("name") or "tool")
        self._tool_start[str(run_id)] = {
            "start": time.perf_counter(),
            "name": str(name),
            "node_rid": node_rid,
        }

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, run_name: str | None = None, **kwargs: Any) -> None:
        info = self._tool_start.pop(str(run_id), None)
        if info is None:
            return
        dur = time.perf_counter() - info["start"]
        buf = self._node_buf.get(info["node_rid"])
        if buf is not None:
            buf["ingredients"].append(f"TOOL {info['name']}: {dur:.2f}s")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        info = self._tool_start.pop(str(run_id), None)
        if info is not None:
            buf = self._node_buf.get(info["node_rid"])
            if buf is not None:
                buf["ingredients"].append(f"TOOL {info['name']} FAILED")

    # -- summary ---------------------------------------------------------------

    def log_summary(self, total_seconds: float) -> None:
        """Log the arrow summary line. Call after the graph run completes."""
        if not self.node_timings:
            return
        parts = [f"{name} ({secs:.2f}s)" for name, secs in self.node_timings]
        logger.info("input -> %s -> output | total %.2fs", " -> ".join(parts), total_seconds)
