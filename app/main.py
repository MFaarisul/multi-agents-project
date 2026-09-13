import json
import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from langchain_core.messages import HumanMessage

from app.core.trace import LLMTraceHandler
from app.graph import get_app_graph
from app.schemas import MessageRequest, MessageResponse
from app.tools.pdf_tools import OUTPUT_DIR

# App loggers propagate to the root logger, which uvicorn does not configure —
# without basicConfig only WARNING+ would surface via Python's last-resort
# handler. Guarded so a custom logging setup is never clobbered.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Enterprise Intelligence System")

# Ingested document images, mounted read-only into this container (mimics an
# S3 bucket prefix). Images are SERVED from here — never copied.
RAG_IMAGES_DIR = os.getenv("IMAGES_DIR", "/app/temp_storage/rag_images")


def _image_urls(names: list[str]) -> list[str]:
    return [f"/api/v1/rag-images/{os.path.basename(n)}" for n in names or []]


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

NODE_NAMES = {
    "supervisor",
    "smalltalk",
    "qa_rag",
    "supervisor_evaluator",
    "text_to_sql",
    "synthesis",
}


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/rag-images/{filename}")
async def get_rag_image(filename: str):
    """Serve an ingested document image straight from the rag_images folder."""
    safe = os.path.basename(filename)
    if not safe.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG images can be served.")
    path = os.path.join(RAG_IMAGES_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/png", filename=safe)


@app.get("/api/v1/reports/{filename}")
async def download_report(filename: str):
    """Download a generated PDF report (persisted on the host volume)."""
    # Path-traversal guard: strip any directory components.
    safe = os.path.basename(filename)
    if not safe.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF reports can be downloaded.")
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="application/pdf", filename=safe)


@app.post("/api/v1/chat/stream")
async def stream_query(payload: MessageRequest):
    user_query = payload.message
    thread_id = payload.thread_id

    # Per-request execution trace (stages + LLM calls + tools) -> docker logs
    trace = LLMTraceHandler()
    t0 = time.perf_counter()

    # LangGraph execution context tied to persistent thread_id
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [trace]}

    # Append new turn to conversation history
    # Per-turn reset: stale values from earlier turns on this thread must not
    # leak into this response (e.g. executed_sql from a previous SQL question).
    input_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "thread_id": thread_id,
        "output_mode": "concise_text",
        "requires_sql_data": False,
        "requires_rag_context": False,
        "sql_error_count": 0,
        "business_rules": None,
        "retrieved_docs": None,
        "retrieved_images": [],
        "retrieved_data": None,
        "executed_sql": None,
        "pdf_file_path": None,
        "generated_charts": None,
    }

    graph = await get_app_graph()

    async def event_generator():
        try:
            async for event in graph.astream_events(
                input_state, config=config, version="v2"
            ):
                event_type = event.get("event")
                node_name = event.get("name")

                if (
                    event_type == "on_chain_start"
                    and node_name in NODE_NAMES
                ):
                    yield f"data: {json.dumps({'status': 'node_started', 'node': node_name, 'thread_id': thread_id})}\n\n"
                elif event_type == "on_chain_end" and node_name == "qa_rag":
                    images = (event.get("data", {}).get("output") or {}).get(
                        "retrieved_images"
                    ) or []
                    if images:
                        yield f"data: {json.dumps({'status': 'images', 'images': _image_urls(images)})}\n\n"
                elif event_type == "on_chain_end" and node_name in ("synthesis", "smalltalk"):
                    output = event.get("data", {}).get("output", {})
                    yield f"data: {json.dumps({'status': 'completed', 'result': output})}\n\n"
            trace.log_summary(time.perf_counter() - t0)
        except Exception as e:
            logger.exception("SSE stream failed")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/chat/invoke", response_model=MessageResponse)
async def invoke_query(payload: MessageRequest):
    """Non-streaming convenience endpoint that returns the final state."""
    user_query = payload.message
    thread_id = payload.thread_id

    trace = LLMTraceHandler()
    t0 = time.perf_counter()

    config = {"configurable": {"thread_id": thread_id}, "callbacks": [trace]}
    # Per-turn reset: stale values from earlier turns on this thread must not
    # leak into this response (e.g. executed_sql from a previous SQL question).
    input_state = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "thread_id": thread_id,
        "output_mode": "concise_text",
        "requires_sql_data": False,
        "requires_rag_context": False,
        "sql_error_count": 0,
        "business_rules": None,
        "retrieved_docs": None,
        "retrieved_images": [],
        "retrieved_data": None,
        "executed_sql": None,
        "pdf_file_path": None,
        "generated_charts": None,
    }

    graph = await get_app_graph()
    try:
        result = await graph.ainvoke(input_state, config=config)
        trace.log_summary(time.perf_counter() - t0)
        return MessageResponse(
            status="completed",
            thread_id=thread_id,
            final_response=result.get("final_response"),
            output_mode=result.get("output_mode"),
            pdf_file_path=result.get("pdf_file_path"),
            executed_sql=result.get("executed_sql"),
            images=_image_urls(result.get("retrieved_images") or []) or None,
        )
    except Exception:
        logger.exception("invoke_query failed")
        return JSONResponse(
            {"status": "error", "message": "Internal error - see server logs"},
            status_code=500,
        )
