from __future__ import annotations

import json
import os

from app.core.state import SystemState
from app.tools.rag_tools import query_vector_store

# Adaptive image surfacing: keep up to IMAGE_MAX image hits, dropping any
# whose dense cosine distance exceeds the best hit by more than
# IMAGE_DISTANCE_MARGIN — "adaptive" because clearly-worse figures drop out.
IMAGE_MAX = 3
IMAGE_DISTANCE_MARGIN = 0.2


def _select_image_files(image_hits: list[dict]) -> list[str]:
    """Pick relevant image basenames from the dense image channel."""
    files, dists = [], []
    for r in image_hits:
        meta = r.get("metadata") or {}
        name = meta.get("image_file")
        if name:
            files.append(os.path.basename(name))
            distance = (meta.get("retrieval") or {}).get("cosine_distance")
            dists.append(distance if distance is not None else 2.0)
    if not files:
        return []
    best = min(dists)
    kept = [f for f, d in zip(files, dists) if d <= best + IMAGE_DISTANCE_MARGIN]
    return kept[:IMAGE_MAX] or files[:1]


async def node(state: SystemState) -> dict:
    """Retrieve unstructured documents + images from ChromaDB and store as
    context.

    This node does NOT generate a final answer — it retrieves RAG context and
    semantic image matches. The supervisor_evaluator decides whether to
    continue to text_to_sql or synthesis.
    """
    user_query = state["user_query"]

    # Text channel: over-fetch so the image records that still ride in the
    # hybrid ranking can't displace text chunks (3 kept). Image matches come
    # from a dedicated dense channel on captions.
    raw = query_vector_store.invoke({"query": user_query, "n_results": 6})
    payload = json.loads(raw)

    results = payload.get("results", []) if payload.get("status") == "success" else []
    retrieved_docs = [
        r["content"] for r in results
        if (r.get("metadata") or {}).get("type") != "image"
    ][:3]
    retrieved_images = _select_image_files(
        payload.get("images", []) if payload.get("status") == "success" else []
    )

    combined_context = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else "None"

    return {
        "retrieved_docs": retrieved_docs,
        "retrieved_images": retrieved_images,
        "business_rules": combined_context,
    }
