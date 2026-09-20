import json
import os
import re
import threading

import chromadb
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings
from app.schemas.tools import RAGQueryInput


EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: " #BGE models expect this instruction prefix on the QUERY (check the docs)

# Hybrid retrieval configuration
CANDIDATES_PER_RETRIEVER = 10
RRF_K = 60  # standard RRF smoothing constant (Cormack et al., 2009)
IMAGE_TOP_K = 3

_STOPWORDS = frozenset(
    "a an the is are was were be been being of to in on at for with and or not it "
    "its this that these those as by from for into about over under between".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]

_embed_model = SentenceTransformer(EMBED_MODEL_ID)
_reranker = CrossEncoder(RERANK_MODEL_ID)


_bm25_cache: dict = {"count": None, "index": None, "ids": [], "docs": [], "metas": []}
_bm25_lock = threading.Lock()


def _get_bm25_state(collection):
    """In-process BM25 index over the collection's chunks, refreshed when the
    collection size changes. Production scaling path: OpenSearch/Elasticsearch."""
    total = collection.count()
    with _bm25_lock:
        if _bm25_cache["count"] == total and _bm25_cache["index"] is not None:
            return _bm25_cache
        data = collection.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas") or [{} for _ in ids]
        index = BM25Okapi([_tokenize(d) for d in docs]) if docs else None
        _bm25_cache.update(count=total, index=index, ids=ids, docs=docs, metas=metas)
        return _bm25_cache


def _get_collection():
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", settings.chroma_host),
        port=int(os.getenv("CHROMA_PORT", settings.chroma_port)),
    )
    return client.get_collection(name=settings.chroma_collection)


# --- Retrieval stages ---------------------------------------------------------


def _query_embedding(query: str) -> list[float]:
    return _embed_model.encode(
        BGE_QUERY_PREFIX + query, normalize_embeddings=True
    ).tolist()


def _dense_candidates(collection, embedding: list[float], k: int) -> list[dict]:
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, max(collection.count(), 1)),
    )
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = (results.get("metadatas") or [[{}]])[0]
    return [
        {"id": cid, "doc": doc, "meta": meta or {}, "retriever": "dense"}
        for cid, doc, meta in zip(ids, docs, metas)
    ]


def _image_candidates(collection, embedding: list[float], k: int) -> list[dict]:
    """Dense-only search restricted to image records: captions embed with the
    same BGE model, so cosine distance is a direct semantic match between the
    query and what the figure shows. The ms-marco cross-encoder is skipped —
    it clusters short figure queries too tightly to discriminate."""
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"type": "image"},
        )
    except Exception:
        return []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = (results.get("metadatas") or [[{}]])[0]
    dists = (results.get("distances") or [[None]])[0]
    return [
        {"id": cid, "doc": doc, "meta": meta or {}, "distance": dist}
        for cid, doc, meta, dist in zip(ids, docs, metas, dists)
    ]


def _bm25_candidates(collection, query: str, k: int) -> list[dict]:
    try:
        cache = _get_bm25_state(collection)
        if cache["index"] is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = cache["index"].get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "id": cache["ids"][i],
                "doc": cache["docs"][i],
                "meta": cache["metas"][i] or {},
                "retriever": "bm25",
            }
            for i in order
            if scores[i] > 0
        ]
    except Exception:
        return []  # sparse failure degrades to dense-only


def _rrf_fuse(rankings: list[list[dict]]) -> list[dict]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_r(d))."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for pos, cand in enumerate(ranking, start=1):
            entry = fused.setdefault(
                cand["id"],
                {"id": cand["id"], "doc": cand["doc"], "meta": cand["meta"], "rrf": 0.0, "ranks": {}},
            )
            entry["rrf"] += 1.0 / (RRF_K + pos)
            entry["ranks"][cand["retriever"]] = pos
    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    pairs = [(query, c["doc"]) for c in candidates]
    scores = _reranker.predict(pairs)
    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


@tool("query_vector_store", args_schema=RAGQueryInput)
def query_vector_store(query: str, n_results: int = 3) -> str:
    """Hybrid retrieval over ChromaDB: dense (BGE) + sparse (BM25) candidates
    fused with Reciprocal Rank Fusion, then cross-encoder reranked. Returns the
    most relevant corporate policy and glossary chunks with retrieval metadata,
    plus semantically matched document figures (image captions, dense-only)."""
    try:
        collection = _get_collection()
        total = collection.count()
        if total == 0:
            return json.dumps({"status": "success", "results": [], "images": []})

        # Shared BGE query embedding for both channels.
        embedding = _query_embedding(query)

        # Stage 1: candidate generation (dense + sparse) for TEXT records.
        rankings = [_dense_candidates(collection, embedding, CANDIDATES_PER_RETRIEVER)]
        rankings.append(_bm25_candidates(collection, query, CANDIDATES_PER_RETRIEVER))

        # Stage 2: RRF fusion
        fused = _rrf_fuse(rankings)[:CANDIDATES_PER_RETRIEVER]

        # Stage 3: cross-encoder rerank (fallback to RRF order on failure)
        method = "hybrid+rerank"
        try:
            ranked = _rerank(query, fused)
        except Exception:
            ranked = fused
            method = "hybrid (rerank unavailable)"

        top = ranked[:n_results]

        output = []
        for pos, cand in enumerate(top, start=1):
            meta = dict(cand.get("meta") or {})
            meta["retrieval"] = {
                "method": method,
                "final_rank": pos,
                "bm25_rank": cand["ranks"].get("bm25"),
                "dense_rank": cand["ranks"].get("dense"),
                "rrf_score": round(cand["rrf"], 4),
                "rerank_score": round(cand["rerank_score"], 4)
                if "rerank_score" in cand
                else None,
            }
            output.append({"content": cand["doc"], "metadata": meta})

        # IMAGE channel: dense cosine match on VLM captions.
        image_hits = _image_candidates(collection, embedding, IMAGE_TOP_K)
        images = []
        for pos, hit in enumerate(image_hits, start=1):
            meta = dict(hit.get("meta") or {})
            meta["retrieval"] = {
                "method": "dense-image",
                "dense_rank": pos,
                "cosine_distance": (
                    round(hit["distance"], 4) if hit.get("distance") is not None else None
                ),
            }
            images.append({"content": hit["doc"], "metadata": meta})

        return json.dumps({"status": "success", "results": output, "images": images})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
