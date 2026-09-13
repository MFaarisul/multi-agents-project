"""Document ingestion pipeline for the ChromaDB vector store.

Flow:
  resources/rag/* (pdf, txt, md)
    → Docling parse (OCR + picture extraction)
    → HybridChunker (tokenizer-aware, BGE tokenizer, 512 max tokens)
    → BAAI/bge-small-en-v1.5 embeddings (explicit — Chroma default embedder bypassed)
    → ChromaDB collection (reset on every run to guarantee one consistent
      embedding space; corpus is small so re-ingestion is cheap)

Extracted images are saved to temp_storage/rag_images/ and become FIRST-CLASS
retrievable records: each image gets a VLM-generated caption (LLM_VLM_MODEL via
the gateway, OpenAI-style image_url blocks) embedded with the same BGE model, so
retrieval matches images to queries SEMANTICALLY. The image file itself is
never copied at query time — it is served straight from temp_storage/rag_images/.

Images are referenced by BASENAME in chunk metadata; the app resolves them
against its own rag_images mount (IMAGES_DIR).
"""

import base64
import glob
import logging
import os

import chromadb
import httpx
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "enterprise_knowledge_base"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_MAX_TOKENS = 512  # BGE context window

SUPPORTED_EXTENSIONS = ("*.pdf", "*.txt", "*.md")

VLM_CAPTION_PROMPT = (
    "Describe this figure in 2-3 factual sentences for a retrieval index: "
    "what it shows, key labels, and any trends. No headings, no markdown. "
    "Output only the final description, no reasoning or preamble."
)
VLM_TIMEOUT_S = 120.0
VLM_ATTEMPTS = 2

# Lines injected by some inference gateways into completions — never store
# them as captions.
_GATEWAY_NOISE = ("Karpathy", "multica-ai", "karpathy-skills")


def _resolve_dirs() -> tuple[str, str]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rag_dir = os.getenv(
        "DATA_SOURCES_DIR", os.path.join(base_dir, "resources", "rag")
    )
    images_dir = os.getenv(
        "IMAGES_DIR", os.path.join(base_dir, "temp_storage", "rag_images")
    )
    os.makedirs(images_dir, exist_ok=True)
    return rag_dir, images_dir


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0
    return DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _strip_noise_lines(text: str) -> str:
    """Drop gateway-injected notice lines so they never become captions."""
    lines = [
        line
        for line in (text or "").splitlines()
        if not any(marker in line for marker in _GATEWAY_NOISE)
        and not line.strip().startswith("💡")
    ]
    return "\n".join(lines).strip()


def _vlm_caption(image_path: str) -> str | None:
    """Caption one image via the gateway vision model.

    Uses the OpenAI-style image_url format (data URI) — the gateway's
    Anthropic-style "image" block route is unreliable. mimo additionally
    reasons before answering, hence the generous max_tokens. Returns None
    on any failure so the caller can fall back to context text.
    """
    base_url = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY") or ""
    model = os.getenv("LLM_VLM_MODEL") or "haiku-4.5"
    if not base_url or not api_key:
        logger.warning("LLM_BASE_URL/LLM_API_KEY unset — VLM captions disabled.")
        return None

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": VLM_CAPTION_PROMPT},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(1, VLM_ATTEMPTS + 1):
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=VLM_TIMEOUT_S,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            caption = _strip_noise_lines(content)
            if caption:
                return caption
            logger.warning(
                "VLM caption empty for %s (attempt %d).",
                os.path.basename(image_path),
                attempt,
            )
        except Exception as e:
            logger.warning(
                "VLM caption attempt %d failed for %s: %s",
                attempt,
                os.path.basename(image_path),
                e,
            )
    return None


def _page_text_map(doc) -> dict[int, str]:
    """Group document text by page number — fallback captions for images
    whose VLM captioning failed."""
    pages: dict[int, list[str]] = {}
    for item in getattr(doc, "texts", None) or []:
        prov = getattr(item, "prov", None)
        text = (getattr(item, "text", "") or "").strip()
        if not prov or not text:
            continue
        page_no = getattr(prov[0], "page_no", None)
        if page_no is not None:
            pages.setdefault(page_no, []).append(text)
    return {p: " ".join(chunks) for p, chunks in pages.items()}


def _picture_page(pic) -> int | None:
    prov = getattr(pic, "prov", None)
    return getattr(prov[0], "page_no", None) if prov else None


def _extract_images(doc, stem: str, images_dir: str) -> list[tuple[str, int | None]]:
    """Save every picture in the document to disk; return (path, page_no)."""
    saved: list[tuple[str, int | None]] = []
    for idx, pic in enumerate(doc.pictures):
        try:
            image = pic.image.pil_image if pic.image is not None else None
            if image is None:
                continue
            path = os.path.join(images_dir, f"{stem}_img_{idx}.png")
            image.save(path)
            saved.append((path, _picture_page(pic)))
            logger.info("saved image: %s", os.path.basename(path))
        except Exception as e:
            logger.warning("image %d extraction failed: %s", idx, e)
    return saved


def _chunk_plain_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Token-aware chunking for plain .txt files (Docling has no txt backend).

    Accumulates words until the token budget is reached — mirrors the
    HybridChunker sizing so chunks embed identically.
    """
    chunks: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join(current + [word])
        if current and len(
            tokenizer.encode(candidate, add_special_tokens=False)
        ) > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def run_ingestion():
    chroma_host = os.getenv("CHROMA_HOST", "vector_db")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    rag_dir, images_dir = _resolve_dirs()

    # rag_images is a derived-artifact directory — wipe stale PNGs so
    # re-ingestion never leaves orphans from removed/changed documents.
    # (.gitkeep is not a .png and survives.)
    for stale in glob.glob(os.path.join(images_dir, "*.png")):
        os.remove(stale)

    logger.info("Embedding model: %s", EMBED_MODEL_ID)
    embed_model = SentenceTransformer(EMBED_MODEL_ID)

    chunker = HybridChunker(
        tokenizer=EMBED_MODEL_ID,
        max_tokens=EMBED_MAX_TOKENS,
        merge_peers=True,
    )
    converter = _build_converter()

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    # The embedding model changed (BGE) — the old collection's vectors live in
    # a different space. Reset the collection so the store is always consistent.
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Reset existing collection '%s'.", COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    doc_files: list[str] = []
    for ext in SUPPORTED_EXTENSIONS:
        doc_files.extend(glob.glob(os.path.join(rag_dir, ext)))
    doc_files.sort()

    if not doc_files:
        logger.warning("No documents found in %s. Skipping.", rag_dir)
        return

    total_chunks = 0
    for filepath in doc_files:
        filename = os.path.basename(filepath)
        stem = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        logger.info("Parsing %s ...", filename)

        image_paths: list[tuple[str, int | None]] = []
        try:
            if ext == ".txt":
                # Docling has no plain-text backend — use the token-aware
                # fallback with the BGE tokenizer.
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                chunks = _chunk_plain_text(
                    raw_text, embed_model.tokenizer, EMBED_MAX_TOKENS - 12
                )
                chunk_texts = chunks
            else:
                doc = converter.convert(filepath).document
                image_paths = _extract_images(doc, stem, images_dir)
                dl_chunks = list(chunker.chunk(dl_doc=doc))
                chunk_texts = [c.text for c in dl_chunks]
        except Exception as e:
            logger.error("FAILED to parse %s: %s — skipping.", filename, e)
            continue

        logger.info("%d chunks, %d images", len(chunk_texts), len(image_paths))

        # Semantic image records: caption each figure (VLM, fallback to page
        # text) and embed the caption — retrieval then matches images to the
        # QUERY, not to chunk positions.
        page_texts = (
            _page_text_map(doc) if (ext != ".txt" and image_paths) else {}
        )
        for img_path, img_page in image_paths:
            caption = _vlm_caption(img_path)
            caption_method = "vlm"
            page_no = img_page if img_page is not None else 1
            if caption is None:
                caption_method = "context"
                context = (page_texts.get(page_no) or "")[:400]
                source_note = f"Figure from document {filename}"
                caption = f"{source_note}: {context}".strip() or source_note
                logger.warning(
                    "VLM caption failed for %s — using context text fallback.",
                    os.path.basename(img_path),
                )
            embed_text = f"{caption}\nFigure from document {filename}."
            try:
                collection.add(
                    ids=[f"{stem}_image_{os.path.basename(img_path)}"],
                    documents=[embed_text],
                    embeddings=[embed_model.encode(embed_text, normalize_embeddings=True).tolist()],
                    metadatas=[
                        {
                            "type": "image",
                            "image_file": os.path.basename(img_path),
                            "source": filename,
                            "page": page_no,
                            "caption_method": caption_method,
                        }
                    ],
                )
                logger.info("indexed image record: %s", os.path.basename(img_path))
            except Exception as e:
                logger.error("image record %s failed: %s", os.path.basename(img_path), e)

        ids, documents, embeddings, metadatas = [], [], [], []
        for idx, chunk_text in enumerate(chunk_texts):
            if ext == ".txt":
                contextualized = chunk_text
            else:
                contextualized = chunker.contextualize(dl_chunks[idx])
            embedding = embed_model.encode(
                contextualized, normalize_embeddings=True
            )
            ids.append(f"{stem}_chunk_{idx}")
            documents.append(chunk_text)
            embeddings.append(embedding.tolist())
            meta = {
                "source": filename,
                "chunk_index": idx,
            }
            if ext != ".txt":
                headings = getattr(dl_chunks[idx].meta, "headings", None)
                meta["headings"] = " > ".join(headings) if headings else ""
            metadatas.append(meta)

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunk_texts)

    logger.info(
        "Knowledge base ingestion complete: %d chunks across %d documents.",
        total_chunks,
        len(doc_files),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    run_ingestion()
