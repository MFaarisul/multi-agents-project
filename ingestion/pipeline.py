"""
Initialize the chromaDB (Vector DB) during project starting

Flow:
  resources/rag/* (pdf, txt, md)
    → Docling parse (OCR + picture extraction)
    → Extract pictures to temp_storage/rag_images/*.png
    → Generate captions for each picture using LLM_VLM_MODEL
    → HybridChunker (tokenizer-aware, BGE tokenizer, 512 max tokens)
    → BAAI/bge-small-en-v1.5 embeddings
    → Store to ChromaDB collection 
"""

import base64
import glob
import io
import logging
import os

import chromadb
from app.core.config import settings
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from openai import OpenAI
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ("*.pdf", "*.txt", "*.md")


def _strip_gateway_noise(text: str) -> str:
    """Drop gateway-injected notice lines."""
    _GATEWAY_NOISE = ("Karpathy", "multica-ai", "karpathy-skills")
    lines = [
        line
        for line in (text or "").splitlines()
        if not any(marker in line for marker in _GATEWAY_NOISE)
        and not line.strip().startswith("💡")
    ]
    return "\n".join(lines).strip()


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    return DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _generate_caption(image_path: str) -> str | None:
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url
    )

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    VLM_CAPTION_PROMPT = """
    Describe this figure in 2-3 factual sentences for a retrieval index:
    what it shows, key labels, and any trends.
    No headings, no markdown. 
    Output only the final description, no reasoning or preamble.
    """

    try:
        response = client.chat.completions.create(
            model=settings.llm_vlm_model,
            messages=[
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
            temperature=0,
            max_tokens=2000,
            extra_body={"enable_thinking": False},
        )
        caption = _strip_gateway_noise(response.choices[0].message.content)

        if caption:
            return caption

        logger.warning(
            "VLM caption empty or noise-only for %s.",
            os.path.basename(image_path),
        )
    except Exception as e:
        logger.warning(
            "VLM caption error for %s: %s",
            os.path.basename(image_path),
            e,
        )

    return None


def _picture_page(pic) -> int | None:
    prov = getattr(pic, "prov", None)
    return getattr(prov[0], "page_no", None) if prov else None


def _extract_images(doc, stem: str, images_dir: str) -> list[tuple[str, int | None, str]]:
    """Save every picture in the document to disk; return (path, page_no, document_caption)."""
    saved: list[tuple[str, int | None, str]] = []
    for idx, pic in enumerate(doc.pictures):
        try:
            image = pic.image.pil_image if pic.image is not None else None
            if image is None:
                continue
            path = os.path.join(images_dir, f"{stem}_img_{idx}.png")
            image.save(path)
            logger.info("saved image: %s", os.path.basename(path))

            default_caption = pic.caption_text(doc).strip()
            saved.append((path, _picture_page(pic), default_caption))
        except Exception as e:
            logger.warning("image %d extraction failed: %s", idx, e)
    return saved


def run_ingestion():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rag_dir = os.path.join(base_dir, "resources", "rag")
    images_dir = os.path.join(base_dir, "temp_storage", "rag_images")

    # Remove stale images from previous runs
    for filename in os.listdir(images_dir):
        if filename.endswith(".png"):
            os.remove(os.path.join(images_dir, filename))

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    try:
        client.delete_collection(settings.chroma_collection)
        logger.info("Reset existing collection '%s'.", settings.chroma_collection)
    except Exception:
        logger.info(
            "Collection '%s' not present yet — nothing to reset.",
            settings.chroma_collection,
        )
    embed_model = SentenceTransformer(settings.embed_model)

    chunker = HybridChunker(
        tokenizer=settings.embed_model,
        max_tokens=settings.embed_max_tokens,
        merge_peers=True,
    )
    
    converter = _build_converter()

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
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

        image_data: list[tuple[str, int | None, str]] = []
        try:
            if ext == ".txt":
                with open(filepath, "rb") as f:
                    raw = f.read()
                doc = converter.convert(
                    DocumentStream(name=f"{stem}.md", stream=io.BytesIO(raw))
                ).document
            else:
                doc = converter.convert(filepath).document
                image_data = _extract_images(doc, stem, images_dir)
            dl_chunks = list(chunker.chunk(dl_doc=doc))
            chunk_texts = [c.text for c in dl_chunks]
        except Exception as e:
            logger.error("FAILED to parse %s: %s — skipping.", filename, e)
            continue

        for img_path, img_page, temp_caption in image_data:
            vlm_caption = _generate_caption(img_path)
            caption = vlm_caption or temp_caption # If VLM fails, fall back to the temporary caption (from docs)
            page_no = img_page if img_page is not None else 1
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
                            "caption_source": "vlm" if vlm_caption else "document",
                            "vlm_caption_status": "success" if vlm_caption else "failed",
                        }
                    ],
                )
                logger.info("successfully indexed image %s", os.path.basename(img_path))
            except Exception as e:
                logger.error("image record %s failed: %s", os.path.basename(img_path), e)

        ids, documents, embeddings, metadatas = [], [], [], []
        for idx, chunk_text in enumerate(chunk_texts):
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
