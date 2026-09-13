# Multi-Agent Enterprise Intelligence System

A LangGraph-powered multi-agent analytics platform that combines structured SQL
analysis, unstructured RAG document search (ChromaDB), and multi-format response
generation (text, Markdown, PDF) behind a FastAPI streaming gateway.

## Architecture

```
                           ┌─────────────────────────────┐
                           │ Supervisor (LLM router)     │
                           │ prompt JSON (no tool calls) │
                           └──────────────┬──────────────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              ▼ (rag or sql)                                          ▼ (neither)
   ┌──────────────────────┐                                ┌──────────────────────┐
   │  qa_rag (agent)      │                                │ smalltalk (agent)    │
   │  ChromaDB retrieval  │                                │  direct LLM answer   │
   └──────────┬───────────┘                                └──────────┬───────────┘
              │                                                       │
              ▼                                                       ▼
   ┌──────────────────────────┐                                       END
   │ Supervisor Evaluator     │
   │ (LLM re-judge)           │
   └──────┬─────────────┬─────┘
          │             │
    SQL_NEEDED    RAG_SUFFICIENT
          ▼             │
   ┌──────────────┐     │
   │ text_to_sql  │     │
   │ (agent)      │     │
   └──────┬───────┘     │
          ▼             ▼
   ┌─────────────────────────┐
   │ synthesis (agent)       │
   │ Text | Markdown | PDF   │
   └────────────┬────────────┘
                ▼
   ┌───────────────────────────┐
   │ PostgresSaver Checkpoint  │
   │ (state persistent memory) │
   └───────────────────────────┘
```

### Routing

**3 agents** — `smalltalk` + `qa_rag` (the conversational agent in two nodes), `text_to_sql`,
`synthesis` — plus **2 LLM routers**: `supervisor` and `supervisor_evaluator`.

- **Supervisor** decides initial intent by returning a JSON object declared in
  its prompt (`SupervisorDecision`): `requires_rag_context`, `requires_sql_data`,
  and `output_mode`. No hardcoded keyword lists — all routing decisions come
  from typed LLM output, parsed and validated with Pydantic.
- **`qa_rag`** retrieves corporate documents from ChromaDB and stores them as
  context; it does not generate the final answer. Retrieval is **hybrid**:
  dense (BGE embeddings) + sparse (BM25) candidates fused with Reciprocal Rank
  Fusion (k=60), then re-scored by a cross-encoder reranker
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Retrieval scores flow through the
  chunk metadata.
- **Supervisor Evaluator** re-judges after retrieval with the prompt-declared
  JSON decision (`EvaluatorDecision`): if the retrieved RAG context
  already answers the query, it routes to `synthesis` and skips SQL entirely
  (course-correction of the supervisor's pre-retrieval guess).
- **`smalltalk`** is terminal: small talk, light conversation, and platform guidance — answered directly from conversation
  history, bypassing synthesis.
- The PostgresSaver checkpointer persists the full conversation state after each
  super-step, enabling multi-turn thread memory.

### Semantic image retrieval

Document figures are **first-class retrievable items**, not chunk by-products:

1. **Ingestion** — Docling extracts each figure to `temp_storage/rag_images/`,
   then a vision model (`LLM_VLM_MODEL`, e.g. `haiku-4.5`) writes a factual
   caption. The caption is embedded with the same BGE model and stored as its
   own Chroma record (`type: "image"`, `image_file` basename, source, page).
   If captioning fails, ingestion falls back to the image's page text.
2. **Retrieval** — captions compete in the same hybrid pipeline (dense + BM25
   → RRF → cross-encoder), so images match the *meaning* of the query, not a
   chunk position. `qa_rag` adaptively surfaces the top 2-3 image hits.
3. **Serving** — files are streamed straight from the mounted folder (never
   copied): `GET /api/v1/rag-images/{filename}`. The chat response carries an
   `images` array of these URLs; `/chat/stream` emits an `images` SSE event
   right after retrieval.

### Response format

Every `final_response` is **markdown** — clients must render markdown. Key
numbers are bolded; multi-fact answers may use short bullet lists. Retrieved
figures are referenced inline at the exact spot they belong via
`![Figure](/api/v1/rag-images/<file>)` links (substituted server-side from
exact-match markers — hallucinated links are impossible), and are also listed
in the response's `images` array and the stream's `images` SSE event for
programmatic clients. PDF report bodies receive a plain-text version, so
markdown symbols never reach the rendered document.

**Two storage engines:**
- **PostgreSQL 16** (`analytics_db`) — structured domain data, LangGraph state
  checkpoints via `PostgresSaver`.
- **ChromaDB** (`chroma_db`) — dense embeddings for unstructured corporate
  documents (RAG). Cosine distance, persistent volume.

## Read-Only Security

SQL injection / mutation risk is blocked at two independent layers:

1. **Role sandboxing** — `agent_readonly` user has `SELECT`-only privileges
   (enforced in `resources/sql/init_db.sql`).
2. **Transaction enforcement** — driver sets
   `default_transaction_read_only=on` and `statement_timeout=5000`.

## Project Layout

```
app/            FastAPI + LangGraph application
  core/         Config (Pydantic settings) & SystemState
  agents/       6 graph nodes:
                  supervisor.py            LLM router (structured output)
                  smalltalk.py             Small talk & light Q&A persona (terminal)
                  qa_rag.py                ChromaDB retrieval (context phase)
                  supervisor_evaluator.py  LLM re-judge after RAG (SQL vs RAG)
                  text_to_sql.py           SQL generation + self-correction
                  synthesis.py             Text | Markdown | PDF formatting
  tools/        SQL, vector-retrieval, PDF tools
  db/           psycopg connection pool
ingestion/      Document ingestion pipeline (Docling parse → HybridChunker →
                BAAI/bge-small-en-v1.5 embeddings → ChromaDB; images extracted
                to temp_storage/rag_images/)
resources/      Static project assets
  rag/          Corporate documents for RAG ingestion (pdf/txt/md via Docling)
  sql/          init_db.sql (schema + seed + readonly role; mounted into Postgres first boot)
temp_storage/   Derived artifacts (local stand-in for S3; gitignored)
  rag_images/   Images extracted from documents during ingestion
  reports/      Generated PDF reports (downloadable via /api/v1/reports/{filename})
tests/          Multi-turn memory & SQL guardrail tests
```

## Quick Start

### 1. Configure environment
```bash
cp .env.example .env
# Edit .env: set LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL if not using OpenAI)
```

### 2. Run everything with Docker (recommended)
```bash
docker compose up --build
```

This boots PostgreSQL, ChromaDB, runs the ingestion worker once (seeds the
vector store from `resources/rag/`), then launches FastAPI on `:8080`.

**Adding documents to RAG:** drop a file into `resources/rag/` (the folder is
bind-mounted — no image rebuild needed), then rerun the worker:
`docker compose run --rm ingestion_worker`. The vector collection is reset
and re-ingested in full.

**Generated reports:** PDF reports persist on the host in
`temp_storage/reports/` and are downloadable via
`GET /api/v1/reports/{filename}` (the `pdf_file_path` in the API response
contains the filename).

### 3. Query the API (SSE stream)
```bash
curl -N -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is our total revenue from completed orders?", "thread_id": "sess-1"}'
```

## Local Development (without Docker)

```bash
pip install -e ".[dev]"
# Start Postgres + ChromaDB (e.g. via docker compose up db vector_db)
python -m ingestion.pipeline      # seed vector store
uvicorn app.main:app --reload --port 8080
pytest -q
```

Logs are plain text via the stdlib `logging` module; set `LOG_LEVEL=DEBUG`
(default `INFO`) in `.env` to increase verbosity.

Every request emits an execution trace to the logs — per-node durations with
their LLM calls and tool runs, plus a total:

```
INFO [trace] supervisor: 2.31s [LLM glm-5.2: 2.13s]
INFO [trace] qa_rag: 0.42s [TOOL query_vector_store: 0.38s]
INFO [trace] supervisor_evaluator: 5.91s [LLM glm-5.2: 5.87s]
INFO [trace] text_to_sql: 2.52s [LLM glm-5.2: 2.48s | TOOL execute_read_only_sql: 0.06s]
INFO [trace] synthesis: 4.60s [LLM glm-5.2: 4.59s]
INFO [trace] input -> supervisor (2.31s) -> qa_rag (0.42s) -> supervisor_evaluator (5.91s) -> text_to_sql (2.52s) -> synthesis (4.60s) -> output | total 15.76s
```

## API

### `POST /api/v1/chat/stream`
Server-Sent Events stream of graph node lifecycle + final result.
```json
{ "message": "...", "thread_id": "session-id" }
```

Node lifecycle events fire for every node start (`supervisor`, `smalltalk`,
`qa_rag`, `supervisor_evaluator`, `text_to_sql`, `synthesis`). The
`completed` event fires from whichever node produced the final answer —
`smalltalk` (light conversation) or `synthesis` (RAG/SQL paths).

### `POST /api/v1/chat/invoke`
Non-streaming endpoint returning the final state as JSON.
```json
{ "message": "Total revenue by country", "thread_id": "sess-1" }
```
Response:
```json
{
  "status": "completed",
  "thread_id": "sess-1",
  "final_response": "...",
  "output_mode": "markdown_table",
  "pdf_file_path": null,
  "executed_sql": "SELECT ..."
}
```

> **Note:** the output format (`concise_text` / `markdown_table` /
> `pdf_report`) is decided dynamically by the supervisor LLM based on query
> intent — it is not set by the client.
>
> **Memory:** reuse the same `thread_id` across calls to continue the same
> conversation; a new `thread_id` starts a fresh chat.

### `GET /api/v1/health`
Liveness probe.
