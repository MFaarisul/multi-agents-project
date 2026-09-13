from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import List

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import SystemState
from app.tools.pdf_tools import render_bar_chart, render_pdf

llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.llm_api_key, base_url=settings.llm_base_url)

SYNTHESIS_PROMPT_TEXT = """You are the Synthesis Agent. Using the user query, any retrieved
SQL data, and any RAG business-rule context, answer the user's question directly.

Retrieved SQL data (JSON):
{sql_data}

RAG / business-rule context:
{rag_context}

Retrieved images (filenames): {images}

User query:
{query}

Respond in markdown, 1 to 4 sentences: **bold** the key number(s), and use a
short bullet list only when the answer has 2+ distinct facts.
Answer exactly what the user asked — do not pad with background context or
tangential details. If the SQL data is an empty list ([]), do NOT mention missing
or empty database records — answer from the RAG context alone.
Figure references: when a listed image is relevant, place the marker
[figure: filename.png] inline exactly where the figure belongs (use ONLY
filenames from "Retrieved images").
NEVER write format labels ("concise_text:", "markdown_table:", "Summary:").
No section headings, no tables.
"""

SYNTHESIS_PROMPT_TABLE = """You are the Synthesis Agent. Using the user query, any retrieved
SQL data, and any RAG business-rule context, summarize the findings for the user.

Retrieved SQL data (JSON):
{sql_data}

RAG / business-rule context:
{rag_context}

Retrieved images (filenames): {images}

User query:
{query}

Write a 1 to 2 sentence summary of the findings (include the key numbers),
in markdown — **bold** the key number(s).
Answer exactly what the user asked — do not pad with background context or
tangential details. If the SQL data is an empty list ([]), do NOT mention missing
or empty database records — answer from the RAG context alone.
Figure references: when a listed image is relevant, place the marker
[figure: filename.png] inline exactly where the figure belongs (use ONLY
filenames from "Retrieved images").
Do NOT add a markdown table — the data table is rendered automatically from
the retrieved rows. NEVER write format labels ("concise_text:", "markdown_table:",
"Summary:"). No section headings.
"""

REPORT_PROMPT = """You are the Report Writer for Apex Dynamics. Using the user query, the
retrieved SQL data, and any RAG business-rule context, write a professional analytics report.

Retrieved SQL data (JSON):
{sql_data}

RAG / business-rule context:
{rag_context}

Retrieved images (filenames): {images}

User query:
{query}

Structure — your reply must contain exactly two parts and nothing else:
1. A summary paragraph of 2 to 4 sentences stating the headline result and what it means.
2. The key findings as 3 to 6 short factual lines (one insight per line, include the numbers).

Rules:
- Answer exactly what the user asked — do not pad with background context or
  tangential details.
- Write in markdown: you may **bold** key numbers. Do NOT write headings,
  bullet lists, or pipe tables — structure is added automatically.
- Figure references: when a listed image is relevant, place the marker
  [figure: filename.png] inline exactly where the figure belongs (use ONLY
  filenames from "Retrieved images").
- If the SQL data is an empty list ([]), do NOT mention missing or empty database
  records — answer from the RAG context alone.
- Do NOT write any section titles or headings (e.g. "Executive Summary", "Key Findings",
  "1. Executive Summary") — headings are added automatically around your text.
- NEVER write the labels "concise_text" or "markdown_table".
- Tables and charts are rendered separately from the data — describe insights in words.
- Stay precise: do not characterize the data beyond what it shows (e.g. do not call
  subscriptions "active" unless the data says so). Do not invent figures.
"""

# Lines injected by some inference gateways into model completions — never
# let them reach users or reports.
_NOISE_PATTERNS = (
    re.compile(r"^.*Andrej Karpathy.*$", re.MULTILINE),
    re.compile(r"^.*multica-ai.*$", re.MULTILINE),
    re.compile(r"^.*karpathy-skills.*$", re.MULTILINE),
    re.compile(r"^\s*💡.*$", re.MULTILINE),
    # Residual format labels the model may echo despite the prompt
    re.compile(r"^\s*\*{0,2}(concise_text|markdown_table|Summary)\*{0,2}\s*:?\s*$", re.MULTILINE | re.IGNORECASE),
)


def _strip_noise(text: str) -> str:
    """Remove gateway-injected notice lines and format labels from model output."""
    cleaned = text or ""
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _embed_figure_links(text: str, image_files: List[str]) -> str:
    """Replace exact [figure: name] markers with markdown image links; strip
    any marker that does not match a retrieved image (no hallucinated links)."""
    cleaned = text or ""
    for name in image_files or []:
        cleaned = cleaned.replace(
            f"[figure: {name}]", f"![Figure](/api/v1/rag-images/{name})"
        )
    return re.sub(r"\[figure:\s*[^\]]+\]\s*", "", cleaned)


def _markdown_to_plain(text: str) -> str:
    """Plain-text view of a markdown answer — used for the PDF report body so
    markdown symbols never reach the rendered document."""
    cleaned = _IMAGE_LINK_RE.sub("the figure", text or "")
    cleaned = cleaned.replace("**", "")
    return re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)


def _data_to_markdown(data: List[dict]) -> str:
    if not data:
        return "_No data._"
    columns = list(data[0].keys())
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in data]
    return "\n".join([header, separator, *rows])


async def node(state: SystemState) -> dict:
    """Format the final response according to output_mode."""
    query = state["user_query"]
    data = state.get("retrieved_data") or []
    rag_ctx = state.get("business_rules") or "None"
    output_mode = state.get("output_mode", "concise_text")

    sql_data_str = json.dumps(data, default=str) if data else "[]"

    if output_mode == "pdf_report":
        system_prompt = REPORT_PROMPT
    elif output_mode == "markdown_table":
        system_prompt = SYNTHESIS_PROMPT_TABLE
    else:
        system_prompt = SYNTHESIS_PROMPT_TEXT

    image_names = ", ".join(state.get("retrieved_images") or []) or "none"
    prompt = system_prompt.format(
        sql_data=sql_data_str, rag_context=rag_ctx, query=query, images=image_names
    )
    ai_response = await llm.ainvoke([{"role": "user", "content": prompt}])
    text_answer = _embed_figure_links(
        _strip_noise(ai_response.content),
        state.get("retrieved_images") or [],
    )

    update: dict = {
        "final_response": text_answer,
    }

    if output_mode == "markdown_table":
        # Table only makes sense for SQL-sourced rows; RAG-sourced answers are
        # pure narrative (the summary sentence already covers the findings).
        if data:
            table = _data_to_markdown(data)
            update["final_response"] = f"{text_answer}\n\n{table}"

    elif output_mode == "pdf_report":
        chart_paths: List[str] = []
        # Build a simple chart if we have two-dimensional numeric-ish data.
        # SQL NUMERIC arrives as Decimal — treat it as numeric too.
        if data and len(data) > 1:
            first = data[0]
            label_col = next((k for k in first if isinstance(first[k], str)), None)
            num_col = next(
                (
                    k
                    for k in first
                    if k != label_col
                    and isinstance(first[k], (int, float, Decimal))
                    and not isinstance(first[k], bool)
                ),
                None,
            )
            if label_col and num_col:
                labels = [str(row.get(label_col, "")) for row in data[:10]]
                values = [float(row.get(num_col, 0) or 0) for row in data[:10]]
                try:
                    chart_paths.append(
                        render_bar_chart(labels, values, title=num_col)
                    )
                except Exception:
                    chart_paths = []

        subtitle = (
            f"Generated {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')}"
            f" · Query: {query}"
        )

        try:
            pdf_path = render_pdf(
                title="Apex Dynamics — Analytics Report",
                subtitle=subtitle,
                body_text=_markdown_to_plain(text_answer),
                data=data or None,
                chart_paths=chart_paths,
            )
            update["pdf_file_path"] = pdf_path
            update["generated_charts"] = chart_paths
            update["final_response"] = (
                f"{text_answer}\n\nPDF report generated: {os.path.basename(pdf_path)}"
            )
        except Exception as e:
            update["final_response"] = f"{text_answer}\n\n[PDF generation failed: {e}]"

    # Store the full reply (including table / PDF additions) as the
    # conversation turn so follow-up questions keep that context.
    update["messages"] = [AIMessage(content=update["final_response"])]
    return update
