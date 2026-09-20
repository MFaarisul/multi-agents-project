"""LLM decision contracts enforced via tool-calling structured output."""

from typing import Literal

from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    """Structured routing decision produced by the supervisor LLM."""
    requires_rag_context: bool = Field(
        description=(
            "True if unstructured document/glossary retrieval (RAG) is needed "
            "to answer the query."
        )
    )
    requires_sql_data: bool = Field(
        description=(
            "True if structured SQL data from the database is needed to "
            "answer the query."
        )
    )
    output_mode: Literal["concise_text", "markdown_table", "pdf_report"] = Field(
        description=(
            "concise_text for quick factual or conversational answers, "
            "markdown_table when the user wants data in a table or breakdown, "
            "pdf_report when the user explicitly asks for a report or detailed document."
        )
    )


class EvaluatorDecision(BaseModel):
    """Structured decision produced by the supervisor evaluator LLM."""
    requires_sql_data: bool = Field(
        description=(
            "True if structured SQL data from the database is still needed to "
            "answer the user's query after reviewing the retrieved RAG context. "
            "False if the RAG context alone is sufficient."
        )
    )
