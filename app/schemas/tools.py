"""LLM-facing argument schemas for LangChain tools — the Field descriptions
become the tool parameter documentation the model sees."""

from pydantic import BaseModel, Field


class SQLExecutionInput(BaseModel):
    query: str = Field(description="Strict read-only SQL query to run.")


class RAGQueryInput(BaseModel):
    query: str = Field(description="Semantic query string to search inside ChromaDB.")
    n_results: int = Field(default=3, description="Number of document chunks to retrieve.")
