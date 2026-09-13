from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SystemState(TypedDict):
    # Persistent Conversation Memory Across Turns (add_messages appends history)
    messages: Annotated[List[BaseMessage], add_messages]

    # Session Request Context
    user_query: str
    thread_id: str
    output_mode: Literal["concise_text", "markdown_table", "pdf_report"]
    requires_sql_data: bool
    requires_rag_context: bool

    # Internal Execution Context
    business_rules: Optional[str]
    retrieved_docs: Optional[List[str]]
    retrieved_images: Optional[List[str]]
    retrieved_data: Optional[List[Dict[str, Any]]]
    executed_sql: Optional[str]
    sql_error_count: int

    # Generated Artifacts
    generated_charts: Optional[List[str]]
    final_response: Optional[str]
    pdf_file_path: Optional[str]
