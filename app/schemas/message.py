from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    """Body for the chat endpoints."""

    message: str = Field(min_length=1, description="The user's chat message")
    thread_id: str = Field(
        default="default_session",
        description="Conversation thread id — reuse it across calls to keep context",
    )


class MessageResponse(BaseModel):
    """Final result returned by /api/v1/chat/invoke."""

    status: str
    thread_id: str
    final_response: str | None = None
    output_mode: str | None = None
    pdf_file_path: str | None = None
    executed_sql: str | None = None
    images: list[str] | None = Field(
        default=None,
        description="URLs of retrieved RAG images relevant to the query",
    )
