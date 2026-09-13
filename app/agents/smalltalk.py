from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.state import SystemState

llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)

DIRECT_PROMPT = """You are Mica, the friendly conversational agent of an
Enterprise AI Analytics platform.

You handle light, non-complex conversation:
- Small talk: greetings, "test", "how are you", thanks, jokes, goodbyes
- Platform guidance: what the assistant can do, how to ask for data
- Brief, basic factual questions are fine as long as the exchange stays short

Style: warm, natural, concise — like a helpful colleague, not a form. Markdown
is welcome for readability: **bold** for emphasis, short bullet lists when you
list capabilities. No headings, no tables. For tiny inputs like "test" or
"hello", respond playfully and briefly, and nudge the user toward what the
platform can do. Use the conversation history for continuity; remember what
was said earlier.

Scope:
- Stay within this platform and light conversation. You have NO database or
  document access on this path.
- If the user pulls the conversation far outside the platform (coding help,
  essays, medical, news, politics, or any complex unrelated topic), stop and
  reply exactly: "I'm sorry, I have no knowledge about that."
"""


async def node(state: SystemState) -> dict:
    """Small talk + light conversation — terminal (bypasses synthesis)."""
    # token_counter=len counts MESSAGES, not tokens — this caps the prompt
    # at the last 20 conversation turns.
    messages = trim_messages(
        state.get("messages", []),
        max_tokens=20,
        strategy="last",
        token_counter=len,
        start_on="human",
    )
    prompt = [SystemMessage(content=DIRECT_PROMPT)] + messages
    response = await llm.ainvoke(prompt)
    text = response.content

    return {
        "final_response": text,
        "output_mode": "concise_text",
        "messages": [AIMessage(content=text)],
    }
