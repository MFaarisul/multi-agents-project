from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_json_model(text: str, model: type[T]) -> T:
    """Parse an LLM reply into a Pydantic model.

    Tolerates markdown fences and stray text around the JSON object (e.g. the
    gateway's injected notice lines) by slicing from the first "{" to the
    last "}". Fails loud — no silent defaults, a malformed reply raises.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in model reply: {text[:200]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model reply is not valid JSON: {text[:200]!r}") from exc
    return model.model_validate(data)
