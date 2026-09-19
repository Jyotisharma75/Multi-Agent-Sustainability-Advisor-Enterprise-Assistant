"""Structured (JSON) output with validation and bounded repair.

Models are asked for a single JSON object. The reply is extracted, parsed and
validated against a pydantic schema. When validation fails, the model is shown
its own reply and the validation error once or twice (configurable) before
the call is declared a failure. The repair instruction text comes from the
prompt library, not from code.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from sustainability_advisor.domain.errors import LLMError, LLMOutputError
from sustainability_advisor.llm.base import ChatMessage, LLMProvider, LLMResponse

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> str:
    """Return the first balanced JSON object in ``text``."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise LLMOutputError("The model reply did not contain a JSON object.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : index + 1]
    raise LLMOutputError("The model reply contained an unterminated JSON object.")


def parse_structured(text: str, schema: type[T]) -> T:
    raw = extract_json_object(text)
    try:
        return schema.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMOutputError(
            "The model reply did not match the expected schema.",
            details={"error": str(exc)[:500]},
        ) from exc


async def complete_structured(
    provider: LLMProvider,
    messages: list[ChatMessage],
    schema: type[T],
    *,
    task: str,
    repair_template: str,
    repair_attempts: int,
    max_tokens: int | None = None,
) -> tuple[T, list[LLMResponse]]:
    """Return a validated object and every response that contributed to it."""
    conversation = list(messages)
    responses: list[LLMResponse] = []
    last_error: LLMError | None = None
    for _ in range(repair_attempts + 1):
        response = await provider.complete(
            conversation, task=task, json_mode=True, max_tokens=max_tokens
        )
        responses.append(response)
        try:
            return parse_structured(response.text, schema), responses
        except LLMOutputError as exc:
            last_error = exc
            conversation = [
                *conversation,
                ChatMessage(role="assistant", content=response.text),
                ChatMessage(
                    role="user",
                    content=repair_template.replace("$error", str(exc.details.get("error", exc))),
                ),
            ]
    assert last_error is not None
    raise last_error
