"""Deterministic scripted provider for tests and offline evaluation.

Responses are keyed by the ``task`` a caller passes to ``complete``. A value
is either a fixed string or a callable that receives the messages, so a test
can produce output that depends on the prompt. Every call is recorded for
assertions. A task without a scripted response raises
``LLMUnavailableError`` so callers exercise their fallback paths.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sustainability_advisor.config.models import RetryConfig
from sustainability_advisor.domain.errors import LLMError, LLMUnavailableError
from sustainability_advisor.domain.models import TokenUsage
from sustainability_advisor.llm.base import ChatMessage, LLMProvider

Responder = str | Callable[[list[ChatMessage]], str]

_NO_RETRY = RetryConfig(max_attempts=1, initial_backoff_seconds=0, max_backoff_seconds=0)


@dataclass
class MockCall:
    task: str
    messages: list[ChatMessage]
    json_mode: bool


class MockLLM(LLMProvider):
    def __init__(
        self,
        responses: dict[str, Responder] | None = None,
        *,
        name: str = "mock",
        model: str = "mock-model",
        fail_tasks: set[str] | None = None,
        available: bool = True,
    ) -> None:
        super().__init__(_NO_RETRY)
        self.responses: dict[str, Responder] = dict(responses or {})
        self.calls: list[MockCall] = []
        self.fail_tasks: set[str] = set(fail_tasks or set())
        self._name = name
        self._model = model
        self._available = available

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return self._available

    def set(self, task: str, response: Responder) -> None:
        self.responses[task] = response

    async def _complete(
        self,
        messages: list[ChatMessage],
        *,
        task: str,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, TokenUsage]:
        self.calls.append(MockCall(task=task, messages=list(messages), json_mode=json_mode))
        if task in self.fail_tasks:
            raise LLMError(f"scripted failure for {task}")
        responder = self.responses.get(task)
        if responder is None:
            raise LLMUnavailableError(f"No scripted response for task {task!r}.")
        text = responder(messages) if callable(responder) else responder
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        return text, TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=len(text.split()))

    def calls_for(self, task: str) -> list[MockCall]:
        return [c for c in self.calls if c.task == task]
