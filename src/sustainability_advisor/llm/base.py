"""Language model provider contract.

Every provider exposes the same asynchronous ``complete`` call. The base class
adds what every call needs regardless of vendor: retry with backoff on
transient failures, a latency measurement, token accounting and a telemetry
record. Unavailability is not retried here; the ``ProviderChain`` falls back
to the next configured provider instead.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from sustainability_advisor.config.models import RetryConfig
from sustainability_advisor.domain.errors import LLMError, LLMUnavailableError
from sustainability_advisor.domain.models import TokenUsage
from sustainability_advisor.observability.telemetry import LLMCallRecord, record_llm_call


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: TokenUsage
    latency_ms: float


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, LLMError) and not isinstance(exc, LLMUnavailableError)


class LLMProvider(ABC):
    """A chat completion backend."""

    def __init__(self, retry: RetryConfig) -> None:
        self._retry = retry

    @property
    @abstractmethod
    def name(self) -> str:
        """Configuration key of the provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model or deployment identifier."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider is configured and its dependencies exist."""

    @abstractmethod
    async def _complete(
        self,
        messages: list[ChatMessage],
        *,
        task: str,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, TokenUsage]:
        """Run one completion without retry."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        task: str,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.is_available():
            raise LLMUnavailableError(f"The {self.name} model provider is not available.")
        status = "error"
        usage = TokenUsage()
        text = ""
        started = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._retry.max_attempts),
                wait=wait_exponential(
                    multiplier=self._retry.initial_backoff_seconds,
                    max=self._retry.max_backoff_seconds,
                ),
                retry=retry_if_exception(_retryable),
                reraise=True,
            ):
                with attempt:
                    text, usage = await self._complete(
                        messages,
                        task=task,
                        json_mode=json_mode,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
            status = "success"
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.name} failed: {type(exc).__name__}") from exc
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            record_llm_call(
                LLMCallRecord(
                    provider=self.name,
                    model=self.model_name,
                    task=task,
                    latency_ms=latency_ms,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    status=status,
                )
            )
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model_name,
            usage=usage,
            latency_ms=latency_ms,
        )

    async def probe(self) -> str:
        """Return a short readiness description without generating text."""
        return "available" if self.is_available() else "unavailable"
