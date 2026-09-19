"""Per request telemetry.

A ``RequestTelemetry`` collects agent executions, tool executions, model calls,
retrieval scores and the final confidence for one request. It is stored in a
context variable so deep components record into it without being passed a
handle, and every record is also emitted as a structured log event.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, Field

from sustainability_advisor.domain.models import TokenUsage
from sustainability_advisor.observability.logging import get_logger

logger = get_logger("telemetry")


class ExecutionRecord(BaseModel):
    name: str
    kind: str
    status: str
    latency_ms: float
    attempts: int = 1
    error: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class LLMCallRecord(BaseModel):
    provider: str
    model: str
    task: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    status: str


class RequestTelemetry(BaseModel):
    request_id: str
    conversation_id: str | None = None
    executions: list[ExecutionRecord] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    citation_count: int = 0
    final_confidence: float | None = None

    @property
    def token_usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=sum(c.prompt_tokens for c in self.llm_calls),
            completion_tokens=sum(c.completion_tokens for c in self.llm_calls),
        )


_current: ContextVar[RequestTelemetry | None] = ContextVar("request_telemetry", default=None)


def start_request(request_id: str, conversation_id: str | None) -> RequestTelemetry:
    telemetry = RequestTelemetry(request_id=request_id, conversation_id=conversation_id)
    _current.set(telemetry)
    return telemetry


def current() -> RequestTelemetry | None:
    return _current.get()


def record_execution(record: ExecutionRecord) -> None:
    telemetry = _current.get()
    if telemetry is not None:
        telemetry.executions.append(record)
    logger.info(
        f"{record.kind}.completed",
        name=record.name,
        status=record.status,
        latency_ms=round(record.latency_ms, 2),
        attempts=record.attempts,
        error=record.error,
        **record.attributes,
    )


def record_llm_call(record: LLMCallRecord) -> None:
    telemetry = _current.get()
    if telemetry is not None:
        telemetry.llm_calls.append(record)
    logger.info(
        "llm.call",
        provider=record.provider,
        model=record.model,
        task=record.task,
        status=record.status,
        latency_ms=round(record.latency_ms, 2),
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
    )


def record_retrieval(scores: list[float], query_mode: str, result_count: int) -> None:
    telemetry = _current.get()
    if telemetry is not None:
        telemetry.retrieval_scores.extend(scores)
    logger.info(
        "retrieval.result",
        mode=query_mode,
        results=result_count,
        top_score=round(max(scores), 4) if scores else None,
        scores=[round(s, 4) for s in scores],
    )


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Yield a mapping whose ``ms`` key holds the elapsed time on exit."""
    holder: dict[str, float] = {"ms": 0.0}
    started = time.perf_counter()
    try:
        yield holder
    finally:
        holder["ms"] = (time.perf_counter() - started) * 1000.0
