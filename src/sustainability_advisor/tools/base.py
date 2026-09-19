"""Typed tool contract and registry.

A tool has a pydantic input model, a pydantic output model and one ``run``
method. Agents never call tools directly: they call ``ToolRegistry.invoke``,
which

1. checks the calling agent is permitted to use the tool
2. validates the input against the tool's input model
3. enforces the tool's timeout and retry policy from configuration
4. validates the output against the tool's output model
5. records a telemetry event with latency, attempts and status

Only transient failures are retried; invalid input, missing data and SQL
safety rejections fail immediately.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from sustainability_advisor.config.models import ToolSpec
from sustainability_advisor.domain.errors import (
    AdvisorError,
    DependencyUnavailableError,
    InputRejectedError,
    LLMError,
    RetrievalError,
    ToolExecutionError,
    ToolPermissionError,
    ToolTimeoutError,
)
from sustainability_advisor.guardrails.tool_permissions import ToolPermissionPolicy
from sustainability_advisor.observability.telemetry import ExecutionRecord, record_execution

In = TypeVar("In", bound=BaseModel)
Out = TypeVar("Out", bound=BaseModel)


class ToolContext(BaseModel):
    agent: str
    request_id: str | None = None
    conversation_id: str | None = None


class Tool(ABC, Generic[In, Out]):
    name: ClassVar[str]
    input_model: type[In]
    output_model: type[Out]

    @abstractmethod
    async def run(self, payload: In, ctx: ToolContext) -> Out:
        """Execute the tool."""


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, LLMError):
        return exc.code in {"llm_error", "llm_timeout"}
    return isinstance(exc, DependencyUnavailableError | RetrievalError | ToolTimeoutError)


class ToolRegistry:
    def __init__(self, specs: dict[str, ToolSpec], policy: ToolPermissionPolicy) -> None:
        self._specs = specs
        self._policy = policy
        self._tools: dict[str, Tool[Any, Any]] = {}

    def register(self, tool: Tool[Any, Any]) -> None:
        if tool.name not in self._specs:
            raise ToolPermissionError(f"Tool {tool.name!r} has no configuration entry.")
        self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self, agent: str) -> list[dict[str, Any]]:
        """Schemas of the tools an agent may use."""
        return [
            {
                "name": name,
                "description": self._specs[name].description,
                "input_schema": self._tools[name].input_model.model_json_schema(),
            }
            for name in self._policy.allowed(agent)
            if name in self._tools
        ]

    async def invoke(
        self,
        agent: str,
        name: str,
        payload: BaseModel | dict[str, Any],
        output: type[Out],
        *,
        ctx: ToolContext | None = None,
    ) -> Out:
        self._policy.check(agent, name)
        tool = self._tools.get(name)
        if tool is None:
            raise ToolPermissionError(f"Tool {name!r} is not registered.")
        spec = self._specs[name]
        raw = payload.model_dump() if isinstance(payload, BaseModel) else payload
        try:
            validated = tool.input_model.model_validate(raw)
        except ValidationError as exc:
            raise InputRejectedError(
                f"Invalid input for tool {name}.", details={"errors": exc.errors()[:5]}
            ) from exc
        context = ctx or ToolContext(agent=agent)
        started = time.perf_counter()
        attempts = 0
        status = "error"
        error: str | None = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(spec.retry.max_attempts),
                wait=wait_exponential(
                    multiplier=spec.retry.initial_backoff_seconds,
                    max=spec.retry.max_backoff_seconds,
                ),
                retry=retry_if_exception(_transient),
                reraise=True,
            ):
                with attempt:
                    attempts += 1
                    try:
                        result = await asyncio.wait_for(
                            tool.run(validated, context), timeout=spec.timeout_seconds
                        )
                    except TimeoutError as exc:
                        raise ToolTimeoutError(
                            f"Tool {name} exceeded {spec.timeout_seconds}s."
                        ) from exc
            status = "success"
        except AdvisorError as exc:
            error = exc.code
            raise
        except Exception as exc:
            error = type(exc).__name__
            raise ToolExecutionError(f"Tool {name} failed unexpectedly.") from exc
        finally:
            record_execution(
                ExecutionRecord(
                    name=name,
                    kind="tool",
                    status=status,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    attempts=attempts,
                    error=error,
                    attributes={"agent": agent},
                )
            )
        if not isinstance(result, output):
            result = output.model_validate(result.model_dump())
        return result
