"""Agent contract.

Every agent has, from configuration, a responsibility, a list of allowed tools,
a timeout, a retry policy, a maximum number of iterations and a minimum
confidence. In code it declares its input requirements and the schema of the
``data`` it returns. ``BaseAgent.execute`` enforces all of that uniformly:

* required entities are checked before any work
* the run is bounded by the agent timeout
* transient failures are retried per the agent retry policy
* every tool or model call consumes one iteration; exceeding the budget stops
  the agent, which prevents unbounded loops
* the returned ``data`` is validated against the declared output schema
* any failure becomes an ``AgentResult`` with status ``failed``; exceptions
  never escape into the graph
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, Field, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from sustainability_advisor.config.models import AgentSpec, Settings
from sustainability_advisor.domain.errors import (
    AdvisorError,
    AgentExecutionError,
    DependencyUnavailableError,
    LLMError,
    RetrievalError,
    ToolTimeoutError,
)
from sustainability_advisor.domain.models import AgentResult, AgentStatus, TokenUsage
from sustainability_advisor.llm.base import LLMResponse
from sustainability_advisor.llm.factory import ProviderChain
from sustainability_advisor.observability.logging import get_logger
from sustainability_advisor.observability.telemetry import ExecutionRecord, record_execution
from sustainability_advisor.prompts.loader import PromptLibrary
from sustainability_advisor.tools.base import ToolContext, ToolRegistry

logger = get_logger(__name__)

Out = TypeVar("Out", bound=BaseModel)
Schema = TypeVar("Schema", bound=BaseModel)


class RequestEntities(BaseModel):
    """Everything the orchestrator resolved from the request."""

    facility_ids: list[str] = Field(default_factory=list)
    period_start: date
    period_end: date
    period_explicit: bool = False
    kpis: list[str] | None = None
    metrics: list[str] | None = None
    horizon: int | None = None
    scopes: list[int] | None = None
    levers: dict[str, float] = Field(default_factory=dict)
    jurisdiction: str | None = None


class AgentInput(BaseModel):
    request_id: str
    conversation_id: str
    query: str
    intent: str
    entities: RequestEntities
    upstream: list[AgentResult] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    forced_intent: bool = False


@dataclass(frozen=True, slots=True)
class AgentDependencies:
    settings: Settings
    tools: ToolRegistry
    llm: ProviderChain
    prompts: PromptLibrary


class IterationBudget:
    def __init__(self, agent: str, limit: int) -> None:
        self._agent = agent
        self._limit = limit
        self.used = 0

    def consume(self) -> None:
        if self.used >= self._limit:
            raise AgentExecutionError(
                f"Agent {self._agent} reached its limit of {self._limit} iterations."
            )
        self.used += 1


class RunState:
    """Per execution scratch space: iteration budget and token usage."""

    def __init__(self, agent: str, limit: int, ctx: ToolContext) -> None:
        self.budget = IterationBudget(agent, limit)
        self.usage = TokenUsage()
        self.ctx = ctx

    def add_usage(self, responses: list[LLMResponse]) -> None:
        for response in responses:
            self.usage = self.usage.add(response.usage)


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, LLMError):
        return exc.code in {"llm_error", "llm_timeout"}
    return isinstance(exc, ToolTimeoutError | DependencyUnavailableError | RetrievalError)


class BaseAgent(ABC):
    name: ClassVar[str]
    output_data_model: ClassVar[type[BaseModel]]
    requires_facilities: ClassVar[bool] = False

    def __init__(self, spec: AgentSpec, deps: AgentDependencies) -> None:
        self.spec = spec
        self.deps = deps

    @property
    def settings(self) -> Settings:
        return self.deps.settings

    # -- helpers for subclasses ------------------------------------------
    async def call_tool(
        self, state: RunState, tool: str, payload: BaseModel | dict[str, Any], output: type[Out]
    ) -> Out:
        state.budget.consume()
        return await self.deps.tools.invoke(self.name, tool, payload, output, ctx=state.ctx)

    async def call_llm(
        self, state: RunState, prompt: str, schema: type[Schema], **variables: object
    ) -> Schema:
        state.budget.consume()
        messages = self.deps.prompts.messages(prompt, **variables)
        result, responses = await self.deps.llm.structured(messages, schema, task=prompt)
        state.add_usage(responses)
        return result

    def result(
        self,
        state: RunState,
        *,
        summary: str,
        confidence: float,
        status: AgentStatus = AgentStatus.SUCCESS,
        **fields: Any,
    ) -> AgentResult:
        return AgentResult(
            agent=self.name,
            status=status,
            summary=summary,
            confidence=round(max(0.0, min(confidence, 1.0)), 4),
            iterations=state.budget.used,
            token_usage=state.usage,
            **fields,
        )

    # -- contract ----------------------------------------------------------
    @abstractmethod
    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        """Do the agent's work."""

    async def execute(self, inp: AgentInput) -> AgentResult:
        started = time.perf_counter()
        ctx = ToolContext(
            agent=self.name, request_id=inp.request_id, conversation_id=inp.conversation_id
        )
        state = RunState(self.name, self.spec.max_iterations, ctx)
        attempts = 0
        try:
            if not self.spec.enabled:
                return self._finish(
                    self.result(
                        state, summary="Agent disabled.", confidence=0.0, status=AgentStatus.SKIPPED
                    ),
                    started,
                    attempts,
                )
            if self.requires_facilities and not inp.entities.facility_ids:
                return self._finish(
                    self.result(
                        state,
                        summary="No facility could be identified for this analysis.",
                        confidence=0.0,
                        status=AgentStatus.SKIPPED,
                    ),
                    started,
                    attempts,
                )
            result: AgentResult | None = None
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.spec.retry.max_attempts),
                wait=wait_exponential(
                    multiplier=self.spec.retry.initial_backoff_seconds,
                    max=self.spec.retry.max_backoff_seconds,
                ),
                retry=retry_if_exception(_transient),
                reraise=True,
            ):
                with attempt:
                    attempts += 1
                    state = RunState(self.name, self.spec.max_iterations, ctx)
                    result = await asyncio.wait_for(
                        self._run(inp, state), timeout=self.spec.timeout_seconds
                    )
            assert result is not None
            self.output_data_model.model_validate(result.data)
            if (
                result.status == AgentStatus.SUCCESS
                and result.confidence < self.spec.min_confidence
            ):
                result = result.model_copy(update={"status": AgentStatus.PARTIAL})
            return self._finish(result, started, attempts)
        except TimeoutError:
            return self._failure(state, started, attempts, "timeout", "The agent timed out.")
        except ValidationError as exc:
            logger.error("agent.output_invalid", agent=self.name, error=str(exc)[:300])
            return self._failure(
                state, started, attempts, "invalid_output", "The agent produced invalid output."
            )
        except AdvisorError as exc:
            return self._failure(state, started, attempts, exc.code, exc.message)
        except Exception as exc:
            logger.exception("agent.unexpected_error", agent=self.name)
            return self._failure(
                state, started, attempts, type(exc).__name__, "The agent failed unexpectedly."
            )

    def _failure(
        self, state: RunState, started: float, attempts: int, code: str, message: str
    ) -> AgentResult:
        return self._finish(
            self.result(
                state,
                summary=message,
                confidence=0.0,
                status=AgentStatus.FAILED,
                errors=[code],
            ),
            started,
            attempts,
        )

    def _finish(self, result: AgentResult, started: float, attempts: int) -> AgentResult:
        latency = (time.perf_counter() - started) * 1000.0
        result = result.model_copy(update={"latency_ms": round(latency, 2)})
        record_execution(
            ExecutionRecord(
                name=self.name,
                kind="agent",
                status=result.status.value,
                latency_ms=latency,
                attempts=max(attempts, 1),
                error=result.errors[0] if result.errors else None,
                attributes={
                    "iterations": result.iterations,
                    "confidence": result.confidence,
                    "evidence": len(result.evidence),
                    "citations": len(result.citations),
                },
            )
        )
        return result
