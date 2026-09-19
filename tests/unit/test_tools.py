"""Typed tool registry: permissions, validation, timeout, retry and telemetry."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import ClassVar

import pytest
from pydantic import BaseModel

from sustainability_advisor.config.models import RetryConfig, Settings, ToolSpec
from sustainability_advisor.container import Container
from sustainability_advisor.domain.errors import (
    DependencyUnavailableError,
    InputRejectedError,
    ToolPermissionError,
    ToolTimeoutError,
)
from sustainability_advisor.guardrails.tool_permissions import ToolPermissionPolicy
from sustainability_advisor.observability import telemetry
from sustainability_advisor.tools.base import Tool, ToolContext, ToolRegistry
from sustainability_advisor.tools.data_tools import KpiLookupInput, KpiLookupOutput

pytestmark = pytest.mark.unit


class EchoIn(BaseModel):
    value: int


class EchoOut(BaseModel):
    value: int


class FlakyTool(Tool[EchoIn, EchoOut]):
    name: ClassVar[str] = "flaky"
    input_model = EchoIn
    output_model = EchoOut

    def __init__(self, failures: int, delay: float = 0.0) -> None:
        self.failures = failures
        self.delay = delay
        self.calls = 0

    async def run(self, payload: EchoIn, ctx: ToolContext) -> EchoOut:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.calls <= self.failures:
            raise DependencyUnavailableError("transient")
        return EchoOut(value=payload.value * 2)


def _registry(
    settings: Settings, tool: FlakyTool, *, timeout: float = 1.0, attempts: int = 3
) -> ToolRegistry:
    spec = ToolSpec(
        enabled=True,
        description="test",
        timeout_seconds=timeout,
        retry=RetryConfig(max_attempts=attempts, initial_backoff_seconds=0, max_backoff_seconds=0),
    )
    agents = dict(settings.agents)
    agents["sql_agent"] = agents["sql_agent"].model_copy(update={"allowed_tools": ["flaky"]})
    registry = ToolRegistry({"flaky": spec}, ToolPermissionPolicy(agents, {"flaky": spec}))
    registry.register(tool)
    return registry


async def test_retry_recovers_transient_failures(settings: Settings) -> None:
    tool = FlakyTool(failures=2)
    tracker = telemetry.start_request("req_test_tools", None)
    out = await _registry(settings, tool).invoke("sql_agent", "flaky", {"value": 4}, EchoOut)
    assert out.value == 8 and tool.calls == 3
    record = tracker.executions[-1]
    assert record.kind == "tool" and record.status == "success" and record.attempts == 3


async def test_retry_budget_exhausted(settings: Settings) -> None:
    tool = FlakyTool(failures=5)
    with pytest.raises(DependencyUnavailableError):
        await _registry(settings, tool, attempts=2).invoke(
            "sql_agent", "flaky", {"value": 1}, EchoOut
        )
    assert tool.calls == 2


async def test_timeout_is_enforced(settings: Settings) -> None:
    tool = FlakyTool(failures=0, delay=0.5)
    with pytest.raises(ToolTimeoutError):
        await _registry(settings, tool, timeout=0.05, attempts=1).invoke(
            "sql_agent", "flaky", {"value": 1}, EchoOut
        )


async def test_permission_denied_for_other_agent(settings: Settings) -> None:
    with pytest.raises(ToolPermissionError):
        await _registry(settings, FlakyTool(0)).invoke(
            "recommendation", "flaky", {"value": 1}, EchoOut
        )


async def test_invalid_input_rejected_before_execution(settings: Settings) -> None:
    tool = FlakyTool(0)
    with pytest.raises(InputRejectedError):
        await _registry(settings, tool).invoke("sql_agent", "flaky", {"value": "abc"}, EchoOut)
    assert tool.calls == 0


async def test_kpi_tool_rejects_unknown_facility(container: Container) -> None:
    with pytest.raises(InputRejectedError):
        await container.tools.invoke(
            "data_analyst",
            "sustainability_kpi_lookup",
            KpiLookupInput(
                facility_ids=["FAC-999"],
                period_start=date(2024, 7, 1),
                period_end=date(2025, 6, 30),
            ),
            KpiLookupOutput,
        )


async def test_kpi_tool_returns_typed_output(container: Container) -> None:
    out = await container.tools.invoke(
        "data_analyst",
        "sustainability_kpi_lookup",
        {
            "facility_ids": ["FAC-002"],
            "kpis": ["renewable_share"],
            "period_start": "2024-07-01",
            "period_end": "2025-06-30",
        },
        KpiLookupOutput,
    )
    assert out.values[0].kpi == "renewable_share"
    assert out.values[0].unit == "%"


def test_every_configured_tool_is_registered(container: Container) -> None:
    assert set(container.tools.names) == set(container.settings.tools)


def test_tool_description_is_agent_scoped(container: Container) -> None:
    described = {t["name"] for t in container.tools.describe("document_intelligence")}
    assert described == {"document_search", "vector_search"}
