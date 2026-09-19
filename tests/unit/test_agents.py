"""Agent behaviour: contracts, fallbacks, limits and evidence."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from sustainability_advisor.agents.base import AgentInput, RequestEntities
from sustainability_advisor.agents.orchestrator import OrchestratorOutput
from sustainability_advisor.container import Container
from sustainability_advisor.domain.models import (
    AgentResult,
    AgentStatus,
    Evidence,
    EvidenceKind,
)
from sustainability_advisor.llm.mock import MockLLM

pytestmark = pytest.mark.unit

YEAR = {"period_start": date(2024, 7, 1), "period_end": date(2025, 6, 30)}


def make_input(
    query: str, intent: str = "", upstream: list[AgentResult] | None = None, **entities: Any
) -> AgentInput:
    values = {**YEAR, **entities}
    return AgentInput(
        request_id="req_agents_test",
        conversation_id="conv_agents_test",
        query=query,
        intent=intent,
        entities=RequestEntities(**values),
        upstream=upstream or [],
    )


def intent_json(intent: str, confidence: float, **extra: Any) -> str:
    return json.dumps({"intent": intent, "confidence": confidence, **extra})


async def test_orchestrator_uses_model_classification(
    container: Container, azure_llm: MockLLM
) -> None:
    azure_llm.set(
        "intent_classification",
        intent_json(
            "emissions_forecast",
            0.92,
            facility_ids=["FAC-002", "FAC-404"],
            horizon_months=9,
            kpis=["renewable_share", "invented_kpi"],
        ),
    )
    result = await container.agents["orchestrator"].execute(
        make_input("Where are we heading at Monterrey?")
    )
    out = OrchestratorOutput.model_validate(result.data)
    assert result.status == AgentStatus.SUCCESS
    assert out.intent == "emissions_forecast" and out.classification_method == "model"
    assert out.entities.facility_ids == ["FAC-002"]  # unknown id dropped
    assert out.entities.horizon == 9
    assert out.entities.kpis == ["renewable_share"]
    assert out.plan.levels == [["forecasting"], ["recommendation"]]


async def test_orchestrator_falls_back_to_keywords_on_low_confidence(
    container: Container, azure_llm: MockLLM
) -> None:
    azure_llm.set("intent_classification", intent_json("data_query", 0.2))
    result = await container.agents["orchestrator"].execute(
        make_input("Investigate the unusual spike in energy at Rotterdam")
    )
    out = OrchestratorOutput.model_validate(result.data)
    assert out.intent == "anomaly_investigation"
    assert out.classification_method == "keywords"
    assert out.entities.facility_ids == ["FAC-001"]


async def test_orchestrator_without_any_model(
    container: Container, azure_llm: MockLLM, local_llm: MockLLM
) -> None:
    azure_llm._available = False
    local_llm._available = False
    result = await container.agents["orchestrator"].execute(
        make_input("forecast emissions for FAC-003 for the next 2 years")
    )
    out = OrchestratorOutput.model_validate(result.data)
    assert out.intent == "emissions_forecast"
    assert out.entities.horizon == min(24, container.settings.forecasting.max_horizon)
    assert out.entities.facility_ids == ["FAC-003"]


async def test_orchestrator_forced_intent_and_default_period(container: Container) -> None:
    inp = make_input("anything", intent="comprehensive_analysis")
    inp.forced_intent = True
    result = await container.agents["orchestrator"].execute(inp)
    out = OrchestratorOutput.model_validate(result.data)
    assert out.classification_method == "forced"
    # Anchored on the last month with data (June 2025), 12 calendar months.
    assert (out.entities.period_start, out.entities.period_end) == (
        date(2024, 7, 1),
        date(2025, 6, 30),
    )
    assert out.plan.levels == [
        ["anomaly_detection", "compliance_knowledge", "data_analyst", "forecasting"],
        ["sustainability_analyst"],
        ["recommendation"],
    ]


async def test_orchestrator_parses_quarter(container: Container) -> None:
    result = await container.agents["orchestrator"].execute(
        make_input("KPI performance in Q1 2025 at FAC-001")
    )
    out = OrchestratorOutput.model_validate(result.data)
    assert (out.entities.period_start, out.entities.period_end) == (
        date(2025, 1, 1),
        date(2025, 3, 31),
    )


async def test_data_analyst_produces_kpi_evidence_and_rankings(container: Container) -> None:
    result = await container.agents["data_analyst"].execute(
        make_input("kpis", facility_ids=["FAC-001", "FAC-002", "FAC-003"], kpis=["renewable_share"])
    )
    assert result.status == AgentStatus.SUCCESS
    assert len(result.evidence) == 3
    assert all(e.kind == EvidenceKind.KPI for e in result.evidence)
    ranking = result.data["rankings"][0]
    assert ranking["kpi"] == "renewable_share" and len(ranking["best_first"]) == 3


async def test_sustainability_analyst_reuses_upstream_values(container: Container) -> None:
    upstream = await container.agents["data_analyst"].execute(
        make_input("kpis", facility_ids=["FAC-001"], kpis=["renewable_share"])
    )
    result = await container.agents["sustainability_analyst"].execute(
        make_input("trend", upstream=[upstream], facility_ids=["FAC-001"])
    )
    trend = result.data["trends"][0]
    assert trend["current"] == upstream.data["kpis"][0]["value"]
    assert trend["direction"] == "improving"
    assert result.iterations == 1  # only the prior period lookup


async def test_agent_skipped_without_facility(container: Container) -> None:
    result = await container.agents["forecasting"].execute(make_input("forecast"))
    assert result.status == AgentStatus.SKIPPED


async def test_iteration_budget_stops_agent(container: Container) -> None:
    agent = container.agents["anomaly_detection"]
    agent.spec = agent.spec.model_copy(update={"max_iterations": 1})
    result = await agent.execute(
        make_input("x", facility_ids=["FAC-001", "FAC-002"], metrics=["emissions"])
    )
    assert result.status == AgentStatus.FAILED
    assert result.errors == ["agent_failed"]


async def test_agent_timeout_is_contained(container: Container) -> None:
    agent = container.agents["forecasting"]
    agent.spec = agent.spec.model_copy(update={"timeout_seconds": 0.0001})
    result = await agent.execute(make_input("x", facility_ids=["FAC-001"]))
    assert result.status == AgentStatus.FAILED
    assert result.errors == ["timeout"]


async def test_disabled_agent_is_skipped(container: Container) -> None:
    agent = container.agents["document_intelligence"]
    agent.spec = agent.spec.model_copy(update={"enabled": False})
    result = await agent.execute(make_input("policy"))
    assert result.status == AgentStatus.SKIPPED


async def test_forecasting_agent_runs_scenarios_when_levers_present(container: Container) -> None:
    result = await container.agents["forecasting"].execute(
        make_input("what if", facility_ids=["FAC-001"], levers={"energy_efficiency_pct": 10.0})
    )
    assert result.status == AgentStatus.SUCCESS
    assert result.evidence[0].kind == EvidenceKind.SCENARIO
    assert result.data["scenarios"][0]["delta_co2e_tonnes"] < 0


async def test_compliance_agent_keeps_only_cited_obligations(
    container: Container, azure_llm: MockLLM
) -> None:
    azure_llm.set(
        "compliance_extraction",
        json.dumps(
            {
                "obligations": [
                    {"statement": "Report scope 2 using both methods.", "citation_ids": ["C1"]},
                    {"statement": "An obligation citing nothing real.", "citation_ids": ["C99"]},
                ]
            }
        ),
    )
    result = await container.agents["compliance_knowledge"].execute(
        make_input("What are the ESRS E1 scope 2 disclosure requirements?")
    )
    assert result.status == AgentStatus.SUCCESS
    assert [o["statement"] for o in result.data["obligations"]] == [
        "Report scope 2 using both methods."
    ]
    assert result.citations
    doc_types = {c.doc_type for c in result.citations}
    assert doc_types <= set(container.settings.compliance.document_types)
    obligation = next(e for e in result.evidence if e.attributes.get("type") == "obligation")
    assert obligation.citation_ids == [result.citations[0].citation_id]


async def test_compliance_agent_without_model_returns_passages(container: Container) -> None:
    result = await container.agents["compliance_knowledge"].execute(
        make_input("permit fuel deviation investigation")
    )
    assert result.data["extraction"] == "passages_only"
    assert result.citations


def _evidence_result() -> AgentResult:
    evidence = Evidence(
        evidence_id="ev_kpi_1",
        kind=EvidenceKind.KPI,
        source="sustainability_kpi_lookup",
        summary="renewable share for FAC-001 was 11.27 % against a target of 40 (off track).",
        facility_id="FAC-001",
        values={"value": 11.2745, "target": 40.0, "gap_ratio": 0.7181},
        attributes={"kpi": "renewable_share", "status": "off_track"},
        confidence=1.0,
    )
    return AgentResult(
        agent="data_analyst",
        status=AgentStatus.SUCCESS,
        summary="s",
        evidence=[evidence],
        confidence=1.0,
    )


async def test_synthesizer_accepts_grounded_model_answer(
    container: Container, azure_llm: MockLLM
) -> None:
    azure_llm.set(
        "response_synthesis",
        json.dumps(
            {
                "answer": "FAC-001 renewable share is 11.3%, well below the 40% target.",
                "key_findings": ["Renewable share is 71.8% below target."],
                "decision_explanation": "Based on the KPI evidence for FAC-001.",
                "evidence_markers": ["E1"],
            }
        ),
    )
    result = await container.agents["response_synthesizer"].execute(
        make_input("q", upstream=[_evidence_result()])
    )
    assert result.data["method"] == "model"
    assert result.data["grounding"]["grounded"]
    assert result.data["evidence_ids"] == ["ev_kpi_1"]


async def test_synthesizer_rejects_hallucinated_figures(
    container: Container, azure_llm: MockLLM
) -> None:
    azure_llm.set(
        "response_synthesis",
        json.dumps(
            {
                "answer": "Solar would save 1,250,000 EUR per year.",
                "key_findings": [],
                "decision_explanation": "",
            }
        ),
    )
    result = await container.agents["response_synthesizer"].execute(
        make_input("q", upstream=[_evidence_result()])
    )
    assert result.data["method"] == "deterministic"
    assert "1,250,000" not in result.data["answer"]
    assert result.data["grounding"]["grounded"]
    # The model was given one retry with the ungrounded figures listed.
    calls = azure_llm.calls_for("response_synthesis")
    assert len(calls) == 2 and "1,250,000" in calls[1].messages[-1].content


async def test_synthesizer_without_model_is_deterministic(container: Container) -> None:
    result = await container.agents["response_synthesizer"].execute(
        make_input("q", upstream=[_evidence_result()])
    )
    assert result.data["method"] == "deterministic"
    assert "11.27" in result.data["answer"]
