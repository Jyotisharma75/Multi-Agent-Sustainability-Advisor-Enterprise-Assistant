"""Routed SQL generation and cross model verification against seeded SQLite."""

from __future__ import annotations

import json

import pytest

from sustainability_advisor.container import Container
from sustainability_advisor.domain.errors import SQLSafetyError
from sustainability_advisor.llm.factory import AZURE, LOCAL
from sustainability_advisor.llm.mock import MockLLM

pytestmark = pytest.mark.unit

SIMPLE = "How many facilities are there?"
COMPLEX = (
    "Compare total emissions by facility for each quarter of last year and rank the highest "
    "three facilities versus their energy use per month"
)
COUNT_SQL = "SELECT COUNT(facility_id) AS facility_count FROM facilities"
TOTALS_SQL = (
    "SELECT TOP 3 f.name, SUM(e.co2e_tonnes) AS total_co2e FROM emissions e "
    "JOIN facilities f ON f.facility_id = e.facility_id GROUP BY f.name ORDER BY total_co2e DESC"
)


def draft(sql: str, confidence: float = 0.95) -> str:
    return json.dumps({"sql": sql, "confidence": confidence, "tables_used": []})


def review(verdict: str, corrected: str | None = None, confidence: float = 0.9) -> str:
    return json.dumps(
        {"verdict": verdict, "issues": ["x"], "corrected_sql": corrected, "confidence": confidence}
    )


def test_router_sends_simple_to_local_and_complex_to_azure(container: Container) -> None:
    router = container.sql_service._router
    simple = router.route(SIMPLE, "data_query")
    complex_ = router.route(COMPLEX, "data_query")
    assert simple.primary == LOCAL and simple.verifier == AZURE
    assert complex_.primary == AZURE and complex_.verifier == LOCAL
    assert complex_.complexity > simple.complexity


def test_router_falls_back_when_one_provider_is_down(
    container: Container, azure_llm: MockLLM
) -> None:
    azure_llm._available = False
    decision = container.sql_service._router.route(COMPLEX, "data_query")
    assert decision.primary == LOCAL and decision.verifier is None


async def test_simple_question_generated_locally_without_verification(
    container: Container, local_llm: MockLLM, azure_llm: MockLLM
) -> None:
    local_llm.set("sql_generation", draft(COUNT_SQL))
    answer = await container.sql_service.answer(SIMPLE, intent="data_query")
    assert answer.generated_by == LOCAL
    assert answer.verification == "not_required"
    assert answer.result.rows == [[3]]
    assert not azure_llm.calls


async def test_low_confidence_triggers_cross_verification(
    container: Container, local_llm: MockLLM, azure_llm: MockLLM
) -> None:
    local_llm.set("sql_generation", draft(COUNT_SQL, confidence=0.5))
    azure_llm.set("sql_verification", review("approve"))
    answer = await container.sql_service.answer(SIMPLE, intent="data_query")
    assert answer.verification == "approved"
    assert answer.verified_by == AZURE
    assert answer.confidence == pytest.approx((0.5 + 0.9) / 2)


async def test_revision_is_settled_by_dry_run(
    container: Container, local_llm: MockLLM, azure_llm: MockLLM
) -> None:
    local_llm.set(
        "sql_generation", draft("SELECT COUNT(name) AS n FROM facilities WHERE 1 = 0", 0.5)
    )
    azure_llm.set("sql_verification", review("revise", COUNT_SQL))
    answer = await container.sql_service.answer(SIMPLE, intent="data_query")
    # Results differ, so the authoritative (complex route) provider's SQL wins.
    assert answer.verification == "disagreement"
    assert answer.generated_by == AZURE
    assert answer.result.rows == [[3]]
    assert answer.confidence < 0.9


async def test_unsafe_generation_escalates_to_other_model(
    container: Container, local_llm: MockLLM, azure_llm: MockLLM
) -> None:
    local_llm.set("sql_generation", draft("DELETE FROM facilities"))
    azure_llm.set("sql_generation", draft(COUNT_SQL))
    answer = await container.sql_service.answer(SIMPLE, intent="data_query")
    assert answer.generated_by == AZURE
    assert any("escalated" in n for n in answer.notes)
    # The local model got the rejection reason back before escalation.
    feedback = local_llm.calls_for("sql_generation")[-1].messages[-1].content
    assert "not permitted" in feedback
    remaining = await container.sql_service.answer(SIMPLE, intent="data_query")
    assert remaining.result.rows == [[3]]


async def test_all_models_unsafe_raises(
    container: Container, local_llm: MockLLM, azure_llm: MockLLM
) -> None:
    local_llm.set("sql_generation", draft("DROP TABLE emissions"))
    azure_llm.set("sql_generation", draft("SELECT event_id FROM audit_events"))
    with pytest.raises(SQLSafetyError):
        await container.sql_service.answer(SIMPLE, intent="data_query")


async def test_complex_question_uses_azure_and_local_verifies(
    container: Container, local_llm: MockLLM, azure_llm: MockLLM
) -> None:
    azure_llm.set("sql_generation", draft(TOTALS_SQL, 0.9))
    local_llm.set("sql_verification", review("approve", confidence=0.8))
    answer = await container.sql_service.answer(COMPLEX, intent="data_query")
    assert answer.generated_by == AZURE and answer.verified_by == LOCAL
    assert answer.verification == "approved"
    assert len(answer.result.rows) == 3
    assert answer.result.columns == ["name", "total_co2e"]


async def test_rejection_regenerates_with_verifier(
    container: Container, local_llm: MockLLM, azure_llm: MockLLM
) -> None:
    azure_llm.set("sql_generation", draft(TOTALS_SQL, 0.9))
    local_llm.set("sql_verification", review("reject"))
    local_llm.set("sql_generation", draft(TOTALS_SQL, 0.8))
    answer = await container.sql_service.answer(COMPLEX, intent="data_query")
    assert answer.verification == "revised"
    assert answer.generated_by == LOCAL


async def test_database_error_is_repaired_once(
    container: Container, local_llm: MockLLM, azure_llm: MockLLM
) -> None:
    # Valid against the catalogue but fails at runtime in SQLite.
    failing = "SELECT facility_id, SOUNDEX(name) AS n FROM facilities"
    local_llm.set("sql_generation", draft(failing))
    azure_llm.set("sql_generation", draft(COUNT_SQL))
    answer = await container.sql_service.answer(SIMPLE, intent="data_query")
    assert answer.result.rows == [[3]]
    assert any("repaired" in n for n in answer.notes)


async def test_row_cap_marks_truncation(container: Container, local_llm: MockLLM) -> None:
    local_llm.set("sql_generation", draft("SELECT facility_id, co2e_tonnes FROM emissions"))
    answer = await container.sql_service.answer(SIMPLE, intent="data_query", max_rows=5)
    assert answer.result.row_count == 5
    assert answer.result.truncated
