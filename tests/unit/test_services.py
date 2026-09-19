"""Deterministic analytics services against the seeded database."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from sustainability_advisor.container import Container
from sustainability_advisor.domain.errors import DataUnavailableError, InputRejectedError
from sustainability_advisor.domain.models import Evidence, EvidenceKind
from sustainability_advisor.services.anomaly_service import AnomalyService
from sustainability_advisor.services.forecast_service import (
    ForecastService,
    drift,
    mape,
    seasonal_naive,
)
from sustainability_advisor.services.kpi_service import KpiService
from sustainability_advisor.services.periods import Period, add_months, trailing_months
from sustainability_advisor.services.recommendation_service import RecommendationService
from sustainability_advisor.services.scenario_service import ScenarioService

pytestmark = pytest.mark.unit

YEAR = Period(start=date(2024, 7, 1), end=date(2025, 6, 30))


def test_period_helpers() -> None:
    assert add_months(date(2024, 11, 15), 3) == date(2025, 2, 1)
    window = trailing_months(date(2025, 6, 1), 12)
    assert (window.start, window.end, window.months) == (date(2024, 7, 1), date(2025, 6, 1), 12)


def test_kpi_renewable_share_matches_raw_data(container: Container) -> None:
    service = KpiService(container.settings.kpis, container.repositories)
    value = service.compute_one("FAC-001", "renewable_share", YEAR)
    totals = container.repositories.energy.totals("FAC-001", YEAR.start, YEAR.end)
    assert totals is not None
    expected = totals.renewable_mwh / totals.consumption_mwh * 100
    assert value.value == pytest.approx(expected, rel=1e-6)
    assert value.target == 40.0
    assert value.status == "off_track"
    assert value.gap_ratio == pytest.approx((40.0 - expected) / 40.0, rel=1e-4)


def test_kpi_per_month_target_scales_with_period(container: Container) -> None:
    service = KpiService(container.settings.kpis, container.repositories)
    value = service.compute_one("FAC-001", "total_emissions", YEAR)
    assert value.target == pytest.approx(3300 * 12)


def test_kpi_without_data_is_null_not_estimated(container: Container) -> None:
    service = KpiService(container.settings.kpis, container.repositories)
    value = service.compute_one("FAC-003", "energy_cost", YEAR)
    assert value.value is None and value.status == "no_data"
    empty = service.compute_one(
        "FAC-001", "total_emissions", Period(start=date(2010, 1, 1), end=date(2010, 12, 31))
    )
    assert empty.value is None


def test_kpi_unknown_is_rejected(container: Container) -> None:
    service = KpiService(container.settings.kpis, container.repositories)
    with pytest.raises(InputRejectedError):
        service.compute_one("FAC-001", "made_up_kpi", YEAR)


def test_forecast_baselines() -> None:
    history = np.array([1.0, 2, 3, 4])
    assert list(drift(history, 2)) == [5.0, 6.0]
    assert list(seasonal_naive(history, 3, 2)) == [3.0, 4.0, 3.0]
    assert mape(np.array([2.0, 4.0]), np.array([1.0, 4.0])) == pytest.approx(0.25)


def test_forecast_selects_method_and_persists(container: Container) -> None:
    service = ForecastService(container.settings.forecasting, container.repositories)
    outcome = service.forecast("FAC-002", horizon=6)
    assert outcome.method in container.settings.forecasting.methods
    assert len(outcome.points) == 6
    assert all(p.lower <= p.value <= p.upper for p in outcome.points)
    assert outcome.points[0].period_start == date(2025, 7, 1)
    assert outcome.target_value == pytest.approx(3500 * 6)
    stored = container.repositories.forecasts.latest("FAC-002", outcome.metric)
    assert stored is not None and stored.forecast_id == outcome.forecast_id


def test_forecast_rejects_excessive_horizon(container: Container) -> None:
    service = ForecastService(container.settings.forecasting, container.repositories)
    with pytest.raises(InputRejectedError):
        service.forecast("FAC-001", horizon=container.settings.forecasting.max_horizon + 1)


def test_forecast_without_history(container: Container) -> None:
    service = ForecastService(container.settings.forecasting, container.repositories)
    with pytest.raises(DataUnavailableError):
        service.forecast("FAC-UNKNOWN")


def test_anomaly_detects_injected_spike(container: Container) -> None:
    service = AnomalyService(container.settings.anomaly, container.repositories)
    findings, analysed = service.detect("FAC-001", "emissions", YEAR)
    assert analysed == 12
    spikes = [f for f in findings if f.period_start == date(2025, 4, 1)]
    assert spikes, findings
    assert spikes[0].observed_value > spikes[0].expected_value
    assert spikes[0].severity in {"high", "critical"}
    stored = container.repositories.anomalies.list_since("FAC-001", YEAR.start)
    assert {f.anomaly_id for f in stored} == {f.anomaly_id for f in findings}


def test_anomaly_detects_energy_spike_at_second_site(container: Container) -> None:
    service = AnomalyService(container.settings.anomaly, container.repositories)
    findings, _ = service.detect(
        "FAC-002", "energy", Period(start=date(2024, 7, 1), end=date(2025, 6, 30))
    )
    assert any(f.period_start == date(2025, 1, 1) for f in findings)


def test_anomaly_rejects_foreign_sensor(container: Container) -> None:
    service = AnomalyService(container.settings.anomaly, container.repositories)
    with pytest.raises(InputRejectedError):
        service.detect("FAC-001", "sensor:SEN-002-KWH", YEAR)
    findings, analysed = service.detect("FAC-001", "sensor:SEN-001-GAS", YEAR)
    assert analysed > 0 and isinstance(findings, list)


def test_scenario_uses_derived_grid_factor(container: Container) -> None:
    service = ScenarioService(container.settings.scenario, container.repositories)
    outcome = service.run("FAC-001", {"renewable_electricity_pct": 60.0})
    factor = outcome.derived_factors["scope2_tco2e_per_non_renewable_mwh"]
    # The seed derives scope 2 from non renewable electricity at 0.33 t/MWh
    # while gas is counted in consumption, so the derived factor is lower.
    assert factor is not None and 0 < factor < 0.33
    assert outcome.projected_by_scope[2] < outcome.baseline_by_scope[2]
    assert outcome.projected_by_scope[1] == outcome.baseline_by_scope[1]
    assert outcome.delta_co2e_tonnes == pytest.approx(
        outcome.projected_co2e_tonnes - outcome.baseline_co2e_tonnes, abs=1e-3
    )


def test_scenario_levers_validated(container: Container) -> None:
    service = ScenarioService(container.settings.scenario, container.repositories)
    with pytest.raises(InputRejectedError):
        service.run("FAC-001", {"energy_efficiency_pct": 99.0})
    with pytest.raises(InputRejectedError):
        service.run("FAC-001", {"unknown": 1.0})


def test_scenario_unreachable_renewable_target_is_a_limitation(container: Container) -> None:
    service = ScenarioService(container.settings.scenario, container.repositories)
    outcome = service.run("FAC-003", {"renewable_electricity_pct": 1.0})
    assert not outcome.effects
    assert outcome.limitations


def _kpi_evidence(gap: float, kpi: str = "emissions_intensity") -> Evidence:
    return Evidence(
        evidence_id=f"ev_{kpi}",
        kind=EvidenceKind.KPI,
        source="sustainability_kpi_lookup",
        summary="s",
        facility_id="FAC-001",
        values={"value": 0.4, "target": 0.36, "gap_ratio": gap},
        attributes={"kpi": kpi, "status": "off_track" if gap > 0 else "on_track", "unit": "t/t"},
        confidence=1.0,
    )


def test_recommendations_only_from_evidence(container: Container) -> None:
    service = RecommendationService(container.settings.recommendations, container.repositories)
    assert service.rank([_kpi_evidence(-0.1)]) == []
    ranked = service.rank([_kpi_evidence(0.2)])
    assert len(ranked) == 1
    rec = ranked[0]
    assert rec.evidence == ["ev_emissions_intensity"]
    assert rec.severity == "high"
    assert rec.expected_sustainability_impact.value == pytest.approx(0.04)
    # FAC-001 has an approved heat recovery cost reference; nothing is invented.
    assert rec.estimated_cost.value == 420000
    assert rec.actions


def test_recommendation_cost_unknown_without_reference(container: Container) -> None:
    service = RecommendationService(container.settings.recommendations, container.repositories)
    rec = service.rank([_kpi_evidence(0.5, kpi="renewable_share")])[0]
    assert rec.estimated_cost.value is None
    assert "Unknown" in rec.estimated_cost.uncertainty
    assert rec.severity == "critical"


def test_recommendations_ranked_by_priority(container: Container) -> None:
    service = RecommendationService(container.settings.recommendations, container.repositories)
    ranked = service.rank(
        [_kpi_evidence(0.05, "energy_intensity"), _kpi_evidence(0.5, "renewable_share")]
    )
    assert [r.priority_score for r in ranked] == sorted(
        (r.priority_score for r in ranked), reverse=True
    )
