"""Results of deterministic analyses (KPIs, anomalies, forecasts, scenarios)."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from sustainability_advisor.domain.models import Model


class KpiValue(Model):
    facility_id: str
    kpi: str
    description: str
    value: float | None
    unit: str
    target: float | None
    gap_ratio: float | None = Field(
        description="Relative distance from target, positive when performance is worse"
    )
    status: str
    period_start: date
    period_end: date
    sources: list[str]
    note: str | None = None


class AnomalyFinding(Model):
    anomaly_id: str
    facility_id: str
    metric: str
    period_start: date
    observed_value: float
    expected_value: float
    score: float
    severity: str
    method: str
    detected_at: datetime
    status: str = "open"


class ForecastPoint(Model):
    period_start: date
    value: float
    lower: float
    upper: float


class ForecastOutcome(Model):
    forecast_id: str
    facility_id: str
    metric: str
    method: str
    horizon: int
    history_points: int
    backtest_mape: float | None
    interval_coverage: float
    points: list[ForecastPoint]
    candidate_scores: dict[str, float | None]
    generated_at: datetime
    target_value: float | None
    projected_total: float
    target_gap_ratio: float | None


class ScenarioLeverEffect(Model):
    lever: str
    value: float
    unit: str
    delta_co2e_tonnes: float
    assumption: str


class ScenarioOutcome(Model):
    facility_id: str
    baseline_start: date
    baseline_end: date
    baseline_co2e_tonnes: float
    projected_co2e_tonnes: float
    delta_co2e_tonnes: float
    delta_ratio: float
    baseline_by_scope: dict[int, float]
    projected_by_scope: dict[int, float]
    effects: list[ScenarioLeverEffect]
    derived_factors: dict[str, float | None]
    limitations: list[str]
