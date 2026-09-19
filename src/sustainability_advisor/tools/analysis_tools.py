"""Analytical tools: anomaly analysis, emission forecast, scenario analysis and
recommendation ranking."""

from __future__ import annotations

import asyncio
from datetime import date

from pydantic import BaseModel, Field

from sustainability_advisor.domain.analysis import AnomalyFinding, ForecastOutcome, ScenarioOutcome
from sustainability_advisor.domain.models import Evidence, Recommendation
from sustainability_advisor.services.anomaly_service import AnomalyService
from sustainability_advisor.services.forecast_service import ForecastService
from sustainability_advisor.services.periods import Period
from sustainability_advisor.services.recommendation_service import RecommendationService
from sustainability_advisor.services.scenario_service import ScenarioService
from sustainability_advisor.tools.base import Tool, ToolContext


def _period(start: date | None, end: date | None) -> Period | None:
    return Period(start=start, end=end) if start and end else None


class AnomalyAnalysisInput(BaseModel):
    facility_id: str
    metric: str = Field(min_length=1, max_length=100)
    period_start: date | None = None
    period_end: date | None = None


class AnomalyAnalysisOutput(BaseModel):
    facility_id: str
    metric: str
    points_analysed: int
    findings: list[AnomalyFinding]


class AnomalyAnalysisTool(Tool[AnomalyAnalysisInput, AnomalyAnalysisOutput]):
    name = "anomaly_analysis"
    input_model = AnomalyAnalysisInput
    output_model = AnomalyAnalysisOutput

    def __init__(self, service: AnomalyService) -> None:
        self._service = service

    async def run(self, payload: AnomalyAnalysisInput, ctx: ToolContext) -> AnomalyAnalysisOutput:
        findings, points = await asyncio.to_thread(
            self._service.detect,
            payload.facility_id,
            payload.metric,
            _period(payload.period_start, payload.period_end),
        )
        return AnomalyAnalysisOutput(
            facility_id=payload.facility_id,
            metric=payload.metric,
            points_analysed=points,
            findings=findings,
        )


class EmissionForecastInput(BaseModel):
    facility_id: str
    horizon: int | None = Field(default=None, gt=0)
    scopes: list[int] | None = None


class EmissionForecastTool(Tool[EmissionForecastInput, ForecastOutcome]):
    name = "emission_forecast"
    input_model = EmissionForecastInput
    output_model = ForecastOutcome

    def __init__(self, service: ForecastService) -> None:
        self._service = service

    async def run(self, payload: EmissionForecastInput, ctx: ToolContext) -> ForecastOutcome:
        return await asyncio.to_thread(
            lambda: self._service.forecast(
                payload.facility_id, horizon=payload.horizon, scopes=payload.scopes
            )
        )


class ScenarioAnalysisInput(BaseModel):
    facility_id: str
    levers: dict[str, float] = Field(min_length=1)
    baseline_start: date | None = None
    baseline_end: date | None = None


class ScenarioAnalysisTool(Tool[ScenarioAnalysisInput, ScenarioOutcome]):
    name = "scenario_analysis"
    input_model = ScenarioAnalysisInput
    output_model = ScenarioOutcome

    def __init__(self, service: ScenarioService) -> None:
        self._service = service

    async def run(self, payload: ScenarioAnalysisInput, ctx: ToolContext) -> ScenarioOutcome:
        return await asyncio.to_thread(
            self._service.run,
            payload.facility_id,
            payload.levers,
            _period(payload.baseline_start, payload.baseline_end),
        )


class RecommendationRankingInput(BaseModel):
    evidence: list[Evidence]


class RecommendationRankingOutput(BaseModel):
    recommendations: list[Recommendation]
    issues_detected: int


class RecommendationRankingTool(Tool[RecommendationRankingInput, RecommendationRankingOutput]):
    name = "recommendation_ranking"
    input_model = RecommendationRankingInput
    output_model = RecommendationRankingOutput

    def __init__(self, service: RecommendationService) -> None:
        self._service = service

    async def run(
        self, payload: RecommendationRankingInput, ctx: ToolContext
    ) -> RecommendationRankingOutput:
        issues = self._service.identify_issues(payload.evidence)
        ranked = await asyncio.to_thread(self._service.rank, payload.evidence)
        return RecommendationRankingOutput(recommendations=ranked, issues_detected=len(issues))
