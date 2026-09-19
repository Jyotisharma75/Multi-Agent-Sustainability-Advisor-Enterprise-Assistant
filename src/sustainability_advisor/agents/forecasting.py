"""Emission Forecasting Agent.

Responsibility: forward looking emissions analysis.

* forecasting: project monthly emissions for each facility over the requested
  horizon, report the selected method, its backtest error and prediction
  interval, and compare the projection with the approved target. Confidence
  is one minus the backtest MAPE of the selected method
* what-if analysis: when scenario levers are present, apply them to each
  facility's measured baseline with the scenario analysis tool. Scenario
  results are deterministic arithmetic on measured data, so their confidence
  is reduced only by levers that could not be applied
"""

from __future__ import annotations

from pydantic import BaseModel

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RunState
from sustainability_advisor.agents.evidence import evidence_id, fmt
from sustainability_advisor.domain.analysis import ForecastOutcome, ScenarioOutcome
from sustainability_advisor.domain.errors import DataUnavailableError
from sustainability_advisor.domain.models import AgentResult, AgentStatus, Evidence, EvidenceKind
from sustainability_advisor.tools.analysis_tools import EmissionForecastInput, ScenarioAnalysisInput

FORECAST_TOOL = "emission_forecast"
SCENARIO_TOOL = "scenario_analysis"


class ForecastAgentOutput(BaseModel):
    forecasts: list[ForecastOutcome]
    scenarios: list[ScenarioOutcome] = []
    unavailable: dict[str, str]


class ForecastingAgent(BaseAgent):
    name = "forecasting"
    output_data_model = ForecastAgentOutput
    requires_facilities = True

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        if inp.entities.levers:
            return await self._scenarios(inp, state)
        return await self._forecasts(inp, state)

    async def _forecasts(self, inp: AgentInput, state: RunState) -> AgentResult:
        e = inp.entities
        forecasts: list[ForecastOutcome] = []
        unavailable: dict[str, str] = {}
        evidence: list[Evidence] = []
        for facility_id in e.facility_ids:
            try:
                outcome = await self.call_tool(
                    state,
                    FORECAST_TOOL,
                    EmissionForecastInput(
                        facility_id=facility_id, horizon=e.horizon, scopes=e.scopes
                    ),
                    ForecastOutcome,
                )
            except DataUnavailableError as exc:
                unavailable[facility_id] = exc.message
                continue
            forecasts.append(outcome)
            evidence.append(self._evidence(outcome))
        data = ForecastAgentOutput(forecasts=forecasts, unavailable=unavailable)
        if not forecasts:
            return self.result(
                state,
                summary="No facility had enough history to forecast.",
                confidence=0.0,
                status=AgentStatus.PARTIAL,
                data=data.model_dump(mode="json"),
            )
        confidence = sum(max(0.0, 1.0 - (f.backtest_mape or 0.0)) for f in forecasts) / len(
            forecasts
        )
        above = [f.facility_id for f in forecasts if (f.target_gap_ratio or 0) > 0]
        return self.result(
            state,
            summary=(
                f"Forecast {len(forecasts)} facilities; {len(above)} are projected above target."
            ),
            confidence=confidence,
            status=AgentStatus.SUCCESS if not unavailable else AgentStatus.PARTIAL,
            evidence=evidence,
            data=data.model_dump(mode="json"),
        )

    async def _scenarios(self, inp: AgentInput, state: RunState) -> AgentResult:
        e = inp.entities
        scenarios: list[ScenarioOutcome] = []
        unavailable: dict[str, str] = {}
        evidence: list[Evidence] = []
        for facility_id in e.facility_ids:
            try:
                outcome = await self.call_tool(
                    state,
                    SCENARIO_TOOL,
                    ScenarioAnalysisInput(facility_id=facility_id, levers=e.levers),
                    ScenarioOutcome,
                )
            except DataUnavailableError as exc:
                unavailable[facility_id] = exc.message
                continue
            scenarios.append(outcome)
            evidence.append(self._scenario_evidence(outcome))
        data = ForecastAgentOutput(forecasts=[], scenarios=scenarios, unavailable=unavailable)
        if not scenarios:
            return self.result(
                state,
                summary="No facility had a measured baseline for the scenario.",
                confidence=0.0,
                status=AgentStatus.PARTIAL,
                data=data.model_dump(mode="json"),
            )
        requested = len(e.levers) * len(scenarios)
        applied = sum(len(s.effects) for s in scenarios)
        return self.result(
            state,
            summary=f"Evaluated the scenario for {len(scenarios)} facilities.",
            confidence=applied / requested if requested else 0.0,
            status=AgentStatus.SUCCESS if not unavailable else AgentStatus.PARTIAL,
            evidence=evidence,
            data=data.model_dump(mode="json"),
        )

    @staticmethod
    def _scenario_evidence(s: ScenarioOutcome) -> Evidence:
        effects = "; ".join(
            f"{x.lever} {fmt(x.value)} {x.unit} changes emissions by "
            f"{fmt(x.delta_co2e_tonnes)} tCO2e"
            for x in s.effects
        )
        limits = f" Limitations: {' '.join(s.limitations)}" if s.limitations else ""
        summary = (
            f"For {s.facility_id}, the baseline of {fmt(s.baseline_co2e_tonnes)} tCO2e "
            f"({s.baseline_start} to {s.baseline_end}) becomes {fmt(s.projected_co2e_tonnes)} "
            f"tCO2e under the scenario, a change of {fmt(s.delta_co2e_tonnes)} tCO2e "
            f"({s.delta_ratio * 100:.1f}%). {effects}.{limits}"
        )
        return Evidence(
            evidence_id=evidence_id(
                EvidenceKind.SCENARIO,
                s.facility_id,
                s.baseline_start,
                sorted((x.lever, x.value) for x in s.effects),
            ),
            kind=EvidenceKind.SCENARIO,
            source=SCENARIO_TOOL,
            summary=summary,
            facility_id=s.facility_id,
            values={
                "baseline_co2e_tonnes": s.baseline_co2e_tonnes,
                "projected_co2e_tonnes": s.projected_co2e_tonnes,
                "delta_co2e_tonnes": s.delta_co2e_tonnes,
                "delta_ratio": s.delta_ratio,
                **{f"{x.lever}_delta": x.delta_co2e_tonnes for x in s.effects},
                **s.derived_factors,
            },
            attributes={
                "assumptions": [x.assumption for x in s.effects],
                "limitations": s.limitations,
            },
            confidence=len(s.effects) / max(len(s.effects) + len(s.limitations), 1),
        )

    @staticmethod
    def _evidence(f: ForecastOutcome) -> Evidence:
        first, last = f.points[0], f.points[-1]
        target = (
            f" against a prorated target of {fmt(f.target_value)} tCO2e"
            f" ({f.target_gap_ratio * 100:.1f}% gap)"
            if f.target_value is not None and f.target_gap_ratio is not None
            else "; no approved target is recorded"
        )
        mape_text = f"{f.backtest_mape * 100:.1f}%" if f.backtest_mape is not None else "n/a"
        summary = (
            f"{f.facility_id} emissions are projected at {fmt(f.projected_total)} tCO2e over "
            f"{f.horizon} months from {first.period_start} to {last.period_start}{target}. "
            f"Method {f.method}, backtest MAPE {mape_text}, "
            f"{int(f.interval_coverage * 100)}% interval per month up to \u00b1"
            f"{fmt(first.upper - first.value)} tCO2e."
        )
        return Evidence(
            evidence_id=evidence_id(EvidenceKind.FORECAST, f.forecast_id),
            kind=EvidenceKind.FORECAST,
            source=FORECAST_TOOL,
            summary=summary,
            facility_id=f.facility_id,
            values={
                "projected_total": f.projected_total,
                "target_value": f.target_value,
                "target_gap_ratio": f.target_gap_ratio,
                "backtest_mape": f.backtest_mape,
                "first_month": first.value,
                "last_month": last.value,
                "interval_half_width": round(first.upper - first.value, 4),
            },
            attributes={"method": f.method, "horizon": f.horizon, "forecast_id": f.forecast_id},
            confidence=max(0.0, 1.0 - (f.backtest_mape or 0.0)),
        )
