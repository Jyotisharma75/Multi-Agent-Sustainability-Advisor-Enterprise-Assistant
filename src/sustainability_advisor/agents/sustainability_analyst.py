"""Sustainability Analyst Agent.

Responsibility: judge performance, not just report it. For each facility and
KPI it compares the requested period with the immediately preceding period of
equal length and classifies the movement as improving, deteriorating or
stable (within the configured tolerance), taking each KPI's direction into
account. Current period values are taken from the Data Analyst's results when
available so the two agents never disagree on a number.
"""

from __future__ import annotations

from pydantic import BaseModel

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RunState
from sustainability_advisor.agents.evidence import evidence_id, fmt, kpi_evidence
from sustainability_advisor.domain.analysis import KpiValue
from sustainability_advisor.domain.models import AgentResult, Evidence, EvidenceKind
from sustainability_advisor.services.periods import add_months, month_start
from sustainability_advisor.tools.data_tools import KpiLookupInput, KpiLookupOutput

KPI_TOOL = "sustainability_kpi_lookup"


class KpiTrend(BaseModel):
    facility_id: str
    kpi: str
    current: float | None
    previous: float | None
    change_ratio: float | None
    direction: str


class SustainabilityAnalystOutput(BaseModel):
    trends: list[KpiTrend]
    off_track: list[str]


class SustainabilityAnalystAgent(BaseAgent):
    name = "sustainability_analyst"
    output_data_model = SustainabilityAnalystOutput
    requires_facilities = True

    async def _current(self, inp: AgentInput, state: RunState) -> list[KpiValue]:
        for upstream in inp.upstream:
            if upstream.agent == "data_analyst" and upstream.data.get("kpis"):
                return [KpiValue.model_validate(v) for v in upstream.data["kpis"]]
        e = inp.entities
        output = await self.call_tool(
            state,
            KPI_TOOL,
            KpiLookupInput(
                facility_ids=e.facility_ids,
                kpis=e.kpis,
                period_start=e.period_start,
                period_end=e.period_end,
            ),
            KpiLookupOutput,
        )
        return output.values

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        e = inp.entities
        current = await self._current(inp, state)
        months = (
            (e.period_end.year - e.period_start.year) * 12
            + (e.period_end.month - e.period_start.month)
            + 1
        )
        prev_start = add_months(month_start(e.period_start), -months)
        prev_end = add_months(month_start(e.period_start), -1)
        previous_output = await self.call_tool(
            state,
            KPI_TOOL,
            KpiLookupInput(
                facility_ids=e.facility_ids,
                kpis=sorted({v.kpi for v in current}),
                period_start=prev_start,
                period_end=prev_end,
            ),
            KpiLookupOutput,
        )
        previous = {(v.facility_id, v.kpi): v for v in previous_output.values}
        tolerance = self.settings.kpis.on_track_tolerance
        definitions = self.settings.kpis.definitions
        trends: list[KpiTrend] = []
        evidence: list[Evidence] = [
            kpi_evidence(v, source=KPI_TOOL, confidence=1.0) for v in current
        ]
        for value in current:
            prior = previous.get((value.facility_id, value.kpi))
            prior_value = prior.value if prior else None
            change: float | None = None
            direction = "unknown"
            if value.value is not None and prior_value not in (None, 0):
                assert prior_value is not None
                change = (value.value - prior_value) / abs(prior_value)
                better = -change if not definitions[value.kpi].higher_is_better else change
                direction = (
                    "stable"
                    if abs(change) <= tolerance
                    else ("improving" if better > 0 else "deteriorating")
                )
            trends.append(
                KpiTrend(
                    facility_id=value.facility_id,
                    kpi=value.kpi,
                    current=value.value,
                    previous=prior_value,
                    change_ratio=round(change, 6) if change is not None else None,
                    direction=direction,
                )
            )
            if change is not None:
                evidence.append(
                    Evidence(
                        evidence_id=evidence_id(
                            EvidenceKind.KPI, "trend", value.facility_id, value.kpi, prev_start
                        ),
                        kind=EvidenceKind.KPI,
                        source=KPI_TOOL,
                        summary=(
                            f"{value.kpi.replace('_', ' ')} for {value.facility_id} moved from "
                            f"{fmt(prior_value)} ({prev_start} to {prev_end}) to "
                            f"{fmt(value.value)}, a change of {change * 100:.1f}% ({direction})."
                        ),
                        facility_id=value.facility_id,
                        values={
                            "current": value.value,
                            "previous": prior_value,
                            "change_ratio": round(change, 6),
                        },
                        attributes={"kpi": value.kpi, "direction": direction, "status": "trend"},
                        confidence=1.0,
                    )
                )
        known = [t for t in trends if t.direction != "unknown"]
        off_track = sorted({f"{v.facility_id}:{v.kpi}" for v in current if v.status == "off_track"})
        deteriorating = [t for t in known if t.direction == "deteriorating"]
        summary = (
            f"Assessed {len(trends)} KPI trends; {len(deteriorating)} deteriorating and "
            f"{len(off_track)} off target."
        )
        measured = [v for v in current if v.value is not None]
        # Confidence reflects how much of the assessment is backed by data:
        # current values measured, and trends comparable with a prior period.
        coverage = len(measured) / len(current) if current else 0.0
        comparable = len(known) / len(trends) if trends else 0.0
        return self.result(
            state,
            summary=summary,
            confidence=(coverage + comparable) / 2,
            evidence=evidence,
            data=SustainabilityAnalystOutput(trends=trends, off_track=off_track).model_dump(
                mode="json"
            ),
        )
