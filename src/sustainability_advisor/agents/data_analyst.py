"""Data Analyst Agent.

Responsibility: compute the requested sustainability KPIs for the resolved
facilities and period, and rank facilities against each other when more than
one is in scope. Uses the KPI lookup tool only; all numbers come from Azure
SQL through repositories.
"""

from __future__ import annotations

from pydantic import BaseModel

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RunState
from sustainability_advisor.agents.evidence import kpi_evidence
from sustainability_advisor.domain.analysis import KpiValue
from sustainability_advisor.domain.models import AgentResult, AgentStatus
from sustainability_advisor.tools.data_tools import KpiLookupInput, KpiLookupOutput

KPI_TOOL = "sustainability_kpi_lookup"


class KpiRanking(BaseModel):
    kpi: str
    unit: str
    best_first: list[str]


class DataAnalystOutput(BaseModel):
    kpis: list[KpiValue]
    rankings: list[KpiRanking]


class DataAnalystAgent(BaseAgent):
    name = "data_analyst"
    output_data_model = DataAnalystOutput
    requires_facilities = True

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        entities = inp.entities
        output = await self.call_tool(
            state,
            KPI_TOOL,
            KpiLookupInput(
                facility_ids=entities.facility_ids,
                kpis=entities.kpis,
                period_start=entities.period_start,
                period_end=entities.period_end,
            ),
            KpiLookupOutput,
        )
        values = output.values
        measured = [v for v in values if v.value is not None]
        coverage = len(measured) / len(values) if values else 0.0
        evidence = [kpi_evidence(v, source=KPI_TOOL, confidence=1.0) for v in values]
        rankings = self._rankings(measured)
        off_track = [v for v in measured if v.status == "off_track"]
        summary = (
            f"Computed {len(values)} KPI values for {len(entities.facility_ids)} facilities; "
            f"{len(measured)} have data and {len(off_track)} are off target."
        )
        return self.result(
            state,
            summary=summary,
            confidence=coverage,
            status=AgentStatus.SUCCESS if measured else AgentStatus.PARTIAL,
            evidence=evidence,
            data=DataAnalystOutput(kpis=values, rankings=rankings).model_dump(mode="json"),
        )

    def _rankings(self, values: list[KpiValue]) -> list[KpiRanking]:
        definitions = self.settings.kpis.definitions
        by_kpi: dict[str, list[KpiValue]] = {}
        for value in values:
            by_kpi.setdefault(value.kpi, []).append(value)
        rankings = []
        for kpi, items in by_kpi.items():
            if len(items) < 2:
                continue
            higher_better = definitions[kpi].higher_is_better
            ordered = sorted(items, key=lambda v: v.value or 0.0, reverse=higher_better)
            rankings.append(
                KpiRanking(kpi=kpi, unit=items[0].unit, best_first=[v.facility_id for v in ordered])
            )
        return rankings
