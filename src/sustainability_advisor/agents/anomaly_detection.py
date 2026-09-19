"""Anomaly Detection Agent.

Responsibility: investigate unusual observations. For every facility and
metric in scope it runs the anomaly analysis tool and reports each finding
with its observed and expected value, robust score, severity and detection
method. Metrics come from the request or from configuration.
"""

from __future__ import annotations

from pydantic import BaseModel

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RunState
from sustainability_advisor.agents.evidence import evidence_id, fmt
from sustainability_advisor.domain.analysis import AnomalyFinding
from sustainability_advisor.domain.errors import DataUnavailableError
from sustainability_advisor.domain.models import AgentResult, AgentStatus, Evidence, EvidenceKind
from sustainability_advisor.tools.analysis_tools import AnomalyAnalysisInput, AnomalyAnalysisOutput

ANOMALY_TOOL = "anomaly_analysis"


class AnomalyAgentOutput(BaseModel):
    findings: list[AnomalyFinding]
    analysed: dict[str, int]
    unavailable: dict[str, str]


class AnomalyDetectionAgent(BaseAgent):
    name = "anomaly_detection"
    output_data_model = AnomalyAgentOutput
    requires_facilities = True

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        e = inp.entities
        metrics = e.metrics or self.settings.anomaly.default_metrics
        units = self.settings.anomaly.metric_units
        findings: list[AnomalyFinding] = []
        analysed: dict[str, int] = {}
        unavailable: dict[str, str] = {}
        evidence: list[Evidence] = []
        for facility_id in e.facility_ids:
            for metric in metrics:
                key = f"{facility_id}:{metric}"
                try:
                    output = await self.call_tool(
                        state,
                        ANOMALY_TOOL,
                        AnomalyAnalysisInput(
                            facility_id=facility_id,
                            metric=metric,
                            period_start=e.period_start,
                            period_end=e.period_end,
                        ),
                        AnomalyAnalysisOutput,
                    )
                except DataUnavailableError as exc:
                    unavailable[key] = exc.message
                    continue
                analysed[key] = output.points_analysed
                findings.extend(output.findings)
                unit = units.get(metric.split(":", 1)[0], "")
                for finding in output.findings:
                    evidence.append(self._evidence(finding, unit))
                if not output.findings:
                    evidence.append(
                        Evidence(
                            evidence_id=evidence_id(
                                EvidenceKind.ANOMALY, "none", facility_id, metric, e.period_start
                            ),
                            kind=EvidenceKind.ANOMALY,
                            source=ANOMALY_TOOL,
                            summary=(
                                f"No anomalies in {metric} for {facility_id} across "
                                f"{output.points_analysed} periods from {e.period_start} to "
                                f"{e.period_end}."
                            ),
                            facility_id=facility_id,
                            values={"points_analysed": float(output.points_analysed)},
                            attributes={"metric": metric, "detected": False},
                            confidence=1.0,
                        )
                    )
        if not analysed:
            return self.result(
                state,
                summary="There was not enough data to analyse anomalies.",
                confidence=0.0,
                status=AgentStatus.PARTIAL,
                data=AnomalyAgentOutput(
                    findings=[], analysed={}, unavailable=unavailable
                ).model_dump(mode="json"),
            )
        return self.result(
            state,
            summary=f"Analysed {len(analysed)} series and found {len(findings)} anomalies.",
            confidence=len(analysed) / (len(analysed) + len(unavailable)),
            evidence=evidence,
            data=AnomalyAgentOutput(
                findings=findings, analysed=analysed, unavailable=unavailable
            ).model_dump(mode="json"),
        )

    @staticmethod
    def _evidence(f: AnomalyFinding, unit: str) -> Evidence:
        direction = "above" if f.observed_value > f.expected_value else "below"
        return Evidence(
            evidence_id=evidence_id(EvidenceKind.ANOMALY, f.anomaly_id),
            kind=EvidenceKind.ANOMALY,
            source=ANOMALY_TOOL,
            summary=(
                f"{f.metric} for {f.facility_id} in the period starting {f.period_start} was "
                f"{fmt(f.observed_value)} {unit}, {direction} the expected {fmt(f.expected_value)} "
                f"{unit} (robust score {f.score:.2f}, severity {f.severity}, method {f.method})."
            ).replace("  ", " "),
            facility_id=f.facility_id,
            values={
                "observed": f.observed_value,
                "expected": f.expected_value,
                "score": f.score,
            },
            attributes={
                "metric": f.metric,
                "severity": f.severity,
                "method": f.method,
                "period_start": f.period_start.isoformat(),
                "anomaly_id": f.anomaly_id,
                "unit": unit,
                "detected": True,
            },
            confidence=1.0,
        )
