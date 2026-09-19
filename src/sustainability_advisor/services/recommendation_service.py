"""Evidence based recommendation engine.

Recommendations are derived only from evidence produced by tools in the same
request. Three kinds of evidence raise an issue:

* a KPI whose status is ``off_track``        -> issue type ``kpi_off_target``
* a detected anomaly                         -> issue type ``anomaly``
* a forecast whose projection exceeds target -> issue type ``forecast_exceeds_target``

For each issue a playbook from configuration supplies the wording, actions and
complexity; the playbook key is ``<issue_type>.<subject>`` with ``<issue_type>``
as the fallback. Severity comes from configured thresholds on the measured gap
(or the anomaly's own severity), priority from configured weights.

Money is never estimated. ``estimated_cost`` has a value only when the
playbook names a cost reference metric and the facility has an approved value
for it in ``sustainability_metrics``. ``expected_sustainability_impact`` is
the addressable gap measured in the evidence, labelled as such.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from string import Template

from sustainability_advisor.config.models import Playbook, RecommendationConfig
from sustainability_advisor.domain.models import Estimate, Evidence, EvidenceKind, Recommendation
from sustainability_advisor.repositories import Repositories

KPI_ISSUE = "kpi_off_target"
ANOMALY_ISSUE = "anomaly"
FORECAST_ISSUE = "forecast_exceeds_target"


@dataclass
class _Issue:
    issue_type: str
    subject: str
    facility_id: str | None
    magnitude: float
    severity: str
    impact_value: float | None
    impact_unit: str | None
    impact_basis: str
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.facility_id}|{self.issue_type}|{self.subject}"


class RecommendationService:
    def __init__(self, config: RecommendationConfig, repos: Repositories) -> None:
        self._config = config
        self._repos = repos

    # -- issue detection -------------------------------------------------
    def _severity_from_gap(self, gap: float) -> str:
        for name, threshold in sorted(
            self._config.severity_thresholds.items(), key=lambda kv: kv[1], reverse=True
        ):
            if gap >= threshold:
                return name
        return min(self._config.severity_scores, key=lambda n: self._config.severity_scores[n])

    def identify_issues(self, evidence: list[Evidence]) -> list[_Issue]:
        issues: dict[str, _Issue] = {}
        for item in evidence:
            issue = self._issue_from(item)
            if issue is None:
                continue
            existing = issues.get(issue.key)
            if existing is None:
                issues[issue.key] = issue
            else:
                existing.evidence.extend(issue.evidence)
                if issue.magnitude > existing.magnitude:
                    existing.magnitude = issue.magnitude
                    existing.severity = issue.severity
                    existing.impact_value = issue.impact_value
        return list(issues.values())

    def _issue_from(self, item: Evidence) -> _Issue | None:
        attrs, values = item.attributes, item.values
        if item.kind == EvidenceKind.KPI and attrs.get("status") == "off_track":
            gap = float(values.get("gap_ratio") or 0.0)
            value, target = values.get("value"), values.get("target")
            impact = abs(value - target) if value is not None and target is not None else None
            return _Issue(
                issue_type=KPI_ISSUE,
                subject=str(attrs.get("kpi")),
                facility_id=item.facility_id,
                magnitude=gap,
                severity=self._severity_from_gap(gap),
                impact_value=impact,
                impact_unit=attrs.get("unit"),
                impact_basis="Gap between the measured KPI and its approved target.",
                evidence=[item],
            )
        if item.kind == EvidenceKind.ANOMALY and attrs.get("detected") is True:
            observed, expected = values.get("observed"), values.get("expected")
            impact = (
                observed - expected
                if observed is not None and expected is not None and observed > expected
                else None
            )
            severity = str(attrs.get("severity"))
            return _Issue(
                issue_type=ANOMALY_ISSUE,
                subject=str(attrs.get("metric")),
                facility_id=item.facility_id,
                magnitude=float(values.get("score") or 0.0),
                severity=severity if severity in self._config.severity_scores else "medium",
                impact_value=impact,
                impact_unit=attrs.get("unit"),
                impact_basis="Excess of the anomalous observation over its expected value.",
                evidence=[item],
            )
        if item.kind == EvidenceKind.FORECAST:
            forecast_gap = values.get("target_gap_ratio")
            if forecast_gap is None or forecast_gap <= 0:
                return None
            projected, target = values.get("projected_total"), values.get("target_value")
            return _Issue(
                issue_type=FORECAST_ISSUE,
                subject="emissions",
                facility_id=item.facility_id,
                magnitude=float(forecast_gap),
                severity=self._severity_from_gap(float(forecast_gap)),
                impact_value=(projected - target)
                if projected is not None and target is not None
                else None,
                impact_unit="tCO2e",
                impact_basis="Projected emissions above the prorated target over the horizon.",
                evidence=[item],
            )
        return None

    # -- ranking -----------------------------------------------------------
    def _playbook(self, issue: _Issue) -> Playbook | None:
        playbooks = self._config.playbooks
        return playbooks.get(f"{issue.issue_type}.{issue.subject}") or playbooks.get(
            issue.issue_type
        )

    def _cost(self, playbook: Playbook, facility_id: str | None) -> Estimate:
        if playbook.cost_reference_metric and facility_id:
            record = self._repos.metrics.latest(
                facility_id, playbook.cost_reference_metric, date.max
            )
            if record is not None and record.value is not None:
                return Estimate(
                    value=record.value,
                    unit=record.unit,
                    basis=f"Approved reference value {playbook.cost_reference_metric}"
                    f" ({record.source or 'unspecified source'}).",
                    uncertainty="Reference estimate; confirm with a site specific quotation.",
                )
        return Estimate(
            value=None,
            unit=None,
            basis="No approved cost estimate exists in the sustainability data sources.",
            uncertainty="Unknown. A cost must be established before approval.",
        )

    def _band(self, score: float) -> str:
        for band in sorted(self._config.priority_bands, key=lambda b: b.min_score, reverse=True):
            if score >= band.min_score:
                return band.name
        return self._config.priority_bands[-1].name

    def rank(self, evidence: list[Evidence]) -> list[Recommendation]:
        cfg = self._config
        weights = cfg.priority_weights
        recommendations: list[Recommendation] = []
        for issue in self.identify_issues(evidence):
            playbook = self._playbook(issue)
            if playbook is None:
                continue
            coverage = min(len(issue.evidence) / cfg.min_evidence_items, 1.0)
            evidence_confidence = sum(e.confidence for e in issue.evidence) / len(issue.evidence)
            confidence = round(evidence_confidence * coverage, 4)
            severity_score = cfg.severity_scores.get(issue.severity, 0.0)
            ease = 1.0 - cfg.complexity_scores.get(playbook.estimated_complexity, 0.5)
            total_weight = sum(weights.values()) or 1.0
            priority_score = round(
                (
                    weights.get("severity", 0.0) * severity_score
                    + weights.get("confidence", 0.0) * confidence
                    + weights.get("ease", 0.0) * ease
                    + weights.get("evidence", 0.0) * coverage
                )
                / total_weight,
                4,
            )
            variables = {
                "subject": issue.subject.replace("_", " "),
                "facility": issue.facility_id or "the portfolio",
                "magnitude_pct": f"{issue.magnitude * 100:.1f}",
                "severity": issue.severity,
            }
            recommendations.append(
                Recommendation(
                    recommendation_id=_recommendation_id(issue),
                    facility_id=issue.facility_id,
                    issue=Template(playbook.issue_template).safe_substitute(variables),
                    severity=issue.severity,
                    business_impact=Template(playbook.business_impact_template).safe_substitute(
                        variables
                    ),
                    estimated_complexity=playbook.estimated_complexity,
                    estimated_cost=self._cost(playbook, issue.facility_id),
                    expected_sustainability_impact=Estimate(
                        value=round(issue.impact_value, 4)
                        if issue.impact_value is not None
                        else None,
                        unit=issue.impact_unit,
                        basis=issue.impact_basis,
                        uncertainty=(
                            "Upper bound on addressable impact; realised impact depends on "
                            "implementation."
                            if issue.impact_value is not None
                            else "Not quantifiable from the available evidence."
                        ),
                    ),
                    priority=self._band(priority_score),
                    priority_score=priority_score,
                    evidence=[e.evidence_id for e in issue.evidence],
                    confidence=confidence,
                    actions=list(playbook.actions),
                )
            )
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)
        return recommendations[: cfg.max_recommendations]


def _recommendation_id(issue: _Issue) -> str:
    ids = ",".join(sorted(e.evidence_id for e in issue.evidence))
    digest = hashlib.sha1(f"{issue.key}|{ids}".encode(), usedforsecurity=False).hexdigest()[:20]
    return f"rec_{digest}"
