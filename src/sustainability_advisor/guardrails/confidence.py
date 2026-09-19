"""Final confidence and citation requirements."""

from __future__ import annotations

from pydantic import BaseModel

from sustainability_advisor.config.models import GuardrailConfig
from sustainability_advisor.domain.models import AgentResult, AgentStatus, EvidenceKind


class ValidationReport(BaseModel):
    passed: bool
    confidence: float
    low_confidence: bool
    failed_required_agents: list[str]
    issues: list[str]


def citation_requirement_met(results: list[AgentResult], config: GuardrailConfig) -> bool:
    """Document derived evidence must come with at least one citation."""
    if not config.require_citations_for_documents:
        return True
    uses_documents = any(e.kind == EvidenceKind.DOCUMENT for r in results for e in r.evidence)
    has_citations = any(r.citations for r in results)
    return has_citations or not uses_documents


def aggregate_confidence(results: list[AgentResult]) -> float:
    """Mean confidence of agents that produced evidence, weighted by evidence count.

    Failed agents count as zero confidence with weight one, so a failure lowers
    the final score instead of being silently ignored.
    """
    weighted = 0.0
    weights = 0.0
    for result in results:
        if result.status == AgentStatus.SKIPPED:
            continue
        weight = max(len(result.evidence) + len(result.citations), 1)
        score = 0.0 if result.status == AgentStatus.FAILED else result.confidence
        weighted += score * weight
        weights += weight
    return round(weighted / weights, 4) if weights else 0.0


def validate_results(
    results: list[AgentResult],
    required_agents: set[str],
    config: GuardrailConfig,
) -> ValidationReport:
    issues: list[str] = []
    failed_required = sorted(
        r.agent for r in results if r.agent in required_agents and r.status == AgentStatus.FAILED
    )
    if failed_required:
        issues.append(f"Required agents failed: {', '.join(failed_required)}.")
    if not citation_requirement_met(results, config):
        issues.append("Document based findings are missing citations.")
    has_evidence = any(r.evidence or r.citations for r in results)
    if not has_evidence:
        issues.append("No evidence was gathered.")
    confidence = aggregate_confidence(results)
    return ValidationReport(
        passed=not issues,
        confidence=confidence,
        low_confidence=confidence < config.min_final_confidence,
        failed_required_agents=failed_required,
        issues=issues,
    )
