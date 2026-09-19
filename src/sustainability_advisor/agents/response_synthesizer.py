"""Response Synthesizer.

Responsibility: write the final answer from evidence only.

The model sees numbered evidence summaries ``[E1]``, numbered citations
``[C1]`` and the ranked recommendations, and must return a short answer, key
findings and a concise decision explanation (the evidence behind the
conclusion, never its internal reasoning). The draft then passes the grounding
check: every number must trace to evidence and every citation marker must
exist. A draft that fails is sent back once with the specific problems; if it
fails again, or no model is available, the answer is composed
deterministically from the evidence summaries so the user still receives a
correct, cited response.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sustainability_advisor.agents.base import AgentDependencies, AgentInput, BaseAgent, RunState
from sustainability_advisor.config.models import AgentSpec
from sustainability_advisor.domain.errors import LLMError
from sustainability_advisor.domain.models import (
    AgentResult,
    AgentStatus,
    Citation,
    Evidence,
    EvidenceKind,
    Recommendation,
)
from sustainability_advisor.guardrails.grounding import GroundingChecker, GroundingReport

PROMPT = "response_synthesis"
_KIND_ORDER = {
    EvidenceKind.SQL_RESULT: 0,
    EvidenceKind.KPI: 1,
    EvidenceKind.ANOMALY: 2,
    EvidenceKind.FORECAST: 3,
    EvidenceKind.SCENARIO: 4,
    EvidenceKind.DOCUMENT: 5,
    EvidenceKind.RECOMMENDATION: 6,
}


class SynthesisDraft(BaseModel):
    """What the synthesis model must return."""

    answer: str = Field(min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    decision_explanation: str = ""
    evidence_markers: list[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    answer: str
    key_findings: list[str]
    decision_explanation: str
    method: str
    evidence_ids: list[str]
    citation_markers: dict[str, str]
    grounding: GroundingReport


def collect(
    results: list[AgentResult],
) -> tuple[list[Evidence], list[Citation], list[Recommendation]]:
    evidence: dict[str, Evidence] = {}
    citations: dict[str, Citation] = {}
    recommendations: dict[str, Recommendation] = {}
    for result in results:
        for item in result.evidence:
            evidence.setdefault(item.evidence_id, item)
        for citation in result.citations:
            citations.setdefault(citation.citation_id, citation)
        for rec in result.recommendations:
            recommendations.setdefault(rec.recommendation_id, rec)
    ordered = sorted(evidence.values(), key=lambda e: (_KIND_ORDER[e.kind], -e.confidence))
    return ordered, list(citations.values()), list(recommendations.values())


class ResponseSynthesizer(BaseAgent):
    name = "response_synthesizer"
    output_data_model = SynthesisOutput

    def __init__(self, spec: AgentSpec, deps: AgentDependencies) -> None:
        super().__init__(spec, deps)
        self._grounding = GroundingChecker(self.settings.guardrails)

    def _prompt_text(self, key: str, **values: object) -> str:
        return self.deps.prompts.text(PROMPT, key, **values)

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        evidence, citations, recommendations = collect(inp.upstream)
        citation_markers = {f"C{i}": c.citation_id for i, c in enumerate(citations, start=1)}
        evidence_markers = {f"E{i}": e.evidence_id for i, e in enumerate(evidence, start=1)}
        marker_by_citation = {v: k for k, v in citation_markers.items()}

        evidence_block = "\n".join(
            f"[{m}] ({e.kind.value}, confidence {e.confidence:.2f}) {e.summary}"
            + (
                " Sources: "
                + ", ".join(
                    f"[{marker_by_citation[c]}]" for c in e.citation_ids if c in marker_by_citation
                )
                if e.citation_ids
                else ""
            )
            for m, e in zip(evidence_markers, evidence, strict=True)
        )
        citation_block = "\n".join(
            f"[{marker_by_citation[c.citation_id]}] {c.title}"
            + (f", section {c.section}" if c.section else "")
            + (f", page {c.page}" if c.page else "")
            + f" ({c.source_uri})"
            for c in citations
        )
        recommendation_block = "\n".join(
            f"- {r.recommendation_id}: {r.issue} Severity {r.severity}, priority {r.priority}, "
            f"confidence {r.confidence:.2f}. Actions: {'; '.join(r.actions)}"
            for r in recommendations
        )
        agent_block = "\n".join(f"- {r.agent}: {r.status.value}, {r.summary}" for r in inp.upstream)

        draft: SynthesisDraft | None = None
        report: GroundingReport | None = None
        feedback = "none"
        for _ in range(self.spec.max_iterations):
            try:
                draft = await self.call_llm(
                    state,
                    PROMPT,
                    SynthesisDraft,
                    question=inp.query,
                    intent=inp.intent,
                    evidence=evidence_block or "none",
                    citations=citation_block or "none",
                    recommendations=recommendation_block or "none",
                    agents=agent_block or "none",
                    feedback=feedback,
                )
            except LLMError:
                draft = None
                break
            report = self._check(draft, evidence, citations, recommendations, citation_markers)
            if report.grounded:
                break
            feedback = self._prompt_text(
                "grounding_feedback",
                numbers=", ".join(report.ungrounded_numbers) or "none",
                markers=", ".join(report.unknown_citation_markers) or "none",
            )
            draft = None

        method = "model"
        if draft is None or report is None or not report.grounded:
            method = "deterministic"
            draft = self._deterministic(inp, evidence, recommendations, marker_by_citation)
            report = self._check(draft, evidence, citations, recommendations, citation_markers)

        used = [evidence_markers[m] for m in draft.evidence_markers if m in evidence_markers]
        output = SynthesisOutput(
            answer=draft.answer,
            key_findings=draft.key_findings,
            decision_explanation=draft.decision_explanation,
            method=method,
            evidence_ids=used or [e.evidence_id for e in evidence],
            citation_markers=citation_markers,
            grounding=report,
        )
        return self.result(
            state,
            summary=f"Answer composed ({method}).",
            confidence=1.0 if report.grounded else 0.0,
            status=AgentStatus.SUCCESS if report.grounded else AgentStatus.PARTIAL,
            data=output.model_dump(mode="json"),
        )

    def fallback(self, inp: AgentInput) -> AgentResult:
        """Compose the deterministic answer without any model call.

        Used by the graph when the synthesizer fails (for example a model
        timeout), so a request never ends without an evidence based answer.
        """
        evidence, citations, recommendations = collect(inp.upstream)
        citation_markers = {f"C{i}": c.citation_id for i, c in enumerate(citations, start=1)}
        marker_by_citation = {v: k for k, v in citation_markers.items()}
        draft = self._deterministic(inp, evidence, recommendations, marker_by_citation)
        report = self._check(draft, evidence, citations, recommendations, citation_markers)
        output = SynthesisOutput(
            answer=draft.answer,
            key_findings=draft.key_findings,
            decision_explanation=draft.decision_explanation,
            method="deterministic",
            evidence_ids=[e.evidence_id for e in evidence],
            citation_markers=citation_markers,
            grounding=report,
        )
        return AgentResult(
            agent=self.name,
            status=AgentStatus.PARTIAL,
            summary="Answer composed deterministically after the synthesizer failed.",
            confidence=1.0 if report.grounded else 0.0,
            data=output.model_dump(mode="json"),
        )

    def _check(
        self,
        draft: SynthesisDraft,
        evidence: list[Evidence],
        citations: list[Citation],
        recommendations: list[Recommendation],
        markers: dict[str, str],
    ) -> GroundingReport:
        text = "\n".join([draft.answer, *draft.key_findings, draft.decision_explanation])
        return self._grounding.check(text, evidence, citations, recommendations, markers)

    def _deterministic(
        self,
        inp: AgentInput,
        evidence: list[Evidence],
        recommendations: list[Recommendation],
        marker_by_citation: dict[str, str],
    ) -> SynthesisDraft:
        limit = self.settings.guardrails.max_answer_chars
        findings: list[str] = []
        markers: list[str] = []
        for index, item in enumerate(evidence, start=1):
            cites = "".join(
                f" [{marker_by_citation[c]}]" for c in item.citation_ids if c in marker_by_citation
            )
            findings.append(f"{item.summary}{cites}")
            markers.append(f"E{index}")
        if not findings:
            answer = self._prompt_text("fallback_no_evidence")
        else:
            lines = [self._prompt_text("fallback_intro", count=len(findings))]
            size = len(lines[0])
            for finding in findings:
                if size + len(finding) + 3 > limit:
                    break
                lines.append(f"- {finding}")
                size += len(finding) + 3
            if recommendations:
                lines.append(
                    self._prompt_text("fallback_recommendations", count=len(recommendations))
                )
                lines.extend(f"- {r.issue} ({r.priority} priority)" for r in recommendations)
            answer = "\n".join(lines)
        contributing = sorted(
            {
                r.agent
                for r in inp.upstream
                if r.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL)
            }
        )
        explanation = self._prompt_text(
            "fallback_explanation",
            agents=", ".join(contributing) or "none",
            evidence_count=len(evidence),
        )
        return SynthesisDraft(
            answer=answer,
            key_findings=findings,
            decision_explanation=explanation,
            evidence_markers=markers,
        )
