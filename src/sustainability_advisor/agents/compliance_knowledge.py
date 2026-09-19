"""Compliance Knowledge Agent.

Responsibility: compliance oriented analysis. It restricts retrieval to the
regulatory document types in configuration (regulations, disclosure
standards, permits), then asks a model to extract the obligations those
passages state. Each obligation must cite at least one retrieved passage;
obligations citing unknown passages are dropped. If no model is available,
the cited passages themselves are returned as evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sustainability_advisor.agents.base import AgentInput, RunState
from sustainability_advisor.agents.document_intelligence import (
    DocumentIntelligenceAgent,
    citation_evidence,
)
from sustainability_advisor.agents.evidence import evidence_id
from sustainability_advisor.domain.errors import LLMError
from sustainability_advisor.domain.models import (
    AgentResult,
    AgentStatus,
    Evidence,
    EvidenceKind,
)
from sustainability_advisor.retrieval.models import SearchFilters

PROMPT = "compliance_extraction"


class Obligation(BaseModel):
    statement: str = Field(min_length=5, max_length=600)
    citation_ids: list[str] = Field(min_length=1)
    applies_to: str | None = None


class ObligationExtraction(BaseModel):
    obligations: list[Obligation]


class ComplianceOutput(BaseModel):
    obligations: list[Obligation]
    passages: int
    extraction: str


class ComplianceKnowledgeAgent(DocumentIntelligenceAgent):
    name = "compliance_knowledge"
    output_data_model = ComplianceOutput  # type: ignore[assignment]

    def filters(self, inp: AgentInput) -> SearchFilters:
        base = super().filters(inp)
        return base.model_copy(update={"doc_types": list(self.settings.compliance.document_types)})

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        citations, _ = await self.search(inp, state, self.filters(inp))
        if not citations:
            return self.result(
                state,
                summary="No relevant regulatory passages were found.",
                confidence=0.0,
                status=AgentStatus.PARTIAL,
                data=ComplianceOutput(obligations=[], passages=0, extraction="none").model_dump(),
            )
        markers = {f"C{i}": c for i, c in enumerate(citations, start=1)}
        passages = "\n\n".join(
            f"[{marker}] {c.title}"
            + (f" | section: {c.section}" if c.section else "")
            + (f" | page: {c.page}" if c.page else "")
            + f"\n{c.excerpt}"
            for marker, c in markers.items()
        )
        evidence = citation_evidence(citations, "document_search")
        obligations: list[Obligation] = []
        extraction = "model"
        try:
            extracted = await self.call_llm(
                state, PROMPT, ObligationExtraction, question=inp.query, passages=passages
            )
            for item in extracted.obligations:
                valid = [markers[m].citation_id for m in item.citation_ids if m in markers]
                if valid:
                    obligations.append(item.model_copy(update={"citation_ids": valid}))
        except LLMError:
            extraction = "passages_only"
        for obligation in obligations:
            evidence.append(
                Evidence(
                    evidence_id=evidence_id(
                        EvidenceKind.DOCUMENT, "obligation", obligation.statement
                    ),
                    kind=EvidenceKind.DOCUMENT,
                    source=self.name,
                    summary=obligation.statement,
                    values={},
                    attributes={"applies_to": obligation.applies_to, "type": "obligation"},
                    citation_ids=obligation.citation_ids,
                    confidence=max(
                        c.retrieval_score
                        for c in citations
                        if c.citation_id in obligation.citation_ids
                    ),
                )
            )
        confidence = sum(c.retrieval_score for c in citations) / len(citations)
        return self.result(
            state,
            summary=(
                f"Identified {len(obligations)} cited obligations from {len(citations)} "
                "regulatory passages."
                if obligations
                else f"Retrieved {len(citations)} regulatory passages."
            ),
            confidence=confidence,
            evidence=evidence,
            citations=citations,
            data=ComplianceOutput(
                obligations=obligations, passages=len(citations), extraction=extraction
            ).model_dump(),
        )
