"""Document Intelligence Agent.

Responsibility: find the passages in enterprise documents (policies,
procedures, audit reports, site plans) that bear on the question. It runs
hybrid retrieval first; if that yields too few relevant passages it tries a
pure vector search to catch paraphrased content. Every passage it returns is
a citation with document, section, page and score.
"""

from __future__ import annotations

from pydantic import BaseModel

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RunState
from sustainability_advisor.agents.evidence import evidence_id
from sustainability_advisor.domain.models import (
    AgentResult,
    AgentStatus,
    Citation,
    Evidence,
    EvidenceKind,
)
from sustainability_advisor.retrieval.models import RetrievalResult, SearchFilters
from sustainability_advisor.tools.retrieval_tools import DocumentSearchInput

SEARCH_TOOL = "document_search"
VECTOR_TOOL = "vector_search"


class DocumentAgentOutput(BaseModel):
    passages: int
    modes: list[str]
    discarded_low_score: int
    discarded_suspicious: int


def citation_evidence(citations: list[Citation], source: str) -> list[Evidence]:
    return [
        Evidence(
            evidence_id=evidence_id(EvidenceKind.DOCUMENT, c.chunk_id),
            kind=EvidenceKind.DOCUMENT,
            source=source,
            summary=f"{c.title}"
            + (f", section {c.section}" if c.section else "")
            + (f", page {c.page}" if c.page else "")
            + f": {c.excerpt}",
            facility_id=None,
            values={"retrieval_score": c.retrieval_score},
            attributes={"document_id": c.document_id, "doc_type": c.doc_type},
            citation_ids=[c.citation_id],
            confidence=c.retrieval_score,
        )
        for c in citations
    ]


class DocumentIntelligenceAgent(BaseAgent):
    name = "document_intelligence"
    output_data_model = DocumentAgentOutput

    def filters(self, inp: AgentInput) -> SearchFilters:
        facility = inp.entities.facility_ids[0] if len(inp.entities.facility_ids) == 1 else None
        allowed = set(self.settings.retrieval.allowed_filter_fields)
        return SearchFilters(
            facility_id=facility if "facility_id" in allowed else None,
            jurisdiction=inp.entities.jurisdiction if "jurisdiction" in allowed else None,
        )

    async def search(
        self, inp: AgentInput, state: RunState, filters: SearchFilters
    ) -> tuple[list[Citation], DocumentAgentOutput]:
        result = await self.call_tool(
            state,
            SEARCH_TOOL,
            DocumentSearchInput(query=inp.query, filters=filters),
            RetrievalResult,
        )
        results = [result]
        wanted = max(1, self.settings.retrieval.top_k // 2)
        if len(result.citations) < wanted and VECTOR_TOOL in self.spec.allowed_tools:
            results.append(
                await self.call_tool(
                    state,
                    VECTOR_TOOL,
                    DocumentSearchInput(query=inp.query, filters=filters),
                    RetrievalResult,
                )
            )
        citations: dict[str, Citation] = {}
        for r in results:
            for c in r.citations:
                existing = citations.get(c.citation_id)
                if existing is None or c.retrieval_score > existing.retrieval_score:
                    citations[c.citation_id] = c
        ranked = sorted(citations.values(), key=lambda c: c.retrieval_score, reverse=True)
        ranked = ranked[: self.settings.retrieval.top_k]
        summary = DocumentAgentOutput(
            passages=len(ranked),
            modes=[r.mode for r in results],
            discarded_low_score=sum(r.discarded_low_score for r in results),
            discarded_suspicious=sum(r.discarded_suspicious for r in results),
        )
        return ranked, summary

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        citations, output = await self.search(inp, state, self.filters(inp))
        if not citations:
            return self.result(
                state,
                summary="No sufficiently relevant document passages were found.",
                confidence=0.0,
                status=AgentStatus.PARTIAL,
                data=output.model_dump(),
            )
        confidence = sum(c.retrieval_score for c in citations) / len(citations)
        return self.result(
            state,
            summary=f"Found {len(citations)} relevant passages in "
            f"{len({c.document_id for c in citations})} documents.",
            confidence=confidence,
            evidence=citation_evidence(citations, SEARCH_TOOL),
            citations=citations,
            data=output.model_dump(),
        )
