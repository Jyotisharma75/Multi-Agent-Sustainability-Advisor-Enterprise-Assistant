"""Recommendation Agent.

Responsibility: turn evidence from upstream agents into ranked corrective
actions. It does not look at data itself: it passes the evidence it received
to the recommendation ranking tool, which only proposes actions for issues
that evidence demonstrates. When no issue is demonstrated it says so.
"""

from __future__ import annotations

from pydantic import BaseModel

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RunState
from sustainability_advisor.domain.models import AgentResult, AgentStatus, Evidence, Recommendation
from sustainability_advisor.tools.analysis_tools import (
    RecommendationRankingInput,
    RecommendationRankingOutput,
)

RANKING_TOOL = "recommendation_ranking"


class RecommendationAgentOutput(BaseModel):
    recommendations: list[Recommendation]
    issues_detected: int
    evidence_considered: int


class RecommendationAgent(BaseAgent):
    name = "recommendation"
    output_data_model = RecommendationAgentOutput

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        evidence: dict[str, Evidence] = {}
        for upstream in inp.upstream:
            for item in upstream.evidence:
                evidence.setdefault(item.evidence_id, item)
        if not evidence:
            return self.result(
                state,
                summary="No upstream evidence was available to base recommendations on.",
                confidence=0.0,
                status=AgentStatus.SKIPPED,
                data=RecommendationAgentOutput(
                    recommendations=[], issues_detected=0, evidence_considered=0
                ).model_dump(mode="json"),
            )
        output = await self.call_tool(
            state,
            RANKING_TOOL,
            RecommendationRankingInput(evidence=list(evidence.values())),
            RecommendationRankingOutput,
        )
        recs = output.recommendations
        if not recs:
            summary = "The evidence shows no issue that requires corrective action."
            confidence = sum(e.confidence for e in evidence.values()) / len(evidence)
        else:
            summary = (
                f"{len(recs)} recommendations ranked from {output.issues_detected} evidenced "
                "issues."
            )
            confidence = sum(r.confidence for r in recs) / len(recs)
        return self.result(
            state,
            summary=summary,
            confidence=confidence,
            recommendations=recs,
            data=RecommendationAgentOutput(
                recommendations=recs,
                issues_detected=output.issues_detected,
                evidence_considered=len(evidence),
            ).model_dump(mode="json"),
        )
