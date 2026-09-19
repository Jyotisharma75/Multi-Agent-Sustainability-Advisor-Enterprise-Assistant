"""Offline evaluation harness.

Scores the assistant against a golden set file (``evaluation/golden_set.yaml``):

* intent accuracy of the orchestrator
* retrieval hit rate at k and mean reciprocal rank for known relevant documents
* end to end quality: expected intent and agents, grounding of every answer,
  citation coverage for document questions and evidence integrity of every
  recommendation (each must reference evidence present in the response)

Thresholds live in the golden set file; ``passed`` is true only when every
metric meets its threshold. Run with ``sa-evaluate`` against any environment.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from sustainability_advisor.agents.base import AgentInput
from sustainability_advisor.agents.orchestrator import OrchestratorOutput
from sustainability_advisor.container import Container
from sustainability_advisor.observability.context import new_id


class IntentCase(BaseModel):
    question: str
    intent: str


class RetrievalCase(BaseModel):
    query: str
    document_id: str


class EndToEndCase(BaseModel):
    question: str
    intent: str
    agents: list[str]
    requires_citations: bool = False
    facility_ids: list[str] = Field(default_factory=list)


class Thresholds(BaseModel):
    intent_accuracy: float
    retrieval_hit_rate: float
    retrieval_mrr: float
    grounded_rate: float
    citation_coverage: float
    evidence_integrity: float
    end_to_end_intent_accuracy: float


class GoldenSet(BaseModel):
    retrieval_k: int
    thresholds: Thresholds
    intent_cases: list[IntentCase]
    retrieval_cases: list[RetrievalCase]
    end_to_end_cases: list[EndToEndCase]


class EvaluationReport(BaseModel):
    metrics: dict[str, float]
    thresholds: dict[str, float]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return all(self.metrics[k] >= v for k, v in self.thresholds.items())


def load_golden_set(path: Path) -> GoldenSet:
    with path.open(encoding="utf-8") as handle:
        return GoldenSet.model_validate(yaml.safe_load(handle))


async def _intent_accuracy(
    container: Container, cases: list[IntentCase], failures: list[str]
) -> float:
    orchestrator = container.agents["orchestrator"]
    hints = container.assistant.default_hints()
    correct = 0
    for case in cases:
        result = await orchestrator.execute(
            AgentInput(
                request_id=new_id("eval"),
                conversation_id=new_id("eval"),
                query=case.question,
                intent="",
                entities=hints,
            )
        )
        predicted = OrchestratorOutput.model_validate(result.data).intent if result.data else ""
        if predicted == case.intent:
            correct += 1
        else:
            failures.append(f"intent: {case.question!r} -> {predicted} (expected {case.intent})")
    return correct / len(cases) if cases else 1.0


async def _retrieval(
    container: Container, cases: list[RetrievalCase], k: int, failures: list[str]
) -> tuple[float, float]:
    hits = 0
    reciprocal = 0.0
    for case in cases:
        result = await container.retriever.search(case.query, top_k=k, min_score=0.0)
        ranked = list(dict.fromkeys(c.document_id for c in result.citations))
        if case.document_id in ranked[:k]:
            hits += 1
            reciprocal += 1.0 / (ranked.index(case.document_id) + 1)
        else:
            failures.append(f"retrieval: {case.query!r} missed {case.document_id}")
    count = len(cases) or 1
    return hits / count, reciprocal / count


async def _end_to_end(
    container: Container, cases: list[EndToEndCase], failures: list[str]
) -> dict[str, float]:
    intent_ok = grounded = cited = cited_total = integrity_ok = integrity_total = 0
    for case in cases:
        hints = container.assistant.default_hints()
        hints.facility_ids = case.facility_ids
        response = await container.assistant.handle(
            request_id=new_id("eval"),
            message=case.question,
            conversation_id=None,
            hints=hints,
            forced_intent=None,
            endpoint="evaluation",
        )
        ran = {a.agent for a in response.agents}
        if response.intent == case.intent and set(case.agents) <= ran:
            intent_ok += 1
        else:
            failures.append(f"e2e: {case.question!r} intent={response.intent} agents={sorted(ran)}")
        if response.grounded and response.answer:
            grounded += 1
        else:
            failures.append(f"e2e: {case.question!r} answer not grounded")
        if case.requires_citations:
            cited_total += 1
            markers_ok = all(c.marker for c in response.citations)
            if response.citations and markers_ok:
                cited += 1
            else:
                failures.append(f"e2e: {case.question!r} returned no citations")
        evidence_ids = {e.evidence_id for e in response.evidence}
        for rec in response.recommendations:
            integrity_total += 1
            if rec.evidence and set(rec.evidence) <= evidence_ids:
                integrity_ok += 1
            else:
                failures.append(f"e2e: recommendation {rec.recommendation_id} lacks evidence")
    count = len(cases) or 1
    return {
        "end_to_end_intent_accuracy": intent_ok / count,
        "grounded_rate": grounded / count,
        "citation_coverage": cited / cited_total if cited_total else 1.0,
        "evidence_integrity": integrity_ok / integrity_total if integrity_total else 1.0,
    }


async def evaluate(container: Container, golden: GoldenSet) -> EvaluationReport:
    failures: list[str] = []
    metrics: dict[str, float] = {}
    metrics["intent_accuracy"] = await _intent_accuracy(container, golden.intent_cases, failures)
    hit_rate, mrr = await _retrieval(
        container, golden.retrieval_cases, golden.retrieval_k, failures
    )
    metrics["retrieval_hit_rate"] = hit_rate
    metrics["retrieval_mrr"] = mrr
    metrics.update(await _end_to_end(container, golden.end_to_end_cases, failures))
    return EvaluationReport(
        metrics={k: round(v, 4) for k, v in metrics.items()},
        thresholds=golden.thresholds.model_dump(),
        failures=failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the assistant against a golden set.")
    parser.add_argument("--golden", type=Path, default=Path("evaluation/golden_set.yaml"))
    args = parser.parse_args()

    from sustainability_advisor.container import build_container

    async def run() -> bool:
        container = build_container()
        await container.startup()
        try:
            report = await evaluate(container, load_golden_set(args.golden))
        finally:
            container.close()
        print(yaml.safe_dump(report.model_dump(), sort_keys=False))
        return report.passed

    raise SystemExit(0 if asyncio.run(run()) else 1)


if __name__ == "__main__":
    main()
