"""Application service behind the assistant endpoints.

Runs one request through the orchestration graph and takes care of what
surrounds it: conversation history, telemetry, audit events, persistence of
recommendations and the mapping from graph state to the response contract.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date
from typing import Any

from sustainability_advisor.agents.base import RequestEntities
from sustainability_advisor.agents.response_synthesizer import collect
from sustainability_advisor.api.schemas import (
    AgentTrace,
    AssistantResponse,
    CitationView,
    EvidenceView,
    SQLTrace,
)
from sustainability_advisor.config.models import Settings
from sustainability_advisor.domain.errors import DependencyUnavailableError
from sustainability_advisor.domain.models import AgentResult, TokenUsage
from sustainability_advisor.graph.builder import initial_state
from sustainability_advisor.graph.state import AssistantState
from sustainability_advisor.observability import telemetry
from sustainability_advisor.observability.context import bind_conversation, new_id
from sustainability_advisor.observability.logging import get_logger
from sustainability_advisor.repositories import Repositories
from sustainability_advisor.services.audit_service import AuditService, ConversationService
from sustainability_advisor.services.periods import month_end, trailing_months

logger = get_logger(__name__)


class AssistantService:
    def __init__(
        self,
        settings: Settings,
        graph: Any,
        repos: Repositories,
        audit: AuditService,
        conversations: ConversationService,
    ) -> None:
        self._settings = settings
        self._graph = graph
        self._repos = repos
        self._audit = audit
        self._conversations = conversations

    def default_hints(self) -> RequestEntities:
        window = trailing_months(date.today(), self._settings.orchestration.default_lookback_months)
        return RequestEntities(period_start=window.start, period_end=month_end(window.end))

    async def handle(
        self,
        *,
        request_id: str,
        message: str,
        conversation_id: str | None,
        hints: RequestEntities,
        forced_intent: str | None,
        endpoint: str,
    ) -> AssistantResponse:
        started = time.perf_counter()
        conversation = conversation_id or new_id("conv")
        bind_conversation(conversation)
        tracker = telemetry.start_request(request_id, conversation)
        await self._conversations.ensure(conversation)
        history = [
            f"{m.role}: {m.content}" for m in await self._conversations.history(conversation)
        ]
        await self._audit.record(
            "request.received",
            "user",
            {"endpoint": endpoint, "forced_intent": forced_intent, "chars": len(message)},
        )
        state = initial_state(
            request_id=request_id,
            conversation_id=conversation,
            query=message,
            history=history,
            hints=hints,
            forced_intent=forced_intent,
        )
        try:
            final_state: AssistantState = await asyncio.wait_for(
                self._graph.ainvoke(
                    state, config={"recursion_limit": self._settings.orchestration.recursion_limit}
                ),
                timeout=self._settings.api.request_timeout_seconds,
            )
        except TimeoutError as exc:
            await self._audit.record("request.timeout", "system", {})
            raise DependencyUnavailableError("The assistant did not finish in time.") from exc

        response = self._response(final_state, request_id, conversation, tracker, started)
        await self._persist(message, response, endpoint)
        tracker.final_confidence = response.confidence
        tracker.citation_count = len(response.citations)
        logger.info(
            "response.final",
            intent=response.intent,
            confidence=response.confidence,
            low_confidence=response.low_confidence,
            blocked=response.blocked,
            citations=len(response.citations),
            recommendations=len(response.recommendations),
            prompt_tokens=response.token_usage.prompt_tokens,
            completion_tokens=response.token_usage.completion_tokens,
            latency_ms=response.latency_ms,
        )
        return response

    def _response(
        self,
        state: AssistantState,
        request_id: str,
        conversation: str,
        tracker: telemetry.RequestTelemetry,
        started: float,
    ) -> AssistantResponse:
        final = state.get("final", {})
        results: dict[str, AgentResult] = dict(state.get("results", {}))
        traces: list[AgentResult] = []
        if "orchestrator_result" in state:
            traces.append(state["orchestrator_result"])
        traces.extend(results.values())
        if "synthesis" in state:
            traces.append(state["synthesis"])
        evidence, citations, recommendations = collect(list(results.values()))
        markers = {v: k for k, v in final.get("citation_markers", {}).items()}
        sql_result = results.get("sql_agent")
        sql_trace = None
        if sql_result is not None and sql_result.data.get("sql"):
            data = sql_result.data
            sql_trace = SQLTrace(
                query=data["sql"],
                generated_by=data["generated_by"],
                verified_by=data.get("verified_by"),
                verification=data["verification"],
                row_count=data["row_count"],
                truncated=data["truncated"],
            )
        usage = tracker.token_usage
        return AssistantResponse(
            request_id=request_id,
            conversation_id=conversation,
            intent=state.get("intent"),
            answer=final.get("answer", ""),
            key_findings=final.get("key_findings", []),
            decision_explanation=final.get("decision_explanation", ""),
            confidence=final.get("confidence", 0.0),
            low_confidence=final.get("low_confidence", True),
            grounded=bool(final.get("grounded", False)),
            blocked=bool(final.get("blocked")),
            recommendations=recommendations,
            citations=[
                CitationView(**c.model_dump(), marker=markers.get(c.citation_id, ""))
                for c in citations
            ],
            evidence=[
                EvidenceView(
                    evidence_id=e.evidence_id,
                    kind=e.kind.value,
                    source=e.source,
                    summary=e.summary,
                    facility_id=e.facility_id,
                    confidence=e.confidence,
                )
                for e in evidence
            ],
            agents=[
                AgentTrace(
                    agent=r.agent,
                    status=r.status.value,
                    confidence=r.confidence,
                    latency_ms=r.latency_ms,
                    iterations=r.iterations,
                    errors=r.errors,
                )
                for r in traces
            ],
            sql=sql_trace,
            warnings=final.get("warnings", []),
            token_usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
            ),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

    async def _persist(self, message: str, response: AssistantResponse, endpoint: str) -> None:
        conversation = response.conversation_id
        await self._conversations.append(
            conversation, "user", message, {"request_id": response.request_id, "endpoint": endpoint}
        )
        await self._conversations.append(
            conversation,
            "assistant",
            response.answer,
            {
                "request_id": response.request_id,
                "intent": response.intent,
                "confidence": response.confidence,
                "low_confidence": response.low_confidence,
                "citations": [c.citation_id for c in response.citations],
                "recommendations": [r.recommendation_id for r in response.recommendations],
            },
        )
        if response.recommendations:
            try:
                await asyncio.to_thread(
                    self._repos.recommendations.save_many, conversation, response.recommendations
                )
            except DependencyUnavailableError:
                logger.error("recommendations.persist_failed", conversation_id=conversation)
        await self._audit.record(
            "request.blocked" if response.blocked else "response.completed",
            "assistant",
            {
                "intent": response.intent,
                "confidence": response.confidence,
                "agents": {a.agent: a.status for a in response.agents},
                "citations": len(response.citations),
                "recommendations": [r.recommendation_id for r in response.recommendations],
                "warnings": response.warnings,
            },
        )
