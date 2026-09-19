"""Assistant endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from sustainability_advisor.agents.base import RequestEntities
from sustainability_advisor.api.dependencies import get_container, get_request_id, require_api_key
from sustainability_advisor.api.schemas import (
    AnalyzeRequest,
    AssistantResponse,
    ChatRequest,
    ConversationView,
    MessageView,
    ScenarioRequest,
)
from sustainability_advisor.container import Container
from sustainability_advisor.domain.errors import InputRejectedError, NotFoundError
from sustainability_advisor.guardrails.input_validator import InputValidator

router = APIRouter(prefix="/assistant", tags=["assistant"], dependencies=[Depends(require_api_key)])

COMMON_PROMPT = "common"


def _validator(container: Container) -> InputValidator:
    return InputValidator(container.settings.guardrails)


def _conversation_id(container: Container, value: str | None) -> str | None:
    if value is None:
        return None
    return _validator(container).identifiers([value], kind="conversation_id")[0]


@router.post("/chat", response_model=AssistantResponse)
async def chat(
    body: ChatRequest,
    container: Container = Depends(get_container),
    request_id: str = Depends(get_request_id),
) -> AssistantResponse:
    service = container.assistant
    hints = service.default_hints()
    if body.facility_ids:
        hints.facility_ids = _validator(container).identifiers(
            body.facility_ids, kind="facility_id"
        )
    if body.period_start and body.period_end:
        hints.period_start, hints.period_end = body.period_start, body.period_end
        hints.period_explicit = True
    return await service.handle(
        request_id=request_id,
        message=body.message,
        conversation_id=_conversation_id(container, body.conversation_id),
        hints=hints,
        forced_intent=None,
        endpoint="chat",
    )


@router.post("/analyze", response_model=AssistantResponse)
async def analyze(
    body: AnalyzeRequest,
    container: Container = Depends(get_container),
    request_id: str = Depends(get_request_id),
) -> AssistantResponse:
    facility_ids = _validator(container).identifiers(body.facility_ids, kind="facility_id")
    question = body.question or container.prompts.text(
        COMMON_PROMPT,
        "analyze_default_question",
        facilities=", ".join(facility_ids),
        start=body.period_start.isoformat(),
        end=body.period_end.isoformat(),
    )
    hints = RequestEntities(
        facility_ids=facility_ids,
        period_start=body.period_start,
        period_end=body.period_end,
        period_explicit=True,
        kpis=body.kpis,
        metrics=body.metrics,
    )
    return await container.assistant.handle(
        request_id=request_id,
        message=question,
        conversation_id=_conversation_id(container, body.conversation_id),
        hints=hints,
        forced_intent=container.settings.api.analyze_intent,
        endpoint="analyze",
    )


@router.post("/scenario", response_model=AssistantResponse)
async def scenario(
    body: ScenarioRequest,
    container: Container = Depends(get_container),
    request_id: str = Depends(get_request_id),
) -> AssistantResponse:
    facility_id = _validator(container).identifiers([body.facility_id], kind="facility_id")[0]
    levers = container.settings.scenario.levers
    unknown = sorted(set(body.levers) - set(levers))
    if unknown:
        raise InputRejectedError(
            "Unknown scenario levers.", details={"unknown": unknown, "available": sorted(levers)}
        )
    question = body.question or container.prompts.text(
        COMMON_PROMPT,
        "scenario_default_question",
        facility=facility_id,
        levers=", ".join(f"{k}={v}" for k, v in body.levers.items()),
    )
    service = container.assistant
    hints = service.default_hints()
    hints.facility_ids = [facility_id]
    hints.levers = body.levers
    if body.baseline_start and body.baseline_end:
        hints.period_start, hints.period_end = body.baseline_start, body.baseline_end
        hints.period_explicit = True
    return await service.handle(
        request_id=request_id,
        message=question,
        conversation_id=_conversation_id(container, body.conversation_id),
        hints=hints,
        forced_intent=container.settings.api.scenario_intent,
        endpoint="scenario",
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationView)
async def get_conversation(
    conversation_id: str, container: Container = Depends(get_container)
) -> ConversationView:
    conversation_id = _validator(container).identifiers([conversation_id], kind="conversation_id")[
        0
    ]
    conversation = await asyncio.to_thread(
        container.repositories.conversations.get, conversation_id
    )
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    recommendations = await asyncio.to_thread(
        container.repositories.recommendations.list_for_conversation, conversation_id
    )
    return ConversationView(
        conversation_id=conversation.conversation_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageView(
                message_id=m.message_id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                payload=m.payload,
            )
            for m in conversation.messages
        ],
        recommendations=recommendations,
    )
