"""Explicit orchestration state.

Every node reads from and writes to this typed mapping; nothing is passed
between nodes any other way. Keys are overwritten by the node that returns
them. The state is in process only and is discarded after the response.
"""

from __future__ import annotations

from typing import Any, TypedDict

from sustainability_advisor.agents.base import RequestEntities
from sustainability_advisor.agents.orchestrator import TaskPlan
from sustainability_advisor.domain.models import AgentResult
from sustainability_advisor.guardrails.confidence import ValidationReport


class AssistantState(TypedDict, total=False):
    # request
    request_id: str
    conversation_id: str
    query: str
    history: list[str]
    hints: RequestEntities
    forced_intent: str | None

    # guard outcome
    blocked: bool
    block_code: str
    block_message: str

    # orchestration
    intent: str
    intent_confidence: float
    classification_method: str
    entities: RequestEntities
    plan: TaskPlan
    orchestrator_result: AgentResult

    # execution
    pending: list[str]
    results: dict[str, AgentResult]
    iteration: int
    replans: int
    validation: ValidationReport

    # output
    synthesis: AgentResult
    final: dict[str, Any]
    notes: list[str]
