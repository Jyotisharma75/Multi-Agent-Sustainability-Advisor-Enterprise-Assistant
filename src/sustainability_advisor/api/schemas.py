"""HTTP request and response contracts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sustainability_advisor.domain.models import Citation, Recommendation, TokenUsage


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(_Request):
    message: str = Field(min_length=1)
    conversation_id: str | None = Field(default=None, max_length=64)
    facility_ids: list[str] | None = None
    period_start: date | None = None
    period_end: date | None = None

    @model_validator(mode="after")
    def _period(self) -> ChatRequest:
        if (self.period_start is None) != (self.period_end is None):
            raise ValueError("period_start and period_end must be given together")
        if self.period_start and self.period_end and self.period_end < self.period_start:
            raise ValueError("period_end precedes period_start")
        return self


class AnalyzeRequest(_Request):
    facility_ids: list[str] = Field(min_length=1)
    period_start: date
    period_end: date
    question: str | None = None
    kpis: list[str] | None = None
    metrics: list[str] | None = None
    conversation_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _period(self) -> AnalyzeRequest:
        if self.period_end < self.period_start:
            raise ValueError("period_end precedes period_start")
        return self


class ScenarioRequest(_Request):
    facility_id: str
    levers: dict[str, float] = Field(min_length=1)
    baseline_start: date | None = None
    baseline_end: date | None = None
    question: str | None = None
    conversation_id: str | None = Field(default=None, max_length=64)


class CitationView(Citation):
    marker: str


class EvidenceView(BaseModel):
    evidence_id: str
    kind: str
    source: str
    summary: str
    facility_id: str | None
    confidence: float


class AgentTrace(BaseModel):
    agent: str
    status: str
    confidence: float
    latency_ms: float
    iterations: int
    errors: list[str]


class SQLTrace(BaseModel):
    query: str
    generated_by: str
    verified_by: str | None
    verification: str
    row_count: int
    truncated: bool


class AssistantResponse(BaseModel):
    request_id: str
    conversation_id: str
    intent: str | None
    answer: str
    key_findings: list[str]
    decision_explanation: str
    confidence: float
    low_confidence: bool
    grounded: bool
    blocked: bool
    recommendations: list[Recommendation]
    citations: list[CitationView]
    evidence: list[EvidenceView]
    agents: list[AgentTrace]
    sql: SQLTrace | None
    warnings: list[str]
    token_usage: TokenUsage
    latency_ms: float


class MessageView(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime
    payload: dict[str, Any]


class ConversationView(BaseModel):
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageView]
    recommendations: list[Recommendation]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None
    details: dict[str, Any] = Field(default_factory=dict)
