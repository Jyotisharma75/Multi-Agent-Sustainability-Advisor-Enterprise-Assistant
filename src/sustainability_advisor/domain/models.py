"""Domain models shared by repositories, services, tools and agents."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- persisted entities ---------------------------------------------------


class Facility(Model):
    facility_id: str
    name: str
    region: str | None = None
    country: str | None = None
    industry: str | None = None
    capacity: float | None = None
    capacity_unit: str | None = None
    is_active: bool = True


class EmissionRecord(Model):
    facility_id: str
    period_start: date
    period_end: date
    scope: int
    category: str | None = None
    co2e_tonnes: float
    source: str | None = None
    methodology: str | None = None


class EnergyRecord(Model):
    facility_id: str
    period_start: date
    period_end: date
    energy_type: str
    consumption_mwh: float
    renewable_mwh: float
    cost: float | None = None
    currency: str | None = None


class ProductionRecord(Model):
    facility_id: str
    period_start: date
    period_end: date
    product: str | None = None
    output_quantity: float
    output_unit: str | None = None


class Sensor(Model):
    sensor_id: str
    facility_id: str
    sensor_type: str
    unit: str | None = None
    location: str | None = None


class SensorReading(Model):
    sensor_id: str
    recorded_at: datetime
    value: float


class SustainabilityMetric(Model):
    facility_id: str
    metric_name: str
    period_start: date
    period_end: date
    value: float | None
    unit: str | None = None
    target_value: float | None = None
    source: str | None = None


class DocumentRecord(Model):
    document_id: str
    title: str
    doc_type: str
    source_uri: str
    jurisdiction: str | None = None
    facility_id: str | None = None
    effective_date: date | None = None
    version: str | None = None
    checksum: str
    indexed_at: datetime | None = None


class AuditEvent(Model):
    event_id: str
    conversation_id: str | None
    request_id: str | None
    event_type: str
    actor: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ConversationMessage(Model):
    message_id: str
    conversation_id: str
    role: str
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Conversation(Model):
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessage] = Field(default_factory=list)


# -- analysis values --------------------------------------------------------


class SeriesPoint(Model):
    period_start: date
    value: float


class TokenUsage(Model):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class EvidenceKind(StrEnum):
    SQL_RESULT = "sql_result"
    KPI = "kpi"
    FORECAST = "forecast"
    ANOMALY = "anomaly"
    SCENARIO = "scenario"
    DOCUMENT = "document"
    RECOMMENDATION = "recommendation"


class Evidence(Model):
    """A verifiable fact produced by a tool, never by free text generation."""

    evidence_id: str
    kind: EvidenceKind
    source: str
    summary: str
    facility_id: str | None = None
    values: dict[str, float | None] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    citation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class Citation(Model):
    """Identifies the exact document passage that supports a statement."""

    citation_id: str
    document_id: str
    title: str
    source_uri: str
    doc_type: str | None = None
    jurisdiction: str | None = None
    effective_date: date | None = None
    section: str | None = None
    page: int | None = None
    chunk_id: str
    retrieval_score: float
    excerpt: str


class Estimate(Model):
    """A value that may be unknown. ``value`` is null when no source supports it."""

    value: float | None
    unit: str | None = None
    basis: str
    uncertainty: str


class Recommendation(Model):
    recommendation_id: str
    facility_id: str | None
    issue: str
    severity: str
    business_impact: str
    estimated_complexity: str
    estimated_cost: Estimate
    expected_sustainability_impact: Estimate
    priority: str
    priority_score: float
    evidence: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    actions: list[str] = Field(min_length=1)


class AgentStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentResult(Model):
    agent: str
    status: AgentStatus
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    errors: list[str] = Field(default_factory=list)
    iterations: int = 0
    latency_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
