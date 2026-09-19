"""SQL pipeline data contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SQLDraft(BaseModel):
    """What a model must return when asked to write SQL."""

    sql: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    tables_used: list[str] = Field(default_factory=list)


class SQLReview(BaseModel):
    """What a verifier model must return when reviewing SQL."""

    verdict: Literal["approve", "revise", "reject"]
    issues: list[str] = Field(default_factory=list)
    corrected_sql: str | None = None
    confidence: float = Field(ge=0, le=1)


class RoutingDecision(BaseModel):
    primary: str
    verifier: str | None
    complexity: float
    features: dict[str, float]
    reason: str


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    latency_ms: float


class SQLAnswer(BaseModel):
    question: str
    sql: str
    execution_sql: str
    tables: list[str]
    result: QueryResult
    generated_by: str
    verified_by: str | None
    verification: Literal["not_required", "approved", "revised", "disagreement", "unavailable"]
    complexity: float
    confidence: float
    notes: list[str] = Field(default_factory=list)
