"""Retrieval data contracts."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from sustainability_advisor.domain.models import Citation


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    title: str
    source_uri: str
    doc_type: str
    jurisdiction: str | None = None
    facility_id: str | None = None
    effective_date: date | None = None
    section: str | None = None
    page: int | None = None
    ordinal: int
    content: str


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float = Field(ge=0, le=1)
    keyword_rank: int | None = None
    vector_rank: int | None = None


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_types: list[str] | None = None
    jurisdiction: str | None = None
    facility_id: str | None = None
    effective_on_or_after: date | None = None

    def active_fields(self) -> set[str]:
        mapping = {
            "doc_types": "doc_type",
            "jurisdiction": "jurisdiction",
            "facility_id": "facility_id",
            "effective_on_or_after": "effective_date",
        }
        return {mapping[k] for k, v in self.model_dump().items() if v not in (None, [])}


class RetrievalResult(BaseModel):
    query: str
    mode: str
    chunks: list[RetrievedChunk]
    citations: list[Citation]
    discarded_low_score: int = 0
    discarded_suspicious: int = 0
