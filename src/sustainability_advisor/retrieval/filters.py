"""Metadata filters: validation, OData translation and in memory matching."""

from __future__ import annotations

from sustainability_advisor.domain.errors import InputRejectedError
from sustainability_advisor.retrieval.models import DocumentChunk, SearchFilters


def validate_filters(filters: SearchFilters, allowed_fields: list[str]) -> None:
    disallowed = filters.active_fields() - set(allowed_fields)
    if disallowed:
        raise InputRejectedError(
            "Filtering on these fields is not permitted.", details={"fields": sorted(disallowed)}
        )


def _quote(value: str) -> str:
    """Quote an OData string literal, doubling embedded single quotes."""
    return "'" + value.replace("'", "''") + "'"


def to_odata(filters: SearchFilters) -> str | None:
    clauses: list[str] = []
    if filters.doc_types:
        joined = ",".join(t.replace(",", "") for t in filters.doc_types)
        clauses.append(f"search.in(doc_type, {_quote(joined)}, ',')")
    if filters.jurisdiction:
        clauses.append(f"jurisdiction eq {_quote(filters.jurisdiction)}")
    if filters.facility_id:
        clauses.append(f"(facility_id eq {_quote(filters.facility_id)} or facility_id eq null)")
    if filters.effective_on_or_after:
        clauses.append(f"effective_date ge {filters.effective_on_or_after.isoformat()}T00:00:00Z")
    return " and ".join(clauses) if clauses else None


def matches(chunk: DocumentChunk, filters: SearchFilters) -> bool:
    if filters.doc_types and chunk.doc_type not in filters.doc_types:
        return False
    if filters.jurisdiction and chunk.jurisdiction != filters.jurisdiction:
        return False
    if filters.facility_id and chunk.facility_id not in (None, filters.facility_id):
        return False
    return not (
        filters.effective_on_or_after
        and (chunk.effective_date is None or chunk.effective_date < filters.effective_on_or_after)
    )
