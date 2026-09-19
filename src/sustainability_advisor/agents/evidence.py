"""Evidence construction helpers shared by agents."""

from __future__ import annotations

import hashlib
from typing import Any

from sustainability_advisor.domain.analysis import KpiValue
from sustainability_advisor.domain.models import Evidence, EvidenceKind


def evidence_id(kind: EvidenceKind, *parts: object) -> str:
    digest = hashlib.sha1(
        "|".join(str(p) for p in (kind.value, *parts)).encode(), usedforsecurity=False
    ).hexdigest()[:10]
    return f"ev_{kind.value}_{digest}"


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "not reported"
    return f"{value:,.{digits}g}" if abs(value) < 1000 else f"{value:,.1f}"


def kpi_evidence(value: KpiValue, *, source: str, confidence: float) -> Evidence:
    target = f" against a target of {fmt(value.target)}" if value.target is not None else ""
    status = value.status.replace("_", " ")
    summary = (
        f"{value.kpi.replace('_', ' ')} for {value.facility_id} from {value.period_start} to "
        f"{value.period_end} was {fmt(value.value)} {value.unit}{target} ({status})."
    )
    if value.note:
        summary += f" {value.note}"
    return Evidence(
        evidence_id=evidence_id(
            EvidenceKind.KPI, value.facility_id, value.kpi, value.period_start, value.period_end
        ),
        kind=EvidenceKind.KPI,
        source=source,
        summary=summary,
        facility_id=value.facility_id,
        values={"value": value.value, "target": value.target, "gap_ratio": value.gap_ratio},
        attributes={
            "kpi": value.kpi,
            "status": value.status,
            "unit": value.unit,
            "period_start": value.period_start.isoformat(),
            "period_end": value.period_end.isoformat(),
            "sources": value.sources,
        },
        confidence=confidence if value.value is not None else 0.0,
    )


def compact_rows(columns: list[str], rows: list[list[Any]], limit: int) -> list[dict[str, Any]]:
    return [dict(zip(columns, row, strict=False)) for row in rows[:limit]]
