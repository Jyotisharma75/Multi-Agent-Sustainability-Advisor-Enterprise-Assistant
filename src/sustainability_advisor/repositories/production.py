"""Production repository."""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select

from sustainability_advisor.domain.models import ProductionRecord, SeriesPoint
from sustainability_advisor.repositories.base import Repository


class ProductionRepository(Repository):
    def total_output(self, facility_id: str, start: date, end: date) -> float | None:
        t = self._tables.production
        stmt = (
            select(func.sum(t.c.output_quantity))
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
        )
        value = self._fetch(stmt)[0][0]
        return float(value) if value is not None else None

    def monthly_output(self, facility_id: str, start: date, end: date) -> list[SeriesPoint]:
        t = self._tables.production
        stmt = (
            select(t.c.period_start, func.sum(t.c.output_quantity))
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
            .group_by(t.c.period_start)
            .order_by(t.c.period_start)
        )
        return [SeriesPoint(period_start=r[0], value=float(r[1])) for r in self._fetch(stmt)]

    def add_many(self, records: list[ProductionRecord]) -> None:
        self._write(self._tables.production.insert(), [r.model_dump() for r in records])
