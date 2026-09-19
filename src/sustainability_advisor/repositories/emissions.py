"""Emission repository."""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select

from sustainability_advisor.domain.models import EmissionRecord, SeriesPoint
from sustainability_advisor.repositories.base import Repository


class EmissionRepository(Repository):
    def records(
        self,
        facility_id: str,
        start: date,
        end: date,
        scopes: list[int] | None = None,
    ) -> list[EmissionRecord]:
        t = self._tables.emissions
        stmt = (
            select(t)
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
            .order_by(t.c.period_start, t.c.scope)
        )
        if scopes:
            stmt = stmt.where(t.c.scope.in_(scopes))
        rows = self._fetch(stmt)
        return [
            EmissionRecord.model_validate(
                {k: v for k, v in r._mapping.items() if k != "emission_id"}
            )
            for r in rows
        ]

    def monthly_totals(
        self,
        facility_id: str,
        start: date,
        end: date,
        scopes: list[int] | None = None,
    ) -> list[SeriesPoint]:
        t = self._tables.emissions
        stmt = (
            select(t.c.period_start, func.sum(t.c.co2e_tonnes))
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
            .group_by(t.c.period_start)
            .order_by(t.c.period_start)
        )
        if scopes:
            stmt = stmt.where(t.c.scope.in_(scopes))
        return [SeriesPoint(period_start=r[0], value=float(r[1])) for r in self._fetch(stmt)]

    def totals_by_scope(self, facility_id: str, start: date, end: date) -> dict[int, float]:
        t = self._tables.emissions
        stmt = (
            select(t.c.scope, func.sum(t.c.co2e_tonnes))
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
            .group_by(t.c.scope)
        )
        return {int(r[0]): float(r[1]) for r in self._fetch(stmt)}

    def latest_period(self, facility_id: str | None = None) -> date | None:
        t = self._tables.emissions
        stmt = select(func.max(t.c.period_start))
        if facility_id:
            stmt = stmt.where(t.c.facility_id == facility_id)
        rows = self._fetch(stmt)
        value = rows[0][0] if rows else None
        return value if isinstance(value, date) else None

    def add_many(self, records: list[EmissionRecord]) -> None:
        self._write(self._tables.emissions.insert(), [r.model_dump() for r in records])
