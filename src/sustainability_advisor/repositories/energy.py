"""Energy repository."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel
from sqlalchemy import func, select

from sustainability_advisor.domain.models import EnergyRecord, SeriesPoint
from sustainability_advisor.repositories.base import Repository


class EnergyTotals(BaseModel):
    consumption_mwh: float
    renewable_mwh: float
    cost: float | None
    currency: str | None


class EnergyRepository(Repository):
    def records(self, facility_id: str, start: date, end: date) -> list[EnergyRecord]:
        t = self._tables.energy
        stmt = (
            select(t)
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
            .order_by(t.c.period_start, t.c.energy_type)
        )
        return [
            EnergyRecord.model_validate({k: v for k, v in r._mapping.items() if k != "energy_id"})
            for r in self._fetch(stmt)
        ]

    def totals(self, facility_id: str, start: date, end: date) -> EnergyTotals | None:
        t = self._tables.energy
        stmt = (
            select(
                func.sum(t.c.consumption_mwh),
                func.sum(t.c.renewable_mwh),
                func.sum(t.c.cost),
                func.count(t.c.cost),
                func.count(),
                func.min(t.c.currency),
                func.max(t.c.currency),
            )
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
        )
        row = self._fetch(stmt)[0]
        if row[0] is None:
            return None
        # A cost total is only meaningful when every row reports a cost in a
        # single currency; otherwise it is reported as unknown.
        cost_complete = int(row[3]) == int(row[4]) and row[5] == row[6]
        return EnergyTotals(
            consumption_mwh=float(row[0]),
            renewable_mwh=float(row[1] or 0.0),
            cost=float(row[2]) if cost_complete and row[2] is not None else None,
            currency=str(row[5]) if cost_complete and row[5] is not None else None,
        )

    def monthly_consumption(self, facility_id: str, start: date, end: date) -> list[SeriesPoint]:
        t = self._tables.energy
        stmt = (
            select(t.c.period_start, func.sum(t.c.consumption_mwh))
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
            .group_by(t.c.period_start)
            .order_by(t.c.period_start)
        )
        return [SeriesPoint(period_start=r[0], value=float(r[1])) for r in self._fetch(stmt)]

    def add_many(self, records: list[EnergyRecord]) -> None:
        self._write(self._tables.energy.insert(), [r.model_dump() for r in records])
