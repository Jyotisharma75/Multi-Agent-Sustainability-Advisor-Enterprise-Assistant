"""Anomaly repository."""

from __future__ import annotations

from datetime import date

from sqlalchemy import delete, select

from sustainability_advisor.domain.analysis import AnomalyFinding
from sustainability_advisor.repositories.base import Repository


class AnomalyRepository(Repository):
    def replace_for_range(
        self,
        facility_id: str,
        metric: str,
        start: date,
        end: date,
        findings: list[AnomalyFinding],
    ) -> None:
        """Store the latest findings for a range, replacing a previous run."""
        t = self._tables.anomalies
        rows = [f.model_dump() for f in findings]
        with self._engine.begin() as conn:
            conn.execute(
                delete(t)
                .where(t.c.facility_id == facility_id)
                .where(t.c.metric == metric)
                .where(t.c.period_start >= start)
                .where(t.c.period_start <= end)
            )
            if rows:
                conn.execute(t.insert(), rows)

    def list_since(self, facility_id: str, since: date) -> list[AnomalyFinding]:
        t = self._tables.anomalies
        stmt = (
            select(t)
            .where(t.c.facility_id == facility_id)
            .where(t.c.period_start >= since)
            .order_by(t.c.period_start)
        )
        return [AnomalyFinding.model_validate(dict(r._mapping)) for r in self._fetch(stmt)]
