"""Sustainability metric repository (targets, reference values, reported KPIs)."""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from sustainability_advisor.domain.models import SustainabilityMetric
from sustainability_advisor.repositories.base import Repository


class SustainabilityMetricRepository(Repository):
    def values(
        self, facility_id: str, metric_name: str, start: date, end: date
    ) -> list[SustainabilityMetric]:
        t = self._tables.sustainability_metrics
        stmt = (
            select(t)
            .where(t.c.facility_id == facility_id)
            .where(t.c.metric_name == metric_name)
            .where(t.c.period_start >= start)
            .where(t.c.period_start <= end)
            .order_by(t.c.period_start)
        )
        return [
            SustainabilityMetric.model_validate(
                {k: v for k, v in r._mapping.items() if k != "metric_id"}
            )
            for r in self._fetch(stmt)
        ]

    def latest(
        self, facility_id: str, metric_name: str, as_of: date
    ) -> SustainabilityMetric | None:
        """Return the most recent record starting on or before ``as_of``."""
        t = self._tables.sustainability_metrics
        stmt = (
            select(t)
            .where(t.c.facility_id == facility_id)
            .where(t.c.metric_name == metric_name)
            .where(t.c.period_start <= as_of)
            .order_by(t.c.period_start.desc())
            .limit(1)
        )
        rows = self._fetch(stmt)
        if not rows:
            return None
        return SustainabilityMetric.model_validate(
            {k: v for k, v in rows[0]._mapping.items() if k != "metric_id"}
        )

    def add_many(self, metrics: list[SustainabilityMetric]) -> None:
        self._write(self._tables.sustainability_metrics.insert(), [m.model_dump() for m in metrics])
