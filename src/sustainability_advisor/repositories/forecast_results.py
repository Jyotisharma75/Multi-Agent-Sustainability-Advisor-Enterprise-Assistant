"""Forecast result repository."""

from __future__ import annotations

from sqlalchemy import select

from sustainability_advisor.domain.analysis import ForecastOutcome
from sustainability_advisor.repositories.base import Repository, from_json, to_json


class ForecastResultRepository(Repository):
    def save(self, outcome: ForecastOutcome) -> None:
        self._write(
            self._tables.forecast_results.insert().values(
                forecast_id=outcome.forecast_id,
                facility_id=outcome.facility_id,
                metric=outcome.metric,
                method=outcome.method,
                horizon=outcome.horizon,
                backtest_mape=outcome.backtest_mape,
                payload=to_json(outcome.model_dump(mode="json")),
                generated_at=outcome.generated_at,
            )
        )

    def latest(self, facility_id: str, metric: str) -> ForecastOutcome | None:
        t = self._tables.forecast_results
        stmt = (
            select(t.c.payload)
            .where(t.c.facility_id == facility_id)
            .where(t.c.metric == metric)
            .order_by(t.c.generated_at.desc())
            .limit(1)
        )
        rows = self._fetch(stmt)
        return ForecastOutcome.model_validate(from_json(rows[0][0])) if rows else None
