"""Emission forecasting.

Candidate methods are fitted on the history minus a holdout, scored by mean
absolute percentage error on the holdout, and the best one is refitted on the
full history. Prediction intervals are split conformal: the interval half
width is the configured coverage quantile of the absolute holdout errors of
the selected method. With a short holdout the intervals are approximate and
the outcome says so through ``history_points`` and ``backtest_mape``.

Methods:

* ``holt_winters``: additive trend and seasonality (statsmodels); needs two
  full seasonal cycles
* ``seasonal_naive``: repeats the last observed season
* ``drift``: last value plus the average historical change
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from datetime import date

import numpy as np

from sustainability_advisor.config.models import ForecastConfig
from sustainability_advisor.domain.analysis import ForecastOutcome, ForecastPoint
from sustainability_advisor.domain.errors import DataUnavailableError, InputRejectedError
from sustainability_advisor.repositories import Repositories
from sustainability_advisor.repositories.base import utcnow
from sustainability_advisor.services.periods import add_months, trailing_months

Forecaster = Callable[[np.ndarray, int], np.ndarray]


def seasonal_naive(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    season = history[-period:]
    return np.array([season[i % period] for i in range(horizon)], dtype=float)


def drift(history: np.ndarray, horizon: int) -> np.ndarray:
    slope = (history[-1] - history[0]) / max(len(history) - 1, 1)
    return np.array([history[-1] + slope * (i + 1) for i in range(horizon)], dtype=float)


def holt_winters(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            history, trend="add", seasonal="add", seasonal_periods=period
        ).fit(optimized=True)
        return np.asarray(model.forecast(horizon), dtype=float)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


class ForecastService:
    def __init__(self, config: ForecastConfig, repos: Repositories) -> None:
        self._config = config
        self._repos = repos

    def _methods(self, n: int) -> dict[str, Forecaster]:
        period = self._config.seasonal_period
        methods: dict[str, Forecaster] = {}
        for name in self._config.methods:
            if name == "holt_winters" and n >= 2 * period:
                methods[name] = lambda h, k: holt_winters(h, k, period)
            elif name == "seasonal_naive" and n >= period:
                methods[name] = lambda h, k: seasonal_naive(h, k, period)
            elif name == "drift" and n >= 2:
                methods[name] = drift
        return methods

    def forecast(
        self,
        facility_id: str,
        *,
        horizon: int | None = None,
        scopes: list[int] | None = None,
        as_of: date | None = None,
    ) -> ForecastOutcome:
        cfg = self._config
        steps = horizon or cfg.default_horizon
        if steps > cfg.max_horizon:
            raise InputRejectedError(
                f"The forecast horizon is limited to {cfg.max_horizon} months."
            )
        latest = as_of or self._repos.emissions.latest_period(facility_id)
        if latest is None:
            raise DataUnavailableError(f"No emissions history for facility {facility_id}.")
        window = trailing_months(latest, cfg.history_months)
        series = self._repos.emissions.monthly_totals(facility_id, window.start, window.end, scopes)
        if len(series) < cfg.min_history_points:
            raise DataUnavailableError(
                f"At least {cfg.min_history_points} months of history are required; "
                f"{len(series)} are available."
            )
        values = np.array([p.value for p in series], dtype=float)
        holdout = min(cfg.holdout_points, len(values) // 3) or 1
        train, test = values[:-holdout], values[-holdout:]

        scores: dict[str, float | None] = {}
        errors: dict[str, np.ndarray] = {}
        for name, method in self._methods(len(train)).items():
            try:
                predicted = method(train, holdout)
            except (ValueError, np.linalg.LinAlgError):
                scores[name] = None
                continue
            scores[name] = mape(test, predicted)
            errors[name] = np.abs(test - predicted)
        ranked = sorted(
            (n for n, s in scores.items() if s is not None), key=lambda n: scores[n] or 0.0
        )
        if not ranked:
            raise DataUnavailableError("No forecasting method could be fitted to this history.")
        best = ranked[0]
        forecast = self._methods(len(values))[best](values, steps)
        half_width = float(np.quantile(errors[best], cfg.interval_coverage))

        start = add_months(series[-1].period_start, 1)
        points = [
            ForecastPoint(
                period_start=add_months(start, i),
                value=round(float(v), 4),
                lower=round(max(float(v) - half_width, 0.0), 4),
                upper=round(float(v) + half_width, 4),
            )
            for i, v in enumerate(forecast)
        ]
        projected = float(sum(p.value for p in points))
        target = self._target(facility_id, latest, steps)
        gap = (projected - target) / abs(target) if target else None
        digest = hashlib.sha1(
            f"{facility_id}|{scopes}|{latest}|{steps}|{utcnow().isoformat()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:20]
        outcome = ForecastOutcome(
            forecast_id=f"fc_{digest}",
            facility_id=facility_id,
            metric="co2e_tonnes"
            if not scopes
            else f"co2e_tonnes_scope{'_'.join(map(str, scopes))}",
            method=best,
            horizon=steps,
            history_points=len(values),
            backtest_mape=round(scores[best] or 0.0, 6),
            interval_coverage=cfg.interval_coverage,
            points=points,
            candidate_scores={
                k: (round(v, 6) if v is not None else None) for k, v in scores.items()
            },
            generated_at=utcnow(),
            target_value=round(target, 4) if target is not None else None,
            projected_total=round(projected, 4),
            target_gap_ratio=round(gap, 6) if gap is not None else None,
        )
        self._repos.forecasts.save(outcome)
        return outcome

    def _target(self, facility_id: str, as_of: date, steps: int) -> float | None:
        record = self._repos.metrics.latest(facility_id, self._config.target_metric, as_of)
        if record is None or record.target_value is None:
            return None
        if self._config.target_basis == "per_month":
            return record.target_value * steps
        return record.target_value
