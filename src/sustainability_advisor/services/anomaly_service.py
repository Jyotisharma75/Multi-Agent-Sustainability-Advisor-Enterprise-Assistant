"""Anomaly detection on emissions, energy, production and sensor series.

Two complementary detectors:

* robust z score: each point is compared with the median and median absolute
  deviation of the trailing window before it, so one outlier does not hide
  itself by inflating the spread
* isolation forest over (value, change from previous) when enough points
  exist; it catches points that are unusual in combination rather than size

A point is reported when either detector flags it. Severity comes from the
absolute robust z score mapped onto configured bands.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

import numpy as np

from sustainability_advisor.config.models import AnomalyConfig
from sustainability_advisor.domain.analysis import AnomalyFinding
from sustainability_advisor.domain.errors import DataUnavailableError, InputRejectedError
from sustainability_advisor.domain.models import SeriesPoint
from sustainability_advisor.repositories import Repositories
from sustainability_advisor.repositories.base import utcnow
from sustainability_advisor.services.periods import Period, add_months, month_start, trailing_months

MAD_SCALE = 0.6745
METRICS = ("emissions", "energy", "production")
SENSOR_PREFIX = "sensor:"


class AnomalyService:
    def __init__(self, config: AnomalyConfig, repos: Repositories) -> None:
        self._config = config
        self._repos = repos

    def _series(self, facility_id: str, metric: str, period: Period) -> list[SeriesPoint]:
        repos = self._repos
        if metric == "emissions":
            return repos.emissions.monthly_totals(facility_id, period.start, period.end)
        if metric == "energy":
            return repos.energy.monthly_consumption(facility_id, period.start, period.end)
        if metric == "production":
            return repos.production.monthly_output(facility_id, period.start, period.end)
        if metric.startswith(SENSOR_PREFIX):
            sensor_id = metric[len(SENSOR_PREFIX) :]
            sensor = repos.sensors.get(sensor_id)
            if sensor is None or sensor.facility_id != facility_id:
                raise InputRejectedError(f"Sensor {sensor_id} does not belong to {facility_id}.")
            readings = repos.sensors.readings(
                sensor_id,
                datetime.combine(period.start, datetime.min.time()),
                datetime.combine(period.end, datetime.max.time()),
            )
            return [SeriesPoint(period_start=r.recorded_at.date(), value=r.value) for r in readings]
        raise InputRejectedError(
            f"Unknown anomaly metric {metric!r}",
            details={"supported": [*METRICS, f"{SENSOR_PREFIX}<sensor_id>"]},
        )

    def default_period(self, facility_id: str) -> Period:
        latest = self._repos.emissions.latest_period(facility_id)
        if latest is None:
            raise DataUnavailableError(f"No data for facility {facility_id}.")
        return trailing_months(latest, self._config.history_months)

    def robust_z(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the robust z score and expected value (trailing median) per point."""
        window = self._config.rolling_window
        z = np.zeros(len(values))
        expected = values.astype(float).copy()
        for i in range(len(values)):
            history = values[max(0, i - window) : i]
            if len(history) < self._config.min_points - 1:
                continue
            median = float(np.median(history))
            mad = float(np.median(np.abs(history - median)))
            expected[i] = median
            if mad == 0:
                spread = float(np.std(history)) or abs(median) * 1e-6 or 1.0
                z[i] = (values[i] - median) / spread
            else:
                z[i] = MAD_SCALE * (values[i] - median) / mad
        return z, expected

    def _isolation_flags(self, values: np.ndarray) -> np.ndarray:
        cfg = self._config
        if not cfg.isolation_forest_enabled or len(values) < cfg.isolation_forest_min_points:
            return np.zeros(len(values), dtype=bool)
        from sklearn.ensemble import IsolationForest

        changes = np.concatenate([[0.0], np.diff(values)])
        features = np.column_stack([values, changes])
        model = IsolationForest(
            contamination=cfg.isolation_forest_contamination, random_state=cfg.random_state
        )
        return np.asarray(model.fit_predict(features) == -1)

    def severity(self, score: float) -> str:
        for band in sorted(self._config.severity_bands, key=lambda b: b.min_score, reverse=True):
            if score >= band.min_score:
                return band.name
        return self._config.severity_bands[-1].name

    def detect(
        self, facility_id: str, metric: str, period: Period | None = None
    ) -> tuple[list[AnomalyFinding], int]:
        """Return anomalies and the number of points analysed."""
        window = period or self.default_period(facility_id)
        # Detection needs a trailing baseline, so history before the requested
        # window is read; only findings inside the window are reported.
        lookback = Period(
            start=add_months(month_start(window.start), -self._config.rolling_window),
            end=window.end,
        )
        series = self._series(facility_id, metric, lookback)
        if len(series) < self._config.min_points:
            raise DataUnavailableError(
                f"At least {self._config.min_points} points are required; {len(series)} found."
            )
        values = np.array([p.value for p in series], dtype=float)
        z, expected = self.robust_z(values)
        iso = self._isolation_flags(values)
        threshold = self._config.robust_z_threshold
        detected_at = utcnow()
        findings: list[AnomalyFinding] = []
        analysed = 0
        for i, point in enumerate(series):
            if point.period_start < window.start:
                continue
            analysed += 1
            z_flag = abs(z[i]) >= threshold
            # An isolation forest flag alone must still be a noticeable deviation.
            iso_flag = bool(iso[i]) and abs(z[i]) >= threshold / 2
            if not (z_flag or iso_flag):
                continue
            method = "+".join(
                name for name, hit in (("robust_z", z_flag), ("isolation_forest", iso_flag)) if hit
            )
            findings.append(
                AnomalyFinding(
                    anomaly_id=_anomaly_id(facility_id, metric, point.period_start),
                    facility_id=facility_id,
                    metric=metric,
                    period_start=point.period_start,
                    observed_value=round(float(point.value), 4),
                    expected_value=round(float(expected[i]), 4),
                    score=round(abs(float(z[i])), 4),
                    severity=self.severity(abs(float(z[i]))),
                    method=method,
                    detected_at=detected_at,
                )
            )
        self._repos.anomalies.replace_for_range(
            facility_id, metric, window.start, window.end, findings
        )
        return findings, analysed


def _anomaly_id(facility_id: str, metric: str, period_start: date) -> str:
    digest = hashlib.sha1(
        f"{facility_id}|{metric}|{period_start.isoformat()}".encode(), usedforsecurity=False
    ).hexdigest()[:20]
    return f"an_{digest}"
