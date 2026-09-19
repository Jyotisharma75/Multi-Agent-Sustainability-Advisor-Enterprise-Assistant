"""Sensor and sensor reading repository."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from sustainability_advisor.domain.models import Sensor, SensorReading
from sustainability_advisor.repositories.base import Repository


class SensorRepository(Repository):
    def list_for_facility(self, facility_id: str) -> list[Sensor]:
        t = self._tables.sensors
        stmt = select(t).where(t.c.facility_id == facility_id).order_by(t.c.sensor_id)
        return [Sensor.model_validate(dict(r._mapping)) for r in self._fetch(stmt)]

    def get(self, sensor_id: str) -> Sensor | None:
        t = self._tables.sensors
        rows = self._fetch(select(t).where(t.c.sensor_id == sensor_id))
        return Sensor.model_validate(dict(rows[0]._mapping)) if rows else None

    def readings(self, sensor_id: str, start: datetime, end: datetime) -> list[SensorReading]:
        t = self._tables.sensor_readings
        stmt = (
            select(t.c.sensor_id, t.c.recorded_at, t.c.value)
            .where(t.c.sensor_id == sensor_id)
            .where(t.c.recorded_at >= start)
            .where(t.c.recorded_at <= end)
            .order_by(t.c.recorded_at)
        )
        return [SensorReading.model_validate(dict(r._mapping)) for r in self._fetch(stmt)]

    def add_sensors(self, sensors: list[Sensor]) -> None:
        self._write(self._tables.sensors.insert(), [s.model_dump() for s in sensors])

    def add_readings(self, readings: list[SensorReading]) -> None:
        self._write(self._tables.sensor_readings.insert(), [r.model_dump() for r in readings])
