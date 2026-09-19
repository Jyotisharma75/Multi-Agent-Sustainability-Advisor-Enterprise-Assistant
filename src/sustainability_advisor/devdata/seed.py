"""Development and test data seeding.

Generates a synthetic but internally consistent operating history from a seed
specification file (``data/seed/dev_seed.yaml`` by default): monthly emissions
by scope, electricity and gas consumption, production, sensor readings,
targets and reference values. Scope 2 emissions are derived from non
renewable electricity with the facility's grid factor, so scenario analysis
recovers that factor from the data. Anomalies listed in the specification are
injected so detection can be demonstrated and tested.

Seeding refuses to run against Azure SQL unless ``--allow-azure`` is passed.
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sqlalchemy import delete

from sustainability_advisor.config.models import Settings
from sustainability_advisor.db.engine import Database
from sustainability_advisor.domain.models import (
    EmissionRecord,
    EnergyRecord,
    Facility,
    ProductionRecord,
    Sensor,
    SensorReading,
    SustainabilityMetric,
)
from sustainability_advisor.repositories import Repositories
from sustainability_advisor.services.periods import add_months


class SeedAnomaly(BaseModel):
    month_index: int
    metric: str
    factor: float


class SeedSensor(BaseModel):
    sensor_id: str
    sensor_type: str
    unit: str
    location: str
    base: float
    noise: float
    readings_per_month: int = Field(gt=0)


class SeedReference(BaseModel):
    value: float
    unit: str
    source: str


class SeedFacility(BaseModel):
    facility: Facility
    scope1_tco2e: float
    scope3_tco2e: float
    electricity_mwh: float
    gas_mwh: float
    renewable_share: float
    renewable_share_growth_per_month: float
    grid_factor_tco2e_per_mwh: float
    production: float
    production_unit: str
    product: str
    electricity_price: float | None
    gas_price: float | None
    currency: str | None
    trend_per_month: float
    seasonality: float
    noise: float
    anomalies: list[SeedAnomaly] = Field(default_factory=list)
    targets: dict[str, float] = Field(default_factory=dict)
    target_units: dict[str, str] = Field(default_factory=dict)
    references: dict[str, SeedReference] = Field(default_factory=dict)
    sensors: list[SeedSensor] = Field(default_factory=list)


class SeedSpec(BaseModel):
    start_month: date
    months: int = Field(gt=0)
    random_seed: int
    data_source: str
    methodology: str
    facilities: list[SeedFacility]


def load_spec(path: Path) -> SeedSpec:
    with path.open(encoding="utf-8") as handle:
        return SeedSpec.model_validate(yaml.safe_load(handle))


def _month_end(start: date) -> date:
    return add_months(start, 1) - timedelta(days=1)


def _factor(rng: random.Random, f: SeedFacility, index: int) -> float:
    season = 1 + f.seasonality * math.sin(2 * math.pi * (index % 12) / 12)
    trend = (1 + f.trend_per_month) ** index
    noise = 1 + rng.gauss(0, f.noise)
    return max(season * trend * noise, 0.05)


def clear(db: Database) -> None:
    t = db.tables
    with db.app_engine.begin() as conn:
        for table in (
            t.sensor_readings,
            t.sensors,
            t.sustainability_metrics,
            t.emissions,
            t.energy,
            t.production,
            t.anomalies,
            t.forecast_results,
            t.recommendations,
            t.facilities,
        ):
            conn.execute(delete(table))


def seed(spec: SeedSpec, repos: Repositories) -> dict[str, int]:
    rng = random.Random(spec.random_seed)  # noqa: S311 (reproducible synthetic data)
    counts = {"facilities": 0, "emissions": 0, "energy": 0, "production": 0, "readings": 0}
    last_month = add_months(spec.start_month, spec.months - 1)
    repos.facilities.add_many([f.facility for f in spec.facilities])
    counts["facilities"] = len(spec.facilities)
    for f in spec.facilities:
        fid = f.facility.facility_id
        emissions: list[EmissionRecord] = []
        energy: list[EnergyRecord] = []
        production: list[ProductionRecord] = []
        spikes = {(a.month_index, a.metric): a.factor for a in f.anomalies}
        for i in range(spec.months):
            start = add_months(spec.start_month, i)
            end = _month_end(start)
            factor = _factor(rng, f, i)
            output = f.production * factor * (1 + rng.gauss(0, f.noise / 2))
            electricity = f.electricity_mwh * factor * spikes.get((i, "energy"), 1.0)
            gas = f.gas_mwh * factor * spikes.get((i, "energy"), 1.0)
            share = min(f.renewable_share + f.renewable_share_growth_per_month * i, 1.0)
            renewable = electricity * share
            scope1 = f.scope1_tco2e * factor * spikes.get((i, "emissions"), 1.0)
            scope2 = (electricity - renewable) * f.grid_factor_tco2e_per_mwh
            scope3 = f.scope3_tco2e * factor
            for scope, value, category in (
                (1, scope1, "stationary_combustion"),
                (2, scope2, "purchased_electricity"),
                (3, scope3, "upstream_logistics"),
            ):
                emissions.append(
                    EmissionRecord(
                        facility_id=fid,
                        period_start=start,
                        period_end=end,
                        scope=scope,
                        category=category,
                        co2e_tonnes=round(value, 3),
                        source=spec.data_source,
                        methodology=spec.methodology,
                    )
                )
            for energy_type, mwh, renewable_mwh, price in (
                ("electricity", electricity, renewable, f.electricity_price),
                ("natural_gas", gas, 0.0, f.gas_price),
            ):
                energy.append(
                    EnergyRecord(
                        facility_id=fid,
                        period_start=start,
                        period_end=end,
                        energy_type=energy_type,
                        consumption_mwh=round(mwh, 3),
                        renewable_mwh=round(renewable_mwh, 3),
                        cost=round(mwh * price, 2) if price is not None else None,
                        currency=f.currency if price is not None else None,
                    )
                )
            production.append(
                ProductionRecord(
                    facility_id=fid,
                    period_start=start,
                    period_end=end,
                    product=f.product,
                    output_quantity=round(output * spikes.get((i, "production"), 1.0), 3),
                    output_unit=f.production_unit,
                )
            )
        repos.emissions.add_many(emissions)
        repos.energy.add_many(energy)
        repos.production.add_many(production)
        counts["emissions"] += len(emissions)
        counts["energy"] += len(energy)
        counts["production"] += len(production)

        metrics = [
            SustainabilityMetric(
                facility_id=fid,
                metric_name=name,
                period_start=spec.start_month,
                period_end=_month_end(last_month),
                value=None,
                unit=f.target_units.get(name),
                target_value=value,
                source="approved_targets",
            )
            for name, value in f.targets.items()
        ] + [
            SustainabilityMetric(
                facility_id=fid,
                metric_name=name,
                period_start=spec.start_month,
                period_end=_month_end(last_month),
                value=ref.value,
                unit=ref.unit,
                target_value=None,
                source=ref.source,
            )
            for name, ref in f.references.items()
        ]
        repos.metrics.add_many(metrics)

        repos.sensors.add_sensors(
            [
                Sensor(
                    sensor_id=s.sensor_id,
                    facility_id=fid,
                    sensor_type=s.sensor_type,
                    unit=s.unit,
                    location=s.location,
                )
                for s in f.sensors
            ]
        )
        readings: list[SensorReading] = []
        for s in f.sensors:
            for i in range(spec.months):
                month = add_months(spec.start_month, i)
                days = (_month_end(month) - month).days + 1
                for k in range(s.readings_per_month):
                    at = datetime.combine(month, datetime.min.time()) + timedelta(
                        days=k * days / s.readings_per_month
                    )
                    value = s.base * _factor(rng, f, i) * (1 + rng.gauss(0, s.noise))
                    value *= spikes.get((i, f"sensor:{s.sensor_id}"), 1.0)
                    readings.append(
                        SensorReading(sensor_id=s.sensor_id, recorded_at=at, value=round(value, 3))
                    )
        repos.sensors.add_readings(readings)
        counts["readings"] += len(readings)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed development data.")
    parser.add_argument("--spec", type=Path, default=Path("data/seed/dev_seed.yaml"))
    parser.add_argument("--allow-azure", action="store_true")
    args = parser.parse_args()

    from sustainability_advisor.config.loader import get_settings
    from sustainability_advisor.db.engine import build_database
    from sustainability_advisor.repositories import build_repositories

    settings: Settings = get_settings()
    if settings.database.backend == "azure_sql" and not args.allow_azure:
        raise SystemExit("Refusing to seed Azure SQL without --allow-azure.")
    database = build_database(settings.database)
    try:
        clear(database)
        counts: dict[str, Any] = seed(load_spec(args.spec), build_repositories(database))
        print(yaml.safe_dump(counts, sort_keys=False))
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
