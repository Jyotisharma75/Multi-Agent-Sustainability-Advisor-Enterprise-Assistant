"""What-if scenario analysis.

A scenario starts from the facility's measured baseline (emissions by scope,
energy consumption, renewable supply) over the configured number of months
and applies levers defined in configuration. Every factor used is derived
from the facility's own data; when a factor cannot be derived the lever is
not applied and a limitation is reported instead.

Lever kinds, applied in this order:

1. ``production_change``: output changes by the percentage; emissions in the
   lever's scopes scale proportionally (intensity held constant)
2. ``energy_reduction``: energy use falls by the percentage; emissions in the
   lever's scopes fall proportionally
3. ``renewable_share``: renewable share of electricity rises to the target
   percentage; scope 2 emissions follow the facility's derived emission
   factor per non renewable MWh
"""

from __future__ import annotations

from datetime import date

from sustainability_advisor.config.models import ScenarioConfig, ScenarioLever
from sustainability_advisor.domain.analysis import ScenarioLeverEffect, ScenarioOutcome
from sustainability_advisor.domain.errors import DataUnavailableError, InputRejectedError
from sustainability_advisor.repositories import Repositories
from sustainability_advisor.services.periods import Period, trailing_months

_ORDER = {"production_change": 0, "energy_reduction": 1, "renewable_share": 2}


class ScenarioService:
    def __init__(self, config: ScenarioConfig, repos: Repositories) -> None:
        self._config = config
        self._repos = repos

    @property
    def levers(self) -> dict[str, ScenarioLever]:
        return dict(self._config.levers)

    def validate_levers(self, levers: dict[str, float]) -> None:
        if not levers:
            raise InputRejectedError("At least one scenario lever is required.")
        for name, value in levers.items():
            spec = self._config.levers.get(name)
            if spec is None:
                raise InputRejectedError(
                    f"Unknown scenario lever {name!r}",
                    details={"available": sorted(self._config.levers)},
                )
            if not spec.min_value <= value <= spec.max_value:
                raise InputRejectedError(
                    f"{name} must be between {spec.min_value} and {spec.max_value} {spec.unit}."
                )

    def baseline_period(self, facility_id: str, as_of: date | None = None) -> Period:
        latest = as_of or self._repos.emissions.latest_period(facility_id)
        if latest is None:
            raise DataUnavailableError(f"No emissions data for facility {facility_id}.")
        return trailing_months(latest, self._config.baseline_months)

    def run(
        self, facility_id: str, levers: dict[str, float], period: Period | None = None
    ) -> ScenarioOutcome:
        self.validate_levers(levers)
        window = period or self.baseline_period(facility_id)
        baseline = self._repos.emissions.totals_by_scope(facility_id, window.start, window.end)
        if not baseline:
            raise DataUnavailableError(f"No emissions baseline for facility {facility_id}.")
        energy = self._repos.energy.totals(facility_id, window.start, window.end)
        consumption = energy.consumption_mwh if energy else None
        renewable = energy.renewable_mwh if energy else None
        share = renewable / consumption if consumption and renewable is not None else None
        non_renewable = consumption - renewable if consumption and renewable is not None else None
        grid_factor = (
            baseline.get(2, 0.0) / non_renewable if non_renewable and non_renewable > 0 else None
        )

        projected = dict(baseline)
        current_consumption = consumption
        effects: list[ScenarioLeverEffect] = []
        limitations: list[str] = []
        ordered = sorted(levers.items(), key=lambda kv: _ORDER[self._config.levers[kv[0]].kind])
        for name, value in ordered:
            spec = self._config.levers[name]
            before = sum(projected.values())
            if spec.kind in ("production_change", "energy_reduction"):
                factor = (
                    1 + value / 100.0 if spec.kind == "production_change" else 1 - value / 100.0
                )
                for scope in spec.scopes:
                    if scope in projected:
                        projected[scope] *= factor
                if current_consumption is not None:
                    current_consumption *= factor
                scopes = ", ".join(str(s) for s in spec.scopes)
                assumption = (
                    f"Emissions in scopes {scopes} scale in proportion to "
                    f"{'output' if spec.kind == 'production_change' else 'energy use'}."
                )
            else:
                if grid_factor is None or current_consumption is None or share is None:
                    limitations.append(
                        f"{name} was not applied: the facility's electricity emission factor "
                        "cannot be derived from its energy and scope 2 data."
                    )
                    continue
                target_share = value / 100.0
                if target_share <= share:
                    limitations.append(
                        f"{name} was not applied: the current renewable share "
                        f"({share * 100:.1f}%) already meets {value}%."
                    )
                    continue
                projected[2] = grid_factor * current_consumption * (1 - target_share)
                assumption = (
                    f"Scope 2 follows the facility's derived factor of {grid_factor:.4f} "
                    "tCO2e per non renewable MWh."
                )
            effects.append(
                ScenarioLeverEffect(
                    lever=name,
                    value=value,
                    unit=spec.unit,
                    delta_co2e_tonnes=round(sum(projected.values()) - before, 4),
                    assumption=assumption,
                )
            )

        base_total = sum(baseline.values())
        projected_total = sum(projected.values())
        return ScenarioOutcome(
            facility_id=facility_id,
            baseline_start=window.start,
            baseline_end=window.end,
            baseline_co2e_tonnes=round(base_total, 4),
            projected_co2e_tonnes=round(projected_total, 4),
            delta_co2e_tonnes=round(projected_total - base_total, 4),
            delta_ratio=round((projected_total - base_total) / base_total, 6)
            if base_total
            else 0.0,
            baseline_by_scope={k: round(v, 4) for k, v in baseline.items()},
            projected_by_scope={k: round(v, 4) for k, v in projected.items()},
            effects=effects,
            derived_factors={
                "renewable_share": round(share, 6) if share is not None else None,
                "scope2_tco2e_per_non_renewable_mwh": (
                    round(grid_factor, 6) if grid_factor is not None else None
                ),
            },
            limitations=limitations,
        )
