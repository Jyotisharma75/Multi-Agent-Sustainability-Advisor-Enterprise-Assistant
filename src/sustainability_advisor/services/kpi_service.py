"""Sustainability KPI computation.

KPI definitions (kind, numerator, denominator, unit, scale, direction and
target metric) are configuration. Data sources are addressed with a small,
closed vocabulary resolved here against repositories:

* ``emissions.total`` or ``emissions.scope<N>``
* ``energy.consumption_mwh``, ``energy.renewable_mwh``, ``energy.cost``
* ``production.output_quantity``
* ``metric:<metric_name>`` for values held in ``sustainability_metrics``

A KPI whose inputs are missing is reported with ``value = null`` and status
``no_data``; nothing is estimated.
"""

from __future__ import annotations

from sustainability_advisor.config.models import KpiConfig, KpiDefinition
from sustainability_advisor.domain.analysis import KpiValue
from sustainability_advisor.domain.errors import InputRejectedError
from sustainability_advisor.repositories import Repositories
from sustainability_advisor.services.periods import Period


class KpiService:
    def __init__(self, config: KpiConfig, repos: Repositories) -> None:
        self._config = config
        self._repos = repos

    @property
    def available(self) -> list[str]:
        return list(self._config.definitions)

    def _source(self, source: str, facility_id: str, period: Period) -> float | None:
        repos = self._repos
        if source.startswith("emissions."):
            selector = source.split(".", 1)[1]
            totals = repos.emissions.totals_by_scope(facility_id, period.start, period.end)
            if not totals:
                return None
            if selector == "total":
                return sum(totals.values())
            if selector.startswith("scope") and selector[5:].isdigit():
                return totals.get(int(selector[5:]))
        elif source.startswith("energy."):
            totals_e = repos.energy.totals(facility_id, period.start, period.end)
            if totals_e is None:
                return None
            field = source.split(".", 1)[1]
            if field == "consumption_mwh":
                return totals_e.consumption_mwh
            if field == "renewable_mwh":
                return totals_e.renewable_mwh
            if field == "cost":
                return totals_e.cost
        elif source == "production.output_quantity":
            return repos.production.total_output(facility_id, period.start, period.end)
        elif source.startswith("metric:"):
            values = repos.metrics.values(facility_id, source[7:], period.start, period.end)
            measured = [v.value for v in values if v.value is not None]
            return sum(measured) if measured else None
        raise InputRejectedError(f"Unknown KPI data source {source!r}")

    def _target(self, definition: KpiDefinition, facility_id: str, period: Period) -> float | None:
        if not definition.target_metric:
            return None
        record = self._repos.metrics.latest(facility_id, definition.target_metric, period.end)
        if record is None or record.target_value is None:
            return None
        if definition.target_basis == "per_month":
            return record.target_value * period.months
        return record.target_value

    def compute_one(self, facility_id: str, kpi: str, period: Period) -> KpiValue:
        definition = self._config.definitions.get(kpi)
        if definition is None:
            raise InputRejectedError(f"Unknown KPI {kpi!r}", details={"available": self.available})
        numerator = self._source(definition.numerator, facility_id, period)
        sources = [definition.numerator]
        value: float | None
        note: str | None = None
        if definition.kind == "ratio":
            assert definition.denominator is not None
            sources.append(definition.denominator)
            denominator = self._source(definition.denominator, facility_id, period)
            if numerator is None or denominator is None:
                value, note = None, "An input to this ratio is not reported for the period."
            elif denominator == 0:
                value, note = None, "The denominator is zero for the period."
            else:
                value = numerator / denominator * definition.scale
        else:
            value = numerator * definition.scale if numerator is not None else None
            if value is None:
                note = "No data is reported for the period."

        target = self._target(definition, facility_id, period)
        gap: float | None = None
        if value is None:
            status = "no_data"
        elif target is None:
            status = "no_target"
        elif target == 0:
            gap = value if not definition.higher_is_better else -value
            status = "on_track" if gap <= 0 else "off_track"
        else:
            raw = (value - target) / abs(target)
            gap = -raw if definition.higher_is_better else raw
            status = "on_track" if gap <= self._config.on_track_tolerance else "off_track"
        return KpiValue(
            facility_id=facility_id,
            kpi=kpi,
            description=definition.description,
            value=round(value, 6) if value is not None else None,
            unit=definition.unit,
            target=target,
            gap_ratio=round(gap, 6) if gap is not None else None,
            status=status,
            period_start=period.start,
            period_end=period.end,
            sources=sources,
            note=note,
        )

    def compute(self, facility_id: str, kpis: list[str], period: Period) -> list[KpiValue]:
        return [self.compute_one(facility_id, kpi, period) for kpi in kpis]
