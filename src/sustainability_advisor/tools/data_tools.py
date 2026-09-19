"""Structured data tools: facility directory, SQL query and KPI lookup."""

from __future__ import annotations

import asyncio
from datetime import date

from pydantic import BaseModel, Field, model_validator

from sustainability_advisor.domain.analysis import KpiValue
from sustainability_advisor.domain.errors import InputRejectedError
from sustainability_advisor.domain.models import Facility
from sustainability_advisor.repositories import Repositories
from sustainability_advisor.services.kpi_service import KpiService
from sustainability_advisor.services.periods import Period
from sustainability_advisor.sql.models import SQLAnswer
from sustainability_advisor.sql.service import NLToSQLService
from sustainability_advisor.tools.base import Tool, ToolContext


class FacilityDirectoryInput(BaseModel):
    include_inactive: bool = False


class FacilityEntry(BaseModel):
    facility: Facility
    latest_period: date | None


class FacilityDirectoryOutput(BaseModel):
    facilities: list[FacilityEntry]


class FacilityDirectoryTool(Tool[FacilityDirectoryInput, FacilityDirectoryOutput]):
    name = "facility_directory"
    input_model = FacilityDirectoryInput
    output_model = FacilityDirectoryOutput

    def __init__(self, repos: Repositories) -> None:
        self._repos = repos

    def _load(self, include_inactive: bool) -> FacilityDirectoryOutput:
        facilities = self._repos.facilities.list_all(active_only=not include_inactive)
        return FacilityDirectoryOutput(
            facilities=[
                FacilityEntry(
                    facility=f, latest_period=self._repos.emissions.latest_period(f.facility_id)
                )
                for f in facilities
            ]
        )

    async def run(
        self, payload: FacilityDirectoryInput, ctx: ToolContext
    ) -> FacilityDirectoryOutput:
        return await asyncio.to_thread(self._load, payload.include_inactive)


class SQLQueryInput(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    intent: str
    context: str = Field(default="", max_length=2000)
    max_rows: int | None = Field(default=None, gt=0)


class SQLQueryTool(Tool[SQLQueryInput, SQLAnswer]):
    """Answers a data question with guarded, verified, read only SQL on Azure SQL."""

    name = "azure_sql_query"
    input_model = SQLQueryInput
    output_model = SQLAnswer

    def __init__(self, service: NLToSQLService) -> None:
        self._service = service

    async def run(self, payload: SQLQueryInput, ctx: ToolContext) -> SQLAnswer:
        return await self._service.answer(
            payload.question,
            intent=payload.intent,
            context=payload.context,
            max_rows=payload.max_rows,
        )


class KpiLookupInput(BaseModel):
    facility_ids: list[str] = Field(min_length=1)
    kpis: list[str] | None = None
    period_start: date
    period_end: date

    @model_validator(mode="after")
    def _ordered(self) -> KpiLookupInput:
        if self.period_end < self.period_start:
            raise ValueError("period_end precedes period_start")
        return self


class KpiLookupOutput(BaseModel):
    values: list[KpiValue]


class KpiLookupTool(Tool[KpiLookupInput, KpiLookupOutput]):
    name = "sustainability_kpi_lookup"
    input_model = KpiLookupInput
    output_model = KpiLookupOutput

    def __init__(self, service: KpiService, repos: Repositories, default_kpis: list[str]) -> None:
        self._service = service
        self._repos = repos
        self._default_kpis = default_kpis

    def _compute(self, payload: KpiLookupInput) -> KpiLookupOutput:
        known = self._repos.facilities.existing_ids(payload.facility_ids)
        unknown = sorted(set(payload.facility_ids) - known)
        if unknown:
            raise InputRejectedError("Unknown facilities.", details={"facility_ids": unknown})
        period = Period(start=payload.period_start, end=payload.period_end)
        kpis = payload.kpis or self._default_kpis
        values = [
            value
            for facility_id in payload.facility_ids
            for value in self._service.compute(facility_id, kpis, period)
        ]
        return KpiLookupOutput(values=values)

    async def run(self, payload: KpiLookupInput, ctx: ToolContext) -> KpiLookupOutput:
        return await asyncio.to_thread(self._compute, payload)
