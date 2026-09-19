"""Facility repository."""

from __future__ import annotations

from sqlalchemy import select

from sustainability_advisor.domain.models import Facility
from sustainability_advisor.repositories.base import Repository


class FacilityRepository(Repository):
    def list_all(self, *, active_only: bool = True) -> list[Facility]:
        t = self._tables.facilities
        stmt = select(t).order_by(t.c.facility_id)
        if active_only:
            stmt = stmt.where(t.c.is_active.is_(True))
        return [Facility.model_validate(dict(r._mapping)) for r in self._fetch(stmt)]

    def get(self, facility_id: str) -> Facility | None:
        t = self._tables.facilities
        rows = self._fetch(select(t).where(t.c.facility_id == facility_id))
        return Facility.model_validate(dict(rows[0]._mapping)) if rows else None

    def existing_ids(self, facility_ids: list[str]) -> set[str]:
        if not facility_ids:
            return set()
        t = self._tables.facilities
        rows = self._fetch(select(t.c.facility_id).where(t.c.facility_id.in_(facility_ids)))
        return {str(r[0]) for r in rows}

    def add_many(self, facilities: list[Facility]) -> None:
        self._write(self._tables.facilities.insert(), [f.model_dump() for f in facilities])
