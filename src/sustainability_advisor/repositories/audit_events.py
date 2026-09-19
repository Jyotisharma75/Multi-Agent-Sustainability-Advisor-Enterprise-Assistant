"""Audit event repository (append only)."""

from __future__ import annotations

from sqlalchemy import select

from sustainability_advisor.domain.models import AuditEvent
from sustainability_advisor.repositories.base import Repository, from_json, to_json


class AuditEventRepository(Repository):
    def append(self, event: AuditEvent) -> None:
        self._write(
            self._tables.audit_events.insert().values(
                event_id=event.event_id,
                conversation_id=event.conversation_id,
                request_id=event.request_id,
                event_type=event.event_type,
                actor=event.actor,
                payload=to_json(event.payload),
                created_at=event.created_at,
            )
        )

    def list_for_conversation(self, conversation_id: str) -> list[AuditEvent]:
        t = self._tables.audit_events
        stmt = select(t).where(t.c.conversation_id == conversation_id).order_by(t.c.created_at)
        events = []
        for row in self._fetch(stmt):
            data = dict(row._mapping)
            data["payload"] = from_json(data["payload"])
            events.append(AuditEvent.model_validate(data))
        return events
