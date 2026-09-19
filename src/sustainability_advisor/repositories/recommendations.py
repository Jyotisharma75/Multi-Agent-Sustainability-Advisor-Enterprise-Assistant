"""Recommendation repository."""

from __future__ import annotations

from sqlalchemy import select

from sustainability_advisor.domain.models import Recommendation
from sustainability_advisor.repositories.base import Repository, from_json, to_json, utcnow


class RecommendationRepository(Repository):
    def save_many(self, conversation_id: str | None, items: list[Recommendation]) -> None:
        now = utcnow()
        rows = [
            {
                "recommendation_id": r.recommendation_id,
                "facility_id": r.facility_id,
                "conversation_id": conversation_id,
                "issue": r.issue,
                "severity": r.severity,
                "priority": r.priority,
                "confidence": r.confidence,
                "payload": to_json(r.model_dump(mode="json")),
                "created_at": now,
            }
            for r in items
        ]
        self._write(self._tables.recommendations.insert(), rows)

    def list_for_conversation(self, conversation_id: str) -> list[Recommendation]:
        t = self._tables.recommendations
        stmt = (
            select(t.c.payload)
            .where(t.c.conversation_id == conversation_id)
            .order_by(t.c.created_at)
        )
        return [Recommendation.model_validate(from_json(r[0])) for r in self._fetch(stmt)]
