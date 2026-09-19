"""Conversation repository."""

from __future__ import annotations

from sqlalchemy import select, update

from sustainability_advisor.domain.models import Conversation, ConversationMessage
from sustainability_advisor.repositories.base import Repository, from_json, to_json, utcnow


class ConversationRepository(Repository):
    def ensure(self, conversation_id: str) -> None:
        t = self._tables.conversations
        if self._fetch(select(t.c.conversation_id).where(t.c.conversation_id == conversation_id)):
            return
        now = utcnow()
        self._write(
            t.insert().values(conversation_id=conversation_id, created_at=now, updated_at=now)
        )

    def add_message(self, message: ConversationMessage) -> None:
        m = self._tables.conversation_messages
        c = self._tables.conversations
        with self._engine.begin() as conn:
            conn.execute(
                m.insert().values(
                    message_id=message.message_id,
                    conversation_id=message.conversation_id,
                    role=message.role,
                    content=message.content,
                    payload=to_json(message.payload),
                    created_at=message.created_at,
                )
            )
            conn.execute(
                update(c)
                .where(c.c.conversation_id == message.conversation_id)
                .values(updated_at=message.created_at)
            )

    def get(self, conversation_id: str, *, max_messages: int | None = None) -> Conversation | None:
        c = self._tables.conversations
        m = self._tables.conversation_messages
        header = self._fetch(select(c).where(c.c.conversation_id == conversation_id))
        if not header:
            return None
        stmt = (
            select(m).where(m.c.conversation_id == conversation_id).order_by(m.c.created_at.desc())
        )
        if max_messages is not None:
            stmt = stmt.limit(max_messages)
        messages = []
        for row in reversed(self._fetch(stmt)):
            data = dict(row._mapping)
            data["payload"] = from_json(data["payload"])
            messages.append(ConversationMessage.model_validate(data))
        head = dict(header[0]._mapping)
        return Conversation(
            conversation_id=head["conversation_id"],
            created_at=head["created_at"],
            updated_at=head["updated_at"],
            messages=messages,
        )
