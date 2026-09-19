"""Audit trail and conversation persistence."""

from __future__ import annotations

import asyncio
from typing import Any

from sustainability_advisor.config.models import ConversationConfig
from sustainability_advisor.domain.models import AuditEvent, Conversation, ConversationMessage
from sustainability_advisor.observability.context import (
    get_conversation_id,
    get_request_id,
    new_id,
)
from sustainability_advisor.observability.logging import get_logger
from sustainability_advisor.repositories import Repositories
from sustainability_advisor.repositories.base import utcnow

logger = get_logger(__name__)


class AuditService:
    def __init__(self, repos: Repositories) -> None:
        self._repos = repos

    async def record(self, event_type: str, actor: str, payload: dict[str, Any]) -> None:
        event = AuditEvent(
            event_id=new_id("evt"),
            conversation_id=get_conversation_id(),
            request_id=get_request_id(),
            event_type=event_type,
            actor=actor,
            payload=payload,
            created_at=utcnow(),
        )
        try:
            await asyncio.to_thread(self._repos.audit.append, event)
        except Exception as exc:
            # Losing an audit event must be visible but must not fail the request.
            logger.error("audit.write_failed", event_type=event_type, error=type(exc).__name__)


class ConversationService:
    def __init__(self, config: ConversationConfig, repos: Repositories) -> None:
        self._config = config
        self._repos = repos

    async def ensure(self, conversation_id: str) -> None:
        await asyncio.to_thread(self._repos.conversations.ensure, conversation_id)

    async def append(
        self, conversation_id: str, role: str, content: str, payload: dict[str, Any]
    ) -> None:
        message = ConversationMessage(
            message_id=new_id("msg"),
            conversation_id=conversation_id,
            role=role,
            content=content[: self._config.max_message_chars],
            payload=payload,
            created_at=utcnow(),
        )
        await asyncio.to_thread(self._repos.conversations.add_message, message)

    async def history(self, conversation_id: str) -> list[ConversationMessage]:
        if self._config.max_history_turns == 0:
            return []
        conversation = await asyncio.to_thread(
            self._repos.conversations.get,
            conversation_id,
            max_messages=self._config.max_history_turns * 2,
        )
        return conversation.messages if conversation else []

    async def get(self, conversation_id: str) -> Conversation | None:
        return await asyncio.to_thread(self._repos.conversations.get, conversation_id)
