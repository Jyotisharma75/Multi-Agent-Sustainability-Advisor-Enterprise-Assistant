"""Request scoped correlation identifiers carried in context variables."""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token

import structlog

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)


def new_id(prefix: str) -> str:
    """Return a random identifier with a readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"


def get_request_id() -> str | None:
    return _request_id.get()


def get_conversation_id() -> str | None:
    return _conversation_id.get()


def bind_request(request_id: str) -> Token[str | None]:
    structlog.contextvars.bind_contextvars(request_id=request_id)
    return _request_id.set(request_id)


def bind_conversation(conversation_id: str) -> Token[str | None]:
    structlog.contextvars.bind_contextvars(conversation_id=conversation_id)
    return _conversation_id.set(conversation_id)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
    _request_id.set(None)
    _conversation_id.set(None)
