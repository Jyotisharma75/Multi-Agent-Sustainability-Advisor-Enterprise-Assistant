"""Repository base class.

Repositories are the only components that build SQL against the application
engine. They use SQLAlchemy Core expressions exclusively, so every value is a
bound parameter and no user text is ever concatenated into a statement.
Methods are synchronous; async callers run them through ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Engine, Row, Select
from sqlalchemy.exc import SQLAlchemyError

from sustainability_advisor.db.tables import Tables
from sustainability_advisor.domain.errors import DependencyUnavailableError


def utcnow() -> datetime:
    """Return a naive UTC timestamp, the storage convention of this schema."""
    return datetime.now(UTC).replace(tzinfo=None)


def to_json(payload: Any) -> str:
    return json.dumps(payload, default=_json_default, sort_keys=True)


def from_json(raw: str | None) -> Any:
    return json.loads(raw) if raw else {}


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Unserialisable value of type {type(value).__name__}")


class Repository:
    def __init__(self, engine: Engine, tables: Tables) -> None:
        self._engine = engine
        self._tables = tables

    def _fetch(self, statement: Select[Any]) -> Sequence[Row[Any]]:
        try:
            with self._engine.connect() as conn:
                return conn.execute(statement).fetchall()
        except SQLAlchemyError as exc:
            raise DependencyUnavailableError(
                "The sustainability database could not be queried.",
                details={"error": type(exc).__name__},
            ) from exc

    def _write(self, statement: Any, rows: list[dict[str, Any]] | None = None) -> None:
        try:
            with self._engine.begin() as conn:
                if rows is None:
                    conn.execute(statement)
                elif rows:
                    conn.execute(statement, rows)
        except SQLAlchemyError as exc:
            raise DependencyUnavailableError(
                "The sustainability database rejected a write.",
                details={"error": type(exc).__name__},
            ) from exc
