"""Read only execution of validated SQL.

Runs only statements the guard and validator produced, on the read only
engine. A worker thread performs the blocking call; the configured timeout is
enforced by the driver (``connection.timeout`` for pyodbc, a progress handler
for SQLite) and by the caller.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime
from datetime import time as dtime
from decimal import Decimal
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from sustainability_advisor.domain.errors import DependencyUnavailableError, SQLSafetyError
from sustainability_advisor.sql.models import QueryResult

_SQLITE_PROGRESS_STEPS = 10_000


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date | dtime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


class SQLExecutor:
    def __init__(self, engine: Engine, *, timeout_seconds: float) -> None:
        self._engine = engine
        self._timeout = timeout_seconds

    async def execute(self, sql: str, *, max_rows: int) -> QueryResult:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run, sql, max_rows), timeout=self._timeout + 1.0
            )
        except TimeoutError as exc:
            raise DependencyUnavailableError("The query exceeded its time limit.") from exc

    def _run(self, sql: str, max_rows: int) -> QueryResult:
        started = time.perf_counter()
        deadline = time.monotonic() + self._timeout
        try:
            with self._engine.connect() as conn:
                raw = conn.connection.driver_connection
                if self._engine.dialect.name == "sqlite" and raw is not None:
                    raw.set_progress_handler(
                        lambda: 1 if time.monotonic() > deadline else 0, _SQLITE_PROGRESS_STEPS
                    )
                cursor = conn.exec_driver_sql(sql)
                columns = list(cursor.keys())
                fetched = cursor.fetchmany(max_rows + 1)
                if self._engine.dialect.name == "sqlite" and raw is not None:
                    raw.set_progress_handler(None, 0)
        except SQLAlchemyError as exc:
            message = str(getattr(exc, "orig", exc)).splitlines()[0][:300]
            raise SQLSafetyError(
                "The database rejected the generated query.", details={"error": message}
            ) from exc
        truncated = len(fetched) > max_rows
        rows = [[_json_safe(v) for v in row] for row in fetched[:max_rows]]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
