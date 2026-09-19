"""Liveness and readiness probes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from sustainability_advisor.api.dependencies import get_container
from sustainability_advisor.api.schemas import HealthResponse, ReadinessResponse
from sustainability_advisor.container import Container

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(container: Container = Depends(get_container)) -> HealthResponse:
    """Liveness: the process is up. Performs no dependency calls."""
    app = container.settings.app
    return HealthResponse(
        status="ok", service=app.name, version=app.version, environment=app.environment
    )


@router.get("/ready", response_model=ReadinessResponse)
async def ready(container: Container = Depends(get_container)) -> JSONResponse:
    """Readiness: the database answers, the search index answers and at least
    one language model provider is available."""
    checks: dict[str, str] = {}
    try:
        await asyncio.to_thread(container.database.ping)
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
    try:
        count = await container.search_index.count()
        checks["search_index"] = f"ok ({count} chunks)"
    except Exception as exc:
        checks["search_index"] = f"error: {type(exc).__name__}"
    providers = {name: await p.probe() for name, p in container.providers.all().items()}
    for name, status in providers.items():
        checks[f"llm.{name}"] = status
    llm_ready = any(status != "unavailable" for status in providers.values())
    ready_state = (
        checks["database"] == "ok" and checks["search_index"].startswith("ok") and llm_ready
    )
    body = ReadinessResponse(status="ready" if ready_state else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if ready_state else 503, content=body.model_dump())
