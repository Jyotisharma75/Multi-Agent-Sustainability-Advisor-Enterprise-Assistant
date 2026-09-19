"""ASGI application factory.

Run with ``uvicorn sustainability_advisor.main:create_app --factory``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sustainability_advisor.api.errors import install_error_handlers
from sustainability_advisor.api.middleware import RequestContextMiddleware
from sustainability_advisor.api.routes import assistant, health
from sustainability_advisor.config.loader import get_settings
from sustainability_advisor.container import Container, build_container
from sustainability_advisor.observability.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(container: Container | None = None) -> FastAPI:
    app_container = container or build_container(get_settings())
    settings = app_container.settings
    configure_logging(settings.logging)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await app_container.startup()
        logger.info("app.started", environment=settings.app.environment)
        try:
            yield
        finally:
            app_container.close()
            logger.info("app.stopped")

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        lifespan=lifespan,
        docs_url=f"{settings.api.prefix}/docs",
        openapi_url=f"{settings.api.prefix}/openapi.json",
    )
    app.state.container = app_container
    app.add_middleware(RequestContextMiddleware, max_request_bytes=settings.api.max_request_bytes)
    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    install_error_handlers(app)
    app.include_router(health.router, prefix=settings.api.prefix)
    app.include_router(assistant.router, prefix=settings.api.prefix)
    return app
