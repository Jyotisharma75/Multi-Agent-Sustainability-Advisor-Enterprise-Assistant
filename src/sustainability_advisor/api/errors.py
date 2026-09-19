"""Exception to HTTP response mapping. Internal details never leak."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sustainability_advisor.domain.errors import AdvisorError
from sustainability_advisor.observability.logging import get_logger

logger = get_logger("http.errors")


def _body(request: Request, code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "request_id": getattr(request.state, "request_id", None),
        "details": jsonable_encoder(details),
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AdvisorError)
    async def _advisor(request: Request, exc: AdvisorError) -> JSONResponse:
        logger.warning("http.advisor_error", code=exc.code, status=exc.http_status)
        return JSONResponse(
            status_code=exc.http_status, content=_body(request, exc.code, exc.message, exc.details)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()[:10]
        ]
        return JSONResponse(
            status_code=422,
            content=_body(
                request, "invalid_request", "The request is invalid.", {"errors": errors}
            ),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("http.unexpected_error", error=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=_body(request, "internal_error", "An unexpected error occurred.", {}),
        )
