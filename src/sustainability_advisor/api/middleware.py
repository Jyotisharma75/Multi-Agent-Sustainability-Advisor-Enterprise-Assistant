"""HTTP middleware: correlation ids, request size limit and access logging."""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from sustainability_advisor.observability.context import bind_request, clear_context, new_id
from sustainability_advisor.observability.logging import get_logger

logger = get_logger("http")

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_request_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_ID.match(incoming) else new_id("req")
        clear_context()
        bind_request(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()

        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > self._max_bytes:
            response: Response = JSONResponse(
                status_code=413,
                content={
                    "code": "request_too_large",
                    "message": "The request body is too large.",
                    "request_id": request_id,
                    "details": {},
                },
            )
        else:
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        return response
