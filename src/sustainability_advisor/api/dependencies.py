"""Request dependencies: container access, request id and API key auth."""

from __future__ import annotations

import hmac

from fastapi import Request

from sustainability_advisor.config.loader import read_secret
from sustainability_advisor.container import Container
from sustainability_advisor.domain.errors import AuthenticationError


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_request_id(request: Request) -> str:
    return str(request.state.request_id)


def require_api_key(request: Request) -> None:
    """Constant time comparison against keys held in the configured variable.

    The variable holds a comma separated list so keys can be rotated without
    downtime. Deployments behind Azure API Management or Entra ID auth at the
    ingress may disable this check in configuration.
    """
    container = get_container(request)
    auth = container.settings.api.auth
    if not auth.enabled:
        return
    raw = read_secret(auth.api_keys_env) or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    presented = request.headers.get(auth.header_name, "")
    if not keys or not presented:
        raise AuthenticationError("A valid API key is required.")
    if not any(hmac.compare_digest(presented.encode(), key.encode()) for key in keys):
        raise AuthenticationError("A valid API key is required.")
