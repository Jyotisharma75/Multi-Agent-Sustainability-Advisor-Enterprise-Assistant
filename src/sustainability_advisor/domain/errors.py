"""Domain error hierarchy.

Every failure the service raises deliberately is an ``AdvisorError`` with a
stable ``code``. The API maps codes onto HTTP statuses and never returns stack
traces or internal messages for unexpected errors.
"""

from __future__ import annotations

from typing import Any


class AdvisorError(Exception):
    """Base class for expected, classified failures."""

    code = "advisor_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InputRejectedError(AdvisorError):
    code = "input_rejected"
    http_status = 422


class PromptInjectionError(AdvisorError):
    code = "prompt_injection_detected"
    http_status = 400


class AuthenticationError(AdvisorError):
    code = "unauthenticated"
    http_status = 401


class NotFoundError(AdvisorError):
    code = "not_found"
    http_status = 404


class ToolPermissionError(AdvisorError):
    code = "tool_not_permitted"
    http_status = 403


class ToolExecutionError(AdvisorError):
    code = "tool_failed"
    http_status = 502


class ToolTimeoutError(ToolExecutionError):
    code = "tool_timeout"
    http_status = 504


class SQLSafetyError(AdvisorError):
    code = "sql_rejected"
    http_status = 422


class DataUnavailableError(AdvisorError):
    code = "data_unavailable"
    http_status = 404


class LLMError(AdvisorError):
    code = "llm_error"
    http_status = 502


class LLMUnavailableError(LLMError):
    code = "llm_unavailable"
    http_status = 503


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    http_status = 504


class LLMOutputError(LLMError):
    code = "llm_invalid_output"
    http_status = 502


class RetrievalError(AdvisorError):
    code = "retrieval_error"
    http_status = 502


class AgentExecutionError(AdvisorError):
    code = "agent_failed"
    http_status = 502


class DependencyUnavailableError(AdvisorError):
    code = "dependency_unavailable"
    http_status = 503
