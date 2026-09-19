"""Azure OpenAI chat provider.

Authenticates with Microsoft Entra ID through ``DefaultAzureCredential`` by
default, or with an API key read from the environment variable named in
configuration. The deployment name, API version and limits all come from
configuration.
"""

from __future__ import annotations

from typing import Any

import openai
from openai import AsyncAzureOpenAI

from sustainability_advisor.config.loader import read_secret
from sustainability_advisor.config.models import AzureOpenAIConfig
from sustainability_advisor.domain.errors import (
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from sustainability_advisor.domain.models import TokenUsage
from sustainability_advisor.llm.base import ChatMessage, LLMProvider


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, config: AzureOpenAIConfig) -> None:
        super().__init__(config.retry)
        self._config = config
        self._client: AsyncAzureOpenAI | None = None

    @property
    def name(self) -> str:
        return "azure_openai"

    @property
    def model_name(self) -> str:
        return self._config.chat_deployment

    def is_available(self) -> bool:
        if not self._config.enabled or not read_secret(self._config.endpoint_env):
            return False
        return self._config.auth_mode == "entra" or bool(read_secret(self._config.api_key_env))

    def _get_client(self) -> AsyncAzureOpenAI:
        if self._client is not None:
            return self._client
        endpoint = read_secret(self._config.endpoint_env)
        if not endpoint:
            raise LLMUnavailableError(f"{self._config.endpoint_env} is not set.")
        kwargs: dict[str, Any] = {
            "azure_endpoint": endpoint,
            "api_version": self._config.api_version,
            "timeout": self._config.timeout_seconds,
            "max_retries": 0,
        }
        if self._config.auth_mode == "entra":
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), self._config.token_scope
            )
        else:
            kwargs["api_key"] = read_secret(self._config.api_key_env)
        self._client = AsyncAzureOpenAI(**kwargs)
        return self._client

    async def _complete(
        self,
        messages: list[ChatMessage],
        *,
        task: str,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, TokenUsage]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self._config.chat_deployment,
            "messages": [m.model_dump() for m in messages],
            "temperature": self._config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("Azure OpenAI timed out.") from exc
        except openai.AuthenticationError as exc:
            raise LLMUnavailableError("Azure OpenAI rejected the credentials.") from exc
        except openai.NotFoundError as exc:
            raise LLMUnavailableError("The Azure OpenAI deployment was not found.") from exc
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            raise LLMError(f"Azure OpenAI transient failure: {type(exc).__name__}") from exc
        except openai.BadRequestError as exc:
            raise LLMUnavailableError(f"Azure OpenAI refused the request: {exc.code}") from exc
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        return text, TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
