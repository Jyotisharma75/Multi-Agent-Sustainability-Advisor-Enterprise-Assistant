"""Provider registry and fallback chain."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from sustainability_advisor.config.models import LLMConfig
from sustainability_advisor.domain.errors import LLMError, LLMUnavailableError
from sustainability_advisor.llm.azure_openai import AzureOpenAIProvider
from sustainability_advisor.llm.base import ChatMessage, LLMProvider, LLMResponse
from sustainability_advisor.llm.huggingface_local import HuggingFaceLocalProvider
from sustainability_advisor.llm.structured import complete_structured
from sustainability_advisor.observability.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

AZURE = "azure_openai"
LOCAL = "local"
MOCK = "mock"


class ProviderRegistry:
    def __init__(self, providers: dict[str, LLMProvider]) -> None:
        self._providers = providers

    def get(self, name: str) -> LLMProvider:
        provider = self.try_get(name)
        if provider is None:
            raise LLMUnavailableError(f"The model provider {name!r} is not available.")
        return provider

    def try_get(self, name: str) -> LLMProvider | None:
        provider = self._providers.get(name)
        if provider is None or not provider.is_available():
            return None
        return provider

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def all(self) -> dict[str, LLMProvider]:
        return dict(self._providers)


class ProviderChain:
    """Tries providers in configured order, falling back on failure."""

    def __init__(
        self,
        registry: ProviderRegistry,
        order: list[str],
        *,
        repair_template: str,
        repair_attempts: int,
        task_orders: dict[str, list[str]] | None = None,
    ) -> None:
        self._registry = registry
        self._order = order
        self._task_orders = task_orders or {}
        self._repair_template = repair_template
        self._repair_attempts = repair_attempts

    def available(self, task: str | None = None) -> list[LLMProvider]:
        order = self._task_orders.get(task, self._order) if task else self._order
        return [p for name in order if (p := self._registry.try_get(name)) is not None]

    async def structured(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        task: str,
        max_tokens: int | None = None,
    ) -> tuple[T, list[LLMResponse]]:
        providers = self.available(task)
        if not providers:
            raise LLMUnavailableError("No language model provider is available.")
        last: LLMError | None = None
        for provider in providers:
            try:
                return await complete_structured(
                    provider,
                    messages,
                    schema,
                    task=task,
                    repair_template=self._repair_template,
                    repair_attempts=self._repair_attempts,
                    max_tokens=max_tokens,
                )
            except LLMError as exc:
                logger.warning(
                    "llm.provider_fallback", provider=provider.name, task=task, error=exc.code
                )
                last = exc
        assert last is not None
        raise last


def build_registry(config: LLMConfig) -> ProviderRegistry:
    providers: dict[str, LLMProvider] = {
        AZURE: AzureOpenAIProvider(config.azure_openai),
        LOCAL: HuggingFaceLocalProvider(config.local, config.azure_openai.retry),
    }
    return ProviderRegistry(providers)
