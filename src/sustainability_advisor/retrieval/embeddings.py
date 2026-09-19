"""Embedding providers.

* ``azure_openai``: Azure OpenAI embedding deployment (production)
* ``sentence_transformers``: a local Hugging Face sentence embedding model
* ``hash``: signed feature hashing of word unigrams and bigrams. It is a real,
  deterministic lexical embedding with no model download, used in tests and
  air gapped development. It captures vocabulary overlap, not meaning.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
import re
import threading
from abc import ABC, abstractmethod
from typing import Any

from sustainability_advisor.config.loader import read_secret
from sustainability_advisor.config.models import AzureOpenAIConfig, EmbeddingConfig
from sustainability_advisor.domain.errors import RetrievalError

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(ABC):
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm > 0 else vector


class HashEmbedding(EmbeddingProvider):
    def _vector(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in itertools.pairwise(tokens)]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += 1.0 if (value >> 63) & 1 else -1.0
        return _normalise(vector)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RetrievalError(
                            "sentence-transformers is not installed: pip install '.[local]'."
                        ) from exc
                    self._model = SentenceTransformer(self.config.model, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, batch_size=self.config.batch_size, normalize_embeddings=True)
        return [[float(v) for v in row] for row in vectors]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)


class AzureOpenAIEmbedding(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig, azure: AzureOpenAIConfig) -> None:
        super().__init__(config)
        self._azure = azure
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncAzureOpenAI

            endpoint = read_secret(self._azure.endpoint_env)
            if not endpoint:
                raise RetrievalError(f"{self._azure.endpoint_env} is not set.")
            kwargs: dict[str, Any] = {
                "azure_endpoint": endpoint,
                "api_version": self._azure.api_version,
                "timeout": self._azure.timeout_seconds,
            }
            if self._azure.auth_mode == "entra":
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    DefaultAzureCredential(), self._azure.token_scope
                )
            else:
                kwargs["api_key"] = read_secret(self._azure.api_key_env)
            self._client = AsyncAzureOpenAI(**kwargs)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        vectors: list[list[float]] = []
        size = self.config.batch_size
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            try:
                response = await client.embeddings.create(
                    model=self.config.model, input=batch, dimensions=self.dimensions
                )
            except Exception as exc:
                raise RetrievalError(f"Embedding request failed: {type(exc).__name__}") from exc
            vectors.extend([list(item.embedding) for item in response.data])
        return vectors


def build_embedding_provider(
    config: EmbeddingConfig, azure: AzureOpenAIConfig
) -> EmbeddingProvider:
    if config.provider == "azure_openai":
        return AzureOpenAIEmbedding(config, azure)
    if config.provider == "sentence_transformers":
        return SentenceTransformerEmbedding(config)
    return HashEmbedding(config)
