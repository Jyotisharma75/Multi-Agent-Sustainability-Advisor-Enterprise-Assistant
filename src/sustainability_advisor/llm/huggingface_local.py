"""Local Hugging Face provider.

Runs an instruction tuned causal model in this process. Its main job is SQL
generation for simple questions and independent verification of SQL written
by the hosted model; it is also the offline fallback for agent reasoning.

* the model loads lazily and once, behind a lock
* generation is CPU bound, so it runs in a worker thread and is serialised by
  a semaphore sized from configuration
* a generation that outruns its timeout is abandoned by the caller; the
  concurrency limit is what prevents slow work from piling up
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from typing import Any

from sustainability_advisor.config.models import LocalModelConfig, RetryConfig
from sustainability_advisor.domain.errors import LLMTimeoutError, LLMUnavailableError
from sustainability_advisor.domain.models import TokenUsage
from sustainability_advisor.llm.base import ChatMessage, LLMProvider
from sustainability_advisor.observability.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceLocalProvider(LLMProvider):
    def __init__(self, config: LocalModelConfig, retry: RetryConfig) -> None:
        super().__init__(retry)
        self._config = config
        self._lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_error: str | None = None

    @property
    def name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._config.model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def is_available(self) -> bool:
        if not self._config.enabled or self._load_error is not None:
            return False
        return bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        with self._lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                self._load_error = str(exc)
                raise LLMUnavailableError(
                    "The local model requires the optional dependencies: pip install '.[local]'."
                ) from exc
            if self._config.num_threads:
                torch.set_num_threads(self._config.num_threads)
            common: dict[str, Any] = {
                "revision": self._config.revision,
                "trust_remote_code": self._config.trust_remote_code,
            }
            if self._config.cache_dir:
                common["cache_dir"] = self._config.cache_dir
            logger.info("local_model.loading", model_id=self._config.model_id)
            tokenizer: Any
            model: Any
            try:
                tokenizer = AutoTokenizer.from_pretrained(self._config.model_id, **common)
                dtype = (
                    "auto" if self._config.dtype == "auto" else getattr(torch, self._config.dtype)
                )
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        self._config.model_id, dtype=dtype, **common
                    )
                except TypeError:
                    model = AutoModelForCausalLM.from_pretrained(
                        self._config.model_id, torch_dtype=dtype, **common
                    )
            except OSError as exc:
                self._load_error = str(exc)
                raise LLMUnavailableError(
                    f"The local model {self._config.model_id} could not be loaded."
                ) from exc
            model.to(self._config.device)
            model.eval()
            self._tokenizer, self._model = tokenizer, model
            logger.info("local_model.loaded", model_id=self._config.model_id)
            return tokenizer, model

    async def preload(self) -> None:
        if self.is_available():
            await asyncio.to_thread(self._load)

    async def _complete(
        self,
        messages: list[ChatMessage],
        *,
        task: str,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, TokenUsage]:
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self._generate,
                        messages,
                        max_tokens or self._config.max_new_tokens,
                        self._config.temperature if temperature is None else temperature,
                    ),
                    timeout=self._config.timeout_seconds,
                )
            except TimeoutError as exc:
                raise LLMTimeoutError(
                    f"The local model did not finish within {self._config.timeout_seconds}s."
                ) from exc
            except MemoryError as exc:
                raise LLMUnavailableError("The local model ran out of memory.") from exc

    def _generate(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float
    ) -> tuple[str, TokenUsage]:
        import torch

        tokenizer, model = self._load()
        prompt = tokenizer.apply_chat_template(
            [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(self._config.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        new_tokens = generated[0][prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return str(text), TokenUsage(
            prompt_tokens=prompt_tokens, completion_tokens=int(new_tokens.shape[0])
        )

    async def probe(self) -> str:
        if not self.is_available():
            return "unavailable"
        return "loaded" if self.is_loaded else "installed"
