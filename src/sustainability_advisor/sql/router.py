"""Complexity based model routing for SQL generation.

The router scores a question before any SQL exists, using signals that are
already known: how many exposed tables the question mentions, its length,
whether it asks for aggregation, comparison or time windows, and the intent
the orchestrator classified. Each signal is normalised to 0..1 and weighted
by configuration.

Simple questions go to the configured "simple" provider (the local Hugging
Face coder model by default); complex ones to the "complex" provider (Azure
OpenAI by default). The provider that did not generate becomes the verifier,
which keeps cross checking independent. If only one provider is available it
generates and verification is reported as unavailable.
"""

from __future__ import annotations

import re

from sustainability_advisor.config.models import RoutingConfig
from sustainability_advisor.domain.errors import LLMUnavailableError
from sustainability_advisor.llm.factory import ProviderRegistry
from sustainability_advisor.observability.logging import get_logger
from sustainability_advisor.sql.catalog import SchemaCatalog
from sustainability_advisor.sql.models import RoutingDecision

logger = get_logger(__name__)


def _saturate(value: float, saturation: int) -> float:
    return min(float(value) / float(saturation), 1.0) if saturation > 0 else 0.0


def _count_terms(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if re.search(rf"\b{re.escape(t.lower())}\b", text))


class ModelRouter:
    def __init__(
        self, config: RoutingConfig, registry: ProviderRegistry, catalog: SchemaCatalog
    ) -> None:
        self._config = config
        self._registry = registry
        self._catalog = catalog

    def complexity(self, question: str, intent: str) -> tuple[float, dict[str, float]]:
        cfg = self._config
        text = question.lower()
        features = {
            "table_count": _saturate(
                len(self._catalog.mentioned_tables(question)), cfg.table_count_saturation
            ),
            "question_length": _saturate(len(text.split()), cfg.question_length_saturation),
            "aggregation": min(_count_terms(text, cfg.aggregation_keywords), 1),
            "time_expressions": _saturate(
                _count_terms(text, cfg.time_keywords), cfg.time_expression_saturation
            ),
            "comparison": min(_count_terms(text, cfg.comparison_keywords), 1),
            "intent": cfg.intent_complexity.get(intent, cfg.intent_complexity.get("default", 0.5)),
        }
        weights = cfg.weights.model_dump()
        total = sum(weights.values())
        if total <= 0:
            return 0.0, {k: float(v) for k, v in features.items()}
        score = sum(float(features[name]) * weight for name, weight in weights.items()) / total
        return round(min(max(score, 0.0), 1.0), 4), {k: float(v) for k, v in features.items()}

    def route(self, question: str, intent: str) -> RoutingDecision:
        cfg = self._config
        simple = self._registry.try_get(cfg.primary_when_simple)
        complex_ = self._registry.try_get(cfg.primary_when_complex)
        if simple is None and complex_ is None:
            raise LLMUnavailableError("No SQL generation model is available.")
        score, features = self.complexity(question, intent)

        if not cfg.enabled:
            primary = (complex_ or simple).name  # type: ignore[union-attr]
            decision = RoutingDecision(
                primary=primary,
                verifier=None,
                complexity=score,
                features=features,
                reason="routing disabled",
            )
        elif score <= cfg.local_max_complexity and simple is not None:
            decision = RoutingDecision(
                primary=simple.name,
                verifier=complex_.name if complex_ is not None else None,
                complexity=score,
                features=features,
                reason="complexity at or below the simple threshold",
            )
        else:
            primary_provider = complex_ or simple
            assert primary_provider is not None
            other = simple if primary_provider is complex_ else complex_
            decision = RoutingDecision(
                primary=primary_provider.name,
                verifier=other.name if other is not None else None,
                complexity=score,
                features=features,
                reason=(
                    "complexity above the simple threshold"
                    if complex_ is not None
                    else "only the simple provider is available"
                ),
            )
        logger.info(
            "sql.routing",
            primary=decision.primary,
            verifier=decision.verifier,
            complexity=decision.complexity,
            reason=decision.reason,
        )
        return decision

    def should_verify(self, decision: RoutingDecision, confidence: float) -> bool:
        if decision.verifier is None:
            return False
        if decision.complexity >= self._config.always_verify_above_complexity:
            return True
        return confidence < self._config.verify_below_confidence

    @property
    def authoritative_provider(self) -> str:
        """Provider whose SQL wins a disagreement that data cannot settle."""
        return self._config.primary_when_complex
