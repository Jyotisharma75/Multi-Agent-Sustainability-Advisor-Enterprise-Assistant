"""Prompt injection detection.

Applied to user messages (direct injection) and to retrieved document passages
(indirect injection). Each configured pattern carries a weight; the text's
score is the sum of the weights of the patterns it matches, computed on a
normalised copy so spacing, casing and zero width characters do not help an
attacker. Patterns and the blocking threshold are configuration.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from sustainability_advisor.config.models import GuardrailConfig
from sustainability_advisor.domain.errors import PromptInjectionError
from sustainability_advisor.guardrails.input_validator import normalise_text


class InjectionAssessment(BaseModel):
    score: float
    matched: list[str]
    blocked: bool


class InjectionDetector:
    def __init__(self, config: GuardrailConfig) -> None:
        self._enabled = config.injection_enabled
        self._threshold = config.injection_block_threshold
        self._patterns = [
            (re.compile(p.pattern, re.IGNORECASE), p.weight, p.pattern)
            for p in config.injection_patterns
        ]

    def assess(self, text: str) -> InjectionAssessment:
        if not self._enabled:
            return InjectionAssessment(score=0.0, matched=[], blocked=False)
        normalised = re.sub(r"\s+", " ", normalise_text(text).lower())
        matched = [source for regex, _, source in self._patterns if regex.search(normalised)]
        score = sum(weight for regex, weight, _ in self._patterns if regex.search(normalised))
        return InjectionAssessment(
            score=round(score, 4), matched=matched, blocked=score >= self._threshold
        )

    def check(self, text: str) -> InjectionAssessment:
        assessment = self.assess(text)
        if assessment.blocked:
            raise PromptInjectionError(
                "The request was blocked because it resembles an attempt to override the "
                "assistant's instructions.",
                details={"score": assessment.score},
            )
        return assessment

    def is_suspicious(self, text: str) -> bool:
        return self.assess(text).blocked
