"""Hallucination control through numeric grounding and citation checks.

Every number stated in a generated answer must be traceable to a number in
the evidence gathered by tools (or to a citation excerpt), allowing for
rounding within a relative tolerance and for ratios expressed as percentages.
Small counting numbers and calendar years are exempt. Citation markers of the
form ``[C1]`` must refer to citations that were actually retrieved.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from sustainability_advisor.config.models import GuardrailConfig
from sustainability_advisor.domain.models import Citation, Evidence, Recommendation

# Numbers glued to identifiers (FAC-012, tCO2e, v1.2) are not quantitative claims.
_NUMBER = re.compile(r"(?<![\w.\-/])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w\-]|\.\d)")
_CITATION_MARKER = re.compile(r"\[(C\d+)\]")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")


class GroundingReport(BaseModel):
    grounded: bool
    numbers_checked: int
    ungrounded_numbers: list[str]
    unknown_citation_markers: list[str]


def extract_numbers(text: str) -> list[tuple[str, float]]:
    text = _ISO_DATE.sub(" ", _CITATION_MARKER.sub(" ", text))
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        try:
            found.append((raw, float(raw.replace(",", ""))))
        except ValueError:
            continue
    return found


def _collect(value: Any, sink: list[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int | float):
        if math.isfinite(float(value)):
            sink.append(float(value))
    elif isinstance(value, str):
        sink.extend(v for _, v in extract_numbers(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect(item, sink)
    elif isinstance(value, list | tuple):
        for item in value:
            _collect(item, sink)


def evidence_numbers(
    evidence: Iterable[Evidence],
    citations: Iterable[Citation],
    recommendations: Iterable[Recommendation] = (),
) -> list[float]:
    numbers: list[float] = []
    for item in evidence:
        _collect(item.values, numbers)
        _collect(item.attributes, numbers)
        _collect(item.summary, numbers)
    for citation in citations:
        _collect(citation.excerpt, numbers)
    for rec in recommendations:
        _collect(rec.model_dump(mode="json"), numbers)
    # Ratios are often stated as percentages and totals as their sums.
    return numbers + [n * 100 for n in numbers]


class GroundingChecker:
    def __init__(self, config: GuardrailConfig) -> None:
        self._config = config

    def _is_exempt(self, value: float) -> bool:
        if abs(value) < self._config.grounding_ignore_below:
            return True
        return value.is_integer() and 1900 <= value <= 2100

    def _matches(self, value: float, known: list[float]) -> bool:
        tolerance = self._config.grounding_relative_tolerance
        for candidate in known:
            scale = max(abs(candidate), abs(value), 1e-9)
            if abs(candidate - value) / scale <= tolerance:
                return True
            # A figure rounded to fewer decimals than the evidence holds.
            if abs(abs(candidate) - abs(value)) <= 0.5 * 10 ** -_decimals(value):
                return True
        return False

    def check(
        self,
        answer: str,
        evidence: list[Evidence],
        citations: list[Citation],
        recommendations: list[Recommendation],
        citation_markers: dict[str, str],
    ) -> GroundingReport:
        unknown_markers = sorted(
            {m for m in _CITATION_MARKER.findall(answer) if m not in citation_markers}
        )
        if not self._config.grounding_enabled:
            return GroundingReport(
                grounded=not unknown_markers,
                numbers_checked=0,
                ungrounded_numbers=[],
                unknown_citation_markers=unknown_markers,
            )
        known = evidence_numbers(evidence, citations, recommendations)
        ungrounded = []
        checked = 0
        for raw, value in extract_numbers(answer):
            if self._is_exempt(value):
                continue
            checked += 1
            if not self._matches(value, known):
                ungrounded.append(raw)
        return GroundingReport(
            grounded=not ungrounded and not unknown_markers,
            numbers_checked=checked,
            ungrounded_numbers=ungrounded,
            unknown_citation_markers=unknown_markers,
        )


def _decimals(value: float) -> int:
    text = repr(value)
    if "e" in text or "." not in text:
        return 0
    return len(text.split(".")[1].rstrip("0"))
