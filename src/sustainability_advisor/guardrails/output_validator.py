"""Output sanitisation.

Removes any reasoning a model emitted inside configured tags (for example
``<think>``), removes typographic dashes the house style forbids, and bounds
the answer length. The API only ever returns the sanitised text.
"""

from __future__ import annotations

import re

from sustainability_advisor.config.models import GuardrailConfig

_DASHES = {"\u2014": ", ", "\u2013": "-"}


class OutputValidator:
    def __init__(self, config: GuardrailConfig) -> None:
        self._config = config
        self._reasoning = [
            re.compile(rf"<{re.escape(tag)}\b[^>]*>.*?(</{re.escape(tag)}>|$)", re.DOTALL | re.I)
            for tag in config.hidden_reasoning_tags
        ]

    def sanitise(self, text: str) -> str:
        for pattern in self._reasoning:
            text = pattern.sub("", text)
        for dash, replacement in _DASHES.items():
            text = text.replace(f" {dash} ", replacement).replace(dash, replacement)
        text = re.sub(r"[ \t]+\n", "\n", text).strip()
        limit = self._config.max_answer_chars
        if len(text) > limit:
            cut = text[:limit].rsplit(" ", 1)[0]
            text = cut.rstrip(",;:") + "..."
        return text
