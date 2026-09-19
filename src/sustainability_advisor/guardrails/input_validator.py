"""User input validation and normalisation."""

from __future__ import annotations

import re
import unicodedata

from sustainability_advisor.config.models import GuardrailConfig
from sustainability_advisor.domain.errors import InputRejectedError

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def normalise_text(text: str) -> str:
    """NFKC normalise, drop zero width and control characters, trim."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return text.strip()


class InputValidator:
    def __init__(self, config: GuardrailConfig) -> None:
        self._config = config
        self._identifier = re.compile(config.identifier_pattern)

    def message(self, text: str) -> str:
        cleaned = normalise_text(text)
        if len(cleaned) < self._config.min_input_chars:
            raise InputRejectedError("The message is empty or too short.")
        if len(cleaned) > self._config.max_input_chars:
            raise InputRejectedError(
                f"The message exceeds {self._config.max_input_chars} characters."
            )
        return cleaned

    def identifiers(self, values: list[str], *, kind: str) -> list[str]:
        if len(values) > self._config.max_facility_ids:
            raise InputRejectedError(
                f"At most {self._config.max_facility_ids} {kind} values may be given."
            )
        for value in values:
            if not self._identifier.fullmatch(value):
                raise InputRejectedError(f"{kind} {value!r} is not a valid identifier.")
        return values
