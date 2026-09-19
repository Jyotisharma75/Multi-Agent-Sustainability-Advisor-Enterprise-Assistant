"""Versioned prompt library.

Prompts live in ``prompts/<name>/<version>.yaml`` as named templates. Values
are substituted with ``string.Template`` (``$variable``) so JSON examples in
prompts need no escaping. A missing variable is an error, never a silent gap.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

import yaml

from sustainability_advisor.config.models import PromptConfig
from sustainability_advisor.domain.errors import AdvisorError
from sustainability_advisor.llm.base import ChatMessage


class PromptError(AdvisorError):
    code = "prompt_error"


class PromptLibrary:
    def __init__(self, config: PromptConfig) -> None:
        self._root = Path(config.directory)
        self._version = config.version
        self._cache: dict[str, dict[str, str]] = {}

    def templates(self, name: str) -> dict[str, str]:
        if name not in self._cache:
            path = self._root / name / f"{self._version}.yaml"
            if not path.is_file():
                raise PromptError(f"Prompt {name!r} version {self._version!r} not found at {path}")
            with path.open(encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
                raise PromptError(f"Prompt file {path} must map names to strings")
            self._cache[name] = {str(k): str(v) for k, v in data.items()}
        return self._cache[name]

    def text(self, name: str, key: str, **values: object) -> str:
        templates = self.templates(name)
        if key not in templates:
            raise PromptError(f"Prompt {name!r} has no template {key!r}")
        try:
            return Template(templates[key]).substitute({k: str(v) for k, v in values.items()})
        except KeyError as exc:
            raise PromptError(f"Prompt {name}.{key} is missing variable {exc}") from exc

    def messages(self, name: str, **values: object) -> list[ChatMessage]:
        """Render the ``system`` and ``user`` templates of a prompt."""
        return [
            ChatMessage(role="system", content=self.text(name, "system", **values)),
            ChatMessage(role="user", content=self.text(name, "user", **values)),
        ]

    def validate(self, names: list[str]) -> None:
        """Load every named prompt so a missing file fails at start up."""
        for name in names:
            self.templates(name)
