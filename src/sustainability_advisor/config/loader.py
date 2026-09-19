"""Layered configuration loading.

Resolution order, later layers winning:

1. ``configs/base.yaml``
2. ``configs/<SA_ENV>.yaml``
3. environment variables named ``SA_<SECTION>__<KEY>[__<KEY>...]``

An environment value is parsed as JSON when it is valid JSON, so lists,
numbers and booleans can be overridden, and is used as a plain string
otherwise. ``SA_ENV`` selects the overlay and ``SA_CONFIG_DIR`` the folder.
A ``.env`` file in the working directory is read first without overriding
variables that are already set.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from sustainability_advisor.config.models import Settings

ENV_PREFIX = "SA_"
ENV_SELECTOR = "SA_ENV"
CONFIG_DIR_VARIABLE = "SA_CONFIG_DIR"
NESTED_DELIMITER = "__"
_RESERVED = frozenset({ENV_SELECTOR, CONFIG_DIR_VARIABLE})


class ConfigurationError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated with ``overlay``.

    Mappings merge key by key. Any other value in the overlay, lists included,
    replaces the base value entirely so an environment can shrink a list.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ConfigurationError(f"Configuration file {path} must contain a mapping")
    return content


def _parse_env_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Translate ``SA_`` variables into a nested override mapping."""
    overrides: dict[str, Any] = {}
    for name, raw in environ.items():
        # Every section is a mapping, so an override always contains the
        # delimiter. This keeps secret carrying variables such as SA_API_KEYS
        # out of the configuration tree.
        if not name.startswith(ENV_PREFIX) or name in _RESERVED or NESTED_DELIMITER not in name:
            continue
        path = [part.lower() for part in name[len(ENV_PREFIX) :].split(NESTED_DELIMITER)]
        if not all(path):
            continue
        cursor = overrides
        for part in path[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _parse_env_value(raw)
    return overrides


def default_config_dir() -> Path:
    """Return the configuration folder, honouring ``SA_CONFIG_DIR``."""
    configured = os.environ.get(CONFIG_DIR_VARIABLE)
    if configured:
        return Path(configured)
    return Path.cwd() / "configs"


def load_settings(
    environment: str | None = None,
    config_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load, merge and validate configuration."""
    if environ is None:
        load_dotenv(override=False)
        environ = os.environ
    env_name = environment or environ.get(ENV_SELECTOR)
    if not env_name:
        raise ConfigurationError(f"{ENV_SELECTOR} must be set to select a configuration overlay")
    folder = config_dir or default_config_dir()
    merged = deep_merge(_read_yaml(folder / "base.yaml"), _read_yaml(folder / f"{env_name}.yaml"))
    merged = deep_merge(merged, env_overrides(environ))
    try:
        return Settings.model_validate(merged)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid configuration for {env_name!r}: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process wide settings, loaded once."""
    return load_settings()


def read_secret(variable_name: str) -> str | None:
    """Return the value of the environment variable named by a ``*_env`` field."""
    value = os.environ.get(variable_name)
    return value if value else None
