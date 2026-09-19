from __future__ import annotations

from pathlib import Path

import pytest

from sustainability_advisor.config.loader import (
    ConfigurationError,
    deep_merge,
    env_overrides,
    load_settings,
)
from tests.conftest import ROOT, make_settings

pytestmark = pytest.mark.unit


def test_deep_merge_merges_mappings_and_replaces_lists() -> None:
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 1}
    merged = deep_merge(base, {"a": {"c": [3]}, "e": 2})
    assert merged == {"a": {"b": 1, "c": [3]}, "d": 1, "e": 2}


def test_env_overrides_parse_json_and_require_delimiter() -> None:
    overrides = env_overrides(
        {
            "SA_LOGGING__LEVEL": "DEBUG",
            "SA_ROUTING__LOCAL_MAX_COMPLEXITY": "0.25",
            "SA_SQL__ALLOWED_STATEMENTS": '["SELECT"]',
            "SA_API_KEYS": "secret",
            "SA_ENV": "test",
            "OTHER": "x",
        }
    )
    assert overrides == {
        "logging": {"level": "DEBUG"},
        "routing": {"local_max_complexity": 0.25},
        "sql": {"allowed_statements": ["SELECT"]},
    }


@pytest.mark.parametrize("environment", ["development", "test", "production"])
def test_every_environment_loads(environment: str) -> None:
    settings = load_settings(environment=environment, config_dir=ROOT / "configs", environ={})
    assert settings.app.environment == environment
    assert set(settings.agents) >= {"orchestrator", "response_synthesizer"}


def test_environment_variable_overrides_yaml(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, SA_ROUTING__LOCAL_MAX_COMPLEXITY="0.11")
    assert settings.routing.local_max_complexity == pytest.approx(0.11)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        make_settings(tmp_path, SA_LOGGING__NOT_A_SETTING="1")


def test_cross_reference_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="unknown tools"):
        make_settings(tmp_path, SA_AGENTS__SQL_AGENT__ALLOWED_TOOLS='["drop_database"]')


def test_missing_environment_selector_fails() -> None:
    with pytest.raises(ConfigurationError, match="SA_ENV"):
        load_settings(config_dir=ROOT / "configs", environ={})


def test_production_has_no_inline_secrets() -> None:
    text = (ROOT / "configs" / "production.yaml").read_text(encoding="utf-8").lower()
    for marker in ("password:", "api_key:", "secret:", "connectionstring"):
        assert marker not in text
