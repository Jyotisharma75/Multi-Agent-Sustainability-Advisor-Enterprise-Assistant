"""Shared fixtures.

Every test runs against the ``test`` configuration: a SQLite database in a
temporary folder seeded from ``data/seed/dev_seed.yaml``, an in process hybrid
index with hash embeddings over ``data/documents``, and two scripted mock
language models registered under the provider names ``local`` and
``azure_openai`` so routing, verification and fallback paths run exactly as
in production without any network access.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sustainability_advisor.config.loader import load_settings
from sustainability_advisor.config.models import Settings
from sustainability_advisor.container import Container, build_container
from sustainability_advisor.devdata.seed import load_spec, seed
from sustainability_advisor.llm.factory import AZURE, LOCAL, ProviderRegistry
from sustainability_advisor.llm.mock import MockLLM
from sustainability_advisor.main import create_app

ROOT = Path(__file__).resolve().parents[1]
API_KEY = "test-key-0123456789"


@pytest.fixture(autouse=True)
def _project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def make_settings(tmp_path: Path, **overrides: str) -> Settings:
    environ = {"SA_DATABASE__SQLITE__PATH": str(tmp_path / "advisor.db"), **overrides}
    return load_settings(environment="test", config_dir=ROOT / "configs", environ=environ)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def local_llm() -> MockLLM:
    return MockLLM(name=LOCAL, model="mock-local-coder")


@pytest.fixture
def azure_llm() -> MockLLM:
    return MockLLM(name=AZURE, model="mock-azure-gpt")


@pytest.fixture
def registry(local_llm: MockLLM, azure_llm: MockLLM) -> ProviderRegistry:
    return ProviderRegistry({LOCAL: local_llm, AZURE: azure_llm})


@pytest.fixture
async def container(settings: Settings, registry: ProviderRegistry) -> AsyncIterator[Container]:
    built = build_container(settings, providers=registry)
    seed(load_spec(ROOT / "data" / "seed" / "dev_seed.yaml"), built.repositories)
    await built.startup()
    yield built
    built.close()


@pytest.fixture
def client(
    settings: Settings, registry: ProviderRegistry, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv(settings.api.auth.api_keys_env, f"other-key,{API_KEY}")
    built = build_container(settings, providers=registry)
    seed(load_spec(ROOT / "data" / "seed" / "dev_seed.yaml"), built.repositories)
    app = create_app(built)
    with TestClient(app) as test_client:
        test_client.headers.update({settings.api.auth.header_name: API_KEY})
        yield test_client


def as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload)
