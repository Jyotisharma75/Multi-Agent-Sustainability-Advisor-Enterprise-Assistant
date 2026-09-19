"""SQL generation with the real local Hugging Face model.

Skipped unless the ``local`` extra is installed and the configured model is
already in the Hugging Face cache (the test never downloads weights). Run with
``pytest -m local_model``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sustainability_advisor.config.models import Settings
from sustainability_advisor.container import build_container
from sustainability_advisor.devdata.seed import load_spec, seed
from sustainability_advisor.llm.factory import LOCAL, ProviderRegistry
from sustainability_advisor.llm.huggingface_local import HuggingFaceLocalProvider
from tests.conftest import ROOT, make_settings

pytestmark = pytest.mark.local_model


def _cached(model_id: str) -> bool:
    if not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")):
        return False
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


@pytest.fixture
def local_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path, SA_LLM__LOCAL__ENABLED="true")


async def test_local_model_generates_safe_executable_sql(
    local_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _cached(local_settings.llm.local.model_id):
        pytest.skip("local model not installed or not cached")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    provider = HuggingFaceLocalProvider(
        local_settings.llm.local, local_settings.llm.azure_openai.retry
    )
    container = build_container(local_settings, providers=ProviderRegistry({LOCAL: provider}))
    try:
        seed(load_spec(ROOT / "data" / "seed" / "dev_seed.yaml"), container.repositories)
        answer = await container.sql_service.answer(
            "How many facilities are there?", intent="data_query"
        )
        assert answer.generated_by == LOCAL
        assert answer.verification == "not_required"  # no second model available
        assert answer.result.rows == [[3]]
        assert answer.sql.upper().lstrip().startswith("SELECT")
    finally:
        container.close()
