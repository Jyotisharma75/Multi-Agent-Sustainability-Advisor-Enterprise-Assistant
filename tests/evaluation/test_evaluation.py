"""Golden set evaluation. Fails the build when any metric drops below its
threshold in ``evaluation/golden_set.yaml``."""

from __future__ import annotations

import pytest

from sustainability_advisor.container import Container
from sustainability_advisor.evaluation.harness import evaluate, load_golden_set
from tests.conftest import ROOT

pytestmark = pytest.mark.evaluation


async def test_golden_set_meets_thresholds(container: Container) -> None:
    golden = load_golden_set(ROOT / "evaluation" / "golden_set.yaml")
    report = await evaluate(container, golden)
    below = {k: v for k, v in report.metrics.items() if v < report.thresholds[k]}
    assert report.passed, f"metrics below threshold: {below}; failures: {report.failures}"
