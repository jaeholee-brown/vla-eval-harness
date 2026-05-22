from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest


def _load_callable(spec: str):
    module_name, function_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _load_fixture_images(corpus_dir: Path) -> list[np.ndarray]:
    fixtures: list[np.ndarray] = []
    for path in sorted(corpus_dir.glob("*.npy")):
        fixtures.append(np.load(path))
    return fixtures


@pytest.mark.skipif(
    "OPENPI_PREPROCESS_FIXTURE_DIR" not in os.environ
    or "OPENPI_OFFICIAL_PREPROCESS" not in os.environ
    or "OPENPI_HARNESS_PREPROCESS" not in os.environ,
    reason="Set OPENPI_PREPROCESS_FIXTURE_DIR, OPENPI_OFFICIAL_PREPROCESS, and OPENPI_HARNESS_PREPROCESS to run.",
)
def test_openpi_preprocessing_parity():
    corpus_dir = Path(os.environ["OPENPI_PREPROCESS_FIXTURE_DIR"])
    official_preprocess = _load_callable(os.environ["OPENPI_OFFICIAL_PREPROCESS"])
    harness_preprocess = _load_callable(os.environ["OPENPI_HARNESS_PREPROCESS"])

    fixture_images = _load_fixture_images(corpus_dir)
    assert fixture_images, "Expected at least one .npy image in the preprocessing fixture corpus."

    for image in fixture_images:
        expected = official_preprocess(image)
        observed = harness_preprocess(image)
        np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
