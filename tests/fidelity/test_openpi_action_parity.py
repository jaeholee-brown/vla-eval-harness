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


def _load_observation_fixture(path: Path) -> dict[str, object]:
    payload = np.load(path, allow_pickle=True)
    obs: dict[str, object] = {}
    for key in payload.files:
        value = payload[key]
        if value.dtype == object and value.ndim == 0:
            obs[key] = value.item()
        else:
            obs[key] = value
    return obs


@pytest.mark.skipif(
    "OPENPI_ACTION_FIXTURE_DIR" not in os.environ
    or "OPENPI_OFFICIAL_ACTION_CALLABLE" not in os.environ
    or "OPENPI_HARNESS_ACTION_CALLABLE" not in os.environ,
    reason="Set OPENPI_ACTION_FIXTURE_DIR, OPENPI_OFFICIAL_ACTION_CALLABLE, and OPENPI_HARNESS_ACTION_CALLABLE to run.",
)
def test_openpi_action_parity():
    corpus_dir = Path(os.environ["OPENPI_ACTION_FIXTURE_DIR"])
    official_predict = _load_callable(os.environ["OPENPI_OFFICIAL_ACTION_CALLABLE"])
    harness_predict = _load_callable(os.environ["OPENPI_HARNESS_ACTION_CALLABLE"])

    fixture_paths = sorted(corpus_dir.glob("*.npz"))
    assert fixture_paths, "Expected at least one .npz observation fixture in the action corpus."

    for fixture_path in fixture_paths:
        obs = _load_observation_fixture(fixture_path)
        expected = official_predict(obs)["actions"]
        observed = harness_predict(obs)["actions"]
        np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
