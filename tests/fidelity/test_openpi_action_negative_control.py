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
    or "OPENPI_NEGATIVE_CONTROL_ACTION_CALLABLE" not in os.environ,
    reason=(
        "Set OPENPI_ACTION_FIXTURE_DIR, OPENPI_OFFICIAL_ACTION_CALLABLE, and "
        "OPENPI_NEGATIVE_CONTROL_ACTION_CALLABLE to run."
    ),
)
def test_openpi_action_negative_control_separates_from_official():
    corpus_dir = Path(os.environ["OPENPI_ACTION_FIXTURE_DIR"])
    official_predict = _load_callable(os.environ["OPENPI_OFFICIAL_ACTION_CALLABLE"])
    negative_control_predict = _load_callable(os.environ["OPENPI_NEGATIVE_CONTROL_ACTION_CALLABLE"])
    min_abs_diff = float(os.environ.get("OPENPI_NEGATIVE_CONTROL_MIN_ABS_DIFF", "1e-4"))

    fixture_paths = sorted(corpus_dir.glob("*.npz"))
    assert fixture_paths, "Expected at least one .npz observation fixture in the action corpus."

    separated = False
    max_observed_abs_diff = 0.0
    for fixture_path in fixture_paths:
        obs = _load_observation_fixture(fixture_path)
        expected = official_predict(obs)["actions"]
        observed = negative_control_predict(obs)["actions"]
        abs_diff = float(np.max(np.abs(observed - expected)))
        max_observed_abs_diff = max(max_observed_abs_diff, abs_diff)
        if abs_diff > min_abs_diff:
            separated = True
            break

    assert separated, (
        "Negative control did not separate from the official policy output. "
        f"Max observed absolute diff was {max_observed_abs_diff} with threshold {min_abs_diff}."
    )
