"""Smoke test for the scripts/run_episode.py launcher.

Runs the launcher in --dry-run mode for both supported policies. Verifies it
exits cleanly, writes a fairness log, and reports the manifest name.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_episode.py"


def _load_main():
    spec = importlib.util.spec_from_file_location("run_episode_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main


@pytest.mark.parametrize(
    ("policy", "expected_manifest"),
    [
        ("molmoact2_yam", "molmoact2_bimanual_yam"),
        ("openpi_aloha", "openpi_aloha_pen_uncap"),
    ],
)
def test_dry_run_writes_fairness_log(tmp_path, policy, expected_manifest, capsys):
    main = _load_main()
    rc = main(
        [
            "--policy", policy,
            "--embodiment", "fake",
            "--dry-run",
            "--prompt", "smoke test",
            "--max-steps", "2",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr().out
    assert f"manifest: {expected_manifest}" in captured

    logs = list(tmp_path.glob("*/decision_log.json"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text(encoding="utf-8"))
    assert payload["run"]["prompt"] == "smoke test"
    assert len(payload["step_metrics"]) == 2
    assert payload["embodiment"]["backend"] == "in_memory_zero_fill"


def test_list_configs_prints_field_table(capsys):
    main = _load_main()
    rc = main(["--list-configs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--policy molmoact2_yam" in out
    assert "--embodiment yam" in out
    assert "server_url" in out
    assert "control_hz" in out


def test_missing_backend_loader_for_real_embodiment_errors(tmp_path):
    main = _load_main()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--policy", "molmoact2_yam",
                "--embodiment", "yam",
                "--prompt", "x",
                "--output-dir", str(tmp_path),
            ]
        )
    assert "backend-loader" in str(exc.value)
