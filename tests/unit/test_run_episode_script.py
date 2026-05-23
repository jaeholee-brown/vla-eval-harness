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


def test_missing_yam_config_uses_default_loader_and_errors_clearly(tmp_path):
    """--embodiment yam now defaults to the shipped loader, which expects --yam-config."""
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
    msg = str(exc.value)
    assert "--yam-config" in msg


def test_custom_backend_loader_receives_config_path(tmp_path, monkeypatch):
    """A user-supplied --backend-loader must receive the --yam-config kwarg."""
    import sys
    import types

    received: dict[str, object] = {}

    fake_module = types.ModuleType("user_yam_backend_for_test")

    class _FakeBackend:
        def prepare_episode(self, prompt):
            return None

        def capture(self):
            import numpy as np
            from vla_harness.adapters.embodiment._sample_types import BimanualObservationSample
            return BimanualObservationSample(
                camera_frames={
                    "front_camera": np.zeros((8, 8, 3), dtype=np.uint8),
                    "left_camera": np.zeros((8, 8, 3), dtype=np.uint8),
                    "right_camera": np.zeros((8, 8, 3), dtype=np.uint8),
                },
                arm_state_streams={
                    "left_arm": {"joint_position": np.zeros(7, dtype="float32")},
                    "right_arm": {"joint_position": np.zeros(7, dtype="float32")},
                },
            )

        def execute_chunk(self, arm_actions, *, padding_rules, manifest):
            return None

        def close(self):
            return None

    def make_backend(*, config_path=None, **_):
        received["config_path"] = config_path
        return _FakeBackend()

    fake_module.make_backend = make_backend
    monkeypatch.setitem(sys.modules, "user_yam_backend_for_test", fake_module)

    main = _load_main()
    rc = main(
        [
            "--policy", "molmoact2_yam",
            "--embodiment", "yam",
            "--backend-loader", "user_yam_backend_for_test:make_backend",
            "--yam-config", "/etc/yam/test.yaml",
            "--dry-run",  # fake the policy client
            "--prompt", "smoke",
            "--max-steps", "1",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    assert received["config_path"] == "/etc/yam/test.yaml"
