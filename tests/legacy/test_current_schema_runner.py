from __future__ import annotations

import json

import numpy as np
import pytest

from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata
from vla_harness.logging.decision_log import ImagePreprocessMetadata
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata
from vla_harness.legacy.current_schema_runner import CurrentSchemaRunConfig
from vla_harness.legacy.current_schema_runner import CurrentSchemaRunner


class FakePolicyAdapter:
    def __init__(self) -> None:
        self.reset_payloads: list[dict[str, object]] = []
        self.ready_calls = 0

    def assert_ready_for_benchmark(self) -> None:
        self.ready_calls += 1

    def get_server_metadata(self) -> dict[str, object]:
        return {
            "needs_wrist_camera": False,
            "n_external_cameras": 0,
            "needs_stereo_camera": False,
            "needs_session_id": True,
            "action_space": "joint_velocity",
        }

    def infer(self, obs: dict[str, object]) -> dict[str, object]:
        assert "session_id" in obs
        return {"actions": np.zeros((1, 8), dtype=np.float32)}

    def reset(self, reset_info: dict[str, object]) -> None:
        self.reset_payloads.append(reset_info)

    def build_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            family="openpi",
            config_name="pi05_droid",
            checkpoint_ref=None,
            checkpoint_sha256=None,
            checkpoint_sha256_explanation="fixture",
            dtype="bfloat16",
            device="cuda",
            action_space="joint_velocity",
            chunk_size=8,
            prompt_format_source="official_openpi_runtime",
            image_preprocess=ImagePreprocessMetadata(
                resize_resolution=(224, 224),
                resize_filter="bilinear",
                color_space="rgb",
                output_dtype="uint8",
            ),
        )

    def build_validation_metadata(self) -> ValidationMetadata:
        return ValidationMetadata()

    def build_notes(self) -> list[DecisionNote]:
        return []

    def close(self) -> None:
        pass


class FakeEmbodimentAdapter:
    def __init__(self) -> None:
        self.executed: list[tuple[np.ndarray, str]] = []

    def prepare_episode(self, server_metadata: dict[str, object], prompt: str) -> None:
        del server_metadata, prompt

    def build_observation(self, server_metadata: dict[str, object], prompt: str, *, session_id: str | None = None) -> dict[str, object]:
        del server_metadata
        return {
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": prompt,
            "session_id": session_id,
        }

    def execute_action_chunk(self, actions: np.ndarray, *, action_space: str) -> None:
        self.executed.append((actions, action_space))

    def build_embodiment_metadata(self) -> EmbodimentMetadata:
        return EmbodimentMetadata(
            family="dk1",
            backend="fake",
            active_arm="right",
            parked_arm="left",
            parked_arm_rule="robot_native_hold",
            control_hz=15.0,
            chunk_consumption_policy="execute_full_chunk_open_loop",
        )

    def build_notes(self) -> list[DecisionNote]:
        return []


def test_current_schema_runner_writes_log_and_calls_reset(tmp_path):
    policy = FakePolicyAdapter()
    embodiment = FakeEmbodimentAdapter()
    runner = CurrentSchemaRunner(policy, embodiment)

    result = runner.run_episode(
        CurrentSchemaRunConfig(
            prompt="pick",
            max_steps=1,
            run_id="test-run",
            session_id="session-1",
            output_dir=tmp_path,
        )
    )

    assert result.log_path.exists()
    assert policy.ready_calls == 1
    assert policy.reset_payloads == [{"session_id": "session-1"}]
    assert len(embodiment.executed) == 1

    payload = json.loads(result.log_path.read_text(encoding="utf-8"))
    assert payload["step_metrics"][0]["request_bytes"] > 0
    assert payload["step_metrics"][0]["response_bytes"] > 0
    assert payload["runtime"]["concurrency_model"] == "single_rollout_per_process"


def test_current_schema_runner_rejects_reentrant_runs():
    runner = CurrentSchemaRunner(FakePolicyAdapter(), FakeEmbodimentAdapter())
    runner._run_in_progress = True

    with pytest.raises(RuntimeError, match="one active rollout"):
        runner.run_episode(CurrentSchemaRunConfig(prompt="pick"))
