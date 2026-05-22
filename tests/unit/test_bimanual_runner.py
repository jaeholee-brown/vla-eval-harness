from __future__ import annotations

import json

import numpy as np
import pytest

from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata
from vla_harness.logging.decision_log import ImagePreprocessMetadata
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata
from vla_harness.protocol import ActionChunk
from vla_harness.protocol import ActionPacket
from vla_harness.protocol import ActionSemantics
from vla_harness.protocol import ActionStreamSpec
from vla_harness.protocol import ArmActionGroup
from vla_harness.protocol import ArmGroupSpec
from vla_harness.protocol import ArmObservationGroup
from vla_harness.protocol import HarnessManifest
from vla_harness.protocol import ObservationPacket
from vla_harness.protocol import StateStreamSpec
from vla_harness.protocol import TemporalStateSequence
from vla_harness.protocol import TemporalVideoSequence
from vla_harness.protocol import VideoStreamSpec
from vla_harness.runner import BimanualRunConfig
from vla_harness.runner import BimanualRunner


def _manifest() -> HarnessManifest:
    return HarnessManifest(
        name="unit_test_bimanual",
        version="1",
        arm_groups=(
            ArmGroupSpec(name="left_arm", side="left"),
            ArmGroupSpec(name="right_arm", side="right"),
        ),
        video_streams=(VideoStreamSpec(name="top_camera", role="top", order_index=0),),
        state_streams=(
            StateStreamSpec(name="joint_position", arm_group="left_arm", dim=7, layout="joint_position"),
            StateStreamSpec(name="joint_position", arm_group="right_arm", dim=7, layout="joint_position"),
        ),
        action_streams=(
            ActionStreamSpec(
                name="joint_position",
                arm_group="left_arm",
                dim=7,
                semantics=ActionSemantics(representation="absolute", domain="joint", layout="joint_position"),
                horizon=2,
            ),
            ActionStreamSpec(
                name="joint_position",
                arm_group="right_arm",
                dim=7,
                semantics=ActionSemantics(representation="absolute", domain="joint", layout="joint_position"),
                horizon=2,
            ),
        ),
    )


class FakeBimanualPolicy:
    def __init__(self) -> None:
        self.ready_calls = 0
        self.reset_payloads: list[dict[str, object]] = []

    def assert_ready_for_benchmark(self) -> None:
        self.ready_calls += 1

    def build_manifest(self) -> HarnessManifest:
        return _manifest()

    def infer(self, observation: ObservationPacket) -> ActionPacket:
        del observation
        chunk = np.zeros((2, 7), dtype=np.float32)
        return ActionPacket(
            arms={
                "left_arm": ArmActionGroup(streams={"joint_position": ActionChunk(chunk.copy())}),
                "right_arm": ArmActionGroup(streams={"joint_position": ActionChunk(chunk.copy())}),
            }
        )

    def reset(self, reset_info: dict[str, object]) -> None:
        self.reset_payloads.append(reset_info)

    def build_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            family="unit_test_policy",
            config_name="fake",
            checkpoint_ref="none",
            checkpoint_sha256=None,
            checkpoint_sha256_explanation="fixture",
            dtype="float32",
            device="cpu",
            action_space="absolute_joint_bimanual",
            chunk_size=2,
            prompt_format_source="test",
            image_preprocess=ImagePreprocessMetadata(
                resize_resolution=None,
                resize_filter="not_applicable",
                color_space="rgb",
                output_dtype="uint8",
            ),
            runtime_family="unit_test",
            schema_source="tests/unit/test_bimanual_runner.py",
            normalization_tag=None,
        )

    def build_validation_metadata(self) -> ValidationMetadata:
        return ValidationMetadata(action_oracle="cpu_smoke")

    def build_notes(self) -> list[DecisionNote]:
        return [
            DecisionNote(
                topic="policy.test",
                choice="smoke",
                status="adapter",
                rationale="CPU smoke test fixture.",
                evidence="tests/unit/test_bimanual_runner.py",
            )
        ]

    def close(self) -> None:
        return None


class FakeBimanualEmbodiment:
    def __init__(self) -> None:
        self.executed: list[ActionPacket] = []
        self.prepare_calls = 0

    def prepare_episode(self, manifest: HarnessManifest, prompt: str) -> None:
        assert manifest.name == "unit_test_bimanual"
        assert prompt == "pick"
        self.prepare_calls += 1

    def capture_observation(
        self,
        manifest: HarnessManifest,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> ObservationPacket:
        assert manifest.name == "unit_test_bimanual"
        assert prompt == "pick"
        return ObservationPacket(
            video={"top_camera": TemporalVideoSequence(sample_indices=(0,), frames=(np.zeros((6, 6, 3), dtype=np.uint8),))},
            arms={
                "left_arm": ArmObservationGroup(
                    streams={"joint_position": TemporalStateSequence(sample_indices=(0,), values=(np.zeros(7, dtype=np.float32),))}
                ),
                "right_arm": ArmObservationGroup(
                    streams={"joint_position": TemporalStateSequence(sample_indices=(0,), values=(np.zeros(7, dtype=np.float32),))}
                ),
            },
            language={"instruction": prompt},
            session_id=session_id,
        )

    def execute_action(self, manifest: HarnessManifest, action: ActionPacket) -> None:
        assert manifest.name == "unit_test_bimanual"
        self.executed.append(action)

    def build_embodiment_metadata(self) -> EmbodimentMetadata:
        return EmbodimentMetadata(
            family="unit_test_embodiment",
            backend="fake",
            active_arm=None,
            parked_arm=None,
            parked_arm_rule="hold_static",
            control_hz=30.0,
            chunk_consumption_policy="execute_full_chunk_open_loop",
            arm_group_names=("left_arm", "right_arm"),
            camera_roles=("top",),
            camera_role_source="tests/unit/test_bimanual_runner.py",
        )

    def build_notes(self) -> list[DecisionNote]:
        return []

    def close(self) -> None:
        return None


def test_bimanual_runner_writes_fairness_log_and_executes(tmp_path):
    runner = BimanualRunner(FakeBimanualPolicy(), FakeBimanualEmbodiment())

    result = runner.run_episode(
        BimanualRunConfig(
            prompt="pick",
            max_steps=1,
            run_id="bimanual-run",
            session_id="session-1",
            output_dir=tmp_path,
        )
    )

    payload = json.loads(result.log_path.read_text(encoding="utf-8"))
    assert result.manifest_name == "unit_test_bimanual"
    assert payload["policy"]["runtime_family"] == "unit_test"
    assert payload["embodiment"]["arm_group_names"] == ["left_arm", "right_arm"]
    assert payload["runtime"]["transport"] == "internal_protocol+adapter_local_transport"
    assert payload["step_metrics"][0]["request_bytes"] == 0
    assert payload["step_metrics"][0]["response_bytes"] == 0


def test_bimanual_runner_rejects_reentrant_runs():
    runner = BimanualRunner(FakeBimanualPolicy(), FakeBimanualEmbodiment())
    runner._run_in_progress = True

    with pytest.raises(RuntimeError, match="one active rollout"):
        runner.run_episode(BimanualRunConfig(prompt="pick"))
