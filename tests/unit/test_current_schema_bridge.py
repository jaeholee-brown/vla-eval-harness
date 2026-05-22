from __future__ import annotations

import numpy as np

from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata
from vla_harness.logging.decision_log import ImagePreprocessMetadata
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata
from vla_harness.protocol.current_schema_bridge import CurrentSchemaBridgeConfig
from vla_harness.protocol.current_schema_bridge import LegacyCurrentSchemaEmbodimentBridge
from vla_harness.protocol.current_schema_bridge import LegacyCurrentSchemaPolicyBridge
from vla_harness.protocol.current_schema_bridge import action_packet_from_current_schema
from vla_harness.protocol.current_schema_bridge import action_packet_to_current_schema
from vla_harness.protocol.current_schema_bridge import build_current_schema_bridge_manifest
from vla_harness.protocol.current_schema_bridge import observation_from_current_schema
from vla_harness.protocol.current_schema_bridge import observation_to_current_schema


def _server_metadata() -> dict[str, object]:
    return {
        "needs_wrist_camera": True,
        "n_external_cameras": 1,
        "needs_stereo_camera": False,
        "needs_session_id": True,
        "action_space": "joint_velocity",
    }


def _flat_observation() -> dict[str, object]:
    return {
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_1_left": np.ones((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "pick",
        "session_id": "session-1",
    }


class FakeCurrentPolicy:
    def __init__(self) -> None:
        self.seen_observations: list[dict[str, object]] = []

    def assert_ready_for_benchmark(self) -> None:
        return None

    def get_server_metadata(self) -> dict[str, object]:
        return _server_metadata()

    def infer(self, obs: dict[str, object]) -> dict[str, object]:
        self.seen_observations.append(obs)
        return {"actions": np.full((2, 8), 0.5, dtype=np.float32)}

    def reset(self, reset_info: dict[str, object]) -> dict[str, object]:
        return {"reset": reset_info}

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
        return None


class FakeCurrentEmbodiment:
    def __init__(self) -> None:
        self.executed: list[tuple[np.ndarray, str]] = []
        self.prepared: list[tuple[dict[str, object], str]] = []

    def prepare_episode(self, server_metadata: dict[str, object], prompt: str) -> None:
        self.prepared.append((server_metadata, prompt))

    def build_observation(
        self,
        server_metadata: dict[str, object],
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, object]:
        del server_metadata, prompt
        obs = _flat_observation()
        obs["session_id"] = session_id
        return obs

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


def test_current_schema_bridge_manifest_is_bimanual():
    manifest = build_current_schema_bridge_manifest(_server_metadata(), CurrentSchemaBridgeConfig(active_arm="right"))

    assert [arm.name for arm in manifest.arm_groups] == ["right_arm", "left_arm"]
    assert manifest.arm_group("right_arm").control_role == "policy_controlled"
    assert manifest.arm_group("left_arm").control_role == "static_pad_only"
    assert [spec.name for spec in manifest.video_streams] == ["right_arm.wrist_left", "global.exterior_1_left"]


def test_observation_bridge_round_trips_flat_schema():
    config = CurrentSchemaBridgeConfig(active_arm="right")
    server_metadata = _server_metadata()
    packet = observation_from_current_schema(_flat_observation(), server_metadata, config)
    restored = observation_to_current_schema(packet, server_metadata, config)

    assert set(packet.arms) == {"right_arm", "left_arm"}
    assert packet.arms["left_arm"].streams == {}
    assert restored["prompt"] == "pick"
    assert restored["session_id"] == "session-1"
    assert restored["observation/joint_position"].shape == (7,)
    assert restored["observation/wrist_image_left"].shape == (224, 224, 3)


def test_action_bridge_adds_static_padding_rule():
    config = CurrentSchemaBridgeConfig(active_arm="right")
    server_metadata = _server_metadata()
    packet = action_packet_from_current_schema(
        {"actions": np.zeros((2, 8), dtype=np.float32)},
        server_metadata,
        config,
    )
    restored = action_packet_to_current_schema(packet, server_metadata, config)

    assert "left_arm" in packet.padding
    assert packet.padding["left_arm"].strategy == "hold_static"
    assert restored["actions"].shape == (2, 8)


def test_legacy_policy_bridge_projects_through_bimanual_packet():
    bridge = LegacyCurrentSchemaPolicyBridge(FakeCurrentPolicy(), CurrentSchemaBridgeConfig(active_arm="right"))
    server_metadata = _server_metadata()
    observation = observation_from_current_schema(_flat_observation(), server_metadata, bridge.bridge_config)

    action_packet = bridge.infer(observation)

    assert "right_arm" in action_packet.arms
    assert "left_arm" in action_packet.padding
    assert action_packet.arms["right_arm"].streams["policy_action"].values.shape == (2, 8)
    assert any(note.topic == "policy.phase3_bridge" for note in bridge.build_notes())


def test_legacy_embodiment_bridge_projects_capture_and_execute():
    legacy = FakeCurrentEmbodiment()
    bridge = LegacyCurrentSchemaEmbodimentBridge(legacy, CurrentSchemaBridgeConfig(active_arm="right"))
    manifest = build_current_schema_bridge_manifest(_server_metadata(), bridge.bridge_config)

    bridge.prepare_episode(manifest, "pick")
    observation = bridge.capture_observation(manifest, "pick", session_id="abc")
    bridge.execute_action(
        manifest,
        action_packet_from_current_schema({"actions": np.zeros((1, 8), dtype=np.float32)}, _server_metadata(), bridge.bridge_config),
    )

    assert legacy.prepared[0][1] == "pick"
    assert observation.session_id == "abc"
    assert legacy.executed[0][1] == "joint_velocity"
    assert any(note.topic == "embodiment.phase3_bridge" for note in bridge.build_notes())
