from __future__ import annotations

import numpy as np

from vla_harness.adapters.embodiment import BimanualObservationSample
from vla_harness.adapters.embodiment.dk1_bimanual import DK1BimanualAdapter
from vla_harness.adapters.embodiment.dk1_bimanual import LeRobotBiDK1Backend
from vla_harness.protocol import ActionChunk
from vla_harness.protocol import ActionPacket
from vla_harness.protocol import ActionSemantics
from vla_harness.protocol import ActionStreamSpec
from vla_harness.protocol import ArmActionGroup
from vla_harness.protocol import ArmGroupSpec
from vla_harness.protocol import HarnessManifest
from vla_harness.protocol import StateStreamSpec
from vla_harness.protocol import VideoStreamSpec


def _manifest() -> HarnessManifest:
    return HarnessManifest(
        name="dk1_fixture",
        version="1",
        arm_groups=(
            ArmGroupSpec(name="left_arm", side="left"),
            ArmGroupSpec(name="right_arm", side="right"),
        ),
        video_streams=(
            VideoStreamSpec(name="top_camera", role="top", order_index=0),
            VideoStreamSpec(name="left_wrist_camera", role="left_wrist", order_index=1, arm_group="left_arm"),
            VideoStreamSpec(name="right_wrist_camera", role="right_wrist", order_index=2, arm_group="right_arm"),
        ),
        state_streams=(
            StateStreamSpec(name="joint_position", arm_group="left_arm", dim=7, layout="joint_plus_gripper"),
            StateStreamSpec(name="joint_position", arm_group="right_arm", dim=7, layout="joint_plus_gripper"),
        ),
        action_streams=(
            ActionStreamSpec(
                name="joint_position",
                arm_group="left_arm",
                dim=7,
                semantics=ActionSemantics(representation="absolute", domain="joint", layout="joint_plus_gripper"),
            ),
            ActionStreamSpec(
                name="joint_position",
                arm_group="right_arm",
                dim=7,
                semantics=ActionSemantics(representation="absolute", domain="joint", layout="joint_plus_gripper"),
            ),
        ),
    )


class FakeDK1Backend:
    def __init__(self) -> None:
        self.executed: list[tuple[dict[str, dict[str, np.ndarray]], dict[str, str], str]] = []

    def prepare_episode(self, prompt: str) -> None:
        self.prompt = prompt

    def capture(self) -> BimanualObservationSample:
        return BimanualObservationSample(
            camera_frames={
                "head": np.zeros((10, 10, 3), dtype=np.uint8),
                "left_wrist": np.ones((10, 10, 3), dtype=np.uint8),
                "right_wrist": np.full((10, 10, 3), 2, dtype=np.uint8),
            },
            arm_state_streams={
                "left_arm": {"joint_position": np.arange(7, dtype=np.float32)},
                "right_arm": {"joint_position": np.arange(7, 14, dtype=np.float32)},
            },
        )

    def execute_chunk(self, arm_actions, *, padding_rules, manifest) -> None:
        self.executed.append((dict(arm_actions), dict(padding_rules), manifest.name))

    def close(self) -> None:
        return None


def test_dk1_bimanual_adapter_supports_top_alias_and_shapes():
    backend = FakeDK1Backend()
    adapter = DK1BimanualAdapter(backend)
    manifest = _manifest()

    observation = adapter.capture_observation(manifest, "pick up block", session_id="s1")
    action = ActionPacket(
        arms={
            "left_arm": ArmActionGroup(streams={"joint_position": ActionChunk(np.zeros((1, 7), dtype=np.float32))}),
            "right_arm": ArmActionGroup(streams={"joint_position": ActionChunk(np.ones((1, 7), dtype=np.float32))}),
        }
    )
    adapter.execute_action(manifest, action)
    metadata = adapter.build_embodiment_metadata()

    assert observation.video["top_camera"].frames[0].shape == (10, 10, 3)
    assert observation.video["left_wrist_camera"].frames[0].shape == (10, 10, 3)
    assert backend.executed[0][0]["left_arm"]["joint_position"].shape == (1, 7)
    assert metadata.camera_roles == ("head", "right_wrist", "left_wrist")
    assert metadata.control_hz == 200.0


class FakeBiDK1Robot:
    def __init__(self) -> None:
        self.sent_actions: list[dict[str, float]] = []

    def get_observation(self):
        obs = {
            "head": np.zeros((6, 6, 3), dtype=np.uint8),
            "left_wrist": np.ones((6, 6, 3), dtype=np.uint8),
            "right_wrist": np.full((6, 6, 3), 2, dtype=np.uint8),
        }
        for prefix, start in (("left_", 0.0), ("right_", 10.0)):
            for index, key in enumerate(("joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos", "joint_5.pos", "joint_6.pos", "gripper.pos")):
                obs[f"{prefix}{key}"] = start + index
        return obs

    def send_action(self, action: dict[str, float]):
        self.sent_actions.append(dict(action))
        return action

    def disconnect(self):
        return None


def test_lerobot_bi_dk1_backend_maps_prefixed_keys():
    backend = LeRobotBiDK1Backend(FakeBiDK1Robot())
    sample = backend.capture()
    backend.execute_chunk(
        {
            "left_arm": {"joint_position": np.zeros((1, 7), dtype=np.float32)},
            "right_arm": {"joint_position": np.ones((1, 7), dtype=np.float32)},
        },
        padding_rules={},
        manifest=_manifest(),
    )

    assert sample.arm_state_streams["left_arm"]["joint_position"].shape == (7,)
    assert sample.camera_frames["head"].shape == (6, 6, 3)
