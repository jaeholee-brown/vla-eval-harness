from __future__ import annotations

import numpy as np

from vla_harness.adapters.embodiment import BimanualObservationSample
from vla_harness.adapters.embodiment.yam_bimanual import YAMBimanualAdapter
from vla_harness.adapters.embodiment.yam_bimanual import YAMRobotEnvBackend
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
        name="yam_fixture",
        version="1",
        arm_groups=(
            ArmGroupSpec(name="left_arm", side="left"),
            ArmGroupSpec(name="right_arm", side="right"),
        ),
        video_streams=(
            VideoStreamSpec(name="top_camera", role="top", order_index=0),
            VideoStreamSpec(name="left_camera", role="left", order_index=1),
            VideoStreamSpec(name="right_camera", role="right", order_index=2),
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


class FakeYAMBackend:
    def __init__(self) -> None:
        self.executed: list[tuple[dict[str, dict[str, np.ndarray]], dict[str, str], str]] = []
        self.prompts: list[str] = []

    def prepare_episode(self, prompt: str) -> None:
        self.prompts.append(prompt)

    def capture(self) -> BimanualObservationSample:
        return BimanualObservationSample(
            camera_frames={
                "front_camera": np.zeros((8, 8, 3), dtype=np.uint8),
                "left_camera": np.ones((8, 8, 3), dtype=np.uint8),
                "right_camera": np.full((8, 8, 3), 2, dtype=np.uint8),
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


def test_yam_bimanual_adapter_smoke():
    backend = FakeYAMBackend()
    adapter = YAMBimanualAdapter(backend)
    manifest = _manifest()

    adapter.prepare_episode(manifest, "sort candy")
    observation = adapter.capture_observation(manifest, "sort candy", session_id="session-1")
    action = ActionPacket(
        arms={
            "left_arm": ArmActionGroup(streams={"joint_position": ActionChunk(np.zeros((2, 7), dtype=np.float32))}),
            "right_arm": ArmActionGroup(streams={"joint_position": ActionChunk(np.ones((2, 7), dtype=np.float32))}),
        },
    )
    adapter.execute_action(manifest, action)
    metadata = adapter.build_embodiment_metadata()

    assert backend.prompts == ["sort candy"]
    assert observation.video["top_camera"].frames[0].shape == (8, 8, 3)
    assert observation.arms["left_arm"].streams["joint_position"].dim == 7
    assert backend.executed[0][0]["right_arm"]["joint_position"].shape == (2, 7)
    assert metadata.camera_roles == ("top", "left", "right")
    assert metadata.arm_group_names == ("left_arm", "right_arm")


class FakeYAMEnv:
    def __init__(self) -> None:
        self.steps: list[np.ndarray] = []

    def get_obs(self):
        return {
            "joint_positions": np.arange(14, dtype=np.float32),
            "left_camera_rgb": np.ones((6, 6, 3), dtype=np.uint8),
            "front_camera_rgb": np.zeros((6, 6, 3), dtype=np.uint8),
            "right_camera_rgb": np.full((6, 6, 3), 2, dtype=np.uint8),
        }

    def step(self, joint_positions: np.ndarray):
        self.steps.append(np.asarray(joint_positions))
        return {}


def test_yam_robot_env_backend_maps_official_obs_and_step_shape():
    backend = YAMRobotEnvBackend(FakeYAMEnv())
    sample = backend.capture()
    backend.execute_chunk(
        {
            "left_arm": {"joint_position": np.zeros((2, 7), dtype=np.float32)},
            "right_arm": {"joint_position": np.ones((2, 7), dtype=np.float32)},
        },
        padding_rules={},
        manifest=_manifest(),
    )

    assert sample.camera_frames["front_camera"].shape == (6, 6, 3)
    assert sample.arm_state_streams["left_arm"]["joint_position"].shape == (7,)
