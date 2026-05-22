from __future__ import annotations

import numpy as np

from vla_harness.adapters.embodiment.dk1_active_arm import DK1ActiveArmAdapter
from vla_harness.adapters.embodiment.dk1_active_arm import DK1ActiveArmConfig
from vla_harness.adapters.embodiment.dk1_active_arm import DK1Observation


class FakeDK1Backend:
    def __init__(self) -> None:
        self.parked_arms: list[str] = []
        self.executed: list[tuple[str, np.ndarray, str]] = []

    def park_arm(self, arm: str) -> None:
        self.parked_arms.append(arm)

    def capture_active_arm(self, arm: str) -> DK1Observation:
        del arm
        return DK1Observation(
            wrist_image_left=np.zeros((224, 224, 3), dtype=np.uint8),
            exterior_image_lefts=[np.ones((224, 224, 3), dtype=np.uint8)],
            joint_position=np.zeros(7, dtype=np.float32),
            cartesian_position=np.zeros(6, dtype=np.float32),
            gripper_position=np.zeros(1, dtype=np.float32),
        )

    def execute_actions(self, arm: str, actions: np.ndarray, *, action_space: str) -> None:
        self.executed.append((arm, actions, action_space))


def test_dk1_active_arm_maps_current_schema_fields():
    backend = FakeDK1Backend()
    adapter = DK1ActiveArmAdapter(backend, DK1ActiveArmConfig(active_arm="right"))
    metadata = {
        "needs_wrist_camera": True,
        "n_external_cameras": 1,
        "needs_stereo_camera": False,
        "needs_session_id": True,
    }

    adapter.prepare_episode(metadata, "pick")
    obs = adapter.build_observation(metadata, "pick", session_id="session-123")

    assert backend.parked_arms == ["left"]
    assert obs["prompt"] == "pick"
    assert obs["session_id"] == "session-123"
    assert obs["observation/joint_position"].shape == (7,)
    assert obs["observation/cartesian_position"].shape == (6,)
    assert obs["observation/gripper_position"].shape == (1,)
    assert obs["observation/wrist_image_left"].shape == (224, 224, 3)
    assert obs["observation/exterior_image_1_left"].shape == (224, 224, 3)


def test_dk1_active_arm_executes_only_on_configured_arm():
    backend = FakeDK1Backend()
    adapter = DK1ActiveArmAdapter(backend, DK1ActiveArmConfig(active_arm="left"))
    actions = np.zeros((2, 8), dtype=np.float32)

    adapter.execute_action_chunk(actions, action_space="joint_velocity")

    assert len(backend.executed) == 1
    arm, logged_actions, action_space = backend.executed[0]
    assert arm == "left"
    assert action_space == "joint_velocity"
    assert logged_actions.shape == (2, 8)
