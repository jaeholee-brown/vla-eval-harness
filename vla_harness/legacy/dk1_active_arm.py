"""Legacy current-schema DK-1 adapter scoped to one active arm on a bimanual rig.

This adapter is retained as the historical bootstrap used to close Phase 1.
Future DK-1 work should target the bimanual internal representation directly.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Mapping
from typing import Protocol

import numpy as np

from vla_harness.legacy.embodiment_protocol import CurrentSchemaEmbodimentAdapter
from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata


@dataclasses.dataclass(slots=True)
class DK1Observation:
    wrist_image_left: np.ndarray
    exterior_image_lefts: list[np.ndarray]
    joint_position: np.ndarray
    cartesian_position: np.ndarray
    gripper_position: np.ndarray
    wrist_image_right: np.ndarray | None = None
    exterior_image_rights: list[np.ndarray] = dataclasses.field(default_factory=list)


class DK1Backend(Protocol):
    def park_arm(self, arm: str) -> None:
        """Place the non-active arm into a robot-native hold or park mode."""

    def capture_active_arm(self, arm: str) -> DK1Observation:
        """Capture one observation frame for the active arm."""

    def execute_actions(self, arm: str, actions: np.ndarray, *, action_space: str) -> None:
        """Execute one action chunk on the active arm."""


@dataclasses.dataclass(slots=True)
class DK1ActiveArmConfig:
    backend_name: str = "trlc_dk1"
    active_arm: str = "right"
    parked_arm: str | None = None
    parked_arm_rule: str = "robot_native_hold"
    control_hz: float = 15.0
    chunk_consumption_policy: str = "execute_full_chunk_open_loop"

    def resolved_parked_arm(self) -> str:
        if self.parked_arm is not None:
            return self.parked_arm
        return "left" if self.active_arm == "right" else "right"


class DK1ActiveArmAdapter(CurrentSchemaEmbodimentAdapter):
    """Maps one active DK-1 arm into RoboArena's historical flat schema."""

    def __init__(self, backend: DK1Backend, config: DK1ActiveArmConfig | None = None) -> None:
        self._backend = backend
        self._config = config or DK1ActiveArmConfig()
        if self._config.active_arm not in {"left", "right"}:
            raise ValueError("active_arm must be 'left' or 'right'")

    def prepare_episode(self, server_metadata: Mapping[str, Any], prompt: str) -> None:
        del server_metadata, prompt
        self._backend.park_arm(self._config.resolved_parked_arm())

    def build_observation(
        self,
        server_metadata: Mapping[str, Any],
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        sample = self._backend.capture_active_arm(self._config.active_arm)
        obs: dict[str, Any] = {
            "observation/joint_position": sample.joint_position,
            "observation/cartesian_position": sample.cartesian_position,
            "observation/gripper_position": sample.gripper_position,
            "prompt": prompt,
        }

        if bool(server_metadata.get("needs_wrist_camera", False)):
            obs["observation/wrist_image_left"] = sample.wrist_image_left
            if bool(server_metadata.get("needs_stereo_camera", False)):
                if sample.wrist_image_right is None:
                    raise ValueError("Stereo wrist image requested but unavailable from DK-1 backend.")
                obs["observation/wrist_image_right"] = sample.wrist_image_right

        n_external_cameras = int(server_metadata.get("n_external_cameras", 0))
        if n_external_cameras > len(sample.exterior_image_lefts):
            raise ValueError(
                f"Requested {n_external_cameras} exterior cameras but backend only returned "
                f"{len(sample.exterior_image_lefts)} left images."
            )
        for index in range(n_external_cameras):
            obs[f"observation/exterior_image_{index + 1}_left"] = sample.exterior_image_lefts[index]
            if bool(server_metadata.get("needs_stereo_camera", False)):
                if index >= len(sample.exterior_image_rights):
                    raise ValueError("Stereo exterior image requested but unavailable from DK-1 backend.")
                obs[f"observation/exterior_image_{index + 1}_right"] = sample.exterior_image_rights[index]

        if bool(server_metadata.get("needs_session_id", False)):
            if session_id is None:
                raise ValueError("Server metadata requires a session_id, but none was provided.")
            obs["session_id"] = session_id

        return obs

    def execute_action_chunk(self, actions: np.ndarray, *, action_space: str) -> None:
        self._backend.execute_actions(self._config.active_arm, actions, action_space=action_space)

    def build_embodiment_metadata(self) -> EmbodimentMetadata:
        return EmbodimentMetadata(
            family="dk1",
            backend=self._config.backend_name,
            active_arm=self._config.active_arm,
            parked_arm=self._config.resolved_parked_arm(),
            parked_arm_rule=self._config.parked_arm_rule,
            control_hz=self._config.control_hz,
            chunk_consumption_policy=self._config.chunk_consumption_policy,
        )

    def build_notes(self) -> list[DecisionNote]:
        return [
            DecisionNote(
                topic="embodiment.phase1_scope",
                choice="single_active_arm_only",
                status="scoped_out",
                rationale="Phase 1 uses one active arm on a bimanual DK-1 rig and parks the other arm outside the protocol.",
                evidence="docs/spikes/current-schema-gap-matrix.md",
            ),
            DecisionNote(
                topic="embodiment.parked_arm_rule",
                choice=self._config.parked_arm_rule,
                status="adapter",
                rationale="The parked arm is managed entirely by the embodiment layer in phase 1.",
                evidence="vla_harness/legacy/dk1_active_arm.py",
            ),
        ]
