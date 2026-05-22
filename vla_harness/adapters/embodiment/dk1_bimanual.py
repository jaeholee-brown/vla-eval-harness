"""True-bimanual TRLC DK-1 embodiment adapter."""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Mapping
from typing import Protocol

import numpy as np

from vla_harness.adapters.embodiment._sample_types import BimanualObservationSample
from vla_harness.adapters.embodiment._sample_types import coerce_state_sequence
from vla_harness.adapters.embodiment._sample_types import coerce_video_sequence
from vla_harness.adapters.embodiment.bimanual import BimanualEmbodimentAdapter
from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.observation import ArmObservationGroup
from vla_harness.protocol.observation import ObservationPacket


DEFAULT_DK1_CAMERA_ROLE_SOURCE = "trlc-dk1/README.md + lerobot_robot_trlc_dk1/bi_follower.py"


class DK1BimanualBackend(Protocol):
    """Thin backend contract over LeRobot's `bi_dk1_follower` path."""

    def prepare_episode(self, prompt: str) -> None:
        """Prime the embodiment for a new task instruction."""

    def capture(self) -> BimanualObservationSample:
        """Return one snapshot of official DK-1 cameras and arm states."""

    def execute_chunk(
        self,
        arm_actions: Mapping[str, Mapping[str, np.ndarray]],
        *,
        padding_rules: Mapping[str, str],
        manifest: HarnessManifest,
    ) -> None:
        """Execute one action packet through the official DK-1 control path."""

    def close(self) -> None:
        """Release backend resources."""


@dataclasses.dataclass(slots=True)
class LeRobotBiDK1BackendConfig:
    left_arm_name: str = "left_arm"
    right_arm_name: str = "right_arm"
    left_prefix: str = "left_"
    right_prefix: str = "right_"
    joint_feature_order: tuple[str, ...] = (
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "gripper.pos",
    )
    camera_keys: tuple[str, ...] = ("head", "right_wrist", "left_wrist")


class LeRobotBiDK1Backend(DK1BimanualBackend):
    """Concrete backend for the official `bi_dk1_follower` robot object."""

    def __init__(self, robot: Any, config: LeRobotBiDK1BackendConfig | None = None) -> None:
        self._robot = robot
        self._config = config or LeRobotBiDK1BackendConfig()

    def prepare_episode(self, prompt: str) -> None:
        del prompt
        return None

    def capture(self) -> BimanualObservationSample:
        obs = self._robot.get_observation()
        return BimanualObservationSample(
            camera_frames={camera_key: np.asarray(obs[camera_key]) for camera_key in self._config.camera_keys if camera_key in obs},
            arm_state_streams={
                self._config.left_arm_name: {"joint_position": _read_prefixed_joint_vector(obs, self._config.left_prefix, self._config.joint_feature_order)},
                self._config.right_arm_name: {"joint_position": _read_prefixed_joint_vector(obs, self._config.right_prefix, self._config.joint_feature_order)},
            },
            metadata={"source": "official_bi_dk1_follower"},
        )

    def execute_chunk(
        self,
        arm_actions: Mapping[str, Mapping[str, np.ndarray]],
        *,
        padding_rules: Mapping[str, str],
        manifest: HarnessManifest,
    ) -> None:
        del manifest
        current = self._robot.get_observation()
        left_static = _read_prefixed_joint_vector(current, self._config.left_prefix, self._config.joint_feature_order)
        right_static = _read_prefixed_joint_vector(current, self._config.right_prefix, self._config.joint_feature_order)
        horizon = _action_horizon(arm_actions)
        for step_idx in range(horizon):
            left_action = _step_or_static(
                arm_actions,
                self._config.left_arm_name,
                "joint_position",
                step_idx,
                left_static,
                padding_rules.get(self._config.left_arm_name),
            )
            right_action = _step_or_static(
                arm_actions,
                self._config.right_arm_name,
                "joint_position",
                step_idx,
                right_static,
                padding_rules.get(self._config.right_arm_name),
            )
            self._robot.send_action(
                {
                    **_prefixed_action_dict(self._config.left_prefix, self._config.joint_feature_order, left_action),
                    **_prefixed_action_dict(self._config.right_prefix, self._config.joint_feature_order, right_action),
                }
            )

    def close(self) -> None:
        disconnect = getattr(self._robot, "disconnect", None)
        if callable(disconnect):
            disconnect()


@dataclasses.dataclass(slots=True)
class DK1BimanualConfig:
    backend_name: str = "lerobot_bi_dk1_follower"
    control_hz: float = 200.0
    chunk_consumption_policy: str = "execute_full_chunk_open_loop"
    static_padding_rule: str = "hold_static"
    arm_group_names: tuple[str, str] = ("left_arm", "right_arm")
    camera_roles: tuple[str, ...] = ("head", "right_wrist", "left_wrist")
    camera_role_source: str = DEFAULT_DK1_CAMERA_ROLE_SOURCE
    camera_key_by_role: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            "head": "head",
            "top": "head",
            "right_wrist": "right_wrist",
            "left_wrist": "left_wrist",
        }
    )
    state_aliases: Mapping[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {
            "joint_position": ("joint_position",),
            "joint_plus_gripper": ("joint_position",),
        }
    )


class DK1BimanualAdapter(BimanualEmbodimentAdapter):
    """Map the official TRLC DK-1 bimanual layout into the harness protocol."""

    def __init__(self, backend: DK1BimanualBackend, config: DK1BimanualConfig | None = None) -> None:
        self._backend = backend
        self._config = config or DK1BimanualConfig()

    def prepare_episode(self, manifest: HarnessManifest, prompt: str) -> None:
        del manifest
        self._backend.prepare_episode(prompt)

    def capture_observation(
        self,
        manifest: HarnessManifest,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> ObservationPacket:
        sample = self._backend.capture()
        video = {}
        for spec in manifest.ordered_video_streams():
            video[spec.name] = coerce_video_sequence(
                _lookup_camera_stream(sample.camera_frames, self._config.camera_key_by_role, spec.role),
                spec.sample_indices,
            )

        arms = {}
        for arm_spec in manifest.arm_groups:
            state_group: dict[str, Any] = {}
            backend_streams = sample.arm_state_streams.get(arm_spec.name, {})
            for stream_spec in manifest.state_streams_for_arm(arm_spec.name):
                state_group[stream_spec.name] = coerce_state_sequence(
                    _lookup_state_stream(backend_streams, stream_spec.name, stream_spec.layout, self._config.state_aliases),
                    stream_spec.sample_indices,
                )
            arms[arm_spec.name] = ArmObservationGroup(streams=state_group)

        observation = ObservationPacket(
            video=video,
            arms=arms,
            language={"instruction": prompt},
            session_id=session_id,
            metadata=dict(sample.metadata),
        )
        observation.validate_against(manifest)
        return observation

    def execute_action(self, manifest: HarnessManifest, action: ActionPacket) -> None:
        action.validate_against(manifest)
        arm_actions = {
            arm_name: {stream_name: chunk.values for stream_name, chunk in group.streams.items()}
            for arm_name, group in action.arms.items()
        }
        padding_rules = {arm_name: rule.strategy for arm_name, rule in action.padding.items()}
        self._backend.execute_chunk(arm_actions, padding_rules=padding_rules, manifest=manifest)

    def build_embodiment_metadata(self) -> EmbodimentMetadata:
        return EmbodimentMetadata(
            family="dk1",
            backend=self._config.backend_name,
            active_arm=None,
            parked_arm=None,
            parked_arm_rule=self._config.static_padding_rule,
            control_hz=self._config.control_hz,
            chunk_consumption_policy=self._config.chunk_consumption_policy,
            arm_group_names=self._config.arm_group_names,
            camera_roles=self._config.camera_roles,
            camera_role_source=self._config.camera_role_source,
        )

    def build_notes(self) -> list[DecisionNote]:
        return [
            DecisionNote(
                topic="embodiment.camera_roles",
                choice="head,right_wrist,left_wrist with optional top->head alias",
                status="adapter",
                rationale="The official TRLC DK-1 bimanual setup uses head/right_wrist/left_wrist cameras; top is only an adapter alias for policy compatibility.",
                evidence=self._config.camera_role_source,
            ),
            DecisionNote(
                topic="embodiment.control_hz",
                choice=str(self._config.control_hz),
                status="official",
                rationale="The published bimanual teleoperation example runs at 200 Hz and is the closest released control-rate default.",
                evidence="trlc-dk1/examples/bi_teleop.py",
            ),
            DecisionNote(
                topic="embodiment.static_padding_rule",
                choice=self._config.static_padding_rule,
                status="benchmark_default",
                rationale="If a single-arm policy is bridged into a bimanual setup, the uncontrolled arm must stay explicitly static.",
                evidence="User-scoped harness policy for bimanual-only evaluation.",
            ),
        ]

    def close(self) -> None:
        self._backend.close()


def _lookup_camera_stream(
    frames: Mapping[str, Any],
    camera_key_by_role: Mapping[str, str],
    role: str,
) -> Any:
    if role not in camera_key_by_role:
        raise KeyError(f"Unknown DK-1 camera role {role!r}; extend camera_key_by_role to support it.")
    camera_key = camera_key_by_role[role]
    if camera_key not in frames:
        raise KeyError(f"DK-1 backend did not provide required camera key {camera_key!r} for role {role!r}.")
    return frames[camera_key]


def _lookup_state_stream(
    backend_streams: Mapping[str, Any],
    stream_name: str,
    layout: str,
    state_aliases: Mapping[str, tuple[str, ...]],
) -> Any:
    candidates = (stream_name,) + tuple(state_aliases.get(layout, ()))
    for candidate in candidates:
        if candidate in backend_streams:
            return backend_streams[candidate]
    raise KeyError(f"DK-1 backend did not provide a state stream for {stream_name!r}/{layout!r}.")


def _read_prefixed_joint_vector(
    observation: Mapping[str, Any],
    prefix: str,
    feature_order: tuple[str, ...],
) -> np.ndarray:
    return np.asarray([observation[f"{prefix}{feature_name}"] for feature_name in feature_order], dtype=np.float32)


def _prefixed_action_dict(prefix: str, feature_order: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {f"{prefix}{feature_name}": float(values[index]) for index, feature_name in enumerate(feature_order)}


def _action_horizon(arm_actions: Mapping[str, Mapping[str, np.ndarray]]) -> int:
    horizon = 0
    for streams in arm_actions.values():
        for values in streams.values():
            horizon = max(horizon, int(np.asarray(values).shape[0]))
    if horizon <= 0:
        raise ValueError("DK-1 backend received no executable action streams.")
    return horizon


def _step_or_static(
    arm_actions: Mapping[str, Mapping[str, np.ndarray]],
    arm_name: str,
    stream_name: str,
    step_idx: int,
    static_value: np.ndarray,
    padding_rule: str | None,
) -> np.ndarray:
    stream_values = arm_actions.get(arm_name, {}).get(stream_name)
    if stream_values is not None:
        return np.asarray(stream_values[step_idx], dtype=np.float32)
    if padding_rule not in {None, "hold_static"}:
        raise ValueError(f"Unsupported DK-1 padding rule {padding_rule!r}.")
    return np.asarray(static_value, dtype=np.float32)
