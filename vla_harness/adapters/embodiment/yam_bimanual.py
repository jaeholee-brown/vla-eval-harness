"""True-bimanual YAM embodiment adapter backed by official camera/state defaults."""

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


DEFAULT_YAM_CAMERA_ROLE_SOURCE = "gello_software/configs/yam_left.yaml + gello_software/configs/yam_right.yaml"


class YAMBimanualBackend(Protocol):
    """Thin backend contract over the official YAM stack."""

    def prepare_episode(self, prompt: str) -> None:
        """Prime the embodiment for a new task instruction."""

    def capture(self) -> BimanualObservationSample:
        """Return one snapshot of official YAM cameras and arm states."""

    def execute_chunk(
        self,
        arm_actions: Mapping[str, Mapping[str, np.ndarray]],
        *,
        padding_rules: Mapping[str, str],
        manifest: HarnessManifest,
    ) -> None:
        """Execute one action packet through the official YAM control path."""

    def close(self) -> None:
        """Release backend resources."""


class YAMRobotEnvLike(Protocol):
    """Subset of the official YAM `RobotEnv` API used by the harness."""

    def get_obs(self) -> Mapping[str, np.ndarray]:
        """Return the current observation dict."""

    def step(self, joint_positions: np.ndarray) -> Any:
        """Execute one 14-D bimanual joint target."""


@dataclasses.dataclass(slots=True)
class YAMRobotEnvBackendConfig:
    left_arm_name: str = "left_arm"
    right_arm_name: str = "right_arm"
    left_state_slice: slice = dataclasses.field(default_factory=lambda: slice(0, 7))
    right_state_slice: slice = dataclasses.field(default_factory=lambda: slice(7, 14))
    camera_key_by_backend_name: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            "front_camera": "front_camera_rgb",
            "left_camera": "left_camera_rgb",
            "right_camera": "right_camera_rgb",
        }
    )


class YAMRobotEnvBackend(YAMBimanualBackend):
    """Concrete backend for the official YAM `RobotEnv` control loop."""

    def __init__(self, env: YAMRobotEnvLike, config: YAMRobotEnvBackendConfig | None = None) -> None:
        self._env = env
        self._config = config or YAMRobotEnvBackendConfig()

    def prepare_episode(self, prompt: str) -> None:
        del prompt
        return None

    def capture(self) -> BimanualObservationSample:
        obs = self._env.get_obs()
        joints = np.asarray(obs["joint_positions"], dtype=np.float32)
        return BimanualObservationSample(
            camera_frames={
                backend_name: np.asarray(obs[obs_key])
                for backend_name, obs_key in self._config.camera_key_by_backend_name.items()
            },
            arm_state_streams={
                self._config.left_arm_name: {"joint_position": joints[self._config.left_state_slice]},
                self._config.right_arm_name: {"joint_position": joints[self._config.right_state_slice]},
            },
            metadata={"source": "official_yam_robot_env"},
        )

    def execute_chunk(
        self,
        arm_actions: Mapping[str, Mapping[str, np.ndarray]],
        *,
        padding_rules: Mapping[str, str],
        manifest: HarnessManifest,
    ) -> None:
        del manifest
        current = np.asarray(self._env.get_obs()["joint_positions"], dtype=np.float32)
        left_static = current[self._config.left_state_slice]
        right_static = current[self._config.right_state_slice]
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
            self._env.step(np.concatenate([left_action, right_action]).astype(np.float32))

    def close(self) -> None:
        return None


@dataclasses.dataclass(slots=True)
class YAMBimanualConfig:
    backend_name: str = "yam_gello"
    control_hz: float = 30.0
    chunk_consumption_policy: str = "execute_full_chunk_open_loop"
    static_padding_rule: str = "hold_static"
    arm_group_names: tuple[str, str] = ("left_arm", "right_arm")
    camera_roles: tuple[str, ...] = ("top", "left", "right")
    camera_role_source: str = DEFAULT_YAM_CAMERA_ROLE_SOURCE
    camera_key_by_role: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            "top": "front_camera",
            "front": "front_camera",
            "left": "left_camera",
            "right": "right_camera",
        }
    )
    state_aliases: Mapping[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {
            "joint_position": ("joint_position",),
            "joint_plus_gripper": ("joint_position",),
        }
    )


class YAMBimanualAdapter(BimanualEmbodimentAdapter):
    """Map the official YAM camera/state layout into the bimanual harness protocol."""

    def __init__(self, backend: YAMBimanualBackend, config: YAMBimanualConfig | None = None) -> None:
        self._backend = backend
        self._config = config or YAMBimanualConfig()

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
            family="yam",
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
                choice="top->front_camera,left->left_camera,right->right_camera",
                status="adapter",
                rationale="The official YAM release defines left/front/right cameras; the harness exposes top as the front view used by MolmoAct2.",
                evidence=self._config.camera_role_source,
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
        raise KeyError(f"Unknown YAM camera role {role!r}; extend camera_key_by_role to support it.")
    camera_key = camera_key_by_role[role]
    if camera_key not in frames:
        raise KeyError(f"YAM backend did not provide required camera key {camera_key!r} for role {role!r}.")
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
    raise KeyError(f"YAM backend did not provide a state stream for {stream_name!r}/{layout!r}.")


def _action_horizon(arm_actions: Mapping[str, Mapping[str, np.ndarray]]) -> int:
    horizon = 0
    for streams in arm_actions.values():
        for values in streams.values():
            horizon = max(horizon, int(np.asarray(values).shape[0]))
    if horizon <= 0:
        raise ValueError("YAM backend received no executable action streams.")
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
        raise ValueError(f"Unsupported YAM padding rule {padding_rule!r}.")
    return np.asarray(static_value, dtype=np.float32)
