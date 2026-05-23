"""Bridge the quarantined flat schema into the bimanual-first representation.

This bridge exists only so the legacy current-schema adapters can still be
exercised through a ``BimanualRunner`` for parity work. New adapters should
target the bimanual surface directly.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Literal
from typing import Mapping

import numpy as np

from vla_harness.adapters.embodiment.bimanual import BimanualEmbodimentAdapter
from vla_harness.adapters.policy.bimanual import BimanualPolicyAdapter
from vla_harness.legacy.embodiment_protocol import CurrentSchemaEmbodimentAdapter
from vla_harness.legacy.policy_protocol import CurrentSchemaPolicyAdapter
from vla_harness.logging.decision_log import DecisionNote
from vla_harness.protocol.action import ActionChunk
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.action import ArmActionGroup
from vla_harness.protocol.action import PaddingRule
from vla_harness.protocol.manifest import ActionDomain
from vla_harness.protocol.manifest import ActionRepresentation
from vla_harness.protocol.manifest import ActionSemantics
from vla_harness.protocol.manifest import ActionStreamSpec
from vla_harness.protocol.manifest import ArmGroupSpec
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.manifest import LanguageFieldSpec
from vla_harness.protocol.manifest import StateStreamSpec
from vla_harness.protocol.manifest import VideoStreamSpec
from vla_harness.protocol.observation import ArmObservationGroup
from vla_harness.protocol.observation import ObservationPacket
from vla_harness.protocol.observation import TemporalStateSequence
from vla_harness.protocol.observation import TemporalVideoSequence


FlatActionSpace = Literal[
    "joint_position",
    "joint_velocity",
    "cartesian_position",
    "cartesian_velocity",
]


@dataclasses.dataclass(slots=True, frozen=True)
class CurrentSchemaBridgeConfig:
    """How to embed the historical flat schema inside the bimanual representation."""

    active_arm: Literal["left", "right"] = "right"
    inactive_arm_padding: Literal["hold_static", "zero_velocity", "repeat_last"] = "hold_static"
    instruction_field: str = "instruction"

    def active_arm_group(self) -> str:
        return f"{self.active_arm}_arm"

    def inactive_arm(self) -> Literal["left", "right"]:
        return "left" if self.active_arm == "right" else "right"

    def inactive_arm_group(self) -> str:
        return f"{self.inactive_arm()}_arm"


def _action_semantics_for_space(action_space: FlatActionSpace) -> ActionSemantics:
    mapping: dict[FlatActionSpace, tuple[ActionRepresentation, ActionDomain, str]] = {
        "joint_position": ("absolute", "joint", "joint_plus_gripper"),
        "joint_velocity": ("velocity", "joint", "joint_plus_gripper"),
        "cartesian_position": ("absolute", "cartesian", "cartesian_plus_gripper"),
        "cartesian_velocity": ("velocity", "cartesian", "cartesian_plus_gripper"),
    }
    representation, domain, layout = mapping[action_space]
    return ActionSemantics(
        representation=representation,
        domain=domain,
        layout=layout,
        static_pad_strategy="hold_static" if representation == "absolute" else "zero_velocity",
    )


def _action_dim_for_space(action_space: FlatActionSpace) -> int:
    if action_space in {"joint_position", "joint_velocity"}:
        return 8
    return 7


def _legacy_server_metadata(manifest: HarnessManifest) -> Mapping[str, Any]:
    raw = manifest.metadata.get("legacy_server_metadata")
    if not isinstance(raw, Mapping):
        raise ValueError("Manifest does not carry legacy_server_metadata.")
    return raw


def build_current_schema_bridge_manifest(
    server_metadata: Mapping[str, Any],
    config: CurrentSchemaBridgeConfig,
) -> HarnessManifest:
    """Describe the old current-schema path as a bimanual manifest."""

    action_space = str(server_metadata["action_space"])
    if action_space not in {
        "joint_position",
        "joint_velocity",
        "cartesian_position",
        "cartesian_velocity",
    }:
        raise ValueError(f"Unsupported current-schema action_space: {action_space!r}")

    active_arm = config.active_arm_group()
    inactive_arm = config.inactive_arm_group()

    video_streams: list[VideoStreamSpec] = []
    order_index = 0
    if bool(server_metadata.get("needs_wrist_camera", False)):
        video_streams.append(
            VideoStreamSpec(
                name=f"{active_arm}.wrist_left",
                role="wrist_rgb",
                arm_group=active_arm,
                order_index=order_index,
            )
        )
        order_index += 1
        if bool(server_metadata.get("needs_stereo_camera", False)):
            video_streams.append(
                VideoStreamSpec(
                    name=f"{active_arm}.wrist_right",
                    role="wrist_rgb_stereo",
                    arm_group=active_arm,
                    order_index=order_index,
                )
            )
            order_index += 1

    for camera_index in range(int(server_metadata.get("n_external_cameras", 0))):
        video_streams.append(
            VideoStreamSpec(
                name=f"global.exterior_{camera_index + 1}_left",
                role="exterior_rgb",
                order_index=order_index,
            )
        )
        order_index += 1
        if bool(server_metadata.get("needs_stereo_camera", False)):
            video_streams.append(
                VideoStreamSpec(
                    name=f"global.exterior_{camera_index + 1}_right",
                    role="exterior_rgb_stereo",
                    order_index=order_index,
                )
            )
            order_index += 1

    manifest = HarnessManifest(
        name="legacy_current_schema_bridge",
        version="phase3-bootstrap",
        arm_groups=(
            ArmGroupSpec(
                name=active_arm,
                side=config.active_arm,
                control_role="policy_controlled",
                description="Historical flat-schema policy-controlled arm.",
            ),
            ArmGroupSpec(
                name=inactive_arm,
                side=config.inactive_arm(),
                control_role="static_pad_only",
                description="Historical flat-schema inactive arm kept static by explicit padding rule.",
            ),
        ),
        video_streams=tuple(video_streams),
        state_streams=(
            StateStreamSpec(
                name="joint_position",
                arm_group=active_arm,
                dim=7,
                layout="joint_position",
                origin="sensor",
            ),
            StateStreamSpec(
                name="cartesian_position",
                arm_group=active_arm,
                dim=6,
                layout="cartesian_position",
                origin="sensor",
            ),
            StateStreamSpec(
                name="gripper_position",
                arm_group=active_arm,
                dim=1,
                layout="gripper_position",
                origin="sensor",
            ),
        ),
        action_streams=(
            ActionStreamSpec(
                name="policy_action",
                arm_group=active_arm,
                dim=_action_dim_for_space(action_space),  # benchmark-derived from the flat action enum
                semantics=_action_semantics_for_space(action_space),
            ),
        ),
        language_fields=(LanguageFieldSpec(name=config.instruction_field),),
        metadata={
            "legacy_bridge": "current_flat_schema",
            "legacy_server_metadata": dict(server_metadata),
            "active_arm_group": active_arm,
            "inactive_arm_group": inactive_arm,
        },
    )
    return manifest


def observation_from_current_schema(
    observation: Mapping[str, Any],
    server_metadata: Mapping[str, Any],
    config: CurrentSchemaBridgeConfig,
) -> ObservationPacket:
    """Project one flat observation dict into the bimanual-first packet shape."""

    manifest = build_current_schema_bridge_manifest(server_metadata, config)
    active_arm = config.active_arm_group()
    inactive_arm = config.inactive_arm_group()

    video: dict[str, TemporalVideoSequence] = {}
    if bool(server_metadata.get("needs_wrist_camera", False)):
        video[f"{active_arm}.wrist_left"] = TemporalVideoSequence(
            sample_indices=(0,),
            frames=(np.asarray(observation["observation/wrist_image_left"]),),
        )
        if bool(server_metadata.get("needs_stereo_camera", False)):
            video[f"{active_arm}.wrist_right"] = TemporalVideoSequence(
                sample_indices=(0,),
                frames=(np.asarray(observation["observation/wrist_image_right"]),),
            )

    for camera_index in range(int(server_metadata.get("n_external_cameras", 0))):
        left_key = f"observation/exterior_image_{camera_index + 1}_left"
        video[f"global.exterior_{camera_index + 1}_left"] = TemporalVideoSequence(
            sample_indices=(0,),
            frames=(np.asarray(observation[left_key]),),
        )
        if bool(server_metadata.get("needs_stereo_camera", False)):
            right_key = f"observation/exterior_image_{camera_index + 1}_right"
            video[f"global.exterior_{camera_index + 1}_right"] = TemporalVideoSequence(
                sample_indices=(0,),
                frames=(np.asarray(observation[right_key]),),
            )

    packet = ObservationPacket(
        video=video,
        arms={
            active_arm: ArmObservationGroup(
                streams={
                    "joint_position": TemporalStateSequence(
                        sample_indices=(0,),
                        values=(np.asarray(observation["observation/joint_position"]).reshape(-1),),
                    ),
                    "cartesian_position": TemporalStateSequence(
                        sample_indices=(0,),
                        values=(np.asarray(observation["observation/cartesian_position"]).reshape(-1),),
                    ),
                    "gripper_position": TemporalStateSequence(
                        sample_indices=(0,),
                        values=(np.asarray(observation["observation/gripper_position"]).reshape(-1),),
                    ),
                }
            ),
            inactive_arm: ArmObservationGroup(streams={}),
        },
        language={config.instruction_field: str(observation["prompt"])},
        session_id=observation.get("session_id"),
        metadata={
            "bridge_origin": "current_flat_schema",
            "active_arm_group": active_arm,
            "inactive_arm_group": inactive_arm,
        },
    )
    packet.validate_against(manifest)
    return packet


def observation_to_current_schema(
    packet: ObservationPacket,
    server_metadata: Mapping[str, Any],
    config: CurrentSchemaBridgeConfig,
) -> dict[str, Any]:
    """Project one bimanual packet back onto the historical flat observation shape."""

    active_arm = config.active_arm_group()
    if active_arm not in packet.arms:
        raise ValueError(f"ObservationPacket is missing the active arm group {active_arm!r}.")
    group = packet.arms[active_arm]

    observation: dict[str, Any] = {
        "observation/joint_position": group.streams["joint_position"].values[-1],
        "observation/cartesian_position": group.streams["cartesian_position"].values[-1],
        "observation/gripper_position": group.streams["gripper_position"].values[-1],
        "prompt": packet.language[config.instruction_field],
    }
    if packet.session_id is not None:
        observation["session_id"] = packet.session_id

    if bool(server_metadata.get("needs_wrist_camera", False)):
        observation["observation/wrist_image_left"] = packet.video[f"{active_arm}.wrist_left"].frames[-1]
        if bool(server_metadata.get("needs_stereo_camera", False)):
            observation["observation/wrist_image_right"] = packet.video[f"{active_arm}.wrist_right"].frames[-1]

    for camera_index in range(int(server_metadata.get("n_external_cameras", 0))):
        observation[f"observation/exterior_image_{camera_index + 1}_left"] = packet.video[
            f"global.exterior_{camera_index + 1}_left"
        ].frames[-1]
        if bool(server_metadata.get("needs_stereo_camera", False)):
            observation[f"observation/exterior_image_{camera_index + 1}_right"] = packet.video[
                f"global.exterior_{camera_index + 1}_right"
            ].frames[-1]
    return observation


def action_packet_from_current_schema(
    action_dict: Mapping[str, Any],
    server_metadata: Mapping[str, Any],
    config: CurrentSchemaBridgeConfig,
) -> ActionPacket:
    """Project one flat-schema action dict into the bimanual-first packet shape."""

    manifest = build_current_schema_bridge_manifest(server_metadata, config)
    active_arm = config.active_arm_group()
    inactive_arm = config.inactive_arm_group()
    actions = np.asarray(action_dict["actions"])
    packet = ActionPacket(
        arms={
            active_arm: ArmActionGroup(
                streams={
                    "policy_action": ActionChunk(actions),
                }
            )
        },
        padding={
            inactive_arm: PaddingRule(
                strategy=config.inactive_arm_padding,
                reason=(
                    "Legacy current-schema bridge: policy controls one arm only, so the other arm "
                    "is kept static by an explicit padding rule."
                ),
            )
        },
        metadata={
            "bridge_origin": "current_flat_schema",
            "action_space": str(server_metadata["action_space"]),
            "active_arm_group": active_arm,
            "inactive_arm_group": inactive_arm,
        },
    )
    packet.validate_against(manifest)
    return packet


def action_packet_to_current_schema(
    packet: ActionPacket,
    server_metadata: Mapping[str, Any],
    config: CurrentSchemaBridgeConfig,
) -> dict[str, Any]:
    """Project one bimanual action packet back onto the historical flat action dict."""

    del server_metadata
    active_arm = config.active_arm_group()
    if active_arm not in packet.arms:
        raise ValueError(f"ActionPacket is missing the active arm group {active_arm!r}.")
    group = packet.arms[active_arm]
    if set(group.streams) != {"policy_action"}:
        raise ValueError(
            "Legacy current-schema bridge expects exactly one active-arm action stream named 'policy_action'."
        )
    return {"actions": group.streams["policy_action"].values}


@dataclasses.dataclass(slots=True)
class LegacyCurrentSchemaPolicyBridge(BimanualPolicyAdapter):
    """Wrap a current-schema policy adapter behind the bimanual-first protocol."""

    legacy_policy: CurrentSchemaPolicyAdapter
    bridge_config: CurrentSchemaBridgeConfig = dataclasses.field(default_factory=CurrentSchemaBridgeConfig)
    _cached_server_metadata: dict[str, Any] | None = dataclasses.field(default=None, init=False, repr=False)

    def assert_ready_for_benchmark(self) -> None:
        self.legacy_policy.assert_ready_for_benchmark()

    def _server_metadata(self) -> dict[str, Any]:
        if self._cached_server_metadata is None:
            self._cached_server_metadata = dict(self.legacy_policy.get_server_metadata())
        return self._cached_server_metadata

    def build_manifest(self) -> HarnessManifest:
        return build_current_schema_bridge_manifest(self._server_metadata(), self.bridge_config)

    def infer(self, observation: ObservationPacket) -> ActionPacket:
        server_metadata = self._server_metadata()
        flat_observation = observation_to_current_schema(observation, server_metadata, self.bridge_config)
        flat_action = self.legacy_policy.infer(flat_observation)
        return action_packet_from_current_schema(flat_action, server_metadata, self.bridge_config)

    def reset(self, reset_info: dict[str, Any]) -> Any:
        return self.legacy_policy.reset(reset_info)

    def build_policy_metadata(self):
        return self.legacy_policy.build_policy_metadata()

    def build_validation_metadata(self):
        return self.legacy_policy.build_validation_metadata()

    def build_notes(self):
        notes = list(self.legacy_policy.build_notes())
        notes.append(
            DecisionNote(
                topic="policy.phase3_bridge",
                choice=f"legacy_flat_schema_via_{self.bridge_config.active_arm_group()}",
                status="benchmark_default",
                rationale=(
                    "Phase 3 wraps the historical single-arm current-schema policy path inside the "
                    "bimanual representation using an explicit padding rule for the other arm."
                ),
                evidence="vla_harness/legacy/current_schema_bridge.py",
            )
        )
        return notes

    def close(self) -> None:
        self.legacy_policy.close()


@dataclasses.dataclass(slots=True)
class LegacyCurrentSchemaEmbodimentBridge(BimanualEmbodimentAdapter):
    """Wrap a current-schema embodiment adapter behind the bimanual-first protocol."""

    legacy_embodiment: CurrentSchemaEmbodimentAdapter
    bridge_config: CurrentSchemaBridgeConfig = dataclasses.field(default_factory=CurrentSchemaBridgeConfig)

    def prepare_episode(self, manifest: HarnessManifest, prompt: str) -> None:
        self.legacy_embodiment.prepare_episode(_legacy_server_metadata(manifest), prompt)

    def capture_observation(
        self,
        manifest: HarnessManifest,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> ObservationPacket:
        server_metadata = _legacy_server_metadata(manifest)
        flat_observation = self.legacy_embodiment.build_observation(
            server_metadata,
            prompt,
            session_id=session_id if bool(server_metadata.get("needs_session_id", False)) else None,
        )
        return observation_from_current_schema(flat_observation, server_metadata, self.bridge_config)

    def execute_action(self, manifest: HarnessManifest, action: ActionPacket) -> None:
        server_metadata = _legacy_server_metadata(manifest)
        flat_action = action_packet_to_current_schema(action, server_metadata, self.bridge_config)
        self.legacy_embodiment.execute_action_chunk(flat_action["actions"], action_space=str(server_metadata["action_space"]))

    def build_embodiment_metadata(self):
        return self.legacy_embodiment.build_embodiment_metadata()

    def build_notes(self):
        notes = list(self.legacy_embodiment.build_notes())
        notes.append(
            DecisionNote(
                topic="embodiment.phase3_bridge",
                choice=f"legacy_flat_schema_via_{self.bridge_config.active_arm_group()}",
                status="benchmark_default",
                rationale=(
                    "Phase 3 keeps the historical single-active-arm embodiment path available only "
                    "through an explicit bridge into the bimanual representation."
                ),
                evidence="vla_harness/legacy/current_schema_bridge.py",
            )
        )
        return notes

    def close(self) -> None:
        closer = getattr(self.legacy_embodiment, "close", None)
        if callable(closer):
            closer()
