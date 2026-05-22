"""Dataclasses for the bimanual-first harness internal representation."""

from __future__ import annotations

import dataclasses
from typing import Literal


ArmSide = Literal["left", "right"]
ArmControlRole = Literal["policy_controlled", "static_pad_only"]
StateOrigin = Literal["sensor", "derived"]
ActionRepresentation = Literal["absolute", "relative", "velocity", "other"]
ActionDomain = Literal["joint", "eef", "gripper", "cartesian", "other"]


@dataclasses.dataclass(slots=True, frozen=True)
class ArmGroupSpec:
    """One named arm group in a bimanual embodiment."""

    name: str
    side: ArmSide
    control_role: ArmControlRole = "policy_controlled"
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Arm group name must be non-empty.")


@dataclasses.dataclass(slots=True, frozen=True)
class VideoStreamSpec:
    """One ordered camera stream with explicit temporal sampling."""

    name: str
    role: str
    sample_indices: tuple[int, ...] = (0,)
    arm_group: str | None = None
    order_index: int | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Video stream name must be non-empty.")
        if not self.role:
            raise ValueError("Video stream role must be non-empty.")
        if len(self.sample_indices) == 0:
            raise ValueError("Video stream sample_indices must be non-empty.")


@dataclasses.dataclass(slots=True, frozen=True)
class StateStreamSpec:
    """One state stream for one arm group."""

    name: str
    arm_group: str
    dim: int
    layout: str
    sample_indices: tuple[int, ...] = (0,)
    origin: StateOrigin = "sensor"
    derived_from: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("State stream name must be non-empty.")
        if not self.arm_group:
            raise ValueError("State stream arm_group must be non-empty.")
        if self.dim <= 0:
            raise ValueError("State stream dim must be positive.")
        if not self.layout:
            raise ValueError("State stream layout must be non-empty.")
        if len(self.sample_indices) == 0:
            raise ValueError("State stream sample_indices must be non-empty.")


@dataclasses.dataclass(slots=True, frozen=True)
class ActionSemantics:
    """Semantics for one action stream."""

    representation: ActionRepresentation
    domain: ActionDomain
    layout: str
    static_pad_strategy: str = "hold_static"

    def __post_init__(self) -> None:
        if not self.layout:
            raise ValueError("Action semantics layout must be non-empty.")
        if not self.static_pad_strategy:
            raise ValueError("Action semantics static_pad_strategy must be non-empty.")


@dataclasses.dataclass(slots=True, frozen=True)
class ActionStreamSpec:
    """One action stream for one arm group."""

    name: str
    arm_group: str
    dim: int
    semantics: ActionSemantics
    horizon: int | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Action stream name must be non-empty.")
        if not self.arm_group:
            raise ValueError("Action stream arm_group must be non-empty.")
        if self.dim <= 0:
            raise ValueError("Action stream dim must be positive.")
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("Action stream horizon must be positive when provided.")


@dataclasses.dataclass(slots=True, frozen=True)
class LanguageFieldSpec:
    """One language field such as an instruction string."""

    name: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Language field name must be non-empty.")


@dataclasses.dataclass(slots=True, frozen=True)
class HarnessManifest:
    """Transport-neutral schema for one policy/embodiment pairing."""

    name: str
    version: str
    arm_groups: tuple[ArmGroupSpec, ...]
    video_streams: tuple[VideoStreamSpec, ...]
    state_streams: tuple[StateStreamSpec, ...]
    action_streams: tuple[ActionStreamSpec, ...]
    language_fields: tuple[LanguageFieldSpec, ...] = (LanguageFieldSpec(name="instruction"),)
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Manifest name must be non-empty.")
        if not self.version:
            raise ValueError("Manifest version must be non-empty.")
        if len(self.arm_groups) != 2:
            raise ValueError("HarnessManifest currently requires exactly two arm groups (left and right).")
        arm_names = [spec.name for spec in self.arm_groups]
        if len(set(arm_names)) != len(arm_names):
            raise ValueError(f"Duplicate arm group names are not allowed: {arm_names!r}")
        arm_sides = [spec.side for spec in self.arm_groups]
        if sorted(arm_sides) != ["left", "right"]:
            raise ValueError(f"HarnessManifest arm group sides must be exactly left and right: {arm_sides!r}")
        video_names = [spec.name for spec in self.video_streams]
        if len(set(video_names)) != len(video_names):
            raise ValueError(f"Duplicate video stream names are not allowed: {video_names!r}")
        state_keys = [(spec.arm_group, spec.name) for spec in self.state_streams]
        if len(set(state_keys)) != len(state_keys):
            raise ValueError(f"Duplicate state streams are not allowed: {state_keys!r}")
        action_keys = [(spec.arm_group, spec.name) for spec in self.action_streams]
        if len(set(action_keys)) != len(action_keys):
            raise ValueError(f"Duplicate action streams are not allowed: {action_keys!r}")
        language_names = [spec.name for spec in self.language_fields]
        if len(set(language_names)) != len(language_names):
            raise ValueError(f"Duplicate language field names are not allowed: {language_names!r}")
        defined_arms = {spec.name for spec in self.arm_groups}
        for spec in self.state_streams:
            if spec.arm_group not in defined_arms:
                raise ValueError(f"State stream references unknown arm_group {spec.arm_group!r}")
        for spec in self.action_streams:
            if spec.arm_group not in defined_arms:
                raise ValueError(f"Action stream references unknown arm_group {spec.arm_group!r}")
        for spec in self.video_streams:
            if spec.arm_group is not None and spec.arm_group not in defined_arms:
                raise ValueError(f"Video stream references unknown arm_group {spec.arm_group!r}")

    def arm_group(self, name: str) -> ArmGroupSpec:
        for spec in self.arm_groups:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def state_streams_for_arm(self, arm_group: str) -> tuple[StateStreamSpec, ...]:
        return tuple(spec for spec in self.state_streams if spec.arm_group == arm_group)

    def action_streams_for_arm(self, arm_group: str) -> tuple[ActionStreamSpec, ...]:
        return tuple(spec for spec in self.action_streams if spec.arm_group == arm_group)

    def ordered_video_streams(self) -> tuple[VideoStreamSpec, ...]:
        return tuple(
            sorted(
                self.video_streams,
                key=lambda spec: (
                    spec.order_index is None,
                    spec.order_index if spec.order_index is not None else spec.name,
                    spec.name,
                ),
            )
        )
