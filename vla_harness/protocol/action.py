"""Action packet dataclasses for the bimanual-first internal representation."""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Literal

import numpy as np

from vla_harness.protocol.manifest import HarnessManifest


PaddingStrategy = Literal["hold_static", "zero_velocity", "repeat_last"]


@dataclasses.dataclass(slots=True, frozen=True)
class PaddingRule:
    """How to keep one arm static when the policy does not control it directly."""

    strategy: PaddingStrategy
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("PaddingRule.reason must be non-empty.")


@dataclasses.dataclass(slots=True, frozen=True)
class ActionChunk:
    """One `(horizon, dim)` action chunk."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("ActionChunk values must be rank-2, shaped (horizon, dim).")

    @property
    def horizon(self) -> int:
        return int(self.values.shape[0])

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])


@dataclasses.dataclass(slots=True, frozen=True)
class ArmActionGroup:
    """All action streams for one named arm group."""

    streams: dict[str, ActionChunk]

    def __post_init__(self) -> None:
        if len(set(self.streams)) != len(self.streams):
            raise ValueError("ArmActionGroup stream names must be unique.")


@dataclasses.dataclass(slots=True, frozen=True)
class ActionPacket:
    """Transport-neutral action packet grouped by arm."""

    arms: dict[str, ArmActionGroup]
    padding: dict[str, PaddingRule] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("ActionPacket arm group names must be unique.")
        overlap = set(self.arms).intersection(self.padding)
        if overlap:
            raise ValueError(f"An arm cannot have both concrete actions and padding rules: {sorted(overlap)!r}")

    def validate_against(self, manifest: HarnessManifest) -> None:
        for arm_spec in manifest.arm_groups:
            if arm_spec.control_role == "static_pad_only":
                if arm_spec.name not in self.padding:
                    raise ValueError(f"Static-pad arm {arm_spec.name!r} requires an explicit padding rule.")
                continue

            if arm_spec.name not in self.arms:
                raise ValueError(f"Missing action group for policy-controlled arm {arm_spec.name!r}")

            group = self.arms[arm_spec.name]
            for stream_spec in manifest.action_streams_for_arm(arm_spec.name):
                if stream_spec.required and stream_spec.name not in group.streams:
                    raise ValueError(f"Missing required action stream {(arm_spec.name, stream_spec.name)!r}")
                if stream_spec.name in group.streams:
                    chunk = group.streams[stream_spec.name]
                    if chunk.dim != stream_spec.dim:
                        raise ValueError(
                            f"Action stream {(arm_spec.name, stream_spec.name)!r} dim mismatch: "
                            f"{chunk.dim} != {stream_spec.dim}"
                        )
                    if stream_spec.horizon is not None and chunk.horizon != stream_spec.horizon:
                        raise ValueError(
                            f"Action stream {(arm_spec.name, stream_spec.name)!r} horizon mismatch: "
                            f"{chunk.horizon} != {stream_spec.horizon}"
                        )
