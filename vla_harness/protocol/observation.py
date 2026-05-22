"""Observation packet dataclasses for the bimanual-first internal representation."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from vla_harness.protocol.manifest import HarnessManifest


@dataclasses.dataclass(slots=True, frozen=True)
class TemporalVideoSequence:
    """A sequence of HWC frames with explicit sampling indices."""

    sample_indices: tuple[int, ...]
    frames: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.sample_indices) == 0:
            raise ValueError("TemporalVideoSequence.sample_indices must be non-empty.")
        if len(self.sample_indices) != len(self.frames):
            raise ValueError("TemporalVideoSequence sample_indices and frames must have equal length.")
        for frame in self.frames:
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("Each video frame must be shaped (H, W, 3).")


@dataclasses.dataclass(slots=True, frozen=True)
class TemporalStateSequence:
    """A sequence of 1-D state vectors with explicit sampling indices."""

    sample_indices: tuple[int, ...]
    values: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.sample_indices) == 0:
            raise ValueError("TemporalStateSequence.sample_indices must be non-empty.")
        if len(self.sample_indices) != len(self.values):
            raise ValueError("TemporalStateSequence sample_indices and values must have equal length.")
        for value in self.values:
            if value.ndim != 1:
                raise ValueError("Each state sample must be a rank-1 vector.")

    @property
    def dim(self) -> int:
        return int(self.values[0].shape[0])


@dataclasses.dataclass(slots=True, frozen=True)
class ArmObservationGroup:
    """All state streams for one named arm group."""

    streams: dict[str, TemporalStateSequence]

    def __post_init__(self) -> None:
        if len(set(self.streams)) != len(self.streams):
            raise ValueError("ArmObservationGroup stream names must be unique.")


@dataclasses.dataclass(slots=True, frozen=True)
class ObservationPacket:
    """Transport-neutral observation packet grouped by arm and modality."""

    video: dict[str, TemporalVideoSequence]
    arms: dict[str, ArmObservationGroup]
    language: dict[str, str]
    session_id: str | None = None
    step_index: int | None = None
    timestamp_ns: int | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.video)) != len(self.video):
            raise ValueError("ObservationPacket video stream names must be unique.")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("ObservationPacket arm group names must be unique.")

    def validate_against(self, manifest: HarnessManifest) -> None:
        for spec in manifest.video_streams:
            if spec.required and spec.name not in self.video:
                raise ValueError(f"Missing required video stream {spec.name!r}")
            if spec.name in self.video:
                sequence = self.video[spec.name]
                if sequence.sample_indices != spec.sample_indices:
                    raise ValueError(
                        f"Video stream {spec.name!r} sample_indices mismatch: "
                        f"{sequence.sample_indices!r} != {spec.sample_indices!r}"
                    )

        for spec in manifest.language_fields:
            if spec.required and spec.name not in self.language:
                raise ValueError(f"Missing required language field {spec.name!r}")

        for arm_spec in manifest.arm_groups:
            group = self.arms.get(arm_spec.name)
            required_streams = manifest.state_streams_for_arm(arm_spec.name)
            if group is None:
                if any(stream.required for stream in required_streams):
                    raise ValueError(f"Missing required arm group {arm_spec.name!r}")
                continue
            for stream_spec in required_streams:
                if stream_spec.required and stream_spec.name not in group.streams:
                    raise ValueError(f"Missing required state stream {(arm_spec.name, stream_spec.name)!r}")
                if stream_spec.name in group.streams:
                    sequence = group.streams[stream_spec.name]
                    if sequence.sample_indices != stream_spec.sample_indices:
                        raise ValueError(
                            f"State stream {(arm_spec.name, stream_spec.name)!r} sample_indices mismatch: "
                            f"{sequence.sample_indices!r} != {stream_spec.sample_indices!r}"
                        )
                    if sequence.dim != stream_spec.dim:
                        raise ValueError(
                            f"State stream {(arm_spec.name, stream_spec.name)!r} dim mismatch: "
                            f"{sequence.dim} != {stream_spec.dim}"
                        )
