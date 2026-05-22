"""Shared sample types for source-backed bimanual embodiment adapters."""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np

from vla_harness.protocol.observation import TemporalStateSequence
from vla_harness.protocol.observation import TemporalVideoSequence


FrameLike = np.ndarray | Sequence[np.ndarray]
StateLike = np.ndarray | Sequence[np.ndarray]


@dataclasses.dataclass(slots=True)
class BimanualObservationSample:
    """One backend-owned snapshot of ordered cameras and per-arm state streams."""

    camera_frames: Mapping[str, FrameLike]
    arm_state_streams: Mapping[str, Mapping[str, StateLike]]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def coerce_video_sequence(value: FrameLike, sample_indices: tuple[int, ...]) -> TemporalVideoSequence:
    """Convert one backend camera value into a protocol video sequence."""

    if isinstance(value, np.ndarray):
        frames = (value,)
    else:
        frames = tuple(value)
    if len(frames) != len(sample_indices):
        raise ValueError(
            f"Video sample count mismatch: got {len(frames)} frames, expected {len(sample_indices)} "
            f"for sample_indices={sample_indices!r}"
        )
    return TemporalVideoSequence(sample_indices=sample_indices, frames=tuple(np.asarray(frame) for frame in frames))


def coerce_state_sequence(value: StateLike, sample_indices: tuple[int, ...]) -> TemporalStateSequence:
    """Convert one backend state value into a protocol state sequence."""

    if isinstance(value, np.ndarray):
        values = (value,)
    else:
        values = tuple(value)
    if len(values) != len(sample_indices):
        raise ValueError(
            f"State sample count mismatch: got {len(values)} values, expected {len(sample_indices)} "
            f"for sample_indices={sample_indices!r}"
        )
    return TemporalStateSequence(sample_indices=sample_indices, values=tuple(np.asarray(item) for item in values))
