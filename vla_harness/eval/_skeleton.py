"""Runnable skeleton for parity and replay callables used by new adapters."""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.observation import ObservationPacket


EVAL_TEMPLATE_FIELD_GUIDE = {
    "fixture_source": "copy_from_upstream when official recorded fixtures exist; otherwise benchmark_derived from your captured replay corpus",
    "official_callable_source": "copy_from_upstream: direct policy entrypoint, official server, or official health/inference API",
    "negative_control": "benchmark_derived: one deliberate break path that proves the parity battery can fail",
    "tolerances": "benchmark_derived but mandatory: record exact atol/rtol used for the adapter",
    "cpu_smoke_test": "copy_from_this_repo: every adapter ships with a CPU-only smoke test before any GPU oracle",
}


@dataclasses.dataclass(slots=True)
class ReplayFixture:
    name: str
    observation: ObservationPacket
    expected_action: np.ndarray | None = None


ActionCallable = Callable[[ObservationPacket], ActionPacket]


def run_parity_battery(
    fixtures: list[ReplayFixture],
    *,
    official_callable: ActionCallable,
    harness_callable: ActionCallable,
    atol: float,
    rtol: float,
) -> list[tuple[str, float]]:
    """Minimal parity harness for future adapter-specific eval code.

    TODO(copy_from_upstream, cookbook §4.1): replace this with a richer battery once you know the upstream runtime shape.
    TODO(benchmark_derived, cookbook §4.2): add at least one negative control that proves the battery can fail.
    """

    diffs: list[tuple[str, float]] = []
    for fixture in fixtures:
        official = official_callable(fixture.observation)
        harness = harness_callable(fixture.observation)
        official_stream = next(iter(next(iter(official.arms.values())).streams.values())).values
        harness_stream = next(iter(next(iter(harness.arms.values())).streams.values())).values
        max_abs_diff = float(np.max(np.abs(official_stream - harness_stream)))
        if not np.allclose(official_stream, harness_stream, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Parity failed for fixture {fixture.name!r}: max_abs_diff={max_abs_diff}, "
                f"atol={atol}, rtol={rtol}"
            )
        diffs.append((fixture.name, max_abs_diff))
    return diffs
