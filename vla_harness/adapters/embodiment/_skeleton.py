"""Runnable skeleton for future bimanual embodiment adapters.

Every field below is tagged as either:

- `copy_from_upstream`: read this from official embodiment docs, configs, or SDK
- `benchmark_derived`: choose this only when the official embodiment stack is silent
"""

from __future__ import annotations

import dataclasses

from vla_harness.adapters.embodiment.bimanual import BimanualEmbodimentAdapter
from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.observation import ObservationPacket


EMBODIMENT_TEMPLATE_FIELD_GUIDE = {
    "embodiment_family": "copy_from_upstream: robot family / repo / SDK name",
    "backend_name": "copy_from_upstream: official runtime or driver stack name",
    "camera_role_order": "copy_from_upstream: official camera names and their required order",
    "arm_group_names": "copy_from_upstream: left/right arm identifiers exposed by the embodiment stack",
    "state_sources": "copy_from_upstream: which proprio streams the embodiment can read directly",
    "control_hz": "copy_from_upstream when documented; benchmark_derived only if the stack is silent",
    "chunk_consumption_policy": "copy_from_upstream when documented; benchmark_derived otherwise",
    "static_padding_rule": "benchmark_derived: how to keep an uncontrolled arm static if a single-arm policy is bridged",
}


@dataclasses.dataclass(slots=True)
class EmbodimentTemplateConfig:
    embodiment_family: str
    backend_name: str
    control_hz: float | None
    chunk_consumption_policy: str | None
    static_padding_rule: str = "hold_static"


class SkeletonBimanualEmbodimentAdapter(BimanualEmbodimentAdapter):
    """Copy this file and replace the TODOs with one concrete embodiment adapter."""

    def __init__(self, config: EmbodimentTemplateConfig) -> None:
        self._config = config

    def prepare_episode(self, manifest: HarnessManifest, prompt: str) -> None:
        # TODO(copy_from_upstream): home/reset both arms using the official robot stack.
        del manifest, prompt
        return None

    def capture_observation(
        self,
        manifest: HarnessManifest,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> ObservationPacket:
        # TODO(copy_from_upstream): read the official camera feeds and proprio streams, then
        # place them into the ObservationPacket exactly as named by the manifest.
        del manifest, prompt, session_id
        raise NotImplementedError

    def execute_action(self, manifest: HarnessManifest, action: ActionPacket) -> None:
        # TODO(copy_from_upstream): execute each arm's action streams through the official control API.
        # TODO(benchmark_derived): if one arm is covered only by a padding rule, apply the configured
        # static-padding behavior here instead of inventing motion.
        del manifest, action
        raise NotImplementedError

    def build_embodiment_metadata(self) -> EmbodimentMetadata:
        return EmbodimentMetadata(
            family=self._config.embodiment_family,
            backend=self._config.backend_name,
            active_arm=None,
            parked_arm=None,
            parked_arm_rule=self._config.static_padding_rule,
            control_hz=self._config.control_hz,
            chunk_consumption_policy=self._config.chunk_consumption_policy,
        )

    def build_notes(self) -> list[DecisionNote]:
        return [
            DecisionNote(
                topic="embodiment.static_padding_rule",
                choice=self._config.static_padding_rule,
                status="benchmark_default",
                rationale=(
                    "If a bridged single-arm policy leaves one arm uncontrolled, the embodiment "
                    "must keep that arm static explicitly instead of inventing motion."
                ),
                evidence="User-scoped harness policy for bimanual-only evaluation.",
            )
        ]

    def close(self) -> None:
        return None
