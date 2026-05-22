"""Embodiment adapter protocol for the bimanual-first internal representation."""

from __future__ import annotations

from typing import Protocol

from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.observation import ObservationPacket


class BimanualEmbodimentAdapter(Protocol):
    def prepare_episode(self, manifest: HarnessManifest, prompt: str) -> None:
        """Prepare embodiment state for a new episode under the given manifest."""

    def capture_observation(
        self,
        manifest: HarnessManifest,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> ObservationPacket:
        """Capture one transport-neutral observation packet."""

    def execute_action(self, manifest: HarnessManifest, action: ActionPacket) -> None:
        """Execute one transport-neutral action packet."""

    def build_embodiment_metadata(self) -> EmbodimentMetadata:
        """Return structured fairness metadata for the embodiment."""

    def build_notes(self) -> list[DecisionNote]:
        """Return adapter-owned fairness notes."""

    def close(self) -> None:
        """Release adapter resources."""
