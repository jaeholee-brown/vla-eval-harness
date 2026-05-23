"""Minimal embodiment adapter protocol for the historical flat-schema bootstrap.

This Protocol exists to preserve the pinned current-schema reference path.
Future embodiment adapters should target the bimanual internal representation,
not this single-active-arm bootstrap surface.
"""

from __future__ import annotations

from typing import Any
from typing import Mapping
from typing import Protocol

import numpy as np

from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata


class CurrentSchemaEmbodimentAdapter(Protocol):
    def prepare_episode(self, server_metadata: Mapping[str, Any], prompt: str) -> None:
        """Prepare hardware state for a new episode."""

    def build_observation(
        self,
        server_metadata: Mapping[str, Any],
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Collect a flat-schema observation packet from the active arm.

        This method is legacy bootstrap support only.
        """

    def execute_action_chunk(self, actions: np.ndarray, *, action_space: str) -> None:
        """Execute one action chunk on the active arm.

        This method is legacy bootstrap support only.
        """

    def build_embodiment_metadata(self) -> EmbodimentMetadata:
        """Return structured fairness metadata for the embodiment."""

    def build_notes(self) -> list[DecisionNote]:
        """Return adapter-owned fairness notes."""
