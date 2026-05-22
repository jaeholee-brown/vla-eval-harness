"""Policy adapter protocol for the bimanual-first internal representation."""

from __future__ import annotations

from typing import Any
from typing import Protocol

from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.observation import ObservationPacket


class BimanualPolicyAdapter(Protocol):
    def assert_ready_for_benchmark(self) -> None:
        """Fail fast if the adapter's fidelity claims are not actually wired."""

    def build_manifest(self) -> HarnessManifest:
        """Return the transport-neutral manifest for this policy path."""

    def infer(self, observation: ObservationPacket) -> ActionPacket:
        """Run policy inference for one transport-neutral observation packet."""

    def reset(self, reset_info: dict[str, Any]) -> Any:
        """Reset any policy-side rollout state."""

    def build_policy_metadata(self) -> PolicyMetadata:
        """Return structured fairness metadata for the policy."""

    def build_validation_metadata(self) -> ValidationMetadata:
        """Return the configured validation-oracle descriptors."""

    def build_notes(self) -> list[DecisionNote]:
        """Return adapter-owned fairness notes."""

    def close(self) -> None:
        """Release adapter resources."""
