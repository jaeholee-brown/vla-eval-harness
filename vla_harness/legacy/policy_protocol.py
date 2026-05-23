"""Minimal policy adapter protocol for the historical flat-schema bootstrap.

This Protocol is kept for the pinned current-schema reference path. New policy
adapters should target the bimanual internal representation once Phase 3 lands.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol

from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata


class CurrentSchemaPolicyAdapter(Protocol):
    def assert_ready_for_benchmark(self) -> None:
        """Fail fast if the adapter's fidelity claims are not actually wired."""

    def get_server_metadata(self) -> dict[str, Any]:
        """Return the flat-schema metadata advertised by the runtime."""

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Run policy inference for one flat-schema observation dict."""

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
