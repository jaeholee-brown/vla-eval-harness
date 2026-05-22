"""Runnable skeleton for future bimanual policy adapters.

Every field below is tagged as either:

- `copy_from_upstream`: fill this by reading official release code or docs
- `benchmark_derived`: fill this only when the upstream artifact is silent or
  when the policy must be bridged into a bimanual setup
"""

from __future__ import annotations

import dataclasses
from typing import Any

from vla_harness.adapters.policy.bimanual import BimanualPolicyAdapter
from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import ImagePreprocessMetadata
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.observation import ObservationPacket


POLICY_TEMPLATE_FIELD_GUIDE = {
    "policy_family": "copy_from_upstream: repo name / model card / release blog",
    "runtime_family": "copy_from_upstream: official server or policy API entrypoint",
    "schema_source": "copy_from_upstream: file that defines request schema or modality config",
    "checkpoint_ref": "copy_from_upstream: official checkpoint or repo_id",
    "normalization_tag": "copy_from_upstream: official norm tag / stats identifier when present",
    "camera_roles": "copy_from_upstream: official camera names and order",
    "state_layouts": "copy_from_upstream: official state stream names and dims",
    "action_semantics": "copy_from_upstream: official action stream semantics and horizons",
    "single_arm_padding_arm": "benchmark_derived: only if the policy controls one arm and the other must stay static",
    "single_arm_padding_rule": "benchmark_derived: e.g. hold_static or zero_velocity",
}


@dataclasses.dataclass(slots=True)
class PolicyTemplateConfig:
    policy_family: str
    runtime_family: str
    schema_source: str
    checkpoint_ref: str | None
    normalization_tag: str | None
    prompt_format_source: str | None
    dtype: str
    device: str
    single_arm_padding_arm: str | None = None
    single_arm_padding_rule: str | None = None


class SkeletonBimanualPolicyAdapter(BimanualPolicyAdapter):
    """Copy this file and replace the TODOs with one concrete policy adapter."""

    def __init__(self, config: PolicyTemplateConfig) -> None:
        self._config = config

    def assert_ready_for_benchmark(self) -> None:
        # TODO(copy_from_upstream, cookbook §2.2): wire any official runtime-health checks here.
        return None

    def build_manifest(self) -> HarnessManifest:
        # TODO(copy_from_upstream, cookbook §2.3): build manifest fields directly from the official
        # request schema or modality-config source named in self._config.schema_source.
        raise NotImplementedError

    def infer(self, observation: ObservationPacket) -> ActionPacket:
        # TODO(copy_from_upstream, cookbook §2.4): convert ObservationPacket into the official runtime input.
        # TODO(copy_from_upstream, cookbook §2.5): convert the official runtime output back into ActionPacket.
        # TODO(benchmark_derived, cookbook §2.5): if the policy is single-arm only, emit one arm group plus
        # an explicit static-padding rule for the other arm.
        del observation
        raise NotImplementedError

    def reset(self, reset_info: dict[str, Any]) -> Any:
        # TODO(copy_from_upstream, cookbook §2.2): delegate to the official reset/session API if it exists.
        del reset_info
        return None

    def build_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            family=self._config.policy_family,
            config_name=self._config.runtime_family,
            checkpoint_ref=self._config.checkpoint_ref,
            checkpoint_sha256=None,
            checkpoint_sha256_explanation="Fill this when checkpoint hashing is wired for the new adapter.",
            dtype=self._config.dtype,
            device=self._config.device,
            action_space=None,
            chunk_size=None,
            prompt_format_source=self._config.prompt_format_source,
            image_preprocess=ImagePreprocessMetadata(
                resize_resolution=None,
                resize_filter=None,
                color_space=None,
                output_dtype=None,
            ),
        )

    def build_validation_metadata(self) -> ValidationMetadata:
        return ValidationMetadata()

    def build_notes(self) -> list[DecisionNote]:
        notes = [
            DecisionNote(
                topic="policy.schema_source",
                choice=self._config.schema_source,
                status="official",
                rationale="The adapter should declare the exact upstream file that defines its runtime schema.",
                evidence=self._config.schema_source,
            )
        ]
        if self._config.single_arm_padding_arm is not None:
            notes.append(
                DecisionNote(
                    topic="policy.single_arm_padding_rule",
                    choice=f"{self._config.single_arm_padding_arm}:{self._config.single_arm_padding_rule}",
                    status="benchmark_default",
                    rationale=(
                        "This is the one allowed benchmark-side bridge for a single-arm policy on a "
                        "bimanual setup: make the other arm stay static explicitly."
                    ),
                    evidence="User-scoped harness policy for bimanual-only evaluation.",
                )
            )
        return notes

    def close(self) -> None:
        return None
