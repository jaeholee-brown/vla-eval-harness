"""Structured fairness log for current-schema runs."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any
from typing import Literal


DecisionStatus = Literal["official", "adapter", "benchmark_default", "scoped_out"]


@dataclasses.dataclass(slots=True)
class DecisionNote:
    topic: str
    choice: str
    status: DecisionStatus
    rationale: str
    evidence: str


@dataclasses.dataclass(slots=True)
class ImagePreprocessMetadata:
    resize_resolution: tuple[int, int] | None
    resize_filter: str | None
    color_space: str | None
    output_dtype: str | None


@dataclasses.dataclass(slots=True)
class PolicyMetadata:
    family: str
    config_name: str
    checkpoint_ref: str | None
    checkpoint_sha256: str | None
    checkpoint_sha256_explanation: str | None
    dtype: str
    device: str
    action_space: str | None
    chunk_size: int | None
    prompt_format_source: str | None
    image_preprocess: ImagePreprocessMetadata


@dataclasses.dataclass(slots=True)
class EmbodimentMetadata:
    family: str
    backend: str
    active_arm: str | None
    parked_arm: str | None
    parked_arm_rule: str | None
    control_hz: float | None
    chunk_consumption_policy: str | None


@dataclasses.dataclass(slots=True)
class RuntimeMetadata:
    upstream_roboarena_commit: str
    transport: str
    compression: str
    concurrency_model: str


@dataclasses.dataclass(slots=True)
class ValidationMetadata:
    preprocessing_oracle: str | None = None
    preprocessing_allowed_atol: float | None = None
    preprocessing_allowed_rtol: float | None = None
    action_oracle: str | None = None
    action_allowed_atol: float | None = None
    action_allowed_rtol: float | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    passed: bool | None = None


@dataclasses.dataclass(slots=True)
class StepMetric:
    step_index: int
    request_bytes: int
    response_bytes: int
    latency_ms: float
    action_rows: int
    action_dim: int


@dataclasses.dataclass(slots=True)
class RunMetadata:
    run_id: str
    prompt: str
    session_id: str
    started_at_utc: str
    finished_at_utc: str | None = None


@dataclasses.dataclass(slots=True)
class FairnessLog:
    run: RunMetadata
    policy: PolicyMetadata
    embodiment: EmbodimentMetadata
    runtime: RuntimeMetadata
    validation: ValidationMetadata = dataclasses.field(default_factory=ValidationMetadata)
    notes: list[DecisionNote] = dataclasses.field(default_factory=list)
    step_metrics: list[StepMetric] = dataclasses.field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        prompt: str,
        session_id: str,
        policy: PolicyMetadata,
        embodiment: EmbodimentMetadata,
        runtime: RuntimeMetadata,
        validation: ValidationMetadata | None = None,
        notes: list[DecisionNote] | None = None,
    ) -> "FairnessLog":
        return cls(
            run=RunMetadata(
                run_id=run_id,
                prompt=prompt,
                session_id=session_id,
                started_at_utc=dt.datetime.now(dt.UTC).isoformat(),
            ),
            policy=policy,
            embodiment=embodiment,
            runtime=runtime,
            validation=validation or ValidationMetadata(),
            notes=notes or [],
        )

    def add_note(self, note: DecisionNote) -> None:
        self.notes.append(note)

    def record_step(
        self,
        *,
        step_index: int,
        request_bytes: int,
        response_bytes: int,
        latency_ms: float,
        action_rows: int,
        action_dim: int,
    ) -> None:
        self.step_metrics.append(
            StepMetric(
                step_index=step_index,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                latency_ms=latency_ms,
                action_rows=action_rows,
                action_dim=action_dim,
            )
        )

    def finalize(
        self,
        *,
        validation_passed: bool | None = None,
        max_abs_diff: float | None = None,
        max_rel_diff: float | None = None,
    ) -> None:
        self.run.finished_at_utc = dt.datetime.now(dt.UTC).isoformat()
        if validation_passed is not None:
            self.validation.passed = validation_passed
        if max_abs_diff is not None:
            self.validation.max_abs_diff = max_abs_diff
        if max_rel_diff is not None:
            self.validation.max_rel_diff = max_rel_diff

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def write(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output_path
