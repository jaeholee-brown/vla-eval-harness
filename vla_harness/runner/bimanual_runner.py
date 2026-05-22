"""Runner for the bimanual-first internal representation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import time
import uuid

from vla_harness.adapters.embodiment.bimanual import BimanualEmbodimentAdapter
from vla_harness.adapters.policy.bimanual import BimanualPolicyAdapter
from vla_harness.logging.decision_log import FairnessLog
from vla_harness.logging.decision_log import RuntimeMetadata
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.observation import ObservationPacket
from vla_harness.runner.current_schema_runner import UPSTREAM_ROBOARENA_COMMIT


@dataclasses.dataclass(slots=True)
class BimanualRunConfig:
    prompt: str
    max_steps: int = 1
    run_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    output_dir: Path = Path("runs")
    reset_payload: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class BimanualRunResult:
    log_path: Path
    session_id: str
    manifest_name: str


class BimanualRunner:
    """Run bimanual episodes with fairness logging and manifest validation."""

    def __init__(self, policy: BimanualPolicyAdapter, embodiment: BimanualEmbodimentAdapter) -> None:
        self._policy = policy
        self._embodiment = embodiment
        self._run_in_progress = False

    def run_episode(self, config: BimanualRunConfig) -> BimanualRunResult:
        if self._run_in_progress:
            raise RuntimeError("BimanualRunner only allows one active rollout per process.")

        self._run_in_progress = True
        try:
            self._policy.assert_ready_for_benchmark()
            manifest = self._policy.build_manifest()
            self._embodiment.prepare_episode(manifest, config.prompt)

            fairness_log = FairnessLog.create(
                run_id=config.run_id,
                prompt=config.prompt,
                session_id=config.session_id,
                policy=self._policy.build_policy_metadata(),
                embodiment=self._embodiment.build_embodiment_metadata(),
                runtime=RuntimeMetadata(
                    upstream_roboarena_commit=UPSTREAM_ROBOARENA_COMMIT,
                    transport="internal_protocol+adapter_local_transport",
                    compression="adapter_defined",
                    concurrency_model="single_rollout_per_process",
                ),
                validation=self._policy.build_validation_metadata(),
                notes=self._policy.build_notes() + self._embodiment.build_notes(),
            )

            for step_index in range(config.max_steps):
                observation = self._embodiment.capture_observation(
                    manifest,
                    config.prompt,
                    session_id=config.session_id,
                )
                _validate_observation(observation, manifest)
                start_time = time.perf_counter()
                action = self._policy.infer(observation)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                _validate_action(action, manifest)

                action_rows, action_dim = _summarize_action_packet(action)
                fairness_log.record_step(
                    step_index=step_index,
                    request_bytes=0,
                    response_bytes=0,
                    latency_ms=latency_ms,
                    action_rows=action_rows,
                    action_dim=action_dim,
                )
                self._embodiment.execute_action(manifest, action)

            reset_payload = {"session_id": config.session_id, **config.reset_payload}
            self._policy.reset(reset_payload)
            fairness_log.finalize()
            run_dir = Path(config.output_dir) / config.run_id
            log_path = fairness_log.write(run_dir / "decision_log.json")
            return BimanualRunResult(
                log_path=log_path,
                session_id=config.session_id,
                manifest_name=manifest.name,
            )
        finally:
            self._run_in_progress = False


def _validate_observation(observation: ObservationPacket, manifest) -> None:
    observation.validate_against(manifest)


def _validate_action(action: ActionPacket, manifest) -> None:
    action.validate_against(manifest)


def _summarize_action_packet(action: ActionPacket) -> tuple[int, int]:
    rows = 0
    dims = 0
    for group in action.arms.values():
        for chunk in group.streams.values():
            rows = max(rows, chunk.horizon)
            dims += chunk.dim
    return rows, dims
