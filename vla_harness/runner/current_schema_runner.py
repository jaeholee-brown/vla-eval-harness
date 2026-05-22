"""Current flat-schema runner around the pinned RoboArena transport slice."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import time
import uuid

import numpy as np

from vla_harness._upstream.roboarena.utils import msgpack_numpy
from vla_harness.adapters.embodiment.base import CurrentSchemaEmbodimentAdapter
from vla_harness.adapters.policy.base import CurrentSchemaPolicyAdapter
from vla_harness.logging.decision_log import FairnessLog
from vla_harness.logging.decision_log import RuntimeMetadata


UPSTREAM_ROBOARENA_COMMIT = "a07f93d"


@dataclasses.dataclass(slots=True)
class CurrentSchemaRunConfig:
    prompt: str
    max_steps: int = 1
    run_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    output_dir: Path = Path("runs")
    reset_payload: dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class CurrentSchemaRunResult:
    log_path: Path
    session_id: str
    server_metadata: dict[str, object]


class CurrentSchemaRunner:
    """Run current flat-schema episodes with latency and payload logging."""

    def __init__(
        self,
        policy: CurrentSchemaPolicyAdapter,
        embodiment: CurrentSchemaEmbodimentAdapter,
    ) -> None:
        self._policy = policy
        self._embodiment = embodiment
        self._run_in_progress = False

    def run_episode(self, config: CurrentSchemaRunConfig) -> CurrentSchemaRunResult:
        if self._run_in_progress:
            raise RuntimeError("CurrentSchemaRunner only allows one active rollout per process.")

        self._run_in_progress = True
        try:
            server_metadata = self._policy.get_server_metadata()
            self._embodiment.prepare_episode(server_metadata, config.prompt)

            fairness_log = FairnessLog.create(
                run_id=config.run_id,
                prompt=config.prompt,
                session_id=config.session_id,
                policy=self._policy.build_policy_metadata(),
                embodiment=self._embodiment.build_embodiment_metadata(),
                runtime=RuntimeMetadata(
                    upstream_roboarena_commit=UPSTREAM_ROBOARENA_COMMIT,
                    transport="websocket+msgpack",
                    compression="none",
                    concurrency_model="single_rollout_per_process",
                ),
                validation=self._policy.build_validation_metadata(),
                notes=self._policy.build_notes() + self._embodiment.build_notes(),
            )

            for step_index in range(config.max_steps):
                obs = self._embodiment.build_observation(
                    server_metadata,
                    config.prompt,
                    session_id=config.session_id if bool(server_metadata.get("needs_session_id", False)) else None,
                )
                request_bytes = len(msgpack_numpy.packb({"endpoint": "infer", **obs}))
                start_time = time.perf_counter()
                action_dict = self._policy.infer(obs)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                response_bytes = len(msgpack_numpy.packb(action_dict))
                actions = action_dict["actions"]
                if not isinstance(actions, np.ndarray):
                    raise TypeError("Policy response must contain a NumPy array under the 'actions' key.")
                if actions.ndim != 2:
                    raise ValueError("Policy actions must be a rank-2 array shaped (N, D).")

                fairness_log.record_step(
                    step_index=step_index,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                    latency_ms=latency_ms,
                    action_rows=int(actions.shape[0]),
                    action_dim=int(actions.shape[1]),
                )
                self._embodiment.execute_action_chunk(actions, action_space=str(server_metadata["action_space"]))

            reset_payload = {"session_id": config.session_id, **config.reset_payload}
            self._policy.reset(reset_payload)
            fairness_log.finalize()

            run_dir = Path(config.output_dir) / config.run_id
            log_path = fairness_log.write(run_dir / "decision_log.json")
            return CurrentSchemaRunResult(
                log_path=log_path,
                session_id=config.session_id,
                server_metadata=dict(server_metadata),
            )
        finally:
            self._run_in_progress = False
