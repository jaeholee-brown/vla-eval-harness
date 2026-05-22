"""Thin current-schema adapter for official openpi runtimes."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence

import numpy as np

from roboarena.policy_client import WebsocketClientPolicy
from vla_harness.adapters.policy.base import CurrentSchemaPolicyAdapter
from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import ImagePreprocessMetadata
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata


Observation = dict[str, Any]
ActionDict = dict[str, Any]
ClientFactory = Callable[[str, int], WebsocketClientPolicy]
OracleCallable = Callable[[Observation], ActionDict]
PreprocessCallable = Callable[[np.ndarray], np.ndarray]


def identity_preprocess(image: np.ndarray) -> np.ndarray:
    """Fallback image preprocessor when the official path is not wired yet."""

    return image


@dataclasses.dataclass(slots=True)
class OpenPIRuntimeConfig:
    config_name: str = "pi05_droid"
    checkpoint_ref: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_sha256_explanation: str | None = "Checkpoint hashing is not wired yet."
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "bfloat16"
    device: str = "cuda"
    action_space: str | None = None
    chunk_size: int | None = None
    prompt_format_source: str | None = "official_openpi_runtime"
    image_resize_resolution: tuple[int, int] | None = (224, 224)
    image_resize_filter: str | None = "official_openpi_runtime"
    image_color_space: str | None = "rgb"
    image_output_dtype: str | None = "uint8"
    server_command: Sequence[str] | None = None
    startup_timeout_s: float = 20.0
    startup_poll_interval_s: float = 0.25
    env: Mapping[str, str] | None = None
    preprocessing_oracle_name: str | None = None
    action_oracle_name: str | None = None


class OpenPICurrentSchemaAdapter(CurrentSchemaPolicyAdapter):
    """Current-schema adapter that wraps an official openpi runtime."""

    def __init__(
        self,
        config: OpenPIRuntimeConfig,
        *,
        client_factory: ClientFactory | None = None,
        preprocess_callable: PreprocessCallable | None = None,
        oracle_callable: OracleCallable | None = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory or (lambda host, port: WebsocketClientPolicy(host=host, port=port))
        self._preprocess_callable = preprocess_callable or identity_preprocess
        self._oracle_callable = oracle_callable
        self._client: WebsocketClientPolicy | None = None
        self._server_process: subprocess.Popen[str] | None = None

    def _spawn_server_if_needed(self) -> None:
        if self._server_process is not None or self._config.server_command is None:
            return

        env = os.environ.copy()
        if self._config.env is not None:
            env.update(self._config.env)
        self._server_process = subprocess.Popen(
            list(self._config.server_command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

    def _connect(self) -> WebsocketClientPolicy:
        if self._client is not None:
            return self._client

        self._spawn_server_if_needed()
        deadline = time.monotonic() + self._config.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._client = self._client_factory(self._config.host, self._config.port)
                return self._client
            except Exception as exc:  # pragma: no cover - retry path
                last_error = exc
                time.sleep(self._config.startup_poll_interval_s)

        raise RuntimeError(
            f"Failed to connect to openpi runtime at {self._config.host}:{self._config.port}"
        ) from last_error

    def get_server_metadata(self) -> dict[str, Any]:
        return self._connect().get_server_metadata()

    def infer(self, obs: Observation) -> ActionDict:
        return self._connect().infer(obs)

    def reset(self, reset_info: dict[str, Any]) -> Any:
        return self._connect().reset(reset_info)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        return self._preprocess_callable(image)

    def oracle_predict(self, obs: Observation) -> ActionDict:
        if self._oracle_callable is None:
            raise RuntimeError("No direct openpi oracle callable configured for this adapter.")
        return self._oracle_callable(obs)

    def build_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            family="openpi",
            config_name=self._config.config_name,
            checkpoint_ref=self._config.checkpoint_ref,
            checkpoint_sha256=self._config.checkpoint_sha256,
            checkpoint_sha256_explanation=self._config.checkpoint_sha256_explanation,
            dtype=self._config.dtype,
            device=self._config.device,
            action_space=self._config.action_space,
            chunk_size=self._config.chunk_size,
            prompt_format_source=self._config.prompt_format_source,
            image_preprocess=ImagePreprocessMetadata(
                resize_resolution=self._config.image_resize_resolution,
                resize_filter=self._config.image_resize_filter,
                color_space=self._config.image_color_space,
                output_dtype=self._config.image_output_dtype,
            ),
        )

    def build_validation_metadata(self) -> ValidationMetadata:
        return ValidationMetadata(
            preprocessing_oracle=self._config.preprocessing_oracle_name,
            action_oracle=self._config.action_oracle_name,
        )

    def build_notes(self) -> list[DecisionNote]:
        runtime_choice = "spawn_local_server_command" if self._config.server_command is not None else "connect_existing_server"
        return [
            DecisionNote(
                topic="policy.runtime_path",
                choice=runtime_choice,
                status="official" if self._config.server_command is not None else "adapter",
                rationale="Phase 1 keeps openpi on its official runtime path whenever possible.",
                evidence="Current phase-1 architecture and openpi remote-inference pattern.",
            ),
            DecisionNote(
                topic="policy.current_schema_scope",
                choice="single_arm_flat_schema_only",
                status="scoped_out",
                rationale="Phase 1 intentionally targets only the current flat schema bootstrap path.",
                evidence="docs/spikes/current-schema-gap-matrix.md",
            ),
        ]

    def close(self) -> None:
        if self._client is not None:
            websocket = getattr(self._client, "_ws", None)
            if websocket is not None:
                websocket.close()
            self._client = None

        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                self._server_process.kill()
                self._server_process.wait(timeout=5)
            self._server_process = None
