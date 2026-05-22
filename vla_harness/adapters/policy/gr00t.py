"""Managed-local-server GR00T adapter driven by official modality configs."""

from __future__ import annotations

import dataclasses
import importlib
import os
import subprocess
import time
from typing import Any
from typing import Mapping
from typing import Protocol
from typing import Sequence

import numpy as np

from vla_harness.adapters.policy.bimanual import BimanualPolicyAdapter
from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import ImagePreprocessMetadata
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata
from vla_harness.protocol.action import ActionChunk
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.action import ArmActionGroup
from vla_harness.protocol.action import PaddingRule
from vla_harness.protocol.manifest import ActionSemantics
from vla_harness.protocol.manifest import ActionStreamSpec
from vla_harness.protocol.manifest import ArmGroupSpec
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.manifest import LanguageFieldSpec
from vla_harness.protocol.manifest import StateStreamSpec
from vla_harness.protocol.manifest import VideoStreamSpec
from vla_harness.protocol.observation import ObservationPacket


class Gr00tClient(Protocol):
    def ping(self) -> bool:
        """Return whether the managed local server is alive."""

    def get_modality_config(self) -> Mapping[str, Any]:
        """Return the official modality config dict from the server."""

    def get_action(
        self,
        observation: Mapping[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        """Call the official GR00T server."""

    def reset(self, options: dict[str, Any] | None = None) -> Mapping[str, Any]:
        """Reset the official policy runtime."""

    def kill_server(self) -> None:
        """Stop the managed local server if the client owns it."""


@dataclasses.dataclass(slots=True, frozen=True)
class Gr00tVideoBinding:
    manifest_name: str
    gr00t_key: str
    role: str
    arm_group: str | None = None


@dataclasses.dataclass(slots=True, frozen=True)
class Gr00tStateBinding:
    manifest_name: str
    gr00t_key: str
    arm_group: str
    dim: int
    layout: str


@dataclasses.dataclass(slots=True, frozen=True)
class Gr00tActionBinding:
    manifest_name: str
    gr00t_key: str
    arm_group: str
    dim: int
    domain: str
    layout: str
    static_pad_strategy: str = "hold_static"


@dataclasses.dataclass(slots=True, frozen=True)
class Gr00tLanguageBinding:
    manifest_name: str
    gr00t_key: str


@dataclasses.dataclass(slots=True)
class Gr00tRuntimeConfig:
    checkpoint_ref: str
    embodiment_tag: str
    schema_source: str
    dtype: str = "float32"
    device: str = "cuda:0"
    runtime_family: str = "managed_local_server"
    host: str = "127.0.0.1"
    port: int = 5555
    timeout_ms: int = 15000
    strict: bool = False
    startup_timeout_s: float = 30.0
    startup_poll_interval_s: float = 0.5
    server_command: Sequence[str] | None = None
    env: Mapping[str, str] | None = None
    arm_groups: tuple[ArmGroupSpec, ArmGroupSpec] = (
        ArmGroupSpec(name="left_arm", side="left"),
        ArmGroupSpec(name="right_arm", side="right"),
    )
    video_bindings: tuple[Gr00tVideoBinding, ...] = ()
    state_bindings: tuple[Gr00tStateBinding, ...] = ()
    action_bindings: tuple[Gr00tActionBinding, ...] = ()
    language_binding: Gr00tLanguageBinding = Gr00tLanguageBinding(
        manifest_name="instruction",
        gr00t_key="annotation.human.task_description",
    )
    static_padding: Mapping[str, str] = dataclasses.field(default_factory=dict)


class GR00TPolicyAdapter(BimanualPolicyAdapter):
    """Generic GR00T adapter that keeps modality config as the source of truth."""

    def __init__(self, config: Gr00tRuntimeConfig, *, client: Gr00tClient | None = None) -> None:
        self._config = config
        self._client = client
        self._server_process: subprocess.Popen[str] | None = None
        self._modality_cache: Mapping[str, Any] | None = None

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

    def _connect(self) -> Gr00tClient:
        if self._client is not None:
            return self._client
        self._spawn_server_if_needed()
        policy_client_cls = _load_gr00t_policy_client()
        deadline = time.monotonic() + self._config.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                candidate = policy_client_cls(
                    host=self._config.host,
                    port=self._config.port,
                    timeout_ms=self._config.timeout_ms,
                    strict=self._config.strict,
                )
                if candidate.ping():
                    self._client = candidate
                    return candidate
                last_error = RuntimeError("GR00T server did not respond to ping.")
            except Exception as exc:  # pragma: no cover - startup retry
                last_error = exc
            time.sleep(self._config.startup_poll_interval_s)
        raise RuntimeError(
            f"Failed to connect to GR00T managed local server at {self._config.host}:{self._config.port}"
        ) from last_error

    def _modality_configs(self) -> Mapping[str, Any]:
        if self._modality_cache is None:
            self._modality_cache = self._connect().get_modality_config()
        return self._modality_cache

    def assert_ready_for_benchmark(self) -> None:
        client = self._connect()
        if not client.ping():
            raise RuntimeError("GR00T client could not verify server health via ping().")
        modality_configs = self._modality_configs()
        for required_key in ("video", "state", "action", "language"):
            if required_key not in modality_configs:
                raise RuntimeError(f"GR00T modality config missing required key {required_key!r}.")
        _assert_binding_keys(modality_configs["video"], [binding.gr00t_key for binding in self._config.video_bindings], "video")
        _assert_binding_keys(modality_configs["state"], [binding.gr00t_key for binding in self._config.state_bindings], "state")
        _assert_binding_keys(modality_configs["action"], [binding.gr00t_key for binding in self._config.action_bindings], "action")
        _assert_binding_keys(modality_configs["language"], [self._config.language_binding.gr00t_key], "language")

    def build_manifest(self) -> HarnessManifest:
        modality_configs = self._modality_configs()
        video_config = modality_configs["video"]
        state_config = modality_configs["state"]
        action_config = modality_configs["action"]
        language_config = modality_configs["language"]

        action_horizon = len(tuple(action_config.delta_indices))
        action_config_by_key = {
            key: action_cfg for key, action_cfg in zip(action_config.modality_keys, action_config.action_configs or [])
        }

        return HarnessManifest(
            name=f"gr00t_{self._config.embodiment_tag.lower()}",
            version="phase4",
            arm_groups=self._config.arm_groups,
            video_streams=tuple(
                VideoStreamSpec(
                    name=binding.manifest_name,
                    role=binding.role,
                    arm_group=binding.arm_group,
                    sample_indices=tuple(video_config.delta_indices),
                    order_index=index,
                )
                for index, binding in enumerate(self._config.video_bindings)
            ),
            state_streams=tuple(
                StateStreamSpec(
                    name=binding.manifest_name,
                    arm_group=binding.arm_group,
                    dim=binding.dim,
                    layout=binding.layout,
                    sample_indices=tuple(state_config.delta_indices),
                )
                for binding in self._config.state_bindings
            ),
            action_streams=tuple(
                ActionStreamSpec(
                    name=binding.manifest_name,
                    arm_group=binding.arm_group,
                    dim=binding.dim,
                    semantics=ActionSemantics(
                        representation=_coerce_action_representation(action_config_by_key[binding.gr00t_key]),
                        domain=binding.domain,
                        layout=binding.layout,
                        static_pad_strategy=binding.static_pad_strategy,
                    ),
                    horizon=action_horizon,
                )
                for binding in self._config.action_bindings
            ),
            language_fields=(LanguageFieldSpec(name=self._config.language_binding.manifest_name),),
            metadata={
                "runtime_family": self._config.runtime_family,
                "schema_source": self._config.schema_source,
                "embodiment_tag": self._config.embodiment_tag,
                "video_keys": tuple(video_config.modality_keys),
                "state_keys": tuple(state_config.modality_keys),
                "action_keys": tuple(action_config.modality_keys),
                "language_keys": tuple(language_config.modality_keys),
            },
        )

    def infer(self, observation: ObservationPacket) -> ActionPacket:
        manifest = self.build_manifest()
        observation.validate_against(manifest)
        payload = _observation_to_gr00t_payload(observation, self._config)
        action_dict, info = self._connect().get_action(payload, options=None)

        arms: dict[str, ArmActionGroup] = {}
        for binding in self._config.action_bindings:
            action_array = np.asarray(action_dict[binding.gr00t_key], dtype=np.float32)
            if action_array.ndim == 3:
                if action_array.shape[0] != 1:
                    raise ValueError(
                        f"GR00T action batch dimension must be 1 for harness inference. Got {action_array.shape!r} "
                        f"for key {binding.gr00t_key!r}."
                    )
                action_array = action_array[0]
            if action_array.ndim != 2 or action_array.shape[1] != binding.dim:
                raise ValueError(
                    f"GR00T action stream {binding.gr00t_key!r} must be shaped (T, {binding.dim}). "
                    f"Got {action_array.shape!r}."
                )
            arms.setdefault(binding.arm_group, ArmActionGroup(streams={}))
            arms[binding.arm_group].streams[binding.manifest_name] = ActionChunk(action_array)

        padding = {
            arm_name: PaddingRule(strategy=strategy, reason="single-arm bridge on bimanual embodiment")
            for arm_name, strategy in self._config.static_padding.items()
        }
        packet = ActionPacket(arms=arms, padding=padding, metadata={"info": dict(info)})
        packet.validate_against(manifest)
        return packet

    def reset(self, reset_info: dict[str, Any]) -> Any:
        return self._connect().reset(reset_info or None)

    def build_policy_metadata(self) -> PolicyMetadata:
        action_horizon = len(tuple(self._modality_configs()["action"].delta_indices))
        return PolicyMetadata(
            family="gr00t",
            config_name=self._config.embodiment_tag,
            checkpoint_ref=self._config.checkpoint_ref,
            checkpoint_sha256=None,
            checkpoint_sha256_explanation="Checkpoint hashing is not wired for GR00T managed-server checkpoints.",
            dtype=self._config.dtype,
            device=self._config.device,
            action_space="gr00t_managed_server",
            chunk_size=action_horizon,
            prompt_format_source=self._config.schema_source,
            image_preprocess=ImagePreprocessMetadata(
                resize_resolution=None,
                resize_filter="official_gr00t_runtime",
                color_space="official_gr00t_runtime",
                output_dtype="official_gr00t_runtime",
            ),
            runtime_family=self._config.runtime_family,
            schema_source=self._config.schema_source,
            normalization_tag=self._config.embodiment_tag,
        )

    def build_validation_metadata(self) -> ValidationMetadata:
        return ValidationMetadata()

    def build_notes(self) -> list[DecisionNote]:
        notes = [
            DecisionNote(
                topic="policy.runtime_family",
                choice=self._config.runtime_family,
                status="official",
                rationale="GR00T ships an official managed local server path through PolicyServer/PolicyClient.",
                evidence="Isaac-GR00T/getting_started/policy.md",
            ),
            DecisionNote(
                topic="policy.schema_source",
                choice=self._config.schema_source,
                status="official",
                rationale="GR00T modality config files define the observation/action schema and action semantics.",
                evidence=self._config.schema_source,
            ),
            DecisionNote(
                topic="policy.embodiment_tag",
                choice=self._config.embodiment_tag,
                status="official",
                rationale="GR00T requires an explicit embodiment tag to load the correct modality configuration and normalization.",
                evidence="Isaac-GR00T/getting_started/policy.md",
            ),
        ]
        for arm_name, strategy in self._config.static_padding.items():
            notes.append(
                DecisionNote(
                    topic=f"policy.static_padding.{arm_name}",
                    choice=strategy,
                    status="benchmark_default",
                    rationale="This is the one allowed bridge for running a single-arm policy on a bimanual setup: keep the other arm explicitly static.",
                    evidence="User-scoped harness policy for bimanual-only evaluation.",
                )
            )
        return notes

    def close(self) -> None:
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                self._server_process.kill()
                self._server_process.wait(timeout=5)
            self._server_process = None
        if self._client is not None and hasattr(self._client, "kill_server") and self._config.server_command is not None:
            try:
                self._client.kill_server()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


def _load_gr00t_policy_client():
    module = importlib.import_module("gr00t.policy.server_client")
    return module.PolicyClient


def _assert_binding_keys(modality_config: Any, expected_keys: list[str], modality_name: str) -> None:
    available = set(modality_config.modality_keys)
    missing = [key for key in expected_keys if key not in available]
    if missing:
        raise RuntimeError(
            f"GR00T {modality_name} bindings reference keys not present in the official modality config: {missing!r}. "
            f"Available keys: {sorted(available)!r}"
        )


def _coerce_action_representation(action_config: Any) -> str:
    rep = getattr(action_config, "rep", None)
    value = getattr(rep, "value", rep)
    if value is None:
        return "other"
    rep_text = str(value).lower()
    if rep_text == "delta":
        return "relative"
    if rep_text in {"relative", "absolute", "velocity"}:
        return rep_text
    return "other"


def _observation_to_gr00t_payload(observation: ObservationPacket, config: Gr00tRuntimeConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {"video": {}, "state": {}, "language": {}}

    for binding in config.video_bindings:
        sequence = observation.video[binding.manifest_name]
        frames = np.stack(sequence.frames, axis=0).astype(np.uint8)
        payload["video"][binding.gr00t_key] = np.expand_dims(frames, axis=0)

    for binding in config.state_bindings:
        sequence = observation.arms[binding.arm_group].streams[binding.manifest_name]
        values = np.stack(sequence.values, axis=0).astype(np.float32)
        payload["state"][binding.gr00t_key] = np.expand_dims(values, axis=0)

    instruction = observation.language[config.language_binding.manifest_name]
    payload["language"][config.language_binding.gr00t_key] = [[instruction]]
    return payload
