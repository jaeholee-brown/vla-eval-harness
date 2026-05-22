"""Bimanual MolmoAct2 adapter anchored to the official YAM FastAPI server."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess
import time
from typing import Any
from typing import Mapping
from typing import Protocol
from urllib import request

import numpy as np

from vla_harness.adapters.policy.bimanual import BimanualPolicyAdapter
from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import ImagePreprocessMetadata
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import ValidationMetadata
from vla_harness.protocol.action import ActionChunk
from vla_harness.protocol.action import ActionPacket
from vla_harness.protocol.action import ArmActionGroup
from vla_harness.protocol.manifest import ActionSemantics
from vla_harness.protocol.manifest import ActionStreamSpec
from vla_harness.protocol.manifest import ArmGroupSpec
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.manifest import LanguageFieldSpec
from vla_harness.protocol.manifest import StateStreamSpec
from vla_harness.protocol.manifest import VideoStreamSpec
from vla_harness.protocol.observation import ObservationPacket


DEFAULT_REPO_ID = "allenai/MolmoAct2-BimanualYAM"
DEFAULT_NORM_TAG = "yam_dual_molmoact2"
DEFAULT_SCHEMA_SOURCE = "examples/yam/host_server_yam.py"
DEFAULT_CAMERA_ORDER = ("top", "left", "right")


class MolmoAct2ActClient(Protocol):
    def health(self) -> dict[str, Any]:
        """Return the `/act` GET payload from the official server."""

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the `/act` POST endpoint using the official request schema."""


class JsonNumpyHttpActClient(MolmoAct2ActClient):
    """Thin HTTP client for the official MolmoAct2 FastAPI server."""

    def __init__(self, server_url: str) -> None:
        self._server_url = server_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        with request.urlopen(self._server_url, timeout=15) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        import json_numpy

        body = json_numpy.dumps(payload).encode("utf-8")
        req = request.Request(
            self._server_url,
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=60) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
        return json_numpy.loads(raw)


@dataclasses.dataclass(slots=True)
class MolmoAct2YAMRuntimeConfig:
    server_url: str = "http://127.0.0.1:8202/act"
    repo_id: str = DEFAULT_REPO_ID
    normalization_tag: str = DEFAULT_NORM_TAG
    schema_source: str = DEFAULT_SCHEMA_SOURCE
    dtype: str = "bfloat16"
    device: str = "cuda"
    default_chunk_size: int = 10
    state_dim: int = 14
    camera_order: tuple[str, str, str] = DEFAULT_CAMERA_ORDER
    startup_timeout_s: float = 20.0
    startup_poll_interval_s: float = 0.25
    server_command: tuple[str, ...] | None = None
    env: Mapping[str, str] | None = None


class MolmoAct2YAMPolicyAdapter(BimanualPolicyAdapter):
    """Policy adapter for the official MolmoAct2-BimanualYAM FastAPI server."""

    def __init__(
        self,
        config: MolmoAct2YAMRuntimeConfig,
        *,
        client: MolmoAct2ActClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or JsonNumpyHttpActClient(config.server_url)
        self._server_process: subprocess.Popen[str] | None = None
        self._health_cache: dict[str, Any] | None = None

    def _spawn_server_if_needed(self) -> None:
        if self._server_process is not None or self._config.server_command is None:
            return
        env = None
        if self._config.env is not None:
            env = {**dict(self._config.env)}
        self._server_process = subprocess.Popen(
            list(self._config.server_command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

    def _health(self) -> dict[str, Any]:
        if self._health_cache is not None:
            return self._health_cache
        self._spawn_server_if_needed()
        deadline = time.monotonic() + self._config.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._health_cache = self._client.health()
                return self._health_cache
            except Exception as exc:  # pragma: no cover - startup retry
                last_error = exc
                time.sleep(self._config.startup_poll_interval_s)
        raise RuntimeError(f"Failed to reach MolmoAct2 YAM server at {self._config.server_url}") from last_error

    def assert_ready_for_benchmark(self) -> None:
        health = self._health()
        required_pairs = {
            "repo_id": self._config.repo_id,
            "norm_tag": self._config.normalization_tag,
            "num_cameras": len(self._config.camera_order),
            "state_dim": self._config.state_dim,
        }
        for key, expected in required_pairs.items():
            if health.get(key) != expected:
                raise RuntimeError(
                    f"MolmoAct2 YAM server health mismatch for {key!r}: "
                    f"expected {expected!r}, got {health.get(key)!r}"
                )

    def build_manifest(self) -> HarnessManifest:
        return HarnessManifest(
            name="molmoact2_bimanual_yam",
            version="phase4",
            arm_groups=(
                ArmGroupSpec(name="left_arm", side="left"),
                ArmGroupSpec(name="right_arm", side="right"),
            ),
            video_streams=(
                VideoStreamSpec(name="top_camera", role="top", order_index=0),
                VideoStreamSpec(name="left_camera", role="left", arm_group="left_arm", order_index=1),
                VideoStreamSpec(name="right_camera", role="right", arm_group="right_arm", order_index=2),
            ),
            state_streams=(
                StateStreamSpec(name="joint_position", arm_group="left_arm", dim=7, layout="joint_plus_gripper"),
                StateStreamSpec(name="joint_position", arm_group="right_arm", dim=7, layout="joint_plus_gripper"),
            ),
            action_streams=(
                ActionStreamSpec(
                    name="joint_position",
                    arm_group="left_arm",
                    dim=7,
                    semantics=ActionSemantics(
                        representation="absolute",
                        domain="joint",
                        layout="joint_plus_gripper",
                        static_pad_strategy="hold_static",
                    ),
                ),
                ActionStreamSpec(
                    name="joint_position",
                    arm_group="right_arm",
                    dim=7,
                    semantics=ActionSemantics(
                        representation="absolute",
                        domain="joint",
                        layout="joint_plus_gripper",
                        static_pad_strategy="hold_static",
                    ),
                ),
            ),
            language_fields=(LanguageFieldSpec(name="instruction"),),
            metadata={
                "runtime_family": "fastapi+json_numpy",
                "schema_source": self._config.schema_source,
                "repo_id": self._config.repo_id,
                "normalization_tag": self._config.normalization_tag,
                "camera_order": self._config.camera_order,
            },
        )

    def infer(self, observation: ObservationPacket) -> ActionPacket:
        manifest = self.build_manifest()
        observation.validate_against(manifest)
        payload = {
            "top_cam": observation.video["top_camera"].frames[-1],
            "left_cam": observation.video["left_camera"].frames[-1],
            "right_cam": observation.video["right_camera"].frames[-1],
            "instruction": observation.language["instruction"],
            "state": np.concatenate(
                [
                    observation.arms["left_arm"].streams["joint_position"].values[-1],
                    observation.arms["right_arm"].streams["joint_position"].values[-1],
                ]
            ).astype(np.float32),
        }
        response = self._client.act(payload)
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"MolmoAct2 YAM actions must be shaped (N, 14). Got {actions.shape!r}")
        return ActionPacket(
            arms={
                "left_arm": ArmActionGroup(
                    streams={"joint_position": ActionChunk(actions[:, :7])}
                ),
                "right_arm": ArmActionGroup(
                    streams={"joint_position": ActionChunk(actions[:, 7:])}
                ),
            },
            metadata={
                "dt_ms": response.get("dt_ms"),
                "repo_id": self._config.repo_id,
                "normalization_tag": self._config.normalization_tag,
            },
        )

    def reset(self, reset_info: dict[str, Any]) -> dict[str, Any]:
        return {"status": "reset_not_required", "reset_info": dict(reset_info)}

    def build_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            family="molmoact2",
            config_name="MolmoAct2-BimanualYAM",
            checkpoint_ref=self._config.repo_id,
            checkpoint_sha256=None,
            checkpoint_sha256_explanation="Checkpoint hashing is not wired for remote Hugging Face snapshots.",
            dtype=self._config.dtype,
            device=self._config.device,
            action_space="absolute_joint_pose_bimanual",
            chunk_size=self._config.default_chunk_size,
            prompt_format_source="official_molmoact2_server",
            image_preprocess=ImagePreprocessMetadata(
                resize_resolution=None,
                resize_filter="official_molmoact2_server",
                color_space="official_molmoact2_server",
                output_dtype="official_molmoact2_server",
            ),
            runtime_family="fastapi+json_numpy",
            schema_source=self._config.schema_source,
            normalization_tag=self._config.normalization_tag,
        )

    def build_validation_metadata(self) -> ValidationMetadata:
        return ValidationMetadata()

    def build_notes(self) -> list[DecisionNote]:
        return [
            DecisionNote(
                topic="policy.schema_source",
                choice=self._config.schema_source,
                status="official",
                rationale="The official YAM FastAPI server defines the request/response schema and camera order.",
                evidence=self._config.schema_source,
            ),
            DecisionNote(
                topic="policy.camera_order",
                choice="top,left,right",
                status="official",
                rationale="The official MolmoAct2-BimanualYAM server requires ordered cameras top/left/right.",
                evidence=self._config.schema_source,
            ),
            DecisionNote(
                topic="policy.normalization_tag",
                choice=self._config.normalization_tag,
                status="official",
                rationale="The official server health payload reports the normalization tag for this checkpoint.",
                evidence=self._config.schema_source,
            ),
        ]

    def close(self) -> None:
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                self._server_process.kill()
                self._server_process.wait(timeout=5)
            self._server_process = None
