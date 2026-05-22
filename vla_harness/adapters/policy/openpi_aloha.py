"""True-bimanual openpi ALOHA policy adapter (pi0_aloha_pen_uncap)."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import time
from typing import Any
from typing import Callable
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
from vla_harness.protocol.manifest import ActionSemantics
from vla_harness.protocol.manifest import ActionStreamSpec
from vla_harness.protocol.manifest import ArmGroupSpec
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.manifest import LanguageFieldSpec
from vla_harness.protocol.manifest import StateStreamSpec
from vla_harness.protocol.manifest import VideoStreamSpec
from vla_harness.protocol.observation import ObservationPacket


# Upstream defaults sourced from openpi (Physical-Intelligence/openpi @ main):
#   - src/openpi/training/config.py: TrainConfig entry "pi0_aloha_pen_uncap"
#   - src/openpi/models/pi0_config.py: Pi0Config.action_horizon = 50
#   - src/openpi/policies/aloha_policy.py: AlohaInputs / AlohaOutputs
#   - docs/remote_inference.md, examples/aloha_real/{main,env}.py
DEFAULT_CONFIG_NAME = "pi0_aloha_pen_uncap"
DEFAULT_CHECKPOINT_REF = "gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap"
DEFAULT_PROMPT = "uncap the pen"
DEFAULT_PROMPT_FORMAT_SOURCE = (
    "openpi/src/openpi/training/config.py::pi0_aloha_pen_uncap.default_prompt"
)
DEFAULT_SCHEMA_SOURCE = "openpi/src/openpi/policies/aloha_policy.py::AlohaInputs"
DEFAULT_RUNTIME_FAMILY = "openpi_websocket_server"
DEFAULT_ACTION_HORIZON = 50  # Pi0Config().action_horizon
DEFAULT_STATE_DIM_PER_ARM = 7  # 6 joints + 1 gripper
DEFAULT_TOTAL_STATE_DIM = 14
DEFAULT_IMAGE_RESOLUTION = (224, 224)
DEFAULT_CAMERA_ROLES: tuple[tuple[str, str, str | None], ...] = (
    # (manifest_name, openpi_key, arm_group_attachment)
    ("cam_high", "cam_high", None),
    ("cam_left_wrist", "cam_left_wrist", "left_arm"),
    ("cam_right_wrist", "cam_right_wrist", "right_arm"),
)
# AlohaInputs in openpi accepts an optional "cam_low" key; pi0_aloha_pen_uncap
# does not use it, so we omit it from the manifest and let the server's
# AlohaInputs fill a black image with a zero image_mask, per upstream behavior.


class OpenPIAlohaClient(Protocol):
    """Subset of openpi WebsocketClientPolicy used by the adapter."""

    def get_server_metadata(self) -> Mapping[str, Any]:
        """Return the policy server's metadata payload."""

    def infer(self, obs: dict[str, Any]) -> Mapping[str, Any]:
        """Run one policy step over the openpi websocket transport."""

    def reset(self, reset_info: dict[str, Any]) -> Any:
        """Reset the policy server's per-episode state."""


@dataclasses.dataclass(slots=True)
class OpenPIAlohaRuntimeConfig:
    """All upstream-derived defaults for pi0_aloha_pen_uncap."""

    config_name: str = DEFAULT_CONFIG_NAME
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF
    checkpoint_sha256: str | None = None
    checkpoint_sha256_explanation: str | None = (
        "openpi serves checkpoints from gs://openpi-assets; upstream does not publish per-checkpoint sha256."
    )
    default_prompt: str = DEFAULT_PROMPT
    prompt_format_source: str = DEFAULT_PROMPT_FORMAT_SOURCE
    schema_source: str = DEFAULT_SCHEMA_SOURCE
    runtime_family: str = DEFAULT_RUNTIME_FAMILY
    action_horizon: int = DEFAULT_ACTION_HORIZON
    state_dim_per_arm: int = DEFAULT_STATE_DIM_PER_ARM
    image_resolution: tuple[int, int] = DEFAULT_IMAGE_RESOLUTION
    image_resize_filter: str = "official_openpi_image_tools.resize_with_pad"
    image_color_space: str = "rgb"
    image_output_dtype: str = "uint8"
    image_channel_order: str = "CHW"  # AlohaInputs converts [C,H,W] -> [H,W,C] server-side
    adapt_to_pi: bool = True  # AlohaInputs / AlohaOutputs default in upstream
    dtype: str = "bfloat16"
    device: str = "cuda"
    host: str = "127.0.0.1"
    port: int = 8000
    server_command: Sequence[str] | None = None
    env: Mapping[str, str] | None = None
    startup_timeout_s: float = 30.0
    startup_poll_interval_s: float = 0.5


class OpenPIAlohaPolicyAdapter(BimanualPolicyAdapter):
    """Bimanual adapter that drives an official openpi server for pi0_aloha_pen_uncap."""

    def __init__(
        self,
        config: OpenPIAlohaRuntimeConfig | None = None,
        *,
        client: OpenPIAlohaClient | None = None,
        client_factory: Callable[[str, int], OpenPIAlohaClient] | None = None,
    ) -> None:
        self._config = config or OpenPIAlohaRuntimeConfig()
        self._client = client
        self._client_factory = client_factory
        self._server_process: subprocess.Popen[str] | None = None
        self._metadata_cache: Mapping[str, Any] | None = None

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

    def _connect(self) -> OpenPIAlohaClient:
        if self._client is not None:
            return self._client
        self._spawn_server_if_needed()
        factory = self._client_factory or _default_websocket_client_factory
        deadline = time.monotonic() + self._config.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._client = factory(self._config.host, self._config.port)
                return self._client
            except Exception as exc:  # pragma: no cover - startup retry
                last_error = exc
                time.sleep(self._config.startup_poll_interval_s)
        raise RuntimeError(
            f"Failed to connect to openpi ALOHA runtime at {self._config.host}:{self._config.port}"
        ) from last_error

    def assert_ready_for_benchmark(self) -> None:
        metadata = self._server_metadata()
        # The openpi server is free to return an opaque metadata dict, so we
        # only enforce shape invariants we *can* verify from upstream code.
        if not isinstance(metadata, Mapping):
            raise RuntimeError(
                f"openpi server metadata must be a mapping; got {type(metadata).__name__}."
            )
        if self._config.action_horizon <= 0:
            raise RuntimeError("action_horizon must be positive.")
        if self._config.state_dim_per_arm * 2 != DEFAULT_TOTAL_STATE_DIM:
            raise RuntimeError(
                "ALOHA state must be 14-D (left 7 + right 7) per openpi AlohaInputs."
            )

    def _server_metadata(self) -> Mapping[str, Any]:
        if self._metadata_cache is None:
            self._metadata_cache = self._connect().get_server_metadata()
        return self._metadata_cache

    def build_manifest(self) -> HarnessManifest:
        video_streams = tuple(
            VideoStreamSpec(
                name=manifest_name,
                role=manifest_name,
                arm_group=arm_group,
                order_index=index,
            )
            for index, (manifest_name, _openpi_key, arm_group) in enumerate(DEFAULT_CAMERA_ROLES)
        )
        return HarnessManifest(
            name="openpi_aloha_pen_uncap",
            version="phase4",
            arm_groups=(
                ArmGroupSpec(name="left_arm", side="left"),
                ArmGroupSpec(name="right_arm", side="right"),
            ),
            video_streams=video_streams,
            state_streams=(
                StateStreamSpec(
                    name="joint_position",
                    arm_group="left_arm",
                    dim=self._config.state_dim_per_arm,
                    layout="joint_plus_gripper",
                ),
                StateStreamSpec(
                    name="joint_position",
                    arm_group="right_arm",
                    dim=self._config.state_dim_per_arm,
                    layout="joint_plus_gripper",
                ),
            ),
            action_streams=(
                ActionStreamSpec(
                    name="joint_position",
                    arm_group="left_arm",
                    dim=self._config.state_dim_per_arm,
                    semantics=ActionSemantics(
                        representation="absolute",
                        domain="joint",
                        layout="joint_plus_gripper",
                        static_pad_strategy="hold_static",
                    ),
                    horizon=self._config.action_horizon,
                ),
                ActionStreamSpec(
                    name="joint_position",
                    arm_group="right_arm",
                    dim=self._config.state_dim_per_arm,
                    semantics=ActionSemantics(
                        representation="absolute",
                        domain="joint",
                        layout="joint_plus_gripper",
                        static_pad_strategy="hold_static",
                    ),
                    horizon=self._config.action_horizon,
                ),
            ),
            language_fields=(LanguageFieldSpec(name="instruction"),),
            metadata={
                "runtime_family": self._config.runtime_family,
                "schema_source": self._config.schema_source,
                "config_name": self._config.config_name,
                "checkpoint_ref": self._config.checkpoint_ref,
                "camera_keys": tuple(openpi_key for _name, openpi_key, _arm in DEFAULT_CAMERA_ROLES),
                "adapt_to_pi": self._config.adapt_to_pi,
                "image_channel_order": self._config.image_channel_order,
                "image_resolution": self._config.image_resolution,
            },
        )

    def infer(self, observation: ObservationPacket) -> ActionPacket:
        manifest = self.build_manifest()
        observation.validate_against(manifest)

        images: dict[str, np.ndarray] = {}
        for manifest_name, openpi_key, _arm in DEFAULT_CAMERA_ROLES:
            frame_hwc = observation.video[manifest_name].frames[-1]
            images[openpi_key] = _to_openpi_image(frame_hwc, self._config.image_channel_order)

        left_state = observation.arms["left_arm"].streams["joint_position"].values[-1]
        right_state = observation.arms["right_arm"].streams["joint_position"].values[-1]
        if left_state.shape != (self._config.state_dim_per_arm,):
            raise ValueError(
                f"left_arm joint_position must be shape ({self._config.state_dim_per_arm},); got {left_state.shape!r}"
            )
        if right_state.shape != (self._config.state_dim_per_arm,):
            raise ValueError(
                f"right_arm joint_position must be shape ({self._config.state_dim_per_arm},); got {right_state.shape!r}"
            )

        instruction = observation.language.get("instruction", self._config.default_prompt)

        payload: dict[str, Any] = {
            "images": images,
            "state": np.concatenate([left_state, right_state]).astype(np.float32),
            "prompt": instruction,
        }
        response = self._connect().infer(payload)
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != DEFAULT_TOTAL_STATE_DIM:
            raise ValueError(
                f"openpi ALOHA actions must be shaped (H, 14); got {actions.shape!r}."
            )

        per_arm = self._config.state_dim_per_arm
        packet = ActionPacket(
            arms={
                "left_arm": ArmActionGroup(
                    streams={"joint_position": ActionChunk(actions[:, :per_arm].copy())}
                ),
                "right_arm": ArmActionGroup(
                    streams={"joint_position": ActionChunk(actions[:, per_arm:].copy())}
                ),
            },
            metadata={
                "policy_timing": dict(response.get("policy_timing", {})),
                "checkpoint_ref": self._config.checkpoint_ref,
                "config_name": self._config.config_name,
            },
        )
        packet.validate_against(manifest)
        return packet

    def reset(self, reset_info: dict[str, Any]) -> Any:
        return self._connect().reset(dict(reset_info))

    def build_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            family="openpi",
            config_name=self._config.config_name,
            checkpoint_ref=self._config.checkpoint_ref,
            checkpoint_sha256=self._config.checkpoint_sha256,
            checkpoint_sha256_explanation=self._config.checkpoint_sha256_explanation,
            dtype=self._config.dtype,
            device=self._config.device,
            action_space="absolute_joint_pose_bimanual",
            chunk_size=self._config.action_horizon,
            prompt_format_source=self._config.prompt_format_source,
            image_preprocess=ImagePreprocessMetadata(
                resize_resolution=self._config.image_resolution,
                resize_filter=self._config.image_resize_filter,
                color_space=self._config.image_color_space,
                output_dtype=self._config.image_output_dtype,
            ),
            runtime_family=self._config.runtime_family,
            schema_source=self._config.schema_source,
            normalization_tag=self._config.config_name,
        )

    def build_validation_metadata(self) -> ValidationMetadata:
        return ValidationMetadata()

    def build_notes(self) -> list[DecisionNote]:
        return [
            DecisionNote(
                topic="policy.runtime_family",
                choice=self._config.runtime_family,
                status="official",
                rationale="openpi ships an official websocket policy server via scripts/serve_policy.py.",
                evidence="openpi/docs/remote_inference.md",
            ),
            DecisionNote(
                topic="policy.config_name",
                choice=self._config.config_name,
                status="official",
                rationale="pi0_aloha_pen_uncap is the upstream TrainConfig entry for this checkpoint.",
                evidence="openpi/src/openpi/training/config.py",
            ),
            DecisionNote(
                topic="policy.checkpoint_ref",
                choice=self._config.checkpoint_ref,
                status="official",
                rationale="Upstream README pins this gs:// path as the published pi0_aloha_pen_uncap checkpoint.",
                evidence="openpi/README.md",
            ),
            DecisionNote(
                topic="policy.prompt_format_source",
                choice=self._config.prompt_format_source,
                status="official",
                rationale="The training config sets default_prompt='uncap the pen' for this checkpoint.",
                evidence="openpi/src/openpi/training/config.py",
            ),
            DecisionNote(
                topic="policy.chunk_size",
                choice=str(self._config.action_horizon),
                status="official",
                rationale="Pi0Config().action_horizon = 50 is the upstream default chunk size for pi0.",
                evidence="openpi/src/openpi/models/pi0_config.py",
            ),
            DecisionNote(
                topic="policy.camera_keys",
                choice=",".join(openpi_key for _n, openpi_key, _a in DEFAULT_CAMERA_ROLES),
                status="official",
                rationale="pi0_aloha_pen_uncap repack maps cam_high / cam_left_wrist / cam_right_wrist; AlohaInputs treats missing cam_low as masked-off.",
                evidence="openpi/src/openpi/training/config.py + openpi/src/openpi/policies/aloha_policy.py",
            ),
            DecisionNote(
                topic="policy.state_layout",
                choice="left(6 joints + 1 gripper) || right(6 joints + 1 gripper) = 14",
                status="official",
                rationale="AlohaInputs documents state as 14-D with per-arm joint_plus_gripper layout.",
                evidence="openpi/src/openpi/policies/aloha_policy.py",
            ),
            DecisionNote(
                topic="policy.action_semantics",
                choice="absolute_joint_pose_bimanual",
                status="official",
                rationale="AlohaOutputs returns the first 14 dims and undoes the pi gripper mapping; the targets are absolute joint+gripper poses.",
                evidence="openpi/src/openpi/policies/aloha_policy.py",
            ),
            DecisionNote(
                topic="policy.image_preprocess",
                choice=f"{self._config.image_resize_filter}@{self._config.image_resolution[0]}x{self._config.image_resolution[1]}",
                status="official",
                rationale="openpi remote_inference.md prescribes resize_with_pad + uint8 at 224x224; AlohaInputs handles CHW->HWC server-side.",
                evidence="openpi/docs/remote_inference.md + openpi/examples/aloha_real/env.py",
            ),
            DecisionNote(
                topic="policy.adapt_to_pi",
                choice=str(self._config.adapt_to_pi),
                status="official",
                rationale="AlohaInputs(adapt_to_pi=True) is the upstream default and aligns joint signs and gripper space with pi0.",
                evidence="openpi/src/openpi/policies/aloha_policy.py",
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


def _default_websocket_client_factory(host: str, port: int) -> OpenPIAlohaClient:
    from roboarena.policy_client import WebsocketClientPolicy

    return WebsocketClientPolicy(host=host, port=port)


def _to_openpi_image(frame_hwc: np.ndarray, channel_order: str) -> np.ndarray:
    """Convert a harness HWC uint8 frame into the layout AlohaInputs expects."""

    if frame_hwc.ndim != 3 or frame_hwc.shape[-1] != 3:
        raise ValueError(f"ALOHA image frames must be shaped (H, W, 3); got {frame_hwc.shape!r}.")
    if channel_order == "CHW":
        return np.transpose(frame_hwc, (2, 0, 1)).astype(np.uint8, copy=False)
    if channel_order == "HWC":
        return frame_hwc.astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image_channel_order {channel_order!r}; expected 'CHW' or 'HWC'.")
