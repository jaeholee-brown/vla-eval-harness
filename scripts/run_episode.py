"""Combine one policy adapter with one embodiment adapter and run one episode.

Usage examples:

  # Pure dry-run on CPU — no server, no robot. Verifies the wiring only.
  python scripts/run_episode.py \\
      --policy molmoact2_yam --embodiment fake --dry-run \\
      --prompt "pack the blocks" --max-steps 3

  # Live policy + fake embodiment. You bring the upstream server; no robot needed.
  python scripts/run_episode.py \\
      --policy molmoact2_yam --server-url http://127.0.0.1:8202/act \\
      --embodiment fake --prompt "pack the blocks"

  # Full live. You bring the upstream server AND a real robot. The
  # --backend-loader points at a Python callable that returns a backend
  # implementing YAMBimanualBackend (yam) or DK1BimanualBackend-equivalent (dk1).
  python scripts/run_episode.py \\
      --policy molmoact2_yam --server-url http://127.0.0.1:8202/act \\
      --embodiment yam --backend-loader my_robot_setup:build_yam_backend \\
      --prompt "pack the blocks" --max-steps 50

  # Dump every tweakable config field + default for the selected combo.
  python scripts/run_episode.py --list-configs

GR00T is intentionally not exposed via this launcher: its bindings depend on
the chosen embodiment_tag and must be copied verbatim from the upstream
modality config. Use the cookbook + `vla_harness.adapters.policy.gr00t`
directly for that.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vla_harness.adapters.embodiment.bimanual import BimanualEmbodimentAdapter
from vla_harness.adapters.embodiment.yam_bimanual import (
    YAMBimanualAdapter,
    YAMBimanualBackend,
    YAMBimanualConfig,
)
from vla_harness.adapters.embodiment.dk1_bimanual import (
    DK1BimanualAdapter,
    DK1BimanualConfig,
)
from vla_harness.adapters.policy.bimanual import BimanualPolicyAdapter
from vla_harness.adapters.policy.molmoact2_yam import (
    MolmoAct2YAMPolicyAdapter,
    MolmoAct2YAMRuntimeConfig,
)
from vla_harness.adapters.policy.openpi_aloha import (
    OpenPIAlohaPolicyAdapter,
    OpenPIAlohaRuntimeConfig,
)
from vla_harness.logging.decision_log import (
    DecisionNote,
    EmbodimentMetadata,
)
from vla_harness.protocol import (
    ActionChunk,
    ActionPacket,
    ArmActionGroup,
    ArmObservationGroup,
    HarnessManifest,
    ObservationPacket,
    TemporalStateSequence,
    TemporalVideoSequence,
)
from vla_harness.runner import BimanualRunConfig, BimanualRunner


POLICY_CHOICES = ("openpi_aloha", "molmoact2_yam")
EMBODIMENT_CHOICES = ("fake", "yam", "dk1")


def build_policy(args: argparse.Namespace) -> BimanualPolicyAdapter:
    if args.policy == "molmoact2_yam":
        kwargs = _kwargs_from(args, {"server_url": "server_url", "device": "device", "dtype": "dtype"})
        config = MolmoAct2YAMRuntimeConfig(**kwargs)
        client = _FakeMolmoAct2Client() if args.dry_run else None
        return MolmoAct2YAMPolicyAdapter(config, client=client)

    if args.policy == "openpi_aloha":
        kwargs = _kwargs_from(args, {"host": "host", "port": "port", "device": "device", "dtype": "dtype"})
        config = OpenPIAlohaRuntimeConfig(**kwargs)
        client = _FakeOpenPIAlohaClient() if args.dry_run else None
        return OpenPIAlohaPolicyAdapter(config, client=client)

    raise ValueError(f"Unknown policy {args.policy!r}")


def build_embodiment(args: argparse.Namespace, manifest: HarnessManifest) -> BimanualEmbodimentAdapter:
    if args.embodiment == "fake":
        return FakeBimanualEmbodiment(manifest)

    if args.embodiment == "yam":
        backend = _load_backend(args.backend_loader, kind="yam")
        kwargs = _kwargs_from(args, {"control_hz": "control_hz"})
        return YAMBimanualAdapter(backend=backend, config=YAMBimanualConfig(**kwargs))

    if args.embodiment == "dk1":
        backend = _load_backend(args.backend_loader, kind="dk1")
        kwargs = _kwargs_from(args, {"control_hz": "control_hz"})
        return DK1BimanualAdapter(backend=backend, config=DK1BimanualConfig(**kwargs))

    raise ValueError(f"Unknown embodiment {args.embodiment!r}")


def _kwargs_from(args: argparse.Namespace, mapping: Mapping[str, str]) -> dict[str, Any]:
    """Forward CLI args to a dataclass only when the user actually set them."""
    out: dict[str, Any] = {}
    for arg_attr, config_field in mapping.items():
        value = getattr(args, arg_attr, None)
        if value is not None:
            out[config_field] = value
    return out


def _load_backend(spec: str | None, *, kind: str) -> Any:
    if spec is None:
        raise SystemExit(
            f"--embodiment {kind} requires --backend-loader module:function pointing at a callable "
            f"that returns a real {kind} backend. Pass --embodiment fake if you don't have hardware."
        )
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise SystemExit(f"--backend-loader must be 'module:function', got {spec!r}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory()


class FakeBimanualEmbodiment(BimanualEmbodimentAdapter):
    """Generic in-memory embodiment driven by the policy's own manifest.

    Returns zero-filled observations sized to the manifest and accepts any
    ActionPacket. Useful for proving the policy + runner chain without a robot.
    """

    def __init__(self, manifest: HarnessManifest) -> None:
        self._manifest = manifest
        self._executed: list[ActionPacket] = []

    def prepare_episode(self, manifest: HarnessManifest, prompt: str) -> None:
        del manifest, prompt

    def capture_observation(
        self,
        manifest: HarnessManifest,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> ObservationPacket:
        video: dict[str, TemporalVideoSequence] = {}
        for spec in manifest.video_streams:
            frames = tuple(np.zeros((224, 224, 3), dtype=np.uint8) for _ in spec.sample_indices)
            video[spec.name] = TemporalVideoSequence(sample_indices=spec.sample_indices, frames=frames)

        arms: dict[str, ArmObservationGroup] = {}
        for arm_spec in manifest.arm_groups:
            streams: dict[str, TemporalStateSequence] = {}
            for stream_spec in manifest.state_streams_for_arm(arm_spec.name):
                values = tuple(np.zeros(stream_spec.dim, dtype=np.float32) for _ in stream_spec.sample_indices)
                streams[stream_spec.name] = TemporalStateSequence(
                    sample_indices=stream_spec.sample_indices,
                    values=values,
                )
            arms[arm_spec.name] = ArmObservationGroup(streams=streams)

        language = {field.name: prompt for field in manifest.language_fields}

        return ObservationPacket(
            video=video,
            arms=arms,
            language=language,
            session_id=session_id,
        )

    def execute_action(self, manifest: HarnessManifest, action: ActionPacket) -> None:
        del manifest
        self._executed.append(action)

    def build_embodiment_metadata(self) -> EmbodimentMetadata:
        arm_names = tuple(spec.name for spec in self._manifest.arm_groups)
        camera_roles = tuple(spec.role for spec in self._manifest.video_streams)
        return EmbodimentMetadata(
            family="fake_bimanual",
            backend="in_memory_zero_fill",
            active_arm=None,
            parked_arm=None,
            parked_arm_rule="hold_static",
            control_hz=None,
            chunk_consumption_policy="record_only",
            arm_group_names=arm_names,
            camera_roles=camera_roles,
            camera_role_source="scripts/run_episode.py::FakeBimanualEmbodiment",
        )

    def build_notes(self) -> list[DecisionNote]:
        return [
            DecisionNote(
                topic="embodiment.dry_run",
                choice="fake_in_memory",
                status="adapter",
                rationale="Fake embodiment used to dry-run the policy + runner chain without hardware.",
                evidence="scripts/run_episode.py",
            )
        ]

    def close(self) -> None:
        return None


class _FakeMolmoAct2Client:
    """Returns zero actions sized to the upstream-default chunk + 14-D state."""

    def health(self) -> dict[str, Any]:
        return {
            "repo_id": "allenai/MolmoAct2-BimanualYAM",
            "norm_tag": "yam_dual_molmoact2",
            "num_cameras": 3,
            "state_dim": 14,
        }

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return {"actions": np.zeros((10, 14), dtype=np.float32), "dt_ms": 0.0}


class _FakeOpenPIAlohaClient:
    """Returns zero actions sized to the upstream-default ALOHA 50x14 chunk."""

    def get_server_metadata(self) -> Mapping[str, Any]:
        return {"config_name": "pi0_aloha_pen_uncap"}

    def infer(self, obs: dict[str, Any]) -> Mapping[str, Any]:
        del obs
        return {"actions": np.zeros((50, 14), dtype=np.float32), "policy_timing": {"infer_ms": 0.0}}

    def reset(self, reset_info: dict[str, Any]) -> Any:
        del reset_info
        return {"status": "ok"}


def list_configs() -> None:
    print("Available policy adapters and their config fields:\n")
    for name, cls in (("openpi_aloha", OpenPIAlohaRuntimeConfig), ("molmoact2_yam", MolmoAct2YAMRuntimeConfig)):
        print(f"  --policy {name}  ({cls.__module__}.{cls.__name__})")
        _print_fields(cls)
        print()

    print("Available embodiment adapters and their config fields:\n")
    for name, cls in (("yam", YAMBimanualConfig), ("dk1", DK1BimanualConfig)):
        print(f"  --embodiment {name}  ({cls.__module__}.{cls.__name__})")
        _print_fields(cls)
        print()

    print("Run-level fields (always available):")
    print("  --prompt STR              episode language instruction")
    print("  --max-steps INT           number of policy steps to run (default 1)")
    print("  --output-dir PATH         fairness log directory (default runs/)")
    print("  --dry-run                 swap policy client + use fake embodiment")
    print("  --backend-loader M:F      factory for the real yam/dk1 backend")
    print()
    print("Anything not exposed as a CLI flag can be overridden by writing a small")
    print("Python wrapper that constructs the adapter config directly. See README.")


def _print_fields(cls: type) -> None:
    for field in dataclasses.fields(cls):
        default = field.default
        if default is dataclasses.MISSING:
            default_repr = "<required>"
        elif callable(default):
            default_repr = "<default_factory>"
        else:
            default_repr = repr(default)
        print(f"      {field.name}: {field.type} = {default_repr}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one bimanual episode by combining a policy adapter with an embodiment adapter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--policy", choices=POLICY_CHOICES, help="Which policy adapter to run.")
    parser.add_argument(
        "--embodiment",
        choices=EMBODIMENT_CHOICES,
        default="fake",
        help="Which embodiment adapter to run (default: fake = no robot).",
    )
    parser.add_argument("--prompt", help="Language instruction passed to the policy.")
    parser.add_argument("--max-steps", type=int, default=1, help="Number of policy steps to run.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Where the fairness log is written.")

    parser.add_argument("--dry-run", action="store_true",
                        help="Use in-memory fake clients/backends; no server, no robot needed.")
    parser.add_argument("--list-configs", action="store_true",
                        help="Print every tweakable config field with its default and exit.")

    policy_group = parser.add_argument_group("policy knobs (forwarded to the adapter config)")
    policy_group.add_argument("--server-url", help="HTTP server URL (molmoact2_yam).")
    policy_group.add_argument("--host", help="Policy server host (openpi_aloha).")
    policy_group.add_argument("--port", type=int, help="Policy server port (openpi_aloha).")
    policy_group.add_argument("--device", help="Policy device, e.g. cuda or cpu.")
    policy_group.add_argument("--dtype", help="Policy dtype, e.g. bfloat16 or float32.")

    embodiment_group = parser.add_argument_group("embodiment knobs")
    embodiment_group.add_argument("--control-hz", type=float, help="Control loop Hz for yam/dk1.")
    embodiment_group.add_argument(
        "--backend-loader",
        help="module:function that returns a real yam/dk1 backend. Required unless --dry-run or --embodiment fake.",
    )

    args = parser.parse_args(argv)

    if args.list_configs:
        list_configs()
        return 0

    if args.policy is None:
        parser.error("--policy is required (or use --list-configs)")
    if args.prompt is None:
        parser.error("--prompt is required to run an episode")

    policy = build_policy(args)
    manifest = policy.build_manifest()
    embodiment = build_embodiment(args, manifest)

    runner = BimanualRunner(policy, embodiment)
    result = runner.run_episode(
        BimanualRunConfig(
            prompt=args.prompt,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
        )
    )

    print(f"manifest: {result.manifest_name}")
    print(f"session : {result.session_id}")
    print(f"log     : {result.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
