"""Probe the current flat schema against GR00T's official DROID surface.

This is a phase-2 spike artifact, not a production adapter.

It uses official GR00T code for:

- the DROID modality config
- the ZeroMQ PolicyServer / PolicyClient transport

and compares that surface against the harness's current flat-schema bootstrap
contract.

Usage:

    python scripts/spike_gr00t_current_schema.py

Optional:

    python scripts/spike_gr00t_current_schema.py --write-json /tmp/gr00t_spike.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np


DEFAULT_GR00T_REPO = Path("/tmp/vla_sources/Isaac-GR00T")
DEFAULT_PORT = 5565


def _repo_commit(repo_root: Path) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True)
        .strip()
    )


def _load_gr00t(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(repo_root))
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.types import ActionRepresentation
    from gr00t.policy.policy import BasePolicy
    from gr00t.policy.server_client import PolicyClient
    from gr00t.policy.server_client import PolicyServer

    return MODALITY_CONFIGS, ActionRepresentation, BasePolicy, (PolicyClient, PolicyServer)


def _make_current_flat_observation() -> dict[str, Any]:
    return {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "pick up the object",
    }


def _action_stream_dims() -> dict[str, int]:
    return {
        "eef_9d": 9,
        "gripper_position": 1,
        "joint_position": 7,
    }


def _run_transport_probe(modality_config: dict[str, Any], base_policy_cls: Any, transport_classes: tuple[Any, Any], port: int) -> dict[str, Any]:
    PolicyClient, PolicyServer = transport_classes
    dims = _action_stream_dims()

    class ProbePolicy(base_policy_cls):
        def __init__(self) -> None:
            super().__init__(strict=False)

        def check_observation(self, observation: dict[str, Any]) -> None:
            del observation

        def check_action(self, action: dict[str, Any]) -> None:
            del action

        def _get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None):
            del observation, options
            horizon = len(modality_config["action"].delta_indices)
            action = {
                key: np.zeros((1, horizon, dims[key]), dtype=np.float32)
                for key in modality_config["action"].modality_keys
            }
            return action, {}

        def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
            del options
            return {"status": "reset"}

        def get_modality_config(self) -> dict[str, Any]:
            return modality_config

    server = PolicyServer(ProbePolicy(), host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.5)

    client = PolicyClient(host="127.0.0.1", port=port, strict=False)
    try:
        roundtrip_cfg = client.get_modality_config()
        ping_ok = client.ping()
        reset_ok = client.reset()
    finally:
        try:
            client.kill_server()
        except Exception:
            pass
        time.sleep(0.1)

    return {
        "ping_ok": ping_ok,
        "reset_response": reset_ok,
        "video_keys": list(roundtrip_cfg["video"].modality_keys),
        "state_keys": list(roundtrip_cfg["state"].modality_keys),
        "action_keys": list(roundtrip_cfg["action"].modality_keys),
        "video_horizon": len(roundtrip_cfg["video"].delta_indices),
        "state_horizon": len(roundtrip_cfg["state"].delta_indices),
        "action_horizon": len(roundtrip_cfg["action"].delta_indices),
    }


def run_spike(repo_root: Path, port: int) -> dict[str, Any]:
    modality_configs, ActionRepresentation, base_policy_cls, transport_classes = _load_gr00t(repo_root)
    modality_config = modality_configs["oxe_droid_relative_eef_relative_joint"]
    transport = _run_transport_probe(modality_config, base_policy_cls, transport_classes, port)
    current_flat_obs = _make_current_flat_observation()

    action_configs = modality_config["action"].action_configs
    mixed_reps = [cfg.rep.value if hasattr(cfg.rep, "value") else str(cfg.rep) for cfg in action_configs]

    result = {
        "source": {
            "repo_root": str(repo_root),
            "commit": _repo_commit(repo_root),
            "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
        },
        "transport_probe": {
            "protocol": "zmq+msgpack_numpy",
            "roundtrip_ok": transport["ping_ok"],
            "reset_ok": transport["reset_response"] == {"status": "reset"},
            "official_video_keys": transport["video_keys"],
            "official_state_keys": transport["state_keys"],
            "official_action_keys": transport["action_keys"],
            "official_video_horizon": transport["video_horizon"],
            "official_state_horizon": transport["state_horizon"],
            "official_action_horizon": transport["action_horizon"],
        },
        "current_flat_schema": {
            "video_keys": [key for key in current_flat_obs if "image" in key],
            "state_keys": [
                "observation/joint_position",
                "observation/cartesian_position",
                "observation/gripper_position",
            ],
            "action_enum": [
                "joint_position",
                "joint_velocity",
                "cartesian_position",
                "cartesian_velocity",
            ],
        },
        "fit_analysis": {
            "video": {
                "rename_only": False,
                "required_horizon": transport["video_horizon"],
                "current_horizon": 1,
                "required_keys": transport["video_keys"],
                "available_keys": ["observation/exterior_image_1_left", "observation/wrist_image_left"],
                "required_decisions": [
                    "duplicate or buffer frames to satisfy GR00T video horizon 2 from a single current-schema frame",
                    "rename flat keys into nested modality names",
                ],
            },
            "state": {
                "rename_only": False,
                "required_keys": transport["state_keys"],
                "available_keys": [
                    "observation/cartesian_position",
                    "observation/gripper_position",
                    "observation/joint_position",
                ],
                "required_decisions": [
                    "derive eef_9d from cartesian_position(6) using an explicit pose convention and rotation conversion",
                    "keep gripper_position and joint_position as separate named state streams instead of one flat vector",
                ],
                "hard_mismatch": "GR00T expects eef_9d (9D), but the current flat schema only exposes cartesian_position(6) with no embedded rotation-6D convention",
            },
            "action": {
                "rename_only": False,
                "required_keys": transport["action_keys"],
                "required_horizon": transport["action_horizon"],
                "mixed_action_representations": mixed_reps,
                "required_decisions": [
                    "preserve per-stream semantics instead of flattening into one current-schema action array",
                    "encode relative EEF, absolute gripper, and relative joint semantics explicitly",
                ],
                "hard_mismatch": "Current flat schema exposes a single 7D/8D chunk plus one action_space enum; GR00T DROID expects 17D over named streams with mixed relative/absolute semantics and 40-step horizon",
            },
            "language": {
                "rename_only": True,
                "required_key": "annotation.language.language_instruction",
                "available_key": "prompt",
            },
        },
        "phase2_failures": [
            "transport mismatch: GR00T official server is ZeroMQ REQ/REP, not websocket",
            "payload mismatch: official observation is nested by modality with explicit temporal horizons",
            "semantic mismatch: current flat action enum cannot represent mixed relative/absolute action streams",
            "state mismatch: eef_9d is not present in the current flat schema and requires a benchmark-side derivation rule",
        ],
    }

    # Defensive assertion so the spike fails loudly if the official DROID config changes.
    assert transport["action_horizon"] == 40, transport
    assert transport["video_keys"] == ["exterior_image_1_left", "wrist_image_left"], transport
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-repo-root", type=Path, default=DEFAULT_GR00T_REPO)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--write-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_spike(args.gr00t_repo_root, args.port)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
