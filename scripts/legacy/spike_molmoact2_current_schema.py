"""Probe the current flat schema against MolmoAct2's official FastAPI apps.

This is a phase-2 spike artifact, not a production adapter.

It imports the official DROID and YAM server modules, builds the FastAPI apps
with stub policies, and drives them through TestClient so the findings are
anchored to the real request/response code paths without loading model weights.

Usage:

    python scripts/spike_molmoact2_current_schema.py

Optional:

    python scripts/spike_molmoact2_current_schema.py --write-json /tmp/molmoact2_spike.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import torch


DEFAULT_MOLMOACT2_REPO = Path("/tmp/vla_sources/molmoact2")


def _repo_commit(repo_root: Path) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True)
        .strip()
    )


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_current_flat_observation() -> dict[str, Any]:
    return {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "pick up the object",
    }


class _StubModel:
    dtype = torch.float32


class _StubDroidPolicy:
    device = "cpu"
    default_cuda_graph = False
    model = _StubModel()

    def predict(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = 10,
        enable_cuda_graph: bool = False,
    ) -> np.ndarray:
        del external_cam, wrist_cam, instruction, enable_cuda_graph
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (8,):
            raise ValueError(f"state must be shape (8,), got {state_f32.shape}")
        return np.zeros((num_steps, 8), dtype=np.float32)


class _StubYamPolicy:
    device = "cpu"
    default_cuda_graph = False
    model = _StubModel()

    def predict(
        self,
        top_cam: np.ndarray,
        left_cam: np.ndarray,
        right_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = 10,
        enable_cuda_graph: bool = False,
    ) -> np.ndarray:
        del top_cam, left_cam, right_cam, instruction, enable_cuda_graph
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (14,):
            raise ValueError(f"state must be shape (14,), got {state_f32.shape}")
        return np.zeros((num_steps, 14), dtype=np.float32)


def run_spike(repo_root: Path) -> dict[str, Any]:
    droid_module = _load_module(
        "molmoact2_droid_server",
        repo_root / "examples" / "droid" / "host_server_droid.py",
    )
    yam_module = _load_module(
        "molmoact2_yam_server",
        repo_root / "examples" / "yam" / "host_server_yam.py",
    )

    droid_app = droid_module.build_app(_StubDroidPolicy())
    yam_app = yam_module.build_app(_StubYamPolicy())
    droid_client = TestClient(droid_app)
    yam_client = TestClient(yam_app)

    current_flat = _make_current_flat_observation()
    droid_payload = {
        "external_cam": current_flat["observation/exterior_image_1_left"],
        "wrist_cam": current_flat["observation/wrist_image_left"],
        "instruction": current_flat["prompt"],
        "state": np.concatenate(
            [
                current_flat["observation/joint_position"],
                current_flat["observation/gripper_position"],
            ]
        ).astype(np.float32),
    }
    yam_payload = {
        "top_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        "left_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        "right_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        "instruction": current_flat["prompt"],
        "state": np.zeros(14, dtype=np.float32),
    }

    droid_health = droid_client.get("/act")
    droid_ok = droid_client.post("/act", content=droid_module.json_numpy.dumps(droid_payload))
    droid_flat_fail = droid_client.post("/act", content=droid_module.json_numpy.dumps(current_flat))

    yam_health = yam_client.get("/act")
    yam_ok = yam_client.post("/act", content=yam_module.json_numpy.dumps(yam_payload))
    yam_flat_fail = yam_client.post("/act", content=yam_module.json_numpy.dumps(current_flat))

    result = {
        "source": {
            "repo_root": str(repo_root),
            "commit": _repo_commit(repo_root),
        },
        "droid_server": {
            "repo_id": droid_module.REPO_ID,
            "norm_tag": droid_module.NORM_TAG,
            "default_num_steps": droid_module.DEFAULT_NUM_STEPS,
            "health_status_code": droid_health.status_code,
            "health_body": droid_health.json(),
            "mapped_current_schema_status_code": droid_ok.status_code,
            "mapped_current_schema_response_keys": sorted(droid_module.json_numpy.loads(droid_ok.text).keys()),
            "raw_current_schema_status_code": droid_flat_fail.status_code,
            "raw_current_schema_error": droid_module.json_numpy.loads(droid_flat_fail.text)["error"],
        },
        "yam_server": {
            "repo_id": yam_module.REPO_ID,
            "norm_tag": yam_module.NORM_TAG,
            "state_dim": yam_module.STATE_DIM,
            "num_cameras": yam_module.NUM_CAMERAS,
            "health_status_code": yam_health.status_code,
            "health_body": yam_health.json(),
            "official_payload_status_code": yam_ok.status_code,
            "official_payload_response_keys": sorted(yam_module.json_numpy.loads(yam_ok.text).keys()),
            "raw_current_schema_status_code": yam_flat_fail.status_code,
            "raw_current_schema_error": yam_module.json_numpy.loads(yam_flat_fail.text)["error"],
        },
        "fit_analysis": {
            "droid": {
                "rename_only": False,
                "easy_mappings": [
                    "exterior_image_1_left -> external_cam",
                    "wrist_image_left -> wrist_cam",
                    "prompt -> instruction",
                    "concat(joint_position, gripper_position) -> state(8)",
                ],
                "adapter_local_choices": [
                    "drop cartesian_position because the official server does not consume it",
                    "speak HTTP+json_numpy instead of websocket+msgpack",
                ],
            },
            "yam": {
                "rename_only": False,
                "hard_mismatches": [
                    "official server requires three named cameras in fixed order [top, left, right]",
                    "official server requires state(14,) for two 7-D arms",
                    "current flat schema only carries one arm plus one wrist and one exterior camera",
                ],
                "dishonest_mapping_required_for_current_schema": [
                    "invent a top/left/right camera role mapping from wrist/exterior cameras",
                    "duplicate or pad camera streams",
                    "pad or synthesize the missing 7-D arm state",
                ],
                "checkpoint_specific_runtime_branching": [
                    "DROID server calls predict_action(..., action_mode='continuous')",
                    "YAM server calls predict_action(..., inference_action_mode='continuous')",
                ],
            },
        },
        "phase2_failures": [
            "transport mismatch: MolmoAct2 official deployment is HTTP+json_numpy, not websocket",
            "schema mismatch: YAM requires true bimanual state and three ordered cameras",
            "runtime mismatch: per-checkpoint server code differs at the predict_action call signature",
            "fairness-metadata gap: norm_tag and camera order are official deployment defaults that the current fairness log cannot name explicitly",
        ],
    }

    assert result["droid_server"]["mapped_current_schema_status_code"] == 200, result
    assert result["yam_server"]["official_payload_status_code"] == 200, result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molmoact2-repo-root", type=Path, default=DEFAULT_MOLMOACT2_REPO)
    parser.add_argument("--write-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_spike(args.molmoact2_repo_root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
