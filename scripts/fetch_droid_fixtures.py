"""Fetch a small number of real DROID frames for the OpenPI fidelity tests.

This script pulls the ``lerobot/droid_100`` dataset from HuggingFace
(MIT licensed, publicly downloadable, ~464MB) and saves N frames into the
two fixture corpora the fidelity tests consume:

- ``fixtures/openpi_preprocess/frame_NNN.npy``       (single images)
- ``fixtures/openpi_action/obs_NNN.npz``             (full observation dicts)

Run it once after installing the optional dependency ``lerobot``::

    uv pip install lerobot
    python scripts/fetch_droid_fixtures.py --num-frames 5

The script intentionally pins the dataset revision so the fixtures are
reproducible across runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np


DEFAULT_REPO_ID = "lerobot/droid_100"
DEFAULT_NUM_FRAMES = 5
DEFAULT_OUTPUT_DIR = Path("fixtures")


def _ensure_lerobot() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - import guard
            print(
                "lerobot is not installed. Install it with `uv pip install lerobot` "
                "(or `pip install lerobot`) and re-run.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
    return LeRobotDataset


def _frame_to_uint8_image(value: Any) -> np.ndarray:
    """Coerce a lerobot frame field into an HxWx3 uint8 numpy array."""

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = (array * 255.0).clip(0, 255)
        array = array.astype(np.uint8)
    return array


def _build_observation(frame: dict[str, Any]) -> dict[str, Any]:
    """Map a lerobot DROID frame into the flat current schema."""

    exterior_left = _frame_to_uint8_image(frame["observation.images.exterior_image_1_left"])
    wrist_left = _frame_to_uint8_image(frame["observation.images.wrist_image_left"])

    state = np.asarray(frame["observation.state"], dtype=np.float32)
    if state.size != 7:
        raise ValueError(f"Expected a 7-D DROID state, got shape {state.shape}.")

    prompt = frame.get("language_instruction") or frame.get("task") or "follow the language instruction"
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")

    return {
        "observation/exterior_image_1_left": exterior_left,
        "observation/wrist_image_left": wrist_left,
        "observation/joint_position": state.astype(np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": state[-1:].astype(np.float32),
        "prompt": str(prompt),
    }


def fetch_fixtures(repo_id: str, num_frames: int, output_dir: Path) -> None:
    LeRobotDataset = _ensure_lerobot()
    logging.info("Loading dataset %s", repo_id)
    dataset = LeRobotDataset(repo_id)

    preprocess_dir = output_dir / "openpi_preprocess"
    action_dir = output_dir / "openpi_action"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, len(dataset) // max(1, num_frames * 2))
    selected = 0
    for dataset_index in range(0, len(dataset), stride):
        if selected >= num_frames:
            break
        frame = dataset[dataset_index]
        try:
            obs = _build_observation(frame)
        except KeyError as exc:
            logging.warning("Skipping index %d, missing key: %s", dataset_index, exc)
            continue

        frame_id = f"{selected:03d}"
        np.save(preprocess_dir / f"frame_{frame_id}.npy", obs["observation/exterior_image_1_left"])
        np.savez(
            action_dir / f"obs_{frame_id}.npz",
            **{key: value for key, value in obs.items() if key != "prompt"},
            prompt=np.array(obs["prompt"], dtype=object),
        )
        selected += 1
        logging.info("Saved fixture %s (dataset idx %d)", frame_id, dataset_index)

    if selected == 0:
        raise RuntimeError("No frames were saved. Inspect the dataset schema and adjust _build_observation.")

    print(f"Wrote {selected} fixtures into {preprocess_dir} and {action_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace dataset id (default: %(default)s)")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of fixtures to save (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root for fixtures (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fetch_fixtures(args.repo_id, args.num_frames, args.output_dir)


if __name__ == "__main__":
    main()
