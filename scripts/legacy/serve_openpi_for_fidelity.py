"""Serve pi05_droid (or any openpi config) with deterministic-noise support.

The stock openpi websocket server calls ``Policy.infer(obs)`` with no
``noise=`` kwarg, so every call samples a fresh flow-matching initial
noise from the policy's RNG. That makes apples-to-apples action parity
impossible across two independent processes (the in-process Policy used
by ``official_action`` and the server-side Policy used by
``harness_action``) — both observe the same input but their RNG states
diverge after the first call.

This script wraps the loaded Policy so that, when a request includes a
``noise`` key (numpy array), the noise tensor is popped off the obs and
passed to ``Policy.infer(obs, noise=...)``. With identical noise on both
legs, action parity becomes a real test of "is the websocket path
faithfully producing the same output as the in-process path."

Usage (drop-in replacement for openpi's ``serve_policy.py``):

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
        uv run python scripts/serve_openpi_for_fidelity.py \
            --config pi05_droid --checkpoint-dir gs://openpi-assets/checkpoints/pi05_droid \
            --port 8000
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import download
from openpi.training import config as _config


class _NoiseAwarePolicy:
    """Thin wrapper around openpi Policy that lifts ``noise`` out of obs."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        noise = None
        if isinstance(obs, dict) and "noise" in obs:
            obs = dict(obs)
            noise = obs.pop("noise")
        return self._inner.infer(obs, noise=noise)

    def reset(self, reset_info: dict[str, Any]) -> Any:
        return self._inner.reset(reset_info) if hasattr(self._inner, "reset") else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_droid", help="openpi training config name")
    parser.add_argument(
        "--checkpoint-dir",
        default="gs://openpi-assets/checkpoints/pi05_droid",
        help="local path or gs:// URI to the checkpoint",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = _config.get_config(args.config)
    ckpt = download.maybe_download(args.checkpoint_dir)
    policy = policy_config.create_trained_policy(cfg, ckpt)
    wrapped = _NoiseAwarePolicy(policy)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host=args.host,
        port=args.port,
        metadata={"config": args.config, "checkpoint_dir": args.checkpoint_dir, "noise_aware": True},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
