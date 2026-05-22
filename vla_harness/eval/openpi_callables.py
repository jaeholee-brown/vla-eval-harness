"""Callables wired up for the OpenPI fidelity parity tests.

These are the functions the fidelity tests load via env vars of the form
``module:function``. They are imported lazily so the module can be referenced
from environment variables even when the rest of the harness does not import
``openpi`` (for example during unit-test runs).

Required environment variables when running the fidelity suite:

- ``OPENPI_CONFIG_NAME``        e.g. ``pi05_droid``
- ``OPENPI_CHECKPOINT_DIR``     local path or ``gs://`` URI accepted by
                                ``openpi.shared.download.maybe_download``

Optional:

- ``OPENPI_HARNESS_HOST``       host for the harness adapter's websocket
                                policy server (default ``127.0.0.1``)
- ``OPENPI_HARNESS_PORT``       port for the harness adapter's websocket
                                policy server (default ``8000``)
- ``OPENPI_NEGATIVE_CONTROL``   strategy for the negative control callable.
                                One of ``swap_rgb`` (default), ``zero_image``,
                                ``shuffle_prompt``.

The five callables exported here:

- ``official_preprocess(image) -> np.ndarray``
- ``harness_preprocess(image) -> np.ndarray``
- ``official_action(obs) -> {"actions": ...}``
- ``harness_action(obs) -> {"actions": ...}``
- ``negative_control_action(obs) -> {"actions": ...}``
"""

from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"Environment variable {name!r} is required to use vla_harness.eval.openpi_callables."
        )
    return value


@functools.lru_cache(maxsize=1)
def _official_policy() -> Any:
    """Construct the official openpi Policy once and reuse it.

    Importing openpi is intentionally deferred to call time so unit tests can
    reference this module without the openpi package installed.
    """

    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    config_name = _require_env("OPENPI_CONFIG_NAME")
    checkpoint_dir_str = _require_env("OPENPI_CHECKPOINT_DIR")

    train_config = _config.get_config(config_name)
    checkpoint_dir = download.maybe_download(checkpoint_dir_str)
    return policy_config.create_trained_policy(train_config, checkpoint_dir)


@functools.lru_cache(maxsize=1)
def _harness_adapter() -> Any:
    """Construct the harness OpenPI adapter once and reuse it."""

    from vla_harness.adapters.policy.openpi_current_schema import OpenPICurrentSchemaAdapter
    from vla_harness.adapters.policy.openpi_current_schema import OpenPIRuntimeConfig

    config_name = _require_env("OPENPI_CONFIG_NAME")
    host = os.environ.get("OPENPI_HARNESS_HOST", "127.0.0.1")
    port = int(os.environ.get("OPENPI_HARNESS_PORT", "8000"))

    adapter = OpenPICurrentSchemaAdapter(
        OpenPIRuntimeConfig(
            config_name=config_name,
            host=host,
            port=port,
            image_resize_filter="official_openpi_runtime",
            image_color_space="official_openpi_runtime",
            image_output_dtype="official_openpi_runtime",
            preprocessing_oracle_name="vla_harness.eval.openpi_callables:official_preprocess",
            action_oracle_name="vla_harness.eval.openpi_callables:official_action",
        ),
        preprocess_callable=_run_image_through_official_transforms,
    )
    adapter.assert_ready_for_benchmark()
    return adapter


def _run_image_through_official_transforms(image: np.ndarray) -> np.ndarray:
    """Send a single image through the official openpi input pipeline.

    Implementation note: openpi's ``Policy`` class does not expose a public
    method that runs the input transforms in isolation; the transforms live
    on the private ``_input_transform`` attribute. We use that attribute
    directly. If openpi renames it, change the attribute reference below.
    """

    policy = _official_policy()
    minimal_obs: dict[str, Any] = {
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/cartesian_position": np.zeros(6, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "noop",
    }
    transformed = policy._input_transform(minimal_obs)  # noqa: SLF001 - intentional internal access
    for key in ("observation/exterior_image_1_left", "image", "images", "observation/image"):
        if key in transformed:
            return np.asarray(transformed[key])
    raise RuntimeError(
        "Could not locate a preprocessed image in the openpi input transform output; "
        "inspect policy._input_transform(...) for the actual key name on your config."
    )


def official_preprocess(image: np.ndarray) -> np.ndarray:
    """Run the official openpi preprocessing pipeline on a single image."""

    return _run_image_through_official_transforms(image)


def harness_preprocess(image: np.ndarray) -> np.ndarray:
    """Run the harness adapter's preprocessing on a single image."""

    return _harness_adapter().preprocess_image(image)


def official_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Call openpi's Policy.infer directly, with no harness involvement."""

    return _official_policy().infer(obs)


def harness_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Call the harness OpenPI adapter, which routes through the websocket."""

    return _harness_adapter().infer(obs)


def negative_control_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Deliberately perturbed call into the official policy.

    The point of this callable is to prove the action-parity battery can
    distinguish a faithful path from a broken one. The perturbation is
    intentionally large enough that any working parity test must flag it.

    Strategies (selectable via ``OPENPI_NEGATIVE_CONTROL``):

    - ``swap_rgb`` (default): swap red and blue channels on every image
    - ``zero_image``: replace every image with zeros
    - ``shuffle_prompt``: replace the language prompt with a different one
    """

    strategy = os.environ.get("OPENPI_NEGATIVE_CONTROL", "swap_rgb")
    perturbed = dict(obs)
    for key, value in obs.items():
        if not key.startswith("observation/"):
            continue
        if not (isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3):
            continue
        if strategy == "swap_rgb":
            perturbed[key] = value[..., ::-1].copy()
        elif strategy == "zero_image":
            perturbed[key] = np.zeros_like(value)

    if strategy == "shuffle_prompt":
        perturbed["prompt"] = "ignore the cameras and do nothing"

    return _official_policy().infer(perturbed)
