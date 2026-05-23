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
            f"Environment variable {name!r} is required to use vla_harness.legacy.openpi_callables."
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

    from vla_harness.legacy.openpi_current_schema import OpenPICurrentSchemaAdapter
    from vla_harness.legacy.openpi_current_schema import OpenPIRuntimeConfig

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
            preprocessing_oracle_name="vla_harness.legacy.openpi_callables:official_preprocess",
            action_oracle_name="vla_harness.legacy.openpi_callables:official_action",
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

    pi05_droid's transform returns a nested dict where ``image`` itself is
    a mapping of canonical camera names (``base_0_rgb``, ``left_wrist_0_rgb``,
    ``right_wrist_0_rgb``). We extract the exterior view (``base_0_rgb``)
    because that is what our preprocessing fixture corpus contains.
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

    candidate = transformed.get("image", transformed.get("images", transformed.get("observation/image")))
    if isinstance(candidate, dict):
        for inner_key in ("base_0_rgb", "exterior_image_1_left", "image"):
            if inner_key in candidate:
                return np.asarray(candidate[inner_key])
    elif candidate is not None:
        return np.asarray(candidate)
    if "observation/exterior_image_1_left" in transformed:
        return np.asarray(transformed["observation/exterior_image_1_left"])

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


def _deterministic_noise_for(obs: dict[str, Any]) -> np.ndarray:
    """Build a fixed (action_horizon, action_dim) noise tensor for ``obs``.

    pi05 (and other flow-matching openpi models) sample a fresh noise
    tensor inside ``Policy.infer`` on every call. Without pinning that
    noise, two independent processes can't produce matching action
    chunks for the same input, which makes ``test_openpi_action_parity``
    structurally impossible to satisfy.

    We derive the seed from a stable identifier on the obs so each fixture
    gets a different noise tensor (catching any conditioning that depends
    on the noise sample), but a given fixture always sees the same noise
    on both the official and the harness legs.
    """

    policy = _official_policy()
    horizon = policy._model.action_horizon  # noqa: SLF001 - introspection
    dim = policy._model.action_dim  # noqa: SLF001 - introspection
    seed_source = obs.get("prompt", "")
    if isinstance(seed_source, np.ndarray):
        seed_source = seed_source.item() if seed_source.ndim == 0 else str(seed_source.tolist())
    if isinstance(seed_source, bytes):
        seed_source = seed_source.decode("utf-8", errors="replace")
    seed = abs(hash(("openpi-fidelity", str(seed_source)))) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((horizon, dim), dtype=np.float32)


def official_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Call openpi's Policy.infer directly with a deterministic noise."""

    noise = _deterministic_noise_for(obs)
    return _official_policy().infer(obs, noise=noise)


def harness_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Call the harness OpenPI adapter, which routes through the websocket.

    We piggyback the same deterministic noise tensor inside the obs dict.
    ``scripts/legacy/serve_openpi_for_fidelity.py`` strips it off and forwards it
    to ``Policy.infer(obs, noise=noise)``; the stock openpi websocket
    server ignores extra keys, in which case action parity will diverge
    because of independent RNG advancement (this is why the fidelity
    server is the documented entrypoint).
    """

    enriched = dict(obs)
    enriched["noise"] = _deterministic_noise_for(obs)
    return _harness_adapter().infer(enriched)


def negative_control_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Deliberately perturbed call into the official policy.

    The point of this callable is to prove the action-parity battery can
    distinguish a faithful path from a broken one. The perturbation is
    intentionally large enough that any working parity test must flag it.

    Strategies (selectable via ``OPENPI_NEGATIVE_CONTROL``):

    - ``swap_rgb`` (default): swap red and blue channels on every image
    - ``zero_image``: replace every image with zeros
    - ``shuffle_prompt``: replace the language prompt with a different one

    We pin the noise to the same deterministic tensor as ``official_action``
    so the only thing varying is the perturbation itself.
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

    noise = _deterministic_noise_for(obs)
    return _official_policy().infer(perturbed, noise=noise)
