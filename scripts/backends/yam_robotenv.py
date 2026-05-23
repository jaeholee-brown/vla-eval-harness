"""Default ``--backend-loader`` for the official YAM ``RobotEnv``.

Usage from the launcher (assuming the gello/YAM SDK is installed and you have
a robot config YAML on disk):

    python scripts/run_episode.py \\
        --policy molmoact2_yam --server-url http://127.0.0.1:8202/act \\
        --embodiment yam --yam-config /etc/yam/my_rig.yaml \\
        --prompt "pack the blocks" --max-steps 50

What this file does:

1. Imports the upstream YAM ``RobotEnv`` lazily so the launcher's dry-run path
   and other embodiments are unaffected when the YAM SDK is not installed.
2. Constructs the env from a single config path (``--yam-config``).
3. Wraps it in :class:`vla_harness.adapters.embodiment.yam_bimanual.YAMRobotEnvBackend`.

If your install does not match the canonical upstream entry point named in
:data:`UPSTREAM_ROBOT_ENV_IMPORT` (e.g. you forked gello, the module path
moved, or your constructor takes more than a single YAML path), edit the
clearly-marked block in :func:`_build_yam_robot_env` below. The harness
contract you are targeting is documented at the bottom of this docstring.

`YAMRobotEnvLike` contract (the env this file returns must satisfy):

- ``env.get_obs() -> Mapping[str, np.ndarray]`` returning at least:
    * ``"joint_positions"``: shape ``(14,)``, left arm first (``[:7]``), right
      arm second (``[7:]``). Each per-arm 7-D is ``[joint_1 … joint_6, gripper]``.
    * ``"front_camera_rgb"``, ``"left_camera_rgb"``, ``"right_camera_rgb"``:
      HxWx3 uint8 frames. If your install names cameras differently, override
      :class:`YAMRobotEnvBackendConfig.camera_key_by_backend_name` instead of
      renaming inside the env.
- ``env.step(joint_positions: np.ndarray)`` taking a shape ``(14,)`` float32
  joint target (same left-then-right ordering).
"""

from __future__ import annotations

from typing import Any

from vla_harness.adapters.embodiment.yam_bimanual import (
    YAMRobotEnvBackend,
    YAMRobotEnvBackendConfig,
)


# The upstream entry point this file expects. If your install names it
# something else, change this constant AND the import inside _build_yam_robot_env.
UPSTREAM_ROBOT_ENV_IMPORT = "gello.envs.yam_real_env:YAMRealEnv"


def make_backend(*, config_path: str | None = None, **_: Any) -> YAMRobotEnvBackend:
    """Launcher entry point. Forwarded ``--yam-config`` lands in ``config_path``."""
    if config_path is None:
        raise SystemExit(
            "scripts/backends/yam_robotenv.py: pass --yam-config <path> to specify "
            "the YAM robot config YAML (the file your install would normally hand "
            "to launch_yaml_eval_molmoact.py). Use --embodiment fake if you have "
            "no robot."
        )
    env = _build_yam_robot_env(config_path)
    return YAMRobotEnvBackend(env, YAMRobotEnvBackendConfig())


def _build_yam_robot_env(config_path: str) -> Any:
    """Construct the upstream YAM env.

    --- EDIT THIS BLOCK IF YOUR INSTALL DIVERGES ----------------------------
    The canonical upstream entry, as referenced in the harness runbook
    (``docs/runbooks/live-integration.md``) and the gello configs comment in
    ``vla_harness/adapters/embodiment/yam_bimanual.py``, is approximately:

        from gello.envs.yam_real_env import YAMRealEnv
        env = YAMRealEnv(config_path=config_path)

    If your fork uses a different module path, class name, or constructor
    keyword, change the import and call below. The rest of the harness does
    not care, as long as the returned object satisfies ``YAMRobotEnvLike``.
    ------------------------------------------------------------------------
    """
    try:
        from gello.envs.yam_real_env import YAMRealEnv  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Could not import the upstream YAM env "
            f"({UPSTREAM_ROBOT_ENV_IMPORT}). Install the gello/YAM SDK, or "
            "edit scripts/backends/yam_robotenv.py to import from wherever "
            "your fork lives."
        ) from exc

    return YAMRealEnv(config_path=config_path)
