"""Default ``--backend-loader`` for the official LeRobot ``bi_dk1_follower``.

Usage from the launcher (assuming LeRobot is installed and you have a DK-1
robot config on disk):

    python scripts/run_episode.py \\
        --policy openpi_aloha --host 127.0.0.1 --port 8000 \\
        --embodiment dk1 --dk1-config /etc/dk1/my_rig.json \\
        --prompt "uncap the pen" --max-steps 50

What this file does:

1. Imports the upstream ``BiDK1Follower`` robot object lazily.
2. Constructs it from a single config path (``--dk1-config``).
3. Wraps it in
   :class:`vla_harness.adapters.embodiment.dk1_bimanual.LeRobotBiDK1Backend`.

If your install does not match the canonical upstream entry point named in
:data:`UPSTREAM_ROBOT_IMPORT`, edit the clearly-marked block in
:func:`_build_bi_dk1_follower` below. The harness contract you are targeting
is the upstream ``BiDK1Follower`` API; the harness side documents the joint
feature order and camera key set in
:class:`vla_harness.adapters.embodiment.dk1_bimanual.LeRobotBiDK1BackendConfig`
(``joint_1.pos … joint_6.pos`` + ``gripper.pos`` per arm, ``head/right_wrist/
left_wrist`` cameras).
"""

from __future__ import annotations

from typing import Any

from vla_harness.adapters.embodiment.dk1_bimanual import (
    LeRobotBiDK1Backend,
    LeRobotBiDK1BackendConfig,
)


# The upstream entry point this file expects. If your install names it
# something else, change this constant AND the import inside _build_bi_dk1_follower.
UPSTREAM_ROBOT_IMPORT = "lerobot.robots.bi_dk1_follower:BiDK1Follower"


def make_backend(*, config_path: str | None = None, **_: Any) -> LeRobotBiDK1Backend:
    """Launcher entry point. Forwarded ``--dk1-config`` lands in ``config_path``."""
    if config_path is None:
        raise SystemExit(
            "scripts/backends/dk1_bidk1.py: pass --dk1-config <path> to specify "
            "the DK-1 robot config (the file your install would normally hand "
            "to bi_dk1_follower). Use --embodiment fake if you have no robot."
        )
    robot = _build_bi_dk1_follower(config_path)
    return LeRobotBiDK1Backend(robot, LeRobotBiDK1BackendConfig())


def _build_bi_dk1_follower(config_path: str) -> Any:
    """Construct the upstream ``BiDK1Follower``.

    --- EDIT THIS BLOCK IF YOUR INSTALL DIVERGES ----------------------------
    The canonical upstream entry, as referenced in the harness runbook
    (``docs/runbooks/live-integration.md``) and the cookbook, is approximately:

        from lerobot.robots.bi_dk1_follower import BiDK1Follower
        robot = BiDK1Follower(config_path=config_path)

    If your LeRobot version names the module differently (e.g.
    ``lerobot.common.robots.…``) or the constructor takes a different keyword,
    change the import and call below. The rest of the harness does not care,
    as long as the returned object exposes ``get_observation()`` and the
    documented action surface.
    ------------------------------------------------------------------------
    """
    try:
        from lerobot.robots.bi_dk1_follower import BiDK1Follower  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Could not import the upstream BiDK1Follower "
            f"({UPSTREAM_ROBOT_IMPORT}). Install LeRobot, or edit "
            "scripts/backends/dk1_bidk1.py to import from wherever your "
            "fork lives."
        ) from exc

    return BiDK1Follower(config_path=config_path)
