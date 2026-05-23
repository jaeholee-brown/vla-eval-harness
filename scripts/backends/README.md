# Embodiment Backend Loaders

This directory holds the per-embodiment Python glue that turns a real upstream
robot SDK into a harness backend the launcher can drive.

Each file here exposes a ``make_backend(*, config_path: str | None = None, **_)
-> BimanualBackend`` callable. The launcher's ``--backend-loader`` flag points
at one of these (e.g. ``--backend-loader scripts.backends.yam_robotenv:make_backend``).
The corresponding ``--<embodiment>-config PATH`` launcher flag is forwarded as
the ``config_path`` kwarg.

## Shipped templates

| File | For | Upstream entry point |
| --- | --- | --- |
| `yam_robotenv.py` | YAM bimanual (`--embodiment yam`) | `gello.envs.yam_real_env.YAMRealEnv` |
| `dk1_bidk1.py` | DK-1 bimanual (`--embodiment dk1`) | `lerobot.robots.bi_dk1_follower.BiDK1Follower` |

If your install matches the canonical upstream and the SDK is importable, you
do not need to edit these. Just install the upstream SDK and point the
launcher at your robot config:

```bash
python scripts/run_episode.py \
    --policy molmoact2_yam --server-url http://127.0.0.1:8202/act \
    --embodiment yam --yam-config /etc/yam/my_rig.yaml \
    --prompt "pack the blocks" --max-steps 50
```

## When you have to edit a template

The upstream robot SDKs evolve independently of this harness. If a template's
upstream import or constructor keyword no longer matches your install:

1. Open the relevant `*.py` template.
2. Look for the block marked `--- EDIT THIS BLOCK IF YOUR INSTALL DIVERGES ---`.
3. Change the import and constructor call. The rest of the file does not need
   to move.

The harness-side contract the template targets is documented at the top of
each file. As long as the returned object satisfies that contract, the rest of
the harness does not care which upstream version or fork it came from.

## Adding a backend for a new embodiment

When a coding agent adds a new embodiment adapter (see the README's
"Adding a New Adapter With a Coding Agent" section), it should also drop a
matching `scripts/backends/<name>.py` file here that follows the same shape:

- Lazy import of the upstream robot SDK
- A `make_backend(*, config_path=None, **_)` entry point
- A clearly-marked edit block for the constructor call
- A docstring listing the contract the returned object must satisfy
