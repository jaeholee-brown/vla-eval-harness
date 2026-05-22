# Third-Party Notices

This repository vendors a small, pinned transport slice from the RoboArena project and preserves the provenance of code that RoboArena itself copied from `openpi`.

## Vendored upstream slice

- Upstream project: `robo-arena/roboarena`
- Upstream URL: <https://github.com/robo-arena/roboarena>
- Pinned commit: `a07f93d`
- License: MIT

Vendored files:

- `vla_harness/_upstream/roboarena/policy.py`
- `vla_harness/_upstream/roboarena/policy_server.py`
- `vla_harness/_upstream/roboarena/policy_client.py`
- `vla_harness/_upstream/roboarena/utils/msgpack_numpy.py`

## `openpi`-derived files

RoboArena's `policy_client.py` and `utils/msgpack_numpy.py` are marked in-file as copied from `openpi`.

- Upstream project: `Physical-Intelligence/openpi`
- Upstream URL: <https://github.com/Physical-Intelligence/openpi>
- License: Apache-2.0

Affected vendored files:

- `vla_harness/_upstream/roboarena/policy_client.py`
- `vla_harness/_upstream/roboarena/utils/msgpack_numpy.py`

The Apache License, Version 2.0 text is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
