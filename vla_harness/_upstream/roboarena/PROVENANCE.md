# Vendored Upstream Provenance

This directory vendors a pinned slice of the upstream RoboArena repository.

- Upstream repository: <https://github.com/robo-arena/roboarena>
- Pinned commit: `a07f93d`
- Retrieved from the upstream repository on 2026-05-22

Vendored files:

- `policy.py`
- `policy_server.py`
- `policy_client.py`
- `utils/msgpack_numpy.py`

Notes:

- `policy_client.py` and `utils/msgpack_numpy.py` are marked in-file as copied from `openpi`.
- Those files therefore carry Apache-2.0 notice obligations in addition to the MIT license of the RoboArena repo.
- The active harness code should import from `vla_harness.*`. The top-level `roboarena.*` package exists only as a shim so the vendored upstream files can keep their absolute imports unchanged.
