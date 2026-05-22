# `vla-eval-harness`

`vla-eval-harness` is a private fair-inference harness for evaluating frontier vision-language-action policies across robot embodiments.

The repo starts from a pinned, vendored slice of RoboArena's transport layer and builds the harness around that. It does **not** assume RoboArena already provides a rollout engine, embodiment adapters, or a generalized multi-arm protocol.

## What Is Here

- `vla_harness/_upstream/roboarena/`: pinned upstream transport files from RoboArena commit `a07f93d`
- `vla_harness/runner/`: current-schema runner and run orchestration
- `vla_harness/adapters/policy/`: policy adapters, starting with `OpenPI`
- `vla_harness/adapters/embodiment/`: embodiment adapters, starting with a single-active-arm `DK-1` path
- `vla_harness/logging/`: structured fairness logs
- `archive/upstream_roboarena/`: archived upstream docs and non-harness-facing references

## Current Status

Implemented now:

- pinned upstream transport vendoring with provenance and notices
- current-schema gap matrix for `GR00T` and `MolmoAct2`
- current-schema runner and structured fairness log
- `OpenPI` current-schema adapter scaffold
- `DK-1` single-active-arm adapter scaffold
- unit, fidelity-harness, and hardware-smoke test scaffolding
- phase-1.5 fidelity guards for preprocessing claims and configurable action-parity tolerances

Not implemented yet:

- real `GR00T` adapter
- real `MolmoAct2` FastAPI bridge
- true bimanual internal representation
- true bimanual `YAM` or `DK-1` protocol support

## Install

```bash
uv sync
uv pip install -e .
```

## Run Tests

```bash
pytest tests/unit
pytest tests/fidelity
pytest tests/hardware
```

The fidelity and hardware suites are intentionally skip-heavy until the required external dependencies, captured frame corpora, and hardware backends are available.

## Phase 2 Entry Gate

Do not start the `GR00T` / `MolmoAct2` runtime spike until all three of these are true for `OpenPI + DK-1`:

1. preprocessing parity is wired to the real official preprocessing path
2. action parity passes on captured fixtures with configured `atol` / `rtol`
3. a negative-control action test proves the parity battery can detect a deliberately wrong path

Phase 2 is a spike, not full adapter implementation. Its output should be an observed-pain report in `docs/pain/`.

## Upstream Boundary

- The vendored upstream slice is pinned at RoboArena commit `a07f93d`.
- `policy_client.py` and `msgpack_numpy.py` in that slice ultimately originate from `openpi` and carry Apache-2.0 notice obligations.
- Active product code should import from `vla_harness.*`. The top-level `roboarena.*` package remains only as an import shim for the vendored upstream files.

See [archive/upstream_roboarena/README.upstream.md](archive/upstream_roboarena/README.upstream.md) for the original benchmark-facing README.
