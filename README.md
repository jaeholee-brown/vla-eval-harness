# `vla-eval-harness`

`vla-eval-harness` is a private fair-inference harness for evaluating frontier vision-language-action policies across robot embodiments.

The repo starts from a pinned, vendored slice of RoboArena's transport layer and builds the harness around that. It does **not** assume RoboArena already provides a rollout engine, embodiment adapters, or a generalized multi-arm protocol.

Current product scope is **bimanual-only**. The historical current-schema
single-active-arm path remains in-tree only as a bootstrap reference and
fidelity oracle, not as the target shape for future adapters.

## What Is Here

- `vla_harness/_upstream/roboarena/`: pinned upstream transport files from RoboArena commit `a07f93d`
- `vla_harness/runner/`: current-schema runner and run orchestration
- `vla_harness/adapters/policy/`: policy adapters, starting with `OpenPI`
- `vla_harness/adapters/embodiment/`: embodiment adapters, currently including a legacy single-active-arm `DK-1` bootstrap path kept for reference
- `vla_harness/protocol/`: bimanual-first internal representation and legacy flat-schema bridge helpers
- `vla_harness/logging/`: structured fairness logs
- `archive/upstream_roboarena/`: archived upstream docs and non-harness-facing references

## Current Status

Implemented now:

- pinned upstream transport vendoring with provenance and notices
- current-schema gap matrix for `GR00T` and `MolmoAct2`
- current-schema runner and structured fairness log
- `OpenPI` current-schema adapter scaffold
- legacy `DK-1` single-active-arm bootstrap adapter scaffold
- unit, fidelity-harness, and hardware-smoke test scaffolding
- phase-1.5 fidelity guards for preprocessing claims and configurable action-parity tolerances
- phase-1.5 fidelity closure against live `pi05_droid`
- phase-2 runtime spike probes for `GR00T` and `MolmoAct2`
- phase-2 observed pain report
- source-backed phase-2 probe artifacts under `docs/spikes/artifacts/`
- an upstream-default source map for future adapter templates
- bimanual-first internal representation dataclasses and current-schema bridge wrappers
- runnable policy, embodiment, and parity skeletons for future adapters
- adapter-authoring cookbook for future coding agents

Not implemented yet:

- real `MolmoAct2-BimanualYAM` adapter
- real true-bimanual `YAM` embodiment adapter
- real true-bimanual `DK-1` embodiment adapter
- real `GR00T` adapter on the new representation

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

## Current Phase Status

Phase 1.5 has been earned on GPU, and Phase 2 is complete.

Completed Phase-2 outputs:

- one `GR00T` runtime spike using official source-backed probes
- one official `MolmoAct2` runtime spike using the real FastAPI apps
- one observed-pain report in `docs/pain/current-schema-observed-pain.md`
- one upstream-default source map in `docs/spikes/upstream-default-source-map.md`

Step-by-step instructions for running the three gating tests are in
[docs/runbooks/phase-1.5-fidelity.md](docs/runbooks/phase-1.5-fidelity.md).

Step-by-step instructions for the runtime spikes are in
[docs/runbooks/phase-2-runtime-spikes.md](docs/runbooks/phase-2-runtime-spikes.md).

The full implementation plan and phase ordering are in
[docs/implementation-plan.md](docs/implementation-plan.md).

Phase 3 foundations are now implemented: the transport-neutral, bimanual-first
internal representation exists, and the historical flat-schema path can be
bridged into it explicitly.

The next implementation target is the first real true-bimanual adapters on top
of that representation.

The phase-2 source-backed artifacts that justify that work are in:

- [docs/spikes/artifacts/gr00t-current-schema.json](docs/spikes/artifacts/gr00t-current-schema.json)
- [docs/spikes/artifacts/molmoact2-current-schema.json](docs/spikes/artifacts/molmoact2-current-schema.json)
- [docs/spikes/upstream-default-source-map.md](docs/spikes/upstream-default-source-map.md)
- [docs/cookbook/adapter-authoring.md](docs/cookbook/adapter-authoring.md)

## Upstream Boundary

- The vendored upstream slice is pinned at RoboArena commit `a07f93d`.
- `policy_client.py` and `msgpack_numpy.py` in that slice ultimately originate from `openpi` and carry Apache-2.0 notice obligations.
- Active product code should import from `vla_harness.*`. The top-level `roboarena.*` package remains only as an import shim for the vendored upstream files.

See [archive/upstream_roboarena/README.upstream.md](archive/upstream_roboarena/README.upstream.md) for the original benchmark-facing README.
