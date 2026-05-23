# `vla-eval-harness`

`vla-eval-harness` is a private fair-inference harness for evaluating frontier vision-language-action policies across robot embodiments.

The repo starts from a pinned, vendored slice of RoboArena's transport layer and builds the harness around that. It does **not** assume RoboArena already provides a rollout engine, embodiment adapters, or a generalized multi-arm protocol.

Current product scope is **bimanual-only**. The historical current-schema
single-active-arm path remains in-tree only as a bootstrap reference and
fidelity oracle, not as the target shape for future adapters.

## What Is Here

- `vla_harness/_upstream/roboarena/`: pinned upstream transport files from RoboArena commit `a07f93d`
- `vla_harness/runner/`: bimanual run orchestration (`BimanualRunner`)
- `vla_harness/adapters/policy/`: bimanual policy adapters plus `template_policy_adapter.py` — the first-class template for new policies
- `vla_harness/adapters/embodiment/`: bimanual embodiment adapters plus `template_embodiment_adapter.py` — the first-class template for new embodiments
- `vla_harness/protocol/`: bimanual-first internal representation (manifest, observation packet, action packet)
- `vla_harness/logging/`: structured fairness logs
- `vla_harness/legacy/`: quarantined current-schema (single-active-arm flat schema) path — kept for the Phase 1.5 fidelity oracle, **not** a target shape for new adapters
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
- adapter-authoring cookbook with numbered TODO-to-cookbook links
- CPU-only smoke-test coverage for every shipped adapter and runner
- real `MolmoAct2-BimanualYAM` policy adapter on the internal representation
- real true-bimanual `YAM` embodiment adapter plus a thin official-`RobotEnv` backend wrapper
- real true-bimanual `DK-1` embodiment adapter plus a thin official-`bi_dk1_follower` backend wrapper
- real `GR00T` managed-local-server adapter on the internal representation

Not implemented yet:

- live GPU validation for the new `MolmoAct2` and `GR00T` adapters
- live bimanual hardware validation for the `YAM` and `DK-1` embodiment backends
- at least one end-to-end true-bimanual run that closes the phase-4 authoring claim in the real world

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
pytest tests/legacy
```

The fidelity and hardware suites are intentionally skip-heavy until the required external dependencies, captured frame corpora, and hardware backends are available. `tests/legacy/` covers the quarantined current-schema path.

## Adding a New Adapter With a Coding Agent

This harness is designed so that adding a new VLA policy or embodiment is a
*translation* task: copy the template, fill in the fields from upstream code,
ship a CPU-only smoke test. The prompt below has been refined from real runs
and is the recommended way to brief a coding agent. Replace the items in
`{{angle braces}}` and hand the whole block to the agent verbatim.

````
Add a new true-bimanual {policy | embodiment} adapter for {{NAME}}, using the
existing bimanual internal representation in `vla_harness.protocol`.

Target: {{checkpoint id / robot SKU / official release tag}}

Hard requirements:
- Read the official upstream source/docs first and treat them as the only
  source of truth. If upstream is silent on a field, surface that as a
  question — do not invent a value.
- Prefer official runtime delegation (server, trained-policy entrypoint, SDK)
  over reimplementation.
- Target the bimanual protocol (`vla_harness.adapters.policy.bimanual` /
  `vla_harness.adapters.embodiment.bimanual`). Do NOT use anything under
  `vla_harness/legacy/` — that path is quarantined.
- For a policy: do NOT use the single-arm static-padding bridge unless the
  upstream policy is genuinely single-arm. True-bimanual upstream means honest
  true-bimanual here.
- Do NOT modify files outside the new adapter, its test, and the cookbook
  worked-example addition. The protocol, runner, and decision-log surfaces are
  static.

Upstream sources to read first (replace as appropriate for {{NAME}}):
- {{official README}}
- {{official remote-inference / serving doc}}
- {{official training / config file that defines the deployment entry}}
- {{official request/response schema or modality config}}
- {{any worked example in the upstream repo, e.g. examples/<task>/main.py}}

Implement:
1. `vla_harness/adapters/{policy|embodiment}/{{module_name}}.py`, started by
   copying `template_{policy|embodiment}_adapter.py` in the same directory.
2. `tests/unit/test_{{module_name}}.py` — CPU-only smoke test using a fake
   client/backend. The pattern to copy is
   `tests/unit/test_openpi_aloha.py`.
3. A short worked-example addition in
   `docs/cookbook/adapter-authoring.md` under §2.7 (policies) or a new §3.6
   (embodiments), modeled on the existing `pi0_aloha_pen_uncap` entry.

The adapter must:
- delegate to the official runtime; do not duplicate normalization, image
  preprocessing, or action decoding logic
- copy upstream defaults verbatim for checkpoint ref, config name, runtime
  family, schema source, prompt-format source, camera keys, state layout,
  action semantics, and chunk horizon
- populate every `PolicyMetadata` / `EmbodimentMetadata` field that has a
  documented upstream answer
- emit one `DecisionNote` per upstream-derived choice (`status="official"`)
  and one per benchmark-derived choice (`status="benchmark_default"`), with
  the upstream file path or doc URL in `evidence`
- never silently reinterpret action semantics (absolute vs. relative, joint
  vs. EEF, gripper sign/scale)

CPU-only smoke test must prove:
1. construction succeeds
2. the adapter satisfies the `Bimanual{Policy|Embodiment}Adapter` Protocol
3. `build_manifest()` returns a valid `HarnessManifest` matching upstream
   schema
4. one dummy `ObservationPacket` produces an `ActionPacket` of the expected
   per-arm shape (for a policy) — or one dummy `ActionPacket` is accepted and
   one dummy `ObservationPacket` is produced (for an embodiment)
5. every `DecisionNote` in `build_notes()` is fully populated and tagged
   `official` wherever upstream is the source

Acceptance:
- `pytest tests/unit -q` is green
- the new smoke test is in `tests/unit/`
- changes are committed in modular commits: (a) the adapter, (b) the smoke
  test, (c) the cookbook worked example
- if any acceptance step requires a real GPU, a real robot, or a missing
  upstream artifact, stop and report — do not stub it out

Style notes:
- do not add fields, helpers, or abstractions beyond what this single adapter
  needs
- do not write a runtime client from scratch if the upstream one is importable
- do not edit `vla_harness/protocol/`, `vla_harness/runner/`,
  `vla_harness/logging/`, or any file under `vla_harness/legacy/`
````

The first three completed adapters (`openpi_aloha`, `gr00t`, `molmoact2_yam`)
are the canonical worked examples — point the agent at one whose upstream
runtime shape most closely matches the new target (websocket / managed
local server / FastAPI).

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

Phase 3 and the code-delivery part of Phase 4 are now implemented: the
transport-neutral, bimanual-first internal representation exists, the
historical flat-schema path can be bridged into it explicitly, and the first
real true-bimanual adapters are in tree with CPU-only smoke coverage.

The next implementation target is live validation of those adapters on a GPU
machine and real bimanual hardware.

The phase-2 source-backed artifacts that justify that work are in:

- [docs/spikes/artifacts/gr00t-current-schema.json](docs/spikes/artifacts/gr00t-current-schema.json)
- [docs/spikes/artifacts/molmoact2-current-schema.json](docs/spikes/artifacts/molmoact2-current-schema.json)
- [docs/spikes/upstream-default-source-map.md](docs/spikes/upstream-default-source-map.md)
- [docs/cookbook/adapter-authoring.md](docs/cookbook/adapter-authoring.md)
- [docs/runbooks/phase-4-live-integration.md](docs/runbooks/phase-4-live-integration.md)

## Upstream Boundary

- The vendored upstream slice is pinned at RoboArena commit `a07f93d`.
- `policy_client.py` and `msgpack_numpy.py` in that slice ultimately originate from `openpi` and carry Apache-2.0 notice obligations.
- Active product code should import from `vla_harness.*`. The top-level `roboarena.*` package remains only as an import shim for the vendored upstream files.

See [archive/upstream_roboarena/README.upstream.md](archive/upstream_roboarena/README.upstream.md) for the original benchmark-facing README.
