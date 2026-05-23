# `vla-eval-harness`

A fair-inference harness for evaluating frontier vision-language-action (VLA)
policies across bimanual robot embodiments. Each adapter delegates to the
*official* upstream runtime and logs every benchmark-side choice it makes, so
results stay apples-to-apples across policies.

The repo is **bimanual-only**. A legacy single-active-arm flat-schema path is
quarantined under `vla_harness/legacy/` for fidelity work; new adapters do not
touch it.

## Quick Start

```bash
uv sync
uv pip install -e .
pytest tests/unit -q
```

That runs every CPU-only adapter smoke test. If it's green, the harness is
working on your machine. From here you can pick a policy + embodiment pair to
drive end-to-end (see below) or start authoring a new adapter.

## Available Adapter Pairs

All three policies and both embodiments are real and ship a CPU-only smoke
test. Live validation (GPU for the policy, real robot for the embodiment) is
the remaining work and is runbook-driven, not blocked by code.

| Policy adapter | File | Upstream runtime |
| --- | --- | --- |
| openpi `pi0_aloha_pen_uncap` | [openpi_aloha.py](vla_harness/adapters/policy/openpi_aloha.py) | official websocket server (`scripts/serve_policy.py`) |
| `MolmoAct2-BimanualYAM` | [molmoact2_yam.py](vla_harness/adapters/policy/molmoact2_yam.py) | official FastAPI server (`examples/yam/host_server_yam.py`) |
| GR00T | [gr00t.py](vla_harness/adapters/policy/gr00t.py) | official managed local server (`gr00t.policy.server_client`) |

| Embodiment adapter | File | Backend |
| --- | --- | --- |
| YAM bimanual | [yam_bimanual.py](vla_harness/adapters/embodiment/yam_bimanual.py) | official YAM `RobotEnv` |
| DK-1 bimanual | [dk1_bimanual.py](vla_harness/adapters/embodiment/dk1_bimanual.py) | official `bi_dk1_follower` |

To drive a real pair end-to-end, see
[docs/runbooks/live-integration.md](docs/runbooks/live-integration.md).

## Layout

- [vla_harness/protocol/](vla_harness/protocol/) — bimanual internal representation (manifest, observation/action packets)
- [vla_harness/adapters/policy/](vla_harness/adapters/policy/) — policy adapters + [template_policy_adapter.py](vla_harness/adapters/policy/template_policy_adapter.py)
- [vla_harness/adapters/embodiment/](vla_harness/adapters/embodiment/) — embodiment adapters + [template_embodiment_adapter.py](vla_harness/adapters/embodiment/template_embodiment_adapter.py)
- [vla_harness/runner/](vla_harness/runner/) — `BimanualRunner` (one rollout per process)
- [vla_harness/logging/](vla_harness/logging/) — structured fairness log
- [vla_harness/legacy/](vla_harness/legacy/) — quarantined current-schema path (do not use for new work)
- [vla_harness/_upstream/](vla_harness/_upstream/) — vendored RoboArena transport slice (pinned at `a07f93d`)
- [docs/cookbook/](docs/cookbook/) — authoring cookbook
- [docs/runbooks/](docs/runbooks/) — live-integration runbook for GPU + real hardware
- [docs/internal/](docs/internal/) — bootstrap chronicle and phase planning (historical context only)
- [scripts/legacy/](scripts/legacy/) — bootstrap spike scripts and the openpi fidelity launcher

## Adding a New Adapter With a Coding Agent

Adding a new VLA policy or embodiment is a *translation* task: copy the
template, fill in the fields from upstream code, ship a CPU-only smoke test.
The prompt below is the recommended way to brief a coding agent. Replace the
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
   client/backend. The pattern to copy is `tests/unit/test_openpi_aloha.py`.
3. A short worked-example addition in `docs/cookbook/adapter-authoring.md`
   under §2.7 (policies) or a new §3.6 (embodiments), modeled on the existing
   `pi0_aloha_pen_uncap` entry.

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

The three shipped adapters (`openpi_aloha`, `gr00t`, `molmoact2_yam`) are the
canonical worked examples — point the agent at the one whose upstream runtime
shape most closely matches the new target (websocket / managed local server /
FastAPI).

The full cookbook is at
[docs/cookbook/adapter-authoring.md](docs/cookbook/adapter-authoring.md).

## Test Suites

```bash
pytest tests/unit        # CPU-only smoke tests for every shipped adapter
pytest tests/fidelity    # openpi parity oracle (needs openpi + a checkpoint)
pytest tests/hardware    # DK-1 hardware smoke (needs the robot)
pytest tests/legacy      # quarantined current-schema path
```

The fidelity and hardware suites are intentionally skip-heavy until the
required external dependencies, captured frame corpora, and hardware backends
are available.

## Upstream Boundary

- The vendored upstream slice in `vla_harness/_upstream/roboarena/` is pinned
  at RoboArena commit `a07f93d`.
- `policy_client.py` and `msgpack_numpy.py` in that slice ultimately originate
  from `openpi` and carry Apache-2.0 notice obligations (see [NOTICE](NOTICE),
  [LICENSES/](LICENSES/), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- Active product code imports from `vla_harness.*`. The top-level `roboarena.*`
  package is an import shim that exists only so the vendored upstream files
  can resolve each other's relative imports — do not import from it directly.
- The original benchmark-facing README is preserved at
  [archive/upstream_roboarena/README.upstream.md](archive/upstream_roboarena/README.upstream.md).
