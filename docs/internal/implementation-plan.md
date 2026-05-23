# Implementation Plan

This document is the canonical plan for `vla-eval-harness` as of the current repo state.

It is intentionally concrete. It describes:

- what has already been implemented
- what still blocks the next phase
- exactly what to run on a GPU machine before moving forward
- what Phase 2 should do and what it should explicitly **not** do
- the constraints that Phase 3 must satisfy once the pain is known

The plan is based on the actual code in this repo, not on an abstract future architecture.

## North Star

The harness exists so that adding a new VLA or a new robot is a templated translation job, not a design job.

Concretely:

> If I want to add a new arm and a new VLA, I can get a coding agent to easily code up an embodiment adapter and a VLA adapter. It should be easy for the agent to look at the existing published "default settings" or published code that came when the model or arm released, and port those into templates of existing adapters. The agent should have to make minimal decisions.

This is the test every phase exit criterion is judged against. A phase is done not just when its code works, but when the resulting contract makes the next adapter easier to author from official upstream artifacts than the previous one was.

### What that requires the harness to provide

- a small, stable contract per adapter type, small enough to fit in one LLM context
- a contract honest enough to carry diverse policies and embodiments without silent flattening
- an automatic fairness ledger that the adapter declares as part of its contract
- templated verification (parity callables + recorded-trajectory replay) so each new adapter can be smoke-tested without hardware
- a documented mapping from common upstream artifact shapes (websocket server, FastAPI server, in-process Python entrypoint, HuggingFace policy server, managed-local-server) to harness adapter slots

### Authoring invariants any future schema change must hold

- policy adapter Protocol stays around 5–7 methods
- embodiment adapter Protocol stays around 5–7 methods
- every fairness-relevant decision is a typed dataclass field, never a free-form string
- every adapter type has a runnable skeleton file the agent can copy and fill in
- every adapter type has at least two reference implementations before its contract is declared stable, so agents triangulate the pattern instead of overfitting to one example

## Goals

Primary goals:

- preserve official model and embodiment defaults whenever possible
- minimize discretionary benchmark-side choices
- make every non-official choice visible in a fairness log
- build around a pinned upstream transport slice instead of overstating RoboArena as a full harness
- defer the internal representation until real adapter pain has been observed
- target only bimanual embodiments going forward

Non-goals for the current phase:

- public benchmark compatibility promises
- true bimanual protocol support in the current flat schema
- new single-arm embodiment support beyond the historical bootstrap path
- reimplementation of official model runtimes
- early standardization of a generalized transport or payload format

## Current Repo State

Implemented and checked in:

- pinned upstream RoboArena transport slice vendored at commit `a07f93d`
- provenance and third-party notice handling
- archived upstream benchmark-facing docs and tests
- current-schema runner
- structured fairness log with tolerance fields
- `OpenPI` current-schema adapter
- legacy `DK-1` single-active-arm bootstrap adapter
- `OpenPI` parity callables (with deterministic per-fixture noise so the
  flow-matching action parity test is meaningful)
- DROID fixture fetcher
- harness-side noise-aware openpi fidelity server (`scripts/legacy/serve_openpi_for_fidelity.py`)
- phase-1.5 fidelity runbook
- phase-1.5 guard that refuses benchmark runs if fairness metadata claims official preprocessing but the adapter is still using the identity preprocessor
- phase-2 source-backed runtime probes for `GR00T` and `MolmoAct2`
- phase-2 observed-pain report
- phase-2 JSON artifacts under `docs/internal/spikes/artifacts/`
- phase-2 upstream-default source map in `docs/internal/spikes/upstream-default-source-map.md`

Implemented but intentionally incomplete:

- hardware tests are placeholders until the real `DK-1` backend is wired

Closed on a GPU (no longer "incomplete"):

- fidelity tests (preprocessing parity, action parity, negative control)
  pass against a live pi05_droid runtime when the env vars in
  `docs/internal/phase-1.5-fidelity.md` are sourced; the recorded
  tolerances live in that runbook

Not implemented yet:

- first real true-bimanual embodiment and policy adapters on the internal representation

## Upstream Boundary

The harness uses RoboArena in a narrow, pinned way.

Kept from upstream:

- `policy.py`
- `policy_server.py`
- `policy_client.py`
- `utils/msgpack_numpy.py`

These are vendored under `vla_harness/_upstream/roboarena/`.

Important clarifications:

- `policy_client.py` and `msgpack_numpy.py` originate from `openpi`, not RoboArena alone
- the harness does **not** treat upstream RoboArena as a rollout framework or embodiment layer
- the vendored slice is temporary infrastructure and future reference material, not a permanent public compatibility commitment

## Phase Order

### Phase 0: repo bootstrap and provenance

Status: complete.

Deliverables already present:

- vendored upstream slice
- provenance file
- notices
- archived upstream docs
- standalone repo identity

### Phase 0.5: current-schema gap spike

Status: complete.

Deliverable already present:

- `docs/internal/spikes/current-schema-gap-matrix.md`

This spike already established the critical constraint:

- the current flat schema is a temporary bootstrap path only

### Phase 1: current-schema `OpenPI + DK-1`

Status: historical bootstrap complete and fidelity proven on `OpenPI`; real `DK-1` hardware smoke is still separate work.

What Phase 1 means:

- one active arm on a `DK-1` rig
- one parked arm managed entirely by the embodiment layer
- current flat schema only
- no true bimanual claim

Important scope update:

- this path is retained as a bootstrap reference only
- future product work should not add new single-arm adapters
- future `DK-1`, `YAM`, and other embodiments should target the bimanual internal representation directly

Code already present:

- `vla_harness/legacy/current_schema_runner.py`
- `vla_harness/legacy/openpi_current_schema.py`
- `vla_harness/legacy/dk1_active_arm.py`
- `vla_harness/legacy/openpi_callables.py`
- `scripts/legacy/fetch_droid_fixtures.py`

What changed after the GPU run:

- the model-side fidelity gate is closed for `OpenPI`
- the gating artifact is not just “parity exists,” but “parity exists with explicit stochastic-control plumbing and validated negative controls”
- phase 2 does not need to wait for a real `DK-1` backend; the next bottleneck is model-side adapter pressure, not embodiment execution

### Phase 1.5: fidelity gates

Status: closed 2026-05-22 on a single RTX 5090 with pi05_droid.

All three tests pass with the tolerances logged in the runbook; both
fail-on-purpose loops were observed to fail as expected.

The gate is strict: **do not start Phase 2 until all three of these are complete**.

1. Preprocessing parity
   - use real captured DROID frames
   - compare the official `openpi` preprocessing path against the harness preprocessing path
   - if exact reuse is wired, require exact equality
   - if exact reuse is impossible in practice, use explicit tolerances and log them

2. Action parity
   - use real captured flat-schema observations
   - compare `openpi` direct policy outputs against harness-routed outputs
   - keep checkpoint, config, dtype, device, and prompt fixed
   - use explicit `atol` / `rtol`, never silent exact-equality assumptions

3. Negative control
   - prove the parity battery can detect a deliberately wrong path
   - at minimum, verify that one of:
     - `swap_rgb`
     - `zero_image`
     - `shuffle_prompt`
     causes the action-parity battery to separate from the official path

Evidence required to close Phase 1.5:

- preprocessing parity test passes
- action parity test passes
- negative control passes
- at least one fail-on-purpose manual check was performed and observed to fail
- the tolerances that were required to pass are recorded in the fairness log metadata

Earned tolerances (2026-05-22 RTX 5090, pi05_droid, 5 DROID fixtures):

- preprocessing parity: `atol=0.0, rtol=0.0` (byte-identical, both callables route through `_run_image_through_official_transforms`)
- action parity: `atol=2e-2, rtol=2e-2` with deterministic per-fixture noise piped through `scripts/legacy/serve_openpi_for_fidelity.py`; steady-state max abs diff was 3.3e-3, worst cold-cache excursion was 1.12e-2 (cuDNN auto-tuner divergence between two independent JAX processes — verified stable across 8 suite-mode repeats at 2e-2)
- negative control: `min_abs_diff=1e-4` separated cleanly on all three of `swap_rgb`, `zero_image`, `shuffle_prompt`
- preprocess fail-on-purpose: pointing `OPENPI_HARNESS_PREPROCESS` at the identity callable failed the parity test (shape mismatch 180×320 vs 224×224)

Runbook:

- `docs/internal/phase-1.5-fidelity.md`

Tests involved:

- `tests/fidelity/test_openpi_preprocessing_parity.py`
- `tests/fidelity/test_openpi_action_parity.py`
- `tests/fidelity/test_openpi_action_negative_control.py`

### Phase 2: runtime spikes and observed pain report

Status: complete.

Phase 2 is a spike, not adapter delivery.

The only purpose of Phase 2 is to observe what breaks when the current harness shape is forced onto:

- one `GR00T` runtime
- one official `MolmoAct2` server path

#### Phase 2A: `GR00T` spike

Required constraints:

- use `managed_local_server`, not `inprocess`
- do not try to solve true bimanual support yet
- do not build a polished adapter surface

Concrete tasks:

1. Stand up one official `GR00T` runtime path using `managed_local_server`.
2. Pick one embodiment config only.
3. Attempt to project the current flat-schema observation dict into the required GR00T modality config.
4. Record every place where the current policy-adapter shape is awkward or insufficient.
5. Record every action-semantic mismatch, especially relative EEF delta semantics.

Questions this spike must answer:

- Which fields are fundamentally payload-semantic, not transport-semantic?
- What does the current flat schema erase about GR00T’s modality system?
- Can the current fairness log name the important differences, or does it need new structured fields later?
- Does the current runner assume action semantics that break on relative EEF deltas?

Deliverable:

- a section in the observed-pain report dedicated to `GR00T`

#### Phase 2B: `MolmoAct2` spike

Required constraints:

- keep the official FastAPI server path
- do not reimplement the server as websocket
- do not flatten transport and payload into one concern

Concrete tasks:

1. Stand up one official `MolmoAct2` FastAPI server.
2. Use one official embodiment-specific server path first.
3. Try to force-fit that path into the current policy-adapter shape.
4. Record every place where transport assumptions leak into the adapter boundary.
5. Record every place where embodiment-specific camera/state schemas do not fit the current flat shape cleanly.

Questions this spike must answer:

- What must remain transport-local instead of becoming part of the internal representation?
- Which embodiment-specific schema details are too important to hide behind relabeling?
- How much of the current adapter shape is really `OpenPI`-specific?

Deliverable:

- a section in the observed-pain report dedicated to `MolmoAct2`

#### Priors updated by Phase 1.5

These are now assumptions the next phase should start from:

1. **Stochastic-policy parity needs explicit randomness control.**
   The `OpenPI` result showed that websocket-vs-inprocess parity is not meaningful for flow-matching policies unless both legs share deterministic per-example noise. For future model spikes, the first question is not just “what is the inference entrypoint?” but also “how do we control randomness across processes?”

2. **Exact preprocessing parity is realistic when the official transform can be reused directly.**
   `OpenPI` preprocessing parity was byte-identical. That raises the bar for future adapters: if an official preprocessing transform is callable, exact parity should be the target until proven impossible.

3. **Negative controls are mandatory, not nice-to-have.**
   The fail-on-purpose loops carried real signal. Every future parity or replay battery should include at least one deliberate break path.

4. **Offline recorded fixtures are sufficient for early model-side pressure testing.**
   Phase 2 does not need live `DK-1` hardware to discover schema and transport pain. The current next risk is model/runtime mismatch, not robot I/O.

5. **Stock serving paths may be insufficient for fidelity work.**
   `scripts/legacy/serve_openpi_for_fidelity.py` exists because the stock `openpi` server could not expose deterministic-noise control. This makes it more likely that `GR00T` and `MolmoAct2` spikes will need similarly small, spike-only runtime wrappers to make apples-to-apples comparisons possible.

#### What Phase 2 actually ran

- `scripts/legacy/spike_gr00t_current_schema.py`
  - official source root: `/tmp/vla_sources/Isaac-GR00T`
  - commit: `3df8b38`
  - exercised the official DROID modality config and the official ZeroMQ `PolicyServer` / `PolicyClient`
  - wrote `docs/internal/spikes/artifacts/gr00t-current-schema.json`
- `scripts/legacy/spike_molmoact2_current_schema.py`
  - official source root: `/tmp/vla_sources/molmoact2`
  - commit: `804ba37`
  - exercised the official DROID and YAM FastAPI apps through `build_app(...)` and `TestClient`
  - wrote `docs/internal/spikes/artifacts/molmoact2-current-schema.json`

#### Phase 2 outcomes

- `GR00T` pressure is real:
  - nested modality payloads
  - explicit temporal horizons
  - `eef_9d` state
  - mixed relative/absolute multi-stream action semantics
  - ZeroMQ transport instead of websocket
- `MolmoAct2` splits into two very different cases:
  - DROID is bridgeable with a fairly honest current-schema mapping
  - YAM is not honest under the current flat schema and proves true-bimanual structure must be first-class
- the pain report is now specific enough to justify Phase 3
- the source map is now specific enough to tell a future agent where to copy defaults from upstream code instead of inventing harness-local values

#### Phase 2 observed-pain report

File to write:

- `docs/internal/pain/current-schema-observed-pain.md`
- `docs/internal/spikes/upstream-default-source-map.md`

Required sections:

- `OpenPI + DK-1` lessons from Phase 1.5
- payload-shape pain
- transport pain
- control-semantics pain
- history/stereo pain
- true-bimanual pain
- what must move into the future internal representation
- what can remain adapter-local

Phase 2 exit criteria:

- both spikes were actually run
- the report is based on real runtime experience, not only paper reasoning
- the report is specific enough that Phase 3 can be written as implementation work, not fresh exploration

Status update: satisfied. The report is in `docs/internal/pain/current-schema-observed-pain.md`.
The source map is in `docs/internal/spikes/upstream-default-source-map.md`.

### Phase 3: harness internal representation

Status: foundations implemented; first real true-bimanual adapters still pending.

Important naming rule:

- this is the harness internal representation
- it is not “the generalized protocol”

The design must be conservative and justified only by observed pain.

Required properties:

- transport-neutral payload model
- explicit support for multi-arm grouping
- explicit support for named left/right arm groups as first-class structure
- explicit support for relative EEF delta semantics
- no universal assumption that history is encoded in tensor rank
- no universal assumption that stereo is always separate streams or always channels
- migration path from the current flat schema
- every structured field is either directly copyable from upstream artifacts or explicitly marked as a benchmark-derived projection rule
- no design effort spent preserving single-arm ergonomics beyond the migration bridge from the historical bootstrap path

Initial files to add once Phase 3 begins:

- `vla_harness/protocol/manifest.py`
- `vla_harness/protocol/observation.py`
- `vla_harness/protocol/action.py`

Implemented now:

- `vla_harness/protocol/manifest.py`
- `vla_harness/protocol/observation.py`
- `vla_harness/protocol/action.py`
- `vla_harness/legacy/current_schema_bridge.py`
- `vla_harness/adapters/policy/bimanual.py`
- `vla_harness/adapters/embodiment/bimanual.py`

Authoring-goal deliverables (non-optional — Phase 3 does not exit without these):

- runnable policy-adapter template at `vla_harness/adapters/policy/template_policy_adapter.py`, with inline comments marking every spot the author must decide something
- runnable embodiment-adapter template at `vla_harness/adapters/embodiment/template_embodiment_adapter.py`
- runnable parity-callables skeleton at `vla_harness/eval/_skeleton.py`
- first draft of `docs/cookbook/adapter-authoring.md` with sections:
  - "How to add a new VLA"
  - "How to add a new embodiment"
  - "Mapping common upstream artifact shapes" (websocket server, FastAPI server, in-process Python entrypoint, HuggingFace policy server, managed-local-server)
- every field in the internal representation has a one-line note on where in a typical upstream artifact an agent would find the value to put there
- every template field is tagged as `copy_from_upstream` or `benchmark_derived`

Implemented now:

- `vla_harness/adapters/policy/template_policy_adapter.py`
- `vla_harness/adapters/embodiment/template_embodiment_adapter.py`
- `vla_harness/eval/_skeleton.py`
- `docs/cookbook/adapter-authoring.md`

Phase 3 exit criteria:

- the internal representation covers every concrete blocker named in the observed-pain report
- `OpenPI + DK-1` can be migrated without losing parity
- the model can represent true bimanual state/action groupings honestly
- transport remains separable from payload
- the three skeletons exist and are imported by at least one real adapter as a sanity check
- the cookbook first draft is complete

Recommended first implementation order inside Phase 3:

1. define typed dataclasses for stream names, stream sampling, stream semantics, camera roles, and named left/right arm groupings
2. encode the direct-copy fields identified in `docs/internal/spikes/upstream-default-source-map.md`
3. make sure the representation can express `MolmoAct2-BimanualYAM` and a true-bimanual `DK-1` embodiment without projection hacks
4. migrate `OpenPI + DK-1` through the new representation without changing its already-earned parity behavior, but only as a legacy bootstrap bridge
5. only then add the benchmark-derived projection hooks needed for GR00T and DROID-style bridge cases

Status update:

- satisfied for the protocol foundation, bridge helpers, skeletons, and cookbook
- phase-4 code delivery is now complete in-tree
- still unproven with a live true-bimanual policy-plus-embodiment path on GPU/hardware

### Phase 4: full adapters on the internal representation

Order:

1. `MolmoAct2-BimanualYAM`
2. true-bimanual `YAM`
3. true-bimanual `DK-1`
4. `GR00T`

Rules:

- `GR00T` stays on `managed_local_server`
- `MolmoAct2` stays on an HTTP bridge to the official FastAPI path
- no clean-room runtime reimplementation unless the official release forces it
- all adapters must continue emitting structured fairness metadata plus free-form notes

Phase 4 is when the harness earns the right to say it supports more than the current bootstrap path.

Because product scope is bimanual-only, Phase 4 is not done when a single-arm
policy path works through the new representation. It is done when the first
true-bimanual policy-plus-embodiment path is real, documented, and usable as a
template for later adapters.

Authoring-goal deliverables for Phase 4:

- shipped: four real phase-4 reference adapters are now in tree:
  - `MolmoAct2YAMPolicyAdapter`
  - `GR00TPolicyAdapter`
  - `YAMBimanualAdapter`
  - `DK1BimanualAdapter`
- shipped: concrete official-runtime backend wrappers are now in tree where source code made them explicit:
  - `YAMRobotEnvBackend`
  - `LeRobotBiDK1Backend`
- shipped: the cookbook now uses numbered sections and the skeleton TODOs link to them
- shipped: every shipped adapter and runner now has a CPU-only smoke test
- remaining: one outside contributor or coding agent still needs to add a fifth adapter from the cookbook alone to prove the authoring claim in practice

### Phase 4 exit gate — the authoring claim

Phase 4 is not done when `GR00T`, `MolmoAct2`, and `YAM` all work. It is done when an outside contributor — or a coding agent working from the cookbook alone, with no live design help from the harness maintainer — can author a fifth adapter against an unlisted upstream policy or embodiment, using only:

- the cookbook
- the skeletons
- the four reference adapters
- the upstream artifact's own documentation

The fifth adapter does not have to be production-grade. It must:

- compile against the Protocols
- emit a complete fairness log
- pass its own parity gates
- have been authored without the harness maintainer rewriting it

This is the test that proves the harness has earned its authoring claim. Code delivery for Phase 4 is complete, but the authoring claim is not closed until this passes.

## Fairness Rules

These rules apply across phases.

### Rule 1: official defaults beat harness invention

If official model/runtime behavior is available and practical to preserve, preserve it.

### Rule 2: no hidden fidelity claims

If fairness metadata says “official,” the runtime path must actually be wired to the official behavior.

The current `OpenPI` readiness guard enforces this for preprocessing.

### Rule 3: tolerances are part of the result

For parity tests:

- tolerances must be explicit inputs
- tolerances must be logged
- loosened tolerances are not free; they are part of the benchmark record

### Rule 4: parity batteries need negative controls

A parity battery that cannot detect an intentionally wrong path is not a useful parity battery.

### Rule 5: no phase skipping

Do not start Phase 2 until Phase 1.5 is proven on the GPU machine.

Status update: this rule is satisfied for `OpenPI`.

### Rule 6: every exit criterion is re-read against the north star

Before declaring any phase complete, re-read its exit criteria with the north star in mind. If a phase's deliverables would not make the next adapter easier to author from upstream artifacts than the previous one was, the phase is not done — even if its code compiles and its tests pass.

## What To Do On The GPU Machine

Do this before anything else:

1. install `openpi` and `lerobot`
2. fetch fixtures with `scripts/legacy/fetch_droid_fixtures.py`
3. stand up the official `openpi` websocket runtime
4. export the runbook env vars
5. run the three fidelity tests in order
6. run at least one fail-on-purpose manual check
7. record the tolerances and outcome

This step is now complete for `OpenPI`.

The exact commands are already documented in:

- `docs/internal/phase-1.5-fidelity.md`

The next spike checklist is documented in:

- `docs/internal/phase-2-runtime-spikes.md`

## Immediate Next Commit After GPU Validation

After the current code state, the next work is live validation:

1. validate `MolmoAct2YAMPolicyAdapter` against the official YAM FastAPI server on a GPU machine
2. validate `GR00TPolicyAdapter` against the official managed-local-server path on a GPU machine
3. validate `YAMRobotEnvBackend` on a real bimanual YAM setup
4. validate `LeRobotBiDK1Backend` on a real bimanual DK-1 setup
5. then ask a fresh coding agent or outside contributor to add a fifth adapter from the cookbook alone

The live-validation checklist is documented in:

- `docs/runbooks/live-integration.md`
