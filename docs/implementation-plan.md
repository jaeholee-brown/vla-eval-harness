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

Non-goals for the current phase:

- public benchmark compatibility promises
- true bimanual protocol support in the current flat schema
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
- `DK-1` single-active-arm adapter
- `OpenPI` parity callables (with deterministic per-fixture noise so the
  flow-matching action parity test is meaningful)
- DROID fixture fetcher
- harness-side noise-aware openpi fidelity server (`scripts/serve_openpi_for_fidelity.py`)
- phase-1.5 fidelity runbook
- phase-1.5 guard that refuses benchmark runs if fairness metadata claims official preprocessing but the adapter is still using the identity preprocessor

Implemented but intentionally incomplete:

- hardware tests are placeholders until the real `DK-1` backend is wired

Closed on a GPU (no longer "incomplete"):

- fidelity tests (preprocessing parity, action parity, negative control)
  pass against a live pi05_droid runtime when the env vars in
  `docs/runbooks/phase-1.5-fidelity.md` are sourced; the recorded
  tolerances live in that runbook

Not implemented yet:

- real `GR00T` runtime spike
- real `MolmoAct2` runtime spike
- observed-pain report from those spikes
- internal representation
- true-bimanual embodiment support

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

- `docs/spikes/current-schema-gap-matrix.md`

This spike already established the critical constraint:

- the current flat schema is a temporary bootstrap path only

### Phase 1: current-schema `OpenPI + DK-1`

Status: scaffold complete, fidelity not yet proven.

What Phase 1 means:

- one active arm on a `DK-1` rig
- one parked arm managed entirely by the embodiment layer
- current flat schema only
- no true bimanual claim

Code already present:

- `vla_harness/runner/current_schema_runner.py`
- `vla_harness/adapters/policy/openpi_current_schema.py`
- `vla_harness/adapters/embodiment/dk1_active_arm.py`
- `vla_harness/eval/openpi_callables.py`
- `scripts/fetch_droid_fixtures.py`

Phase 1 is not considered closed yet because Phase 1.5 has not been earned.

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
- action parity: `atol=2e-2, rtol=2e-2` with deterministic per-fixture noise piped through `scripts/serve_openpi_for_fidelity.py`; steady-state max abs diff was 3.3e-3, worst cold-cache excursion was 1.12e-2 (cuDNN auto-tuner divergence between two independent JAX processes — verified stable across 8 suite-mode repeats at 2e-2)
- negative control: `min_abs_diff=1e-4` separated cleanly on all three of `swap_rgb`, `zero_image`, `shuffle_prompt`
- preprocess fail-on-purpose: pointing `OPENPI_HARNESS_PREPROCESS` at the identity callable failed the parity test (shape mismatch 180×320 vs 224×224)

Runbook:

- `docs/runbooks/phase-1.5-fidelity.md`

Tests involved:

- `tests/fidelity/test_openpi_preprocessing_parity.py`
- `tests/fidelity/test_openpi_action_parity.py`
- `tests/fidelity/test_openpi_action_negative_control.py`

### Phase 2: runtime spikes and observed pain report

Status: blocked on Phase 1.5.

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

#### Phase 2 observed-pain report

File to write:

- `docs/pain/current-schema-observed-pain.md`

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

### Phase 3: harness internal representation

Status: not started and should remain not started until Phase 2 closes.

Important naming rule:

- this is the harness internal representation
- it is not “the generalized protocol”

The design must be conservative and justified only by observed pain.

Required properties:

- transport-neutral payload model
- explicit support for multi-arm grouping
- explicit support for relative EEF delta semantics
- no universal assumption that history is encoded in tensor rank
- no universal assumption that stereo is always separate streams or always channels
- migration path from the current flat schema

Initial files to add once Phase 3 begins:

- `vla_harness/protocol/manifest.py`
- `vla_harness/protocol/observation.py`
- `vla_harness/protocol/action.py`

Authoring-goal deliverables (non-optional — Phase 3 does not exit without these):

- runnable policy-adapter skeleton at `vla_harness/adapters/policy/_skeleton.py`, with inline comments marking every spot the author must decide something
- runnable embodiment-adapter skeleton at `vla_harness/adapters/embodiment/_skeleton.py`
- runnable parity-callables skeleton at `vla_harness/eval/_skeleton.py`
- first draft of `docs/cookbook/adapter-authoring.md` with sections:
  - "How to add a new VLA"
  - "How to add a new embodiment"
  - "Mapping common upstream artifact shapes" (websocket server, FastAPI server, in-process Python entrypoint, HuggingFace policy server, managed-local-server)
- every field in the internal representation has a one-line note on where in a typical upstream artifact an agent would find the value to put there

Phase 3 exit criteria:

- the internal representation covers every concrete blocker named in the observed-pain report
- `OpenPI + DK-1` can be migrated without losing parity
- the model can represent true bimanual state/action groupings honestly
- transport remains separable from payload
- the three skeletons exist and are imported by at least one real adapter as a sanity check
- the cookbook first draft is complete

### Phase 4: full adapters on the internal representation

Order:

1. `GR00T`
2. `MolmoAct2`
3. `YAM`
4. true-bimanual `DK-1`, only if still needed after `YAM`

Rules:

- `GR00T` stays on `managed_local_server`
- `MolmoAct2` stays on an HTTP bridge to the official FastAPI path
- no clean-room runtime reimplementation unless the official release forces it
- all adapters must continue emitting structured fairness metadata plus free-form notes

Phase 4 is when the harness earns the right to say it supports more than the current bootstrap path.

Authoring-goal deliverables for Phase 4:

- after each adapter ships, the cookbook gains a worked example section showing the exact upstream-artifact → harness-field mapping for that adapter
- the cookbook gains a "Mapping common upstream artifact shapes" appendix with one concrete example per shape category, drawn from the four real adapters now in tree

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

This is the test that proves the harness has earned its authoring claim. Until this passes, Phase 4 is not closed.

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

### Rule 6: every exit criterion is re-read against the north star

Before declaring any phase complete, re-read its exit criteria with the north star in mind. If a phase's deliverables would not make the next adapter easier to author from upstream artifacts than the previous one was, the phase is not done — even if its code compiles and its tests pass.

## What To Do On The GPU Machine

Do this before anything else:

1. install `openpi` and `lerobot`
2. fetch fixtures with `scripts/fetch_droid_fixtures.py`
3. stand up the official `openpi` websocket runtime
4. export the runbook env vars
5. run the three fidelity tests in order
6. run at least one fail-on-purpose manual check
7. record the tolerances and outcome

Only after that should you start the `GR00T` and `MolmoAct2` spikes.

The exact commands are already documented in:

- `docs/runbooks/phase-1.5-fidelity.md`

## Immediate Next Commit After GPU Validation

Assuming Phase 1.5 is proven, the next code/doc change should be:

- add `docs/pain/current-schema-observed-pain.md`

That report should be committed before any internal-representation code is written.
