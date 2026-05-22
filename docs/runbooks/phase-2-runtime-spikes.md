# Phase 2 Runtime Spike Runbook

This runbook is for the first post-fidelity exploration phase.

Phase 1.5 is already closed for `OpenPI`, so the goal now is not to prove another faithful adapter. The goal is to discover where the current harness shape breaks when forced onto new model runtimes.

Outputs:

- one `GR00T` runtime spike
- one `MolmoAct2` runtime spike
- one observed-pain report in `docs/pain/current-schema-observed-pain.md`

This phase is explicitly **not** for shipping polished adapters.

## Rules

1. Do not clean-room reimplement either runtime.
2. Do not expand the current schema yet.
3. Do not start Phase 3 design during the spike.
4. Record pain immediately while it is fresh.
5. Prefer one narrow supported path per model over broad speculative coverage.

## `GR00T` Spike

Target:

- one official `GR00T` runtime path
- `managed_local_server`, not `inprocess`
- one embodiment config only

Why:

- `GR00T`'s modality config system is likely to pressure the current flat schema
- its relative end-effector delta semantics are likely to pressure the current runner and action assumptions

Questions to answer:

- Which required `GR00T` fields cannot be represented honestly in the current flat schema?
- Which mismatches are just renaming, and which are semantic loss?
- Does the current adapter contract need new structured fairness fields for modality config?
- What is the smallest spike-only wrapper needed to drive `GR00T` through the current harness shape?

Artifacts to produce:

- short notes under the `GR00T` section of `docs/pain/current-schema-observed-pain.md`
- if needed, a tiny spike-only helper script under `scripts/` with a clear comment that it is for runtime probing, not production adapter architecture

## `MolmoAct2` Spike

Target:

- one official `MolmoAct2` FastAPI server path
- one embodiment-specific server path first

Why:

- the transport mismatch is real and should remain adapter-local later
- the embodiment-specific request schemas are likely to pressure the current flat policy adapter shape

Questions to answer:

- What assumptions in the current `OpenPI` adapter shape are really websocket-specific?
- Which `MolmoAct2` request fields are payload semantics versus transport glue?
- How awkward is it to express the official embodiment-specific server schema through the current harness shape without lying?
- Is a minimal HTTP bridge obviously enough later, or does the harness need a deeper runtime abstraction?

Artifacts to produce:

- short notes under the `MolmoAct2` section of `docs/pain/current-schema-observed-pain.md`
- if needed, a tiny spike-only helper script under `scripts/` with a clear comment that it is for runtime probing, not production adapter architecture

## Observed-Pain Report

File:

- `docs/pain/current-schema-observed-pain.md`

Minimum sections to fill:

- `OpenPI + DK-1 lessons carried forward`
- `GR00T pain`
- `MolmoAct2 pain`
- `payload-shape pain`
- `transport pain`
- `control-semantics pain`
- `history/stereo pain`
- `true-bimanual pain`
- `what must become internal-representation fields`
- `what can stay adapter-local`

## Exit Criteria

Phase 2 is done when:

- both runtime spikes were actually run
- the pain report is specific enough to drive Phase 3 without fresh broad exploration
- the team can point to at least three concrete current-shape failures that the future internal representation must solve

Phase 2 is not done when:

- a server launches successfully
- a shallow wrapper exists
- a speculative Phase 3 design has been drafted without real pain notes
