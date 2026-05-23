# Current-Schema Observed Pain

This report is based on real phase-2 spike work against official source trees:

- `GR00T`: official code checked out at commit `3df8b38`
- `MolmoAct2`: official code checked out at commit `804ba37`

Probe artifacts used to generate these notes:

- `scripts/legacy/spike_gr00t_current_schema.py`
- `scripts/legacy/spike_molmoact2_current_schema.py`
- `docs/internal/spikes/artifacts/gr00t-current-schema.json`
- `docs/internal/spikes/artifacts/molmoact2-current-schema.json`

Default-source follow-up for future adapter templates:

- `docs/internal/spikes/upstream-default-source-map.md`

Both were run locally against the official source trees. The GR00T spike used the official DROID modality config plus the official ZeroMQ `PolicyServer` / `PolicyClient` transport. The MolmoAct2 spike used the official FastAPI `build_app(...)` functions for both DROID and YAM with stub policies, so the request/response behavior came from the real server modules without loading model weights.

## OpenPI + DK-1 Lessons Carried Forward

- Phase 1.5 proved that stochastic-policy parity needs explicit randomness control.
- Exact preprocessing parity is realistic when the official transform is directly reusable.
- Negative controls are mandatory because a parity battery that cannot fail on purpose is not useful.
- Offline replay fixtures are enough for early model-side pressure testing.
- Small spike-only runtime wrappers are acceptable when the stock serving path cannot expose the controls needed for faithful comparison. `scripts/legacy/serve_openpi_for_fidelity.py` established that precedent.

## GR00T Pain

What was actually run:

- imported `gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS`
- imported the official `gr00t.policy.server_client.PolicyServer` / `PolicyClient`
- stood up a temporary official ZeroMQ server on localhost with a stub policy that served the real DROID modality config
- queried `get_modality_config`, `ping`, and `reset` through the real GR00T transport

Concrete findings:

- The official transport is **ZeroMQ REQ/REP + msgpack-numpy**, not websocket. This is a true transport mismatch with the current current-schema harness shell.
- The official DROID embodiment tag `oxe_droid_relative_eef_relative_joint` expects:
  - video keys: `exterior_image_1_left`, `wrist_image_left`
  - video horizon: `2` via `delta_indices=[-15, 0]`
  - state keys: `eef_9d`, `gripper_position`, `joint_position`
  - state horizon: `1`
  - action keys: `eef_9d`, `gripper_position`, `joint_position`
  - action horizon: `40`
- The current flat schema can rename the two camera keys, but it cannot represent the required video horizon honestly. A single current frame must be duplicated or buffered to satisfy GR00T's two-frame expectation.
- The current flat state surface is not rename-only:
  - it has `cartesian_position(6)`, not `eef_9d(9)`
  - producing `eef_9d` requires a benchmark-side pose convention and rotation conversion rule
  - that is semantic invention, not a transport rename
- The current flat action schema is fundamentally too weak:
  - GR00T DROID action is 17D across three named streams
  - the streams have mixed semantics:
    - `eef_9d`: relative
    - `gripper_position`: absolute
    - `joint_position`: relative
  - the current flat action chunk supports only one array plus one coarse `action_space` enum, which cannot encode mixed per-stream semantics

Source-of-truth code:

- `gr00t/configs/data/embodiment_configs.py`
- `getting_started/policy.md`
- `gr00t/policy/server_client.py`

Questions answered:

- Which required modality fields do not fit the current flat schema?
  - `eef_9d`, temporal video horizons, and multi-stream actions
- Where does the current adapter contract become semantically lossy?
  - as soon as it tries to flatten mixed relative/absolute action streams
- What action semantics break the current runner assumptions?
  - per-stream relative EEF delta + absolute gripper + relative joint

## MolmoAct2 Pain

What was actually run:

- imported `examples/droid/host_server_droid.py`
- imported `examples/yam/host_server_yam.py`
- built both official FastAPI apps via `build_app(...)`
- drove them with `fastapi.testclient.TestClient`
- sent:
  - an honest current-schema-derived DROID request
  - the raw current flat schema to DROID
  - an official-shape synthetic YAM request
  - the raw current flat schema to YAM

Concrete findings:

- The official transport is **HTTP + json_numpy**, not websocket.
- `MolmoAct2-DROID` is a relatively good fit for the current flat schema:
  - `observation/exterior_image_1_left -> external_cam`
  - `observation/wrist_image_left -> wrist_cam`
  - `prompt -> instruction`
  - `concat(joint_position, gripper_position) -> state(8)`
  - the official DROID app accepted that mapped request and returned `{"actions", "dt_ms"}`
- Even in the “good fit” case, some data is simply dropped:
  - `cartesian_position` is present in the current flat schema but unused by the official DROID server
- `MolmoAct2-BimanualYAM` is not an honest fit:
  - official request requires `top_cam`, `left_cam`, `right_cam`
  - camera order matters
  - official state is `(14,)`, representing two 7-D arms
  - the current flat schema only exposes one logical arm and two unnamed camera roles
- A YAM request can only be produced from the current flat schema by making benchmark-side inventions:
  - inventing a top/left/right camera-role mapping
  - duplicating or padding camera streams
  - synthesizing the missing second-arm state
- The model-family abstraction itself is checkpoint-specific at runtime:
  - DROID server calls `predict_action(..., action_mode="continuous")`
  - YAM server calls `predict_action(..., inference_action_mode="continuous")`
  - a future `MolmoAct2` adapter cannot assume one invariant model-call signature for all checkpoints
- The official deployment defaults include embodiment-specific `norm_tag`s:
  - `franka_droid`
  - `yam_dual_molmoact2`
  These are fairness-relevant and currently invisible to the harness as typed fields.

Source-of-truth code:

- `examples/droid/host_server_droid.py`
- `examples/yam/host_server_yam.py`
- `README.md` in the MolmoAct2 repo

## Payload-Shape Pain

- The current flat schema assumes one logical arm and one flat action array.
- GR00T wants nested modality groups with per-modality horizons and multi-stream actions.
- MolmoAct2 YAM wants true bimanual state plus three ordered cameras.
- Conclusion: payload shape is not just “different names.” The harness needs first-class modality grouping and multi-arm structure.

## Transport Pain

- Current bootstrap harness transport is websocket + msgpack.
- GR00T official transport is ZeroMQ REQ/REP + msgpack-numpy.
- MolmoAct2 official transport is HTTP + json_numpy.
- Conclusion: transport must remain adapter-local. The future internal representation must be transport-neutral.

## Control-Semantics Pain

- GR00T DROID uses mixed action semantics within one policy output:
  - relative EEF
  - absolute gripper
  - relative joint
- Current flat `action_space` cannot express mixed semantics.
- MolmoAct2 DROID and YAM are both absolute joint-pose control paths, but they are still embodiment-specific at the server level.
- Conclusion: action semantics must become typed fields in the internal representation, not a single enum.

## History / Stereo Pain

- GR00T video horizon is explicit (`delta_indices=[-15, 0]` for DROID). The current flat schema has no place to say “I need two frames with specific temporal offsets.”
- The current bootstrap path also has no principled way to express whether camera multiplicity is stereo, role-based multi-camera, or temporal history.
- Conclusion: history and camera roles must be explicit structure, not inferred from tensor shape or key naming hacks.

## True-Bimanual Pain

- MolmoAct2 YAM proves the current flat schema cannot honestly represent true bimanual state/action.
- The current single-active-arm `DK-1` path is still fine as a bootstrap embodiment, but it is not a general representation.
- Conclusion: the future internal representation must carry multi-arm groupings explicitly rather than as padded vectors.

## What Must Become Internal-Representation Fields

- transport-independent observation grouping by modality (`video`, `state`, `language`)
- explicit per-stream names
- explicit per-stream temporal horizons / sampling indices
- multi-arm state and action grouping
- typed action semantics per stream:
  - representation (`relative`, `absolute`)
  - action type (`eef`, `non_eef`)
  - format (`xyz_rot6d`, etc.)
- camera order / camera role metadata
- embodiment-specific normalization tag or equivalent deployment metadata
- source provenance for modality config / request schema
- runtime family / transport family
- policy-call signature variant when model families expose checkpoint-specific inference kwargs

## What Can Stay Adapter-Local

- HTTP vs websocket vs ZeroMQ connection mechanics
- server health-check endpoints
- checkpoint-specific weight-loading patches
- local runtime spawning details
- temporary spike-only wrappers used only to expose fidelity controls or probe official server behavior

## Minimum Phase-3 Requirements Implied By This Report

- The internal representation must be nested and transport-neutral.
- The migration layer from the current flat schema must make lossy projections explicit.
- The internal representation should optimize for honest bimanual structure first; the current flat schema is now only a legacy bootstrap bridge.
- The future policy-adapter template must have a place to declare:
  - modality config source
  - normalization tag
  - action stream semantics
  - camera order
  - state projection method
- The future embodiment-adapter template must have a place to declare:
  - arm grouping
  - camera role inventory
  - any benchmark-side state derivation rules

## Minimal Future Adapter Decisions

This report changes the bar for Phase 3 template design.

A future coding agent adding a new VLA should mostly be answering:

- which official file defines the request or modality schema?
- which official file defines transport?
- which official file defines stream names, horizons, and normalization tags?
- which fields still require benchmark-side projection?

The agent should not have to rediscover those categories from scratch.

Concrete consequences from this phase:

- `GR00T` proved the template needs separate slots for:
  - upstream transport family
  - modality-config source
  - per-stream temporal indices
  - per-stream action semantics
  - benchmark-side `eef_9d` derivation rules when upstream embodiment state is richer than the harness bootstrap state
- `MolmoAct2` proved the template needs separate slots for:
  - request-schema source
  - normalization tag
  - ordered camera roles
  - per-checkpoint inference call signature variants
  - explicit dropped-field and projection rules for DROID-like bridge cases
  - explicit refusal to invent missing-arm state for true-bimanual cases

Because product scope is now bimanual-only:

- future embodiment templates should assume named multi-arm groupings, not `active_arm` plus `parked_arm`
- future policy templates should assume that honest true-bimanual payloads are the default target, not an optional extension
