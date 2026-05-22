# Adapter Authoring Cookbook

This cookbook exists to make adding a new VLA adapter or embodiment adapter a
translation job, not a design job.

The default workflow is:

1. read the upstream release code or docs
2. copy values into the skeleton fields tagged `copy_from_upstream`
3. fill the smallest possible set of `benchmark_derived` fields
4. wire a parity or replay battery
5. record every non-official choice in the fairness log

## Internal-Representation Field Guide

Every field in the Phase-3 internal representation should be sourced from one of
these upstream artifacts whenever possible.

| Field | Where to find it upstream | Typical examples |
| --- | --- | --- |
| `arm_groups.side` | embodiment docs / SDK / robot config | left arm, right arm |
| `arm_groups.control_role` | official embodiment or policy scope; otherwise benchmark bridge rule | policy-controlled, static-pad-only |
| `video_streams.name` | request schema / modality config | `top_cam`, `wrist_image_left` |
| `video_streams.role` | request schema / camera docs | top, wrist, exterior |
| `video_streams.order_index` | request schema order or modality-config order | YAM `[top, left, right]` |
| `video_streams.sample_indices` | modality config / history config | GR00T `[-15, 0]` |
| `state_streams.name` | request schema / modality config | `joint_position`, `eef_9d` |
| `state_streams.dim` | validation code / model config | `7`, `9`, `14` |
| `state_streams.layout` | model docs / server validation | `joint_position`, `xyz_rot6d` |
| `state_streams.origin` | embodiment sensors vs benchmark derivation rule | sensor, derived |
| `action_streams.name` | policy output schema | `policy_action`, `eef_9d` |
| `action_streams.dim` | action head or server contract | `8`, `17`, `14` |
| `action_streams.semantics` | modality config / policy docs | relative EEF, absolute gripper |
| `language_fields.name` | request schema | `instruction`, `prompt` |

If you cannot point to an upstream artifact for a field, mark it
`benchmark_derived` and explain why.

## How To Add A New VLA

1. Start from [vla_harness/adapters/policy/_skeleton.py](/Users/jaeholee0404/roboarena/vla_harness/adapters/policy/_skeleton.py).
2. Gather the upstream artifacts:
   - repo or model card
   - official inference server or Python entrypoint
   - request schema or modality-config file
   - checkpoint identifiers
   - normalization tag or equivalent deployment metadata
3. Fill the `copy_from_upstream` fields first:
   - `policy_family`
   - `runtime_family`
   - `schema_source`
   - `checkpoint_ref`
   - `normalization_tag`
   - camera roles, state layouts, action semantics in the manifest
4. Decide whether the policy is:
   - native bimanual
   - single-arm on a bimanual setup
5. If it is single-arm only, set exactly two bridge fields:
   - which arm it controls
   - which static padding rule keeps the other arm still
6. Implement `build_manifest()` from the upstream schema source.
7. Implement `infer()` by converting the harness `ObservationPacket` into the official runtime input.
8. Add a parity or replay battery using [vla_harness/eval/_skeleton.py](/Users/jaeholee0404/roboarena/vla_harness/eval/_skeleton.py).
9. Add at least one negative control.

### VLA examples already informing this repo

- `OpenPI`: historical flat-schema bootstrap and parity oracle
- `GR00T`: modality config with explicit temporal sampling and mixed action semantics
- `MolmoAct2-DROID`: bridgeable single-arm request schema
- `MolmoAct2-BimanualYAM`: honest true-bimanual request schema

## How To Add A New Embodiment

1. Start from [vla_harness/adapters/embodiment/_skeleton.py](/Users/jaeholee0404/roboarena/vla_harness/adapters/embodiment/_skeleton.py).
2. Gather the upstream artifacts:
   - official SDK or robot runtime
   - camera layout docs
   - proprio/state APIs
   - control frequency and chunk-consumption behavior if documented
3. Fill the `copy_from_upstream` fields first:
   - `embodiment_family`
   - `backend_name`
   - camera role order
   - left/right arm identifiers
   - state sources
   - control frequency
4. Implement `capture_observation()` so that both arms and all ordered cameras are represented honestly.
5. Implement `execute_action()` so that:
   - policy-controlled arms consume their streams directly
   - static-padding arms stay still explicitly if a single-arm policy is bridged in
6. Add a dry-run or replay test before live motion.

## Mapping Common Upstream Artifact Shapes

### Websocket server

Typical shape:

- one remote metadata endpoint or handshake
- one inference method returning an action chunk

Example:

- `OpenPI`

Use:

- copy request keys into the manifest
- bridge transport locally
- keep the server-specific transport details out of the manifest

### FastAPI server

Typical shape:

- HTTP route with JSON or `json_numpy` payload
- embodiment-specific request body

Example:

- `MolmoAct2`

Use:

- copy route request fields, camera order, and normalization tag directly
- keep HTTP mechanics adapter-local

### Managed local server

Typical shape:

- subprocess or local server process
- sidecar client
- modality config or embodiment tag in code

Example:

- `GR00T`

Use:

- copy stream names, temporal indices, and action semantics from the modality config
- keep lifecycle and subprocess control adapter-local

### In-process Python entrypoint

Typical shape:

- direct `predict(...)` or `forward(...)`
- no official server wrapper

Use:

- keep the manifest tied to the upstream callable signature
- add a small harness-local wrapper only if needed for parity controls or replay

## When To Use The Single-Arm Bridge

Use it only when all of the following are true:

1. the policy is genuinely single-arm only
2. the embodiment is bimanual
3. you can keep the uncontrolled arm still with an explicit padding rule

Do not use the bridge to fake true-bimanual support. If the upstream policy is
already bimanual, represent both arms honestly in the manifest and action packet.
