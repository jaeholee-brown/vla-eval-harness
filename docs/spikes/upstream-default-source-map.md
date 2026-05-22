# Upstream Default Source Map

This file exists for one reason:

- when a future coding agent adds a new VLA adapter or embodiment adapter, it should spend most of its effort copying values from official upstream artifacts, not inventing harness-local defaults

The entries below record where the important defaults actually came from in the
phase-2 source-backed probes, and which remaining fields would still require
benchmark-side invention under the current flat schema.

## How To Use This File

When designing a future adapter template, every field should land in one of two buckets:

- `copy_from_upstream`: the value is directly readable from official code or official runtime metadata
- `benchmark_derived`: the value only exists after the harness chooses a projection rule

The goal for future templates is that most fields should be `copy_from_upstream`.

Current planning scope:

- future embodiment work is bimanual-only
- the historical single-active-arm `DK-1` bootstrap path remains only as a migration/reference artifact
- future templates should default to named multi-arm groups, not `active_arm` / `parked_arm`

## GR00T

Official source used in the phase-2 probe:

- repo: `Isaac-GR00T`
- commit: `3df8b38`
- files:
  - `gr00t/configs/data/embodiment_configs.py`
  - `gr00t/policy/server_client.py`
  - `getting_started/policy.md`

Fields a future `GR00T` policy adapter should copy directly from upstream:

- `transport_family`
  - value source: `gr00t/policy/server_client.py`
  - observed value: `zmq+msgpack_numpy`
- `embodiment_tag`
  - value source: selected key in `MODALITY_CONFIGS`
  - observed value: `oxe_droid_relative_eef_relative_joint`
- `video_stream_names`
  - value source: `MODALITY_CONFIGS[tag]["video"].modality_keys`
  - observed value: `["exterior_image_1_left", "wrist_image_left"]`
- `video_sampling_indices`
  - value source: `MODALITY_CONFIGS[tag]["video"].delta_indices`
  - observed value: `[-15, 0]`
- `state_stream_names`
  - value source: `MODALITY_CONFIGS[tag]["state"].modality_keys`
  - observed value: `["eef_9d", "gripper_position", "joint_position"]`
- `action_stream_names`
  - value source: `MODALITY_CONFIGS[tag]["action"].modality_keys`
  - observed value: `["eef_9d", "gripper_position", "joint_position"]`
- `action_horizon`
  - value source: `MODALITY_CONFIGS[tag]["action"].delta_indices`
  - observed value: `40`
- `action_stream_semantics`
  - value source: `MODALITY_CONFIGS[tag]["action"].action_configs[*].rep`
  - observed value:
    - `eef_9d -> relative`
    - `gripper_position -> absolute`
    - `joint_position -> relative`

Fields that were still benchmark-derived under the current flat schema:

- `eef_9d_projection_rule`
  - why: current flat schema only carries `cartesian_position(6)`
- `video_history_buffering_rule`
  - why: current flat schema only provides the current frame
- `flat_action_projection_rule`
  - why: current flat schema cannot carry mixed per-stream semantics honestly

Phase-3 consequence:

- the internal representation needs first-class stream names, stream semantics, and explicit temporal sampling

## MolmoAct2 DROID

Official source used in the phase-2 probe:

- repo: `molmoact2`
- commit: `804ba37`
- file: `examples/droid/host_server_droid.py`

Fields a future `MolmoAct2` DROID adapter should copy directly from upstream:

- `transport_family`
  - value source: FastAPI app + `json_numpy` usage in `host_server_droid.py`
  - observed value: `http+json_numpy`
- `checkpoint_family`
  - value source: `REPO_ID`
  - observed value: `allenai/MolmoAct2-DROID`
- `normalization_tag`
  - value source: `NORM_TAG`
  - observed value: `franka_droid`
- `default_chunk_size`
  - value source: `DEFAULT_NUM_STEPS`
  - observed value: `10`
- `camera_roles`
  - value source: request schema in `/act`
  - observed value: `external_cam`, `wrist_cam`
- `instruction_field`
  - value source: request schema in `/act`
  - observed value: `instruction`
- `state_layout`
  - value source: request schema and server-side validation
  - observed value: `state(8)` = `joint_position(7) + gripper_position(1)`

Fields that were still benchmark-derived under the current flat schema:

- `state_projection_rule`
  - current workable rule: `concat(joint_position, gripper_position)`
- `ignored_current_schema_fields`
  - current example: `cartesian_position`

Phase-3 consequence:

- DROID is bridgeable, but the adapter template still needs an explicit place to declare dropped fields and projection rules

## MolmoAct2 YAM

Official source used in the phase-2 probe:

- repo: `molmoact2`
- commit: `804ba37`
- file: `examples/yam/host_server_yam.py`

Fields a future `MolmoAct2` YAM adapter should copy directly from upstream:

- `transport_family`
  - value source: FastAPI app + `json_numpy` usage in `host_server_yam.py`
  - observed value: `http+json_numpy`
- `checkpoint_family`
  - value source: `REPO_ID`
  - observed value: `allenai/MolmoAct2-BimanualYAM`
- `normalization_tag`
  - value source: `NORM_TAG`
  - observed value: `yam_dual_molmoact2`
- `default_chunk_size`
  - value source: `DEFAULT_NUM_STEPS`
  - observed value: `10`
- `camera_roles`
  - value source: request schema in `/act`
  - observed value: `[top_cam, left_cam, right_cam]` in fixed order
- `state_layout`
  - value source: `STATE_DIM` and request validation
  - observed value: `state(14)` for two 7-D arms

Fields that were benchmark-derived under the current flat schema and should stop being invented in Phase 3:

- `top/left/right_camera_mapping_rule`
- `missing_arm_padding_rule`
- `camera_duplication_rule`

Phase-3 consequence:

- the internal representation must be able to say “this embodiment has three ordered cameras and two arms” directly, without a projection hack

## Minimal Future Template Requirement

Phase 3 should encode the distinction above directly in adapter templates.

Every future policy-adapter template should have fields for:

- direct upstream defaults
  - transport family
  - request / modality schema source
  - camera roles and order
  - state stream names and layout
  - action stream names, horizons, and semantics
  - normalization tag or equivalent deployment metadata
- benchmark-derived projections
  - any state derivation rule
  - any camera remapping rule
  - any dropped-field rule
  - any lossy flattening rule

If a field cannot be sourced from upstream and is not declared as a benchmark-derived projection, the template is still too implicit.
