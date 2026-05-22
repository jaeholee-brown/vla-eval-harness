# Current Flat Schema Gap Matrix

This spike forces the current RoboArena `PolicyServerConfig` / flat observation-action schema onto the first pressure-test models before the harness commits to a broader internal representation.

## Current schema summary

The current flat schema can express:

- one logical arm state via:
  - `observation/joint_position: (7,)`
  - `observation/cartesian_position: (6,)`
  - `observation/gripper_position: (1,)`
- one wrist camera stream, optionally stereo
- up to two exterior camera streams, optionally stereo
- one action chunk in one of:
  - `joint_position`
  - `joint_velocity`
  - `cartesian_position`
  - `cartesian_velocity`

It cannot express:

- multiple named arm groups
- mixed action heads in one packet
- embodiment-specific modality naming
- explicit history requirements
- transport-specific request contracts

## `OpenPI`

Fit:

- DROID-style single-arm inputs fit naturally.
- Current websocket/msgpack transport is already derived from `openpi`.
- Single-arm chunked action execution fits the current schema.

Distortion:

- None for the narrow DROID-style single-arm path.

Impossible:

- Honest multi-arm policy exposure through the current flat schema.

## `GR00T N1.7`

Fit:

- A subset of image streams can be flattened into wrist/exterior slots.
- Joint and end-effector state can be squeezed into the current state keys.

Distortion:

- `GR00T` uses named modality configs, not fixed flat keys.
- Relative end-effector delta actions are semantically richer than the current enum.
- Mixed semantics across action heads get flattened away.

Impossible:

- True representation of relative EEF delta actions without an explicit semantic tag.
- Multiple embodiment configs in one process with an in-process default.
- Honest multi-arm action/state grouping.

## `MolmoAct2-DROID`

Fit:

- The DROID server path maps reasonably onto wrist + one exterior camera + 8-D state.

Distortion:

- The official transport is FastAPI/HTTP, not websocket/msgpack.
- The embodiment-specific request schema is hidden if everything is relabeled into flat keys.

Impossible:

- Transport-neutral fidelity claims if the harness assumes websocket is part of the payload contract.

## `MolmoAct2-BimanualYAM`

Fit:

- The three camera streams can be partially projected into wrist/exterior slots.

Distortion:

- Camera order is embodiment-specific and semantically meaningful.
- Bimanual 14-D state must be collapsed into a single-arm state shape.

Impossible:

- Honest representation of true bimanual state and actions.
- Honest representation of embodiment-specific camera contracts.

## Immediate constraints on phase 1

- Phase 1 must stay single active arm only.
- `GR00T` must not be treated as a current-schema-native design target.
- `MolmoAct2` transport must stay adapter-local later; websocket is not the universal runtime.
- The future internal representation must separate:
  - payload semantics
  - transport
  - multi-arm grouping
  - action semantics such as relative EEF deltas
