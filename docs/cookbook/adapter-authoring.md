# Adapter Authoring Cookbook

This cookbook exists to make adding a new VLA adapter or embodiment adapter a
translation job, not a design job.

## 1. Authoring Contract

### 1.1 What a completed adapter must ship

Every new adapter must ship with all of the following:

1. one adapter module copied from the relevant skeleton
2. one CPU-only unit test that proves:
   - the adapter constructs
   - the adapter satisfies the harness Protocol
   - dummy observations or actions produce the right shapes
   - the fairness log metadata is fully populated
3. one fairness-log note set that records every benchmark-derived choice
4. one parity or replay battery plan, even if the full oracle needs a GPU or hardware later

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 1.2 Source-of-truth policy

Every field in an adapter must be tagged mentally as one of:

- `copy_from_upstream`
- `benchmark_derived`
- `scoped_out`

The default rule is simple: if the release blog, repo, model card, config file,
or SDK names a value explicitly, copy that value. Only invent a
`benchmark_derived` value when the upstream artifacts are silent or when the
harness is doing the one allowed single-arm-on-bimanual bridge.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

## 2. New VLA Adapter

### 2.1 Gather the upstream artifacts first

Before touching code, collect:

1. the official repo or model card
2. the official inference entrypoint or server
3. the request schema or modality-config source file
4. the checkpoint identifier
5. the normalization tag, embodiment tag, or equivalent deployment metadata
6. the camera order and state/action key names

Examples already in this repo:

- `OpenPI`: historical flat-schema bootstrap and parity oracle
- `MolmoAct2-BimanualYAM`: official FastAPI server with ordered cameras and a norm tag
- `GR00T`: official managed local server plus modality config

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 2.2 Fill the policy config block before writing logic

Start from [vla_harness/adapters/policy/template_policy_adapter.py](/Users/jaeholee0404/roboarena/vla_harness/adapters/policy/template_policy_adapter.py).

Fill these fields first:

- `policy_family`
- `runtime_family`
- `schema_source`
- `checkpoint_ref`
- `normalization_tag`
- `prompt_format_source`
- `dtype`
- `device`

If the policy is single-arm only, fill exactly two bridge fields:

- which arm it controls
- which static padding rule keeps the other arm still

Do not write runtime logic until these fields are complete and sourced.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 2.3 Build the manifest directly from the upstream schema

`build_manifest()` should be a nearly mechanical projection of the official
request schema or modality config:

- `VideoStreamSpec` entries from the official camera keys and order
- `StateStreamSpec` entries from official state keys and dimensions
- `ActionStreamSpec` entries from official action keys, horizon, and semantics
- `LanguageFieldSpec` from the official instruction key

For `GR00T`, the modality config is the source of truth for:

- `sample_indices`
- action horizon
- action representation

For `MolmoAct2-BimanualYAM`, the server schema is the source of truth for:

- `top_cam`, `left_cam`, `right_cam`
- 14-D concatenated state
- 14-D action chunk

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 2.4 Convert `ObservationPacket` into the official runtime input

`infer()` must do the smallest possible translation:

- reorder or rename cameras exactly as the official runtime expects
- concatenate or split state only when the official runtime requires it
- preserve temporal sampling from the manifest
- preserve the official language key and shape

Do not normalize, resize, or reorder anything unless the upstream runtime
itself requires it.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 2.5 Convert the official runtime output back into `ActionPacket`

Action decoding should also be mechanical:

- map official action keys to harness arm groups
- preserve the official action horizon
- preserve official absolute vs relative semantics
- only use `PaddingRule` when a single-arm policy is bridged onto a bimanual setup

Do not silently reinterpret relative EEF deltas as absolute joint targets, or
vice versa.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 2.6 Ship the CPU-only smoke test

Every policy adapter must ship with a CPU-only unit test patterned after the
unit tests already in this repo. The test must prove:

1. adapter construction works
2. `assert_ready_for_benchmark()` is meaningful
3. `build_manifest()` returns a valid `HarnessManifest`
4. dummy `ObservationPacket` input produces action chunks with the expected shape
5. `build_policy_metadata()` and `build_notes()` fully populate the fairness log surface

The smoke test is the fast “did I wire this right” check. The parity oracle can
be slower and can depend on a GPU or real checkpoints later.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 2.7 Worked example: openpi `pi0_aloha_pen_uncap`

This adapter is the canonical worked example of the rules above: a true-bimanual
openpi policy wired through the official websocket server, with every field
sourced from upstream openpi code.

Upstream artifacts used:

- repo: `Physical-Intelligence/openpi`
- training config: `src/openpi/training/config.py` entry `pi0_aloha_pen_uncap`
  (`default_prompt="uncap the pen"`, three-camera repack, Trossen asset_id)
- model defaults: `src/openpi/models/pi0_config.py` (`action_horizon=50`)
- request schema: `src/openpi/policies/aloha_policy.py`
  (`AlohaInputs(adapt_to_pi=True)` and `AlohaOutputs(adapt_to_pi=True)`,
  14-D state and action with `[arm6, gripper1, arm6, gripper1]`)
- runtime: `docs/remote_inference.md` plus `scripts/serve_policy.py`
- checkpoint: `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`

What the adapter copies verbatim from upstream and tags as `official`:

- `config_name = "pi0_aloha_pen_uncap"`
- `checkpoint_ref = "gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap"`
- `chunk_size / action_horizon = 50`
- camera keys `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- state and action layout per arm: `joint_plus_gripper`, dim 7
- action semantics: `absolute` / `joint`
- image preprocess: `resize_with_pad` at 224×224, uint8 RGB, CHW on the wire
- `adapt_to_pi = True`

Minimal local launch (real GPU machine):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_aloha_pen_uncap \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap
```

Driving the adapter from harness code:

```python
from vla_harness.adapters.policy.openpi_aloha import OpenPIAlohaPolicyAdapter
from vla_harness.adapters.policy.openpi_aloha import OpenPIAlohaRuntimeConfig

adapter = OpenPIAlohaPolicyAdapter(
    OpenPIAlohaRuntimeConfig(host="127.0.0.1", port=8000),
)
adapter.assert_ready_for_benchmark()
manifest = adapter.build_manifest()
action_packet = adapter.infer(observation_packet)  # ObservationPacket built per manifest
```

CPU-only smoke test pattern: see
[tests/unit/test_openpi_aloha.py](/Users/jaeholee0404/roboarena/tests/unit/test_openpi_aloha.py).
It injects a `FakeOpenPIAlohaClient` (no GPU, no network) and checks construction,
protocol conformance, manifest shape, per-arm chunk shapes from the upstream 14-D
action, and that every fairness-log decision is tagged `official`.

If you need to bridge a single-arm openpi checkpoint onto a bimanual embodiment,
do **not** extend this adapter — use the single-arm bridge path instead and
record the static-padding rule explicitly, per §3.4.

## 3. New Embodiment Adapter

### 3.1 Gather the embodiment artifacts first

Before touching code, collect:

1. the official SDK, runtime, or robot backend
2. camera layout docs or config files
3. available proprio/state APIs
4. published control frequency
5. published action consumption or teleop cadence

Examples already in this repo:

- `YAM`: left/front/right cameras at 30 Hz from the published YAML configs
- `DK-1`: `bi_dk1_follower` with head/right_wrist/left_wrist cameras and a 200 Hz teleop example

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 3.2 Fill the embodiment config block first

Start from [vla_harness/adapters/embodiment/template_embodiment_adapter.py](/Users/jaeholee0404/roboarena/vla_harness/adapters/embodiment/template_embodiment_adapter.py).

Fill these fields before writing logic:

- `embodiment_family`
- `backend_name`
- `control_hz`
- `chunk_consumption_policy`
- camera role order
- arm group names
- state sources

Only `static_padding_rule` is expected to be benchmark-derived when you are
bridging a single-arm policy onto a bimanual embodiment.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 3.3 Build `capture_observation()` as an honest projection of the hardware

`capture_observation()` should:

- read the official cameras in their published order
- map each role through an explicit alias table when needed
- read per-arm state streams without fabricating missing values
- preserve temporal sample counts required by the manifest

Examples:

- YAM `top` is a deliberate alias to the published `front_camera`
- DK-1 `top` is a deliberate alias to the published `head` camera

Those aliases belong in config and fairness notes, not in hidden code.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 3.4 Build `execute_action()` so action ownership is explicit

`execute_action()` must:

- pass policy-controlled arm streams directly to the official control backend
- apply static-padding behavior only to the uncontrolled arm
- never invent motion for an arm not covered by the policy

The only allowed benchmark-side bridge is:

- one arm receives real policy outputs
- the other arm receives an explicit static padding rule

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 3.5 Ship the CPU-only smoke test

Every embodiment adapter must ship with a CPU-only unit test that proves:

1. adapter construction works
2. `capture_observation()` returns a protocol-valid `ObservationPacket`
3. `execute_action()` accepts protocol-valid `ActionPacket`s
4. role aliases and arm-group names are wired correctly
5. `build_embodiment_metadata()` and `build_notes()` fully populate the fairness log surface

Use fake backends and dummy frames/states. This test should run in seconds.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

## 4. Evaluation and Validation

### 4.1 Start with the smoke test, then add parity

Use [vla_harness/eval/_skeleton.py](/Users/jaeholee0404/roboarena/vla_harness/eval/_skeleton.py) after the CPU smoke test is green.

The order is:

1. CPU smoke test
2. replay battery
3. official parity oracle
4. negative control

Do not start with the GPU oracle when a broken constructor or shape mismatch can
be detected in two seconds on CPU.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.

### 4.2 Negative controls are mandatory

Every parity battery must include one deliberate break path that proves the
battery can fail. Examples:

- permute camera order
- shift one state dimension
- claim the wrong chunk horizon
- feed the wrong prompt key

The harness should never accept a parity battery that only ever passes.

If you can't fill this in from upstream docs alone, the protocol design has failed and we need to revisit it.
