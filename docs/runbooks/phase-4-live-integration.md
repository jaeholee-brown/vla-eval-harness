# Phase 4 Live Integration Runbook

This runbook starts where the CPU-only smoke tests end. Use it when you are on
a GPU machine or standing next to a real bimanual robot.

## 1. MolmoAct2-BimanualYAM on a GPU machine

Use the official `MolmoAct2-BimanualYAM` FastAPI server as the source of truth.

1. Install the official `molmoact2` environment.
2. Start the official YAM server path that backs `examples/yam/host_server_yam.py`.
3. Instantiate `MolmoAct2YAMPolicyAdapter` with:
   - `repo_id="allenai/MolmoAct2-BimanualYAM"`
   - `normalization_tag="yam_dual_molmoact2"`
   - the real server URL
4. Verify `assert_ready_for_benchmark()` passes.
5. Run one `BimanualRunner` episode with a fake embodiment first.
6. Then pair it with a real `YAMBimanualAdapter` using `YAMRobotEnvBackend`.

Success criteria:

- health check matches `repo_id`, `norm_tag`, `num_cameras`, and `state_dim`
- action packet is `(N, 7)` per arm
- fairness log records `runtime_family`, `schema_source`, and `normalization_tag`

## 2. YAM live embodiment validation

Use the official YAM `RobotEnv` shape from `launch_yaml_eval_molmoact.py`.

1. Bring up the real YAM environment.
2. Wrap it in `YAMRobotEnvBackend`.
3. Run `YAMBimanualAdapter.capture_observation()` once and inspect:
   - top/front image
   - left image
   - right image
   - left/right 7-D joint vectors
4. Run one no-motion action chunk through `execute_action()`.
5. Run one small bounded-motion chunk.

Success criteria:

- `front_camera_rgb` correctly becomes the harness `top` role
- `joint_positions[:7]` and `joint_positions[7:14]` map to left/right arms
- no static-padding path is triggered for honest bimanual policies

## 3. DK-1 live embodiment validation

Use the official `bi_dk1_follower` object as the source of truth.

1. Instantiate the real `BiDK1Follower`.
2. Wrap it in `LeRobotBiDK1Backend`.
3. Run `DK1BimanualAdapter.capture_observation()` once and inspect:
   - `head`
   - `right_wrist`
   - `left_wrist`
   - left/right 7-D joint vectors in official motor order
4. Run one no-motion action chunk through `execute_action()`.
5. Run one tiny bounded-motion chunk.

Success criteria:

- `head` aliasing to `top` works when a policy asks for `top`
- joint order is exactly:
  - `joint_1.pos`, `joint_2.pos`, `joint_3.pos`, `joint_4.pos`, `joint_5.pos`, `joint_6.pos`, `gripper.pos`
- fairness log records `control_hz=200.0` and official camera roles

## 4. GR00T managed-local-server validation

Use the official managed-local-server path, not an in-process shortcut.

1. Install the official `GR00T` environment.
2. Start the official `PolicyServer`.
3. Instantiate `GR00TPolicyAdapter` with:
   - the chosen checkpoint
   - the chosen `embodiment_tag`
   - bindings copied from the official modality config
4. Verify `assert_ready_for_benchmark()` passes.
5. Run one fake-embodiment `BimanualRunner` episode first.
6. Then pair it with a real bimanual embodiment once the modality config is honest for that embodiment.

Success criteria:

- `ping()` passes
- `get_modality_config()` matches the adapter bindings exactly
- video/state/action horizons are preserved from the modality config
- action semantics stay relative/absolute exactly as declared upstream

## 5. Closing the Phase 4 authoring claim

Phase 4 is only truly closed when one outside adapter author can use:

- the cookbook
- the skeletons
- the four real reference adapters

to add a fifth adapter without live architecture help.

The proof artifact should include:

1. the new adapter code
2. its CPU-only smoke test
3. a fairness log
4. a short note naming which upstream artifacts were copied directly
