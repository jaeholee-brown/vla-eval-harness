# Phase 1.5 Fidelity Runbook

This runbook walks through the three tests that gate Phase 2:

1. preprocessing parity on real captured frames
2. action parity with explicit tolerances
3. negative control that proves the parity battery can detect a wrong path

Until all three pass on a real `OpenPI + DK-1` setup, Phase 2 stays blocked.

## Prerequisites

You need three things on the machine running the tests:

1. The `openpi` Python package installed and importable.
2. A trained DROID checkpoint downloaded (or accessible via `gs://`).
3. The `lerobot` package installed for the fixture fetcher.

Install commands:

```bash
uv pip install openpi          # follow upstream openpi install docs if extras are required
uv pip install lerobot
```

The harness only depends on `openpi` and `lerobot` for fidelity testing,
not for unit tests, so the default `uv sync` install does not pull them in.

## Step 1: Fetch real DROID frames as fixtures

```bash
python scripts/fetch_droid_fixtures.py --num-frames 5
```

This downloads the public `lerobot/droid_100` dataset (MIT licensed,
~464MB) and writes:

- `fixtures/openpi_preprocess/frame_NNN.npy` — single uint8 RGB camera frames
- `fixtures/openpi_action/obs_NNN.npz` — full flat-schema observations

The `fixtures/` directory is gitignored. Re-running the script is safe;
existing files are overwritten.

## Step 2: Stand up the fidelity-mode openpi server

The action-parity test compares an in-process `Policy.infer(obs)` call
(`official_action`) against the same call routed over the websocket
(`harness_action`). pi05_droid is a flow-matching policy: each
`Policy.infer` call samples a fresh initial noise from its own RNG, so
two independent processes will produce structurally different action
chunks for the same input unless the noise tensor is pinned on both
legs. Stock `serve_policy.py` does not accept a noise kwarg, so the
harness ships its own thin server that pops `noise` out of the obs
dict and forwards it.

In a separate terminal (the test process needs GPU memory too, so disable
JAX preallocation):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
    uv run python scripts/serve_openpi_for_fidelity.py \
        --config pi05_droid \
        --checkpoint-dir gs://openpi-assets/checkpoints/pi05_droid \
        --port 8000
```

The harness adapter connects to `127.0.0.1:8000` by default; override
with `OPENPI_HARNESS_HOST` / `OPENPI_HARNESS_PORT` if you ran the server
elsewhere. The fidelity server is a drop-in replacement for openpi's
`serve_policy.py` only when noise is provided; if you reuse the stock
server for any reason the action parity test will not be meaningful.

## Step 3: Point the env vars at the callables

The fidelity tests load callables from environment variables. All point
into `vla_harness/eval/openpi_callables.py`. Save the block below to
`.env.fidelity` and `source` it before each run (the file is
gitignored):

```bash
# Strip ROS2 PYTHONPATH leak if you have one — otherwise pytest's plugin
# discovery imports /opt/ros/jazzy launch_testing into our venv and crashes
# before any test collection.
unset PYTHONPATH

# Stop JAX from preallocating 75% of the GPU. The fidelity tests load
# their own in-process Policy alongside the websocket server's Policy,
# so both processes need to share the device.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

# Shared
export OPENPI_CONFIG_NAME=pi05_droid
export OPENPI_CHECKPOINT_DIR=gs://openpi-assets/checkpoints/pi05_droid

# Server
export OPENPI_HARNESS_HOST=127.0.0.1
export OPENPI_HARNESS_PORT=8000

# Preprocessing parity
export OPENPI_PREPROCESS_FIXTURE_DIR=fixtures/openpi_preprocess
export OPENPI_OFFICIAL_PREPROCESS=vla_harness.eval.openpi_callables:official_preprocess
export OPENPI_HARNESS_PREPROCESS=vla_harness.eval.openpi_callables:harness_preprocess

# Action parity — tolerances earned on 2026-05-22 RTX 5090 run.
# Steady-state max abs diff was 3.3e-3 across 5 DROID fixtures with
# deterministic noise on both legs; suite-mode repeats occasionally saw
# up to ~1.1e-2 from cuDNN auto-tuner flutter between the two independent
# processes. 2e-2 was stable across 8 suite-mode repeats.
export OPENPI_ACTION_FIXTURE_DIR=fixtures/openpi_action
export OPENPI_OFFICIAL_ACTION_CALLABLE=vla_harness.eval.openpi_callables:official_action
export OPENPI_HARNESS_ACTION_CALLABLE=vla_harness.eval.openpi_callables:harness_action
export OPENPI_ACTION_PARITY_ATOL=2e-2
export OPENPI_ACTION_PARITY_RTOL=2e-2

# Negative control (uses the same action corpus)
export OPENPI_NEGATIVE_CONTROL_ACTION_CALLABLE=vla_harness.eval.openpi_callables:negative_control_action
export OPENPI_NEGATIVE_CONTROL=swap_rgb     # or zero_image, shuffle_prompt
export OPENPI_NEGATIVE_CONTROL_MIN_ABS_DIFF=1e-4
```

## Step 4: Run the three tests in order

```bash
source .env.fidelity
uv run pytest tests/fidelity/test_openpi_preprocessing_parity.py -v
uv run pytest tests/fidelity/test_openpi_action_parity.py -v
uv run pytest tests/fidelity/test_openpi_action_negative_control.py -v
```

Expected on a faithful harness:

- preprocessing parity passes (byte-identical, the two callables point at the same code)
- action parity passes within the configured tolerances
- negative control passes (deliberately wrong path is detected as different)

Last validated 2026-05-22 on a single RTX 5090 with pi05_droid:

- preprocessing parity: 0 / 5 fixtures violate `atol=0.0, rtol=0.0`
- action parity: 0 / 5 fixtures violate `atol=2e-2, rtol=2e-2` (stable
  across 8 suite-mode repeats; steady-state max abs diff is 3.3e-3,
  worst observed cold-cache excursion was 1.12e-2 — the residual is
  cuDNN auto-tuner divergence between two independent JAX processes)
- negative control: separates from official on all three strategies
  (`swap_rgb`, `zero_image`, `shuffle_prompt`) at the default
  `OPENPI_NEGATIVE_CONTROL_MIN_ABS_DIFF=1e-4`

## Step 5: Earn trust in the test itself

A parity test that always passes is useless. To earn confidence that the
test would actually catch a regression, run two negative-control loops by
hand:

1. Temporarily point `OPENPI_HARNESS_PREPROCESS` at a deliberately-wrong
   callable (e.g. an identity function, or one that swaps RGB channels).
   Re-run the preprocessing parity test. It MUST fail. Then revert.
   ```bash
   source .env.fidelity
   OPENPI_HARNESS_PREPROCESS=vla_harness.adapters.policy.openpi_current_schema:identity_preprocess \
       uv run pytest tests/fidelity/test_openpi_preprocessing_parity.py -v
   ```
2. Change `OPENPI_NEGATIVE_CONTROL` between `swap_rgb`, `zero_image`, and
   `shuffle_prompt` to confirm that any of these perturbations is large
   enough to trip the negative-control threshold.

Only once both fail-on-purpose checks succeed should you declare
preprocessing parity and action parity "earned."

Last validated 2026-05-22:

- preprocess fail-on-purpose with `identity_preprocess`: FAILED with
  `(shapes (180, 320, 3), (224, 224, 3) mismatch)` — the official path
  resizes to 224×224 and identity does not, so the assertion trips
  before any value comparison.
- negative-control sweep: all three of `swap_rgb`, `zero_image`,
  `shuffle_prompt` separate from the official path on the same DROID
  fixture corpus.

## Step 6: Record the result in a fairness log

For any benchmark run that follows, the fairness log will carry:

- `validation.preprocessing_oracle` — set by the adapter config
- `validation.preprocessing_allowed_atol` / `_rtol`
- `validation.action_oracle`
- `validation.action_allowed_atol` / `_rtol`
- `validation.passed`

Make sure these match the tolerances that actually passed in Step 4. If
you loosened tolerances to make the test pass, you have to log the
loosened values — that is the entire point of recording them.

## Troubleshooting

- `RuntimeError: OpenPI adapter claims official preprocessing in fairness
  metadata, but no explicit preprocess callable was wired.` — the runner
  guard fired. Either pass `preprocess_callable=` when constructing the
  adapter (the `_harness_adapter()` helper in `openpi_callables.py`
  already does this) or change `image_resize_filter` away from
  `"official_openpi_runtime"`.

- `Could not locate a preprocessed image in the openpi input transform
  output;` — openpi's internal transform output key changed. Inspect the
  dict returned by `policy._input_transform(...)` in your installed
  openpi version and add the correct key to the lookup in
  `_run_image_through_official_transforms`.

- Negative control fails to separate at `1e-4` — your perturbation is too
  small. Switch to `zero_image`, which is essentially guaranteed to
  produce divergent actions.

- Action parity fails with `Max relative difference ~40x` and 100% of
  elements mismatched — you are pointed at the stock openpi
  `serve_policy.py`, not `scripts/serve_openpi_for_fidelity.py`. The
  stock server ignores the `noise` key in obs, so the two legs sample
  independent flow-matching noise and cannot match.

- `jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED` during
  test collection — the server preallocated most of the GPU. Restart
  both the server and the tests with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  and a `XLA_PYTHON_CLIENT_MEM_FRACTION` that fits two copies of the
  model on your device.

- `ModuleNotFoundError: No module named 'lark'` during pytest startup
  — a ROS2 distribution's `PYTHONPATH` is leaking in. `unset PYTHONPATH`
  before running pytest.

## After Phase 1.5 closes

Only after all three tests pass and the two fail-on-purpose checks confirm
the tests can detect regressions, move on to the Phase 2 spike:

- stand up one `GR00T` `managed_local_server` runtime
- stand up one official `MolmoAct2` FastAPI server
- force-fit both through the current adapter shape
- write the observed pain report into `docs/pain/`
