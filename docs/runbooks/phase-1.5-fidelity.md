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

## Step 2: Stand up an OpenPI websocket server

The harness adapter routes inference through `WebsocketClientPolicy`,
so a server has to be running on the host/port the adapter expects.

In a separate terminal:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid \
    --policy.dir=/path/to/your/checkpoint
```

(That `serve_policy.py` is the script from the `openpi` repo. Replace
`/path/to/your/checkpoint` with the checkpoint dir you downloaded.)

By default the harness adapter connects to `127.0.0.1:8000`, matching
openpi's default. Override with `OPENPI_HARNESS_HOST` / `OPENPI_HARNESS_PORT`
if you ran the server elsewhere.

## Step 3: Point the env vars at the callables

The fidelity tests load callables from environment variables. All point
into `vla_harness/eval/openpi_callables.py`:

```bash
# Shared
export OPENPI_CONFIG_NAME=pi05_droid
export OPENPI_CHECKPOINT_DIR=/path/to/your/checkpoint   # or a gs:// URI

# Preprocessing parity
export OPENPI_PREPROCESS_FIXTURE_DIR=fixtures/openpi_preprocess
export OPENPI_OFFICIAL_PREPROCESS=vla_harness.eval.openpi_callables:official_preprocess
export OPENPI_HARNESS_PREPROCESS=vla_harness.eval.openpi_callables:harness_preprocess

# Action parity
export OPENPI_ACTION_FIXTURE_DIR=fixtures/openpi_action
export OPENPI_OFFICIAL_ACTION_CALLABLE=vla_harness.eval.openpi_callables:official_action
export OPENPI_HARNESS_ACTION_CALLABLE=vla_harness.eval.openpi_callables:harness_action

# Tolerances (defaults are 1e-3 / 1e-3 — adjust based on observed bf16 noise)
export OPENPI_ACTION_PARITY_ATOL=1e-3
export OPENPI_ACTION_PARITY_RTOL=1e-3

# Negative control (uses the same action corpus)
export OPENPI_NEGATIVE_CONTROL_ACTION_CALLABLE=vla_harness.eval.openpi_callables:negative_control_action
export OPENPI_NEGATIVE_CONTROL=swap_rgb     # or zero_image, shuffle_prompt
export OPENPI_NEGATIVE_CONTROL_MIN_ABS_DIFF=1e-4
```

## Step 4: Run the three tests in order

```bash
pytest tests/fidelity/test_openpi_preprocessing_parity.py -v
pytest tests/fidelity/test_openpi_action_parity.py -v
pytest tests/fidelity/test_openpi_action_negative_control.py -v
```

Expected on a faithful harness:

- preprocessing parity passes (byte-identical, the two callables point at the same code)
- action parity passes within the configured tolerances
- negative control passes (deliberately wrong path is detected as different)

## Step 5: Earn trust in the test itself

A parity test that always passes is useless. To earn confidence that the
test would actually catch a regression, run two negative-control loops by
hand:

1. Temporarily point `OPENPI_HARNESS_PREPROCESS` at a deliberately-wrong
   callable (e.g. an identity function, or one that swaps RGB channels).
   Re-run the preprocessing parity test. It MUST fail. Then revert.
2. Change `OPENPI_NEGATIVE_CONTROL` between `swap_rgb`, `zero_image`, and
   `shuffle_prompt` to confirm that any of these perturbations is large
   enough to trip the negative-control threshold.

Only once both fail-on-purpose checks succeed should you declare
preprocessing parity and action parity "earned."

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

## After Phase 1.5 closes

Only after all three tests pass and the two fail-on-purpose checks confirm
the tests can detect regressions, move on to the Phase 2 spike:

- stand up one `GR00T` `managed_local_server` runtime
- stand up one official `MolmoAct2` FastAPI server
- force-fit both through the current adapter shape
- write the observed pain report into `docs/pain/`
