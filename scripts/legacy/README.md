# Legacy Scripts

These scripts target the quarantined current-schema (flat-schema) path and
remain runnable as references:

- `serve_openpi_for_fidelity.py` — launch a deterministic openpi server for the
  Phase 1.5 fidelity oracle.
- `fetch_droid_fixtures.py` — pull DROID frame fixtures used by the fidelity
  parity tests.
- `spike_gr00t_current_schema.py` — Phase 2 runtime spike that probed GR00T
  against the legacy current schema.
- `spike_molmoact2_current_schema.py` — same, for MolmoAct2.

New work should not depend on any of these.
