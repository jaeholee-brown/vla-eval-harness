"""Quarantined current-schema (flat-schema) path.

This package holds the historical single-active-arm bootstrap surface that the
harness was built around before the bimanual internal representation existed.
It is preserved for the Phase 1.5 fidelity oracle against openpi and as a
reference, not as a target shape for new adapters.

New adapters MUST target the bimanual surface in ``vla_harness.adapters`` and
the runner in ``vla_harness.runner.bimanual_runner``. Do not import from here
when authoring new code.
"""
