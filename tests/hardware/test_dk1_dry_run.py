from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("DK1_DRY_RUN") != "1",
    reason="Set DK1_DRY_RUN=1 and wire a real backend to run this hardware smoke test.",
)
def test_dk1_dry_run_placeholder():
    pytest.fail("Replace this placeholder with a real DK-1 dry-run once the hardware backend is wired.")
