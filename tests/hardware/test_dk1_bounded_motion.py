from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("DK1_BOUNDED_MOTION") != "1",
    reason="Set DK1_BOUNDED_MOTION=1 and wire a real backend to run this bounded-motion test.",
)
def test_dk1_bounded_motion_placeholder():
    pytest.fail("Replace this placeholder with a real bounded-motion DK-1 smoke test once the hardware backend is wired.")
