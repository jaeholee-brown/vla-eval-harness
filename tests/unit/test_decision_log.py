from __future__ import annotations

import json

from vla_harness.logging.decision_log import DecisionNote
from vla_harness.logging.decision_log import EmbodimentMetadata
from vla_harness.logging.decision_log import FairnessLog
from vla_harness.logging.decision_log import ImagePreprocessMetadata
from vla_harness.logging.decision_log import PolicyMetadata
from vla_harness.logging.decision_log import RuntimeMetadata


def test_fairness_log_writes_structured_fields(tmp_path):
    log = FairnessLog.create(
        run_id="run-1",
        prompt="pick up the object",
        session_id="session-1",
        policy=PolicyMetadata(
            family="openpi",
            config_name="pi05_droid",
            checkpoint_ref="gs://demo",
            checkpoint_sha256=None,
            checkpoint_sha256_explanation="fixture",
            dtype="bfloat16",
            device="cuda",
            action_space="joint_velocity",
            chunk_size=8,
            prompt_format_source="official_openpi_runtime",
            image_preprocess=ImagePreprocessMetadata(
                resize_resolution=(224, 224),
                resize_filter="bilinear",
                color_space="rgb",
                output_dtype="uint8",
            ),
        ),
        embodiment=EmbodimentMetadata(
            family="dk1",
            backend="fake",
            active_arm="right",
            parked_arm="left",
            parked_arm_rule="robot_native_hold",
            control_hz=15.0,
            chunk_consumption_policy="execute_full_chunk_open_loop",
        ),
        runtime=RuntimeMetadata(
            upstream_roboarena_commit="a07f93d",
            transport="websocket+msgpack",
            compression="none",
            concurrency_model="single_rollout_per_process",
        ),
    )
    log.add_note(
        DecisionNote(
            topic="test",
            choice="demo",
            status="adapter",
            rationale="unit test",
            evidence="tests/unit/test_decision_log.py",
        )
    )
    log.record_step(
        step_index=0,
        request_bytes=123,
        response_bytes=45,
        latency_ms=6.7,
        action_rows=1,
        action_dim=8,
    )
    log.finalize(validation_passed=True, max_abs_diff=0.0, max_rel_diff=0.0)

    path = log.write(tmp_path / "decision_log.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["policy"]["family"] == "openpi"
    assert payload["policy"]["image_preprocess"]["resize_filter"] == "bilinear"
    assert payload["embodiment"]["active_arm"] == "right"
    assert payload["runtime"]["upstream_roboarena_commit"] == "a07f93d"
    assert payload["validation"]["passed"] is True
    assert payload["step_metrics"][0]["request_bytes"] == 123
