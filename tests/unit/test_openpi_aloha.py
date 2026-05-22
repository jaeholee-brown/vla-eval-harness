"""CPU-only smoke test for the bimanual openpi ALOHA adapter."""

from __future__ import annotations

from typing import Any
from typing import Mapping

import numpy as np

from vla_harness.protocol import ArmObservationGroup
from vla_harness.protocol import ObservationPacket
from vla_harness.protocol import TemporalStateSequence
from vla_harness.protocol import TemporalVideoSequence
from vla_harness.adapters.policy.bimanual import BimanualPolicyAdapter
from vla_harness.adapters.policy.openpi_aloha import DEFAULT_ACTION_HORIZON
from vla_harness.adapters.policy.openpi_aloha import DEFAULT_CHECKPOINT_REF
from vla_harness.adapters.policy.openpi_aloha import DEFAULT_CONFIG_NAME
from vla_harness.adapters.policy.openpi_aloha import DEFAULT_PROMPT
from vla_harness.adapters.policy.openpi_aloha import OpenPIAlohaPolicyAdapter
from vla_harness.adapters.policy.openpi_aloha import OpenPIAlohaRuntimeConfig


class FakeOpenPIAlohaClient:
    """Captures the wire-format observation we send to the openpi server."""

    def __init__(self, *, action_horizon: int = DEFAULT_ACTION_HORIZON) -> None:
        self._action_horizon = action_horizon
        self.payloads: list[dict[str, Any]] = []
        self.reset_calls: list[dict[str, Any]] = []

    def get_server_metadata(self) -> Mapping[str, Any]:
        return {
            "config_name": DEFAULT_CONFIG_NAME,
            "checkpoint_ref": DEFAULT_CHECKPOINT_REF,
        }

    def infer(self, obs: dict[str, Any]) -> Mapping[str, Any]:
        self.payloads.append(obs)
        actions = np.tile(
            np.arange(14, dtype=np.float32),
            (self._action_horizon, 1),
        )
        return {"actions": actions, "policy_timing": {"infer_ms": 12.34}}

    def reset(self, reset_info: dict[str, Any]) -> Any:
        self.reset_calls.append(reset_info)
        return {"status": "ok"}


def _make_observation(instruction: str = DEFAULT_PROMPT) -> ObservationPacket:
    return ObservationPacket(
        video={
            "cam_high": TemporalVideoSequence(
                sample_indices=(0,),
                frames=(np.zeros((224, 224, 3), dtype=np.uint8),),
            ),
            "cam_left_wrist": TemporalVideoSequence(
                sample_indices=(0,),
                frames=(np.ones((224, 224, 3), dtype=np.uint8),),
            ),
            "cam_right_wrist": TemporalVideoSequence(
                sample_indices=(0,),
                frames=(np.full((224, 224, 3), 2, dtype=np.uint8),),
            ),
        },
        arms={
            "left_arm": ArmObservationGroup(
                streams={
                    "joint_position": TemporalStateSequence(
                        sample_indices=(0,),
                        values=(np.arange(7, dtype=np.float32),),
                    ),
                }
            ),
            "right_arm": ArmObservationGroup(
                streams={
                    "joint_position": TemporalStateSequence(
                        sample_indices=(0,),
                        values=(np.arange(7, 14, dtype=np.float32),),
                    ),
                }
            ),
        },
        language={"instruction": instruction},
        session_id="session-openpi-aloha",
    )


def test_constructs_and_satisfies_bimanual_protocol():
    adapter = OpenPIAlohaPolicyAdapter(OpenPIAlohaRuntimeConfig(), client=FakeOpenPIAlohaClient())
    # BimanualPolicyAdapter is a static Protocol; assert structural conformance.
    for method in (
        "assert_ready_for_benchmark",
        "build_manifest",
        "infer",
        "reset",
        "build_policy_metadata",
        "build_validation_metadata",
        "build_notes",
        "close",
    ):
        assert callable(getattr(adapter, method)), f"missing protocol method: {method}"
    assert BimanualPolicyAdapter is not None  # imported to anchor the protocol surface
    adapter.assert_ready_for_benchmark()


def test_build_manifest_matches_upstream_aloha_schema():
    adapter = OpenPIAlohaPolicyAdapter(client=FakeOpenPIAlohaClient())
    manifest = adapter.build_manifest()

    assert manifest.name == "openpi_aloha_pen_uncap"
    assert [arm.name for arm in manifest.arm_groups] == ["left_arm", "right_arm"]
    assert [stream.name for stream in manifest.ordered_video_streams()] == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    for state_spec in manifest.state_streams:
        assert state_spec.dim == 7
        assert state_spec.layout == "joint_plus_gripper"
    for action_spec in manifest.action_streams:
        assert action_spec.dim == 7
        assert action_spec.horizon == DEFAULT_ACTION_HORIZON
        assert action_spec.semantics.representation == "absolute"
        assert action_spec.semantics.domain == "joint"
    assert manifest.metadata["runtime_family"] == "openpi_websocket_server"
    assert manifest.metadata["camera_keys"] == ("cam_high", "cam_left_wrist", "cam_right_wrist")
    assert manifest.metadata["adapt_to_pi"] is True


def test_infer_produces_per_arm_chunks_with_correct_shapes():
    client = FakeOpenPIAlohaClient()
    adapter = OpenPIAlohaPolicyAdapter(client=client)

    action = adapter.infer(_make_observation())

    assert set(action.arms) == {"left_arm", "right_arm"}
    left_chunk = action.arms["left_arm"].streams["joint_position"]
    right_chunk = action.arms["right_arm"].streams["joint_position"]
    assert left_chunk.values.shape == (DEFAULT_ACTION_HORIZON, 7)
    assert right_chunk.values.shape == (DEFAULT_ACTION_HORIZON, 7)
    # Left half of the 14-D upstream action goes to the left arm, right half to right.
    assert np.array_equal(left_chunk.values[0], np.arange(7, dtype=np.float32))
    assert np.array_equal(right_chunk.values[0], np.arange(7, 14, dtype=np.float32))

    # Wire-format payload must match the openpi ALOHA repack contract.
    assert len(client.payloads) == 1
    payload = client.payloads[0]
    assert set(payload["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    for image in payload["images"].values():
        assert image.shape == (3, 224, 224)  # CHW, per AlohaInputs (server converts to HWC)
        assert image.dtype == np.uint8
    assert payload["state"].shape == (14,)
    assert payload["state"].dtype == np.float32
    assert np.array_equal(payload["state"], np.arange(14, dtype=np.float32))
    assert payload["prompt"] == DEFAULT_PROMPT


def test_fairness_metadata_is_fully_populated():
    adapter = OpenPIAlohaPolicyAdapter(client=FakeOpenPIAlohaClient())
    policy_meta = adapter.build_policy_metadata()

    assert policy_meta.family == "openpi"
    assert policy_meta.config_name == DEFAULT_CONFIG_NAME
    assert policy_meta.checkpoint_ref == DEFAULT_CHECKPOINT_REF
    assert policy_meta.chunk_size == DEFAULT_ACTION_HORIZON
    assert policy_meta.runtime_family == "openpi_websocket_server"
    assert policy_meta.schema_source.startswith("openpi/src/openpi/policies/aloha_policy.py")
    assert policy_meta.prompt_format_source.endswith("pi0_aloha_pen_uncap.default_prompt")
    assert policy_meta.action_space == "absolute_joint_pose_bimanual"
    assert policy_meta.image_preprocess.resize_resolution == (224, 224)
    assert policy_meta.image_preprocess.color_space == "rgb"
    assert policy_meta.image_preprocess.output_dtype == "uint8"

    notes = adapter.build_notes()
    topics = {note.topic for note in notes}
    expected_topics = {
        "policy.runtime_family",
        "policy.config_name",
        "policy.checkpoint_ref",
        "policy.prompt_format_source",
        "policy.chunk_size",
        "policy.camera_keys",
        "policy.state_layout",
        "policy.action_semantics",
        "policy.image_preprocess",
        "policy.adapt_to_pi",
    }
    assert expected_topics.issubset(topics)
    # Every decision in this adapter must be sourced to upstream openpi.
    assert all(note.status == "official" for note in notes)


def test_reset_forwards_to_client():
    client = FakeOpenPIAlohaClient()
    adapter = OpenPIAlohaPolicyAdapter(client=client)

    result = adapter.reset({"episode": 7})

    assert result == {"status": "ok"}
    assert client.reset_calls == [{"episode": 7}]
