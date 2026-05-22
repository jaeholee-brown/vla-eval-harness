from __future__ import annotations

import numpy as np

from vla_harness.adapters.policy.molmoact2_yam import MolmoAct2YAMPolicyAdapter
from vla_harness.adapters.policy.molmoact2_yam import MolmoAct2YAMRuntimeConfig
from vla_harness.protocol import ArmObservationGroup
from vla_harness.protocol import ObservationPacket
from vla_harness.protocol import TemporalStateSequence
from vla_harness.protocol import TemporalVideoSequence


class FakeMolmoActClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def health(self) -> dict[str, object]:
        return {
            "repo_id": "allenai/MolmoAct2-BimanualYAM",
            "norm_tag": "yam_dual_molmoact2",
            "num_cameras": 3,
            "state_dim": 14,
        }

    def act(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        return {
            "actions": np.arange(28, dtype=np.float32).reshape(2, 14),
            "dt_ms": 12.5,
        }


def _make_observation() -> ObservationPacket:
    return ObservationPacket(
        video={
            "top_camera": TemporalVideoSequence(sample_indices=(0,), frames=(np.zeros((8, 8, 3), dtype=np.uint8),)),
            "left_camera": TemporalVideoSequence(sample_indices=(0,), frames=(np.ones((8, 8, 3), dtype=np.uint8),)),
            "right_camera": TemporalVideoSequence(sample_indices=(0,), frames=(np.full((8, 8, 3), 2, dtype=np.uint8),)),
        },
        arms={
            "left_arm": ArmObservationGroup(
                streams={"joint_position": TemporalStateSequence(sample_indices=(0,), values=(np.arange(7, dtype=np.float32),))}
            ),
            "right_arm": ArmObservationGroup(
                streams={"joint_position": TemporalStateSequence(sample_indices=(0,), values=(np.arange(7, 14, dtype=np.float32),))}
            ),
        },
        language={"instruction": "stack blocks"},
        session_id="session-1",
    )


def test_molmoact2_yam_adapter_constructs_manifest_and_infers():
    client = FakeMolmoActClient()
    adapter = MolmoAct2YAMPolicyAdapter(MolmoAct2YAMRuntimeConfig(), client=client)

    adapter.assert_ready_for_benchmark()
    manifest = adapter.build_manifest()
    action = adapter.infer(_make_observation())
    metadata = adapter.build_policy_metadata()

    assert [stream.name for stream in manifest.ordered_video_streams()] == ["top_camera", "left_camera", "right_camera"]
    assert action.arms["left_arm"].streams["joint_position"].values.shape == (2, 7)
    assert action.arms["right_arm"].streams["joint_position"].values.shape == (2, 7)
    assert np.array_equal(client.payloads[0]["state"], np.arange(14, dtype=np.float32))
    assert metadata.runtime_family == "fastapi+json_numpy"
    assert metadata.schema_source == "examples/yam/host_server_yam.py"
    assert metadata.normalization_tag == "yam_dual_molmoact2"
