from __future__ import annotations

import dataclasses

import numpy as np

from vla_harness.adapters.policy.gr00t import GR00TPolicyAdapter
from vla_harness.adapters.policy.gr00t import Gr00tActionBinding
from vla_harness.adapters.policy.gr00t import Gr00tLanguageBinding
from vla_harness.adapters.policy.gr00t import Gr00tRuntimeConfig
from vla_harness.adapters.policy.gr00t import Gr00tStateBinding
from vla_harness.adapters.policy.gr00t import Gr00tVideoBinding
from vla_harness.protocol import ArmGroupSpec
from vla_harness.protocol import ArmObservationGroup
from vla_harness.protocol import ObservationPacket
from vla_harness.protocol import TemporalStateSequence
from vla_harness.protocol import TemporalVideoSequence


@dataclasses.dataclass
class FakeEnumValue:
    value: str


@dataclasses.dataclass
class FakeActionConfig:
    rep: FakeEnumValue


@dataclasses.dataclass
class FakeModalityConfig:
    delta_indices: list[int]
    modality_keys: list[str]
    action_configs: list[FakeActionConfig] | None = None


class FakeGr00tClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def ping(self) -> bool:
        return True

    def get_modality_config(self) -> dict[str, FakeModalityConfig]:
        return {
            "video": FakeModalityConfig(delta_indices=[-15, 0], modality_keys=["top_cam", "wrist_left", "wrist_right"]),
            "state": FakeModalityConfig(delta_indices=[0], modality_keys=["left_arm", "right_arm"]),
            "action": FakeModalityConfig(
                delta_indices=[0, 1, 2, 3],
                modality_keys=["left_arm", "right_arm"],
                action_configs=[
                    FakeActionConfig(rep=FakeEnumValue("relative")),
                    FakeActionConfig(rep=FakeEnumValue("relative")),
                ],
            ),
            "language": FakeModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
        }

    def get_action(self, observation, options=None):
        del options
        self.payloads.append(observation)
        return (
            {
                "left_arm": np.zeros((1, 4, 7), dtype=np.float32),
                "right_arm": np.ones((1, 4, 7), dtype=np.float32),
            },
            {"server": "ok"},
        )

    def reset(self, options=None):
        return {"reset": options}

    def kill_server(self) -> None:
        return None


def _make_observation() -> ObservationPacket:
    return ObservationPacket(
        video={
            "top_camera": TemporalVideoSequence(
                sample_indices=(-15, 0),
                frames=(np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)),
            ),
            "left_wrist_camera": TemporalVideoSequence(
                sample_indices=(-15, 0),
                frames=(np.full((8, 8, 3), 2, dtype=np.uint8), np.full((8, 8, 3), 3, dtype=np.uint8)),
            ),
            "right_wrist_camera": TemporalVideoSequence(
                sample_indices=(-15, 0),
                frames=(np.full((8, 8, 3), 4, dtype=np.uint8), np.full((8, 8, 3), 5, dtype=np.uint8)),
            ),
        },
        arms={
            "left_arm": ArmObservationGroup(
                streams={"joint_position": TemporalStateSequence(sample_indices=(0,), values=(np.arange(7, dtype=np.float32),))}
            ),
            "right_arm": ArmObservationGroup(
                streams={"joint_position": TemporalStateSequence(sample_indices=(0,), values=(np.arange(7, 14, dtype=np.float32),))}
            ),
        },
        language={"instruction": "assemble"},
    )


def test_gr00t_adapter_smoke_and_manifest():
    adapter = GR00TPolicyAdapter(
        Gr00tRuntimeConfig(
            checkpoint_ref="nvidia/GR00T-N1.7-3B",
            embodiment_tag="NEW_EMBODIMENT",
            schema_source="examples/SO100/so100_config.py",
            arm_groups=(
                ArmGroupSpec(name="left_arm", side="left"),
                ArmGroupSpec(name="right_arm", side="right"),
            ),
            video_bindings=(
                Gr00tVideoBinding(manifest_name="top_camera", gr00t_key="top_cam", role="top"),
                Gr00tVideoBinding(manifest_name="left_wrist_camera", gr00t_key="wrist_left", role="left_wrist", arm_group="left_arm"),
                Gr00tVideoBinding(manifest_name="right_wrist_camera", gr00t_key="wrist_right", role="right_wrist", arm_group="right_arm"),
            ),
            state_bindings=(
                Gr00tStateBinding(manifest_name="joint_position", gr00t_key="left_arm", arm_group="left_arm", dim=7, layout="joint_position"),
                Gr00tStateBinding(manifest_name="joint_position", gr00t_key="right_arm", arm_group="right_arm", dim=7, layout="joint_position"),
            ),
            action_bindings=(
                Gr00tActionBinding(manifest_name="joint_position", gr00t_key="left_arm", arm_group="left_arm", dim=7, domain="joint", layout="joint_position"),
                Gr00tActionBinding(manifest_name="joint_position", gr00t_key="right_arm", arm_group="right_arm", dim=7, domain="joint", layout="joint_position"),
            ),
            language_binding=Gr00tLanguageBinding(
                manifest_name="instruction",
                gr00t_key="annotation.human.task_description",
            ),
        ),
        client=FakeGr00tClient(),
    )

    adapter.assert_ready_for_benchmark()
    manifest = adapter.build_manifest()
    action = adapter.infer(_make_observation())
    metadata = adapter.build_policy_metadata()

    assert manifest.video_streams[0].sample_indices == (-15, 0)
    assert manifest.action_streams[0].horizon == 4
    assert manifest.action_streams[0].semantics.representation == "relative"
    assert action.arms["left_arm"].streams["joint_position"].values.shape == (4, 7)
    assert metadata.runtime_family == "managed_local_server"
    assert metadata.normalization_tag == "NEW_EMBODIMENT"


def test_gr00t_adapter_can_static_pad_one_arm():
    adapter = GR00TPolicyAdapter(
        Gr00tRuntimeConfig(
            checkpoint_ref="nvidia/GR00T-N1.7-3B",
            embodiment_tag="NEW_EMBODIMENT",
            schema_source="examples/SO100/so100_config.py",
            arm_groups=(
                ArmGroupSpec(name="left_arm", side="left"),
                ArmGroupSpec(name="right_arm", side="right", control_role="static_pad_only"),
            ),
            video_bindings=(Gr00tVideoBinding(manifest_name="top_camera", gr00t_key="top_cam", role="top"),),
            state_bindings=(Gr00tStateBinding(manifest_name="joint_position", gr00t_key="left_arm", arm_group="left_arm", dim=7, layout="joint_position"),),
            action_bindings=(Gr00tActionBinding(manifest_name="joint_position", gr00t_key="left_arm", arm_group="left_arm", dim=7, domain="joint", layout="joint_position"),),
            language_binding=Gr00tLanguageBinding(
                manifest_name="instruction",
                gr00t_key="annotation.human.task_description",
            ),
            static_padding={"right_arm": "hold_static"},
        ),
        client=FakeGr00tClient(),
    )

    observation = ObservationPacket(
        video={"top_camera": TemporalVideoSequence(sample_indices=(-15, 0), frames=(np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)))},
        arms={"left_arm": ArmObservationGroup(streams={"joint_position": TemporalStateSequence(sample_indices=(0,), values=(np.arange(7, dtype=np.float32),))})},
        language={"instruction": "assemble"},
    )
    action = adapter.infer(observation)

    assert action.padding["right_arm"].strategy == "hold_static"
