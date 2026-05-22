from __future__ import annotations

import pytest

from vla_harness.protocol.manifest import ActionSemantics
from vla_harness.protocol.manifest import ActionStreamSpec
from vla_harness.protocol.manifest import ArmGroupSpec
from vla_harness.protocol.manifest import HarnessManifest
from vla_harness.protocol.manifest import StateStreamSpec
from vla_harness.protocol.manifest import VideoStreamSpec


def test_harness_manifest_requires_exactly_two_arm_groups():
    with pytest.raises(ValueError, match="exactly two arm groups"):
        HarnessManifest(
            name="bad",
            version="1",
            arm_groups=(ArmGroupSpec(name="left_arm", side="left"),),
            video_streams=(),
            state_streams=(),
            action_streams=(),
        )


def test_harness_manifest_orders_video_streams():
    manifest = HarnessManifest(
        name="ok",
        version="1",
        arm_groups=(
            ArmGroupSpec(name="left_arm", side="left"),
            ArmGroupSpec(name="right_arm", side="right"),
        ),
        video_streams=(
            VideoStreamSpec(name="wrist", role="wrist_rgb", order_index=2, arm_group="right_arm"),
            VideoStreamSpec(name="top", role="top_rgb", order_index=0),
            VideoStreamSpec(name="left_ctx", role="context_rgb", order_index=1, arm_group="left_arm"),
        ),
        state_streams=(
            StateStreamSpec(name="joint_position", arm_group="left_arm", dim=7, layout="joint_position"),
            StateStreamSpec(name="joint_position", arm_group="right_arm", dim=7, layout="joint_position"),
        ),
        action_streams=(
            ActionStreamSpec(
                name="policy_action",
                arm_group="left_arm",
                dim=8,
                semantics=ActionSemantics(
                    representation="velocity",
                    domain="joint",
                    layout="joint_plus_gripper",
                ),
            ),
            ActionStreamSpec(
                name="policy_action",
                arm_group="right_arm",
                dim=8,
                semantics=ActionSemantics(
                    representation="velocity",
                    domain="joint",
                    layout="joint_plus_gripper",
                ),
            ),
        ),
    )

    assert [spec.name for spec in manifest.ordered_video_streams()] == ["top", "left_ctx", "wrist"]
