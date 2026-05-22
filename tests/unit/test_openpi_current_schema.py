from __future__ import annotations

import numpy as np

from vla_harness.adapters.policy.openpi_current_schema import OpenPICurrentSchemaAdapter
from vla_harness.adapters.policy.openpi_current_schema import OpenPIRuntimeConfig


class FakePolicyClient:
    def __init__(self) -> None:
        self.reset_payloads: list[dict[str, object]] = []
        self.infer_payloads: list[dict[str, object]] = []

    def get_server_metadata(self) -> dict[str, object]:
        return {"action_space": "joint_velocity"}

    def infer(self, obs: dict[str, object]) -> dict[str, object]:
        self.infer_payloads.append(obs)
        return {"actions": np.zeros((1, 8), dtype=np.float32)}

    def reset(self, reset_info: dict[str, object]) -> str:
        self.reset_payloads.append(reset_info)
        return "ok"


def test_openpi_adapter_forwards_explicit_reset_payload():
    client = FakePolicyClient()
    adapter = OpenPICurrentSchemaAdapter(
        OpenPIRuntimeConfig(config_name="pi05_droid"),
        client_factory=lambda host, port: client,
    )

    adapter.get_server_metadata()
    adapter.reset({"session_id": "abc123"})

    assert client.reset_payloads == [{"session_id": "abc123"}]


def test_openpi_adapter_exposes_structured_policy_metadata():
    adapter = OpenPICurrentSchemaAdapter(OpenPIRuntimeConfig(config_name="pi05_droid", chunk_size=8))
    metadata = adapter.build_policy_metadata()
    validation = adapter.build_validation_metadata()

    assert metadata.family == "openpi"
    assert metadata.config_name == "pi05_droid"
    assert metadata.chunk_size == 8
    assert validation.preprocessing_oracle is None
