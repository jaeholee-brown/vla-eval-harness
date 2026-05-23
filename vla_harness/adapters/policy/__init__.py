"""Policy adapters.

Bimanual is the canonical surface. To author a new adapter, copy
``template_policy_adapter.py`` into this package and fill in the TODOs.

Concrete adapters (``GR00TPolicyAdapter``, ``OpenPIAlohaPolicyAdapter``,
``MolmoAct2YAMPolicyAdapter``) are intentionally not re-exported here — import
them by their full module path to keep this package free of per-adapter churn.
"""

from vla_harness.adapters.policy.bimanual import BimanualPolicyAdapter
from vla_harness.adapters.policy.template_policy_adapter import PolicyTemplateConfig
from vla_harness.adapters.policy.template_policy_adapter import TemplatePolicyAdapter

__all__ = [
    "BimanualPolicyAdapter",
    "PolicyTemplateConfig",
    "TemplatePolicyAdapter",
]
