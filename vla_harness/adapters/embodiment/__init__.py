"""Embodiment adapters.

Bimanual is the canonical surface. To author a new adapter, copy
``template_embodiment_adapter.py`` into this package and fill in the TODOs.

Concrete adapters (``YAMBimanualAdapter``, ``DK1BimanualAdapter``) are
intentionally not re-exported here — import them by their full module path to
keep this package free of per-adapter churn.
"""

from vla_harness.adapters.embodiment._sample_types import BimanualObservationSample
from vla_harness.adapters.embodiment.bimanual import BimanualEmbodimentAdapter
from vla_harness.adapters.embodiment.template_embodiment_adapter import EmbodimentTemplateConfig
from vla_harness.adapters.embodiment.template_embodiment_adapter import TemplateEmbodimentAdapter

__all__ = [
    "BimanualEmbodimentAdapter",
    "BimanualObservationSample",
    "EmbodimentTemplateConfig",
    "TemplateEmbodimentAdapter",
]
