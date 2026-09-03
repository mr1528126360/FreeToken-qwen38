"""GLM-5.3-Flash (model_type glm5_next). Text-only by default; set
``FREETOKEN_LOAD_VISION=1`` to also build/load the ``model.visual.*`` 24-block ViT
and serve image inputs (see vision.py).
"""

from .config import parse_config
from .model import Glm5NextForCausalLM
from .weight import iter_visual_weights, iter_weights, load_nvfp4_expert_sources

__all__ = [
    "Glm5NextForCausalLM",
    "iter_visual_weights",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "parse_config",
]
