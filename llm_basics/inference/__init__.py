"""Inference layer: autoregressive sampling and text generation.

- ``sampling.sample_next_token``: temperature scaling plus top-k and top-p
  (nucleus) truncation.
- ``generator.generate``: the autoregressive generation loop.
- ``generator.load_model_from_checkpoint``: checkpoint weight loading.
"""

from .generator import generate, load_model_from_checkpoint
from .sampling import sample_next_token

__all__ = ["sample_next_token", "generate", "load_model_from_checkpoint"]
