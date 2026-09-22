"""Model layer: the minimal composable operators of the Transformer LM.

Organised bottom-up along the dependency chain:

- primitives: ``Linear``, ``Embedding``, ``RMSNorm``, ``SwiGLU``
- positions: ``RotaryPositionalEmbedding`` (RoPE)
- attention: ``softmax``, ``scaled_dot_product_attention``,
  ``CausalMultiHeadSelfAttention``
- composition: ``TransformerBlock``, ``TransformerLM``
"""

from .attention import scaled_dot_product_attention, softmax
from .embedding import Embedding
from .linear import Linear
from .mha import CausalMultiHeadSelfAttention
from .rms_norm import RMSNorm
from .rope import RotaryPositionalEmbedding
from .swiglu import SwiGLU
from .transformer import TransformerBlock, TransformerLM

__all__ = [
    "Embedding",
    "Linear",
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "softmax",
    "scaled_dot_product_attention",
    "CausalMultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerLM",
]
