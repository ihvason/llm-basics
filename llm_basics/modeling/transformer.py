"""Transformer block and the full decoder-only language model."""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .embedding import Embedding
from .linear import Linear
from .mha import CausalMultiHeadSelfAttention
from .rms_norm import RMSNorm
from .rope import RotaryPositionalEmbedding
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """One pre-norm block: attention sub-layer plus feed-forward sub-layer.

    Each sub-layer normalizes its input before transforming it and adds its
    output back to the residual stream.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Build the two sub-layers and their normalizations.

        Args:
            d_model: Model width.
            num_heads: Number of attention heads.
            d_ff: Hidden width of the feed-forward network.
            device: Device for the parameters.
            dtype: Dtype for the parameters.
        """
        super().__init__()
        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Run the block over a sequence of hidden states.

        Args:
            x: Input of shape ``(..., seq_len, d_model)``.
            token_positions: Position of each token, forwarded to attention.
            rope: RoPE module shared across blocks.

        Returns:
            Tensor of shape ``(..., seq_len, d_model)``.
        """
        norm_x = self.attn_norm(x)
        attn_out = self.attn(norm_x, token_positions=token_positions, rope=rope)
        x = x + attn_out

        norm_x = self.ffn_norm(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model returning next-token logits.

    The stack is: token embedding, ``num_layers`` pre-norm blocks sharing one
    RoPE module, a final RMSNorm, and a bias-free linear head projecting to
    the vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Build the embedding, the block stack and the output head.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            context_length: Maximum sequence length the model accepts.
            d_model: Model width.
            num_layers: Number of Transformer blocks.
            num_heads: Number of attention heads per block.
            d_ff: Hidden width of each feed-forward network.
            theta: RoPE wavelength base.
            device: Device for the parameters.
            dtype: Dtype for the parameters.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: Int[Tensor, "... seq_len"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        """Compute next-token logits for every position.

        Args:
            token_ids: Token ids of shape ``(..., seq_len)``.
            token_positions: Position of each token, defaulting to
                ``0, 1, ...`` along the last axis.

        Returns:
            Logits of shape ``(..., seq_len, vocab_size)``.

        Raises:
            AssertionError: If the sequence is longer than ``context_length``.
        """
        seq_len = token_ids.shape[-1]
        assert seq_len <= self.context_length, (
            f"Sequence length {seq_len} exceeds context length {self.context_length}"
        )

        if token_positions is None:
            token_positions = torch.arange(
                seq_len, device=token_ids.device, dtype=torch.long
            )
            token_positions = token_positions.expand_as(token_ids)

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions, rope=self.rope)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
