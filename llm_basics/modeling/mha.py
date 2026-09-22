"""Causal multi-head self-attention with rotary position embeddings."""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .attention import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention that cannot attend to future positions.

    Queries, keys and values are projected from the same input, split into
    heads, rotated with RoPE, and combined through a lower-triangular mask.
    All four projections are bias-free.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Create the projections and, optionally, an internal RoPE module.

        Args:
            d_model: Model width; must be divisible by ``num_heads``.
            num_heads: Number of attention heads.
            max_seq_len: Longest sequence for the internal RoPE tables. When
                ``None`` no internal RoPE is created and a module must be
                passed to :meth:`forward` instead.
            theta: RoPE wavelength base.
            device: Device for the parameters.
            dtype: Dtype for the parameters.

        Raises:
            AssertionError: If ``d_model`` is not divisible by ``num_heads``.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        if max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Attend each position to itself and all earlier positions.

        Args:
            x: Input of shape ``(..., seq_len, d_model)``.
            token_positions: Position of each token, used by RoPE. Defaults to
                ``0, 1, ...`` when omitted.
            rope: RoPE module to use; takes precedence over the internal one.

        Returns:
            Tensor of shape ``(..., seq_len, d_model)``.
        """
        *batch_dims, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split the model dimension into heads: (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k).
        q = q.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        v = v.view(*batch_dims, seq_len, self.num_heads, self.d_v).transpose(-3, -2)

        # RoPE is applied to queries and keys only; the head axis broadcasts as a batch dimension.
        active_rope = rope if rope is not None else self.rope
        if active_rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
                token_positions = token_positions.expand(*batch_dims, seq_len)

            # Add a head axis so positions broadcast against (..., num_heads, seq_len, d_k).
            token_positions_with_head = token_positions.unsqueeze(-2)
            q = active_rope(q, token_positions_with_head)
            k = active_rope(k, token_positions_with_head)

        # Lower-triangular mask: position i may attend to positions j <= i.
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        )

        attn_out = scaled_dot_product_attention(q=q, k=k, v=v, mask=causal_mask)

        # Merge heads back into the model dimension: (..., num_heads, seq_len, d_v) -> (..., seq_len, d_model).
        attn_out = (
            attn_out.transpose(-3, -2)
            .contiguous()
            .view(*batch_dims, seq_len, self.d_model)
        )

        return self.out_proj(attn_out)
