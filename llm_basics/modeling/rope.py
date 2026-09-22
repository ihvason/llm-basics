"""Rotary position embedding (RoPE).

Rotates query and key vectors by an angle proportional to their position, so
attention scores depend on relative distance. Cosine and sine tables are
precomputed once and reused for every forward pass.
"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary position embeddings to a tensor of head vectors.

    The embedding is parameter-free: only the cached cosine and sine tables
    are stored, and they are registered as non-persistent buffers so they stay
    out of the model's ``state_dict``.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """Precompute the rotation tables.

        Args:
            theta: Base wavelength of the geometric frequency ladder.
            d_k: Dimension of each attention head.
            max_seq_len: Longest sequence that can be looked up.
            device: Device for the cached tables.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Angular frequencies, one per rotated pair of channels: shape (d_k // 2,).
        dim_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (theta ** (dim_indices / d_k))

        # Rotation angle per (position, frequency) pair: shape (max_seq_len, d_k // 2).
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """Rotate ``x`` according to ``token_positions``.

        Args:
            x: Tensor of shape ``(..., seq_len, d_k)``.
            token_positions: Position index for each element of the sequence,
                broadcastable against ``x``'s leading dimensions.

        Returns:
            Rotated tensor of the same shape and dtype as ``x``.
        """
        # Look the cached tables up by position: (..., seq_len, d_k // 2).
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # The tables live in float32; match the activation dtype instead.
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # Split adjacent channel pairs: each of shape (..., seq_len, d_k // 2).
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Rotate each pair within its 2D plane.
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Re-interleave the pairs: stack to (..., seq_len, d_k // 2, 2), then flatten.
        x_out = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return x_out
