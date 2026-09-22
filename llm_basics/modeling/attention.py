"""Numerically stable softmax and scaled dot-product attention."""

import math

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """Softmax along ``dim``, stabilized against overflow.

    The maximum is subtracted before exponentiating, which keeps every
    exponent at or below zero.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to normalize.

    Returns:
        Tensor with the same shape as ``x``, normalized along ``dim``.
    """
    # Subtract the per-slice maximum so exp() cannot overflow.
    max_val = torch.max(x, dim=dim, keepdim=True).values
    shifted_x = x - max_val

    exp_x = torch.exp(shifted_x)

    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    q: Float[Tensor, "... seq_len_q d_k"],
    k: Float[Tensor, "... seq_len_k d_k"],
    v: Float[Tensor, "... seq_len_k d_v"],
    mask: Bool[Tensor, "seq_len_q seq_len_k"] | None = None,
) -> Float[Tensor, "... seq_len_q d_v"]:
    """Scaled dot-product attention with arbitrary leading batch dimensions.

    Args:
        q: Queries of shape ``(..., seq_len_q, d_k)``.
        k: Keys of shape ``(..., seq_len_k, d_k)``.
        v: Values of shape ``(..., seq_len_k, d_v)``.
        mask: Optional boolean mask of shape ``(seq_len_q, seq_len_k)``
            broadcastable over the batch dimensions. ``True`` allows
            attention, ``False`` masks the position out.

    Returns:
        Tensor of shape ``(..., seq_len_q, d_v)``.
    """
    d_k = q.shape[-1]

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Masked positions become -inf so softmax assigns them exactly zero.
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)

    return torch.matmul(attn_weights, v)
