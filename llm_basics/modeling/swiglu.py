"""SwiGLU position-wise feed-forward network."""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def get_d_ff(d_model: int) -> int:
    """Derive the feed-forward width conventionally used with GLU variants.

    The width is ``8/3 * d_model`` rounded up to a multiple of 64, which keeps
    the parameter count close to a classic 4x MLP while staying friendly to
    memory alignment.

    Args:
        d_model: Model width.

    Returns:
        The feed-forward hidden width.
    """
    raw_d_ff = int(2 * 4 * d_model / 3)
    return ((raw_d_ff + 63) // 64) * 64


class SwiGLU(nn.Module):
    """Feed-forward block computing ``W_down(silu(W_gate x) * W_up x)``.

    Also known as the gated linear unit with SiLU activation; it replaces the
    usual two-matrix MLP in modern Transformer stacks.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Create the three projection matrices.

        Args:
            d_model: Model width, used for both the input and output size.
            d_ff: Hidden width; derived from ``d_model`` when omitted.
            device: Device for the weight tensors.
            dtype: Dtype for the weight tensors.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else get_d_ff(d_model)

        factory_kwargs = {"device": device, "dtype": dtype, "bias": False}
        # W_gate (w1), W_up (w3), W_down (w2)
        self.w_gate = nn.Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.w_up = nn.Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.w_down = nn.Linear(self.d_ff, self.d_model, **factory_kwargs)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Apply the gated feed-forward transformation.

        Args:
            x: Tensor of shape ``(..., d_model)``.

        Returns:
            Tensor of shape ``(..., d_model)``.
        """
        gate = self.w_gate(x)
        gate_act = gate * torch.sigmoid(gate)

        up = self.w_up(x)

        return self.w_down(gate_act * up)
