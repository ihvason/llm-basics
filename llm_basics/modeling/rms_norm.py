"""Root mean square layer normalization."""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):
    """Normalize activations by their root mean square, then scale by a gain.

    Unlike LayerNorm this subtracts no mean. The statistics are computed in
    float32 and cast back, which keeps the reduction accurate in low-precision
    runs.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Create the learnable gain vector.

        Args:
            d_model: Size of the normalized (last) dimension.
            eps: Constant added to the mean square for numerical stability.
            device: Device for the gain tensor.
            dtype: Dtype for the gain tensor.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Normalize the last dimension of ``x``.

        Args:
            x: Tensor of shape ``(..., d_model)``.

        Returns:
            Tensor of the same shape and dtype as ``x``.
        """
        in_dtype = x.dtype

        x_fp32 = x.to(torch.float32)

        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + self.eps)

        normed = (x_fp32 * rms).to(in_dtype)
        return normed * self.weight
