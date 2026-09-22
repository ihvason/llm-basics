"""Linear layer with truncated-normal initialization and no bias term."""

import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """A bias-free linear projection, ``y = x @ W.T``.

    Weights are sampled from a truncated normal with standard deviation
    ``sqrt(2 / (in_features + out_features))``, which keeps activations at a
    stable scale across the depth of the network.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Create the weight matrix and initialize it.

        Args:
            in_features: Size of the last input dimension.
            out_features: Size of the last output dimension.
            device: Device for the weight tensor.
            dtype: Dtype for the weight tensor.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        weight = torch.empty(out_features, in_features, **factory_kwargs)

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` along its last dimension.

        Args:
            x: Tensor of shape ``(..., in_features)``.

        Returns:
            Tensor of shape ``(..., out_features)``.
        """
        return torch.einsum("... i, o i -> ... o", x, self.weight)
