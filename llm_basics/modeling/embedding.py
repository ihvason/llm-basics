"""Token embedding table."""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class Embedding(nn.Module):
    """Lookup table mapping token ids to dense vectors.

    Weights are drawn from a truncated normal with unit standard deviation,
    truncated at three standard deviations.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Create the embedding matrix.

        Args:
            num_embeddings: Vocabulary size (number of rows).
            embedding_dim: Width of each embedding vector.
            device: Device for the weight tensor.
            dtype: Dtype for the weight tensor.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        weight = torch.empty(num_embeddings, embedding_dim, **factory_kwargs)

        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        """Embed a batch of token ids.

        Args:
            token_ids: Integer tensor of arbitrary shape.

        Returns:
            Tensor with a trailing ``embedding_dim`` axis appended to the
            shape of ``token_ids``.
        """
        return self.weight[token_ids]
