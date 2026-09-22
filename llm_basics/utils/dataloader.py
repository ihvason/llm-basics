"""Batch sampling from a flat, pre-tokenized dataset."""

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[
    Int[Tensor, "batch_size context_length"],
    Int[Tensor, "batch_size context_length"],
]:
    """Sample input sequences and their next-token targets from a token array.

    Targets are the inputs shifted one position to the left, so a single
    memory-mapped array is enough to train a next-token predictor.

    Args:
        dataset: One-dimensional array of token ids.
        batch_size: Number of sequences to draw.
        context_length: Length of each drawn sequence.
        device: Target PyTorch device string, e.g. ``"cpu"``, ``"cuda:0"``, ``"mps"``.

    Returns:
        An ``(inputs, targets)`` pair, each of shape
        ``(batch_size, context_length)`` and placed on ``device``.
    """
    n = len(dataset)

    max_start_idx = n - context_length

    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    inputs = np.stack([dataset[i : i + context_length] for i in start_indices]).astype(np.int64)
    targets = np.stack(
        [dataset[i + 1 : i + context_length + 1] for i in start_indices]
    ).astype(np.int64)

    inputs_tensor = torch.from_numpy(inputs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)

    return inputs_tensor, targets_tensor
