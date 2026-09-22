"""Checkpoint serialization for model weights, optimizer state and step count."""

import os
import typing

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """Write model weights, optimizer state and the current iteration.

    Args:
        model: Model whose ``state_dict`` is saved.
        optimizer: Optimizer whose state is saved.
        iteration: Training step the checkpoint corresponds to.
        out: Destination path or writable binary file object.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """Restore model and optimizer state in place.

    Args:
        src: Source path or readable binary file object.
        model: Model to load the weights into.
        optimizer: Optimizer to load the state into.

    Returns:
        The iteration stored in the checkpoint, so training can resume from it.
    """
    checkpoint = torch.load(src, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
