"""Numerically stable cross-entropy loss."""

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """Cross-entropy between logits and integer targets.

    Computed through the log-sum-exp identity after subtracting the maximum
    logit, so it stays finite for large logits.

    Args:
        logits: Unnormalized log probabilities of shape ``(..., vocab_size)``.
        targets: Ground-truth class indices of shape ``(...)``.

    Returns:
        Scalar tensor with the loss averaged over all leading dimensions.
    """
    # Per-position maximum, used to keep exp() bounded: shape (...).
    max_logits = torch.max(logits, dim=-1, keepdim=True).values

    shifted_logits = logits - max_logits

    # log(sum(exp(shifted_logits))): shape (...).
    sum_exp = torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)

    # Gather the logit of the true class: shape (..., 1).
    targets_expanded = targets.unsqueeze(-1)
    target_shifted_logit = torch.gather(shifted_logits, dim=-1, index=targets_expanded)

    loss = log_sum_exp - target_shifted_logit

    return loss.mean()
