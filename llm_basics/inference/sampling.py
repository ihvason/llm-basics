"""Sampling the next token from a single step of logits.

The pipeline is temperature scaling, then top-k truncation, then top-p
(nucleus) truncation, then a multinomial draw.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Draw one token id from unnormalized logits.

    Args:
        logits: Unnormalized logits of shape ``(vocab_size,)``. Modified in
            place by the truncation steps.
        temperature: Softmax temperature. Values at or below ``1e-5`` fall
            back to greedy decoding.
        top_k: Keep only the ``k`` highest-scoring candidates; ``<= 0`` keeps
            all of them.
        top_p: Keep the smallest prefix of candidates whose cumulative
            probability exceeds this value; ``>= 1.0`` keeps all of them.

    Returns:
        The sampled token id.
    """
    if temperature <= 1e-5:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature

    if top_k > 0:
        val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        k_min = val[-1]
        logits[logits < k_min] = -float("Inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right by one so the first token past the threshold is kept.
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
