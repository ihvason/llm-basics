"""Autoregressive text generation and checkpoint loading."""

from __future__ import annotations

from typing import Any

import torch

from ..modeling.transformer import TransformerLM
from ..tokenizer import BPETokenizer
from .sampling import sample_next_token


def load_model_from_checkpoint(
    checkpoint_path: str,
    model: TransformerLM,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> TransformerLM:
    """Load model weights from a checkpoint file into an existing model.

    Three state-dict layouts are accepted: ``{"model": ...}``,
    ``{"model_state_dict": ...}``, and a bare state dict.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model built with the matching architecture, weights not yet loaded.
        device: ``map_location`` passed to :func:`torch.load`.
        strict: Whether the state dict must match the model exactly.

    Returns:
        The same model instance, with weights loaded.
    """
    checkpoint: Any = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return model


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_ids: set[int] | None = None,
    stream: bool = True,
) -> str:
    """Generate text by repeatedly feeding the whole context back in.

    There is no KV cache: each step runs a fresh forward pass over the last
    ``context_length`` tokens. Adding one would change the contract of
    :meth:`TransformerLM.forward` and is out of scope here.

    Args:
        model: Language model with weights already loaded.
        tokenizer: Tokenizer used to encode the prompt and decode the output.
        prompt: Seed text.
        max_new_tokens: Upper bound on newly generated tokens.
        context_length: Model context window; earlier tokens are dropped when
            the sequence grows past it.
        device: Device to run generation on.
        temperature: Sampling temperature.
        top_k: Top-k truncation threshold.
        top_p: Nucleus sampling threshold.
        eos_token_ids: Token ids that stop generation when produced.
        stream: Whether to print each token as it is produced.

    Returns:
        The full text, prompt included.
    """
    model.eval()

    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]

    token_ids = list(input_ids)
    if stream:
        print(f"\n[Prompt]: {prompt}", end="", flush=True)

    for _ in range(max_new_tokens):
        context_ids = token_ids[-context_length:]
        x = torch.tensor([context_ids], dtype=torch.long, device=device)

        logits = model(x)
        next_logits = logits[0, -1, :]

        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if eos_token_ids is not None and next_token in eos_token_ids:
            break

        token_ids.append(next_token)

        if stream:
            print(tokenizer.decode([next_token]), end="", flush=True)

    if stream:
        print("\n")
    return tokenizer.decode(token_ids)
