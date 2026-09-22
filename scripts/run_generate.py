"""Generation entry point: ``python scripts/run_generate.py``.

Argument priority matches the training entry point: default config, then the
experiment file, then command-line flags.
"""

from __future__ import annotations

import argparse

from llm_basics.inference import generate, load_model_from_checkpoint
from llm_basics.modeling import TransformerLM
from llm_basics.tokenizer import BPETokenizer
from llm_basics.utils.config import build_cli_defaults, default_config_path, load_config
from llm_basics.utils.device import resolve_device
from llm_basics.utils.seed import set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse generation arguments.

    Loads ``--config`` first and uses the merged result as the argparse
    defaults, so explicit flags still override the config.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(default_config_path()))
    pre_known, _ = pre_parser.parse_known_args(argv)

    config = load_config(pre_known.config)
    
    defaults = build_cli_defaults(config, exclude={"special_tokens", "device"})

    parser = argparse.ArgumentParser(description="Autoregressive text generation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path()),
        help="Experiment config; specify only fields that differ from the defaults",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--context_length", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--rope_theta", type=float)

    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--vocab_path", type=str, help="Tokenizer vocab.json")
    parser.add_argument("--merges_path", type=str, help="Tokenizer merges.txt")
    parser.add_argument(
        "--device",
        type=str,
        default=config.generate.device,
        help="Device for generation; defaults to the generate section's device",
    )
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--stop_words", type=str, nargs="*")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--top_p", type=float)

    parser.set_defaults(**defaults)
    parser.set_defaults(config=pre_known.config)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load a checkpoint and generate text from a prompt.

    Both the vocabulary size and the stop-token ids are read from the tokenizer
    rather than the config, so a config that drifted from the artifacts cannot
    silently produce garbage.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    set_seed(args.seed)

    # Paths have no built-in defaults: report any that the caller did not
    # provide rather than handing None to the file APIs below.
    missing = [
        option
        for option, value in (
            ("--checkpoint_path", args.checkpoint_path),
            ("--vocab_path", args.vocab_path),
            ("--merges_path", args.merges_path),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "error: these paths have no default and were not provided: "
            + ", ".join(missing)
            + ". Pass them on the command line or set them in the config file."
        )

    device, device_str = resolve_device(args.device)
    print(f"Device: {device} ({device_str}) | seed: {args.seed}")

    print(f"Loading tokenizer: {args.vocab_path}")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=args.stop_words,
    )
    actual_vocab_size = len(tokenizer)

    eos_token_ids = set()
    for word in args.stop_words:
        encoded = tokenizer.encode(word)
        if encoded:
            eos_token_ids.add(encoded[0])
    print(f"Vocabulary size: {actual_vocab_size} | stop token ids: {eos_token_ids}")

    print("Building model...")
    model = TransformerLM(
        vocab_size=actual_vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint_path}")
    load_model_from_checkpoint(args.checkpoint_path, model, device)

    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_ids=eos_token_ids,
    )


if __name__ == "__main__":
    main()
