"""Corpus preparation entry point: ``python scripts/prepare_data.py``.

A two-step pipeline; the order matters:

1. Train a BPE tokenizer on a corpus you provide.
2. Encode the corpora with that tokenizer into ``uint16`` binaries.

Step 2 must reuse the tokenizer produced by step 1. A mismatched pair leaves
token ids that no longer agree with the vocabulary, which raises no error and
only shows up as a training loss that never improves.

Where the inputs come from and where the outputs go is entirely up to the
caller: every path is taken from ``--flag`` or from the config file, and this
script assumes nothing about the on-disk layout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from llm_basics.tokenizer import BPETokenizer
from llm_basics.utils.config import (
    Config,
    build_cli_defaults,
    default_config_path,
    load_config,
)
from llm_basics.utils.paths import resolve_path

#: Path arguments, split by the config section that provides their defaults.
_TOKENIZER_PATH_ARGS = (
    "raw_train_path",
    "vocab_output_path",
    "merges_output_path",
)
_DATA_PATH_ARGS = (
    "train_text_path",
    "val_text_path",
    "train_bin_output_path",
    "val_bin_output_path",
)


@dataclass
class PrepareResult:
    """Summary of one preprocessing run, useful for logging or assertions."""

    vocab_size: int
    num_merges: int
    train_tokens: int
    val_tokens: int
    vocab_path: str
    merges_path: str
    train_bin_path: str
    val_bin_path: str


def prepare_dataset(config: Config) -> PrepareResult:
    """Train a tokenizer on the supplied corpus and encode it.

    Every path is taken from the caller's configuration or command line. The
    corpus location is the caller's choice; this function neither searches for
    data nor falls back to any default directory.

    Args:
        config: Merged configuration; the ``tokenizer`` and ``data`` sections
            provide the input and output paths.

    Returns:
        A :class:`PrepareResult` describing what was produced.

    Raises:
        ValueError: If a required input or output path is unset.
        FileNotFoundError: If an input file does not exist.
    """
    tok_cfg = config.tokenizer
    data_cfg = config.data

    raw_train_path = resolve_path(tok_cfg.raw_train_path)
    vocab_output_path = resolve_path(tok_cfg.vocab_output_path)
    merges_output_path = resolve_path(tok_cfg.merges_output_path)
    train_text_path = resolve_path(data_cfg.train_text_path)
    val_text_path = resolve_path(data_cfg.val_text_path)
    train_bin_path = resolve_path(data_cfg.train_bin_output_path)
    val_bin_path = resolve_path(data_cfg.val_bin_output_path)

    required = {
        "--raw_train_path (tokenizer.raw_train_path)": raw_train_path,
        "--train_text_path (data.train_text_path)": train_text_path,
        "--train_bin_output_path (data.train_bin_output_path)": train_bin_path,
        "--vocab_output_path (tokenizer.vocab_output_path)": vocab_output_path,
        "--merges_output_path (tokenizer.merges_output_path)": merges_output_path,
    }
    unset = [label for label, value in required.items() if value is None]
    if unset:
        raise ValueError(
            "these paths have no default and were not provided: "
            + ", ".join(unset)
            + ". Pass them on the command line or set them in the config file."
        )

    for label, path in (
        ("--raw_train_path", raw_train_path),
        ("--train_text_path", train_text_path),
        ("--val_text_path", val_text_path),
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{label} points at a file that does not exist: {path}")

    print(f"--> Training tokenizer on: {raw_train_path}")
    vocab, merges = BPETokenizer.train(
        input_path=str(raw_train_path),
        vocab_size=tok_cfg.vocab_size,
        special_tokens=list(tok_cfg.special_tokens),
        save_vocab_path=str(vocab_output_path),
        save_merges_path=str(merges_output_path),
    )
    print(
        f"Tokenizer saved: {vocab_output_path} / {merges_output_path} "
        f"(vocab {len(vocab)}, merges {len(merges)})"
    )

    tokenizer = BPETokenizer.from_files(
        vocab_filepath=str(vocab_output_path),
        merges_filepath=str(merges_output_path),
        special_tokens=list(tok_cfg.special_tokens),
    )

    dtype = getattr(np, data_cfg.dtype)

    print(f"--> Encoding train split to: {train_bin_path}")
    train_tokens = tokenizer.encode_to_bin(
        input_text_path=str(train_text_path),
        output_bin_path=str(train_bin_path),
        chunk_lines=data_cfg.chunk_lines,
        dtype=dtype,
    )

    if val_text_path is None or val_bin_path is None:
        print("--> No validation split configured; skipping it")
        val_tokens = 0
    else:
        print(f"--> Encoding validation split to: {val_bin_path}")
        val_tokens = tokenizer.encode_to_bin(
            input_text_path=str(val_text_path),
            output_bin_path=str(val_bin_path),
            chunk_lines=data_cfg.chunk_lines,
            dtype=dtype,
        )

    print(f"Data preparation complete. Tokens: train={train_tokens:,}, val={val_tokens:,}")

    return PrepareResult(
        vocab_size=len(vocab),
        num_merges=len(merges),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        vocab_path=str(vocab_output_path),
        merges_path=str(merges_output_path),
        train_bin_path=str(train_bin_path),
        val_bin_path=str(val_bin_path) if val_bin_path is not None else "",
    )


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse preprocessing arguments.

    The config file supplies the defaults; command-line flags override them.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A ``(args, special_tokens)`` pair, where ``special_tokens`` falls back
        to the configured list when the flag is not given.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(default_config_path()))
    pre_known, _ = pre_parser.parse_known_args(argv)

    config = load_config(pre_known.config)
    special_tokens = list(config.tokenizer.special_tokens)
    defaults = build_cli_defaults(
        config, exclude={"special_tokens", "stop_words"}, keep_vocab_size=True
    )

    parser = argparse.ArgumentParser(description="Tokenizer training and corpus preprocessing")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path()),
        help="Experiment config; specify only fields that differ from the defaults",
    )

    # Input and output paths. All of them default to None so that an
    # unspecified flag is distinguishable from a flag set on the command line.
    parser.add_argument(
        "--raw_train_path", type=str, default=None, help="Corpus to train the tokenizer on"
    )
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument(
        "--vocab_output_path",
        type=str,
        default=None,
        help="Where to write the tokenizer vocabulary",
    )
    parser.add_argument(
        "--merges_output_path",
        type=str,
        default=None,
        help="Where to write the tokenizer merge rules",
    )
    parser.add_argument("--special_tokens", type=str, nargs="*", default=None)

    parser.add_argument(
        "--train_text_path", type=str, default=None, help="Corpus to encode into the training binary"
    )
    parser.add_argument(
        "--val_text_path",
        type=str,
        default=None,
        help="Optional corpus to encode into the validation binary",
    )
    parser.add_argument(
        "--train_bin_output_path",
        type=str,
        default=None,
        help="Where to write the training binary",
    )
    parser.add_argument(
        "--val_bin_output_path",
        type=str,
        default=None,
        help="Where to write the validation binary",
    )
    parser.add_argument("--chunk_lines", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)

    parser.set_defaults(**defaults)
    parser.set_defaults(config=pre_known.config)

    args = parser.parse_args(argv)
    if args.special_tokens is None:
        args.special_tokens = special_tokens
    return args, args.special_tokens


def explicit_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Collect only the values the caller actually supplied.

    Used to re-merge the final configuration. Values filled in from the config
    defaults are deliberately excluded, otherwise a caller who points at one
    corpus would silently drag every other configured path along with it.

    Args:
        args: Parsed arguments.

    Returns:
        A mapping of override keys to values, always including ``special_tokens``.
        The ``config`` key is omitted: it selects the config file rather than
        overriding a value inside it.
    """
    return {
        key: value
        for key, value in vars(args).items()
        if value is not None and key != "config"
    }


def main(argv: list[str] | None = None) -> None:
    """Run the preprocessing pipeline.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Raises:
        SystemExit: If a required path is unset or an input file is missing.
            Both are caller-supplied-input problems, so they are reported as a
            single-line message rather than a traceback.
    """
    args, _special_tokens = parse_args(argv)

    final_config = load_config(args.config, cli_overrides=explicit_overrides(args))

    try:
        prepare_dataset(final_config)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
