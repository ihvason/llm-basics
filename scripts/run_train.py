"""Training entry point: ``python scripts/run_train.py``.

Argument priority, lowest first: ``configs/default.yaml``, the experiment file
passed via ``--config``, then command-line flags. Parsing happens in two stages
because the defaults come from the config file: first isolate ``--config``,
load and merge, then feed the result to argparse as defaults. Anything passed
explicitly on the command line still wins.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from llm_basics.modeling import TransformerLM
from llm_basics.tokenizer import BPETokenizer
from llm_basics.training import (
    AdamW,
    ExperimentTracker,
    clip_gradient,
    cross_entropy,
    get_lr_cosine_schedule,
    load_checkpoint,
    save_checkpoint,
)
from llm_basics.utils.config import build_cli_defaults, default_config_path, load_config
from llm_basics.utils.dataloader import get_batch
from llm_basics.utils.device import setup_device
from llm_basics.utils.paths import resolve_path
from llm_basics.utils.seed import set_seed

#: Filename of the convenience pointer to the most recent checkpoint, written
#: inside ``checkpoint_dir``. Generation targets this when it does not want to
#: name a specific run.
LATEST_CHECKPOINT_NAME = "latest.pt"


def format_lr_tag(lr: float) -> str:
    """Format a learning rate as a compact scientific-notation tag.

    ``3.0e-4`` becomes ``3e-4`` rather than ``0.0003``: experiment names are read
    side by side, and exponent form keeps them short and aligned. Trailing
    mantissa zeros and the exponent's zero padding are both stripped, so
    ``3.0e-4`` gives ``3e-4`` while ``2.5e-4`` keeps its significant digit
    instead of collapsing to the same tag.

    Args:
        lr: Learning rate.

    Returns:
        The formatted tag, e.g. ``3e-4`` or ``2.5e-4``.
    """
    mantissa, exponent = f"{lr:.10e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".") or "0"
    sign = "-" if exponent.startswith("-") else ""
    digits = exponent.lstrip("+-").lstrip("0") or "0"
    return f"{mantissa}e{sign}{digits}"


def build_exp_name(
    num_layers: int, d_model: int, lr: float, now: time.struct_time
) -> str:
    """Build the automatic experiment name for a run.

    Shape: ``lm_L<layers>_D<d_model>_lr<lr>_<YYYYMMDDHHMM>``. The timestamp is
    minute-resolution and ordered most-significant first, which keeps the name
    sortable while separating runs started close together.

    Args:
        num_layers: Number of Transformer layers.
        d_model: Model width.
        lr: Learning rate.
        now: Timestamp to embed, as returned by :func:`time.localtime`.

    Returns:
        The experiment name, e.g. ``lm_L2_D128_lr3e-4_202509221430``.
    """
    return (
        f"lm_L{num_layers}_D{d_model}_lr{format_lr_tag(lr)}"
        f"_{time.strftime('%Y%m%d%H%M', now)}"
    )


def uniquify_exp_name(exp_name: str, roots: Iterable[Path]) -> str:
    """Return ``exp_name``, or ``exp_name-2``, ``-3``, ... if it is already taken.

    Timestamp granularity cannot guarantee uniqueness on its own: two runs of the
    same configuration started within the same minute would otherwise share a
    name, and the later run would silently append to the earlier run's metrics
    and overwrite its checkpoints. Bumping the name here makes a collision
    impossible however coarse the timestamp is.

    Applied only to automatically generated names. A name the caller supplied
    explicitly is honoured verbatim, since they may be resuming a previous run.

    Args:
        exp_name: Candidate experiment name.
        roots: Roots in which ``<root>/<exp_name>`` must not already exist.

    Returns:
        A name whose directory is free under every root.
    """
    roots = tuple(roots)

    def taken(candidate: str) -> bool:
        return any((root / candidate).exists() for root in roots)

    if not taken(exp_name):
        return exp_name
    counter = 2
    while True:
        candidate = f"{exp_name}-{counter}"
        if not taken(candidate):
            return candidate
        counter += 1


def update_latest_pointer(checkpoint_dir: Path, target: Path) -> Path:
    """Point ``<checkpoint_dir>/latest.pt`` at ``target``.

    Written as a *relative* symlink so the whole checkpoint tree stays portable:
    moving the root, or the repo, must not leave a dangling absolute link. A copy
    is used only when the filesystem refuses symlinks, since checkpoints are tens
    of megabytes and duplicating one per save would double the footprint.

    Args:
        checkpoint_dir: Directory holding the pointer file.
        target: Checkpoint the pointer should resolve to.

    Returns:
        Path to the pointer file.
    """
    link = checkpoint_dir / LATEST_CHECKPOINT_NAME
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        link.symlink_to(target.relative_to(checkpoint_dir))
    except (OSError, ValueError):
        link.write_bytes(target.read_bytes())
    return link


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, object]:
    """Parse training arguments.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        An ``(args, config)`` pair, where ``config`` is the merged config the
        defaults were derived from.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(default_config_path()))
    pre_known, _ = pre_parser.parse_known_args(argv)

    config = load_config(pre_known.config)
    # `device` exists in both the train and generate sections, so it has no
    # flat key; drop it from the shared defaults and set it per section below.
    defaults = build_cli_defaults(
        config, exclude={"special_tokens", "stop_words", "device"}
    )
    # Training reads the tokenizer artifacts, not the inference-time tokenizer
    # paths, so these defaults come from the tokenizer section.
    defaults["vocab_path"] = str(resolve_path(config.tokenizer.vocab_output_path))
    defaults["merges_path"] = str(resolve_path(config.tokenizer.merges_output_path))

    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path()),
        help="Experiment config; specify only fields that differ from the defaults",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deterministic kernels (reproducible but slower)",
    )

    parser.add_argument("--vocab_path", type=str, help="Tokenizer vocab.json")
    parser.add_argument("--merges_path", type=str, help="Tokenizer merges.txt")

    parser.add_argument("--context_length", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--rope_theta", type=float)

    parser.add_argument(
        "--train_bin_path", type=str, help="Token .bin written by prepare_data.py"
    )
    parser.add_argument(
        "--val_bin_path", type=str, help="Optional validation token .bin"
    )
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--runs_dir", type=str)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=config.train.device,
        help="Device for training; defaults to the train section's device",
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_iters", type=int)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--warmup_iters", type=int)
    parser.add_argument("--cosine_cycle_iters", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction)

    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--eval_iters", type=int)
    parser.add_argument("--save_interval", type=int)

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_project", type=str)

    parser.set_defaults(**defaults)
    # Only `--config` itself is pinned here; every other default comes from the
    # config, so it must not be defaulted again or the config values are lost.
    parser.set_defaults(config=pre_known.config)

    args = parser.parse_args(argv)
    return args, config


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
    device_str: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    """Evaluate mean cross-entropy over a few random validation batches.

    Args:
        model: Model to evaluate; its training mode is restored on return.
        data: Validation token array.
        batch_size: Number of sequences per batch.
        context_length: Sequence length.
        eval_iters: How many batches to average over.
        device: Device to run on.
        device_str: Device name, used by ``torch.autocast``.
        amp_dtype: Autocast dtype.
        use_amp: Whether autocast is enabled.

    Returns:
        Mean validation loss as a float.
    """
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, str(device))
        with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main(argv: list[str] | None = None) -> None:
    """Run the training loop.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Raises:
        FileNotFoundError: If the tokenizer artifacts, the training data or a
            requested resume checkpoint are missing.
    """
    args, _config = parse_args(argv)
    set_seed(args.seed, deterministic=args.deterministic)

    # Paths have no built-in defaults: every one of them must come from the
    # command line or the config file. Validate them up front so a missing
    # value reports which option is absent instead of failing later inside
    # os.path.exists(None) or Path(None).
    missing = [
        option
        for option, value in (
            ("--vocab_path", args.vocab_path),
            ("--merges_path", args.merges_path),
            ("--train_bin_path", args.train_bin_path),
            ("--checkpoint_dir", args.checkpoint_dir),
            ("--runs_dir", args.runs_dir),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "error: these paths have no default and were not provided: "
            + ", ".join(missing)
            + ". Pass them on the command line or set them in the config file."
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Device and numeric precision.
    device, device_str, amp_dtype, use_amp = setup_device(args.device, force_amp=args.use_amp)
    print(
        f"Device: {device} ({device_str}) | AMP: {use_amp} | dtype: {amp_dtype} | "
        f"seed: {args.seed} | deterministic: {args.deterministic}"
    )

    # The vocabulary size comes from the tokenizer itself, not from the config.
    if not os.path.exists(args.vocab_path) or not os.path.exists(args.merges_path):
        raise FileNotFoundError(
            f"Tokenizer artifacts not found: {args.vocab_path} / {args.merges_path}\n"
            "Run: python scripts/prepare_data.py"
        )
    print(f"Loading tokenizer: {args.vocab_path}")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
    )
    actual_vocab_size = len(tokenizer)
    print(f"Vocabulary size inferred from tokenizer: {actual_vocab_size}")

    # Experiment tracking: local JSONL, plus wandb when enabled. The automatic
    # name carries architecture, learning rate and a minute-resolution timestamp,
    # and is bumped with a numeric suffix if that name is already taken, so two
    # runs of the same configuration can never share a directory. An explicit
    # name is honoured verbatim, since the caller may be resuming a previous run.
    runs_root_dir = resolve_path(args.runs_dir)
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = uniquify_exp_name(
            build_exp_name(args.num_layers, args.d_model, args.lr, time.localtime()),
            (runs_root_dir, checkpoint_dir),
        )
    runs_dir = runs_root_dir / exp_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints are namespaced by experiment. Without this, two runs sharing a
    # save schedule would both write `ckpt_iter_50.pt` into the same directory
    # and the later run would silently destroy the earlier run's weights.
    run_checkpoint_dir = checkpoint_dir / exp_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_dict = vars(args).copy()
    config_dict["vocab_size"] = actual_vocab_size
    config_dict["seed"] = args.seed

    tracker = ExperimentTracker(
        exp_name=exp_name,
        config=config_dict,
        log_dir=str(runs_dir),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    print(f"Run directory: {runs_dir}")

    # Training data, memory-mapped to avoid copying it into RAM.
    if not os.path.exists(args.train_bin_path):
        raise FileNotFoundError(
            f"Training data not found: {args.train_bin_path}\n"
            "Run: python scripts/prepare_data.py"
        )
    train_data = np.memmap(args.train_bin_path, dtype=np.uint16, mode="r")
    val_data = (
        np.memmap(args.val_bin_path, dtype=np.uint16, mode="r")
        if (args.val_bin_path and os.path.exists(args.val_bin_path))
        else None
    )
    print(f"Training tokens: {len(train_data):,}")

    # Model and optimizer.
    model = TransformerLM(
        vocab_size=actual_vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=str(device),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    tracker.log({"train/num_params": num_params}, step=0)

    # Optional resume from a checkpoint.
    start_iter = 0
    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        start_iter = load_checkpoint(args.resume_checkpoint, model, optimizer)
        print(f"Resumed at iteration {start_iter}")

    # Training loop.
    model.train()
    t0 = time.time()
    cosine_cycle_iters = args.cosine_cycle_iters or args.max_iters

    for it in range(start_iter, args.max_iters):
        current_lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, str(device))

        with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0.0:
            clip_gradient(model.parameters(), max_norm=args.grad_clip)

        optimizer.step()

        if (it + 1) % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = (
                args.batch_size * args.context_length * args.log_interval
            ) / max(dt, 1e-6)

            tracker.log(
                {
                    "train/loss": float(loss.item()),
                    "train/lr": current_lr,
                    "train/tokens_per_sec": tokens_per_sec,
                },
                step=it + 1,
            )
            print(
                f"Iter {it + 1:6d}/{args.max_iters:6d} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {tokens_per_sec:,.0f} tokens/s"
            )

        if val_data is not None and (it + 1) % args.eval_interval == 0:
            val_loss = evaluate_loss(
                model=model,
                data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                eval_iters=args.eval_iters,
                device=device,
                device_str=device_str,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )
            val_ppl = math.exp(min(val_loss, 20.0))
            tracker.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=it + 1)
            print(
                f"--> [Validation] Iter {it + 1:6d} | "
                f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
            )

        if (it + 1) % args.save_interval == 0 or (it + 1) == args.max_iters:
            ckpt_path = run_checkpoint_dir / f"ckpt_iter_{it + 1}.pt"
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            # Convenience pointer so generation can target "the newest run"
            # without naming an experiment.
            pointer = update_latest_pointer(checkpoint_dir, ckpt_path)
            print(f"==> Checkpoint saved: {ckpt_path}")
            print(f"    Latest pointer:  {pointer}")

    tracker.close()


if __name__ == "__main__":
    main()
