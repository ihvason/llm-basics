"""Device selection and numerical-precision policy, shared by training and inference.

Keeping both the ``auto`` probing order and the mixed-precision decision in one
place stops training and inference from drifting apart on hardware handling.
"""

from __future__ import annotations

import torch


def resolve_device(device_arg: str = "auto") -> tuple[torch.device, str]:
    """Resolve the target device, probing CUDA then MPS then CPU for ``"auto"``.

    Args:
        device_arg: ``"auto"`` or an explicit device string such as ``"cpu"``.

    Returns:
        A ``(torch.device, device_str)`` pair.

    Raises:
        RuntimeError: If CUDA is requested explicitly but unavailable.
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        device_str = device_arg

    device = torch.device(device_str)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device_str!r} was requested but torch.cuda.is_available() is False. "
            "Use --device cpu/mps or set device to auto."
        )
    return device, device_str


def resolve_amp(device_str: str, want_amp: bool) -> tuple[torch.dtype, bool]:
    """Decide the autocast dtype and whether mixed precision should be enabled.

    Mixed precision is enabled only on CUDA and only when explicitly requested:
    the payoff on CPU/MPS is unreliable, and some MPS operators fail outright
    with bfloat16.

    Args:
        device_str: Resolved device string, e.g. ``"cuda"``.
        want_amp: Whether the caller asked for mixed precision.

    Returns:
        An ``(amp_dtype, use_amp)`` pair.
    """
    if device_str.startswith("cuda") and want_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return amp_dtype, True
    return torch.float32, False


def setup_device(
    device_arg: str = "auto", force_amp: bool = False
) -> tuple[torch.device, str, torch.dtype, bool]:
    """Resolve the device and the precision policy in a single call.

    Args:
        device_arg: ``"auto"`` or an explicit device string.
        force_amp: Whether mixed precision was requested.

    Returns:
        ``(device, device_str, amp_dtype, use_amp)``.
    """
    device, device_str = resolve_device(device_arg)
    amp_dtype, use_amp = resolve_amp(device_str, force_amp)
    return device, device_str, amp_dtype, use_amp
