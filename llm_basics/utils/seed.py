"""Global random-seed control for reproducible training and sampling.

Covers all three sources of randomness in this project: Python's ``random``
module, NumPy, and PyTorch (CPU plus every CUDA device).
``PYTHONHASHSEED`` only takes effect before the interpreter starts, so it is
set here for child processes or subsequent runs rather than the current one.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every random number generator used by the project.

    Args:
        seed: The seed value applied to Python, NumPy and PyTorch.
        deterministic: When True, also request deterministic algorithms.
            This disables cuDNN benchmarking and may raise on operators that
            have no deterministic implementation; expect slower steps.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
