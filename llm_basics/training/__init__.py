"""Training layer: loss, optimizer, learning-rate schedule, checkpoints and logging.

- ``loss.cross_entropy``: numerically stable cross-entropy.
- ``optimizer``: hand-written ``AdamW``, cosine schedule and gradient clipping.
- ``checkpoint``: saving and loading model/optimizer state.
- ``logger.ExperimentTracker``: local JSONL plus optional wandb reporting.
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .logger import ExperimentTracker
from .loss import cross_entropy
from .optimizer import AdamW, clip_gradient, get_lr_cosine_schedule

# Re-exported so callers can write `from llm_basics.training import get_batch`.
# The implementation lives in `llm_basics.utils.dataloader`.
from llm_basics.utils.dataloader import get_batch

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "cross_entropy",
    "AdamW",
    "get_lr_cosine_schedule",
    "clip_gradient",
    "ExperimentTracker",
    "get_batch",
]
