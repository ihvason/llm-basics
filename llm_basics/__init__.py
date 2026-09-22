"""CS336 Assignment 1: a Transformer language model built from scratch.

Layer conventions
-----------------

- **Domain layers** (``modeling`` / ``tokenizer`` / ``training`` /
  ``inference``): one subpackage each. Implementations inside a layer stay
  independent; cross-layer access goes through public imports from
  ``llm_basics``.
- **Shared utilities** (``utils``): path resolution, seeding, device selection,
  batch sampling and config loading. See ``llm_basics.utils`` for the criteria.
- **Entry points** (``scripts/``, outside the package): argument parsing and
  config merging only, no algorithm logic.

Subpackages
-----------

- ``llm_basics.modeling``: model operators and ``TransformerLM``.
- ``llm_basics.tokenizer``: the BPE tokenizer and pre-tokenization.
- ``llm_basics.training``: loss, AdamW, schedule, checkpoints, experiment logging.
- ``llm_basics.inference``: autoregressive sampling and generation.
- ``llm_basics.utils``: paths, seed, device, dataloader, config.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "__version__",
    "load_config",
    "repo_root",
    "data_root",
    "checkpoint_root",
    "runs_root",
]


def _resolve_version() -> str:
    """Return the installed package version, or a placeholder when not installed."""
    try:
        return version("llm_basics")
    except PackageNotFoundError:  # Running from a source checkout without installation.
        return "0.0.0+unknown"


__version__ = _resolve_version()


def load_config(*args, **kwargs):
    """Forward to :func:`llm_basics.utils.config.load_config`.

    The import is deferred so that callers who only need path handling do not
    pull in PyYAML, which ``config`` depends on but ``utils.paths`` does not.

    Returns:
        The merged ``Config`` produced by the underlying loader.
    """
    from .utils.config import load_config as _load_config

    return _load_config(*args, **kwargs)


def repo_root(*args, **kwargs):
    """Forward to :func:`llm_basics.utils.paths.repo_root`."""
    from .utils.paths import repo_root as _repo_root

    return _repo_root(*args, **kwargs)


def data_root(*args, **kwargs):
    """Forward to :func:`llm_basics.utils.paths.data_root`."""
    from .utils.paths import data_root as _data_root

    return _data_root(*args, **kwargs)


def checkpoint_root(*args, **kwargs):
    """Forward to :func:`llm_basics.utils.paths.checkpoint_root`."""
    from .utils.paths import checkpoint_root as _checkpoint_root

    return _checkpoint_root(*args, **kwargs)


def runs_root(*args, **kwargs):
    """Forward to :func:`llm_basics.utils.paths.runs_root`."""
    from .utils.paths import runs_root as _runs_root

    return _runs_root(*args, **kwargs)
