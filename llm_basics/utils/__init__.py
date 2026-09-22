"""Shared utilities: small, reusable capabilities used across the package.

Inclusion criteria
------------------
1. **Used in more than one place**: at least two subpackages or the script
   layer need it. A helper with a single caller belongs next to that caller.
2. **Independent of modelling logic**: it does not encode LLM semantics.

Note that ``config`` is project-specific rather than generic. It lives here so
that path handling and configuration sit in one layer; moving it back to the
package top level only requires updating the lazy forwarders in
``llm_basics/__init__.py``.

Contents
--------
- ``paths``: repository, data root and checkpoint root resolution, ``${VAR}``
  expansion, relative-to-absolute conversion.
- ``seed``: seeding Python, NumPy and PyTorch.
- ``device``: device probing and the autocast precision policy.
- ``dataloader``: sampling ``(inputs, targets)`` batches from a token array.
- ``config``: YAML loading, deep merge, typed sections and CLI defaults.

Import note
-----------
``config`` depends on PyYAML. To keep that dependency out of the way of
callers that only need path handling, this module does **not** re-export the
config symbols; import them explicitly::

    from llm_basics.utils.config import load_config
"""

from .dataloader import get_batch
from .device import resolve_amp, resolve_device, setup_device
from .paths import (
    checkpoint_root,
    data_root,
    resolve_path,
    resolve_paths,
    repo_root,
    runs_root,
)
from .seed import set_seed

__all__ = [
    "repo_root",
    "data_root",
    "checkpoint_root",
    "runs_root",
    "resolve_path",
    "resolve_paths",
    "set_seed",
    "resolve_device",
    "resolve_amp",
    "setup_device",
    "get_batch",
]
