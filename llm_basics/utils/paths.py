"""Project path resolution: the single entry point for relative-to-absolute paths.

Design notes
------------
1. **One anchor for the repository root.** The root is found by walking upward
   for ``pyproject.toml``. Module files live one level below it, in
   ``llm_basics/``, so ``__file__.parent`` is not the repository root.
2. **The data root can live elsewhere.** The ``LLM_BASICS_DATA_ROOT``
   environment variable redirects ``data/`` to another disk, which matters
   because the raw corpora run to tens of gigabytes. Defaults to ``<repo>/data``.
3. **Each artefact tree has its own root, and they are siblings.**
   ``<repo>/data`` holds corpora and token binaries, ``<repo>/checkpoints`` holds
   model checkpoints and ``<repo>/runs`` holds per-run logs. Each is redirected
   by its own environment variable -- ``LLM_BASICS_DATA_ROOT``,
   ``LLM_BASICS_CHECKPOINT_ROOT``, ``LLM_BASICS_RUNS_ROOT`` -- so no root
   implies another and any one of them can move to a different disk.
4. **Configs use relative paths.** Paths in YAML are written relative to the
   repository root, e.g. ``data/processed/train.bin``; only the built-in
   ``${DATA_ROOT}`` and ``${CHECKPOINT_ROOT}`` references resolve against the
   external roots.
5. **Absolute paths pass through untouched**, so command-line overrides can
   point anywhere.

Conventions
-----------
- Resolution is read-only and never creates directories; callers create them.
- A missing repository root raises ``RuntimeError`` rather than falling back to
  the working directory, which would silently write data in the wrong place.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Markers identifying the repository root, in priority order.
ROOT_MARKERS = ("pyproject.toml", ".git")

#: Environment variable pointing at the data root.
DATA_ROOT_ENV = "LLM_BASICS_DATA_ROOT"

#: Environment variable pointing at the checkpoint root.
CHECKPOINT_ROOT_ENV = "LLM_BASICS_CHECKPOINT_ROOT"

#: Environment variable pointing at the runs (experiment log) root.
RUNS_ROOT_ENV = "LLM_BASICS_RUNS_ROOT"

#: Matches ``${VAR}``, ``$VAR`` and ``${VAR:default}`` in path strings.
_VAR_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)


def repo_root() -> Path:
    """Return the repository root.

    Walks up from this file until a directory containing one of
    :data:`ROOT_MARKERS` is found. Recomputed on every call rather than cached:
    the result is cheap (a few filesystem stats) and callers are one-shot, so
    caching would only risk returning a stale root after the checkout or mount
    point changes within a process.

    Returns:
        Absolute path to the repository root.

    Raises:
        RuntimeError: If no marker is found in any parent directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if any((parent / marker).exists() for marker in ROOT_MARKERS):
            return parent
    raise RuntimeError(
        f"Could not locate the repository root above {current}: "
        f"none of {ROOT_MARKERS} was found."
    )


def project_root() -> Path:
    """Alias of :func:`repo_root`, kept for existing call sites."""
    return repo_root()


def data_root() -> Path:
    """Return the data root: ``$LLM_BASICS_DATA_ROOT`` if set, else ``<repo>/data``.

    Recomputed on every call, so changing ``LLM_BASICS_DATA_ROOT`` mid-process
    takes effect immediately instead of returning a cached earlier answer.

    Returns:
        Absolute path to the directory holding raw corpora and derived data.
    """
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "data"


def checkpoint_root() -> Path:
    """Return the checkpoint root: ``$LLM_BASICS_CHECKPOINT_ROOT`` if set, else
    ``<repo>/checkpoints``.

    Deliberately a sibling of :func:`data_root` rather than a directory inside
    it: corpora and token binaries are inputs that can be shared read-only,
    while checkpoints are per-run outputs that are written continuously. Giving
    each its own root lets them live on different disks and be redirected
    independently.

    Recomputed on every call, for the same reason as :func:`data_root`.

    Returns:
        Absolute path to the directory holding checkpoints.
    """
    override = os.environ.get(CHECKPOINT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "checkpoints"


def runs_root() -> Path:
    """Return the runs root: ``$LLM_BASICS_RUNS_ROOT`` if set, else ``<repo>/runs``.

    The third sibling root, holding per-experiment logs rather than the
    checkpoints themselves. Kept separate so a cheap, text-only log tree can
    live on a different disk from the multi-gigabyte weight files.

    Recomputed on every call, for the same reason as :func:`data_root`.

    Returns:
        Absolute path to the directory holding per-run experiment logs.
    """
    override = os.environ.get(RUNS_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "runs"


def expandvars(text: str) -> str:
    """Expand ``${VAR}`` and ``$VAR``, with an optional ``${VAR:default}`` form.

    Unlike :func:`os.path.expandvars`, an undefined variable raises instead of
    expanding to nothing, so a typo in a config cannot silently resolve to the
    wrong path. ``${VAR:default}`` provides an explicit fallback.

    Args:
        text: String that may contain variable references.

    Returns:
        The string with every reference replaced.

    Raises:
        KeyError: If a referenced variable is undefined and has no default.
    """

    def _replace(match: re.Match[str]) -> str:
        """Resolve one variable reference to its value.

        Args:
            match: A match of :data:`_VAR_PATTERN`.

        Returns:
            The environment value, the inline default, or the data root.

        Raises:
            KeyError: If the variable is undefined and has no inline default.
        """
        name = match.group(1) or match.group(3)
        default = match.group(2)
        value = os.environ.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        # DATA_ROOT, CHECKPOINT_ROOT and RUNS_ROOT are built in: they resolve
        # even without an environment variable.
        if name == "DATA_ROOT":
            return str(data_root())
        if name == "CHECKPOINT_ROOT":
            return str(checkpoint_root())
        if name == "RUNS_ROOT":
            return str(runs_root())
        raise KeyError(
            f"Path variable ${{{name}}} is undefined; set the environment variable "
            f"or write ${{{name}:default}}."
        )

    return _VAR_PATTERN.sub(_replace, text)


def resolve_path(path: str | os.PathLike[str] | None, base: Path | None = None) -> Path | None:
    """Resolve one configured path to an absolute path.

    ``~`` is expanded, ``${VAR}`` references are substituted first, and a
    relative result is anchored to ``base`` (the repository root by default).
    Symlinks are deliberately not resolved so the configured location is kept
    as written.

    Args:
        path: Path to resolve, or ``None`` to pass through unchanged.
        base: Directory used to anchor relative paths.

    Returns:
        The absolute path, or ``None`` if ``path`` was ``None`` or empty.
    """
    if path is None:
        return None
    expanded = expandvars(os.fspath(path)).strip()
    if not expanded:
        return None
    candidate = Path(expanded).expanduser()
    if candidate.is_absolute():
        return candidate
    anchor = base if base is not None else repo_root()
    return anchor / candidate


def resolve_paths(
    paths: dict[str, str | None], base: Path | None = None
) -> dict[str, Path | None]:
    """Resolve a mapping of paths, preserving its keys.

    Args:
        paths: Mapping from key to path string.
        base: Directory used to anchor relative paths.

    Returns:
        A new mapping with the same keys and resolved values.
    """
    return {key: resolve_path(value, base=base) for key, value in paths.items()}
