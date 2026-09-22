"""Configuration system: YAML loading, deep merge and typed access.

Three sources, lowest priority first:

1. ``configs/default.yaml`` -- the project defaults, which are the single
   source of truth for default values.
2. The experiment file passed via ``--config`` -- it only spells out the fields
   that differ from the defaults.
3. Command-line flags -- the highest-priority, single-field overrides.

Design notes
------------
- **Deep merge instead of whole-section replacement.** An experiment file can
  therefore be as short as the diff it represents.
- **Templated paths.** Write ``data/processed/train.bin`` in YAML, or
  ``${DATA_ROOT}/raw/...`` when the corpus lives on another disk;
  ``llm_basics.utils.paths`` expands both.
- **Flat keys remain supported.** ``cfg["context_length"]`` is mapped onto the
  corresponding section field, so older call sites keep working. A field name
  shared by several sections gets no flat key, since flattening it would
  silently drop all but one section; see :data:`_FLAT_KEYS`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .paths import repo_root, resolve_path

#: Default config filename, relative to the repository root.
DEFAULT_CONFIG_NAME = "configs/default.yaml"


# ==============================================================================
# YAML IO and deep merge
# ==============================================================================
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``, recursing into nested mappings.

    Args:
        base: Lower-priority mapping, not modified.
        override: Higher-priority mapping.

    Returns:
        A new mapping in which nested keys are merged rather than replaced.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping from disk.

    Args:
        path: File to read.

    Returns:
        The parsed mapping; an empty file yields an empty mapping.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document's top level is not a mapping.
    """
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with open(resolved, encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {resolved}")
    return content


def dump_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a mapping to a YAML file, creating parent directories.

    Args:
        data: Mapping to serialize.
        path: Destination file.

    Returns:
        The resolved destination path.
    """
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return resolved


# ==============================================================================
# Typed config sections
# ==============================================================================
@dataclass
class TokenizerConfig:
    """Tokenizer settings: raw corpus in, BPE vocab and merges out.

    Every path is ``None`` unless the caller supplies it, either on the command
    line or in a config file. Nothing here presumes a data directory layout, so
    a run cannot silently read or write a location the caller never named.
    """

    raw_train_path: str | None = None
    vocab_size: int = 10000
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    vocab_output_path: str | None = None
    merges_output_path: str | None = None


@dataclass
class DataConfig:
    """Corpus preprocessing settings: text in, token binaries out.

    Paths default to ``None`` for the same reason as
    :class:`TokenizerConfig`: the caller decides where the corpus lives and
    where the binaries go.
    """

    train_text_path: str | None = None
    val_text_path: str | None = None
    # Names carry the "output" marker so they cannot be confused with
    # `train.train_bin_path` / `train.val_bin_path`, which name the *same* files
    # on the reading side. The flat-key table requires those names to be unique.
    train_bin_output_path: str | None = None
    val_bin_output_path: str | None = None
    chunk_lines: int = 200
    dtype: str = "uint16"


@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters."""

    context_length: int = 64
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 2
    d_ff: int = 512
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    """Optimization and training-loop hyperparameters.

    Paths default to ``None``: the caller names the training data, the
    checkpoint directory and any resume checkpoint. Hyperparameters keep their
    defaults because they describe how to optimize, not where data lives.
    """

    train_bin_path: str | None = None
    val_bin_path: str | None = None
    checkpoint_dir: str | None = None
    runs_dir: str | None = None
    resume_checkpoint: str | None = None
    device: str = "auto"

    batch_size: int = 4
    max_iters: int = 50

    lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    warmup_iters: int = 10
    cosine_cycle_iters: int = 50

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    use_amp: bool = False

    log_interval: int = 10
    eval_interval: int = 25
    eval_iters: int = 10
    save_interval: int = 50

    exp_name: str | None = None
    use_wandb: bool = False
    wandb_project: str = "llm_basics"


@dataclass
class GenerateConfig:
    """Autoregressive sampling settings.

    Paths default to ``None``; the caller supplies the checkpoint and the
    tokenizer artifacts it wants to decode with.
    """

    checkpoint_path: str | None = None
    vocab_path: str | None = None
    merges_path: str | None = None
    device: str = "auto"
    prompt: str = "Once upon a time, there was a little girl named Lily"
    max_new_tokens: int = 64
    stop_words: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


_SECTION_TYPES: dict[str, type] = {
    "tokenizer": TokenizerConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "generate": GenerateConfig,
}


def _build_section(section_type: type, payload: dict[str, Any] | None) -> Any:
    """Instantiate one section, rejecting unknown field names.

    Args:
        section_type: The dataclass describing the section.
        payload: Raw mapping read from YAML.

    Returns:
        An instance of ``section_type``.

    Raises:
        ValueError: If ``payload`` contains a field the dataclass lacks, which
            almost always means a typo in the config file.
    """
    payload = payload or {}
    known = {f.name for f in fields(section_type)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"{section_type.__name__} received unknown fields {sorted(unknown)}; "
            f"available fields: {sorted(known)}"
        )
    return section_type(**payload)


# ==============================================================================
# Flat key mapping
# ==============================================================================
#: Flat key -> (section name, field name, expected type).
#:
#: A flat key exists only when it names exactly one field in exactly one
#: section. If the same field name appears in several sections, it gets no flat
#: key and must be read through its section, because a single flattened key
#: would silently select one section and discard the others. ``device`` is the
#: current example: read ``config.train.device`` or ``config.generate.device``.
_FLAT_KEYS: dict[str, tuple[str, str, type]] = {
    # tokenizer
    "raw_train_path": ("tokenizer", "raw_train_path", str),
    "vocab_size": ("tokenizer", "vocab_size", int),
    "special_tokens": ("tokenizer", "special_tokens", list),
    "vocab_output_path": ("tokenizer", "vocab_output_path", str),
    "merges_output_path": ("tokenizer", "merges_output_path", str),
    # Tokenizer paths used at inference time, which may differ from training artifacts.
    "vocab_path": ("generate", "vocab_path", str),
    "merges_path": ("generate", "merges_path", str),
    # data
    "train_text_path": ("data", "train_text_path", str),
    "val_text_path": ("data", "val_text_path", str),
    "train_bin_output_path": ("data", "train_bin_output_path", str),
    "val_bin_output_path": ("data", "val_bin_output_path", str),
    "chunk_lines": ("data", "chunk_lines", int),
    "dtype": ("data", "dtype", str),
    # model (no vocab_size: it is inferred from the tokenizer at runtime)
    "context_length": ("model", "context_length", int),
    "d_model": ("model", "d_model", int),
    "num_layers": ("model", "num_layers", int),
    "num_heads": ("model", "num_heads", int),
    "d_ff": ("model", "d_ff", int),
    "rope_theta": ("model", "rope_theta", float),
    # train
    "train_bin_path": ("train", "train_bin_path", str),
    "val_bin_path": ("train", "val_bin_path", str),
    "checkpoint_dir": ("train", "checkpoint_dir", str),
    "runs_dir": ("train", "runs_dir", str),
    "resume_checkpoint": ("train", "resume_checkpoint", str),
    # NOTE: no flat key for `device`. Both `train` and `generate` define one,
    # so a single flat key would silently pick one section and discard the
    # other. Read `config.train.device` or `config.generate.device` explicitly.
    "batch_size": ("train", "batch_size", int),
    "max_iters": ("train", "max_iters", int),
    "lr": ("train", "lr", float),
    "min_lr": ("train", "min_lr", float),
    "warmup_iters": ("train", "warmup_iters", int),
    "cosine_cycle_iters": ("train", "cosine_cycle_iters", int),
    "weight_decay": ("train", "weight_decay", float),
    "beta1": ("train", "beta1", float),
    "beta2": ("train", "beta2", float),
    "grad_clip": ("train", "grad_clip", float),
    "use_amp": ("train", "use_amp", bool),
    "log_interval": ("train", "log_interval", int),
    "eval_interval": ("train", "eval_interval", int),
    "eval_iters": ("train", "eval_iters", int),
    "save_interval": ("train", "save_interval", int),
    "exp_name": ("train", "exp_name", str),
    "use_wandb": ("train", "use_wandb", bool),
    "wandb_project": ("train", "wandb_project", str),
    # generate
    "checkpoint_path": ("generate", "checkpoint_path", str),
    "prompt": ("generate", "prompt", str),
    "max_new_tokens": ("generate", "max_new_tokens", int),
    "stop_words": ("generate", "stop_words", list),
    "temperature": ("generate", "temperature", float),
    "top_k": ("generate", "top_k", int),
    "top_p": ("generate", "top_p", float),
}

#: Flat keys holding paths, which are resolved to absolute paths. A ``None``
#: value stays ``None``.
_PATH_KEYS = frozenset(
    {
        "raw_train_path",
        "vocab_output_path",
        "merges_output_path",
        "vocab_path",
        "merges_path",
        "train_text_path",
        "val_text_path",
        "train_bin_output_path",
        "val_bin_output_path",
        "train_bin_path",
        "val_bin_path",
        "checkpoint_dir",
        "runs_dir",
        "resume_checkpoint",
        "checkpoint_path",
    }
)


class Config:
    """A merged configuration: typed sections plus a flat-key view."""

    def __init__(
        self,
        raw: dict[str, Any],
        source: Path | None = None,
        default_source: Path | None = None,
    ):
        """Build the typed sections from a merged mapping.

        Args:
            raw: Fully merged configuration mapping.
            source: Path of the experiment config, if one was loaded.
            default_source: Path of the default config, if one was loaded.

        Raises:
            ValueError: If the mapping contains an unknown section.
        """
        self.raw = raw
        self.source = source
        self.default_source = default_source
        self.sections: dict[str, Any] = {
            name: _build_section(section_type, raw.get(name))
            for name, section_type in _SECTION_TYPES.items()
        }
        unknown_sections = set(raw) - set(_SECTION_TYPES)
        if unknown_sections:
            raise ValueError(
                f"Config contains unknown sections {sorted(unknown_sections)}; "
                f"available sections: {sorted(_SECTION_TYPES)}"
            )

    @property
    def tokenizer(self) -> TokenizerConfig:
        """Tokenizer section."""
        return self.sections["tokenizer"]

    @property
    def data(self) -> DataConfig:
        """Corpus preprocessing section."""
        return self.sections["data"]

    @property
    def model(self) -> ModelConfig:
        """Model architecture section."""
        return self.sections["model"]

    @property
    def train(self) -> TrainConfig:
        """Training section."""
        return self.sections["train"]

    @property
    def generate(self) -> GenerateConfig:
        """Generation section."""
        return self.sections["generate"]

    def __getitem__(self, key: str) -> Any:
        """Look up a value by flat key or section name.

        Args:
            key: A flat key such as ``"d_model"`` or a section name.

        Returns:
            The corresponding value.

        Raises:
            KeyError: If the key is neither a known flat key nor a section.
        """
        if key in _FLAT_KEYS:
            section, attribute, _ = _FLAT_KEYS[key]
            return getattr(self.sections[section], attribute)
        if key in self.sections:
            return self.sections[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Like :meth:`__getitem__`, but returns ``default`` for unknown keys."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        """Return whether ``key`` is a known flat key or section name."""
        return key in _FLAT_KEYS or key in self.sections

    def to_dict(
        self,
        resolve: bool = False,
        for_cli: bool = False,
        keep_vocab_size: bool = False,
    ) -> dict[str, Any]:
        """Export the configuration as a plain dictionary.

        Args:
            resolve: Resolve path fields to absolute paths.
            for_cli: Emit flat keys instead of nested sections.
            keep_vocab_size: Keep ``vocab_size`` in a flat export. Training and
                inference infer the vocabulary size from the tokenizer and do
                not want it as a CLI default, whereas ``prepare_data`` does.

        Returns:
            A JSON-serializable mapping.
        """
        if for_cli:
            payload: dict[str, Any] = {}
            for flat_key, (section, attribute, _cast) in _FLAT_KEYS.items():
                value = getattr(self.sections[section], attribute)
                if resolve and flat_key in _PATH_KEYS and value is not None:
                    value = str(resolve_path(value))
                payload[flat_key] = value
            if not keep_vocab_size:
                payload.pop("vocab_size", None)
            return payload

        payload = {name: dataclasses.asdict(section) for name, section in self.sections.items()}
        if resolve:
            for section_name in ("tokenizer", "data", "train", "generate"):
                section = payload[section_name]
                for key in list(section):
                    if key in _PATH_KEYS and section[key] is not None:
                        section[key] = str(resolve_path(section[key]))
        return payload

    def to_json(self, resolve: bool = False) -> str:
        """Serialize the configuration to a JSON string."""
        return json.dumps(self.to_dict(resolve=resolve), ensure_ascii=False, indent=2)

    def __repr__(self) -> str:
        """Return a short description of the config and its sources."""
        return f"Config(source={self.source}, sections={sorted(self.sections)})"


# ==============================================================================
# Loading entry points
# ==============================================================================
def default_config_path() -> Path:
    """Return the default config path, ``<repo>/configs/default.yaml``."""
    return repo_root() / DEFAULT_CONFIG_NAME


def load_config(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    use_default: bool = True,
) -> Config:
    """Load and merge a configuration.

    Args:
        config_path: Experiment config to layer on top of the defaults. ``None``
            uses defaults only. The strings ``"none"``, ``"null"`` and ``""``
            explicitly disable the default file.
        cli_overrides: Highest-priority overrides, given either as flat keys
            (``{"lr": 1e-3}``) or as whole sections (``{"model": {...}}``).
        use_default: Whether to layer ``configs/default.yaml`` underneath.

    Returns:
        The merged :class:`Config`.

    Raises:
        FileNotFoundError: If an explicitly given ``config_path`` is missing.
        KeyError: If a flat override key is not recognized.
    """
    raw: dict[str, Any] = {}
    default_source: Path | None = None

    if use_default:
        default_source = default_config_path()
        if default_source.exists():
            raw = load_yaml(default_source)

    source: Path | None = None
    if config_path is not None:
        text = os.fspath(config_path).strip()
        if text and text.lower() not in {"none", "null"}:
            source = Path(text).expanduser()
            raw = deep_merge(raw, load_yaml(source))

    overrides = dict(cli_overrides or {})
    section_overrides: dict[str, Any] = {}
    flat_overrides: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _SECTION_TYPES and isinstance(value, dict):
            section_overrides[key] = value
        else:
            flat_overrides[key] = value

    raw = deep_merge(raw, section_overrides)

    if flat_overrides:
        sectioned: dict[str, Any] = {name: {} for name in _SECTION_TYPES}
        for flat_key, value in flat_overrides.items():
            if flat_key not in _FLAT_KEYS:
                raise KeyError(
                    f"Unknown config key {flat_key!r}; "
                    f"see llm_basics.utils.config._FLAT_KEYS for the valid keys."
                )
            section, attribute, cast = _FLAT_KEYS[flat_key]
            sectioned[section][attribute] = value if value is None else cast(value)
        sectioned = {name: payload for name, payload in sectioned.items() if payload}
        raw = deep_merge(raw, sectioned)

    # Materialize every section so dataclass defaults apply uniformly.
    for name in _SECTION_TYPES:
        raw.setdefault(name, {})

    return Config(raw, source=source, default_source=default_source)


def build_cli_defaults(
    config: Config,
    exclude: set[str] | None = None,
    keep_vocab_size: bool = False,
) -> dict[str, Any]:
    """Convert a config into flat defaults for ``parser.set_defaults(**...)``.

    Because ``argparse`` needs its defaults before ``parse_args`` runs, while
    the defaults depend on ``--config``, each CLI performs a two-stage parse:
    pre-parse ``--config``, load and merge, then call this function to obtain
    the defaults for the real parser.

    Args:
        config: Merged configuration.
        exclude: Flat keys to drop, e.g. ``stop_words`` when the caller encodes
            them separately.
        keep_vocab_size: Whether to keep ``vocab_size``. Training and inference
            infer it from the tokenizer; ``prepare_data`` needs it.

    Returns:
        Flat key to value mapping with path fields resolved to absolute paths.
        Entries whose value is ``None`` are omitted, since argparse cannot
        express a missing option as ``None``.
    """
    payload = config.to_dict(resolve=True, for_cli=True, keep_vocab_size=keep_vocab_size)
    for key in exclude or ():
        payload.pop(key, None)
    return {key: value for key, value in payload.items() if value is not None}
