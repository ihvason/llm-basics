"""Command-line scripts: argument parsing, config merging and entry forwarding.

This layer only parses arguments and calls into ``llm_basics``; all algorithm
and business logic lives in the package. Run the scripts directly:

    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/run_train.py    --config configs/default.yaml
    python scripts/run_generate.py --config configs/default.yaml

``sys.path`` bootstrap
----------------------
Several modules in this package are invoked as entry points by name, which
means ``sys.path[0]`` is the interpreter's binary directory rather than the
repository root. This package therefore inserts the repository root into
``sys.path`` on import, so the sibling ``llm_basics`` package resolves
regardless of the working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
