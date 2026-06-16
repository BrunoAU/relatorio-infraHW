from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_deps():
    root = Path(__file__).resolve().parents[1]
    deps = root / ".deps"
    deps_str = str(deps)
    in_local_venv = str(Path(sys.prefix).resolve()).startswith(str((root / ".venv").resolve()))
    if deps.exists() and deps_str not in sys.path and not in_local_venv:
        sys.path.insert(0, deps_str)
