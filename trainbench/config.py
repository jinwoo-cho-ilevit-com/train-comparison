"""Config loading that works everywhere, including inside framework images.

Deliberately free of Hydra and OmegaConf. Hydra pins `antlr4==4.9.*` and axolotl
pins `antlr4==4.13.2`, so a Hydra dependency here would make that image
unbuildable. Composition happens where experiments are defined; a pod receives an
already-resolved config and only validates it. See trainbench/compose.py.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trainbench.config_schema import BenchConfig


def to_bench_config(mapping: Mapping[str, Any]) -> BenchConfig:
    """Validate a plain mapping. Raises before any work starts."""
    return BenchConfig.model_validate(dict(mapping))


def load_bench_config(path: str | Path) -> BenchConfig:
    """Load a resolved config JSON, as written by scripts/compose_config.py."""
    return to_bench_config(json.loads(Path(path).read_text()))


def git_state() -> dict[str, Any]:
    """Commit recorded with every run, and whether its tree was clean (convention 07).

    `dirty` matters as much as the hash: a run started from a modified working
    tree records a commit that does not contain the code that produced the number.
    It is None when unknowable — inside an image there is no .git, so the
    orchestrator passes the commit in and the image digest carries the real
    identity of the code.
    """
    from_env = os.environ.get("TRAINBENCH_GIT_COMMIT")
    if from_env:
        return {"commit": from_env, "dirty": None, "source": "env"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None, "source": "unavailable"}
    return {"commit": commit, "dirty": bool(status), "source": "git"}
