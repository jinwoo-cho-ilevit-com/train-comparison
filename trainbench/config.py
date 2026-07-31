"""Config loading that works everywhere, including inside framework images.

Deliberately free of Hydra and OmegaConf. Hydra pins `antlr4==4.9.*` and axolotl
pins `antlr4==4.13.2`, so a Hydra dependency here would make that image
unbuildable. Composition happens where experiments are defined; a pod receives an
already-resolved config and only validates it. See trainbench/compose.py.
"""

from __future__ import annotations

import json
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


def git_commit() -> str:
    """Commit hash recorded with every run (convention 07).

    Returns 'unknown' outside a repo — which is the normal case inside an image,
    where the commit is instead passed in by the orchestrator.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()
