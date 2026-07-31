"""Bridge from Hydra's DictConfig to the validated schema."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from trainbench.config_schema import BenchConfig

CONFIG_DIR = "../configs"
CONFIG_NAME = "config"


def to_bench_config(cfg: DictConfig) -> BenchConfig:
    """Resolve interpolations and validate. Raises before any work starts."""
    resolved: Any = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return BenchConfig.model_validate(resolved)


def output_dir() -> Path:
    """The run's output directory, owned by Hydra (`hydra.run.dir`).

    Falls back to the working directory when called outside `@hydra.main`, which is
    the case in unit tests.
    """
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    return Path.cwd()


def git_commit() -> str:
    """Commit hash recorded with every run (convention 07). 'unknown' outside a repo."""
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
