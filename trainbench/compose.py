"""Hydra composition. Local/orchestrator side only.

Importing this inside a framework image would drag Hydra's `antlr4==4.9.*` pin in
and break the axolotl build. Keep the import at the edge — entry scripts that run
on a laptop or the orchestrator, never the probe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from trainbench.config import to_bench_config
from trainbench.config_schema import BenchConfig

CONFIG_DIR = "../configs"
CONFIG_NAME = "config"


def resolve(cfg: DictConfig) -> tuple[BenchConfig, dict[str, Any]]:
    """Resolve interpolations, validate, and return both the model and the plain
    mapping. The mapping is what gets shipped to a pod, so it must be exactly what
    was validated here."""
    container: Any = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return to_bench_config(container), container


def output_dir() -> Path:
    """The run's output directory, owned by Hydra (`hydra.run.dir`).

    Falls back to the working directory outside `@hydra.main`, as in unit tests.
    """
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    return Path.cwd()
