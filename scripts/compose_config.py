"""Compose an experiment variant with Hydra and write the resolved config JSON.

The pod side never composes: it receives this file. That keeps Hydra out of the
framework images and guarantees the config that ran is the config recorded.

    python scripts/compose_config.py model=gemma4_e2b framework=unsloth run=probe
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from rich.console import Console

from trainbench.compose import CONFIG_DIR, CONFIG_NAME, output_dir, resolve
from trainbench.record import write_json

console = Console()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    config, container = resolve(cfg)
    path = write_json(output_dir() / "resolved_config.json", container)
    console.print(f"{config.framework.name} x {config.model.name} ({config.run.purpose}) -> {path}")


if __name__ == "__main__":
    main()
