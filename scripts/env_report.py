"""Resolve a config, report the environment it would run in, and record it.

This is the smallest end-to-end path through the harness: Hydra composition ->
schema validation -> device resolution -> seeded state -> atomic result write.
It loads no model; framework x model probing is `scripts/verify_env.py`.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from trainbench.compose import CONFIG_DIR, CONFIG_NAME, output_dir, resolve
from trainbench.device import get_device
from trainbench.record import build_record, write_json
from trainbench.seed import set_seed

console = Console()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    config, _ = resolve(cfg)
    device = get_device(config.device)
    set_seed(config.train.seed, deterministic=config.train.deterministic)

    record = build_record(config, device)
    path = write_json(output_dir() / "env_report.json", record)

    table = Table(title="environment", show_header=False)
    table.add_row("model", f"{config.model.name} ({config.model.arch})")
    table.add_row("framework", config.framework.name)
    table.add_row("purpose", config.run.purpose)
    table.add_row("device", str(device))
    table.add_row("deterministic", str(config.train.deterministic))
    table.add_row("data.limit", str(config.data.limit))
    table.add_row("git", record["git_commit"][:12])
    for name, version in record["packages"].items():
        table.add_row(name, version)
    console.print(table)
    console.print(f"wrote {path}")


if __name__ == "__main__":
    main()
