"""Build the fixed MMEB subset and push it to a private dataset repo.

Speed measurement does not need the full corpus, but it does need a *fixed*
sample whose length and image-resolution distribution matches the real thing.
Sampling "short rows first" would make every downstream number optimistic, so the
subset is drawn proportionally across all 20 MMEB configs from a shuffled stream.

    python scripts/prepare_data.py data=speed

Upstream is MrZilinXiao/MMEB_train_with_image: the authoritative TIGER-Lab
MMEB-train stores image *paths* only, and the images are not in that repo. That
mirror is community-maintained, so its commit is pinned and recorded rather than
tracked by branch.
"""

from __future__ import annotations

import math
from typing import Any

import hydra
from omegaconf import DictConfig
from rich.console import Console

from trainbench.compose import CONFIG_DIR, CONFIG_NAME, output_dir, resolve
from trainbench.config_schema import BenchConfig
from trainbench.record import write_json

console = Console()

# The 20 training configs of MMEB. Listed explicitly so a change upstream shows up
# as a mismatch rather than silently altering the subset's composition.
MMEB_CONFIGS = [
    "A-OKVQA",
    "ChartQA",
    "CIRR",
    "DocVQA",
    "HatefulMemes",
    "ImageNet_1K",
    "InfographicsVQA",
    "MSCOCO",
    "MSCOCO_i2t",
    "MSCOCO_t2i",
    "N24News",
    "NIGHTS",
    "OK-VQA",
    "SUN397",
    "VisDial",
    "Visual7W",
    "VisualNews_i2t",
    "VisualNews_t2i",
    "VOC2007",
    "WebQA",
]

SPLIT = "original"
# Rows held in memory per config while shuffling the stream. Large enough that the
# sample is not just the head of the file, small enough to stream 55GB.
SHUFFLE_BUFFER = 2000


def config_row_counts(source_repo: str) -> dict[str, int]:
    """Row count per config, from the Hub rather than a hardcoded table."""
    from datasets import get_dataset_config_info

    counts = {}
    for name in MMEB_CONFIGS:
        info = get_dataset_config_info(source_repo, config_name=name)
        counts[name] = info.splits[SPLIT].num_examples
    return counts


def proportional_quota(counts: dict[str, int], total_rows: int) -> dict[str, int]:
    """Rows to take per config so the subset mirrors upstream composition."""
    grand_total = sum(counts.values())
    quota = {name: math.ceil(total_rows * n / grand_total) for name, n in counts.items()}
    # Ceil overshoots; trim from the largest configs so the total lands exactly.
    overshoot = sum(quota.values()) - total_rows
    for name in sorted(quota, key=lambda k: -quota[k]):
        if overshoot <= 0:
            break
        take = min(overshoot, quota[name] - 1)
        quota[name] -= take
        overshoot -= take
    return quota


# MMEB configs carry between 4 and 8 columns; only these four exist everywhere.
# Explicit hard negatives (neg_text / neg_image_path) are dropped: the loss under
# study is in-batch-negatives InfoNCE, where batch composition supplies the
# negatives. Add them back if a hard-negative axis is ever introduced.
SUBSET_COLUMNS = ["qry", "qry_image", "pos_text"]


def sample_subset(config: BenchConfig, quota: dict[str, int]) -> tuple[Any, dict[str, int]]:
    from datasets import Dataset, concatenate_datasets, load_dataset

    parts, taken = [], {}
    for name, want in quota.items():
        stream = load_dataset(
            config.data.source_repo, name=name, split=SPLIT, streaming=True
        ).shuffle(seed=config.data.sample_seed, buffer_size=SHUFFLE_BUFFER)
        rows = [{column: row.get(column) for column in SUBSET_COLUMNS} for row in stream.take(want)]
        if not rows:
            taken[name] = 0
            console.print(f"  {name}: 0/{want} (empty stream)")
            continue
        part = Dataset.from_list(rows)
        part = part.add_column("mmeb_config", [name] * len(part))
        parts.append(part)
        taken[name] = len(part)
        console.print(f"  {name}: {len(part)}/{want}")
    if not parts:
        raise RuntimeError("no rows sampled from any config")
    return concatenate_datasets(parts), taken


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    config, _ = resolve(cfg)

    console.print(f"counting rows in {config.data.source_repo}")
    counts = config_row_counts(config.data.source_repo)
    quota = proportional_quota(counts, config.data.subset_rows)

    console.print(f"sampling {config.data.subset_rows} rows across {len(quota)} configs")
    subset, taken = sample_subset(config, quota)

    if config.data.push_subset:
        console.print(f"pushing to {config.data.repo_id}")
        commit = subset.push_to_hub(config.data.repo_id, private=True)
        revision = getattr(commit, "oid", None) or str(commit)
    else:
        revision = None
        console.print("[yellow]not pushed (data.push_subset=false)[/yellow]")
    manifest = {
        "source_repo": config.data.source_repo,
        "subset_repo": config.data.repo_id,
        "subset_revision": revision,
        "sample_seed": config.data.sample_seed,
        "requested_rows": config.data.subset_rows,
        "upstream_row_counts": counts,
        "quota": quota,
        "taken": taken,
        "shuffle_buffer": SHUFFLE_BUFFER,
    }
    path = write_json(output_dir() / "data_manifest.json", manifest)
    console.print(f"revision {revision}")
    console.print(f"wrote {path}")
    console.print("[bold]pin this revision in configs/data/*.yaml[/bold]")


if __name__ == "__main__":
    main()
