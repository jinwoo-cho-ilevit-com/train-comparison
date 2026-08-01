"""Build the fixed MMEB subset, measure it, and refuse to push a bad one.

Speed measurement does not need the full corpus, but it does need a *fixed*
sample whose length and image-resolution distribution matches the real thing.
Sampling "short rows first" would make every downstream number optimistic, so the
subset is drawn proportionally across all 20 MMEB configs from a shuffled stream.

    python scripts/prepare_data.py data=speed
    python scripts/prepare_data.py data=speed data.limit=40   # small sample

Upstream is MrZilinXiao/MMEB_train_with_image: the authoritative TIGER-Lab
MMEB-train stores image *paths* only, and the images are not in that repo. That
mirror is community-maintained, so its commit is pinned and recorded rather than
tracked by branch.

**`pos_text` and `qry` are stored verbatim, MMEB placeholder markup included** —
`"<|image_1|>\\nRepresent the given image.\\n"`. That markup is MMEB's, not any
model's: each model under test needs its own `apply_chat_template` conversion
(different image tokens, different generation-prompt handling, per
`docs/model-spec.md`). Converting here would bake one model's template into the
shared corpus. The loader that does the conversion is Wave 3's, not this script's.

Why the quality gate exists: the subset pushed before this rewrite was validated
on row count and config coverage alone, reported `2048/2048, 20/20`, and was
corrupt — `pos_image` had been dropped, so 466 rows shared one placeholder
positive (defect D1, `docs/review-findings.md`). Counting rows is not measuring
data. Everything below either has a declared expectation that raises on drift or
a threshold that refuses the push.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from rich.console import Console

from trainbench.compose import CONFIG_DIR, CONFIG_NAME, output_dir, resolve
from trainbench.config_schema import BenchConfig
from trainbench.record import write_json

console = Console()

SPLIT = "original"
# Rows held in memory per config while shuffling the stream. Large enough that the
# sample is not just the head of the file, small enough to stream 55GB.
SHUFFLE_BUFFER = 2000
# A small-sample run draws a handful of rows per config; filling a 2000-row buffer
# to take 2 of them would download gigabytes of images to answer a smoke test. The
# buffer still covers 20x the draw, so the sample is never the head of the file.
SHUFFLE_BUFFER_PER_ROW = 20
MIN_SHUFFLE_BUFFER = 64

QUERY_IMAGE = "qry_image"
POSITIVE_IMAGE = "pos_image"

# Columns the subset keeps. Explicit hard negatives (neg_text / neg_image) are
# dropped: the loss under study is in-batch-negatives InfoNCE, where batch
# composition supplies the negatives. Add them back if a hard-negative axis is
# ever introduced.
KEPT_COLUMNS = ("qry", QUERY_IMAGE, "pos_text", POSITIVE_IMAGE)

# Read for metrics only, never stored. The upstream path names the positive image
# uniquely, which makes duplicate retrieval targets countable without hashing
# decoded pixels.
METRIC_COLUMNS = ("pos_image_path",)

# The pushed dataset's columns, derived from KEPT_COLUMNS rather than written out
# again where the dataset is built. A second hand-written column list next to the
# first is the mechanism that produced D1, and it does not stop being that
# mechanism because the second list happens to sit in the same file.
SUBSET_COLUMN_NAMES = ("mmeb_config", *KEPT_COLUMNS)

# Per-config schema, read off the Hub's dataset-info endpoint (2026-08-01) and
# declared here rather than discovered at runtime, for the same reason
# MMEB_CONFIGS is declared: an upstream change must surface as a mismatch instead
# of silently altering what the subset contains.
#
# Four of the 20 configs are text-to-image retrieval and carry no `qry_image`;
# seven have an image positive and carry `pos_image`. Defect D1 came from a single
# union-of-columns list plus `row.get()`, which turned "this config has no such
# column" and "this column was dropped" into the same silent None.
CONFIG_COLUMNS: dict[str, tuple[str, ...]] = {
    "A-OKVQA": ("qry", "qry_image", "pos_text"),
    "ChartQA": ("qry", "qry_image", "pos_text"),
    "CIRR": ("qry", "qry_image", "pos_text", "pos_image"),
    "DocVQA": ("qry", "qry_image", "pos_text"),
    "HatefulMemes": ("qry", "qry_image", "pos_text"),
    "ImageNet_1K": ("qry", "qry_image", "pos_text"),
    "InfographicsVQA": ("qry", "qry_image", "pos_text"),
    "MSCOCO": ("qry", "qry_image", "pos_text", "pos_image"),
    "MSCOCO_i2t": ("qry", "qry_image", "pos_text"),
    "MSCOCO_t2i": ("qry", "pos_text", "pos_image"),
    "N24News": ("qry", "qry_image", "pos_text"),
    "NIGHTS": ("qry", "qry_image", "pos_text", "pos_image"),
    "OK-VQA": ("qry", "qry_image", "pos_text"),
    "SUN397": ("qry", "qry_image", "pos_text"),
    "VisDial": ("qry", "pos_text", "pos_image"),
    "Visual7W": ("qry", "qry_image", "pos_text"),
    "VisualNews_i2t": ("qry", "qry_image", "pos_text"),
    "VisualNews_t2i": ("qry", "pos_text", "pos_image"),
    "VOC2007": ("qry", "qry_image", "pos_text"),
    "WebQA": ("qry", "pos_text", "pos_image"),
}

# The 20 training configs of MMEB, in the order the subset is assembled.
MMEB_CONFIGS = sorted(CONFIG_COLUMNS)

# --- Quality thresholds ------------------------------------------------------
#
# These live in code rather than in `configs/data/*.yaml` because `DataConfig`
# (trainbench/config_schema.py) is frozen by docs/CONTRACTS.md and forbids extra
# keys; moving them into config is a contract change, not a lane-A edit.
#
# Zero tolerance on the first two is deliberate and is not a distribution claim:
# a row with no positive has nothing for InfoNCE to pull toward, and a null image
# in a config whose schema declares that image is a materialisation failure, not
# a property of the data. One such row means the pipeline dropped content, and a
# pipeline that drops one row drops others.
MAX_ROWS_WITHOUT_POSITIVE = 0
MAX_ROWS_WITHOUT_QUERY_IMAGE = 0
# An image column that holds something unreadable is worse than a null: it counts
# as present everywhere above. Zero for the same reason as the two before it.
MAX_ROWS_WITH_UNREADABLE_IMAGE = 0

# The collapse gate for image positives, stated as the share held by the single
# most common target rather than as a duplicate ratio.
#
# A duplicate ratio is a function of how much of the pool is drawn. MMEB holds
# several captions per target image, so drawing 100 rows from a 100k-row config
# collides almost never while drawing 3300 (which `data=quality` does) collides
# constantly on the same upstream data — a constant ratio threshold would refuse
# the quality subset for being large, and pass a tiny draw that is entirely
# duplicated. The share of one value does not move that way: caption multiplicity
# bounds the largest legitimate group at a handful of rows whatever the draw size,
# while a collapsed column puts one value at ~100%.
MAX_SINGLE_POSITIVE_SHARE = 0.20
# Under this many rows the largest group is decided by the draw (two rows out of
# ten is already 20%), so the share is not evaluated. Small samples are checked by
# presence and schema, which do not depend on how many rows were drawn.
MIN_ROWS_FOR_SHARE_GATE = 50

# There is deliberately no threshold on duplicate *text* positives. The legitimate
# cardinality is a property of each task — ImageNet_1K has 1000 labels, VOC2007
# has 20, and VQA answers are dominated by "yes"/"no"/small integers — so any
# number chosen here would either be vacuous or reject real MMEB, and this repo
# does not put unmeasured numbers in the path of a gate (convention 16). The
# counts are recorded per config instead, which is what makes the first clean
# regeneration a reference the next one can be compared against.

# Revisions of the subset repo that must never be trained on again. Named rather
# than deleted from the Hub: a run whose result JSON records this revision was
# measured on the corrupt corpus, and that has to stay decidable after the fact.
KNOWN_CORRUPT_REVISIONS = {
    "b750b9c3263e9ef5dce225fd50aa25d7c58f1d5f": (
        "defect D1: pos_image was dropped, so 466 rows share one placeholder "
        "positive and 644 rows lost their query image"
    ),
}

SIZE_ENDPOINT = "https://datasets-server.huggingface.co/size"
DATA_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "data"


def data_config_pins() -> dict[str, dict[str, Any]]:
    import yaml

    return {
        path.name: yaml.safe_load(path.read_text()) or {}
        for path in sorted(DATA_CONFIG_DIR.glob("*.yaml"))
    }


def corrupt_pins(configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Data configs still pinning a revision this script knows to be bad.

    Reported on every run so the warning clears itself: it disappears the moment
    the regenerated revision is pinned, and until then it is in front of whoever
    is regenerating the subset.
    """
    return [
        f"{name} pins {config.get('revision')} — {KNOWN_CORRUPT_REVISIONS[str(config['revision'])]}"
        for name, config in sorted(configs.items())
        if str(config.get("revision")) in KNOWN_CORRUPT_REVISIONS
    ]


def config_row_counts(source_repo: str) -> dict[str, int]:
    """Row count per config, read from the Hub's precomputed size index.

    Not `datasets.get_dataset_config_info`: that reaches into the data for each
    config and does not return in reasonable time against a 55GB repo (measured:
    no output at all across 20 configs). One HTTP call answers for all of them in
    well under a second.
    """
    from huggingface_hub import get_session

    response = get_session().get(SIZE_ENDPOINT, params={"dataset": source_repo}, timeout=60)
    response.raise_for_status()
    size = response.json()["size"]

    wanted = set(MMEB_CONFIGS)
    counts = {
        entry["config"]: entry["num_rows"]
        for entry in size.get("splits", [])
        if entry.get("config") in wanted and entry.get("split") == SPLIT
    }
    missing = wanted - set(counts)
    if missing:
        raise RuntimeError(
            f"{source_repo} is missing expected configs for split '{SPLIT}': {sorted(missing)}. "
            "Upstream composition changed; the subset would no longer match MMEB."
        )
    return counts


def proportional_quota(counts: dict[str, int], total_rows: int) -> dict[str, int]:
    """Rows to take per config so the subset mirrors upstream composition.

    Largest-remainder apportionment: floor every share, then hand the leftover
    rows to the largest fractional parts. The sum is exactly `total_rows` for any
    input, including `total_rows < len(counts)` — there the floors are all zero
    and the leftover is the whole request, so the largest configs get one row each
    and the rest get none. That case is legal for a small sample and refused for a
    push (`gate_violations`), because a subset missing configs is not MMEB's
    composition. The previous ceil-then-trim kept a floor of one row per config
    and therefore returned *more* rows than were asked for.
    """
    if total_rows <= 0:
        raise ValueError(f"total_rows must be positive, got {total_rows}")
    grand_total = sum(counts.values())
    if grand_total <= 0:
        raise ValueError("upstream row counts sum to zero; nothing to sample from")

    exact = {name: total_rows * n / grand_total for name, n in counts.items()}
    quota = {name: math.floor(share) for name, share in exact.items()}
    leftover = total_rows - sum(quota.values())
    # Ties broken by upstream size then name so the same request always yields the
    # same quota; an unstable tiebreak would make the subset unreproducible.
    order = sorted(counts, key=lambda name: (-(exact[name] - quota[name]), -counts[name], name))
    for name in order[:leftover]:
        quota[name] += 1
    return quota


def shuffle_buffer(want: int) -> int:
    return min(SHUFFLE_BUFFER, max(MIN_SHUFFLE_BUFFER, want * SHUFFLE_BUFFER_PER_ROW))


def check_columns(name: str, available: Iterable[str]) -> tuple[str, ...]:
    """Compare the declared schema against what upstream actually offers.

    Equality, not containment: a config that *gains* a keepable column silently
    drops real content out of the subset, which is the same failure as losing one.
    """
    declared = CONFIG_COLUMNS[name]
    available = set(available)
    keepable = tuple(column for column in KEPT_COLUMNS if column in available)
    if keepable != declared:
        raise RuntimeError(
            f"{name}: upstream columns changed. Declared {list(declared)}, "
            f"upstream offers {list(keepable)} of {list(KEPT_COLUMNS)}. "
            "Update CONFIG_COLUMNS deliberately; the subset's contents depend on it."
        )
    absent = [column for column in METRIC_COLUMNS if column not in available]
    if absent:
        raise RuntimeError(f"{name}: upstream is missing metric columns {absent}")
    return declared


def take_row(name: str, row: dict[str, Any], declared: Sequence[str]) -> dict[str, Any]:
    """Read exactly the declared columns, raising on any that is absent.

    `row.get(column)` is what made defect D1 invisible: a dropped column became a
    None that looked like missing data upstream. Absent columns that the config
    genuinely does not have are filled in below *after* this check, so "not in
    this config" and "lost on the way here" stay distinguishable.
    """
    kept: dict[str, Any] = {}
    for column in (*declared, *METRIC_COLUMNS):
        if column not in row:
            raise RuntimeError(
                f"{name}: row is missing declared column {column!r}; "
                f"row has {sorted(row)}. The subset would silently lose this field."
            )
        kept[column] = row[column]
    for column in KEPT_COLUMNS:
        kept.setdefault(column, None)
    kept["mmeb_config"] = name
    return kept


def stream_rows(config: BenchConfig, name: str, want: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    stream = load_dataset(config.data.source_repo, name=name, split=SPLIT, streaming=True)
    if not stream.features:
        raise RuntimeError(
            f"{name}: upstream reports no features, so the declared schema cannot be "
            "checked. Sampling blind is how the columns were lost the first time."
        )
    declared = check_columns(name, stream.features)
    stream = stream.shuffle(seed=config.data.sample_seed, buffer_size=shuffle_buffer(want))
    return [take_row(name, row, declared) for row in stream.take(want)]


def build_dataset(rows_by_config: dict[str, list[dict[str, Any]]]) -> Any:
    """One dataset with one schema, so a null image is a null image everywhere.

    Features are given explicitly rather than inferred: a config whose rows all
    have `pos_image=None` infers a null column, which will not concatenate with an
    image column from the next config.
    """
    from datasets import Dataset, Features, Image, Value, concatenate_datasets

    features = Features(
        {
            name: Image() if name in (QUERY_IMAGE, POSITIVE_IMAGE) else Value("string")
            for name in SUBSET_COLUMN_NAMES
        }
    )
    parts = []
    for name in sorted(rows_by_config):
        rows = [{key: row[key] for key in features} for row in rows_by_config[name]]
        if rows:
            parts.append(Dataset.from_list(rows, features=features))
    if not parts:
        raise RuntimeError("no rows sampled from any config")
    return concatenate_datasets(parts)


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation: these are token and pixel counts
    read by a human comparing two regenerations, not statistics to be smoothed."""
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _image_size(value: Any) -> tuple[int, int] | None:
    """Width and height of a decoded image, or None if this is not one.

    A column can hold a value that is not an image — the authoritative TIGER-Lab
    MMEB-train stores paths, and `source_repo` is a config field. Presence checks
    cannot tell that apart from a real image, so unreadable values are counted.
    """
    size = getattr(value, "size", None)
    if isinstance(size, tuple | list) and len(size) == 2:
        return int(size[0]), int(size[1])
    return None


def _duplicate_ratio(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return sum(n for n in counts.values() if n > 1) / len(values)


def _largest_share(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    return Counter(values).most_common(1)[0][1] / len(values)


def positive_identity(name: str, row: dict[str, Any]) -> Any:
    """What makes this row's positive distinct from another's.

    For an image positive that is the target image, not `pos_text` — every row of
    an image-positive config shares the same instruction string, so counting
    distinct `pos_text` there measures the template, not the data.
    """
    if POSITIVE_IMAGE in CONFIG_COLUMNS[name]:
        return ("image", row["pos_image_path"])
    return ("text", (row["pos_text"] or "").strip())


def missing_positive(name: str, row: dict[str, Any]) -> bool:
    """A row with no positive content of the kind its config declares.

    Config-driven rather than pattern-matched: for an image-positive config the
    positive *is* the image, and `pos_text` there is MMEB's instruction template,
    which is present and meaningless whether or not the image survived. This is
    the check D1 would have failed on every one of its 466 rows.
    """
    if POSITIVE_IMAGE in CONFIG_COLUMNS[name]:
        return row[POSITIVE_IMAGE] is None
    return not (row["pos_text"] or "").strip()


def config_metrics(name: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    declared = CONFIG_COLUMNS[name]
    has_query_image = QUERY_IMAGE in declared
    pos_texts = [(row["pos_text"] or "") for row in rows]
    identities = [positive_identity(name, row) for row in rows]

    metrics: dict[str, Any] = {
        "rows": len(rows),
        "declares_query_image": has_query_image,
        "declares_positive_image": POSITIVE_IMAGE in declared,
        # Only counted where the config has the column. Where it does not, a
        # text-only query is the config, and counting it as a defect is what makes
        # a global "31.4% missing images" number unreadable.
        "rows_without_query_image": (
            sum(1 for row in rows if row[QUERY_IMAGE] is None) if has_query_image else 0
        ),
        "rows_without_positive_content": sum(1 for row in rows if missing_positive(name, row)),
        "distinct_pos_text_count": len(set(pos_texts)),
        "duplicate_pos_text_ratio": round(_duplicate_ratio(pos_texts), 4),
        "distinct_positive_count": len(set(identities)),
        "duplicate_positive_ratio": round(_duplicate_ratio(identities), 4),
        "max_single_positive_share": round(_largest_share(identities), 4),
        "qry_chars": distribution([len(row["qry"] or "") for row in rows]),
        "pos_text_chars": distribution([len(text) for text in pos_texts]),
    }
    for column, label in ((QUERY_IMAGE, "qry_image"), (POSITIVE_IMAGE, "pos_image")):
        present = [row[column] for row in rows if row[column] is not None]
        sizes = [size for value in present if (size := _image_size(value))]
        metrics[f"rows_with_unreadable_{label}"] = len(present) - len(sizes)
        metrics[f"{label}_width"] = distribution([w for w, _ in sizes])
        metrics[f"{label}_height"] = distribution([h for _, h in sizes])
        metrics[f"{label}_pixels"] = distribution([w * h for w, h in sizes])
    return metrics


def subset_metrics(rows_by_config: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Per-config metrics plus an overall roll-up.

    Per config first because every threshold here is only meaningful against the
    schema of the config it applies to; the roll-up exists so a reviewer sees one
    number per gate without reading twenty tables.
    """
    per_config = {name: config_metrics(name, rows) for name, rows in sorted(rows_by_config.items())}
    all_rows = [row for rows in rows_by_config.values() for row in rows]
    pos_texts = [(row["pos_text"] or "") for row in all_rows]
    return {
        "overall": {
            "rows": len(all_rows),
            "rows_without_query_image": sum(
                metrics["rows_without_query_image"] for metrics in per_config.values()
            ),
            # Reported apart from the count above: these configs are text-to-image
            # retrieval and have no query image by design.
            "rows_with_text_only_query_by_design": sum(
                metrics["rows"]
                for metrics in per_config.values()
                if not metrics["declares_query_image"]
            ),
            "rows_without_positive_content": sum(
                metrics["rows_without_positive_content"] for metrics in per_config.values()
            ),
            "rows_with_unreadable_image": sum(
                metrics["rows_with_unreadable_qry_image"]
                + metrics["rows_with_unreadable_pos_image"]
                for metrics in per_config.values()
            ),
            "distinct_pos_text_count": len(set(pos_texts)),
            "duplicate_pos_text_ratio": round(_duplicate_ratio(pos_texts), 4),
            "qry_chars": distribution([len(row["qry"] or "") for row in all_rows]),
            "pos_text_chars": distribution([len(text) for text in pos_texts]),
        },
        "per_config": per_config,
    }


def push_blockers(quota: dict[str, int], data: Any) -> list[str]:
    """Reasons this draw is not the subset, whatever its quality turns out to be.

    Separate from `quality_violations` because these are properties of the
    *request*, not of the data: a small sample is supposed to be a small sample.
    Folding them together would make every `data.limit` smoke run exit non-zero,
    and an exit code that is always red stops being read.
    """
    blockers: list[str] = []
    if data.limit is not None and data.limit != data.subset_rows:
        blockers.append(
            f"data.limit={data.limit} != data.subset_rows={data.subset_rows}: this is a "
            "small-sample run, and a truncated draw must not become the pushed subset"
        )
    empty = sorted(name for name, want in quota.items() if want <= 0)
    if empty:
        blockers.append(
            f"{len(empty)} configs got no quota ({empty[:3]}): "
            f"{sum(quota.values())} rows cannot cover {len(quota)} MMEB configs proportionally"
        )
    return blockers


def quality_violations(
    metrics: dict[str, Any],
    quota: dict[str, int],
    taken: dict[str, int],
) -> list[str]:
    """Measured defects in the drawn rows. Empty means the draw is usable.

    Returned as a list rather than raised one at a time so a regeneration reports
    all of its damage in one pass; fixing them one round-trip at a time against a
    55GB stream is how a corrupt subset gets shipped out of impatience.
    """
    violations: list[str] = []
    overall = metrics["overall"]

    # An empty draw satisfies every threshold below by having nothing to violate.
    # `data-pinned` passed the same way for weeks — no data configs existed, so
    # "every data config pins a commit sha" was vacuously true. A check whose
    # input set can be empty has to say so first.
    if not metrics["per_config"] or overall["rows"] == 0:
        return ["no rows were drawn at all; there is nothing to measure, let alone push"]

    short = sorted(name for name, want in quota.items() if want > 0 and taken.get(name, 0) < want)
    if short:
        detail = ", ".join(f"{name} {taken.get(name, 0)}/{quota[name]}" for name in short[:3])
        violations.append(
            f"{len(short)} configs yielded fewer rows than their quota ({detail}): "
            "the subset no longer mirrors upstream composition"
        )
    if overall["rows_without_positive_content"] > MAX_ROWS_WITHOUT_POSITIVE:
        violations.append(
            f"{overall['rows_without_positive_content']} rows have no positive content "
            f"(max {MAX_ROWS_WITHOUT_POSITIVE}); in-batch InfoNCE has nothing to pull toward"
        )
    if overall["rows_without_query_image"] > MAX_ROWS_WITHOUT_QUERY_IMAGE:
        violations.append(
            f"{overall['rows_without_query_image']} rows have no query image in configs that "
            f"declare one (max {MAX_ROWS_WITHOUT_QUERY_IMAGE}); speed would be measured on "
            "text-only rows and read as image throughput"
        )
    if overall["rows_with_unreadable_image"] > MAX_ROWS_WITH_UNREADABLE_IMAGE:
        violations.append(
            f"{overall['rows_with_unreadable_image']} image values could not be read as "
            f"images (max {MAX_ROWS_WITH_UNREADABLE_IMAGE}); the column is populated with "
            "something else, which every presence check above counts as an image"
        )
    for name, per_config in metrics["per_config"].items():
        if not per_config["declares_positive_image"]:
            continue
        share = per_config["max_single_positive_share"]
        if per_config["rows"] >= MIN_ROWS_FOR_SHARE_GATE and share > MAX_SINGLE_POSITIVE_SHARE:
            violations.append(
                f"{name}: one positive image accounts for {share:.1%} of its rows "
                f"(max {MAX_SINGLE_POSITIVE_SHARE:.0%}); identical positives inside a "
                "batch become their own negatives"
            )
    return violations


def report(metrics: dict[str, Any], violations: Sequence[str], blockers: Sequence[str]) -> None:
    overall = metrics["overall"]
    console.print("\n[bold]subset quality[/bold]")
    for key in (
        "rows",
        "rows_without_query_image",
        "rows_with_text_only_query_by_design",
        "rows_without_positive_content",
        "rows_with_unreadable_image",
        "distinct_pos_text_count",
        "duplicate_pos_text_ratio",
    ):
        console.print(f"  {key:42} {overall[key]}")
    for key in ("qry_chars", "pos_text_chars"):
        spread = overall[key]
        if spread:
            console.print(f"  {key:42} p50={spread['p50']} p95={spread['p95']}")
    for name, per_config in metrics["per_config"].items():
        pixels = per_config["qry_image_pixels"]
        # Keyed off the declaration, not off whether pixels came back: a config
        # that declares a query image and produced no readable one is a defect,
        # and printing it as "text query" would show that defect as the design.
        if not per_config["declares_query_image"]:
            spread = "text query by design"
        elif pixels:
            spread = f"qry px p50={pixels['p50']} p95={pixels['p95']}"
        else:
            spread = "[red]declares a query image, none readable[/red]"
        console.print(
            f"  [dim]{name:18}[/dim] rows={per_config['rows']:5} "
            f"dup_pos={per_config['duplicate_positive_ratio']:.2f} "
            f"top_pos={per_config['max_single_positive_share']:.2f} {spread}"
        )
    if violations:
        console.print("\n[bold red]quality gate failed[/bold red]")
        for problem in violations:
            console.print(f"  [red]-[/red] {problem}")
    else:
        console.print("\n[bold green]quality gate passed[/bold green]")
    for problem in blockers:
        console.print(f"  [yellow]not pushable:[/yellow] {problem}")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    config, _ = resolve(cfg)
    data = config.data
    # effective_rows, not subset_rows: `data.limit` is how convention 04 asks for a
    # small sample, and a stage that ignores it cannot be smoke-tested.
    requested = data.effective_rows

    for warning in corrupt_pins(data_config_pins()):
        console.print(f"[bold yellow]still pinned to a corrupt subset:[/bold yellow] {warning}")

    console.print(f"counting rows in {data.source_repo}")
    counts = config_row_counts(data.source_repo)
    quota = proportional_quota(counts, requested)

    console.print(f"sampling {requested} rows across {len(quota)} configs")
    rows_by_config: dict[str, list[dict[str, Any]]] = {}
    for name in MMEB_CONFIGS:
        want = quota[name]
        if want <= 0:
            console.print(f"  {name}: 0 (quota empty at {requested} rows)")
            continue
        rows_by_config[name] = stream_rows(config, name, want)
        console.print(f"  {name}: {len(rows_by_config[name])}/{want}")
    taken = {name: len(rows) for name, rows in rows_by_config.items()}

    metrics = subset_metrics(rows_by_config)
    violations = quality_violations(metrics, quota, taken)
    blockers = push_blockers(quota, data)
    report(metrics, violations, blockers)

    manifest = {
        "source_repo": data.source_repo,
        "subset_repo": data.repo_id,
        "subset_revision": None,
        "pushed": False,
        "sample_seed": data.sample_seed,
        "requested_rows": requested,
        "subset_rows": data.subset_rows,
        "limit": data.limit,
        "upstream_row_counts": counts,
        "quota": quota,
        "taken": taken,
        "declared_columns": {name: list(columns) for name, columns in CONFIG_COLUMNS.items()},
        "shuffle_buffer": {name: shuffle_buffer(want) for name, want in quota.items() if want},
        "thresholds": {
            "max_rows_without_positive": MAX_ROWS_WITHOUT_POSITIVE,
            "max_rows_without_query_image": MAX_ROWS_WITHOUT_QUERY_IMAGE,
            "max_rows_with_unreadable_image": MAX_ROWS_WITH_UNREADABLE_IMAGE,
            "max_single_positive_share": MAX_SINGLE_POSITIVE_SHARE,
            "min_rows_for_share_gate": MIN_ROWS_FOR_SHARE_GATE,
        },
        "quality": metrics,
        "violations": violations,
        "push_blockers": blockers,
    }
    # Written before the push so the evidence survives a failed one, and rewritten
    # after so the revision lands next to the numbers that justify it.
    path = write_json(output_dir() / "data_manifest.json", manifest)
    console.print(f"wrote {path}")

    if violations:
        # Non-zero exit even when no push was requested. The draw itself is
        # damaged, which is a finding whether or not this invocation intended to
        # publish anything — the wording says so rather than reporting a refusal
        # to do something that was never asked for.
        raise SystemExit(
            f"draw is damaged and will not be pushed: {len(violations)} quality "
            f"violations (see {path})"
        )
    if blockers and data.push_subset:
        raise SystemExit(f"refusing to push: {len(blockers)} push blockers (see {path})")
    if not data.push_subset:
        console.print("[yellow]not pushed (data.push_subset=false)[/yellow]")
        return

    console.print(f"pushing to {data.repo_id}")
    commit = build_dataset(rows_by_config).push_to_hub(data.repo_id, private=True)
    revision = getattr(commit, "oid", None) or str(commit)
    manifest |= {"subset_revision": revision, "pushed": True}
    write_json(path, manifest)
    console.print(f"revision {revision}")
    console.print("[bold]pin this revision in configs/data/*.yaml[/bold]")


if __name__ == "__main__":
    main()
