"""Build the fixed MMEB subset, measure it, and refuse to push a bad one.

Speed measurement does not need the full corpus, but it does need a *fixed*
sample whose length and image-resolution distribution matches the real thing.
Sampling "short rows first" would make every downstream number optimistic, so the
subset is drawn proportionally across all 20 MMEB configs from a shuffled stream.

    python scripts/prepare_data.py data=speed
    python scripts/prepare_data.py data=speed data.limit=40   # small sample

Each config's draw is cached as a parquet shard (`sample_config`), so an
interrupted run resumes where it stopped and the quality gate can be re-measured
against the same rows without touching the network. `TRAINBENCH_SHARD_CACHE`
moves the cache off the checkout, which the 65536-row draw needs on a machine
without ~4GB free there.

Upstream is MrZilinXiao/MMEB_train_with_image: the authoritative TIGER-Lab
MMEB-train stores image *paths* only, and the images are not in that repo. That
mirror is community-maintained, so its commit is pinned and recorded rather than
tracked by branch.

**`pos_text` and `qry` are stored verbatim, MMEB placeholder markup included** —
`"<|image_1|>\\nRepresent the given image.\\n"`. That markup is MMEB's, not any
model's: each model under test needs its own prompt-format conversion (different
image tokens, different generation-prompt handling, and gemma-4 has no chat
template at all, per `docs/model-spec.md`). Converting here would bake one model's
format into the shared corpus. The loader that converts is `trainbench/prompt.py`.

Why the quality gate exists: the subset pushed before this rewrite was validated
on row count and config coverage alone, reported `2048/2048, 20/20`, and was
corrupt — `pos_image` had been dropped, so 466 rows shared one placeholder
positive (defect D1, `docs/review-findings.md`). Counting rows is not measuring
data. Everything below either has a declared expectation that raises on drift or
a threshold that refuses the push.
"""

from __future__ import annotations

import gc
import hashlib
import io
import math
import os
import resource
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from rich.console import Console

from trainbench.compose import CONFIG_DIR, CONFIG_NAME, output_dir, resolve
from trainbench.config import git_state
from trainbench.config_schema import CORRUPT_DATA_REVISIONS, BenchConfig
from trainbench.metrics import percentile
from trainbench.record import package_versions, write_json

console = Console()

SPLIT = "original"
# The split the subset is published under. Upstream's is "original"; renaming it
# here would be gratuitous, but the pushed subset is what every run loads and
# `train` is what a consumer expects to find.
SUBSET_SPLIT = "train"
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
IMAGE_COLUMNS = (QUERY_IMAGE, POSITIVE_IMAGE)

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

# What a cached shard holds: the pushed columns plus the metric-only ones, which
# is exactly what `take_row` produces. Derived for the same reason as above.
SHARD_COLUMNS = (*SUBSET_COLUMN_NAMES, *METRIC_COLUMNS)

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

# Revisions that must never be trained on again. Defined in config_schema, which
# refuses to build a config that pins one — a second copy here would be a second
# definition of what "corrupt" means, and this file already has a defect named
# after exactly that (D1, two column lists).
KNOWN_CORRUPT_REVISIONS = CORRUPT_DATA_REVISIONS

SIZE_ENDPOINT = "https://datasets-server.huggingface.co/size"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CONFIG_DIR = REPO_ROOT / "configs" / "data"

# Where each config's draw is parked so a killed run resumes instead of streaming
# it again. Under the git-ignored `/data/`, and overridable by environment because
# the draw is ~4GB and the machine preparing it may not have that where the
# checkout lives.
#
# A module constant rather than a `DataConfig` field: that schema is frozen by
# docs/CONTRACTS.md and forbids extra keys, and a scratch path is not an
# experiment parameter — nothing about the pushed subset changes with it. Same
# reasoning as the quality thresholds above.
# Rows pulled out of the pushed parquet at a time while verifying it. Small enough
# that a 65536-row subset is checked without holding it, large enough that the
# per-batch overhead does not dominate the image decoding.
ARTIFACT_BATCH_ROWS = 256

SHARD_CACHE_ENV = "TRAINBENCH_SHARD_CACHE"
DEFAULT_SHARD_CACHE = REPO_ROOT / "data" / "subset-shards"


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


def open_stream(config: BenchConfig, name: str) -> tuple[Any, tuple[str, ...]]:
    """Open the upstream stream and check its schema against what this config declares.

    Split out of `stream_rows` because the schema check has to run whether or not
    the rows are about to be drawn. It reads `features` and no rows, so a config
    already satisfied from the shard cache still pays only one HTTP request.
    """
    from datasets import load_dataset

    stream = load_dataset(
        config.data.source_repo,
        name=name,
        split=SPLIT,
        streaming=True,
        revision=config.data.source_revision,
    )
    if not stream.features:
        raise RuntimeError(
            f"{name}: upstream reports no features, so the declared schema cannot be "
            "checked. Sampling blind is how the columns were lost the first time."
        )
    return stream, check_columns(name, stream.features)


def stream_rows(config: BenchConfig, name: str, want: int) -> list[dict[str, Any]]:
    """Draw `want` rows, holding their images as encoded bytes rather than pixels.

    The cast is the difference between a run that finishes and one the kernel
    kills. `datasets` decodes an `Image` column on every row it hands out —
    `Image.decode_example` calls `PIL.Image.load()` explicitly (features/image.py,
    "to avoid too many open files") — and `main` accumulates every config's rows
    until the push. Measured on MSCOCO_i2t: 270.9 KiB/row decoded against
    15.9 KiB/row encoded, a factor of 17. At `data=quality`'s 65536 rows that is
    tens of GB, and the first attempt was SIGKILLed at 40377 rows with 48GB of RAM
    and 20GB of swap exhausted. `data=speed` never hit it because 2048 rows is 32x
    smaller, not because the accumulation is bounded.

    Nothing downstream needs pixels: `_image_size` reads the header, and the push
    stores the same encoded bytes it would have re-encoded.
    """
    from datasets import Image

    stream, declared = open_stream(config, name)
    for column in IMAGE_COLUMNS:
        if column in stream.features:
            stream = stream.cast_column(column, Image(decode=False))
    stream = stream.shuffle(seed=config.data.sample_seed, buffer_size=shuffle_buffer(want))
    return [take_row(name, row, declared) for row in stream.take(want)]


def shard_cache_dir() -> Path:
    return Path(os.environ.get(SHARD_CACHE_ENV) or DEFAULT_SHARD_CACHE)


def shard_path(data: Any, name: str, want: int) -> Path:
    """Where this config's draw is cached.

    Every input that changes which rows are drawn is in the filename: a different
    upstream repo, upstream commit, seed, or quota reads a different file rather
    than silently reusing the last one. `want` moves with the total, so
    `data=speed` and `data=quality` never share a shard even though they share a
    seed — and it also fixes the shuffle buffer, so the two are different draws
    rather than one draw read to two depths.
    """
    source = data.source_repo.replace("/", "__")
    return shard_cache_dir() / (
        f"{source}@{data.source_revision[:12]}--{SPLIT}"
        f"--seed{data.sample_seed}--{name}--{want}.parquet"
    )


def _shard_schema() -> Any:
    import pyarrow as pa

    # The same struct `datasets` writes for an undecoded image column, so a shard
    # is readable by anything that reads the pushed subset.
    image = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    return pa.schema(
        [(column, image if column in IMAGE_COLUMNS else pa.string()) for column in SHARD_COLUMNS]
    )


def write_shard(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write the draw, then rename it into place.

    Atomic because the failure this cache exists for is a killed process: a shard
    half-written when the kernel arrived would be trusted on the next run and the
    subset would quietly lose the rest of that config's quota.

    The staging name carries the pid. Two processes preparing the same shard
    otherwise write the same staging file, and whichever renames second dies on a
    FileNotFoundError that nothing catches.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(f".{os.getpid()}.partial")
    try:
        pq.write_table(pa.Table.from_pylist(list(rows), schema=_shard_schema()), staging)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def read_shard(path: Path) -> list[dict[str, Any]]:
    """Read a cached draw, refusing a file whose columns are not the ones expected.

    `pq.read_table(..., schema=...)` fills a column the file does not have with
    nulls rather than failing, which would let a shard written by older code come
    back as rows with a silently absent `pos_image` — defect D1's shape, reached
    through the resume path instead of the sampler. `take_row` raises on the same
    situation for exactly this reason, and the two have to agree.
    """
    import pyarrow.parquet as pq

    stored = tuple(pq.read_schema(path).names)
    if stored != SHARD_COLUMNS:
        raise RuntimeError(
            f"{path.name}: cached shard holds columns {list(stored)}, expected "
            f"{list(SHARD_COLUMNS)}. It was written by different code; delete it."
        )
    return pq.read_table(path, schema=_shard_schema()).to_pylist()


def sample_config(config: BenchConfig, name: str, want: int) -> tuple[list[dict[str, Any]], str]:
    """This config's rows, from the cache if they were already drawn.

    Convention 04 asks every pipeline stage to save and resume; this one did not,
    and 24 minutes of streaming were lost to a single kill. Resuming is also what
    makes the draw cheap to re-measure: the quality gate can be re-run against the
    same rows without touching the network.

    The upstream schema is checked on every call, cache hit or not. It was checked
    only inside `stream_rows` at first, so a second run against a cached draw could
    not see that upstream had dropped a column — the manifest went on recording
    `CONFIG_COLUMNS` as the declared schema while nothing had compared it to
    anything. That is defect D1's shape with the guard moved out of the path
    rather than deleted, and the check costs one request because it reads no rows.
    """
    open_stream(config, name)
    path = shard_path(config.data, name, want)
    if path.exists():
        try:
            rows = read_shard(path)
        except Exception as problem:
            # A shard the cache cannot read is worse than no shard: every retry
            # fails in the same place, so the cache built to survive a kill would
            # make the failure permanent. Drop it and draw again.
            console.print(f"  [yellow]{name}: discarding unreadable shard[/yellow] ({problem})")
            path.unlink(missing_ok=True)
        else:
            if len(rows) == want:
                return rows, "cached"
            # Not repaired in place: a short shard means the draw it holds is not
            # the one this quota asks for, and topping it up would take rows from
            # a different position in the shuffled stream.
            path.unlink()
    rows = stream_rows(config, name, want)
    write_shard(rows, path)
    return rows, "streamed"


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


def distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _image_size(value: Any) -> tuple[int, int] | None:
    """Width and height of a decodable image, or None if this is not one.

    The pixels are decoded and thrown away rather than only the header parsed.
    `PIL.Image.open` is lazy — a file with an intact header and a corrupt payload
    returns a size and raises only on `load()` — so a header-only check made
    `MAX_ROWS_WITH_UNREADABLE_IMAGE = 0` mean "0 rows whose header parsed", not
    "0 rows that cannot be read". Before rows were held undecoded, `datasets`
    called `load()` on every row itself and a broken payload surfaced upstream of
    this gate; holding encoded bytes removed that without replacing it.

    One image is decoded at a time and not retained, so this does not reintroduce
    the accumulation it replaced (measured: 2320 images in 2.2s for `data=speed`).

    A column can hold a value that is not an image — the authoritative TIGER-Lab
    MMEB-train stores paths, and `source_repo` is a config field. Presence checks
    cannot tell that apart from a real image, so unreadable values are counted.
    """
    from PIL import Image as PILImage

    data = value.get("bytes") if isinstance(value, Mapping) else None
    if not data:
        return None
    try:
        with PILImage.open(io.BytesIO(data)) as image:
            width, height = image.size
            image.load()
    # PIL's decode failures are not one hierarchy, and every name here was reached
    # by an actual malformed input rather than added defensively:
    #   OSError                 truncated payload; UnidentifiedImageError derives from it
    #   SyntaxError             a corrupt chunk found partway through load()
    #   ValueError              a header PIL parses but cannot turn into a mode
    #   DecompressionBombError  derives from Exception, so it escaped the first
    #                           three and killed the run *after* all streaming and
    #                           caching — and the cached shard then reproduced that
    #                           death on every retry
    except (OSError, SyntaxError, ValueError, PILImage.DecompressionBombError):
        return None
    return int(width), int(height)


def image_missing(value: Any) -> bool:
    """No image at all, as against one that is present but cannot be read.

    Undecoded values arrive as `{"bytes": ..., "path": ...}`. A value carrying a
    path but no bytes is upstream storing a reference instead of the image, which
    is counted as unreadable (`MAX_ROWS_WITH_UNREADABLE_IMAGE`) rather than absent
    — collapsing the two would let a repo full of dangling paths pass as a repo
    full of images.
    """
    if value is None:
        return True
    if isinstance(value, Mapping):
        return not value.get("bytes") and not value.get("path")
    return False


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
        return image_missing(row[POSITIVE_IMAGE])
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
            sum(1 for row in rows if image_missing(row[QUERY_IMAGE])) if has_query_image else 0
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
        present = [row[column] for row in rows if not image_missing(row[column])]
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


def _peak_rss_bytes() -> int:
    """Peak resident set size so far, in bytes.

    `ru_maxrss` is bytes on macOS and kilobytes on Linux, and this number is meant
    to be compared across the two, so the unit is normalised rather than recorded
    as whatever the host returns.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def ungated_configs(metrics: dict[str, Any]) -> dict[str, int]:
    """Image-positive configs whose row count leaves the collapse gate unevaluated.

    Recorded rather than inferred from the thresholds because the exemption is
    invisible in the output that reports the gate as passed: `data=speed` draws 30
    NIGHTS rows and 33 WebQA rows, so 2 of its 7 image-positive configs are exempt,
    and the "largest single image positive" number quoted from that draw came from
    one of the exempt two. A gate that did not run is not a gate that passed.
    """
    return {
        name: per_config["rows"]
        for name, per_config in metrics["per_config"].items()
        if per_config["declares_positive_image"] and per_config["rows"] < MIN_ROWS_FOR_SHARE_GATE
    }


def artifact_identity(name: str, row: dict[str, Any]) -> Any:
    """`positive_identity` recomputed from what the pushed subset actually carries.

    `pos_image_path` is metric-only and never pushed, so an image positive is
    identified downstream by the digest of its bytes. The two groupings are not
    interchangeable in general — one path is always one byte string, but two paths
    can hold identical bytes — so digest grouping can only merge groups, never
    split them. The share it produces is therefore >= the manifest's, which makes
    it a sound check to apply the same threshold to.
    """
    if POSITIVE_IMAGE in CONFIG_COLUMNS[name]:
        value = row[POSITIVE_IMAGE]
        data = value.get("bytes") if isinstance(value, Mapping) else None
        return ("image", hashlib.sha256(data).hexdigest() if data else None)
    return ("text", (row["pos_text"] or "").strip())


def _declared_columns(data: Any, revision: str) -> tuple[str, ...]:
    """The column names the pushed revision's card advertises.

    Read on its own and first because this is the check that would have caught
    defect D7 in seconds: `Dataset.push_to_hub` kept the repo's existing
    `dataset_info`, so a corrected five-column subset shipped under the
    four-column card left over from D1 and `load_dataset` raised CastError on it.
    Building the streaming dataset reads the card, not the data.
    """
    from datasets import load_dataset

    dataset = load_dataset(data.repo_id, revision=revision, split=SUBSET_SPLIT, streaming=True)
    return tuple(sorted(dataset.features))


def _artifact_batches(data: Any, revision: str) -> Iterable[list[dict[str, Any]]]:
    """The pushed rows, in batches, downloaded the fast way.

    Not `load_dataset(streaming=True)`: that reads through fsspec as ranged GETs on
    one connection and measured 0.2-0.7 MB/s here, which put a 4.75GB subset out of
    reach at hours per check — and a check that takes hours is one that gets
    skipped, which is the same as not having it. `hf_hub_download` goes through the
    xet client, which fetches chunks in parallel: 12.3 MiB/s measured on the same
    connection, ~20x, turning the full pass into minutes.

    Batched rather than read whole so memory stays flat regardless of subset size.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        name
        for name in HfApi().list_repo_files(data.repo_id, repo_type="dataset", revision=revision)
        if name.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"{data.repo_id}@{revision} has no parquet files to verify")
    for name in files:
        local = hf_hub_download(data.repo_id, name, repo_type="dataset", revision=revision)
        parquet = pq.ParquetFile(local)
        stored = tuple(sorted(parquet.schema_arrow.names))
        if stored != tuple(sorted(SUBSET_COLUMN_NAMES)):
            raise RuntimeError(
                f"{name}: holds columns {list(stored)}, expected {sorted(SUBSET_COLUMN_NAMES)}"
            )
        for batch in parquet.iter_batches(batch_size=ARTIFACT_BATCH_ROWS):
            yield batch.to_pylist()


def verify_pushed(data: Any, revision: str, metrics: dict[str, Any]) -> list[str]:
    """Read the revision that was just pushed and re-derive the gate from it.

    The reason this exists: a push that returned a revision was treated as a
    finished subset, and the revision it returned could not be loaded at all (D7).
    Every number justifying that pin came from rows in memory; nothing had read the
    artifact. `data-pinned` was green throughout, because it checks that the
    revision string looks like a sha.

    Two stages, cheap one first, so a broken card fails in seconds with a message
    about the card rather than after a full download.
    """
    problems: list[str] = []

    declared = _declared_columns(data, revision)
    if declared != tuple(sorted(SUBSET_COLUMN_NAMES)):
        problems.append(
            f"the pushed card declares columns {list(declared)}, not "
            f"{sorted(SUBSET_COLUMN_NAMES)}; consumers cannot read this revision"
        )
        # Every count below would be derived from columns that are not there.
        return problems

    seen: Counter[str] = Counter()
    identities: dict[str, Counter[Any]] = {}
    no_query_image: Counter[str] = Counter()
    no_positive: Counter[str] = Counter()
    unreadable: Counter[str] = Counter()
    for rows in _artifact_batches(data, revision):
        for row in rows:
            name = row["mmeb_config"]
            if name not in CONFIG_COLUMNS:
                problems.append(f"the pushed subset carries an unknown config {name!r}")
                return problems
            seen[name] += 1
            identities.setdefault(name, Counter())[artifact_identity(name, row)] += 1
            if QUERY_IMAGE in CONFIG_COLUMNS[name] and image_missing(row[QUERY_IMAGE]):
                no_query_image[name] += 1
            if missing_positive(name, row):
                no_positive[name] += 1
            for column in IMAGE_COLUMNS:
                if not image_missing(row[column]) and _image_size(row[column]) is None:
                    unreadable[name] += 1

    for name, per_config in metrics["per_config"].items():
        for label, measured, expected in (
            ("rows", seen[name], per_config["rows"]),
            (
                "rows_without_query_image",
                no_query_image[name],
                per_config["rows_without_query_image"],
            ),
            (
                "rows_without_positive_content",
                no_positive[name],
                per_config["rows_without_positive_content"],
            ),
            (
                "rows_with_unreadable_image",
                unreadable[name],
                per_config["rows_with_unreadable_qry_image"]
                + per_config["rows_with_unreadable_pos_image"],
            ),
        ):
            if measured != expected:
                problems.append(
                    f"{name}: the artifact has {label}={measured}, the manifest says {expected}"
                )
        if not per_config["declares_positive_image"] or name not in identities:
            continue
        share = identities[name].most_common(1)[0][1] / max(1, seen[name])
        if seen[name] >= MIN_ROWS_FOR_SHARE_GATE and share > MAX_SINGLE_POSITIVE_SHARE:
            problems.append(
                f"{name}: one positive image is {share:.1%} of the pushed rows "
                f"(max {MAX_SINGLE_POSITIVE_SHARE:.0%})"
            )
    extra = sorted(set(seen) - set(metrics["per_config"]))
    if extra:
        problems.append(f"the artifact carries configs the manifest does not describe: {extra}")
    return problems


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
    exempt = ungated_configs(metrics)
    if exempt:
        detail = ", ".join(f"{name} {rows}" for name, rows in sorted(exempt.items()))
        console.print(
            f"\n[yellow]collapse gate not evaluated[/yellow] for {len(exempt)} image-positive "
            f"config(s) under {MIN_ROWS_FOR_SHARE_GATE} rows: {detail}"
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
    # Read through `config.data` rather than a local alias. The alias was shorter and
    # it made every knob it touched invisible to `config-consumed`, which follows the
    # attribute chain — so `data.subset_rows` and `data.push_subset` were reported as
    # settings nothing reads while the run was reading them.
    #
    # effective_rows, not subset_rows: `data.limit` is how convention 04 asks for a
    # small sample, and a stage that ignores it cannot be smoke-tested.
    requested = config.data.effective_rows

    for warning in corrupt_pins(data_config_pins()):
        console.print(f"[bold yellow]still pinned to a corrupt subset:[/bold yellow] {warning}")

    console.print(f"counting rows in {config.data.source_repo}")
    counts = config_row_counts(config.data.source_repo)
    quota = proportional_quota(counts, requested)

    console.print(f"sampling {requested} rows across {len(quota)} configs")
    console.print(f"[dim]shard cache: {shard_cache_dir()}[/dim]")
    rows_by_config: dict[str, list[dict[str, Any]]] = {}
    for name in MMEB_CONFIGS:
        want = quota[name]
        if want <= 0:
            console.print(f"  {name}: 0 (quota empty at {requested} rows)")
            continue
        rows, source = sample_config(config, name, want)
        rows_by_config[name] = rows
        console.print(f"  {name}: {len(rows)}/{want} [dim]({source})[/dim]")
    taken = {name: len(rows) for name, rows in rows_by_config.items()}

    metrics = subset_metrics(rows_by_config)
    violations = quality_violations(metrics, quota, taken)
    blockers = push_blockers(quota, config.data)
    report(metrics, violations, blockers)

    git = git_state()
    manifest = {
        # The three fields that make this file evidence rather than a note: which
        # code drew the subset, whether that code was actually committed, and what
        # it was asked for. Named as `trainbench/record.py` names them so the two
        # kinds of record in docs/evidence/ answer "where did this come from" the
        # same way — and `audit_plan.py`'s `evidence-committed` requires them.
        #
        # `config` is the data section alone, not the whole composed config. The
        # draw depends on nothing else, and recording `attn.name` next to a subset
        # that never built a model would imply it mattered.
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_source": git["source"],
        "config": config.data.model_dump(mode="json"),
        # datasets decides the shuffle and the image encoding, pillow decodes, and
        # pyarrow writes the shards. A subset is not reproducible across changes in
        # any of them, so the versions are part of the record.
        "packages": package_versions(),
        "source_repo": config.data.source_repo,
        "source_revision": config.data.source_revision,
        "subset_repo": config.data.repo_id,
        "subset_revision": None,
        "pushed": False,
        "sample_seed": config.data.sample_seed,
        "requested_rows": requested,
        "subset_rows": config.data.subset_rows,
        "limit": config.data.limit,
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
        # An image-positive config under the row floor gets no collapse gate, and
        # the output above still reports the gate as passed. Named here so a
        # reviewer can see which numbers no gate stood behind.
        "collapse_gate_not_evaluated": ungated_configs(metrics),
        "violations": violations,
        "push_blockers": blockers,
        # Kernel-maintained high-water mark, not a sample. The evidence that the
        # undecoded-rows fix worked was `ps -o rss=` every 120s, which cannot see a
        # peak inside a sampling interval and in fact recorded a *lower* figure at
        # 65536 rows than at 40377 — impossible for a monotone accumulator, so both
        # samples were measuring allocator noise rather than the accumulation.
        "peak_rss_bytes": _peak_rss_bytes(),
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
    if blockers and config.data.push_subset:
        raise SystemExit(f"refusing to push: {len(blockers)} push blockers (see {path})")
    if not config.data.push_subset:
        console.print("[yellow]not pushed (data.push_subset=false)[/yellow]")
        return

    from datasets import DatasetDict

    console.print(f"pushing to {config.data.repo_id}")
    # Pushed as a DatasetDict, not a bare Dataset. `Dataset.push_to_hub` onto a repo
    # that already carries a card keeps the card's existing `dataset_info`
    # (datasets/arrow_dataset.py, `info_to_dump = repo_info`), so a repush declares
    # the *previous* schema. That is how a corrected five-column subset landed under
    # defect D1's four-column card and could not be loaded at all. DatasetDict passes
    # `remove_other_splits=True`, which takes the branch that rebuilds features.
    commit = DatasetDict({SUBSET_SPLIT: build_dataset(rows_by_config)}).push_to_hub(
        config.data.repo_id, private=True
    )
    revision = getattr(commit, "oid", None) or str(commit)
    manifest |= {"subset_revision": revision, "pushed": True}
    write_json(path, manifest)
    console.print(f"revision {revision}")

    # The draw has been published and the metrics are already computed, so nothing
    # below needs the rows — and holding them makes the verification slower, not
    # just fatter. Measured on `data=quality`: with ~5GB of rows still resident the
    # artifact came down at 3.4 MB/s against 12.3 MB/s for the same file fetched by
    # itself, on a host whose swap was full. Verification is the one stage that
    # cannot be resumed from the shard cache, so it is the one that should run lean.
    rows_by_config.clear()
    gc.collect()

    console.print("verifying the pushed revision")
    problems = verify_pushed(config.data, revision, metrics)
    manifest |= {"artifact_verified": not problems, "artifact_problems": problems}
    write_json(path, manifest)
    if problems:
        for problem in problems:
            console.print(f"  [red]-[/red] {problem}")
        raise SystemExit(
            f"pushed {revision} but it does not match the manifest: {len(problems)} "
            f"problem(s) (see {path}). Do not pin this revision."
        )
    console.print("[bold green]artifact matches the manifest[/bold green]")
    console.print(f"[bold]pin {revision} in configs/data/*.yaml[/bold]")


if __name__ == "__main__":
    main()
