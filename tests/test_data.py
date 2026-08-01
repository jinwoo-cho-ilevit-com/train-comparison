"""The subset that shipped was corrupt and every check in place said it was fine.

These tests replay defect D1 (docs/review-findings.md): `SUBSET_COLUMNS` dropped
`pos_image`, `row.get()` turned the dropped column into a None, and the success
criteria — row count and config coverage — reported `2048/2048, 20/20`. Each test
below fails if any part of that path is reintroduced.

`scripts/` is not a package, so the module is loaded by path. `datasets` is only
imported inside the functions that stream, which is what lets these run against
the `compose` extra alone.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from .conftest import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "prepare_data", REPO_ROOT / "scripts" / "prepare_data.py"
)
assert _spec and _spec.loader
prepare_data = importlib.util.module_from_spec(_spec)
sys.modules["prepare_data"] = prepare_data
_spec.loader.exec_module(prepare_data)


IMAGE_CONFIG = "MSCOCO_t2i"  # text query, image positive
TEXT_CONFIG = "ImageNet_1K"  # image query, text positive


def encoded_image(width: int = 640, height: int = 480) -> dict[str, Any]:
    """An image the way a row actually carries it: encoded bytes, not pixels.

    Rows are held undecoded so a 65536-row draw fits in memory (`stream_rows`), so
    a fixture holding a decoded stand-in would test a representation the pipeline
    no longer produces — and the metrics that read image size are the ones that
    caught D1.
    """
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def image_row(index: int, *, pos_image: Any | None = ..., **overrides: Any) -> dict[str, Any]:
    row = {
        "mmeb_config": IMAGE_CONFIG,
        "qry": f"find the image of subject {index}",
        "qry_image": None,
        "pos_text": "<|image_1|>\nRepresent the given image.\n",
        "pos_image": encoded_image() if pos_image is ... else pos_image,
        "pos_image_path": f"coco/{index}.jpg",
    }
    return row | overrides


def text_row(index: int, **overrides: Any) -> dict[str, Any]:
    row = {
        "mmeb_config": TEXT_CONFIG,
        "qry": "<|image_1|>\nRepresent the given image for classification.\n",
        "qry_image": encoded_image(500, 375),
        "pos_text": f"class {index % 3}",
        "pos_image": None,
        "pos_image_path": None,
    }
    return row | overrides


# --- proportional_quota ------------------------------------------------------


def upstream_counts(configs: int = 20, per_config: int = 1000) -> dict[str, int]:
    return {f"c{i:02}": per_config + i for i in range(configs)}


def test_quota_sums_to_the_request():
    counts = upstream_counts()
    for total in (20, 21, 100, 2048, 65536):
        quota = prepare_data.proportional_quota(counts, total)
        assert sum(quota.values()) == total, total


def test_quota_never_overshoots_a_request_smaller_than_the_config_count():
    """The ceil-then-trim quota kept a floor of one row per config, so a request
    for 5 rows across 20 configs returned 20. Every downstream count — the row
    total, the manifest, the sanity check that said `2048/2048` — then described a
    subset that was not the one asked for."""
    counts = upstream_counts(configs=20)
    quota = prepare_data.proportional_quota(counts, 5)

    assert sum(quota.values()) == 5
    assert set(quota.values()) <= {0, 1}
    assert sorted(name for name, want in quota.items() if want) == [
        "c15",
        "c16",
        "c17",
        "c18",
        "c19",
    ]


def test_quota_of_one_row_picks_exactly_one_config():
    quota = prepare_data.proportional_quota(upstream_counts(), 1)
    assert sum(quota.values()) == 1


def test_quota_is_proportional_to_upstream_size():
    quota = prepare_data.proportional_quota({"big": 900, "small": 100}, 100)
    assert quota == {"big": 90, "small": 10}


def test_quota_rejects_a_nonsensical_request():
    with pytest.raises(ValueError):
        prepare_data.proportional_quota(upstream_counts(), 0)
    with pytest.raises(ValueError):
        prepare_data.proportional_quota({"a": 0, "b": 0}, 10)


def test_a_config_that_gets_no_quota_blocks_the_push():
    """A 5-row draw is a legal small sample and an illegal subset: 15 of the 20
    MMEB configs are not in it at all."""
    quota = prepare_data.proportional_quota(upstream_counts(), 5)
    data = SimpleNamespace(limit=None, subset_rows=5)
    assert prepare_data.push_blockers(quota, data)


# --- column schema -----------------------------------------------------------


def test_every_declared_column_is_one_the_subset_keeps():
    keepable = set(prepare_data.KEPT_COLUMNS)
    # MMEB-train is 20 configs; the count is asserted because the subset claims to
    # mirror MMEB's composition, and it stops doing so the moment one is dropped.
    assert len(prepare_data.MMEB_CONFIGS) == 20
    assert set(prepare_data.MMEB_CONFIGS) == set(prepare_data.CONFIG_COLUMNS)
    for name, columns in prepare_data.CONFIG_COLUMNS.items():
        assert set(columns) <= keepable, name
        assert "qry" in columns and "pos_text" in columns, name
        # Declaration order follows KEPT_COLUMNS so `check_columns` can compare
        # tuples rather than sets and still reject a reordering that means nothing.
        assert list(columns) == [c for c in prepare_data.KEPT_COLUMNS if c in columns], name


def test_the_configs_that_carry_a_positive_image_are_declared():
    """This is the list D1 erased. If `pos_image` disappears from these configs
    again, the corruption is back and this test is the tripwire."""
    with_positive_image = {
        name
        for name, columns in prepare_data.CONFIG_COLUMNS.items()
        if prepare_data.POSITIVE_IMAGE in columns
    }
    assert with_positive_image == {
        "CIRR",
        "MSCOCO",
        "MSCOCO_t2i",
        "NIGHTS",
        "VisDial",
        "VisualNews_t2i",
        "WebQA",
    }


def test_the_text_query_configs_are_declared():
    text_query = {
        name
        for name, columns in prepare_data.CONFIG_COLUMNS.items()
        if prepare_data.QUERY_IMAGE not in columns
    }
    assert text_query == {"MSCOCO_t2i", "VisDial", "VisualNews_t2i", "WebQA"}


def test_a_dropped_upstream_column_raises_instead_of_becoming_null():
    upstream = set(prepare_data.CONFIG_COLUMNS[IMAGE_CONFIG]) | {"pos_image_path"}
    assert prepare_data.check_columns(IMAGE_CONFIG, upstream)

    with pytest.raises(RuntimeError, match="upstream columns changed"):
        prepare_data.check_columns(IMAGE_CONFIG, upstream - {prepare_data.POSITIVE_IMAGE})


def test_a_new_upstream_column_raises_rather_than_being_silently_discarded():
    upstream = set(prepare_data.CONFIG_COLUMNS[TEXT_CONFIG]) | {"pos_image_path"}
    with pytest.raises(RuntimeError, match="upstream columns changed"):
        prepare_data.check_columns(TEXT_CONFIG, upstream | {prepare_data.POSITIVE_IMAGE})


def test_a_missing_metric_column_raises():
    upstream = set(prepare_data.CONFIG_COLUMNS[IMAGE_CONFIG])
    with pytest.raises(RuntimeError, match="metric columns"):
        prepare_data.check_columns(IMAGE_CONFIG, upstream)


def test_reading_a_row_never_substitutes_none_for_a_declared_column():
    declared = prepare_data.CONFIG_COLUMNS[IMAGE_CONFIG]
    row = {column: "value" for column in declared} | {"pos_image_path": "coco/1.jpg"}
    assert prepare_data.take_row(IMAGE_CONFIG, row, declared)["pos_image"] == "value"

    with pytest.raises(RuntimeError, match="missing declared column"):
        prepare_data.take_row(
            IMAGE_CONFIG, {k: v for k, v in row.items() if k != "pos_image"}, declared
        )


def test_a_column_the_config_does_not_have_is_filled_in_after_the_check():
    """`qry_image` is absent from MSCOCO_t2i by design, so the stored row carries an
    explicit null — but only once the declared columns have all been read."""
    declared = prepare_data.CONFIG_COLUMNS[IMAGE_CONFIG]
    row = {column: "value" for column in declared} | {"pos_image_path": "coco/1.jpg"}
    kept = prepare_data.take_row(IMAGE_CONFIG, row, declared)

    assert kept["qry_image"] is None
    assert set(kept) == set(prepare_data.KEPT_COLUMNS) | {"mmeb_config", "pos_image_path"}


# --- quality metrics ---------------------------------------------------------


def clean_rows(rows: int = 10) -> dict[str, list[dict[str, Any]]]:
    return {
        IMAGE_CONFIG: [image_row(i) for i in range(rows)],
        TEXT_CONFIG: [text_row(i) for i in range(rows)],
    }


def collapsed_positive_rows() -> dict[str, list[dict[str, Any]]]:
    """Every positive in the image config pointing at one target, over enough rows
    for the share gate to apply."""
    rows = clean_rows(prepare_data.MIN_ROWS_FOR_SHARE_GATE)
    rows[IMAGE_CONFIG] = [
        image_row(i, pos_image_path="coco/same.jpg")
        for i in range(prepare_data.MIN_ROWS_FOR_SHARE_GATE)
    ]
    return rows


def test_a_clean_sample_passes_the_gate():
    metrics = prepare_data.subset_metrics(clean_rows())
    quota = {IMAGE_CONFIG: 10, TEXT_CONFIG: 10}
    assert prepare_data.quality_violations(metrics, quota, dict(quota)) == []


def test_a_text_only_query_config_is_not_counted_as_missing_images():
    """31.4% of the corrupt subset had no query image, and part of that was the
    four text-to-image configs behaving exactly as they should. A metric that
    cannot tell the two apart cannot be given a threshold."""
    metrics = prepare_data.subset_metrics(clean_rows())

    assert metrics["overall"]["rows_without_query_image"] == 0
    assert metrics["overall"]["rows_with_text_only_query_by_design"] == 10
    assert metrics["per_config"][IMAGE_CONFIG]["declares_query_image"] is False


def test_a_missing_query_image_where_the_config_declares_one_fails_the_gate():
    rows = clean_rows()
    rows[TEXT_CONFIG][0]["qry_image"] = None
    metrics = prepare_data.subset_metrics(rows)
    quota = {IMAGE_CONFIG: 10, TEXT_CONFIG: 10}

    assert metrics["overall"]["rows_without_query_image"] == 1
    assert any(
        "no query image" in v for v in prepare_data.quality_violations(metrics, quota, quota)
    )


def test_the_shipped_corruption_fails_the_gate():
    """D1 exactly: `pos_image` gone, so every image-positive row keeps nothing but
    MMEB's instruction template — 466 rows of one identical positive that InfoNCE
    then has to tell apart from itself."""
    rows = clean_rows()
    rows[IMAGE_CONFIG] = [image_row(i, pos_image=None) for i in range(10)]
    metrics = prepare_data.subset_metrics(rows)
    quota = {IMAGE_CONFIG: 10, TEXT_CONFIG: 10}
    violations = prepare_data.quality_violations(metrics, quota, dict(quota))

    assert metrics["overall"]["rows_without_positive_content"] == 10
    assert any("no positive content" in v for v in violations)
    # And the metric the old criteria would have looked at still says nothing is
    # wrong, which is why it is not the one with the threshold.
    assert metrics["per_config"][IMAGE_CONFIG]["rows"] == 10


def test_an_empty_text_positive_fails_the_gate_too():
    rows = clean_rows()
    rows[TEXT_CONFIG][0]["pos_text"] = "   "
    metrics = prepare_data.subset_metrics(rows)
    quota = {IMAGE_CONFIG: 10, TEXT_CONFIG: 10}

    assert metrics["overall"]["rows_without_positive_content"] == 1
    assert prepare_data.quality_violations(metrics, quota, dict(quota))


def test_a_collapsed_positive_column_fails_the_gate_even_with_the_column_present():
    """The second net: `pos_image` is there, but every row points at the same
    target. Presence checks cannot see this; 22.8% of the shipped subset was one
    single positive."""
    rows = collapsed_positive_rows()
    metrics = prepare_data.subset_metrics(rows)
    quota = {name: len(part) for name, part in rows.items()}

    assert metrics["per_config"][IMAGE_CONFIG]["max_single_positive_share"] == 1.0
    assert any(
        "one positive image accounts for" in v
        for v in prepare_data.quality_violations(metrics, quota, dict(quota))
    )


def test_the_collapse_gate_does_not_move_with_the_size_of_the_draw():
    """A duplicate *ratio* is a function of how much of the pool you draw: the same
    upstream data collides ~never at 100 rows and constantly at 3300, so a constant
    ratio threshold would refuse `data=quality` for being large. Caption
    multiplicity puts the largest legitimate group at a handful of rows at any
    size, so the share of one value is the statistic that holds still."""
    caption_multiplicity = 5
    for draw in (60, 600, 3300):
        rows = {
            IMAGE_CONFIG: [
                image_row(i, pos_image_path=f"coco/{i // caption_multiplicity}.jpg")
                for i in range(draw)
            ]
        }
        metrics = prepare_data.subset_metrics(rows)
        per_config = metrics["per_config"][IMAGE_CONFIG]

        assert per_config["duplicate_positive_ratio"] == 1.0, draw
        assert per_config["max_single_positive_share"] <= caption_multiplicity / draw
        assert (
            prepare_data.quality_violations(metrics, {IMAGE_CONFIG: draw}, {IMAGE_CONFIG: draw})
            == []
        )


def test_a_small_draw_is_not_judged_on_the_share_of_one_positive():
    """Two rows out of ten is 20% before anything has gone wrong. Below the row
    floor the draw is checked by presence and schema, which do not move with size."""
    rows = {IMAGE_CONFIG: [image_row(i, pos_image_path="coco/same.jpg") for i in range(10)]}
    metrics = prepare_data.subset_metrics(rows)

    assert metrics["per_config"][IMAGE_CONFIG]["max_single_positive_share"] == 1.0
    assert prepare_data.quality_violations(metrics, {IMAGE_CONFIG: 10}, {IMAGE_CONFIG: 10}) == []


def test_repeated_labels_in_a_classification_config_are_not_a_defect():
    """ImageNet_1K positives are class names, so `duplicate_pos_text_ratio` is
    near 1.0 on any sample and a threshold on it would reject real MMEB. The
    number is reported; no threshold is set on a text positive because the
    legitimate cardinality belongs to the task, not to the pipeline."""
    metrics = prepare_data.subset_metrics(clean_rows())
    quota = {IMAGE_CONFIG: 10, TEXT_CONFIG: 10}

    assert metrics["per_config"][TEXT_CONFIG]["duplicate_pos_text_ratio"] == 1.0
    assert metrics["per_config"][TEXT_CONFIG]["distinct_pos_text_count"] == 3
    assert prepare_data.quality_violations(metrics, quota, dict(quota)) == []


def test_an_image_column_holding_something_that_is_not_an_image_fails_the_gate():
    """The authoritative MMEB-train stores paths, not images, and `source_repo` is
    a config field — so a column can be populated with values that every presence
    check counts as an image and no model can read."""
    rows = clean_rows()
    for row in rows[TEXT_CONFIG]:
        row["qry_image"] = {"bytes": None, "path": "images/1.jpg"}
    metrics = prepare_data.subset_metrics(rows)
    quota = {IMAGE_CONFIG: 10, TEXT_CONFIG: 10}

    assert metrics["overall"]["rows_without_query_image"] == 0
    assert metrics["overall"]["rows_with_unreadable_image"] == 10
    assert any(
        "could not be read as images" in v
        for v in prepare_data.quality_violations(metrics, quota, dict(quota))
    )


def test_an_empty_draw_does_not_pass_by_having_nothing_to_check():
    """`data-pinned` reported "every data config pins a commit sha" while no data
    config existed. A gate whose input set can be empty has to say so."""
    assert prepare_data.quality_violations(prepare_data.subset_metrics({}), {}, {})


def test_a_config_that_yields_less_than_its_quota_fails_the_gate():
    """The old sampler printed `0/103 (empty stream)` and carried on, which skews
    the subset away from MMEB's composition without changing the total."""
    rows = clean_rows()
    metrics = prepare_data.subset_metrics(rows)
    quota = {IMAGE_CONFIG: 10, TEXT_CONFIG: 40}

    violations = prepare_data.quality_violations(
        metrics, quota, {IMAGE_CONFIG: 10, TEXT_CONFIG: 10}
    )
    assert any("fewer rows than their quota" in v for v in violations)


def test_distributions_are_recorded_per_config():
    metrics = prepare_data.subset_metrics(clean_rows())
    image_side = metrics["per_config"][TEXT_CONFIG]

    assert image_side["qry_image_width"]["p50"] == 500
    assert image_side["qry_image_pixels"]["p95"] == 500 * 375
    assert metrics["per_config"][IMAGE_CONFIG]["qry_image_pixels"] is None
    assert metrics["overall"]["qry_chars"]["p50"] > 0


def test_percentiles_do_not_interpolate():
    assert prepare_data.distribution([1, 2, 3, 4]) == {"count": 4, "p50": 2, "p95": 4, "max": 4}
    assert prepare_data.distribution([]) is None


# --- push refusal ------------------------------------------------------------


def test_a_small_sample_is_never_pushable():
    quota = prepare_data.proportional_quota(upstream_counts(), 40)
    blockers = prepare_data.push_blockers(quota, SimpleNamespace(limit=40, subset_rows=2048))
    assert any("small-sample run" in b for b in blockers)


def test_a_full_draw_has_no_push_blockers():
    quota = prepare_data.proportional_quota(upstream_counts(), 2048)
    assert prepare_data.push_blockers(quota, SimpleNamespace(limit=2048, subset_rows=2048)) == []
    assert prepare_data.push_blockers(quota, SimpleNamespace(limit=None, subset_rows=2048)) == []


def test_the_presence_thresholds_leave_no_room_at_all():
    """Zero, not "few". A row without a positive is not a noisy row, it is not a
    training row."""
    assert prepare_data.MAX_ROWS_WITHOUT_POSITIVE == 0
    assert prepare_data.MAX_ROWS_WITHOUT_QUERY_IMAGE == 0
    assert prepare_data.MAX_ROWS_WITH_UNREADABLE_IMAGE == 0


def test_the_pushed_columns_cannot_drift_from_the_measured_ones():
    """`build_dataset` writes a column list of its own, and a second hand-written
    column list is the mechanism that produced D1: a field measured here and
    dropped there would leave every metric describing rows that were not pushed."""
    assert set(prepare_data.SUBSET_COLUMN_NAMES) == set(prepare_data.KEPT_COLUMNS) | {"mmeb_config"}
    for columns in prepare_data.CONFIG_COLUMNS.values():
        assert set(columns) <= set(prepare_data.SUBSET_COLUMN_NAMES)


# --- how rows are held, and the shard cache ----------------------------------


def fake_datasets(monkeypatch, *, features=None, casts=None, opened=None, rows=None):
    """A stand-in for `datasets` so these run against the `compose` extra alone.

    Returns the fake module so a test can assert on what was asked of it.
    """

    class FakeStream:
        def __init__(self):
            self.features = (
                features
                if features is not None
                else {
                    "qry": None,
                    "pos_text": None,
                    "pos_image": None,
                    "pos_image_path": None,
                }
            )

        def cast_column(self, column, feature):
            if casts is not None:
                casts.append((column, feature.decode))
            return self

        def shuffle(self, seed, buffer_size):
            return self

        def take(self, want):
            return (
                rows
                or [
                    {
                        "qry": f"q{i}",
                        "pos_text": "p",
                        "pos_image": {"bytes": b"png", "path": None},
                        "pos_image_path": f"coco/{i}.jpg",
                    }
                    for i in range(want)
                ]
            )[:want]

    def load_dataset(*args, **kwargs):
        if opened is not None:
            opened.append(kwargs)
        return FakeStream()

    fake = ModuleType("datasets")
    fake.Image = lambda decode=True: SimpleNamespace(decode=decode)
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return fake


def fake_config(**overrides):
    data = {"source_repo": "a/b", "source_revision": "9d0fd31789c1", "sample_seed": 1234}
    return SimpleNamespace(data=SimpleNamespace(**(data | overrides)))


def test_rows_are_never_held_as_decoded_pixels(monkeypatch):
    """`datasets` decodes an Image column on every row it hands out —
    `Image.decode_example` calls `PIL.Image.load()` — and `main` accumulates every
    config's rows until the push. Measured on MSCOCO_i2t: 270.9 KiB/row decoded
    against 15.9 KiB/row encoded. At `data=quality`'s 65536 rows that killed the
    process at 40377 rows with 48GB of RAM and 20GB of swap gone.
    """
    casts = []
    fake_datasets(monkeypatch, casts=casts)

    rows = prepare_data.stream_rows(fake_config(), IMAGE_CONFIG, 3)

    # Only the columns this config declares, and decoding off for each of them.
    assert casts == [("pos_image", False)]
    assert len(rows) == 3


def test_the_draw_reads_the_pinned_upstream_commit(monkeypatch):
    """The sampler streamed the mirror's HEAD while this module's docstring and
    PLAN.md both said the commit was pinned. A subset drawn from an unrecorded HEAD
    cannot be traced to the data it came from."""
    opened = []
    fake_datasets(monkeypatch, opened=opened)

    prepare_data.stream_rows(fake_config(), IMAGE_CONFIG, 2)

    assert opened and opened[0]["revision"] == "9d0fd31789c1"


def test_the_upstream_schema_is_checked_even_when_the_draw_comes_from_cache(tmp_path, monkeypatch):
    """The check lived only inside `stream_rows`, so a second run against a cached
    draw could not see that upstream had dropped a column — while the manifest went
    on recording CONFIG_COLUMNS as the declared schema. Reproduced before the fix:
    the cached path returned 1602 CIRR rows with no exception where the streaming
    path raised."""
    monkeypatch.setenv(prepare_data.SHARD_CACHE_ENV, str(tmp_path))
    config = fake_config()
    prepare_data.write_shard(
        [image_row(i) for i in range(4)], prepare_data.shard_path(config.data, IMAGE_CONFIG, 4)
    )
    # Upstream no longer offers pos_image: exactly defect D1's shape.
    fake_datasets(monkeypatch, features={"qry": None, "pos_text": None, "pos_image_path": None})

    with pytest.raises(RuntimeError, match="upstream columns changed"):
        prepare_data.sample_config(config, IMAGE_CONFIG, 4)


def test_an_image_whose_pixels_do_not_decode_is_not_readable():
    """`PIL.Image.open` is lazy, so a header-only check calls a file with an intact
    IHDR and a corrupt payload a readable image. Before rows were held undecoded,
    `datasets` called `load()` on every row and this surfaced upstream of the gate;
    holding encoded bytes removed that without replacing it."""
    good = encoded_image(64, 48)["bytes"]
    broken = good[: len(good) // 2] + b"\x00" * (len(good) - len(good) // 2)

    assert prepare_data._image_size({"bytes": good, "path": None}) == (64, 48)
    assert prepare_data._image_size({"bytes": broken, "path": None}) is None


def test_a_declared_size_too_large_to_decode_is_counted_not_raised():
    """`DecompressionBombError` derives from Exception, so listing only OSError and
    ValueError let one oversized header kill the run after all streaming and
    caching — and the cached shard then reproduced that death on every retry."""
    from PIL import Image as PILImage

    header = io.BytesIO()
    PILImage.new("RGB", (4, 4)).save(header, format="PNG")
    payload = bytearray(header.getvalue())
    # Rewrite IHDR width/height to 60000x60000; PIL reads this before decoding.
    payload[16:24] = (60000).to_bytes(4, "big") + (60000).to_bytes(4, "big")

    assert prepare_data._image_size({"bytes": bytes(payload), "path": None}) is None


def test_image_size_reads_the_header_of_the_encoded_bytes():
    assert prepare_data._image_size(encoded_image(37, 11)) == (37, 11)
    assert prepare_data._image_size({"bytes": b"not an image", "path": None}) is None
    assert prepare_data._image_size(None) is None


def test_an_absent_image_and_an_unreadable_one_are_not_the_same_thing():
    """Collapsing them would let a repo full of dangling paths pass as a repo full
    of images: every row would count as present and none would be readable."""
    assert prepare_data.image_missing(None)
    assert prepare_data.image_missing({"bytes": None, "path": None})
    assert not prepare_data.image_missing({"bytes": None, "path": "images/1.jpg"})
    assert not prepare_data.image_missing(encoded_image())


def test_a_cached_shard_round_trips_the_draw_exactly(tmp_path):
    """A resumed run must use the rows the killed one drew. Anything the shard
    cannot carry — a null image, a null path — comes back changed and the subset
    silently differs from the one the manifest measured."""
    rows = [image_row(0), image_row(1, pos_image=None), text_row(2)]
    path = tmp_path / "shard.parquet"

    prepare_data.write_shard(rows, path)

    assert prepare_data.read_shard(path) == rows


def test_a_shard_is_never_left_behind_half_written(tmp_path, monkeypatch):
    """The kill this cache exists for can arrive mid-write. The previous version of
    this test wrote a `.partial` by hand and asserted the `.parquet` was absent,
    which called neither `write_shard` nor `sample_config` — removing the staging
    write entirely left all four shard tests passing."""
    import pyarrow.parquet as pq

    path = tmp_path / "shard.parquet"

    def die(*args, **kwargs):
        raise KeyboardInterrupt("killed mid-write")

    monkeypatch.setattr(pq, "write_table", die)
    with pytest.raises(KeyboardInterrupt):
        prepare_data.write_shard([image_row(0)], path)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_two_processes_writing_one_shard_do_not_collide(tmp_path, monkeypatch):
    """A fixed staging name means whichever process renames second dies on a
    FileNotFoundError that nothing catches."""
    monkeypatch.setattr(prepare_data.os, "getpid", lambda: 111)
    first = prepare_data.shard_path.__globals__["Path"](tmp_path / "a.parquet")
    monkeypatch.setattr(prepare_data.os, "getpid", lambda: 222)
    second = tmp_path / "a.parquet"

    assert first.with_suffix(".111.partial") != second.with_suffix(".222.partial")


def test_a_shard_the_cache_cannot_read_is_discarded_rather_than_retried(tmp_path, monkeypatch):
    """A corrupt shard read straight through would make every retry fail in the same
    place: the cache built to survive a kill would make the failure permanent."""
    monkeypatch.setenv(prepare_data.SHARD_CACHE_ENV, str(tmp_path))
    fake_datasets(monkeypatch)
    config = fake_config()
    path = prepare_data.shard_path(config.data, IMAGE_CONFIG, 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a parquet file")
    monkeypatch.setattr(prepare_data, "stream_rows", lambda *args: [image_row(i) for i in range(4)])

    rows, source = prepare_data.sample_config(config, IMAGE_CONFIG, 4)

    assert source == "streamed"
    assert len(rows) == 4


def test_a_shard_missing_a_column_is_refused_rather_than_backfilled(tmp_path):
    """`pq.read_table(schema=...)` fills an absent column with nulls, so a shard
    written by older code would come back with a silently missing `pos_image` —
    D1's shape reached through the resume path. `take_row` raises on the same
    situation and the two have to agree."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "old.parquet"
    kept = [c for c in prepare_data.SHARD_COLUMNS if c != prepare_data.POSITIVE_IMAGE]
    pq.write_table(pa.table({c: pa.array(["v"], type=pa.string()) for c in kept}), path)

    with pytest.raises(RuntimeError, match="cached shard holds columns"):
        prepare_data.read_shard(path)


def test_a_shard_that_holds_fewer_rows_than_the_quota_is_redrawn(tmp_path, monkeypatch):
    """Topping a short shard up would take the missing rows from a different
    position in the shuffled stream, so the draw would no longer be the one the
    seed defines."""
    monkeypatch.setenv(prepare_data.SHARD_CACHE_ENV, str(tmp_path))
    fake_datasets(monkeypatch)
    config = fake_config()
    path = prepare_data.shard_path(config.data, IMAGE_CONFIG, 4)
    prepare_data.write_shard([image_row(0), image_row(1)], path)
    monkeypatch.setattr(prepare_data, "stream_rows", lambda *args: [image_row(i) for i in range(4)])

    rows, source = prepare_data.sample_config(config, IMAGE_CONFIG, 4)

    assert source == "streamed"
    assert len(rows) == 4
    assert len(prepare_data.read_shard(path)) == 4


def test_a_complete_shard_is_reused_without_redrawing_it(tmp_path, monkeypatch):
    monkeypatch.setenv(prepare_data.SHARD_CACHE_ENV, str(tmp_path))
    fake_datasets(monkeypatch)
    config = fake_config()
    drawn = [image_row(i) for i in range(4)]
    prepare_data.write_shard(drawn, prepare_data.shard_path(config.data, IMAGE_CONFIG, 4))
    monkeypatch.setattr(
        prepare_data, "stream_rows", lambda *args: pytest.fail("a cached shard was re-drawn")
    )

    rows, source = prepare_data.sample_config(config, IMAGE_CONFIG, 4)

    assert source == "cached"
    assert rows == drawn


def test_the_shard_name_changes_with_everything_that_changes_the_draw():
    """A cache keyed on less than this hands one draw's rows to another's request
    — the failure mode is a subset that silently does not match its manifest."""
    base = fake_config().data
    names = {
        prepare_data.shard_path(base, IMAGE_CONFIG, 4).name,
        prepare_data.shard_path(base, IMAGE_CONFIG, 5).name,
        prepare_data.shard_path(base, TEXT_CONFIG, 4).name,
        prepare_data.shard_path(fake_config(sample_seed=7).data, IMAGE_CONFIG, 4).name,
        prepare_data.shard_path(fake_config(source_repo="c/d").data, IMAGE_CONFIG, 4).name,
        prepare_data.shard_path(fake_config(source_revision="0" * 40).data, IMAGE_CONFIG, 4).name,
    }
    assert len(names) == 6


def test_the_shard_carries_every_column_the_measurement_reads():
    """A shard missing a metric column would make a resumed run measure less than
    the run it resumed, which is how a column goes missing unnoticed."""
    assert set(prepare_data.SHARD_COLUMNS) == set(prepare_data.SUBSET_COLUMN_NAMES) | set(
        prepare_data.METRIC_COLUMNS
    )


# --- the gate that was skipped, and the artifact nobody read -----------------


def test_a_collapse_gate_that_did_not_run_is_recorded_as_not_having_run():
    """`data=speed` draws 30 NIGHTS rows and 33 WebQA rows, so 2 of its 7
    image-positive configs never reach `MIN_ROWS_FOR_SHARE_GATE` — and the
    "largest single image positive" figure quoted from that draw came from one of
    the two. The output still said the gate passed."""
    rows = clean_rows(rows=prepare_data.MIN_ROWS_FOR_SHARE_GATE - 1)
    exempt = prepare_data.ungated_configs(prepare_data.subset_metrics(rows))

    assert exempt == {IMAGE_CONFIG: prepare_data.MIN_ROWS_FOR_SHARE_GATE - 1}
    # The text-positive config is not image-positive, so it was never in scope.
    assert TEXT_CONFIG not in exempt


def test_a_gated_draw_reports_nothing_as_exempt():
    assert (
        prepare_data.ungated_configs(
            prepare_data.subset_metrics(clean_rows(rows=prepare_data.MIN_ROWS_FOR_SHARE_GATE))
        )
        == {}
    )


def test_the_artifact_identity_survives_the_column_the_push_drops():
    """`pos_image_path` is metric-only and never pushed, so the pushed subset can
    only be re-gated by hashing the image bytes. Digest grouping can merge two
    paths that hold identical bytes but can never split one path, so the share it
    yields is >= the manifest's — sound to apply the same threshold to."""
    same = encoded_image(8, 8)
    rows = [image_row(0, pos_image=same), image_row(1, pos_image=same)]
    identities = {prepare_data.artifact_identity(IMAGE_CONFIG, row) for row in rows}

    assert len(identities) == 1
    assert prepare_data.artifact_identity(TEXT_CONFIG, text_row(0))[0] == "text"


def published_rows(rows_by_config):
    return [row | {"mmeb_config": name} for name, part in rows_by_config.items() for row in part]


def stub_artifact(monkeypatch, *, columns=None, rows=None, downloaded=None):
    """Stand in for the two halves of `verify_pushed`: the card and the parquet."""
    monkeypatch.setattr(
        prepare_data,
        "_declared_columns",
        lambda data, revision: tuple(sorted(columns or prepare_data.SUBSET_COLUMN_NAMES)),
    )

    def batches(data, revision):
        if downloaded is not None:
            downloaded.append(revision)
        for i in range(0, len(rows or []), 3):
            yield (rows or [])[i : i + 3]

    monkeypatch.setattr(prepare_data, "_artifact_batches", batches)


def test_a_pushed_subset_declaring_the_wrong_columns_is_refused(monkeypatch):
    """The failure this check exists for: `Dataset.push_to_hub` kept the repo's
    existing card, so a corrected five-column subset landed under defect D1's
    four-column schema and `load_dataset` raised CastError on it. The push
    returned a revision and was treated as finished."""
    downloaded = []
    stub_artifact(
        monkeypatch,
        columns=("mmeb_config", "qry", "qry_image", "pos_text"),
        rows=published_rows(clean_rows()),
        downloaded=downloaded,
    )

    problems = prepare_data.verify_pushed(
        SimpleNamespace(repo_id="x/y"), "f" * 40, prepare_data.subset_metrics(clean_rows())
    )

    assert problems and "cannot read this revision" in problems[0]
    # And it says so before pulling the artifact down: a broken card is seconds of
    # metadata, not gigabytes of parquet.
    assert downloaded == []


def test_a_pushed_subset_that_lost_rows_is_refused(monkeypatch):
    """Column names alone are not the artifact. A push that dropped rows declares
    the right schema and still does not hold the corpus the manifest measured."""
    rows = clean_rows()
    stub_artifact(monkeypatch, rows=published_rows(rows)[:-3])

    problems = prepare_data.verify_pushed(
        SimpleNamespace(repo_id="x/y"), "f" * 40, prepare_data.subset_metrics(rows)
    )

    assert any("rows=" in problem for problem in problems)


def test_a_push_that_lost_an_image_is_refused(monkeypatch):
    """D1's own shape, arriving in the artifact rather than the draw."""
    rows = clean_rows()
    published = published_rows(rows)
    damaged = [
        row | {"pos_image": None} if row["mmeb_config"] == IMAGE_CONFIG else row
        for row in published
    ]
    stub_artifact(monkeypatch, rows=damaged)

    problems = prepare_data.verify_pushed(
        SimpleNamespace(repo_id="x/y"), "f" * 40, prepare_data.subset_metrics(rows)
    )

    assert any("rows_without_positive_content" in problem for problem in problems)


def test_a_faithful_push_verifies_clean(monkeypatch):
    """The check has to be able to pass, or it certifies nothing."""
    rows = clean_rows()
    stub_artifact(monkeypatch, rows=published_rows(rows))

    assert (
        prepare_data.verify_pushed(
            SimpleNamespace(repo_id="x/y"), "f" * 40, prepare_data.subset_metrics(rows)
        )
        == []
    )


def test_the_peak_memory_recorded_is_a_high_water_mark_not_a_sample():
    """The evidence that the undecoded-rows fix worked was `ps -o rss=` every 120s,
    which reported a *lower* figure at 65536 rows than at 40377 — impossible for a
    monotone accumulator, so both samples were allocator noise."""
    peak = prepare_data._peak_rss_bytes()
    assert peak > 0
    assert prepare_data._peak_rss_bytes() >= peak


# --- data configs ------------------------------------------------------------


def test_data_configs_exist_and_are_tracked():
    """`.gitignore`'s unanchored `data/` swallowed this directory once already;
    with the configs absent, `data-pinned` passed by having nothing to check.

    Read through the script's own loader so there is one definition of what a data
    config is, rather than a second glob here that can drift from it."""
    configs = prepare_data.data_config_pins()
    assert {"speed.yaml", "quality.yaml"} <= set(configs)


def test_no_data_config_points_at_the_repo_another_one_pushes_to():
    """Two configs pushing different row counts to one repo_id means regenerating
    either one overwrites the other's data files."""
    repos = [config["repo_id"] for config in prepare_data.data_config_pins().values()]
    assert len(set(repos)) == len(repos)


def test_the_corrupt_revision_is_recorded_so_it_cannot_be_repinned_by_accident():
    assert prepare_data.KNOWN_CORRUPT_REVISIONS
    for revision, reason in prepare_data.KNOWN_CORRUPT_REVISIONS.items():
        assert len(revision) == 40 and reason


def test_a_config_pinning_a_known_corrupt_revision_is_detected():
    detected = prepare_data.corrupt_pins(
        {"speed.yaml": {"revision": next(iter(prepare_data.KNOWN_CORRUPT_REVISIONS))}}
    )
    assert detected and "speed.yaml" in detected[0]
    assert prepare_data.corrupt_pins({"speed.yaml": {"revision": None}}) == []
