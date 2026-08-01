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
import sys
from types import SimpleNamespace
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


class FakeImage:
    """Stands in for a decoded PIL image. Only `.size` is read for metrics."""

    def __init__(self, width: int = 640, height: int = 480):
        self.size = (width, height)


def image_row(index: int, *, pos_image: Any | None = ..., **overrides: Any) -> dict[str, Any]:
    row = {
        "mmeb_config": IMAGE_CONFIG,
        "qry": f"find the image of subject {index}",
        "qry_image": None,
        "pos_text": "<|image_1|>\nRepresent the given image.\n",
        "pos_image": FakeImage() if pos_image is ... else pos_image,
        "pos_image_path": f"coco/{index}.jpg",
    }
    return row | overrides


def text_row(index: int, **overrides: Any) -> dict[str, Any]:
    row = {
        "mmeb_config": TEXT_CONFIG,
        "qry": "<|image_1|>\nRepresent the given image for classification.\n",
        "qry_image": FakeImage(500, 375),
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
