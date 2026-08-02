"""The wave gate itself. Six lanes will trust whatever this says.

Written after a review found that the audit certified the one defect it was
built to catch: the config-consumed check matched bare identifiers, so every
ablation axis counted as consumed by any file that happened to use the word
`name`, and `--update-baseline` replaced each entry's resolving wave with a
placeholder.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import audit_plan  # noqa: E402
from audit_plan import (  # noqa: E402
    Result,
    _demanded_imports,
    _normalise,
    _reads_dotted,
    anchor_holds,
    classify,
    merge_baseline,
    missing_plan_files,
    model_spec_problems,
    prebuilt_wheel_problems,
    undocumented_files,
    verdict_ledger_problems,
)


def failing(*names, count=None):
    return [Result(name, False, "", count=count) for name in names]


def passing(*names):
    return [Result(name, True, "") for name in names]


def accepted(**notes):
    """A baseline in the stored shape: a note plus the size it was accepted at."""
    return {name: {"note": note, "count": count} for name, (note, count) in notes.items()}


def test_a_bare_identifier_is_not_a_consumer():
    """`name` appears in eighteen files here. Counting those as reads is how
    eight unwired axes passed the check written to find them."""
    code = "def f(name):\n    return Check(name=name, mode='x')\n"

    assert not _reads_dotted(code, "attn", "name")
    assert not _reads_dotted(code, "compile", "mode")


def test_a_config_access_is_a_consumer():
    assert _reads_dotted("model = load(config.attn.name)", "attn", "name")
    assert _reads_dotted('x = cfg["attn"]["name"]', "attn", "name")
    assert _reads_dotted('getattr(config.attn, "name")', "attn", "name")


def test_a_similar_name_is_not_the_same_knob():
    assert not _reads_dotted("config.attn.names", "attn", "name")
    assert not _reads_dotted("config.kernel.name", "attn", "name")


# A reviewer defeated the previous version of this check three ways, all of them
# by writing the knob's name somewhere the interpreter never reads it. The
# defence at the time was the baseline's recorded count, which flags a shrink —
# so a lane editing its own baseline line removed it. These fix the check itself.
def test_a_string_that_names_a_knob_is_not_a_read():
    """Stripping prose removed docstrings, which are `ast.Expr`. Assigning the
    same string to a name makes it an `ast.Assign`, and it survived."""
    assert not _reads_dotted('_NOTE = "see config.data.subset_rows"', "data", "subset_rows")
    assert not _reads_dotted('"""Reads config.attn.name."""', "attn", "name")
    assert not _reads_dotted("x = 1  # config.attn.name", "attn", "name")


def test_a_branch_that_cannot_be_entered_is_not_a_read():
    assert not _reads_dotted("if False:\n    _ = config.data.subset_rows\n", "data", "subset_rows")
    assert _reads_dotted("if True:\n    _ = config.data.subset_rows\n", "data", "subset_rows")


def test_another_objects_attributes_are_not_the_config():
    """The attribute form was unanchored while the `getattr` form was anchored,
    so any object with the right-shaped attribute chain satisfied it."""
    code = "def _unrelated(anything):\n    return anything.data.subset_rows\n"

    assert not _reads_dotted(code, "data", "subset_rows")
    assert not _reads_dotted('payload["data"]["subset_rows"]', "data", "subset_rows")
    assert not _reads_dotted('getattr(model, "name")', "model", "name")


def test_a_hash_inside_a_string_does_not_hide_the_rest_of_the_line():
    """Comments were stripped by cutting each line at its first `#`, which also
    cut lines whose `#` was inside a literal. Fail-closed rather than fail-open,
    so it corrupted nothing — it just reported a read knob as unread."""
    assert _reads_dotted('log("#start", config.attn.name)', "attn", "name")


def test_updating_the_baseline_keeps_the_schedule():
    """Each entry names the wave that resolves it. That is what makes the
    baseline a schedule rather than an excuse, and overwriting it with a
    placeholder erases the plan for every lane at once."""
    baseline = accepted(
        **{"axis-packages": ("Wave 2 (F)", 3), "config-consumed": ("Wave 2 (D)", 13)}
    )

    merged = merge_baseline(baseline, failing("axis-packages", "config-consumed", count=3))

    assert [entry["note"] for entry in merged.values()] == ["Wave 2 (F)", "Wave 2 (D)"]


def test_a_newly_failing_check_enters_as_unscheduled():
    merged = merge_baseline(
        accepted(**{"axis-packages": ("Wave 2 (F)", 4)}),
        failing("axis-packages", "data-pinned", count=2),
    )

    assert merged == {
        "axis-packages": {"note": "Wave 2 (F)", "count": 2},
        "data-pinned": {"note": "unscheduled", "count": 2},
    }


def test_a_check_that_now_passes_leaves_the_baseline():
    merged = merge_baseline(accepted(**{"data-pinned": ("Wave 1 (A)", 1)}), passing("data-pinned"))

    assert merged == {}


def test_a_partial_run_cannot_write_the_baseline(tmp_path, monkeypatch):
    """`--only x --update-baseline` used to delete every other entry, after which
    the accepted failures were gone and the next full run had no record of them."""
    baseline = tmp_path / "audit-baseline.json"
    baseline.write_text(json.dumps({"data-pinned": "Wave 1 (A)", "doc-commands": "Wave 1 (E)"}))
    monkeypatch.setattr(audit_plan, "BASELINE", baseline)

    assert audit_plan.main(["--only", "data-pinned", "--update-baseline"]) == 1
    assert json.loads(baseline.read_text()) == {
        "data-pinned": "Wave 1 (A)",
        "doc-commands": "Wave 1 (E)",
    }


def test_selecting_no_checks_is_an_error_not_a_pass():
    assert audit_plan.main(["--only", "data-pinned", "--skip", "data-pinned"]) == 1


def test_a_new_failure_blocks_and_a_known_one_does_not():
    regressions, fixed, grew, shrank, unreadable = classify(
        failing("a", "b"), accepted(a=("Wave 1", None))
    )

    assert regressions == ["b"]
    assert (fixed, grew, shrank, unreadable) == ([], [], [], [])


def test_a_baseline_entry_that_starts_passing_blocks():
    """A stale baseline grants amnesty to whatever breaks there next."""
    regressions, fixed, grew, shrank, unreadable = classify(passing("a"), accepted(a=("Wave 1", 3)))

    assert (regressions, fixed, grew, shrank, unreadable) == ([], ["a"], [], [], [])


def test_an_accepted_failure_that_got_worse_blocks():
    """Membership alone made the gate blind to everything a wave did inside a check
    that was already failing: deleting Wave 2's entire capture layer left
    `7/11 passing, 0 new failure(s), 0 newly fixed` identical to the byte, because
    `axis-wired` failed before and after."""
    regressions, fixed, grew, shrank, unreadable = classify(
        failing("a", count=22), accepted(a=("Wave 2 (D)", 2))
    )

    assert grew == ["a 2->22"]
    assert (regressions, fixed, shrank, unreadable) == ([], [], [], [])


def test_an_accepted_failure_that_shrank_blocks_as_a_stale_baseline():
    """Same reason a newly passing check blocks: an entry accepting 12 problems
    when 2 remain grants amnesty for 10 that no longer exist."""
    regressions, fixed, grew, shrank, unreadable = classify(
        failing("a", count=2), accepted(a=("Wave 2 (D)", 12))
    )

    assert shrank == ["a 12->2"]
    assert (regressions, fixed, grew, unreadable) == ([], [], [], [])


def test_a_check_that_does_not_count_is_not_compared_by_size():
    """`assert-called` either finds the entry point or does not; inventing a count
    for it would make the comparison fire on nothing."""
    assert classify(failing("a"), accepted(a=("Wave 3 (G)", None)))[2:] == ([], [], [])


def test_a_baseline_count_that_is_not_a_number_is_reported_not_raised(
    tmp_path, monkeypatch, capsys
):
    """`docs/audit-baseline.json` is hand-edited between waves, and a quoted count
    there used to take the gate down: `classify` compares with `>`, so
    `"count": "3"` raised `TypeError: '>' not supported between instances of 'int'
    and 'str'` and the run printed no result for any check — including the twelve
    that had nothing to do with the malformed entry. A gate that dies on its own
    input is the failure mode this ledger exists for.

    Both halves are asserted, because the pure function returning a list is worth
    nothing if `main` still cannot get to the end: `True` is caught as well as
    `"3"`, since `bool` is an `int` and `True == 1` would have silently accepted a
    count of one.
    """
    assert classify(failing("a", count=2), accepted(a=("Wave 2 (D)", "3")))[4] == ["a count='3'"]
    assert classify(failing("a", count=1), accepted(a=("Wave 2 (D)", True)))[4] == ["a count=True"]

    baseline = tmp_path / "audit-baseline.json"
    baseline.write_text(json.dumps({"fake": {"note": "Wave 2 (D)", "count": "3"}}))
    monkeypatch.setattr(audit_plan, "BASELINE", baseline)
    monkeypatch.setattr(
        audit_plan, "CHECKS", {"fake": lambda: Result("fake", False, "two problems", count=2)}
    )

    assert audit_plan.main([]) == 1
    printed = capsys.readouterr().out
    assert "BLOCKED: baseline entries with an uncomparable count: fake count='3'" in printed


def test_an_old_string_baseline_still_loads(tmp_path, monkeypatch):
    """Entries written before counts existed must keep working, with the size
    comparison disabled for them rather than the whole file rejected."""
    baseline = tmp_path / "audit-baseline.json"
    baseline.write_text(json.dumps({"a": "Wave 1 (A)"}))
    monkeypatch.setattr(audit_plan, "BASELINE", baseline)

    assert audit_plan.load_baseline() == {"a": {"note": "Wave 1 (A)", "count": None}}


LAYOUT = """train-comparison/
├── PLAN.md                    # this document
├── trainbench/
│   ├── config.py              # exists
│   └── device.py              # missing
└── scripts/
    └── bench.py               # missing
"""


def test_a_declared_file_must_exist_where_the_tree_puts_it(tmp_path):
    """`config.py` exists 73 times in this repository. Matching the basename
    anywhere keeps the check green after the declared file is deleted — and red
    for a file that exists under a directory the check never looked at."""
    (tmp_path / "trainbench").mkdir()
    (tmp_path / "trainbench" / "config.py").touch()
    (tmp_path / "PLAN.md").touch()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "device.py").touch()

    assert missing_plan_files(LAYOUT, tmp_path) == [
        "trainbench/device.py",
        "scripts/",
        "scripts/bench.py",
    ]


def test_a_declared_directory_must_exist_too(tmp_path):
    """Only entries with a file extension were checkable, so a directory entry
    was used as a parent path and never verified: `configs/nonexistent/` sat in
    the block and passed."""
    (tmp_path / "PLAN.md").touch()

    assert "trainbench/" in missing_plan_files(LAYOUT, tmp_path)


def test_a_directory_does_not_leak_into_its_sibling(tmp_path):
    """Depth decides the parent. Without that, `bench.py` under `scripts/` would
    be looked for under whichever directory was named last."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bench.py").touch()
    (tmp_path / "trainbench").mkdir()
    (tmp_path / "trainbench" / "config.py").touch()
    (tmp_path / "PLAN.md").touch()

    assert missing_plan_files(LAYOUT, tmp_path) == ["trainbench/device.py"]


# The other direction. `missing_plan_files` only asks whether what is written
# down exists, so the block is kept true by mentioning less — which is how
# `trainbench/metrics/`, `scripts/bench.py` and six test modules were all absent
# from it while the check passed.
OPAQUE_LAYOUT = """train-comparison/
├── PLAN.md
├── trainbench/
│   ├── config.py
│   └── probe/
└── docker/
"""


def test_a_file_the_tree_does_not_name_is_reported():
    tracked = ["PLAN.md", "trainbench/config.py", "trainbench/record.py"]

    assert undocumented_files(OPAQUE_LAYOUT, tracked) == ["trainbench/record.py"]


def test_a_directory_listed_without_children_is_documented_as_a_unit():
    """`docker/`, `envs/` and each config group are named but not enumerated.
    Holding their contents to the tree would mean listing every Dockerfile and
    every axis variant in PLAN.md, which is the noise this scope avoids."""
    tracked = ["docker/entrypoint.sh", "docker/Dockerfile.base", "trainbench/probe/native.py"]

    assert undocumented_files(OPAQUE_LAYOUT, tracked) == []


def test_an_undocumented_directory_is_reported_once_not_per_file():
    tracked = ["trainbench/metrics/__init__.py", "trainbench/metrics/mfu.py"]

    assert undocumented_files(OPAQUE_LAYOUT, tracked) == ["trainbench/metrics/"]


def test_package_markers_and_dot_entries_are_out_of_scope():
    """Stated in the check's own output, because a reader who sees this pass
    needs to know what it did not examine."""
    tracked = ["trainbench/__init__.py", ".github/workflows/build.yml", ".pre-commit-config.yaml"]

    assert undocumented_files(OPAQUE_LAYOUT, tracked) == []


# `axis-values` was vacuous for the whole dataloader axis: it called `assemble`
# with no dataset, and `axes._dataloader` returns at `if dataset is None` before
# it reaches packing or pretokenize. Gutting `PackedCollate.__call__` left the
# check's output identical to the byte. Both tests below fail if the fixture or
# the batch draw is removed.
def test_the_axis_value_fixture_is_a_dataset_the_loader_axis_accepts():
    """`_dataloader` asks two separate questions — what the dataset declares, and
    what its first row hands over. A fixture answering only the first is refused
    with `UnappliedAxis`, which would read as the axis being inapplicable."""
    from trainbench import axes

    rows = audit_plan._AxisValueRows()

    assert axes.tokenized_columns(rows) == ["input_ids"]
    assert axes.tokenized_row(rows)
    assert len(rows) == audit_plan._BATCH_ROWS
    packed = axes.PackedCollate()([rows[i] for i in range(len(rows))])
    assert packed["input_ids"].shape == (1, audit_plan._BATCH_ROWS * audit_plan._ROW_TOKENS)


def test_the_two_axis_value_fixtures_differ_only_in_carrying_an_image():
    """Applicability is not always a property of the axis alone, so the check asks
    twice. If the fixtures differed in a second thing, a value refused for that
    other reason would be misreported as depending on the data."""
    from trainbench import axes

    text_only = audit_plan._AxisValueRows()
    with_images = audit_plan._AxisValueRowsWithImages()

    assert axes.image_columns(text_only) == []
    assert axes.image_columns(with_images) == ["qry_image"]
    # Everything the loader axis reads is identical; only the image column is added.
    assert axes.tokenized_columns(text_only) == axes.tokenized_columns(with_images) == ["input_ids"]
    assert len(text_only) == len(with_images)
    for index in range(len(text_only)):
        assert text_only[index]["input_ids"].equal(with_images[index]["input_ids"])
        # A tensor, not None: `image_columns` reads None as the column being empty
        # and torch's default collate cannot stack it.
        assert with_images[index]["qry_image"] is not None


def test_a_value_applicable_to_only_one_data_shape_is_named_not_counted(monkeypatch):
    """The branch that names a data-dependent value instead of counting it.

    `loss/cached_mnrl` was the live case and is not one any more — `axes._split_rows`
    attributes `pixel_values` to rows from the per-row image counts the collate
    records, so the value now applies to both fixtures. The case is therefore made
    rather than waited for: `_gradcache_needs_splittable_data` is given back the
    refusal it used to raise, and the check has to notice that the value passes on
    one shape of data and not the other.

    Written this way rather than deleted because the branch is the whole point of
    asking twice. Without a test it would go on reporting whatever it reported when
    the next value starts depending on its data.
    """
    from trainbench import axes

    splittable = axes._gradcache_needs_splittable_data

    def refuses_images(dataset):
        splittable(dataset)
        if axes.image_columns(dataset):
            raise axes.UnappliedAxis("synthetic: this value refuses rows carrying images")

    monkeypatch.setattr(axes, "_gradcache_needs_splittable_data", refuses_images)

    result = audit_plan.CHECKS["axis-values"]()

    assert "loss/cached_mnrl" in result.detail
    assert "applicable only to data this study does not measure" in result.detail
    assert "loss 1/2" in result.detail


def test_gradcache_is_counted_applicable_on_both_data_shapes():
    """The break for the test above, and the state the check reports today.

    With the refusal really gone the two fixtures no longer separate any value, so
    a `_split_rows` that quietly went back to refusing pixels — or a check that
    stopped trying the image-carrying fixture at all — shows up as this assertion
    rather than as a number a reader would take for progress.
    """
    result = audit_plan.CHECKS["axis-values"]()

    assert "loss 2/2" not in result.detail, "groups are only listed when a value is unusable"
    assert "loss/cached_mnrl" not in result.detail
    assert "applicable only to data this study does not measure" not in result.detail


def test_the_flag_knobs_in_the_train_group_are_counted_as_axis_values(monkeypatch):
    """`train` varies by dotted override, and the check used to skip it entirely.

    `axis-values` enumerates variant files per config group and `train` is in
    `NON_AXIS_GROUPS`, so none of `train.gradient_checkpointing`'s three values was
    ever pushed through the four call sites — the axis's only real gate was
    `tests/test_axes.py`. The values come off the schema rather than a list here,
    so a knob added to one of these sections is tried without being remembered
    into anything.

    The second half is the break. Applicability alone is a weak reading: an apply
    site gutted to do nothing raises no exception, so counting would go on saying
    3/3. That is why these knobs are also read back off what was built.
    """
    from trainbench import axes

    knobs, unenumerable = audit_plan.flag_knob_values()

    assert unenumerable == []
    assert knobs["train.gradient_checkpointing"] == ["none", "full", "selective"]
    assert knobs["train.offload"] == ["none", "optimizer", "param", "both"]

    result = audit_plan.CHECKS["axis-values"]()
    # Listed only when a value is unusable, so silence here is all three counted.
    assert "train.gradient_checkpointing" not in result.detail

    monkeypatch.setattr(axes, "_gradient_checkpointing", lambda model, config: [])
    gutted = audit_plan.CHECKS["axis-values"]()

    assert not gutted.ok
    assert "train.gradient_checkpointing/full" in gutted.detail
    assert "read back 'none'" in gutted.detail


def test_axis_values_draws_a_batch_so_the_collate_actually_runs(monkeypatch):
    """Building a loader does not call its collate, and packing lives entirely in
    the collate — so passing a dataset without drawing a batch leaves the axis
    certified by code that never ran."""
    from trainbench import axes

    def gutted(self, rows):
        raise NotImplementedError("packing gutted")

    monkeypatch.setattr(axes.PackedCollate, "__call__", gutted)

    result = audit_plan.CHECKS["axis-values"]()

    assert not result.ok
    assert "dataloader/torch_packed" in result.detail
    assert "packing gutted" in result.detail


def _entry_point(directory, source):
    """`source` written where `assert-called` looks for the entry point."""
    entry = directory / "scripts" / "bench.py"
    entry.parent.mkdir(parents=True)
    entry.write_text(source)
    return entry


def test_a_harness_that_never_calls_the_loss_it_built_is_reported(tmp_path, monkeypatch):
    """`assert-called` asked only whether the five hooks were called. A loop that
    calls `assemble` and then computes the loss itself passes every one of them
    while `applied.capture` certifies `loss.name` off a `built.loss_fn` nothing
    consumed — so `loss=cached_mnrl` would be reported over ordinary in-batch
    negatives rather than crashing.

    The real entry point is mutated rather than a sketch of one written, so the
    check is aimed at the file it names. The pass half comes first: without it the
    failure below is satisfied by any file at all.
    """
    source = audit_plan.BENCH_ENTRY_POINT.read_text()
    mutated = source.replace("built.loss_fn(", "info_nce(")
    assert mutated != source, "the entry point does not call built.loss_fn at all any more"

    monkeypatch.setattr(audit_plan, "REPO", tmp_path / "real")
    monkeypatch.setattr(audit_plan, "BENCH_ENTRY_POINT", _entry_point(tmp_path / "real", source))
    assert audit_plan.CHECKS["assert-called"]().ok

    monkeypatch.setattr(audit_plan, "REPO", tmp_path / "mutated")
    monkeypatch.setattr(
        audit_plan, "BENCH_ENTRY_POINT", _entry_point(tmp_path / "mutated", mutated)
    )
    result = audit_plan.CHECKS["assert-called"]()

    assert not result.ok
    assert "computes loss outside built.loss_fn" in result.detail


def test_a_harness_that_binds_no_loss_from_the_built_one_is_reported(tmp_path, monkeypatch):
    """The other half of the same hole, and the one the hooks hide best: every
    named call is there, and the loss is another function's from the first line.
    The mutation above leaves the GradCache branch reading `built.loss_fn`, so this
    is what exercises the "never consumed at all" branch."""
    monkeypatch.setattr(audit_plan, "REPO", tmp_path)
    monkeypatch.setattr(
        audit_plan,
        "BENCH_ENTRY_POINT",
        _entry_point(
            tmp_path,
            "def measure(model, config, batch):\n"
            "    model = patch(model, config)\n"
            "    kwargs = load_kwargs(config)\n"
            "    built, applied = assemble(model, config, **kwargs)\n"
            "    with step_context(config):\n"
            "        loss = info_nce(batch.queries, batch.documents)\n"
            "        loss.backward()\n"
            "    assert_matches(applied, config)\n"
            "    return loss\n",
        ),
    )

    result = audit_plan.CHECKS["assert-called"]()

    assert not result.ok
    assert "binds no loss from built.loss_fn" in result.detail


# `doc-commands` asked whether the `uv sync` line carried `--extra compose`,
# justified as "but tests import hydra". It reported `5 documented command(s)
# install what the tests need` while `peft`, `datasets` and `transformers` were
# in an extra no documented command asks for. A rule naming one package cannot
# answer a question about all of them.
def test_the_test_imports_are_collected_from_the_tests_not_a_list():
    """A hand-written list of what the tests need is a list of what to forget.
    All three gaps were function-level imports, which is why they are collected."""
    modules = _demanded_imports()

    assert {"torch", "pytest", "hydra", "peft", "transformers"} <= set(modules)
    assert "tests/test_applied.py" in modules["peft"]
    # First-party and stdlib are not distributions and must not be demanded of a lock.
    assert not {"trainbench", "audit_plan", "json", "pathlib", "__future__"} & set(modules)


def test_a_lazy_third_party_import_in_the_package_is_demanded_of_the_documented_setup(monkeypatch):
    """The scan read `tests/` alone, and the question it answered was used as the
    answer to a wider one.

    `optim=muon` imports `pytorch-optimizer` in exactly one place — inside
    `axes._optimizer`, at call time — and no test imports it. So it appeared in no
    scan, no documented `uv sync` was ever asked for it, and `doc-commands` kept
    reporting that the documented command installs everything while a clean clone
    got the optim axis refusing itself and an ablation one row short.

    Three properties, because the middle one is where a narrower scan slips
    through: the import is found, it is found although it is lazy, and the check
    demands what the scan returns rather than a second list of its own.
    """
    modules = _demanded_imports()

    assert "trainbench/axes.py" in modules["pytorch_optimizer"]
    # Lazy: a scan that read only the import block at the top of each file — which
    # is what "collect the imports" usually means — would find nothing here.
    tree = ast.parse((REPO / "trainbench" / "axes.py").read_text())
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "pytorch_optimizer" not in top_level
    # The per-image adapters are not demanded of the root lock: unsloth is pinned
    # by `envs/unsloth` and installed in that image only.
    assert "unsloth" not in modules

    monkeypatch.setattr(audit_plan, "_locked_distributions", lambda flags: ({"hydra-core"}, None))
    monkeypatch.setattr(
        audit_plan, "_demanded_imports", lambda: {"pytorch_optimizer": {"trainbench/axes.py"}}
    )
    result = audit_plan.CHECKS["doc-commands"]()

    assert not result.ok
    assert "pytorch_optimizer" in result.detail


def test_distribution_names_are_normalised_before_comparison():
    """`uv export` prints `pytorch-optimizer`, the metadata says
    `pytorch_optimizer`, and comparing them unnormalised finds nothing."""
    assert _normalise("pytorch_optimizer") == _normalise("pytorch-optimizer")
    assert _normalise("PyYAML") == "pyyaml"
    assert _normalise("hydra.core") == "hydra-core"


def _dockerfile(tmp_path, *lines):
    path = tmp_path / "Dockerfile.framework"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_uv_failing_for_another_reason_is_not_read_as_staleness(monkeypatch):
    """Two different problems, and the report has to say which. Reading every
    non-zero exit as staleness sends the next lane to `uv lock` for a uv that is
    not on the PATH, and the lock it did not check stays unchecked."""

    class _Run:
        def __init__(self, returncode, stderr):
            self.returncode, self.stderr = returncode, stderr

    answers = {}
    monkeypatch.setattr(audit_plan.subprocess, "run", lambda cmd, **kw: answers[tuple(cmd)])

    command = ("uv", "lock", "--check")
    answers[command] = _Run(0, "")
    assert audit_plan._lock_is_current(REPO) is None

    answers[command] = _Run(
        2, "Resolved 142 packages\nThe lockfile at `uv.lock` needs to be updated"
    )
    assert audit_plan._lock_is_current(REPO) == "stale: regenerate with `uv lock`"

    answers[command] = _Run(1, "error: No such file or directory (os error 2)")
    assert (
        audit_plan._lock_is_current(REPO)
        == "unverified: error: No such file or directory (os error 2)"
    )


def test_a_stale_lock_is_named_rather_than_counted_silently(monkeypatch, tmp_path):
    """The state this check was written for: the lock an image installs from no
    longer agrees with the `pyproject.toml` it was resolved from, and the build
    said nothing because `uv sync --frozen` never asks."""
    monkeypatch.setattr(
        audit_plan,
        "_lock_is_current",
        lambda d: "stale: regenerate with `uv lock`" if d.name == "native" else None,
    )
    monkeypatch.setattr(
        audit_plan,
        "FRAMEWORK_DOCKERFILE",
        _dockerfile(tmp_path, "RUN cd envs/x && uv sync --locked"),
    )

    result = audit_plan.CHECKS["env-locks"]()

    assert not result.ok
    assert "envs/native/uv.lock is stale" in result.detail
    assert result.count == 1


def test_a_sync_that_does_not_assert_its_lock_is_a_failure(monkeypatch, tmp_path):
    """`--frozen` installs from the lock without checking it is current. A
    Dockerfile that only claims the check in a comment is the defect itself."""
    monkeypatch.setattr(audit_plan, "_lock_is_current", lambda directory: None)
    monkeypatch.setattr(
        audit_plan,
        "FRAMEWORK_DOCKERFILE",
        _dockerfile(
            tmp_path,
            "# --frozen so a stale lock fails the build",
            "RUN cd envs/x && uv sync --frozen --only-group build",
            "RUN cd envs/x && uv sync --locked",
        ),
    )

    result = audit_plan.CHECKS["env-locks"]()

    assert not result.ok
    assert "does not pass --locked" in result.detail
    assert result.count == 1


def test_a_dockerfile_that_only_talks_about_syncing_examines_nothing(monkeypatch, tmp_path):
    """Prose is not an instruction. Counting a comment as an invocation would let
    a Dockerfile that runs no sync at all satisfy the emptiness guard."""
    monkeypatch.setattr(audit_plan, "_lock_is_current", lambda directory: None)
    monkeypatch.setattr(
        audit_plan,
        "FRAMEWORK_DOCKERFILE",
        _dockerfile(tmp_path, "# RUN cd envs/x && uv sync --locked"),
    )

    result = audit_plan.CHECKS["env-locks"]()

    assert not result.ok
    assert "found no `uv sync` invocations" in result.detail


def test_a_lock_uv_could_not_answer_for_is_not_a_pass(monkeypatch, tmp_path):
    """An audit that cannot reach the answer has not got the answer — reporting a
    pass there is how a check goes hollow the first time a tool is unavailable."""
    monkeypatch.setattr(
        audit_plan, "_lock_is_current", lambda directory: "unverified: No such file or directory"
    )
    monkeypatch.setattr(
        audit_plan,
        "FRAMEWORK_DOCKERFILE",
        _dockerfile(tmp_path, "RUN cd envs/x && uv sync --locked"),
    )

    result = audit_plan.CHECKS["env-locks"]()

    assert not result.ok
    assert "unverified" in result.detail


def test_model_spec_compares_values_not_words():
    """Checking that the field name appears somewhere passes whether the value is
    true or false — the drift it was written to stop."""
    spec = {"m": {"hf_id": "org/m", "config": {"add_generation_prompt": True}}}

    assert model_spec_problems(spec, {"m": {"hf_id": "org/m", "add_generation_prompt": True}}) == []
    problems = model_spec_problems(spec, {"m": {"hf_id": "org/m", "add_generation_prompt": False}})
    assert problems == ["m.add_generation_prompt: spec True != config False"]


def test_model_spec_notices_a_model_on_only_one_side():
    assert model_spec_problems({"a": {}}, {}) == ["a is only in docs/model-spec.yaml"]
    assert model_spec_problems({}, {"a": {}}) == ["a is only in configs/model"]


def test_model_spec_notices_a_missing_field():
    spec = {"m": {"config": {"padding_side": "left"}}}

    assert model_spec_problems(spec, {"m": {}}) == [
        "m.padding_side is specified but absent from the config"
    ]


# The property, stated once over every check rather than over a hand-written
# subset. The subset version listed five names and omitted `plan-files`,
# `doc-commands`, `evidence-committed` and `axis-wired` — which were exactly the
# four that violated it. A list of what to check is a list of what to forget.
@pytest.fixture
def empty_repository(tmp_path, monkeypatch):
    """A repository with nothing in it, from every check's point of view."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs").mkdir()
    for name in ("PLAN.md", "README.md", "AGENTS.md"):
        (tmp_path / name).write_text("nothing here\n")
    monkeypatch.setattr(audit_plan, "REPO", tmp_path)
    monkeypatch.setattr(audit_plan, "CONFIGS", tmp_path / "configs")
    monkeypatch.setattr(audit_plan, "BENCH_ENTRY_POINT", tmp_path / "scripts" / "bench.py")
    # The axis registry is imported from the package, not read off REPO, so an
    # empty repository does not empty it.
    from trainbench import applied, axes, config_schema

    monkeypatch.setattr(config_schema, "axis_knobs", dict)
    monkeypatch.setattr(applied, "_CAPTURES", {})
    monkeypatch.setattr(axes, "IMPLEMENTED", frozenset())
    return tmp_path


@pytest.mark.parametrize("name", sorted(audit_plan.CHECKS))
def test_every_check_fails_on_an_empty_repository(name, empty_repository):
    """Vacuous truth is not evidence. A check that goes green when the thing it
    inspects disappears is worse than no check: it certifies the absence.

    Every one of these has been observed passing on nothing. `data-pinned`
    announced that every data config pins a commit sha when `.gitignore` had
    swallowed the directory; `axis-wired` reports "all 0 axes are applied and
    verified"; deleting `docs/support-matrix.md` made `evidence-committed` green;
    and adding a language tag to a markdown fence disabled `plan-files`.
    """
    result = audit_plan.CHECKS[name]()

    assert not result.ok, f"{name} certified an empty repository: {result.detail!r}"
    assert result.detail


def test_a_language_tag_on_the_fence_does_not_disable_plan_files(tmp_path, monkeypatch):
    """The fence regex required a bare ```; writing ```text is an ordinary edit
    that turned the check off and made it report the absence as a pass."""
    layout = "```text\ntrain-comparison/\n├── missing.py\n```\n"
    (tmp_path / "PLAN.md").write_text(layout)
    monkeypatch.setattr(audit_plan, "REPO", tmp_path)

    result = audit_plan.CHECKS["plan-files"]()

    assert not result.ok
    assert "missing.py" in result.detail


def test_the_data_config_group_is_committed():
    """It was ignored and untracked for the whole of Wave 0, so the gate only
    passed on the one checkout that happened to have the files locally. A clean
    clone could not compose a run at all."""
    tracked = subprocess.run(
        ["git", "ls-files", "configs/data"], cwd=REPO, capture_output=True, text=True
    ).stdout.split()

    assert [Path(p).name for p in tracked] == ["quality.yaml", "speed.yaml"]


@pytest.mark.parametrize("name", sorted(audit_plan.CHECKS))
def test_every_check_returns_a_result_named_after_itself(name):
    """The baseline is keyed by name, so a check whose Result carries a different
    one is permanently unmatched: never known, never fixed."""
    result = audit_plan.CHECKS[name]()

    assert result.name == name
    assert result.detail, "a failure with no detail cannot be acted on"


# `verdicts-closed`. Round two of the axis re-verifications returned six
# conditional verdicts, each carrying a list of things to do before merge, and
# the lists were not read because the gate was green. These pin the ledger that
# replaces that memory — and, more importantly, every way it could go hollow.
def item(item_id="i", anchor=None, closed=None, **overrides):
    """A ledger item in the stored shape, with only the field under test varied."""
    base = {
        "id": item_id,
        "axis": "kernel",
        "finding": "F1",
        "owner": "D",
        "verdict": "r2-rv-kernel.md",
        "summary": "무언가가 열려 있다",
        "closes_when": {
            "criterion": "변이가 빨개질 것",
            "command": "pytest -p mut",
            "expected": "빨강",
            "observed": "593 passed",
        },
        "anchor": anchor or {"kind": "test", "file": "tests/t.py", "names": ["test_absent"]},
        "closed": closed,
    }
    return {**base, **overrides}


def ledger(*items):
    return {"items": list(items)}


@pytest.fixture
def anchored(tmp_path):
    """A tree where one anchor holds and one does not."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text("def test_present():\n    pass\n")
    return tmp_path


def test_an_open_item_keeps_the_gate_red(anchored):
    """The whole point. An item whose anchor is not in the tree is open, and one
    open item is enough."""
    problems, open_ids, closed = verdict_ledger_problems(ledger(item()), anchored, set())

    assert (problems, open_ids, closed) == ([], ["i"], [])


def test_prose_cannot_close_an_item(anchored):
    """Writing that it is fixed changes nothing: state is derived from the anchor.
    A `closed` record over an anchor that does not hold is reported as a fix that
    was landed and then lost, which is the same event as a regression."""
    closed = {"date": "2026-08-02", "by": "누군가", "evidence": "고쳤다"}

    problems, open_ids, _ = verdict_ledger_problems(ledger(item(closed=closed)), anchored, set())

    assert open_ids == []
    assert problems == ["i: recorded closed but its anchor is gone (tests/t.py)"]


def test_landing_the_fix_without_recording_the_run_still_blocks(anchored):
    """Same reason a baseline entry that starts passing blocks. The anchor is the
    artifact; the mutation run behind it is the evidence, and this repository's
    failure was skipping the second one while the first looked done."""
    landed = {"kind": "test", "file": "tests/t.py", "names": ["test_present"]}

    problems, open_ids, closed = verdict_ledger_problems(
        ledger(item(anchor=landed)), anchored, set()
    )

    assert (open_ids, closed) == ([], [])
    assert problems == ["i: anchor now holds; record the run in `closed` and close it"]


def test_an_item_is_closed_only_with_both_the_anchor_and_the_record(anchored):
    landed = {"kind": "test", "file": "tests/t.py", "names": ["test_present"]}
    record = {"date": "2026-08-02", "by": "감사 레인", "evidence": "변이가 빨개짐"}

    problems, open_ids, closed = verdict_ledger_problems(
        ledger(item(anchor=landed, closed=record)), anchored, set()
    )

    assert (problems, open_ids, closed) == ([], [], ["i"])


def test_deleting_an_item_is_reported_from_git_history(anchored):
    """The obvious way to make this green is to delete the entry, and it is what
    happened to two other checks here: `plan-files` stayed true by mentioning
    less, `doc-commands` was satisfied by removing the import. History is the
    second witness — an id that was ever committed has to still be there."""
    problems, open_ids, _ = verdict_ledger_problems(ledger(item()), anchored, {"i", "deleted-one"})

    assert open_ids == ["i"]
    assert problems == ["deleted-one: was committed to this ledger and is now absent"]


def test_git_history_pins_every_id_the_ledger_ever_carried(tmp_path):
    """Two commits, the second one written without indentation so that every id
    lands on a single line. A per-line pattern finds one id there and pins
    nothing else, which would let the rest be deleted."""
    path = tmp_path / audit_plan.OPEN_VERDICTS
    path.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    def commit(payload, message, **dump):
        path.write_text(json.dumps(payload, **dump))
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(
            # --no-verify: the developer's own hooks are not this fixture's rules.
            ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", message, "-n"],
            cwd=tmp_path,
            check=True,
        )

    commit(ledger(item(item_id="first"), item(item_id="second")), "one", indent=2)
    commit(ledger(item(item_id="first"), item(item_id="second"), item(item_id="third")), "two")
    path.write_text(json.dumps(ledger(item(item_id="first"))))

    committed = audit_plan._committed_verdict_ids(tmp_path)

    assert committed == {"first", "second", "third"}
    problems, _, _ = verdict_ledger_problems(json.loads(path.read_text()), tmp_path, committed)
    assert problems == [
        "second: was committed to this ledger and is now absent",
        "third: was committed to this ledger and is now absent",
    ]


def test_an_empty_ledger_is_not_a_pass(anchored, monkeypatch):
    """Convention §6: a check that iterates a set fails when the set is empty.
    Emptying the file is deleting every item at once."""
    monkeypatch.setattr(audit_plan, "REPO", anchored)
    path = anchored / audit_plan.OPEN_VERDICTS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"items": []}))

    result = audit_plan.CHECKS["verdicts-closed"]()

    assert not result.ok
    assert "nothing to examine" in result.detail


def test_an_unreadable_ledger_does_not_pass_quietly(anchored, monkeypatch):
    monkeypatch.setattr(audit_plan, "REPO", anchored)
    path = anchored / audit_plan.OPEN_VERDICTS
    path.parent.mkdir(parents=True)
    path.write_text("{not json")

    result = audit_plan.CHECKS["verdicts-closed"]()

    assert not result.ok
    assert "does not parse" in result.detail


@pytest.mark.parametrize(
    "broken, expected",
    [
        (item(closes_when={"criterion": "?"}), "closes_when needs"),
        (item(anchor={"kind": "vibes", "file": "tests/t.py"}), "anchor must name a kind"),
        (item(anchor={"kind": "test", "file": "tests/t.py"}), "anchor must name a kind"),
        ({"id": "i", "axis": "kernel"}, "missing"),
        ("just a string", "is not an object"),
        # An empty name list is a subset of every file's functions, so this item
        # would close on nothing — `_nothing_to_check`'s failure one level down.
        (item(anchor={"kind": "test", "file": "tests/t.py", "names": []}), "closes on nothing"),
        # A gate that raises on one item's typo stops answering for the rest.
        (item(anchor={"kind": "text", "file": "tests/t.py", "pattern": "["}), "does not compile"),
    ],
)
def test_a_malformed_item_is_a_failure_not_a_skip(anchored, broken, expected):
    """An item the check cannot evaluate must not drop out of the count. Skipping
    it silently is a hole shaped exactly like deleting it."""
    problems, open_ids, closed = verdict_ledger_problems(ledger(broken), anchored, set())

    assert (open_ids, closed) == ([], [])
    assert len(problems) == 1 and expected in problems[0]


def test_two_items_cannot_share_an_id(anchored):
    """History pins ids, so a duplicate makes a deleted item indistinguishable
    from a surviving one."""
    problems, open_ids, _ = verdict_ledger_problems(ledger(item(), item()), anchored, set())

    assert open_ids == ["i"]
    assert problems == ["i: duplicate id"]


def test_a_test_anchor_needs_the_test_to_be_defined_not_merely_mentioned(anchored):
    """`grep` for a name is satisfied by the name in a comment, a docstring or a
    skip marker. The names come out of the AST."""
    (anchored / "tests" / "t.py").write_text(
        '"""test_mentioned is what we should add."""\n_NOTE = "test_mentioned"\n'
    )
    anchor = {"kind": "test", "file": "tests/t.py", "names": ["test_mentioned"]}

    assert not anchor_holds(anchor, anchored)


def test_a_test_anchor_wants_every_name_it_lists(anchored):
    """Four surviving mutants need four tests, and three of them is not closed."""
    anchor = {"kind": "test", "file": "tests/t.py", "names": ["test_present", "test_absent"]}

    assert not anchor_holds(anchor, anchored)
    assert anchor_holds({**anchor, "names": ["test_present"]}, anchored)


def test_an_absent_anchor_holds_only_once_the_wording_is_gone(anchored):
    """For items whose deliverable is a retraction: a sentence claiming more than
    the code checks, or a plan describing the wrong discriminator."""
    (anchored / "PLAN.md").write_text("distinguished by `context_fn`\n")
    anchor = {"kind": "absent", "file": "PLAN.md", "pattern": "`context_fn`"}

    assert not anchor_holds(anchor, anchored)
    (anchored / "PLAN.md").write_text("distinguished by the policy it carries\n")
    assert anchor_holds(anchor, anchored)


def test_an_anchor_whose_file_is_gone_does_not_close_an_item(anchored):
    """Including the `absent` kind, where a missing file makes the pattern
    trivially absent. Deleting the document is not doing the work."""
    assert not anchor_holds({"kind": "absent", "file": "gone.md", "pattern": "x"}, anchored)
    assert not anchor_holds({"kind": "text", "file": "gone.md", "pattern": "x"}, anchored)


def test_the_repositorys_own_ledger_has_open_items_and_names_them():
    """The live state. Six conditional verdicts, and the gate says so out loud
    rather than leaving it to whoever remembers reading them."""
    result = audit_plan.CHECKS["verdicts-closed"]()

    assert not result.ok
    assert "open:" in result.detail
    assert result.count and result.count > 0


# envs/native installs flash-attn as a wheel this project compiled, pinned by URL.
# uv checks nothing about a URL wheel, so these are the comparisons that stand
# between a drifted lock and a CUDA error on a paid pod.
WHEEL_URL = (
    "https://github.com/jinwoo-cho-ilevit-com/train-comparison/releases/download/"
    "flash-attn-2.8.3.post1-cu130-torch2.13.0-cp313/"
    "flash_attn-2.8.3.post1-cp313-cp313-linux_x86_64.whl"
)
WHEEL_SHA = "166a27d0090ab036029673202f518b2aaa2ca405c45c033cb5b58c0ede9b3d2a"
WHEEL_RECORD = {
    "env": "native",
    "package": "flash-attn",
    "version": "2.8.3.post1",
    "url": WHEEL_URL,
    "sha256": WHEEL_SHA,
    "abi": {"torch": "2.13.0", "cuda": "cu130", "python": "cp313", "cuda_archs": [80, 90, 100]},
}
WHEEL_PACKAGES = {
    "flash-attn": {
        "name": "flash-attn",
        "version": "2.8.3.post1",
        "source": {"url": WHEEL_URL},
        "wheels": [{"url": WHEEL_URL, "hash": f"sha256:{WHEEL_SHA}"}],
    },
    "torch": {"name": "torch", "version": "2.13.0+cu130"},
}


def _wheel_problems(record=None, packages=None, requires_python="==3.13.*", archs="80;90;100"):
    return prebuilt_wheel_problems(
        record or WHEEL_RECORD, requires_python, packages or WHEEL_PACKAGES, archs
    )


def test_a_prebuilt_wheel_that_matches_its_lock_has_nothing_to_report():
    assert _wheel_problems() == []


def test_a_torch_bump_invalidates_the_binary_and_is_named_as_that():
    """The mutation with no other alarm. The lock re-resolves cleanly, the image
    builds, and the wheel's C++ ABI is gone — which surfaces as a CUDA error
    mid-run, on a pod, after the pod-hour is spent."""
    bumped = WHEEL_PACKAGES | {"torch": {"name": "torch", "version": "2.14.0+cu130"}}

    problems = _wheel_problems(packages=bumped)

    assert len(problems) == 1
    assert "the lock resolves torch 2.14.0" in problems[0]


def test_a_cuda_build_the_wheel_was_not_compiled_for_is_caught():
    on_cu128 = WHEEL_PACKAGES | {"torch": {"name": "torch", "version": "2.13.0+cu128"}}

    assert "the wheel was built for cu130" in "; ".join(_wheel_problems(packages=on_cu128))


def test_requires_python_must_admit_the_wheels_interpreter_and_no_other():
    """Containment alone is not the property: `>=3.13` admits 3.13 and also 3.14,
    and a cp313 binary loaded by 3.14 is exactly the failure this stops."""
    assert _wheel_problems(requires_python="==3.13.*") == []

    for widened in (">=3.13", ">=3.12,<3.14", ""):
        problems = _wheel_problems(requires_python=widened)
        assert len(problems) == 1, widened
        assert "is not 3.13 alone" in problems[0]


def test_a_lock_that_went_back_to_building_from_source_is_caught():
    """Reverting [tool.uv.sources] to a plain version spec re-locks cleanly and
    costs the next native image build 13,663 s. The record outliving the URL is
    the whole signal."""
    from_pypi = WHEEL_PACKAGES | {
        "flash-attn": {
            "name": "flash-attn",
            "version": "2.8.3.post1",
            "source": {"registry": "https://pypi.org/simple"},
            "sdist": {"url": "https://files.pythonhosted.org/flash_attn-2.8.3.post1.tar.gz"},
        }
    }

    problems = "; ".join(_wheel_problems(packages=from_pypi))

    assert "installs flash-attn from registry, not from the recorded wheel URL" in problems


def test_a_package_the_lock_no_longer_carries_is_a_record_that_outlived_it():
    assert _wheel_problems(packages={"torch": WHEEL_PACKAGES["torch"]}) == [
        "docs/prebuilt-wheels.yaml native/flash-attn: envs/native/uv.lock has no flash-attn; "
        "the record outlived what it records"
    ]


def test_the_recorded_digest_must_be_the_one_the_lock_pins():
    swapped = WHEEL_RECORD | {"sha256": "0" * 64}

    assert "the lock pins" in "; ".join(_wheel_problems(record=swapped))


def test_a_record_may_not_claim_an_abi_its_own_artifact_denies():
    """The filename and the release tag are the artifact's own statement about
    what it is. A record is free to be wrong; it is not free to be wrong in a way
    the URL it points at contradicts."""
    lying = WHEEL_RECORD | {"abi": WHEEL_RECORD["abi"] | {"python": "cp312"}}

    problems = "; ".join(_wheel_problems(record=lying))

    assert "the wheel filename says python tag cp313, recorded cp312" in problems


def test_a_release_tag_that_states_no_abi_is_not_a_pass():
    """A wheel filename cannot carry a torch version, so the tag is the only place
    the ABI is written down. Untagged means nothing says what this binary was
    built against — which is a failure, not an absence of one."""
    untagged = WHEEL_RECORD | {
        "url": "https://example.invalid/v1/flash_attn-2.8.3.post1-cp313-cp313-linux_x86_64.whl"
    }
    lock = WHEEL_PACKAGES | {
        "flash-attn": WHEEL_PACKAGES["flash-attn"]
        | {"source": {"url": untagged["url"]}, "wheels": [{"hash": f"sha256:{WHEEL_SHA}"}]}
    }

    problems = "; ".join(_wheel_problems(record=untagged, packages=lock))

    assert "states no torch/CUDA ABI" in problems


def test_an_image_arch_the_wheel_has_no_kernels_for_is_a_dead_pod():
    """flash-attn emits code=sm_XX with no PTX — measured on this very wheel: 72
    fatbins, 216 SASS entries at sm_80/90/100, zero PTX entries. Widening the
    image's declared archs without rebuilding gives "no kernel image is
    available", and the audit is the last place that can say so."""
    assert "the image declares CUDA archs" in "; ".join(_wheel_problems(archs="80;90;100;120"))
    assert "declares no TRAINBENCH_CUDA_ARCHS" in "; ".join(_wheel_problems(archs=None))


def test_the_repositorys_own_prebuilt_wheel_agrees_with_its_lock():
    """The live state, not a fixture. envs/native/uv.lock must install exactly the
    wheel docs/prebuilt-wheels.yaml records, for the torch it resolves."""
    result = audit_plan.CHECKS["prebuilt-wheels"]()

    assert result.ok, result.detail
    assert result.count == 0
