"""The wave gate itself. Six lanes will trust whatever this says.

Written after a review found that the audit certified the one defect it was
built to catch: the config-consumed check matched bare identifiers, so every
ablation axis counted as consumed by any file that happened to use the word
`name`, and `--update-baseline` replaced each entry's resolving wave with a
placeholder.
"""

from __future__ import annotations

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
    _reads_dotted,
    classify,
    merge_baseline,
    missing_plan_files,
    model_spec_problems,
)


def failing(*names):
    return [Result(name, False, "") for name in names]


def passing(*names):
    return [Result(name, True, "") for name in names]


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


def test_updating_the_baseline_keeps_the_schedule():
    """Each entry names the wave that resolves it. That is what makes the
    baseline a schedule rather than an excuse, and overwriting it with a
    placeholder erases the plan for every lane at once."""
    baseline = {"axis-packages": "Wave 2 (F)", "config-consumed": "Wave 2 (D)"}

    merged = merge_baseline(baseline, failing("axis-packages", "config-consumed"))

    assert merged == baseline


def test_a_newly_failing_check_enters_as_unscheduled():
    merged = merge_baseline(
        {"axis-packages": "Wave 2 (F)"}, failing("axis-packages", "data-pinned")
    )

    assert merged == {"axis-packages": "Wave 2 (F)", "data-pinned": "unscheduled"}


def test_a_check_that_now_passes_leaves_the_baseline():
    merged = merge_baseline({"data-pinned": "Wave 1 (A)"}, passing("data-pinned"))

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
    regressions, fixed = classify(failing("a", "b"), {"a": "Wave 1"})

    assert regressions == ["b"]
    assert fixed == []


def test_a_baseline_entry_that_starts_passing_blocks():
    """A stale baseline grants amnesty to whatever breaks there next."""
    regressions, fixed = classify(passing("a"), {"a": "Wave 1"})

    assert (regressions, fixed) == ([], ["a"])


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

    assert missing_plan_files(LAYOUT, tmp_path) == ["trainbench/device.py", "scripts/bench.py"]


def test_a_directory_does_not_leak_into_its_sibling(tmp_path):
    """Depth decides the parent. Without that, `bench.py` under `scripts/` would
    be looked for under whichever directory was named last."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bench.py").touch()
    (tmp_path / "trainbench").mkdir()
    (tmp_path / "trainbench" / "config.py").touch()
    (tmp_path / "PLAN.md").touch()

    assert missing_plan_files(LAYOUT, tmp_path) == ["trainbench/device.py"]


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


# Checks that answer a question about a set of files. Each one reported success
# when its set was empty: `configs/data/` was matched by an unanchored `data/` in
# .gitignore and had never been committed, so `data-pinned` announced that every
# data config pins a commit sha — of which there were none.
def test_an_orchestrator_manifest_is_not_a_config_knob(tmp_path, monkeypatch):
    """`configs/experiment/` holds pod work orders, not run settings. Treating its
    keys as unread knobs charged lane C's files to lane D's audit item, which D
    could then never clear."""
    configs = tmp_path / "configs"
    (configs / "attn").mkdir(parents=True)
    (configs / "experiment").mkdir()
    (configs / "config.yaml").write_text("defaults:\n  - attn: sdpa\n")
    (configs / "attn" / "sdpa.yaml").write_text("name: sdpa\n")
    (configs / "experiment" / "job.yaml").write_text("phase: 0\naxis: attn\n")
    monkeypatch.setattr(audit_plan, "CONFIGS", configs)

    assert audit_plan._config_leaf_keys() == {"attn": {"name"}}


def test_a_config_directory_nothing_composes_must_be_declared(tmp_path, monkeypatch):
    """Narrowing to composed groups would otherwise fail open: an axis group added
    to configs/ but forgotten in `defaults` would vanish from every check at once
    while also never reaching a run."""
    configs = tmp_path / "configs"
    (configs / "newaxis").mkdir(parents=True)
    (configs / "config.yaml").write_text("defaults:\n  - attn: sdpa\n")
    monkeypatch.setattr(audit_plan, "CONFIGS", configs)

    result = audit_plan.CHECKS["config-groups"]()

    assert not result.ok
    assert "newaxis" in result.detail


SET_CHECKS = ("config-consumed", "axis-fields", "axis-packages", "data-pinned", "model-spec")


@pytest.mark.parametrize("name", SET_CHECKS)
def test_a_check_with_nothing_to_examine_fails(name, monkeypatch, tmp_path):
    """Vacuous truth is not evidence. A check that goes green when the thing it
    inspects disappears is worse than no check: it certifies the absence."""
    monkeypatch.setattr(audit_plan, "CONFIGS", tmp_path / "configs")
    monkeypatch.setattr(audit_plan, "REPO", tmp_path)

    result = audit_plan.CHECKS[name]()

    # The property is the verdict, not the wording: `model-spec` reports its
    # missing spec file first, which is the same refusal for a nearer reason.
    assert not result.ok, f"{name} certified an empty repository"


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
