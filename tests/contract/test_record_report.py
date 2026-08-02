"""The `record-report` boundary: what a run writes and what the merge may read.

`trainbench/metrics/` and `scripts/bench.py` (lane-d) write the run record.
`scripts/report.py` (lane-b) is the only thing that reads it back. Neither lane
owns this file, so neither can move the boundary while working against it.

`tests/fixtures/run_record.sample.json` is the payload itself: a tevatron x
qwen3_5_0_8b timing run on one H100. Its `config` block was composed by Hydra
from `configs/` and its `metrics` block by `metrics.summarise`, so the numbers
are self-consistent rather than invented. Variations - the frozen graph, the OOM,
the other stack, the record with no identity - are built here from that one file.

Five groups are pinned, each because a measurement of the current tree showed it
unpinned:

1. **Identity.** `report.load_artifacts` falls back to `path.stat().st_mtime`,
   and none of the 40 artifacts in the results repo carry `recorded_at`. With
   every mtime equal - a clean clone downloaded in one go - 8 of 18 cells select
   the previous campaign's artifact.
2. **Token accounting.** Real and padded tokens are separate fields, and every
   rate is recomputed from the counters rather than taken from a framework.
3. **The training-validity gate.** Three unsloth cells passed `infonce_backward`
   with `params_with_grad=0` and `trainable_params=0`. A run that trained nothing
   must not be readable as a speed result.
4. **Peak memory and OOM.** Memory travels beside speed, and OOM is its own
   result category - not "slow", not "unsupported", not "never attempted".
5. **The resolved stack.** The six images cannot be unified (transformers 5.14.1
   / 5.12.1 / 5.5.0). `report.py` ranks only cells sharing a stack, so the stack
   key has to be mechanically derivable from the record.

Also carried: per-axis states including `framework_owned` (the vocabulary is
owned by the `applied-axes` boundary; this one pins that the record carries it),
and the scope a figure is true within - a framework's speed ratio inverts with
sequence length and GPU count.

The `xfail(strict=True)` tests are the parts of this contract the tree does not
satisfy today. They run, they fail for a named reason, and they turn into errors
the moment a lane implements them - which is how the lane learns to drop the
marker rather than leaving a green test that asserts nothing.
"""

from __future__ import annotations

import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir

from trainbench.compose import resolve
from trainbench.record import build_record

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "tests" / "fixtures" / "run_record.sample.json"

sys.path.insert(0, str(REPO / "scripts"))

import report  # noqa: E402

# The record field that says which campaign an artifact belongs to. The whole of
# group 1 is that this exists and that nothing downstream may substitute a clock
# the filesystem happens to be carrying.
IDENTITY_FIELD = "recorded_at"

# When the campaign the stored sample belongs to ran. Written out rather than read
# back out of the sample: a test that orders artifacts must keep working when the
# mutation under examination is the removal of that very field.
CAMPAIGN_AT = 1785974400.0

# What every artifact carries, whatever it is a record of.
REQUIRED_TOP_LEVEL = (
    IDENTITY_FIELD,
    "git_commit",
    "config",
    "applied",
    "device",
    "packages",
    "host",
)

# What a record that reports figures carries in `metrics`, and why each is here.
REQUIRED_METRICS = (
    # group 2 - the denominator, stated twice because the two answers differ
    "tokens_per_step",
    "padded_tokens_per_step",
    "tokens_per_second",
    "metric_definitions",
    # group 3 - the validity gate
    "grad_norm",
    "trainable_params",
    "total_params",
    "loss_first",
    "loss_last",
    # group 4 - memory beside speed
    "peak_memory_bytes",
    # how the figure was produced, without which two summaries are not comparable
    "steps_timed",
    "steps_discarded",
    "steps_measured",
    "step_seconds_p50",
    "step_seconds_mean",
    "profiled",
)

# A count and the rate derived from it. `summarise` computes every rate as
# count / mean step time; a framework's own tokens/sec would not land on that.
COUNT_AND_RATE = (
    ("tokens_per_step", "tokens_per_second"),
    ("padded_tokens_per_step", "padded_tokens_per_second"),
    ("rows_per_step", "rows_per_second"),
    ("samples_per_step", "samples_per_second"),
    ("images_per_step", "images_per_second"),
)

# Config fields `scripts/report.py` reads out of every record. Renaming one of
# these is an interface change even though it happens inside the schema.
CONSUMED_CONFIG_FIELDS = (
    ("framework", "name"),
    ("model", "name"),
    ("run", "purpose"),
    ("run", "profiler"),
    ("data", "max_seq_len"),
    ("parallel", "strategy"),
    ("peft", "mode"),
)

# The four things that can be true of a record instead of "it measured something".
# They are mutually exclusive and none of them may be rendered as another.
STATUS_NO_RESULT = "no_result"  # publish_result.fallback_record: no file came back
STATUS_REFUSED = "axis-refused"  # scripts/bench.py: the setting could not be applied
STATUS_OOM = "oom"  # the run hit the memory ceiling - a result, not an absence

# The axis-ownership state that says an axis is not ours to apply on this cell
# (tevatron's DenseModel.forward owns the loss and the cross-device gather).
FRAMEWORK_OWNED = "framework_owned"

_DELETE = object()


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in over.items():
        if value is _DELETE:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def record(**over: Any) -> dict[str, Any]:
    """The stored payload, with overrides merged in. `_DELETE` removes a key."""
    return _merge(json.loads(SAMPLE.read_text()), over)


def probe_record(checks: int, **over: Any) -> dict[str, Any]:
    """The sample turned into what a probe pod uploads.

    `checks` many passing checks, so which artifact a merge selected is legible
    from the rendered cell rather than from the file it came out of.
    """
    payload = record(config={"run": {"purpose": "probe"}}, metrics=_DELETE, **over)
    config = payload["config"]
    payload["probe"] = {
        "framework": config["framework"]["name"],
        "model": config["model"]["name"],
        "all_ok": True,
        "unexpected_passes": [],
        "applied": payload["applied"],
        "checks": [
            {
                "name": f"check_{index}",
                "ok": True,
                "expected_failure": False,
                "detail": {},
                "error": None,
                "error_type": None,
                "traceback": None,
            }
            for index in range(checks)
        ],
    }
    return payload


def write(
    root: Path,
    payload: dict[str, Any],
    *,
    pod: str,
    label: str | None = None,
    mtime: float | None = None,
    name: str = report.RESULT_NAME,
) -> Path:
    """File one artifact where a pod would have published it."""
    config = payload["config"]
    payload = _merge(payload, {"host": {"runpod_pod_id": pod}})
    parts = [config["framework"]["name"], config["model"]["name"], pod]
    if label is not None:
        parts.append(label)
    path = root.joinpath("results", *parts, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def merged(root: Path) -> str:
    """Run the documented merge and return the document it wrote."""
    matrix = root / "support-matrix.md"
    matrix.write_text("# 지원 매트릭스\n\n손으로 쓴 부분.\n")
    assert report.main(["--results", str(root), "--matrix", str(matrix)]) == 0
    return matrix.read_text()


def matrix_cell(document: str, framework: str, model: str) -> str:
    """The rendered cell for one combination of the generated matrix."""
    column = report.MODELS.index(model) + 1
    for line in document.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == framework:
            return cells[column]
    raise AssertionError(f"no row for {framework} in\n{document}")


def tables(document: str) -> list[list[str]]:
    """Consecutive runs of table lines. One run is one ranking."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in document.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def composed_config():
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=qwen3_5_0_8b",
                "framework=native",
                "run=timing",
                "data.limit=512",
                "train.batch_size=16",
            ],
        )
        return resolve(cfg)[0]


def stack_key(payload: dict[str, Any]) -> tuple[str, str] | None:
    """What stack a figure was measured on, or None if the record cannot say.

    The two packages that differ across the six framework images. A record that
    cannot answer this cannot be ranked against anything, which is a finding and
    not a reason to guess.
    """
    packages = payload.get("packages") or {}
    torch_version = packages.get("torch")
    transformers_version = packages.get("transformers")
    if not torch_version or not transformers_version:
        return None
    return str(torch_version), str(transformers_version)


def scope_label(payload: dict[str, Any]) -> str:
    """The scope a figure is true within.

    A framework's speed ratio is a function of (model, sequence length, GPU
    count) and inverts inside it, so a number quoted without these is not a
    result. Derived rather than stored: every part is already in the record, and
    a stored label can disagree with the run it labels.
    """
    config = payload["config"]
    host = payload["host"]
    gpus = host["torch_device_count"]
    gpu = (host.get("gpu") or {}).get("name") or "unknown GPU"
    return (
        f"{gpus}x {gpu}, {config['parallel']['strategy']}, seqlen={config['data']['max_seq_len']}"
    )


def training_verdict(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Whether this run trained. The contract's own definition, kept executable.

    Four checks, from the reproduction that found zero gradient norms behind a
    published 46,000 tokens/second figure, plus the one this repository measured:
    a fully frozen graph stays differentiable through the embedding output, so
    `loss.backward()` returns normally and `params_with_grad` alone does not
    catch it.
    """
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False, ["no `metrics`: nothing was reported to check"]
    reasons = []

    grad_norm = metrics.get("grad_norm")
    if not isinstance(grad_norm, (int, float)) or isinstance(grad_norm, bool):
        reasons.append("`grad_norm` is absent")
    elif not math.isfinite(grad_norm) or grad_norm <= 0:
        reasons.append(f"`grad_norm`={grad_norm}: the backward reached no parameter")

    trainable = metrics.get("trainable_params")
    total = metrics.get("total_params")
    if not isinstance(trainable, int) or isinstance(trainable, bool) or trainable <= 0:
        reasons.append(f"`trainable_params`={trainable!r}: this model cannot learn")
    elif isinstance(total, int) and not isinstance(total, bool):
        mode = payload["config"]["peft"]["mode"]
        if mode == "full" and trainable != total:
            reasons.append(
                f"peft.mode=full but {trainable} of {total} parameter tensors train; "
                "a full finetune that froze part of the model is a different workload"
            )
        if mode in ("lora", "qlora") and trainable >= total:
            reasons.append(
                f"peft.mode={mode} but {trainable} of {total} parameter tensors train; "
                "the adapter did not narrow anything"
            )

    first, last = metrics.get("loss_first"), metrics.get("loss_last")
    if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
        reasons.append("`loss_first`/`loss_last` are absent")
    elif not (math.isfinite(first) and math.isfinite(last)):
        reasons.append(f"loss is not finite: {first} -> {last}")
    elif last >= first:
        reasons.append(f"loss did not fall: {first} -> {last}")

    if str(payload.get("device", "")).startswith("cuda"):
        peak = metrics.get("peak_memory_bytes")
        if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
            reasons.append(f"`peak_memory_bytes`={peak!r} on a CUDA run")

    return not reasons, reasons


# ---------------------------------------------------------------------------
# The sample is what the producer writes, not a shape invented for the test.
# ---------------------------------------------------------------------------


def test_the_stored_sample_is_the_shape_the_producer_writes():
    """Every key `build_record` emits, and every config field the merge reads.

    A fixture that drifted from the producer would let both lanes pass against a
    record neither of them will ever see.
    """
    config = composed_config()
    produced = build_record(config, torch.device("cpu"))
    payload = record()
    unseen = sorted(set(produced) - set(payload))
    assert not unseen, f"the producer writes keys the sample does not carry: {unseen}"

    dumped = config.model_dump(mode="json")
    for group, field in CONSUMED_CONFIG_FIELDS:
        assert field in dumped[group], f"the schema no longer declares {group}.{field}"
        assert field in payload["config"][group], f"the sample does not carry {group}.{field}"

    for key in REQUIRED_TOP_LEVEL:
        assert key in payload, f"the sample does not carry `{key}`"
    for key in REQUIRED_METRICS:
        assert key in payload["metrics"], f"the sample's `metrics` does not carry `{key}`"


# ---------------------------------------------------------------------------
# 1. Artifact identity
# ---------------------------------------------------------------------------


def test_the_recorded_identity_and_not_the_file_clock_picks_the_current_campaign(tmp_path):
    """The defect measured today, in the condition that reproduces it.

    A clean clone downloads every artifact in one go, so the mtimes carry no
    campaign order - here the *older* campaign is given the *newer* mtime, which
    is the worse half of the same coin. The cell must still read as the campaign
    that ran second.
    """
    old = probe_record(3, recorded_at=CAMPAIGN_AT - 30 * 86400)
    new = probe_record(7, recorded_at=CAMPAIGN_AT)
    write(tmp_path, old, pod="pod-previous-campaign", mtime=2_000_000_000)
    write(tmp_path, new, pod="pod-this-campaign", mtime=1_000_000_000)

    document = merged(tmp_path)
    assert matrix_cell(document, "tevatron", "qwen3_5_0_8b") == "OK (7 checks)", (
        "the merge picked the previous campaign's artifact; only the file clock "
        "distinguishes them and the file clock is not campaign order"
    )


def test_the_identity_is_a_clock_the_run_read_not_one_the_filesystem_carries(tmp_path):
    """`load_artifacts` must order on the record, so equal mtimes change nothing."""
    earlier = probe_record(3, recorded_at=CAMPAIGN_AT - 86400)
    later = probe_record(7, recorded_at=CAMPAIGN_AT)
    write(tmp_path, earlier, pod="a", mtime=1_700_000_000)
    write(tmp_path, later, pod="b", mtime=1_700_000_000)

    artifacts, skipped = report.load_artifacts(tmp_path)
    assert not skipped
    ordered = sorted(artifacts, key=lambda a: a.timestamp)
    assert [len(report.checks_of(a)) for a in ordered] == [3, 7]
    assert ordered[0].timestamp != ordered[1].timestamp, (
        "both artifacts share one timestamp, so their order is whatever `sorted` "
        "happened to do - the identity is not being read"
    )


# ---------------------------------------------------------------------------
# 2. Token accounting
# ---------------------------------------------------------------------------


def test_real_and_padded_tokens_never_collapse_into_one_number():
    """Padding is up to 89% of a batch on some corpora, and it flips the packing axis."""
    metrics = record()["metrics"]
    real, padded = metrics["tokens_per_step"], metrics["padded_tokens_per_step"]
    assert isinstance(real, float) and isinstance(padded, float)
    assert padded > real, (
        "this batch pads, so the two counts must differ; equal counts mean one "
        "field is standing in for the other"
    )

    definitions = metrics["metric_definitions"]
    assert definitions["tokens"] != definitions["padded_tokens"]
    assert "padding" in definitions["tokens"]
    assert "padding included" in definitions["padded_tokens"]

    assert "tokens" not in metrics, (
        "a bare `tokens` key is what lets a number mean real tokens in one record "
        "and rows in the next; the counts are named `*_per_step` / `*_per_second`"
    )
    assert metrics["rows_per_step"] == 2 * metrics["samples_per_step"], (
        "`rows` is queries + positives and `samples` is the pairs, so one is twice "
        "the other; if they are equal one name is being used for both quantities"
    )


def test_every_rate_is_recomputed_from_this_records_own_counters():
    """No framework's `tokens/sec` may enter. Arithmetic is what says it did not."""
    metrics = record()["metrics"]
    mean = metrics["step_seconds_mean"]
    for count_key, rate_key in COUNT_AND_RATE:
        count, rate = metrics[count_key], metrics[rate_key]
        assert rate == pytest.approx(count / mean, rel=1e-9), (
            f"{rate_key}={rate} is not {count_key}/{mean} = {count / mean}; a rate "
            "this record did not compute came from somewhere else"
        )


# ---------------------------------------------------------------------------
# 3. The training-validity gate
# ---------------------------------------------------------------------------


def test_the_gate_fields_travel_with_every_measurement():
    payload = record()
    metrics = payload["metrics"]
    assert isinstance(metrics["grad_norm"], float) and metrics["grad_norm"] > 0
    assert isinstance(metrics["trainable_params"], int) and metrics["trainable_params"] > 0
    assert metrics["total_params"] >= metrics["trainable_params"]
    assert math.isfinite(metrics["loss_first"]) and math.isfinite(metrics["loss_last"])
    assert isinstance(metrics["peak_memory_bytes"], int) and metrics["peak_memory_bytes"] > 0
    assert training_verdict(payload) == (True, [])


@pytest.mark.parametrize(
    ("name", "over", "expected"),
    [
        (
            "the frozen graph three unsloth cells passed today",
            {"metrics": {"grad_norm": 0.0, "trainable_params": 0, "loss_last": 2.8734}},
            ("grad_norm", "trainable_params"),
        ),
        (
            "a full finetune that quietly froze most of the model",
            {"metrics": {"trainable_params": 12}},
            ("peft.mode=full",),
        ),
        (
            "a LoRA run in which everything trains",
            {"config": {"peft": {"mode": "lora", "r": 16, "alpha": 32}}},
            ("peft.mode=lora",),
        ),
        (
            "a loss that did not fall over the measured window",
            {"metrics": {"loss_last": 2.9}},
            ("loss did not fall",),
        ),
        (
            "a CUDA run reporting no peak memory",
            {"metrics": {"peak_memory_bytes": None}},
            ("peak_memory_bytes",),
        ),
    ],
)
def test_the_gate_refuses_a_run_that_did_not_train(name, over, expected):
    ok, reasons = training_verdict(record(**over))
    assert not ok, f"{name} passed the gate"
    joined = " ".join(reasons)
    for fragment in expected:
        assert fragment in joined, f"{name}: no reason mentions {fragment}; got {reasons}"


# ---------------------------------------------------------------------------
# 4. Peak memory, and 5. the resolved stack
# ---------------------------------------------------------------------------


def test_peak_memory_is_reported_beside_the_speed_it_was_bought_with(tmp_path):
    """Frameworks buy speed with VRAM in different amounts. Speed alone is not a comparison."""
    write(tmp_path, record(), pod="pod-a", label="canonical")
    document = merged(tmp_path)
    assert "peak mem (GiB)" in document
    assert "32.00" in document, (
        "the 34359738368-byte peak the record carries is not in the document; a "
        "speed table without it compares runs that bought their speed differently"
    )


def test_the_stack_a_number_was_measured_on_is_derivable_from_the_record():
    """Two packages, both already in `record.package_versions`. Nothing new is needed.

    Measured: native / sentence_transformers / tevatron / axolotl are on
    transformers 5.14.1 + torch 2.13.0, ms_swift on 5.12.1, unsloth on 5.5.0 +
    torch 2.11.0. They cannot be unified, so the key has to be readable instead.
    """
    assert stack_key(record()) == ("2.13.0", "5.14.1")
    unsloth = record(packages={"torch": "2.11.0", "transformers": "5.5.0"})
    assert stack_key(unsloth) != stack_key(record())
    assert stack_key(record(packages={"transformers": _DELETE})) is None, (
        "a record that cannot name its stack must say so; ranking it against "
        "another stack is the thing this key exists to prevent"
    )


def test_the_scope_a_figure_is_true_within_travels_with_it():
    """(GPU count, sequence length) is where a framework's speed ratio inverts."""
    payload = record()
    assert payload["config"]["data"]["max_seq_len"] > 0
    assert payload["config"]["parallel"]["strategy"]
    assert payload["host"]["torch_device_count"] >= 1
    assert (payload["host"].get("gpu") or {}).get("name")
    assert scope_label(payload) == "1x NVIDIA H100 80GB HBM3, single, seqlen=2048"


def test_every_axis_carries_its_state_and_framework_owned_is_not_a_mismatch():
    """The third state, carried into the record.

    tevatron's `DenseModel.forward` does the encoding, pooling, scoring, InfoNCE
    and the distributed gather itself, so on that cell `loss.name` and
    `parallel.cross_device_negatives` are not ours. With only "applied" and "not
    applied" the run is refused as a mismatch; with the state carried, the cell
    is measurable and the report can show which axes it lost. The vocabulary
    belongs to the `applied-axes` boundary - what is pinned here is that the
    record carries it and that it does not read as a failed axis.
    """
    applied = record()["applied"]
    for axis in applied["axes"]:
        assert isinstance(axis.get("state"), str) and axis["state"], (
            f"{axis['axis']} carries no `state`; the report cannot tell an axis "
            "this framework owns from one that failed to apply"
        )
    owned = [a for a in applied["axes"] if a["state"] == FRAMEWORK_OWNED]
    assert {a["axis"] for a in owned} == {"loss.name", "parallel.cross_device_negatives"}
    assert all(a["matches"] is False for a in owned)
    assert applied["all_matched"] is True, (
        "a framework-owned axis reads as a mismatch, which is what refuses the run"
    )
    assert applied["missing"] == []


# ---------------------------------------------------------------------------
# What the tree does not satisfy today. Each names the lane that closes it.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="lane-d: `record.build_record` writes no `recorded_at`, so 0 of the 40 "
    "artifacts in the results repo carry one",
)
def test_the_producer_stamps_the_identity():
    produced = build_record(composed_config(), torch.device("cpu"))
    assert IDENTITY_FIELD in produced
    assert isinstance(produced[IDENTITY_FIELD], float) and produced[IDENTITY_FIELD] > 0


@pytest.mark.xfail(
    strict=True,
    reason="lane-b: `report.load_artifacts` falls back to `path.stat().st_mtime`, "
    "which is the download time in a clean clone and not campaign order",
)
def test_an_artifact_without_the_identity_is_refused_rather_than_dated_by_its_file(tmp_path):
    write(tmp_path, probe_record(3, recorded_at=_DELETE), pod="pod-a")
    artifacts, skipped = report.load_artifacts(tmp_path)
    assert not artifacts
    assert skipped and IDENTITY_FIELD in skipped[0]


@pytest.mark.xfail(
    strict=True,
    reason="lane-d/lane-b: nothing between the record and the table reads "
    "`grad_norm` or `trainable_params`, so a frozen graph is published as a speed",
)
def test_a_run_that_trained_nothing_is_not_published_as_a_speed_result(tmp_path):
    frozen = record(
        metrics={
            "grad_norm": 0.0,
            "trainable_params": 0,
            "loss_last": 2.8734,
            "step_seconds_p50": 0.1111,
        }
    )
    assert training_verdict(frozen)[0] is False
    write(tmp_path, record(), pod="pod-a", label="valid")
    write(tmp_path, frozen, pod="pod-a", label="frozen")

    document = merged(tmp_path)
    assert "frozen" in document, "the run must be reported, not dropped"
    assert "0.1111" not in document, (
        "a step time is published for a run in which no parameter received a "
        "gradient; that number is the cost of a forward pass, not of training"
    )


@pytest.mark.xfail(
    strict=True,
    reason="lane-b: an OOM record carries no metrics and no probe, so `report.cell` "
    "falls through to `launch_state` and renders the combination as never attempted",
)
def test_oom_is_its_own_result_category(tmp_path):
    write(tmp_path, record(status=STATUS_OOM, metrics=_DELETE), pod="pod-a")
    cell = matrix_cell(merged(tmp_path), "tevatron", "qwen3_5_0_8b")
    assert cell not in (
        report.NOT_ATTEMPTED,
        report.NO_RESULT,
        report.LAUNCH_FAILED,
        report.UNSUPPORTED,
        report.NO_METRICS,
    ), f"the OOM reads as `{cell}` - a spent pod-hour filed as something else"
    assert STATUS_OOM in cell.lower()


@pytest.mark.xfail(
    strict=True,
    reason="lane-b: `report.render_measurements` tables every unprofiled run "
    "together, so transformers 5.5.0 and 5.14.1 are ranked side by side",
)
def test_two_stacks_are_never_ranked_in_one_table(tmp_path):
    write(tmp_path, record(), pod="pod-a", label="on-5-14")
    write(
        tmp_path,
        record(packages={"torch": "2.11.0", "transformers": "5.5.0"}),
        pod="pod-b",
        label="on-5-5",
    )
    document = merged(tmp_path)
    assert "5.14.1" in document and "5.5.0" in document, "neither table says which stack it is"
    for block in tables(document):
        joined = "\n".join(block)
        assert not ("| on-5-14 |" in joined and "| on-5-5 |" in joined), (
            "two stacks share one ranking; the image is a confound and the table "
            "presents it as a result"
        )


@pytest.mark.xfail(
    strict=True,
    reason="lane-b/lane-c: `scripts/report.py` never reads `applied`, so a cell "
    "missing an axis to its framework looks like a cell that ran it",
)
def test_an_axis_the_framework_owns_is_visible_in_the_report(tmp_path):
    write(tmp_path, record(), pod="pod-a", label="canonical")
    document = merged(tmp_path)
    assert "loss.name" in document
    assert FRAMEWORK_OWNED in document or "프레임워크 소유" in document
