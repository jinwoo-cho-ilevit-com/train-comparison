"""What the merge does with a measurement, and with a pod that cannot be trusted.

The probe matrix was the only thing `scripts/report.py` could read. A timing run
that produced every figure arrived as `결과 없음(기동됨)` — "launched, produced
nothing" — because the merge looked for probe checks and found none, and the
cross-pod deviation rule that AGENTS.md and PLAN.md both state existed nowhere in
the repository as code.

The fixtures build their figures with the real `metrics.summarise`, not a
hand-written dict. A summary key renamed in one place and read in the other is
exactly how a number goes missing from a report that still renders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trainbench.metrics import summarise

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import report  # noqa: E402

BASELINE_LABEL = "baseline-canonical"


def timing_metrics(step_seconds: float, *, discard: int = 2, dropped: float = 0.0, **over):
    """A summary shaped by the code that writes real ones.

    Every duration is identical so p50, p95 and the mean are all `step_seconds`:
    the gate's arithmetic is what is under test, not the percentile's.
    """
    summary = summarise(
        [step_seconds] * (10 + discard),
        discard=discard,
        rows_per_step=64.0,
        tokens_per_step=8192.0,
        peak_bytes=21474836480,
        extra_counts={
            "samples": 32.0,
            "padded_tokens": 9000.0,
            "images": 4.0,
            "images_dropped": dropped,
        },
        totals={"images_read_total": 40, "images_dropped_total": int(dropped * 10)},
    )
    summary["profiled"] = False
    summary.update(over)
    return summary


def timing_payload(framework, model, pod, metrics=None, purpose="timing"):
    payload = {
        "config": {
            "framework": {"name": framework},
            "model": {"name": model},
            "run": {"purpose": purpose},
        },
        "host": {"runpod_pod_id": pod},
        "packages": {"torch": "2.9.0", "transformers": "4.57.0"},
    }
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def probe_payload(framework, model, checks, recorded_at=1.0, **extra):
    return {
        "config": {
            "framework": {"name": framework},
            "model": {"name": model},
            "run": {"purpose": "probe"},
        },
        "recorded_at": recorded_at,
        "packages": {"torch": "2.9.0", "transformers": "4.57.0"},
        "probe": {"framework": framework, "model": model, "checks": checks, **extra},
    }


def write(root, framework, model, pod, payload, label=None, name=report.RESULT_NAME):
    """One artifact where `publish_result` would have put it."""
    directory = root / "results" / framework / model / pod
    if label is not None:
        directory = directory / label
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload))
    return path


def merged(root):
    """Everything `main` does to a results directory, without the document."""
    artifacts, skipped = report.load_artifacts(root)
    lanes = report.split_lanes(artifacts)
    chosen, duplicates = report.newest_per_combination(lanes.matrix)
    rendered = report.render(
        chosen,
        {},
        duplicates,
        skipped,
        measured=lanes.measured,
        baselines=lanes.baselines,
    )
    return lanes, chosen, duplicates, rendered


def baseline_pod(root, pod, step_seconds, **kwargs):
    """A pod that ran the canonical baseline, filed exactly as the sweep files it."""
    return write(
        root,
        "native",
        "qwen3_5_0_8b",
        pod,
        timing_payload("native", "qwen3_5_0_8b", pod, timing_metrics(step_seconds, **kwargs)),
        label=BASELINE_LABEL,
    )


# --- the timing lane -------------------------------------------------------


def test_a_timing_record_reports_its_figures_instead_of_no_result(tmp_path):
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        timing_payload("native", "gemma4_e2b", "podA", timing_metrics(0.5)),
        label="attn-fa3",
    )
    lanes, chosen, _, rendered = merged(tmp_path)

    assert len(lanes.measured) == 1
    # The defect: this artifact used to be ranked into the matrix, where it could
    # only be described by the checks it does not have.
    assert chosen == {}
    matrix_row = next(line for line in rendered.splitlines() if line.startswith("| native |"))
    assert report.NO_RESULT not in matrix_row
    assert "### 측정 결과" in rendered
    rows = [line for line in rendered.splitlines() if line.startswith("| attn-fa3 |")]
    # Whole rows, not substrings: `"10" in rendered` passes on any four-digit
    # duration and would survive every figure being wrong.
    # 32 samples and 8192 tokens per step over 0.5s; 20 GiB of 2**30 bytes.
    assert rows == [
        "| attn-fa3 | podA | native x gemma4_e2b | timing | 0.5000 | 0.5000 | 0.5000 "
        "| 64.00 | 16384.0 | 20.00 | 12/2/10 | 기준선 없음 |",
        "| attn-fa3 | podA | 64.0 | 9000.0 | 8192.0 | 4.00 | 0.00 | 0 | 없음 |",
    ]


def test_a_measuring_run_with_no_metrics_is_not_passed_over_silently(tmp_path):
    """The same record with `metrics` removed. A merge that renders it as a blank
    row, or drops it, turns a spent pod-hour into an absence nobody looks at."""
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        timing_payload("native", "gemma4_e2b", "podA", metrics=None),
        label="attn-fa3",
    )
    lanes, _, _, rendered = merged(tmp_path)

    assert len(lanes.measured) == 1
    assert report.NO_METRICS in rendered
    assert "attn-fa3" in rendered
    # Nothing invented in the figure table: the table is not rendered at all.
    assert "| 런 | 파드 | 프레임워크 x 모델 | 목적 |" not in rendered


def test_the_metric_definitions_travel_into_the_document(tmp_path):
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        timing_payload("native", "gemma4_e2b", "podA", timing_metrics(0.5)),
        label="attn-fa3",
    )
    _, _, _, rendered = merged(tmp_path)

    assert "### 지표 정의" in rendered
    # The two definitions a reader gets wrong without them, verbatim from the
    # summary the run carried.
    assert "instruction_prompt" in rendered
    assert "twice `samples`" in rendered
    assert "no per-model FLOP formula" in rendered  # why MFU is empty


def test_a_run_whose_definitions_disagree_with_another_is_named(tmp_path):
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        timing_payload("native", "gemma4_e2b", "podA", timing_metrics(0.5)),
        label="old",
    )
    drifted = timing_metrics(0.5)
    drifted["metric_definitions"] = {**drifted["metric_definitions"], "tokens": "everything"}
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        timing_payload("native", "gemma4_e2b", "podA", drifted),
        label="new",
    )
    _, _, _, rendered = merged(tmp_path)
    assert "런마다 정의가 다르다" in rendered


def test_images_dropped_says_the_model_read_a_text_only_view(tmp_path):
    write(
        tmp_path,
        "native",
        "qwen3_5_0_8b",
        "podA",
        timing_payload("native", "qwen3_5_0_8b", "podA", timing_metrics(0.5, dropped=4.0)),
        label="dataloader-torch",
    )
    _, _, _, rendered = merged(tmp_path)
    assert "이미지를 텍스트로만 읽은 런" in rendered
    assert "버려진 이미지 40장" in rendered


def test_a_profiled_run_is_kept_out_of_the_comparable_table(tmp_path):
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        timing_payload(
            "native",
            "gemma4_e2b",
            "podA",
            timing_metrics(9.0, profiled=True),
            purpose="profile",
        ),
        label="profile-run",
    )
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podB",
        timing_payload("native", "gemma4_e2b", "podB", timing_metrics(0.5)),
        label="timing-run",
    )
    _, _, _, rendered = merged(tmp_path)

    heading = "#### 프로파일러가 켜진 런 — 비교표에 인용하지 않는다"
    assert heading in rendered
    # The profiled figure appears only after that heading, never in the table a
    # reader compares settings in.
    assert rendered.index("9.0000") > rendered.index(heading)
    assert rendered.index("0.5000") < rendered.index(heading)


def test_a_run_that_reported_nothing_does_not_make_the_cell_read_as_measured(tmp_path):
    """The cell claims a combination trained only if a run reported figures.

    Counting every measuring artifact instead would let a pod that launched and
    produced nothing overwrite the ledger's honest `결과 없음(기동됨)` with `측정
    1건` — the same collapse of "launched" into "answered" the module opens with.
    """
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        timing_payload("native", "gemma4_e2b", "podA", metrics=None),
        label="attn-fa3",
    )
    artifacts, skipped = report.load_artifacts(tmp_path)
    lanes = report.split_lanes(artifacts)
    chosen, duplicates = report.newest_per_combination(lanes.matrix)
    rendered = report.render(
        chosen,
        {("native", "gemma4_e2b"): [{"pod_id": "podA", "launch_error": None}]},
        duplicates,
        skipped,
        measured=lanes.measured,
        baselines=lanes.baselines,
    )
    row = next(line for line in rendered.splitlines() if line.startswith("| native |"))
    assert row.endswith(f"| {report.NO_RESULT} |")
    assert "측정 1건" not in row


def test_a_combination_measured_but_never_probed_does_not_read_as_untried(tmp_path):
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        timing_payload("native", "gemma4_e2b", "podA", timing_metrics(0.5)),
        label="attn-fa3",
    )
    _, _, _, rendered = merged(tmp_path)
    row = next(line for line in rendered.splitlines() if line.startswith("| native |"))
    # Not 미시도: a run trained on this combination, so loading is not in question.
    assert "측정 1건(probe 없음" in row
    assert report.NOT_ATTEMPTED not in row.split("|")[-2]


# --- the baseline gate -----------------------------------------------------


def test_a_pod_beyond_the_limit_is_invalidated_and_its_other_results_say_so(tmp_path):
    baseline_pod(tmp_path, "podA", 1.00)
    baseline_pod(tmp_path, "podB", 1.00)
    baseline_pod(tmp_path, "slowpod", 1.06)  # 6%, twice the limit
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "slowpod",
        timing_payload("native", "gemma4_e2b", "slowpod", timing_metrics(2.0)),
        label="attn-fa3",
    )
    lanes, _, _, rendered = merged(tmp_path)
    verdicts, reference, _ = report.baseline_gate(lanes.baselines, lanes.measured)

    assert reference == pytest.approx(1.00)
    assert verdicts["slowpod"].status == report.POD_INVALID
    assert verdicts["slowpod"].deviation == pytest.approx(0.06)
    assert verdicts["podA"].status == report.POD_OK
    assert verdicts["podB"].status == report.POD_OK
    assert "### 파드 baseline 편차 게이트" in rendered
    gate_row = next(line for line in rendered.splitlines() if line.startswith("| slowpod |"))
    assert gate_row.startswith(f"| slowpod | 1.0600 | 6.00% | {report.POD_INVALID} |")
    # The measured row carries the verdict too; a reader of that table alone
    # still sees that these numbers are not comparable.
    measured_row = next(
        line for line in rendered.splitlines() if "attn-fa3" in line and "2.0000" in line
    )
    assert report.POD_INVALID in measured_row


def test_a_pod_inside_the_limit_is_not_thrown_out(tmp_path):
    baseline_pod(tmp_path, "podA", 1.00)
    baseline_pod(tmp_path, "podB", 1.00)
    baseline_pod(tmp_path, "podC", 1.02)  # 2%, under the threshold
    lanes, _, _, _ = merged(tmp_path)
    verdicts, _, _ = report.baseline_gate(lanes.baselines, [])
    assert {v.status for v in verdicts.values()} == {report.POD_OK}


def test_the_reference_is_the_median_so_one_slow_host_cannot_invalidate_the_rest(tmp_path):
    """Ranking on the first pod would make launch order decide the campaign."""
    baseline_pod(tmp_path, "podA", 2.00)  # the outlier, and alphabetically first
    baseline_pod(tmp_path, "podB", 1.00)
    baseline_pod(tmp_path, "podC", 1.00)
    lanes, _, _, _ = merged(tmp_path)
    verdicts, reference, _ = report.baseline_gate(lanes.baselines, [])
    assert reference == pytest.approx(1.00)
    assert verdicts["podA"].status == report.POD_INVALID
    assert verdicts["podB"].status == report.POD_OK
    assert verdicts["podC"].status == report.POD_OK


def test_the_threshold_is_published_as_uncalibrated(tmp_path):
    baseline_pod(tmp_path, "podA", 1.00)
    baseline_pod(tmp_path, "podB", 1.00)
    _, _, _, rendered = merged(tmp_path)
    assert "미교정" in rendered
    assert "docs/methodology.md" in rendered
    assert "5회 반복" in rendered


def test_one_pod_alone_is_not_reported_as_an_agreement(tmp_path):
    baseline_pod(tmp_path, "podA", 1.00)
    lanes, _, _, rendered = merged(tmp_path)
    _, _, notes = report.baseline_gate(lanes.baselines, [])
    assert any("비교 대상의 부재" in note for note in notes)
    assert "비교 대상의 부재" in rendered


def test_a_pod_that_measured_without_a_baseline_is_flagged_not_assumed_fine(tmp_path):
    baseline_pod(tmp_path, "podA", 1.00)
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "lonepod",
        timing_payload("native", "gemma4_e2b", "lonepod", timing_metrics(0.5)),
        label="attn-fa3",
    )
    lanes, _, _, rendered = merged(tmp_path)
    verdicts, _, _ = report.baseline_gate(lanes.baselines, lanes.measured)
    assert verdicts["lonepod"].status == report.POD_NO_BASELINE
    assert "canonical baseline 레코드가 없다" in rendered


def test_a_baseline_with_no_figures_is_unjudged_rather_than_passing(tmp_path):
    write(
        tmp_path,
        "native",
        "qwen3_5_0_8b",
        "podA",
        timing_payload("native", "qwen3_5_0_8b", "podA", metrics=None),
        label=BASELINE_LABEL,
    )
    lanes, _, _, _ = merged(tmp_path)
    verdicts, reference, notes = report.baseline_gate(lanes.baselines, lanes.measured)
    assert verdicts["podA"].status == report.POD_UNJUDGED
    assert reference is None
    assert notes


def test_a_profiled_baseline_cannot_stand_for_the_hosts_speed(tmp_path):
    baseline_pod(tmp_path, "podA", 1.00)
    write(
        tmp_path,
        "native",
        "qwen3_5_0_8b",
        "podB",
        timing_payload(
            "native",
            "qwen3_5_0_8b",
            "podB",
            timing_metrics(1.00, profiled=True),
            purpose="profile",
        ),
        label=BASELINE_LABEL,
    )
    lanes, _, _, _ = merged(tmp_path)
    verdicts, _, _ = report.baseline_gate(lanes.baselines, lanes.measured)
    assert verdicts["podB"].status == report.POD_UNJUDGED


# --- baselines are filed by pod, not by cell -------------------------------


def test_a_baseline_does_not_displace_the_probe_that_graded_its_cell(tmp_path):
    """Every pod's baseline names `native x qwen3_5_0_8b`, so left in the matrix
    they collide with each other and with that cell's probe."""
    write(
        tmp_path,
        "native",
        "qwen3_5_0_8b",
        "probepod",
        probe_payload("native", "qwen3_5_0_8b", [{"name": "load", "ok": True}]),
    )
    baseline_pod(tmp_path, "podA", 1.00)
    baseline_pod(tmp_path, "podB", 1.00)
    lanes, chosen, duplicates, rendered = merged(tmp_path)

    assert len(lanes.baselines) == 2
    assert list(chosen) == [("native", "qwen3_5_0_8b")]
    assert chosen[("native", "qwen3_5_0_8b")].pod == "probepod"
    # The baselines are not silently discarded as duplicates of the probe.
    assert duplicates == []
    assert "OK (1 checks)" in rendered
    assert "| podA |" in rendered and "| podB |" in rendered


def test_a_baseline_is_attributed_to_its_pod_even_without_a_recorded_pod_id(tmp_path):
    """`RUNPOD_POD_ID` unset falls back to the directory the artifact was filed in,
    which is the pod's own — `publish_result.result_dir_in_repo` puts it there."""
    payload = timing_payload("native", "qwen3_5_0_8b", "unused", timing_metrics(1.0))
    payload["host"] = {"runpod_pod_id": None}
    write(tmp_path, "native", "qwen3_5_0_8b", "podFromPath", payload, label=BASELINE_LABEL)
    artifacts, _ = report.load_artifacts(tmp_path)
    assert artifacts[0].pod == "podFromPath"
    assert artifacts[0].is_baseline


def test_a_results_subtree_and_a_repo_root_resolve_to_the_same_pod(tmp_path):
    baseline_pod(tmp_path, "podA", 1.00)
    from_root, _ = report.load_artifacts(tmp_path)
    from_subtree, _ = report.load_artifacts(tmp_path / "results")
    assert from_root[0].pod == from_subtree[0].pod == "podA"
    assert from_root[0].label == from_subtree[0].label == BASELINE_LABEL


# --- the probe path must not move ------------------------------------------

# The exact bytes `render` produced for probe-only input before the measurement
# and baseline lanes existed. Phase 0 uploads eighteen probes into this path and
# a change there is a regression, not an improvement.
PROBE_ONLY = "\n".join(
    [
        report.MARKER,
        "",
        report.GENERATED_HEADING,
        "",
        "결과 3건, 아티팩트 4건. `미시도`는 pod을 띄운 적이 없는 조합, "
        "`결과 없음(기동됨)`는 띄웠으나 결과 파일이 올라오지 않은 조합, "
        "`미지원(문서화됨)`는 모든 체크가 문서화된 한계였던 조합이다.",
        "",
        "| | qwen3_vl_emb_2b | qwen3_5_0_8b | gemma4_e2b |",
        "|---|---|---|---|",
        "| native | 미시도 | 미시도 | OK (2 checks) |",
        "| unsloth | 미지원(문서화됨) (2건) | 미시도 | 미시도 |",
        "| ms_swift | 미시도 | FAIL load (OSError) | 미시도 |",
        "| sentence_transformers | 미시도 | 미시도 | 미시도 |",
        "| tevatron | 미시도 | 미시도 | 결과 없음(기동됨) |",
        "| axolotl | 미시도 | 결과 없음(기동됨) | 미시도 |",
        "",
        "### 지원 매트릭스가 틀렸다 — 실패할 것으로 표시한 체크가 통과했다",
        "",
        "문서화된 한계가 사라졌다는 뜻이므로, 해당 셀의 근거를 다시 확인해야 한다.",
        "",
        "- **unsloth x qwen3_vl_emb_2b** — vlm",
        "",
        "### 실행 환경별 해석 버전",
        "",
        "| 조합 | torch | transformers | 프레임워크 |",
        "|---|---|---|---|",
        "| ms_swift x qwen3_5_0_8b | 2.9.0 | 4.57.0 | - |",
        "| native x gemma4_e2b | 2.9.0 | 4.57.0 | 4.57.0 |",
        "| tevatron x gemma4_e2b | 2.9.0 | 4.57.0 | - |",
        "| unsloth x qwen3_vl_emb_2b | 2.9.0 | 4.57.0 | - |",
        "",
        "### 실패 상세",
        "",
        "- **ms_swift x qwen3_5_0_8b / load** — OSError",
        "  - `boom`",
        "",
        "### 병합에서 제외한 파일",
        "",
        "- 판독 불가: broken/result.json: JSONDecodeError",
        "",
    ]
)


def test_probe_only_input_renders_byte_for_byte_as_before(tmp_path):
    write(
        tmp_path,
        "native",
        "gemma4_e2b",
        "podA",
        probe_payload(
            "native",
            "gemma4_e2b",
            [
                {"name": "load", "ok": True, "detail": {}},
                {"name": "framework_version", "ok": True, "detail": {"version": "4.57.0"}},
            ],
        ),
    )
    write(
        tmp_path,
        "unsloth",
        "qwen3_vl_emb_2b",
        "podB",
        probe_payload(
            "unsloth",
            "qwen3_vl_emb_2b",
            [
                {
                    "name": "load",
                    "ok": False,
                    "expected_failure": True,
                    "error_type": "ValueError",
                    "detail": {},
                },
                {"name": "vlm", "ok": True, "expected_failure": True, "detail": {}},
            ],
            unexpected_passes=["vlm"],
        ),
    )
    write(
        tmp_path,
        "ms_swift",
        "qwen3_5_0_8b",
        "podC",
        probe_payload(
            "ms_swift",
            "qwen3_5_0_8b",
            [
                {
                    "name": "load",
                    "ok": False,
                    "error_type": "OSError",
                    "error": "boom\nsecond line",
                    "detail": {},
                }
            ],
        ),
    )
    write(
        tmp_path,
        "tevatron",
        "gemma4_e2b",
        "podD",
        probe_payload("tevatron", "gemma4_e2b", []),
        name=report.STARTED_NAME,
    )

    artifacts, skipped = report.load_artifacts(tmp_path)
    lanes = report.split_lanes(artifacts)
    assert lanes.measured == [] and lanes.baselines == []
    chosen, duplicates = report.newest_per_combination(lanes.matrix)
    rendered = report.render(
        chosen,
        {("axolotl", "qwen3_5_0_8b"): [{"pod_id": "podE", "launch_error": None}]},
        duplicates,
        skipped + ["broken/result.json: JSONDecodeError"],
        measured=lanes.measured,
        baselines=lanes.baselines,
    )
    assert rendered == PROBE_ONLY
