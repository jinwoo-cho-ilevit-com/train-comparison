"""Pod lifecycle, experiment manifests, and the reporting that reads their output.

Written after a review found that `is_finished` returned True for a pod that had
not started yet: the sweep would have terminated all eighteen pods seconds after
launch and filed every combination as producing no result.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from trainbench import pods

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import orchestrate  # noqa: E402
import publish_result  # noqa: E402
import report  # noqa: E402

PULLING = {"desiredStatus": "RUNNING", "runtime": None, "lastStatusChange": "started"}
LIVE = {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 12}}
STOPPED = {"desiredStatus": "RUNNING", "runtime": None}
DONE = {"desiredStatus": "EXITED", "runtime": None}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def watcher(responses, clock=None, timeout=600):
    """A watcher reading a scripted sequence of API answers per pod."""
    clock = clock or FakeClock()
    remaining = {pod_id: list(states) for pod_id, states in responses.items()}

    def get_pod(pod_id):
        state = remaining[pod_id].pop(0)
        if isinstance(state, Exception):
            raise state
        return state

    return pods.PodWatch(
        timeout_seconds=timeout,
        get_pod=get_pod,
        clock=clock,
        sleep=lambda seconds: clock.advance(seconds),
        poll_seconds=10,
    )


# --- D2: a pod that has not started is not a pod that has finished -------------


def test_a_pod_still_pulling_its_image_is_not_finished():
    assert pods.observe("p", lambda _: PULLING) == pods.PENDING
    assert pods.is_finished(pods.PENDING, ever_ran=False) is False


def test_a_pod_whose_runtime_disappeared_is_finished():
    assert pods.is_finished(pods.PENDING, ever_ran=True) is True


def test_an_exited_desired_status_is_finished_even_without_a_transition():
    assert pods.observe("p", lambda _: DONE) == pods.EXITED
    assert pods.is_finished(pods.EXITED, ever_ran=False) is True


def test_a_pod_the_api_forgot_is_finished():
    assert pods.observe("p", lambda _: None) == pods.GONE
    assert pods.is_finished(pods.GONE, ever_ran=False) is True


def test_the_whole_sweep_is_not_terminated_seconds_after_launch():
    """The defect end to end: eighteen pods, all still pulling."""
    ids = [f"pod-{i}" for i in range(18)]
    watch = watcher({pid: [PULLING] for pid in ids})
    for pid in ids:
        watch.track(pid)
    assert watch.poll() == []
    assert sorted(watch.watching) == sorted(ids)


# --- a failed API call is not an answer ---------------------------------------


def test_an_api_error_is_an_unknown_sentinel_not_a_crash():
    def explode(_):
        raise RuntimeError("502 from the control plane")

    assert pods.observe("p", explode) == pods.UNKNOWN


@pytest.mark.parametrize("payload", [["not", "a", "mapping"], "RUNNING", 7])
def test_a_payload_of_the_wrong_shape_is_a_non_answer_not_a_crash(payload):
    """It raised through poll into a main with no handler, killing a live sweep."""
    assert pods.observe("p", lambda _: payload) == pods.UNKNOWN


def test_an_unparseable_reading_does_not_take_the_orchestrator_down_with_it():
    """The path that mattered: the exception escaped poll, not observe."""
    watch = watcher({"p": [["not", "a", "mapping"], PULLING]})
    watch.track("p")
    assert watch.poll() == []
    assert watch.watching == ["p"]


def test_an_unknown_status_never_reads_as_finished():
    assert pods.is_finished(pods.UNKNOWN, ever_ran=True) is False
    assert pods.is_finished(pods.UNKNOWN, ever_ran=False) is False


def test_a_pod_survives_a_transient_api_failure_and_still_completes():
    clock = FakeClock()
    watch = watcher({"p": [RuntimeError("502"), LIVE, STOPPED, STOPPED]}, clock=clock)
    watch.track("p")
    assert watch.poll() == []
    outcomes = watch.wait_for_any()
    assert [o.reason for o in outcomes] == [pods.REASON_STOPPED]
    assert outcomes[0].ever_ran is True


# --- an inferred ending has to repeat before it is acted on --------------------


def test_one_reading_of_a_vanished_pod_does_not_end_the_watch():
    """`gone` is an absence, and so is the transient read that mimics it."""
    watch = watcher({"p": [None, LIVE]})
    watch.track("p")
    assert watch.poll() == []
    assert watch.watching == ["p"]
    assert watch.poll() == []


def test_a_pod_that_stays_gone_is_still_recorded_as_gone():
    watch = watcher({"p": [None, None]})
    watch.track("p")
    watch.poll()
    outcomes = watch.poll()
    assert [o.reason for o in outcomes] == [pods.REASON_GONE]
    assert watch.watching == []


def test_a_runtime_that_blinks_does_not_terminate_a_running_pod():
    watch = watcher({"p": [LIVE, STOPPED, LIVE, LIVE]})
    watch.track("p")
    for _ in range(4):
        assert watch.poll() == []
    assert watch.watching == ["p"]


def test_a_status_the_api_states_outright_needs_no_second_reading():
    """`desiredStatus: EXITED` is a statement, not an inference from an absence."""
    watch = watcher({"p": [DONE]})
    watch.track("p")
    assert [o.reason for o in watch.poll()] == [pods.REASON_EXITED]


# --- per-pod deadlines ---------------------------------------------------------


def test_each_pod_is_charged_only_its_own_waiting_time():
    clock = FakeClock()
    watch = watcher({"early": [PULLING] * 20, "late": [PULLING] * 20}, clock=clock, timeout=100)
    watch.track("early")
    clock.advance(90)
    watch.track("late")
    clock.advance(20)
    outcomes = watch.poll()
    assert [o.pod_id for o in outcomes] == ["early"]
    assert outcomes[0].reason == pods.REASON_TIMEOUT
    # The pod launched last still has its full allowance left.
    assert watch.watching == ["late"]


def test_waiting_always_terminates_even_when_nothing_ever_starts():
    clock = FakeClock()
    watch = watcher({"p": [PULLING] * 100}, clock=clock, timeout=50)
    watch.track("p")
    outcomes = watch.wait_for_any()
    assert [o.reason for o in outcomes] == [pods.REASON_TIMEOUT]
    assert watch.watching == []


def test_watching_nothing_returns_immediately():
    assert pods.PodWatch(timeout_seconds=1, get_pod=lambda _: None).wait_for_any() == []


# --- experiment manifests ------------------------------------------------------


def manifest(tmp_path, name, **overrides):
    body = {
        "phase": "phase0",
        "model": "qwen3_5_0_8b",
        "framework": "native",
        "run": "probe",
        "baseline": "none",
    }
    body.update(overrides)
    path = tmp_path / f"{name}.yaml"
    path.write_text(json.dumps(body))
    return path


def test_every_shipped_manifest_loads():
    experiments = orchestrate.load_experiments()
    assert experiments, "configs/experiment holds no manifest"
    assert all(e.framework in orchestrate.IMAGE_SUFFIX for e in experiments)


def test_every_shipped_manifest_composes_and_validates():
    """A variant that cannot be built must fail here, not on a billing GPU."""
    baselines = orchestrate.load_baselines(orchestrate.BASELINES_PATH)
    for exp in orchestrate.load_experiments():
        runs = orchestrate.plan_runs(exp, baselines)
        assert runs
        for run in runs:
            assert run.config["run"]["purpose"] in ("probe", "timing", "quality", "profile")


def test_a_measuring_pod_must_name_a_canonical_baseline(tmp_path):
    path = manifest(tmp_path, "phase2-x", run="timing", baseline="none")
    with pytest.raises(orchestrate.ManifestError, match="canonical baseline"):
        orchestrate.load_manifest(path)


def test_the_baseline_is_the_same_workload_on_every_pod():
    """It is a fixed reference, so it must not vary with the pod's own model."""
    baselines = orchestrate.load_baselines(orchestrate.BASELINES_PATH)
    measuring = [e for e in orchestrate.load_experiments() if e.baseline != orchestrate.NO_BASELINE]
    assert measuring, "no manifest exercises the canonical baseline"
    dumped = {
        json.dumps(orchestrate.plan_runs(e, baselines)[0].config, sort_keys=True) for e in measuring
    }
    assert len(dumped) == 1


def test_a_probe_manifest_may_not_name_a_canonical_baseline(tmp_path):
    """It validated, and the pod then ran the baseline's model in this pod's image."""
    path = manifest(tmp_path, "phase0-x", run="probe", baseline="canonical")
    with pytest.raises(orchestrate.ManifestError, match="baseline: none"):
        orchestrate.load_manifest(path)


def test_the_pod_executes_its_own_run_and_not_the_baseline():
    """`runs[0]` was the baseline, which names its own model and framework."""
    exp = orchestrate.Experiment(
        name="phase2-loss-gemma4_e2b",
        phase="phase2",
        model="gemma4_e2b",
        framework="native",
        purpose="timing",
        baseline="canonical",
        axis="loss",
        settings={"mnrl": ["loss=mnrl"], "cached_mnrl": ["loss=cached_mnrl"]},
    )
    runs = orchestrate.plan_runs(exp, orchestrate.load_baselines(orchestrate.BASELINES_PATH))
    assert runs[0].role == orchestrate.ROLE_BASELINE
    assert runs[0].config["model"]["name"] != exp.model
    args = orchestrate.argparse.Namespace(
        result_repo="acct/results",
        infisical_env="dev",
        infisical_project_id="project-id",
        timeout_minutes=60,
    )
    env = orchestrate.pod_env(exp, runs, "img", "sha256:abc", "token", "cafe1234", args)
    executed = json.loads(env["TRAINBENCH_CONFIG_JSON"])
    assert executed["model"]["name"] == exp.model
    assert executed["framework"]["name"] == exp.framework
    assert executed["run"]["purpose"] == exp.purpose


def test_the_plan_carries_a_resolved_config_for_every_setting():
    """No pod image has Hydra, so override strings alone leave a sweep unable to run."""
    exp = orchestrate.Experiment(
        name="phase2-loss-gemma4_e2b",
        phase="phase2",
        model="gemma4_e2b",
        framework="native",
        purpose="timing",
        baseline="canonical",
        axis="loss",
        settings={"mnrl": ["loss=mnrl"], "cached_mnrl": ["loss=cached_mnrl"]},
    )
    runs = orchestrate.plan_runs(exp, orchestrate.load_baselines(orchestrate.BASELINES_PATH))
    plan = [r.summary() for r in runs]
    assert len(plan) == 3
    assert all(entry["config"]["run"]["purpose"] for entry in plan)
    assert {entry["config"]["loss"]["name"] for entry in plan[1:]} == {"mnrl", "cached_mnrl"}
    assert [entry["role"] for entry in plan] == [
        orchestrate.ROLE_BASELINE,
        orchestrate.ROLE_EXPERIMENT,
        orchestrate.ROLE_EXPERIMENT,
    ]


def test_a_typo_in_a_manifest_is_refused_not_ignored(tmp_path):
    path = manifest(tmp_path, "phase0-x", gpu_type="NVIDIA B200")
    with pytest.raises(orchestrate.ManifestError, match="unknown key"):
        orchestrate.load_manifest(path)


def test_settings_without_an_axis_are_refused(tmp_path):
    path = manifest(tmp_path, "phase0-x", settings={"a": ["loss=mnrl"]})
    with pytest.raises(orchestrate.ManifestError, match="axis"):
        orchestrate.load_manifest(path)


def test_one_axis_split_across_two_pods_is_refused():
    def experiment(name):
        return orchestrate.Experiment(
            name=name,
            phase="phase2",
            model="gemma4_e2b",
            framework="native",
            purpose="timing",
            baseline="canonical",
            axis="attn",
        )

    with pytest.raises(orchestrate.ManifestError, match="split across pods"):
        orchestrate.check_axis_not_split([experiment("a"), experiment("b")])


def test_an_unknown_baseline_name_is_refused():
    exp = orchestrate.Experiment(
        name="x",
        phase="phase2",
        model="gemma4_e2b",
        framework="native",
        purpose="timing",
        baseline="does_not_exist",
    )
    with pytest.raises(orchestrate.ManifestError, match="unknown baseline"):
        orchestrate.plan_runs(exp, {})


# --- D5: the pod gets what it needs, and nothing it must not have --------------


def pod_environment(**extra):
    exp = orchestrate.load_experiments()[0]
    runs = orchestrate.plan_runs(exp, orchestrate.load_baselines(orchestrate.BASELINES_PATH))
    args = orchestrate.argparse.Namespace(
        result_repo="acct/results",
        infisical_env="dev",
        infisical_project_id="project-id",
        timeout_minutes=60,
        **extra,
    )
    return orchestrate.pod_env(exp, runs, "img", "sha256:abc", "token-value", "cafe1234", args)


def test_the_pod_receives_the_token_whose_absence_looked_like_an_unsupported_model():
    env = pod_environment()
    assert env["INFISICAL_TOKEN"] == "token-value"
    assert env["INFISICAL_PROJECT_ID"] == "project-id"


def test_the_pod_receives_the_commit_and_digest_a_result_is_traced_by():
    env = pod_environment()
    assert env["TRAINBENCH_GIT_COMMIT"] == "cafe1234"
    assert env["TRAINBENCH_IMAGE_DIGEST"] == "sha256:abc"
    assert env["TRAINBENCH_IMAGE"] == "img"


def test_the_account_wide_key_never_reaches_an_experiment_pod():
    assert "RUNPOD_API_KEY" not in pod_environment()
    assert "GITHUB_TOKEN" not in pod_environment()


def test_the_pod_kills_itself_before_the_orchestrator_gives_up():
    env = pod_environment()
    assert int(env["TRAINBENCH_TIMEOUT_SECONDS"]) < 60 * 60


@pytest.mark.parametrize("minutes", [1, 2, 3, 4, 5, 10, 59, 60, 61, 120, 600])
def test_no_deadline_lets_the_pod_outlive_its_watcher(minutes):
    """`max(120, minutes*60 - 120)` inverted below three minutes: the pod got 120s
    against the orchestrator's 60, so the watcher gave up first and the pod billed on.
    Only the 60-minute default was pinned, which is where it happened to be right."""
    args = orchestrate.argparse.Namespace(timeout_minutes=minutes)
    pod = orchestrate.pod_timeout_seconds(args)
    assert 0 < pod < minutes * 60


def test_a_deadline_of_zero_minutes_is_refused():
    with pytest.raises(orchestrate.argparse.ArgumentTypeError):
        orchestrate.positive_minutes("0")


# --- publishing ----------------------------------------------------------------


def resolved_config():
    return {"framework": {"name": "native"}, "model": {"name": "gemma4_e2b"}}


def test_a_run_that_produced_nothing_still_produces_a_traceable_record(monkeypatch):
    monkeypatch.setenv("TRAINBENCH_GIT_COMMIT", "cafe1234")
    record = publish_result.fallback_record(resolved_config(), "probe died")
    assert record["git_commit"] == "cafe1234"
    assert record["config"] == resolved_config()
    assert record["status"] == "no_result"
    assert record["probe"]["checks"][0]["ok"] is False


def test_a_started_record_carries_the_fields_that_make_it_traceable(monkeypatch):
    monkeypatch.setenv("TRAINBENCH_GIT_COMMIT", "cafe1234")
    record = publish_result.started_record(resolved_config())
    assert {"git_commit", "config", "recorded_at"} <= set(record)


def test_a_missing_commit_is_recorded_as_unknown_not_omitted(monkeypatch):
    monkeypatch.delenv("TRAINBENCH_GIT_COMMIT", raising=False)
    record = publish_result.provenance(resolved_config())
    assert record["git_commit"] == "unknown"
    assert record["git_source"] == "unavailable"


def test_an_upload_is_retried_before_it_is_given_up_on():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("429")
        return "ok"

    assert publish_result.with_retry(flaky, "upload", sleep=lambda _: None) == "ok"
    assert len(attempts) == 3


def test_a_permanently_failing_upload_raises_rather_than_reporting_success():
    def broken():
        raise ConnectionError("429")

    with pytest.raises(RuntimeError, match="failed after"):
        publish_result.with_retry(broken, "upload", sleep=lambda _: None)


# --- reporting -----------------------------------------------------------------


def write_artifact(root, framework, model, pod, name, payload):
    path = root / "results" / framework / model / pod / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def probe_payload(framework, model, checks, recorded_at=1.0, **extra):
    return {
        "config": {"framework": {"name": framework}, "model": {"name": model}},
        "recorded_at": recorded_at,
        "probe": {"framework": framework, "model": model, "checks": checks, **extra},
    }


def test_an_unreadable_file_is_skipped_not_fatal(tmp_path):
    write_artifact(
        tmp_path,
        "native",
        "gemma4_e2b",
        "a",
        "result.json",
        probe_payload("native", "gemma4_e2b", [{"name": "load", "ok": True}]),
    )
    broken = tmp_path / "results" / "native" / "gemma4_e2b" / "b" / "result.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{ truncated")
    artifacts, skipped = report.load_artifacts(tmp_path)
    assert len(artifacts) == 1
    assert len(skipped) == 1


def test_the_newest_artifact_wins_and_the_older_one_is_named(tmp_path):
    write_artifact(
        tmp_path,
        "native",
        "gemma4_e2b",
        "old",
        "result.json",
        probe_payload(
            "native", "gemma4_e2b", [{"name": "load", "ok": False, "error_type": "OSError"}], 1.0
        ),
    )
    write_artifact(
        tmp_path,
        "native",
        "gemma4_e2b",
        "new",
        "result.json",
        probe_payload("native", "gemma4_e2b", [{"name": "load", "ok": True}], 99.0),
    )
    artifacts, _ = report.load_artifacts(tmp_path)
    chosen, duplicates = report.newest_per_combination(artifacts)
    assert chosen[("native", "gemma4_e2b")].timestamp == 99.0
    assert len(duplicates) == 1
    assert "old" in duplicates[0]


def test_a_documented_limitation_does_not_read_as_a_broken_cell(tmp_path):
    path = write_artifact(
        tmp_path,
        "unsloth",
        "qwen3_vl_emb_2b",
        "a",
        "result.json",
        probe_payload(
            "unsloth",
            "qwen3_vl_emb_2b",
            [
                {"name": "load", "ok": True},
                {"name": "vlm", "ok": False, "expected_failure": True, "error_type": "ValueError"},
            ],
        ),
    )
    artifacts, _ = report.load_artifacts(path.parents[3])
    chosen, _ = report.newest_per_combination(artifacts)
    assert report.cell(chosen[("unsloth", "qwen3_vl_emb_2b")], None).startswith("OK")


def test_a_documented_limitation_that_started_passing_is_surfaced(tmp_path):
    path = write_artifact(
        tmp_path,
        "unsloth",
        "qwen3_vl_emb_2b",
        "a",
        "result.json",
        probe_payload(
            "unsloth",
            "qwen3_vl_emb_2b",
            [{"name": "vlm", "ok": True, "expected_failure": True}],
            unexpected_passes=["vlm"],
        ),
    )
    artifacts, _ = report.load_artifacts(path.parents[3])
    chosen, _ = report.newest_per_combination(artifacts)
    assert report.unexpected_passes(chosen[("unsloth", "qwen3_vl_emb_2b")]) == ["vlm"]
    rendered = report.render(chosen, {}, [], [])
    assert "지원 매트릭스가 틀렸다" in rendered


def test_a_cell_with_nothing_but_documented_limits_is_unsupported_not_ok(tmp_path):
    """Unsloth rejecting a VLM at load leaves no other check to grade."""
    path = write_artifact(
        tmp_path,
        "unsloth",
        "qwen3_vl_emb_2b",
        "a",
        "result.json",
        probe_payload(
            "unsloth",
            "qwen3_vl_emb_2b",
            [{"name": "load", "ok": False, "expected_failure": True, "error_type": "ValueError"}],
        ),
    )
    artifacts, _ = report.load_artifacts(path.parents[3])
    chosen, _ = report.newest_per_combination(artifacts)
    assert report.cell(chosen[("unsloth", "qwen3_vl_emb_2b")], None).startswith(report.UNSUPPORTED)


def test_a_committed_report_names_no_local_path(tmp_path):
    broken = tmp_path / "results" / "native" / "gemma4_e2b" / "a" / "result.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{ truncated")
    _, skipped = report.load_artifacts(tmp_path)
    assert skipped == ["results/native/gemma4_e2b/a/result.json: JSONDecodeError"]


def test_launched_but_silent_is_not_the_same_as_never_attempted():
    assert report.cell(None, [{"pod_id": "abc", "launch_error": None}]) == report.NO_RESULT
    assert report.cell(None, None) == report.NOT_ATTEMPTED
    assert report.cell(None, []) == report.NOT_ATTEMPTED
    failed = [{"pod_id": None, "launch_error": "no capacity"}]
    assert report.cell(None, failed) == report.LAUNCH_FAILED


# --- one cell, more than one manifest -------------------------------------------


def ledger_file(tmp_path, entries):
    path = tmp_path / "orchestrate.json"
    path.write_text(json.dumps({"experiments": entries}))
    return path


def test_every_ledger_entry_survives_a_combination_named_twice(tmp_path):
    """Three of twenty-one entries were dropped by keying a dict on the combination."""
    entries = [
        {"experiment": "phase0-native-gemma4_e2b", "framework": "native", "model": "gemma4_e2b"},
        {"experiment": "phase2-loss-gemma4_e2b", "framework": "native", "model": "gemma4_e2b"},
        {"experiment": "phase0-unsloth-gemma4_e2b", "framework": "unsloth", "model": "gemma4_e2b"},
    ]
    ledger = report.load_ledger(ledger_file(tmp_path, entries))
    kept = sum(len(group) for group in ledger.values())
    assert kept == len(entries)
    assert [e["experiment"] for e in ledger[("native", "gemma4_e2b")]] == [
        "phase0-native-gemma4_e2b",
        "phase2-loss-gemma4_e2b",
    ]


def test_a_launched_probe_is_not_relabelled_by_a_sweep_that_never_started(tmp_path):
    """The reported symptom: a spent pod-hour reading as a pod that never launched."""
    entries = [
        {
            "experiment": "phase0-native-gemma4_e2b",
            "framework": "native",
            "model": "gemma4_e2b",
            "pod_id": "pod-abc",
            "launch_error": None,
        },
        {
            "experiment": "phase2-loss-gemma4_e2b",
            "framework": "native",
            "model": "gemma4_e2b",
            "pod_id": None,
            "launch_error": "purpose 'timing' has no entry point in the pod image yet",
        },
    ]
    ledger = report.load_ledger(ledger_file(tmp_path, entries))
    assert report.cell(None, ledger[("native", "gemma4_e2b")]) == report.NO_RESULT


def test_the_whole_shipped_sweep_keeps_every_cell_it_launched(tmp_path):
    """End to end over the real manifests, which is where the 21 -> 18 was found."""
    entries = []
    for exp in orchestrate.load_experiments():
        runnable = exp.purpose in orchestrate.RUNNABLE_PURPOSES
        entries.append(
            {
                **exp.summary(),
                "pod_id": f"pod-{exp.name}" if runnable else None,
                "launch_error": None if runnable else "no entry point yet",
            }
        )
    ledger = report.load_ledger(ledger_file(tmp_path, entries))
    assert sum(len(group) for group in ledger.values()) == len(entries)
    probed = {
        (e.framework, e.model) for e in orchestrate.load_experiments() if e.purpose == "probe"
    }
    for key in probed:
        assert report.cell(None, ledger[key]) == report.NO_RESULT


def test_a_later_run_without_probe_checks_does_not_take_over_the_cell(tmp_path):
    """A timing artifact is newer and answers a different question."""
    write_artifact(
        tmp_path,
        "native",
        "gemma4_e2b",
        "probe-pod",
        "result.json",
        probe_payload("native", "gemma4_e2b", [{"name": "load", "ok": True}], recorded_at=1.0),
    )
    timing = {
        "config": {"framework": {"name": "native"}, "model": {"name": "gemma4_e2b"}},
        "recorded_at": 99.0,
        "metrics": {"steps_per_second": 1.0},
    }
    write_artifact(tmp_path, "native", "gemma4_e2b", "timing-pod", "result.json", timing)
    artifacts, _ = report.load_artifacts(tmp_path)
    chosen, duplicates = report.newest_per_combination(artifacts)
    assert chosen[("native", "gemma4_e2b")].path.parts[-2] == "probe-pod"
    assert report.cell(chosen[("native", "gemma4_e2b")], None).startswith("OK")
    assert len(duplicates) == 1


# --- the document the merge writes into ------------------------------------------


def test_a_hand_written_matrix_above_the_marker_stops_the_merge():
    """Two tables, same heading, contradicting cells, hand-written one read first."""
    existing = (
        "# 지원 매트릭스\n\n"
        f"## {report.MATRIX_HEADING}\n\n"
        "| | gemma4_e2b |\n|---|---|\n| native | **OK (7/7)** |\n\n"
        f"{report.MARKER}\n\n{report.GENERATED_HEADING}\n"
    )
    with pytest.raises(ValueError, match=report.MATRIX_HEADING):
        report.document_head(existing)


def test_a_second_generated_section_stops_the_merge():
    existing = f"# 문서\n\n{report.MARKER}\n\n어제 생성분\n\n{report.MARKER}\n\n오늘 생성분\n"
    with pytest.raises(ValueError, match="markers"):
        report.document_head(existing)


def test_prose_above_the_marker_survives_the_merge():
    existing = f"# 문서\n\n손으로 쓴 분석.\n\n{report.MARKER}\n\n낡은 생성분\n"
    assert report.document_head(existing) == "# 문서\n\n손으로 쓴 분석."


def test_the_merge_refuses_rather_than_writing_a_self_contradicting_document(tmp_path):
    matrix = tmp_path / "support-matrix.md"
    matrix.write_text(f"# 지원 매트릭스\n\n## {report.MATRIX_HEADING}\n\n| | a |\n|---|---|\n")
    before = matrix.read_text()
    results = tmp_path / "results"
    results.mkdir()
    ledger = ledger_file(
        tmp_path,
        [{"experiment": "x", "framework": "native", "model": "gemma4_e2b", "pod_id": "p"}],
    )
    code = report.main(
        ["--results", str(results), "--ledger", str(ledger), "--matrix", str(matrix)]
    )
    assert code == 2
    assert matrix.read_text() == before


def test_a_fallback_record_reads_as_launched_and_silent(tmp_path):
    payload = publish_result.fallback_record(resolved_config(), "probe died")
    path = write_artifact(tmp_path, "native", "gemma4_e2b", "a", "result.json", payload)
    artifacts, _ = report.load_artifacts(path.parents[3])
    chosen, _ = report.newest_per_combination(artifacts)
    assert report.cell(chosen[("native", "gemma4_e2b")], None) == report.NO_RESULT


# --- the pod entrypoint ----------------------------------------------------------

ENTRYPOINT = REPO / "docker" / "entrypoint.sh"


def run_entrypoint(tmp_path, env):
    """Run the real entrypoint with a stub PATH, so nothing leaves the machine.

    `uv`, `infisical` and `python3` are replaced by a script that logs its argv and
    succeeds; the assertions are about control flow, which is where the defect was.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    for name in ("uv", "infisical", "python3", "timeout"):
        stub = bin_dir / name
        stub.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "{calls}"\nexit 0\n')
        stub.chmod(0o755)
    result_dir = tmp_path / "result"
    proc = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "TRAINBENCH_RESULT_DIR": str(result_dir),
            "TRAINBENCH_RESULT_REPO": "acct/results",
            **env,
        },
    )
    logged = calls.read_text().splitlines() if calls.exists() else []
    return proc, logged


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_a_pod_with_no_env_dir_still_puts_itself_on_record(tmp_path):
    """`cd "${TRAINBENCH_ENV_DIR}"` aborted at the expansion under `set -u`, so the
    `||` guard never ran, its message was unreachable, the exit code was 1 instead
    of 2 — and, sitting above the announce, it uploaded nothing at all."""
    proc, logged = run_entrypoint(
        tmp_path,
        {"TRAINBENCH_CONFIG_JSON": json.dumps(resolved_config())},
    )
    assert proc.returncode == 2, proc.stderr
    assert "== announce ==" in proc.stdout
    assert "<unset> is not a directory" in proc.stderr
    assert any("--mode start" in line for line in logged)
    assert any("--mode fallback" in line for line in logged)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_a_pod_with_no_config_has_nothing_to_name_the_combination_by(tmp_path):
    proc, logged = run_entrypoint(tmp_path, {})
    assert proc.returncode == 2
    assert "TRAINBENCH_CONFIG_JSON is not set" in proc.stderr
    assert logged == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_a_probe_run_reaches_the_run_and_the_publish(tmp_path):
    env_dir = tmp_path / "envs" / "native"
    env_dir.mkdir(parents=True)
    proc, logged = run_entrypoint(
        tmp_path,
        {
            "TRAINBENCH_CONFIG_JSON": json.dumps(resolved_config()),
            "TRAINBENCH_ENV_DIR": str(env_dir),
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert any("verify_env.py" in line for line in logged)
    assert any("--mode start" in line for line in logged)


def test_a_started_file_alone_means_the_pod_ran_and_said_nothing(tmp_path):
    payload = publish_result.started_record(resolved_config())
    path = write_artifact(tmp_path, "native", "gemma4_e2b", "a", "started.json", payload)
    artifacts, _ = report.load_artifacts(path.parents[3])
    chosen, _ = report.newest_per_combination(artifacts)
    assert report.cell(chosen[("native", "gemma4_e2b")], None) == report.NO_RESULT
