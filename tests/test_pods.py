"""Pod lifecycle, experiment manifests, and the reporting that reads their output.

Written after a review found that `is_finished` returned True for a pod that had
not started yet: the sweep would have terminated all eighteen pods seconds after
launch and filed every combination as producing no result.
"""

from __future__ import annotations

import json
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


def test_an_unknown_status_never_reads_as_finished():
    assert pods.is_finished(pods.UNKNOWN, ever_ran=True) is False
    assert pods.is_finished(pods.UNKNOWN, ever_ran=False) is False


def test_a_pod_survives_a_transient_api_failure_and_still_completes():
    clock = FakeClock()
    watch = watcher({"p": [RuntimeError("502"), LIVE, STOPPED]}, clock=clock)
    watch.track("p")
    assert watch.poll() == []
    outcomes = watch.wait_for_any()
    assert [o.reason for o in outcomes] == [pods.REASON_STOPPED]
    assert outcomes[0].ever_ran is True


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
    ledger = {("native", "gemma4_e2b"): {"pod_id": "abc", "launch_error": None}}
    assert report.cell(None, ledger[("native", "gemma4_e2b")]) == report.NO_RESULT
    assert report.cell(None, None) == report.NOT_ATTEMPTED
    failed = {"pod_id": None, "launch_error": "no capacity"}
    assert report.cell(None, failed) == report.LAUNCH_FAILED


def test_a_fallback_record_reads_as_launched_and_silent(tmp_path):
    payload = publish_result.fallback_record(resolved_config(), "probe died")
    path = write_artifact(tmp_path, "native", "gemma4_e2b", "a", "result.json", payload)
    artifacts, _ = report.load_artifacts(path.parents[3])
    chosen, _ = report.newest_per_combination(artifacts)
    assert report.cell(chosen[("native", "gemma4_e2b")], None) == report.NO_RESULT


def test_a_started_file_alone_means_the_pod_ran_and_said_nothing(tmp_path):
    payload = publish_result.started_record(resolved_config())
    path = write_artifact(tmp_path, "native", "gemma4_e2b", "a", "started.json", payload)
    artifacts, _ = report.load_artifacts(path.parents[3])
    chosen, _ = report.newest_per_combination(artifacts)
    assert report.cell(chosen[("native", "gemma4_e2b")], None) == report.NO_RESULT
