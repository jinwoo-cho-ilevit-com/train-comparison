"""Pod lifecycle, experiment manifests, and the reporting that reads their output.

Written after a review found that `is_finished` returned True for a pod that had
not started yet: the sweep would have terminated all eighteen pods seconds after
launch and filed every combination as producing no result.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import urllib.error
from collections import namedtuple
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


# --- what actually goes on the wire when a pod is created ----------------------
#
# The first campaign lost all fifteen launches here while 592 tests passed, because
# every test that touched `create` replaced it with a stub. So these tests inspect
# the request bytes and nothing else pretends to.

# The env of a real launch, and the reason the old path died. `TRAINBENCH_CONFIG_JSON`
# is a whole resolved config document; the rest are shapes that break naive string
# assembly in a different way each.
HOSTILE_ENV = {
    "TRAINBENCH_CONFIG_JSON": json.dumps(
        {"model": {"name": "qwen3_5_0_8b"}, "train": {"lr": 1e-5}, "purpose": "timing"}
    ),
    "TRAINBENCH_PLAN_JSON": json.dumps([{"name": "baseline", "axis": "attn.name"}]),
    "WITH_A_NEWLINE": "first\nsecond",
    "WITH_A_BACKSLASH": r"C:\models\qwen",
    "WITH_A_BARE_QUOTE": 'the operator said "run it"',
}

KEY = "rpa-sentinel-not-a-real-key"


def spec(**overrides):
    fields = {
        "name": "trainbench-phase0-native-qwen3_5_0_8b",
        "image": "ghcr.io/org/trainbench-native@sha256:" + "0" * 64,
        "gpu_type_id": "NVIDIA A100 80GB PCIe",
        "env": dict(HOSTILE_ENV),
    }
    fields.update(overrides)
    return pods.PodSpec(**fields)


def sent(pod_spec=None, reply=None):
    """Create a pod against a transport that records the request instead of sending."""
    captured = []

    def transport(request):
        captured.append(request)
        return reply if reply is not None else {"id": "abc123"}

    pod = pods.create(pod_spec or spec(), transport=transport)
    assert len(captured) == 1
    return captured[0], pod


class FakeResponse:
    def __init__(self, status=201, payload=b'{"id": "abc123"}'):
        self.status = status
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, body):
    return urllib.error.HTTPError(
        f"{pods.REST_BASE_URL}/pods", code, "rejected", {}, io.BytesIO(body)
    )


def test_a_resolved_config_document_survives_the_trip_into_a_pod_env():
    """The assertion this whole change exists for.

    Not "the dict we built has the right keys" — the bytes on the wire parse, and
    every value comes back the way it went in. The previous transport could not
    make this claim for any value containing a quote, and `TRAINBENCH_CONFIG_JSON`
    is nothing but quotes.
    """
    request, _ = sent()
    body = json.loads(request.body)
    assert body["env"] == HOSTILE_ENV
    # And the document is still a document, not a string that merely looks like one.
    assert json.loads(body["env"]["TRAINBENCH_CONFIG_JSON"])["model"]["name"] == "qwen3_5_0_8b"


@pytest.mark.parametrize("value", sorted(HOSTILE_ENV.values()))
def test_every_hostile_env_value_arrives_unaltered(value):
    request, _ = sent(spec(env={"TRAINBENCH_CONFIG_JSON": value}))
    assert json.loads(request.body)["env"]["TRAINBENCH_CONFIG_JSON"] == value


def test_the_sdk_path_this_replaced_still_puts_the_document_in_raw():
    """Why the transport changed, pinned so the claim is checkable rather than told.

    `runpod` builds its mutation with `f'{{ key: "{k}", value: "{v}" }}'`. The
    value's own quotes close the GraphQL string and the remainder becomes syntax —
    the fifteen launches all returned `Syntax Error: Expected ":", found String
    ": {"`, which is a fragment of this very document.

    If this test fails because the SDK started escaping, that removes one of the
    two reasons `create` has its own transport. The other one is `get`: REST
    reports no runtime, so the module would still need both.
    """
    mutations = pytest.importorskip("runpod.api.mutations.pods")
    mutation = mutations.generate_pod_deployment_mutation(
        name="trainbench-phase0-native-qwen3_5_0_8b",
        image_name="ghcr.io/org/trainbench-native",
        gpu_type_id="NVIDIA A100 80GB PCIe",
        env=dict(HOSTILE_ENV),
    )
    document = HOSTILE_ENV["TRAINBENCH_CONFIG_JSON"]
    assert f'value: "{document}"' in mutation
    assert '\\"' not in mutation, "the SDK escaped nothing; the values go in raw"
    # The literal fragment RunPod's parser reported: Expected ":", found String ": {".
    assert '": {"' in mutation


def test_the_request_is_the_documented_create_endpoint():
    request, _ = sent()
    assert request.method == "POST"
    assert request.url == "https://rest.runpod.io/v1/pods"
    assert request.headers["Content-Type"] == "application/json"


def test_the_spec_lands_in_the_documented_field_names():
    request, _ = sent(spec(container_disk_gb=200, data_center_id="EU-RO-1"))
    body = json.loads(request.body)
    assert body["name"] == "trainbench-phase0-native-qwen3_5_0_8b"
    assert body["imageName"].startswith("ghcr.io/org/trainbench-native@sha256:")
    # Plural and a list, unlike the GraphQL `gpuTypeId`/`dataCenterId` they replace.
    assert body["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"]
    assert body["dataCenterIds"] == ["EU-RO-1"]
    assert body["cloudType"] == "SECURE"
    assert body["computeType"] == "GPU"
    assert body["containerDiskInGb"] == 200
    assert body["gpuCount"] == 1


def test_a_pod_without_a_data_centre_is_not_pinned_to_one():
    assert "dataCenterIds" not in json.loads(sent()[0].body)


def test_no_pod_ever_asks_for_a_volume():
    """The NVMe rule, enforced on the bytes rather than in a comment.

    REST defaults `volumeInGb` to 20, so omitting the field would quietly attach a
    disk that training data could land on — and a dataloader axis measured off a
    network-backed volume measures the volume.
    """
    body = json.loads(sent()[0].body)
    assert body["volumeInGb"] == 0
    assert "networkVolumeId" not in body


def test_nothing_is_exposed_on_a_measuring_pod():
    # REST defaults to `8888/http,22/tcp`; the old call said `start_ssh=False`.
    assert json.loads(sent()[0].body)["ports"] == []


def test_the_account_key_never_enters_the_request_object(monkeypatch):
    """A Request is held by the caller and can end up in a log or a ledger entry."""
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    request, _ = sent()
    assert KEY not in repr(request)
    assert KEY not in request.body.decode()
    assert not any(KEY in value for value in request.headers.values())


def test_the_key_reaches_the_wire_as_a_bearer_header(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    seen = []

    def urlopen(http, timeout=None):
        seen.append(http)
        return FakeResponse()

    pods.send(pods.create_request(spec()), urlopen=urlopen)
    (http,) = seen
    assert http.get_header("Authorization") == f"Bearer {KEY}"
    assert http.get_method() == "POST"


def test_a_missing_key_is_named_before_anything_is_sent(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    def urlopen(http, timeout=None):
        raise AssertionError("a request was sent without a credential")

    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY is not set"):
        pods.send(pods.create_request(spec()), urlopen=urlopen)


def test_a_rejected_launch_reports_the_status_and_the_reason(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)

    def urlopen(http, timeout=None):
        raise http_error(400, b'{"error": "gpuTypeIds is required"}')

    with pytest.raises(RuntimeError) as caught:
        pods.send(pods.create_request(spec()), urlopen=urlopen)
    assert "400" in str(caught.value)
    assert "gpuTypeIds is required" in str(caught.value)


def test_a_rejection_never_carries_the_key_into_the_ledger(monkeypatch):
    """`launch_error` is `str(exc)`, and the orchestrator writes it to disk."""
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)

    def urlopen(http, timeout=None):
        raise http_error(401, b"unauthorized")

    with pytest.raises(RuntimeError) as caught:
        pods.send(pods.create_request(spec()), urlopen=urlopen)
    assert KEY not in str(caught.value)


def test_an_oversized_rejection_body_is_cut_rather_than_dumped(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)

    def urlopen(http, timeout=None):
        raise http_error(400, b"x" * 10_000)

    with pytest.raises(RuntimeError) as caught:
        pods.send(pods.create_request(spec()), urlopen=urlopen)
    assert str(caught.value).count("x") == pods.ERROR_BODY_CHARS


def test_an_empty_reply_is_reported_as_a_pod_that_may_be_billing(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    empty = lambda http, timeout=None: FakeResponse(payload=b"")  # noqa: E731
    with pytest.raises(RuntimeError, match="may have been created"):
        pods.send(pods.create_request(spec()), urlopen=empty)


def test_a_reply_without_an_id_is_a_launch_failure_not_a_ledger_entry():
    # The orchestrator indexes `pod["id"]`; a KeyError there happens beside a pod
    # that is already billing.
    with pytest.raises(RuntimeError, match="returned no id"):
        sent(reply={"desiredStatus": "RUNNING"})


def test_a_failed_launch_does_not_echo_the_pod_token_back_into_the_message():
    """A created pod echoes its own env, and that env holds an Infisical token."""
    token = "st.sentinel.pod.token"
    with pytest.raises(RuntimeError) as caught:
        sent(spec(env={"INFISICAL_TOKEN": token}), reply={"env": {"INFISICAL_TOKEN": token}})
    assert token not in str(caught.value)


def test_a_created_pod_is_returned_whole():
    _, pod = sent(reply={"id": "abc123", "desiredStatus": "RUNNING"})
    assert pod["id"] == "abc123"


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
    exp = loss_sweep_experiment()
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
    plan = sweep_plan()
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


def measuring(name, **overrides):
    """A pod that produces numbers, so both new guards apply to it."""
    body = {
        "name": name,
        "phase": "phase2",
        "model": "gemma4_e2b",
        "framework": "native",
        "purpose": "timing",
        "baseline": "canonical",
        "gpu_type_id": "NVIDIA B200",
    }
    body.update(overrides)
    return orchestrate.Experiment(**body)


def test_two_pods_pinning_different_values_of_one_axis_are_refused():
    """The way through the old guard: `axis` is required only alongside
    `settings`, so a manifest that puts `loss=cached_mnrl` straight in
    `overrides` declares nothing, and a second pinning `loss=mnrl` collides with
    nothing. Half a comparison ran on each host and the check said nothing."""
    split = [
        measuring("a", overrides=["loss=mnrl"]),
        measuring("b", overrides=["loss=cached_mnrl"]),
    ]
    with pytest.raises(orchestrate.ManifestError, match="loss"):
        orchestrate.check_axis_not_split(split)


def test_a_pinned_value_beside_a_sweep_of_the_same_axis_is_refused():
    """The sweep is one pod's, the pinned value is another's, and the report puts
    all three numbers in one column."""
    pods_ = [
        measuring("sweep", axis="loss", settings={"a": ["loss=mnrl"], "b": ["loss=cached_mnrl"]}),
        measuring("pinned", overrides=["loss=cached_mnrl"]),
    ]
    with pytest.raises(orchestrate.ManifestError, match="split across pods"):
        orchestrate.check_axis_not_split(pods_)


def test_a_knob_that_is_an_axis_counts_even_written_out_in_full():
    """`train.gradient_checkpointing` is an axis with no config group of its own,
    so a guard that only understood `group=value` overrides would miss it."""
    pods_ = [
        measuring("a", overrides=["train.gradient_checkpointing=full"]),
        measuring("b", overrides=["train.gradient_checkpointing=selective"]),
    ]
    with pytest.raises(orchestrate.ManifestError, match="train.gradient_checkpointing"):
        orchestrate.check_axis_not_split(pods_)


def test_an_override_that_moves_no_axis_does_not_collide():
    """Every Phase 0 manifest carries `data.limit` and `train.batch_size`. Neither
    is an axis, and a guard that refused them would refuse the shipped sweep."""
    orchestrate.check_axis_not_split(
        [
            measuring("a", overrides=["data.limit=8", "train.batch_size=8"]),
            measuring("b", overrides=["data.limit=8", "train.batch_size=8"]),
        ]
    )


def test_two_models_may_run_the_same_axis_on_their_own_pods():
    """The rule is one axis per model per pod, not one axis per campaign — the
    shipped `loss` sweep is three pods, one per model."""
    orchestrate.check_axis_not_split(
        [
            measuring("a", model="gemma4_e2b", overrides=["loss=mnrl"]),
            measuring("b", model="qwen3_5_0_8b", overrides=["loss=mnrl"]),
        ]
    )


def test_a_manifest_whose_declared_axis_is_not_the_one_it_moves_is_refused(tmp_path):
    path = manifest(
        tmp_path,
        "phase2-x",
        run="timing",
        baseline="canonical",
        gpu_type_id="NVIDIA B200",
        axis="loss",
        settings={"fa3": ["attn=fa3"], "sdpa": ["attn=sdpa"]},
    )
    with pytest.raises(orchestrate.ManifestError, match="declares axis 'loss'"):
        orchestrate.load_manifest(path)


def test_one_baseline_may_not_be_compared_across_two_gpu_types():
    """The baseline exists so that pods can be compared; run it on two
    accelerators and the 3% gate discards whichever pod was right."""
    with pytest.raises(orchestrate.ManifestError, match="two GPU types"):
        orchestrate.check_one_baseline_one_gpu(
            [
                measuring("a", gpu_type_id="NVIDIA B200"),
                measuring("b", model="qwen3_5_0_8b", gpu_type_id="NVIDIA A100-SXM4-80GB"),
            ]
        )


def test_two_campaigns_on_two_gpu_types_are_not_a_mix():
    """Phase 0 probes on A100 and Phase 2 sweeps on B200 is the plan, not a
    defect. Nothing compares a probe to a timing run, and the probes say so by
    naming no baseline."""
    orchestrate.check_one_baseline_one_gpu(
        [
            measuring(
                "probe", purpose="probe", baseline="none", gpu_type_id="NVIDIA A100-SXM4-80GB"
            ),
            measuring("sweep", gpu_type_id="NVIDIA B200"),
        ]
    )


def test_a_measuring_pod_that_names_no_gpu_is_refused():
    """`--gpu-type-id` is one value for the whole invocation, so an omission puts
    this pod on whatever the caller typed and its cohort on what they declared."""
    with pytest.raises(orchestrate.ManifestError, match="must declare gpu_type_id"):
        orchestrate.check_one_baseline_one_gpu([measuring("a", gpu_type_id=None)])


def test_the_shipped_manifests_pass_both_guards():
    """A guard nothing exercises is a guard nobody has read the output of."""
    experiments = orchestrate.load_experiments()
    measured = [e for e in experiments if e.baseline != orchestrate.NO_BASELINE]
    assert measured, "no manifest produces a number for the GPU cohort rule to hold together"
    assert {e.gpu_type_id for e in measured} == {"NVIDIA B200"}
    assert any(orchestrate.axes_touched(e) for e in experiments), "no manifest moves an axis"


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


# --- what the pod's token can reach -------------------------------------------
#
# `assert_pod_scope_is_safe` is the only code enforcing the constraint written at
# the top of trainbench/pods.py — an experiment pod has no business holding an
# account-wide credential — and it shipped with no test at all. It runs on the
# path to every launch, so a wrong answer either leaks the account onto a probe
# pod or blocks the whole campaign.
#
# Nothing here handles a secret value. The names are the scope, the names are what
# the guard compares, and a test that fabricated values would be the first place
# to write one down.


def reaching(*names):
    """A stand-in for the Infisical answer: the names the pod's token can read."""
    return lambda token, project_id, env="dev": set(names)


# Captured before the fixture below replaces it, so the tests about the binding
# probe can still reach the real one.
BINDING_PROBE = orchestrate.token_is_bound_to_one_environment


@pytest.fixture(autouse=True)
def _token_honours_env(monkeypatch):
    """Most tests here are about scope, and a stub scope reader would otherwise make
    the binding probe answer for an environment that does not exist. Tests about
    the binding itself override this."""
    monkeypatch.setattr(orchestrate, "token_is_bound_to_one_environment", lambda t, p: False)


def test_a_refusal_names_every_forbidden_secret_the_token_reaches(monkeypatch):
    """All of them, not the first one found: the fix is to rescope the identity,
    and an operator who is told one name at a time rescopes it one round trip at
    a time."""
    monkeypatch.setattr(
        orchestrate,
        "pod_reachable_secret_names",
        reaching("HF_TOKEN", *orchestrate.FORBIDDEN_ON_POD),
    )
    with pytest.raises(RuntimeError) as raised:
        orchestrate.assert_pod_scope_is_safe("t", "p")
    assert all(name in str(raised.value) for name in orchestrate.FORBIDDEN_ON_POD)


def test_a_scoped_token_passes_and_reports_what_it_reaches(monkeypatch):
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", reaching("HF_TOKEN"))
    assert orchestrate.assert_pod_scope_is_safe("t", "p") == {"HF_TOKEN"}


@pytest.mark.parametrize("secret", orchestrate.FORBIDDEN_ON_POD)
def test_every_forbidden_name_is_caught_on_its_own(monkeypatch, secret):
    """One name per assertion. A guard that only catches the first of a list is
    indistinguishable from a working one until the day a different one leaks."""
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", reaching("HF_TOKEN", secret))
    with pytest.raises(RuntimeError, match=secret):
        orchestrate.assert_pod_scope_is_safe("t", "p")


# The reason this check is an allowlist. A deny list holds the names someone
# remembered; the environment it guards grows without asking it. Measured
# 2026-08-02: the project's `dev` environment injects 27 names, four of which
# FORBIDDEN_ON_POD knows and one of which the pod uses — the other 22 passed.
@pytest.mark.parametrize(
    "secret",
    ["AWS_SECRET_ACCESS_KEY", "DATABASE_URL", "OPENAI_API_KEY", "SLACK_WEBHOOK_URL"],
)
def test_a_secret_no_deny_list_ever_named_is_still_refused(monkeypatch, secret):
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", reaching("HF_TOKEN", secret))
    with pytest.raises(RuntimeError, match=secret):
        orchestrate.assert_pod_scope_is_safe("t", "p")
    assert secret not in orchestrate.FORBIDDEN_ON_POD, "this name must not be on the deny list"


def test_the_refusal_counts_the_extras_so_rescoping_is_one_round_trip(monkeypatch):
    extras = [f"UNRELATED_{i}" for i in range(22)]
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", reaching("HF_TOKEN", *extras))
    with pytest.raises(RuntimeError) as raised:
        orchestrate.assert_pod_scope_is_safe("t", "p")
    assert "22 secret(s)" in str(raised.value)
    assert all(name in str(raised.value) for name in extras)


def test_a_token_that_cannot_read_hf_token_is_refused(monkeypatch):
    """The failure already on this repository's record: with no HF_TOKEN every
    gated checkpoint answers 401 and the combination is filed as unsupported. A
    pod that cannot answer its question must not be paid for."""
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", reaching())
    with pytest.raises(RuntimeError, match="cannot read HF_TOKEN"):
        orchestrate.assert_pod_scope_is_safe("t", "p")


def test_an_empty_scope_is_not_a_clean_bill(monkeypatch):
    """A check whose subject is an empty set passes by having nothing to examine.
    Under a deny list that is exactly what an unreachable Infisical looked like."""
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", reaching())
    with pytest.raises(RuntimeError):
        orchestrate.assert_pod_scope_is_safe("t", "p")


# --- the probe must measure what Infisical injected, not what the child saw -----
#
# Found by running the guard against the real `pod` environment, which holds one
# secret. It was refused for holding three: the operating system adds LC_CTYPE and
# __CF_USER_TEXT_ENCODING to any child process, and subtracting only the names we
# set counted them as secrets the pod had reached. The deny list never noticed
# because locale variables were not on it — the allowlist is what exposed it.

OS_ADDED = ["LC_CTYPE", "__CF_USER_TEXT_ENCODING"]


def scope_reader(monkeypatch, injected):
    """Stub the two subprocesses `pod_reachable_secret_names` compares.

    Both answers carry what the OS adds to a child, because that is the condition
    the defect lived in: identical on both sides and therefore not a secret.
    """
    Done = namedtuple("Done", "returncode stdout stderr")
    clean = ["PATH", "HOME", "INFISICAL_TOKEN"]

    def run(command, env, capture_output, text, timeout):
        under_infisical = command[0] == "infisical"
        names = [*clean, *OS_ADDED, *(injected if under_infisical else [])]
        return Done(0, "\n".join(names), "")

    monkeypatch.setattr(orchestrate.subprocess, "run", run)


def test_what_the_operating_system_adds_to_a_child_is_not_a_secret(monkeypatch):
    scope_reader(monkeypatch, ["HF_TOKEN"])
    assert orchestrate.pod_reachable_secret_names("t", "p", "pod") == {"HF_TOKEN"}


def test_a_correctly_scoped_environment_is_not_refused_for_holding_locale_variables(monkeypatch):
    """The reported symptom: `--infisical-env pod` refused with 'extra: LC_CTYPE,
    __CF_USER_TEXT_ENCODING' while Infisical itself reported injecting 1 secret."""
    scope_reader(monkeypatch, ["HF_TOKEN"])
    assert orchestrate.assert_pod_scope_is_safe("t", "p", "pod") == {"HF_TOKEN"}


def test_a_real_extra_secret_is_still_refused_among_the_locale_variables(monkeypatch):
    """The other half. A fix that stopped counting anything would pass too."""
    scope_reader(monkeypatch, ["HF_TOKEN", "AWS_SECRET_ACCESS_KEY"])
    with pytest.raises(RuntimeError, match="AWS_SECRET_ACCESS_KEY"):
        orchestrate.assert_pod_scope_is_safe("t", "p", "pod")


# --- a token that ignores --env -------------------------------------------------
#
# Measured 2026-08-02, counts only: a machine-identity token answers 26 names for
# `--env=dev`, 1 for `--env=pod`, and errors for an environment that does not
# exist. A service token stored in `dev` answers 26 to all three. `infisical_token`
# preferred the ambient INFISICAL_TOKEN, and the documented way to run the
# orchestrator (`infisical run --env=dev -- python scripts/orchestrate.py`) puts
# exactly that service token there — so every pod got a dev-wide token whatever
# --infisical-env said.


def bound_to_one_environment(*names):
    """A token that answers with the same secrets whatever environment is asked."""
    return lambda token, project_id, env="dev": set(names)


def test_a_token_that_answers_for_an_environment_that_cannot_exist_is_bound(monkeypatch):
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", bound_to_one_environment("A"))
    assert BINDING_PROBE("t", "p") is True


def test_a_token_the_cli_rejects_for_a_missing_environment_honours_env(monkeypatch):
    def refuse(token, project_id, env="dev"):
        raise RuntimeError(f"environment '{env}' not found")

    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", refuse)
    assert BINDING_PROBE("t", "p") is False


def test_a_bound_token_is_refused_even_when_its_scope_looks_clean(monkeypatch):
    """The case that outlives the current mess. Today `dev` holds 27 secrets, so a
    dev-bound token is refused for the extras — a fact about dev's contents, not
    about the token. Tidy dev down to HF_TOKEN and the binding would pass
    silently, and `--infisical-env pod` would go on meaning nothing."""
    monkeypatch.setattr(orchestrate, "token_is_bound_to_one_environment", lambda t, p: True)
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", reaching("HF_TOKEN"))
    with pytest.raises(RuntimeError, match="ignores --env"):
        orchestrate.assert_pod_scope_is_safe("t", "p", "pod")


def test_no_pod_is_created_for_a_token_that_ignores_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrate, "token_is_bound_to_one_environment", lambda t, p: True)
    code, created = launch(tmp_path, monkeypatch, ["HF_TOKEN"])
    assert code == 2
    assert created == []


def test_the_orchestrators_own_token_is_not_handed_to_the_pod(monkeypatch):
    """`INFISICAL_TOKEN` names the caller's token. Reusing it is how a dev-bound
    service token reached every pod through the documented invocation."""
    monkeypatch.setenv("INFISICAL_TOKEN", "the-callers-dev-bound-token")
    monkeypatch.delenv("TRAINBENCH_POD_INFISICAL_TOKEN", raising=False)
    monkeypatch.delenv("INFISICAL_UNIVERSAL_AUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="universal-auth identity"):
        orchestrate.infisical_token()


def test_a_pod_token_supplied_under_its_own_name_is_used(monkeypatch):
    monkeypatch.setenv("TRAINBENCH_POD_INFISICAL_TOKEN", "the-pods-token")
    monkeypatch.setenv("INFISICAL_TOKEN", "the-callers-token")
    assert orchestrate.infisical_token() == "the-pods-token"


def test_the_allowlist_is_what_the_pod_is_documented_to_need():
    """`.env.example` is the repository's own statement of which secret an
    experiment pod uses. Two places saying it means one can drift."""
    documented = {
        line.split("=")[0]
        for line in (REPO / ".env.example").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    }
    assert orchestrate.ALLOWED_ON_POD < documented
    assert orchestrate.ALLOWED_ON_POD.isdisjoint(orchestrate.FORBIDDEN_ON_POD)


def test_the_refusal_names_the_secret_and_never_the_token(monkeypatch):
    monkeypatch.setattr(
        orchestrate, "pod_reachable_secret_names", reaching("HF_TOKEN", "GITHUB_TOKEN")
    )
    with pytest.raises(RuntimeError) as raised:
        orchestrate.assert_pod_scope_is_safe("st.the-actual-token", "p")
    assert "GITHUB_TOKEN" in str(raised.value)
    assert "st.the-actual-token" not in str(raised.value)


def test_the_scope_is_read_with_the_pod_token_alone(monkeypatch):
    """The orchestrator runs under `infisical run` itself, so its own environment
    holds every forbidden name. Inherited by the subprocess, the answer would be
    the orchestrator's scope wearing the pod's name, and the guard would report
    a leak on a correctly scoped identity — or hide one."""
    seen = {}

    def capture(command, env, capture_output, text, timeout):
        # Both subprocesses must start from the same sanitised environment, or the
        # difference between them would be the orchestrator's own variables.
        seen.setdefault("envs", []).append(env)
        injected = ["HF_TOKEN"] if command[0] == "infisical" else []
        return namedtuple("Done", "returncode stdout stderr")(
            0, "\n".join(["PATH", "HOME", "INFISICAL_TOKEN", *injected]), ""
        )

    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    monkeypatch.setattr(orchestrate.subprocess, "run", capture)
    reachable = orchestrate.pod_reachable_secret_names("pod-token", "project")
    assert reachable == {"HF_TOKEN"}
    assert len(seen["envs"]) == 2
    for env in seen["envs"]:
        assert set(env) == {"PATH", "HOME", "INFISICAL_TOKEN"}
        assert env["INFISICAL_TOKEN"] == "pod-token"


def test_a_launch_with_no_flag_hands_the_pod_its_own_environment(tmp_path, monkeypatch):
    """What the pod gets when nobody passes `--infisical-env`.

    The literal `"pod"` is asserted rather than `POD_INFISICAL_ENV`, which would
    pass just as well after someone set that constant back to `dev`. `dev` is the
    orchestrator's own environment, and its 27 secrets are not the pod's.

    Both halves are checked together: the environment the pod is handed and the
    one the pre-launch check asks about have to be the same, or the guard vouches
    for a scope no pod will ever run in.
    """
    asked = []
    created = []

    def scope(token, project_id, env=orchestrate.POD_INFISICAL_ENV):
        asked.append(env)
        return {"HF_TOKEN"}

    monkeypatch.setattr(orchestrate, "infisical_token", lambda: "pod-token")
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", scope)
    monkeypatch.setattr(orchestrate, "image_digest", lambda image: "sha256:" + "ab" * 32)
    monkeypatch.setattr(
        orchestrate.pods, "create", lambda spec: created.append(spec) or {"id": "p"}
    )
    monkeypatch.setattr(orchestrate.pods, "terminate", lambda pod_id: None)
    monkeypatch.setattr(orchestrate.pods, "PodWatch", FakeWatch)
    code = orchestrate.main(
        [
            "--experiment",
            "phase2-loss-gemma4_e2b",
            "--allow-dirty",
            "--infisical-project-id",
            "project",
            "--out",
            str(tmp_path / "orchestrate.json"),
        ]
    )
    assert code == 0
    assert [spec.env["INFISICAL_ENV"] for spec in created] == ["pod"]
    assert asked == ["pod"]


def test_the_scope_is_read_from_the_environment_the_pod_is_handed(monkeypatch, tmp_path):
    """Separating the pod's secrets into their own Infisical environment is what
    the refusal message asks for, and `--infisical-env` is how that reaches the
    pod. The probe read `INFISICAL_ENV` off the orchestrator's own process, so
    acting on the advice pointed the pod at one environment and the check at
    another — the guard would have gone blind at the moment it was obeyed."""
    asked = {}
    real_run = subprocess.run

    def capture(command, **kwargs):
        if command[0] != "infisical":
            return real_run(command, **kwargs)
        asked["env_flag"] = next(a for a in command if a.startswith("--env="))
        return namedtuple("Done", "returncode stdout stderr")(0, "HF_TOKEN\n", "")

    monkeypatch.setenv("INFISICAL_ENV", "dev")
    monkeypatch.setattr(orchestrate.subprocess, "run", capture)
    monkeypatch.setattr(orchestrate, "infisical_token", lambda: "pod-token")
    monkeypatch.setattr(orchestrate, "image_digest", lambda image: "sha256:" + "ab" * 32)
    monkeypatch.setattr(orchestrate.pods, "create", lambda spec: {"id": "p"})
    monkeypatch.setattr(orchestrate.pods, "terminate", lambda pod_id: None)
    monkeypatch.setattr(orchestrate.pods, "PodWatch", FakeWatch)
    orchestrate.main(
        [
            "--experiment",
            "phase2-loss-gemma4_e2b",
            "--allow-dirty",
            "--infisical-env",
            "pod-only",
            "--infisical-project-id",
            "project",
            "--out",
            str(tmp_path / "orchestrate.json"),
        ]
    )
    assert asked["env_flag"] == "--env=pod-only"


# --- the guard sits on the path to a launch, not beside it ----------------------


class FakeWatch:
    """Every tracked pod finishes on the first wait, so `main`'s loop ends."""

    def __init__(self, timeout_seconds):
        self.watching = []

    def track(self, pod_id):
        self.watching.append(pod_id)

    def wait_for_any(self):
        done, self.watching = self.watching, []
        return [pods.PodOutcome(p, pods.REASON_EXITED, pods.EXITED, True, 1.0) for p in done]


def launch(tmp_path, monkeypatch, reachable):
    """Run the real `main` over one shipped manifest with nothing real behind it."""
    created = []
    monkeypatch.setattr(orchestrate, "infisical_token", lambda: "pod-token")
    monkeypatch.setattr(orchestrate, "pod_reachable_secret_names", reaching(*reachable))
    monkeypatch.setattr(orchestrate, "image_digest", lambda image: "sha256:" + "ab" * 32)
    monkeypatch.setattr(
        orchestrate.pods, "create", lambda spec: created.append(spec) or {"id": "p"}
    )
    monkeypatch.setattr(orchestrate.pods, "terminate", lambda pod_id: None)
    monkeypatch.setattr(orchestrate.pods, "PodWatch", FakeWatch)
    code = orchestrate.main(
        [
            "--experiment",
            "phase2-loss-gemma4_e2b",
            "--allow-dirty",
            "--infisical-project-id",
            "project",
            "--out",
            str(tmp_path / "orchestrate.json"),
        ]
    )
    return code, created


def test_no_pod_is_created_while_the_token_can_read_a_forbidden_secret(tmp_path, monkeypatch):
    code, created = launch(tmp_path, monkeypatch, ["HF_TOKEN", "RUNPOD_API_KEY"])
    assert code == 2
    assert created == []


def test_a_scoped_token_reaches_the_launch(tmp_path, monkeypatch):
    """The other half: a guard that refuses everything also creates no pod."""
    code, created = launch(tmp_path, monkeypatch, ["HF_TOKEN"])
    assert code == 0
    assert [spec.name for spec in created] == ["trainbench-phase2-loss-gemma4_e2b"]


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


def stub_bin(bin_dir, calls, forward):
    """The pod's tools, replaced so nothing leaves the machine.

    `forward=False` logs the argv and succeeds, which is all an assertion about
    control flow needs. `forward=True` also hands the wrapped command to the
    interpreter running the tests, so the entrypoint's own JSON handling and the
    real `publish_result.py` actually run — a sweep is not observable otherwise.

    `uv run --frozen python` and `timeout --signal=… --kill-after=… <seconds>`
    each take three arguments of their own before the command they wrap.
    """
    forwarding = {
        "uv": f'shift 3\nexec "{sys.executable}" "$@"',
        "infisical": 'while [[ $# -gt 0 && "$1" != "--" ]]; do shift; done\nshift\nexec "$@"',
        "timeout": 'shift 3\nexec "$@"',
        "python3": f'exec "{sys.executable}" "$@"',
    }
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, forwards in forwarding.items():
        stub = bin_dir / name
        stub.write_text(
            f'#!/usr/bin/env bash\necho "{name} $*" >> "{calls}"\n'
            f"{forwards if forward else 'exit 0'}\n"
        )
        stub.chmod(0o755)


def run_entrypoint(tmp_path, env, forward=False):
    """Run the real entrypoint against `stub_bin`'s PATH."""
    bin_dir = tmp_path / "bin"
    calls = tmp_path / "calls.log"
    stub_bin(bin_dir, calls, forward)
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


# --- the sweep arm: one pod, every setting of one axis ---------------------------
#
# The arm shipped without a test and with two defects that a test would have
# caught: it handed `bench.py` the plan item instead of the resolved config inside
# it, and it published `result.json`, a path the sweep never writes.

FAKE_BENCH = '''\
"""A stand-in for scripts/bench.py, which this repository does not have yet.

Validates what the entrypoint handed it the way the real entry point will —
through `BenchConfig` — so the test asserts on the shape rather than on a message,
and writes a result only for a config it accepted.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "__REPO__")

from trainbench.config_schema import BenchConfig

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

payload = json.loads(args.config.read_text())
entry = {"out": args.out.name, "config": payload, "error": None}
try:
    BenchConfig.model_validate(payload)
except Exception as exc:
    entry["error"] = type(exc).__name__
with open(os.environ["FAKE_BENCH_LOG"], "a") as handle:
    handle.write(json.dumps(entry) + "\\n")

if entry["error"] or args.out.stem in os.environ.get("FAKE_BENCH_FAIL", "").split(","):
    raise SystemExit(1)
args.out.write_text(json.dumps({"status": "ok", "config": payload}))
'''

FAKE_HUB = '''\
"""A Hub that records where a file was asked to go, and moves nothing."""

import json
import os


class HfApi:
    def __init__(self, token=None):
        self.token = token

    def create_repo(self, **kwargs):
        return None

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
        with open(os.environ["FAKE_HUB_LOG"], "a") as handle:
            body = json.loads(open(path_or_fileobj).read())
            handle.write(json.dumps({"path": path_in_repo, "body": body}) + "\\n")
'''

PUBLISH_SHIM = '''\
"""The real publish_result.py, with a fake `huggingface_hub` ahead of it on the path."""

import runpy
import sys

sys.path.insert(0, "__HUB__")
runpy.run_path("__PUBLISH__", run_name="__main__")
'''

Sweep = namedtuple("Sweep", "proc bench uploads calls")

RESULT_FILE = f"/{publish_result.RESULT_NAME}"


def loss_sweep_experiment():
    """One pod owning one axis: a baseline run plus every value of `loss`."""
    return orchestrate.Experiment(
        name="phase2-loss-gemma4_e2b",
        phase="phase2",
        model="gemma4_e2b",
        framework="native",
        purpose="timing",
        baseline="canonical",
        axis="loss",
        settings={"mnrl": ["loss=mnrl"], "cached_mnrl": ["loss=cached_mnrl"]},
    )


def sweep_plan():
    runs = orchestrate.plan_runs(
        loss_sweep_experiment(), orchestrate.load_baselines(orchestrate.BASELINES_PATH)
    )
    return [r.summary() for r in runs]


def pod_config(plan):
    """What the orchestrator puts in `TRAINBENCH_CONFIG_JSON`: the pod's own first
    run, never the baseline, which names a different model and framework."""
    own = [item for item in plan if item["role"] == orchestrate.ROLE_EXPERIMENT]
    return own[0]["config"]


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def deadlines_handed_out(sweep):
    """The seconds each `timeout` was given, in call order.

    `timeout --signal=TERM --kill-after=<grace> <seconds> <command...>`, logged by
    the stub as `timeout <argv>`.
    """
    return [int(line.split()[3]) for line in sweep.calls if line.startswith("timeout --signal=")]


def sweep_pod(tmp_path, plan, purpose="timing", fail="", config=None, budget=None, floor=None):
    """Run the entrypoint over `plan` with a fake bench.py and a fake Hub.

    The image the pod would run in has neither, so both are supplied through the
    knobs the orchestrator already uses: `TRAINBENCH_REPO_DIR` points at a scripts
    directory holding the stand-in, and `publish_result.py` there is the real one
    with an inert Hub in front of it — so what a test reads is the destination the
    shipped code computed, not one the test made up.
    """
    scripts = tmp_path / "image-repo" / "scripts"
    scripts.mkdir(parents=True)
    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "huggingface_hub.py").write_text(FAKE_HUB)
    (scripts / "bench.py").write_text(FAKE_BENCH.replace("__REPO__", str(REPO)))
    (scripts / "publish_result.py").write_text(
        PUBLISH_SHIM.replace("__HUB__", str(hub)).replace(
            "__PUBLISH__", str(REPO / "scripts" / "publish_result.py")
        )
    )
    env_dir = tmp_path / "envs" / "native"
    env_dir.mkdir(parents=True)
    bench_log = tmp_path / "bench.jsonl"
    hub_log = tmp_path / "hub.jsonl"
    env = {
        "TRAINBENCH_CONFIG_JSON": json.dumps(config if config is not None else pod_config(plan)),
        "TRAINBENCH_PLAN_JSON": json.dumps(plan),
        "TRAINBENCH_PURPOSE": purpose,
        "TRAINBENCH_ENV_DIR": str(env_dir),
        "TRAINBENCH_REPO_DIR": str(scripts.parent),
        "FAKE_BENCH_LOG": str(bench_log),
        "FAKE_HUB_LOG": str(hub_log),
        "FAKE_BENCH_FAIL": fail,
    }
    if budget is not None:
        env["TRAINBENCH_TIMEOUT_SECONDS"] = str(budget)
    if floor is not None:
        env["TRAINBENCH_MIN_SETTING_SECONDS"] = str(floor)
    proc, calls = run_entrypoint(tmp_path, env, forward=True)
    return Sweep(proc, read_records(bench_log), read_records(hub_log), calls)


def published_results(sweep):
    """Every upload that is a result, keyed by the path it landed on."""
    return {u["path"]: u["body"] for u in sweep.uploads if u["path"].endswith(RESULT_FILE)}


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_a_sweep_hands_bench_the_resolved_config_and_not_the_plan_item(tmp_path):
    """A plan item is `Run.summary()` — {name, role, overrides, config} — while
    `bench.py` validates a `BenchConfig`. Dumped whole, every setting of every
    sweep is rejected before it starts."""
    plan = sweep_plan()
    sweep = sweep_pod(tmp_path, plan)
    assert [entry["error"] for entry in sweep.bench] == [None] * len(plan), sweep.proc.stderr
    assert [entry["config"] for entry in sweep.bench] == [item["config"] for item in plan]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_every_setting_of_a_sweep_reaches_the_results_repo(tmp_path):
    """The loop writes `result-<i>.json`; the publish block read `result.json`, so
    the sweep always took the fallback branch and uploaded "no result file"."""
    plan = sweep_plan()
    sweep = sweep_pod(tmp_path, plan)
    results = published_results(sweep)
    assert len(results) == len(plan), sweep.uploads
    assert all(body["status"] == "ok" for body in results.values())
    for item in plan:
        slug = publish_result.setting_dir(item["name"])
        assert any(path.endswith(f"/{slug}{RESULT_FILE}") for path in results)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_one_setting_failing_does_not_take_the_rest_of_the_axis_with_it(tmp_path):
    plan = sweep_plan()
    sweep = sweep_pod(tmp_path, plan, fail="result-1")
    assert len(sweep.bench) == len(plan)
    results = published_results(sweep)
    assert len(results) == len(plan)
    failed = publish_result.setting_dir(plan[1]["name"])
    by_setting = {path.split("/")[-2]: body for path, body in results.items()}
    assert by_setting[failed]["status"] == "no_result"
    assert plan[1]["name"] in by_setting[failed]["probe"]["checks"][0]["error"]
    assert sorted(b["status"] for s, b in by_setting.items() if s != failed) == ["ok", "ok"]
    assert "run exited 1" in sweep.proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_a_plan_item_without_a_resolved_config_stops_only_that_setting(tmp_path):
    """Writing `null` instead would hand bench.py an empty config, which reads as a
    setting that was configured rather than one that never started."""
    plan = sweep_plan()
    # Read before the damage: the orchestrator resolves the pod's own config
    # separately, so a broken plan item does not leave the pod unnameable.
    config = pod_config(plan)
    del plan[1]["config"]
    sweep = sweep_pod(tmp_path, plan, config=config)
    assert "carries no resolved config" in sweep.proc.stderr
    assert len(sweep.bench) == len(plan) - 1
    results = published_results(sweep)
    assert len(results) == len(plan)
    skipped = publish_result.setting_dir(plan[1]["name"])
    by_setting = {path.split("/")[-2]: body for path, body in results.items()}
    assert by_setting[skipped]["status"] == "no_result"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_the_pod_budget_is_shared_between_the_settings_not_handed_to_each(tmp_path):
    """`DEADLINE_SECONDS` is the whole pod's budget — the orchestrator sets it just
    under its own deadline. Given to `timeout` once per setting, an N-setting sweep
    can bill N times it, and the guard that exists to bound the bill stops
    bounding it."""
    plan = sweep_plan()
    budget = 600
    sweep = sweep_pod(tmp_path, plan, budget=budget, floor=1)
    slices = deadlines_handed_out(sweep)
    assert len(slices) == len(plan)
    assert all(0 < s <= budget for s in slices), slices
    # The first setting gets what is left divided by the settings still to run, a
    # second or two of startup already spent.
    fair_share = budget // len(plan)
    assert fair_share - 5 <= slices[0] <= fair_share, slices


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_a_setting_the_budget_cannot_fit_is_not_started_and_says_so(tmp_path):
    """Skipping silently would file the setting as absent, which reads exactly like
    a setting nobody ran — the ambiguity every record in this file exists to
    remove."""
    plan = sweep_plan()
    sweep = sweep_pod(tmp_path, plan, budget=0, floor=1)
    assert deadlines_handed_out(sweep) == []
    assert sweep.bench == []
    results = published_results(sweep)
    assert len(results) == len(plan)
    for body in results.values():
        assert body["status"] == "no_result"
        assert "budget" in body["probe"]["checks"][0]["error"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_an_empty_plan_still_records_that_the_pod_produced_nothing(tmp_path):
    sweep = sweep_pod(tmp_path, [], config=resolved_config())
    assert sweep.proc.returncode == 0, sweep.proc.stderr
    assert "nothing to measure" in sweep.proc.stderr
    results = published_results(sweep)
    assert list(results) == [f"results/native/gemma4_e2b/local{RESULT_FILE}"]
    assert next(iter(results.values()))["status"] == "no_result"


def test_two_settings_of_one_pod_do_not_publish_to_the_same_path():
    directory = publish_result.result_dir_in_repo(resolved_config())
    baseline = publish_result.result_path_in_repo(directory, "baseline:canonical")
    setting = publish_result.result_path_in_repo(directory, "mnrl")
    assert baseline != setting
    # `:` never reaches a path segment; an unlabelled pod keeps the old destination.
    assert ":" not in baseline
    assert publish_result.result_path_in_repo(directory, None) == f"{directory}{RESULT_FILE}"


def test_a_label_with_nothing_usable_in_it_is_refused_rather_than_shared(tmp_path):
    with pytest.raises(ValueError):
        publish_result.setting_dir("//")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(resolved_config()))
    result = tmp_path / "result.json"
    result.write_text("{}")
    code = publish_result.main(
        ["--repo", "acct/r", "--config", str(config), "--result", str(result), "--label", "//"]
    )
    assert code == 2


def test_a_sweeps_per_setting_results_stay_readable_by_the_report(tmp_path):
    """Why a directory per setting and not a `result-<setting>.json`:
    `report.load_artifacts` reads `result.json` and skips every other name, so a
    renamed file would upload cleanly and never reach the matrix."""
    for setting in ("mnrl", "cached_mnrl"):
        write_artifact(
            tmp_path,
            "native",
            "gemma4_e2b",
            f"pod/{setting}",
            publish_result.RESULT_NAME,
            probe_payload("native", "gemma4_e2b", [{"name": "c", "ok": True}]),
        )
    artifacts, skipped = report.load_artifacts(tmp_path / "results")
    assert len(artifacts) == 2
    assert skipped == []
