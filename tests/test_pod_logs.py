"""A pod's container log has to survive the pod. Fetch, parse, and persistence.

Written for lane L: `scripts/orchestrate.py` used to call `pods.terminate` with
nothing having read the pod's own log first, and three of the four defects
blocking a whole campaign were sitting in exactly that log, unread. These tests
are meant to be merged into `tests/test_pods.py` (out of scope for this lane —
another lane holds that file).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from trainbench import pods

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import orchestrate  # noqa: E402

KEY = "rpa-sentinel-not-a-real-key"


# --- fakes -----------------------------------------------------------------


class ChunkedResponse:
    """A fake `urlopen` response handing out one chunk per `.read()` call.

    `then`, once the chunks run out, is either `None` (the server closed the
    connection — `.read()` returns `b""`) or an exception instance to raise
    (an idle timeout).
    """

    def __init__(self, chunks, then=None):
        self._chunks = list(chunks)
        self._then = then

    def read(self, n=-1):
        if self._chunks:
            return self._chunks.pop(0)
        if self._then is not None:
            raise self._then
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SteadyResponse:
    """A fake response that never stops sending data. Counts its own reads so a
    wall-clock bound can be proven to have fired rather than assumed."""

    def __init__(self, chunk=b'data: {"source": "container", "line": "x", "ts": "t"}\n\n'):
        self._chunk = chunk
        self.reads = 0

    def read(self, n=-1):
        self.reads += 1
        return self._chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def counting_clock(step=1.0, start=0.0):
    """A fake `clock()` that advances by `step` every call. Deterministic and
    finite, unlike a fake response that never stops — this is what lets a
    wall-clock-bound test terminate instead of spinning forever."""
    state = {"t": start}

    def clock():
        state["t"] += step
        return state["t"]

    return clock


def http_error(url, code, body):
    return urllib.error.HTTPError(url, code, "rejected", {}, io.BytesIO(body))


def hf_api_stub(monkeypatch, fail=None):
    """Install a fake `HfApi` in `orchestrate`'s namespace and return the list of
    `(path_in_repo, body, repo_id)` uploads it received. `fail` names which call
    should raise instead ('create_repo' or 'upload_file')."""
    uploads: list[tuple[str, bytes, str]] = []

    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            if fail == "create_repo":
                raise RuntimeError("hub is down")

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
            if fail == "upload_file":
                raise RuntimeError("upload rejected")
            uploads.append((path_in_repo, path_or_fileobj, repo_id))

    monkeypatch.setattr(orchestrate, "HfApi", FakeHfApi)
    return uploads


def make_exp(name="exp0", framework="native", model="qwen3_5_0_8b"):
    return orchestrate.Experiment(
        name=name,
        phase="phase0",
        model=model,
        framework=framework,
        purpose="timing",
        baseline="canonical-a100",
    )


def make_plans(exp):
    run = orchestrate.Run(
        name=exp.name,
        role=orchestrate.ROLE_EXPERIMENT,
        overrides=(),
        config={"framework": {"name": exp.framework}, "model": {"name": exp.model}},
    )
    return {exp.name: [run]}


# --- logs_request: the documented v2 endpoint -------------------------------


def test_logs_request_targets_the_v2_log_endpoint():
    request = pods.logs_request("pod123")
    assert request.method == "GET"
    assert request.url.startswith("https://v2-rest.runpod.io/v2/pods/pod123/logs?")
    assert request.body is None


def test_logs_request_omits_source_by_default_asking_for_both():
    request = pods.logs_request("pod123")
    assert "source=" not in request.url
    assert f"tail={pods.LOG_TAIL_LINES}" in request.url


def test_logs_request_can_ask_for_one_source_only():
    request = pods.logs_request("pod123", source="system")
    assert "source=system" in request.url


def test_logs_request_encodes_a_hostile_pod_id():
    request = pods.logs_request("pod/needs?encoding")
    assert "pod/needs?encoding" not in request.url
    assert "pod%2Fneeds%3Fencoding" in request.url


# --- parse_log_sse: RunPod's own documented frame shape ---------------------


def test_parse_log_sse_labels_each_frame_by_source():
    raw = (
        b'id: t1\ndata: {"source": "container", "line": "Model loaded.", "ts": "t1"}\n\n'
        b'id: t2\ndata: {"source": "system", "line": "pulling image", "ts": "t2"}\n\n'
    )
    events = pods.parse_log_sse(raw)
    assert [e["source"] for e in events] == ["container", "system"]
    assert events[0]["line"] == "Model loaded."


def test_parse_log_sse_joins_a_multiline_data_field():
    raw = b'data: {"source": "container",\ndata:  "line": "two lines", "ts": "t"}\n\n'
    (event,) = pods.parse_log_sse(raw)
    assert event == {"source": "container", "line": "two lines", "ts": "t"}


def test_parse_log_sse_keeps_a_non_json_frame_verbatim():
    raw = b"data: not json at all\n\n"
    (event,) = pods.parse_log_sse(raw)
    assert event == {"raw": "not json at all"}


def test_parse_log_sse_keeps_a_non_object_json_frame_verbatim():
    """`data: 42` parses as JSON but is not a log frame's shape."""
    raw = b"data: 42\n\n"
    (event,) = pods.parse_log_sse(raw)
    assert event == {"raw": "42"}


def test_parse_log_sse_ignores_an_id_only_block():
    raw = b'id: t1\n\ndata: {"source": "container", "line": "x", "ts": "t"}\n\n'
    events = pods.parse_log_sse(raw)
    assert len(events) == 1


def test_parse_log_sse_of_empty_bytes_is_no_events():
    assert pods.parse_log_sse(b"") == []


# --- stream_log: bounded, never past its bound -------------------------------


def test_stream_log_reads_until_the_server_closes_the_connection():
    response = ChunkedResponse([b"data: a\n\n", b"data: b\n\n"])
    raw, truncated = pods.stream_log(
        pods.logs_request("pod123"), urlopen=lambda http, timeout=None: response
    )
    assert raw == b"data: a\n\ndata: b\n\n"
    assert truncated is False


def test_stream_log_treats_an_idle_timeout_as_a_clean_end():
    """No byte for the idle window is the ordinary case: a container that has
    already exited is not going to produce another line."""
    response = ChunkedResponse([b"data: a\n\n"], then=TimeoutError("timed out"))
    raw, truncated = pods.stream_log(
        pods.logs_request("pod123"), urlopen=lambda http, timeout=None: response
    )
    assert raw == b"data: a\n\n"
    assert truncated is False


def test_stream_log_stops_at_the_byte_cap_and_says_so(monkeypatch):
    monkeypatch.setattr(pods, "LOG_MAX_BYTES", 10)
    response = SteadyResponse(chunk=b"x" * 8)
    raw, truncated = pods.stream_log(
        pods.logs_request("pod123"), urlopen=lambda http, timeout=None: response
    )
    assert truncated is True
    assert len(raw) >= 10
    # Stopped, not merely capped after the fact: a runaway stream cannot make
    # this loop read forever.
    assert response.reads <= 3


def test_stream_log_stops_at_the_wall_clock_deadline_even_while_data_keeps_arriving():
    """A stream that never idles and never hits the byte cap must still end.

    Otherwise a pod whose container keeps a connection busy forever could hold
    up its own termination — the one outcome this module must never produce.
    """
    response = SteadyResponse()
    clock = counting_clock(step=1.0)
    raw, truncated = pods.stream_log(
        pods.logs_request("pod123"), urlopen=lambda http, timeout=None: response, clock=clock
    )
    assert raw  # it did read something before stopping
    assert truncated is False
    # Bounded: the deadline is `LOG_TOTAL_SECONDS` past the first clock reading,
    # advancing by 1 per read, so the loop cannot run past roughly that many
    # reads plus the one that observes the deadline has passed.
    assert 1 <= response.reads <= int(pods.LOG_TOTAL_SECONDS) + 2


def test_stream_log_raises_on_a_real_http_refusal(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)

    def urlopen(http, timeout=None):
        raise http_error("https://v2-rest.runpod.io/v2/pods/pod123/logs", 404, b"pod not found")

    with pytest.raises(RuntimeError, match="404") as caught:
        pods.stream_log(pods.logs_request("pod123"), urlopen=urlopen)
    assert "pod not found" in str(caught.value)
    assert KEY not in str(caught.value)


# --- fetch_log: request + stream + parse together ----------------------------


def test_fetch_log_counts_container_and_system_lines_separately():
    raw = (
        b'data: {"source": "container", "line": "a", "ts": "1"}\n\n'
        b'data: {"source": "container", "line": "b", "ts": "2"}\n\n'
        b'data: {"source": "system", "line": "c", "ts": "3"}\n\n'
        b"data: not json\n\n"
    )
    response = ChunkedResponse([raw])
    fetch = pods.fetch_log("pod123", urlopen=lambda http, timeout=None: response)
    assert fetch.container_lines == 2
    assert fetch.system_lines == 1
    assert len(fetch.events) == 4
    assert fetch.truncated is False


def test_fetch_log_propagates_truncated(monkeypatch):
    monkeypatch.setattr(pods, "LOG_MAX_BYTES", 5)
    response = SteadyResponse(chunk=b"x" * 8)
    fetch = pods.fetch_log("pod123", urlopen=lambda http, timeout=None: response)
    assert fetch.truncated is True


def test_fetch_log_of_an_empty_stream_is_zero_lines_not_an_error():
    """The pod printed nothing. That is a fact, not a failure — distinguishing
    it from a failed fetch is `capture_pod_log`'s job in `scripts/orchestrate.py`."""
    response = ChunkedResponse([])
    fetch = pods.fetch_log("pod123", urlopen=lambda http, timeout=None: response)
    assert fetch.events == ()
    assert fetch.container_lines == 0
    assert fetch.system_lines == 0
    assert fetch.truncated is False


# --- orchestrate.pod_log_destination -----------------------------------------


def test_pod_log_destination_is_keyed_by_the_pod_id_not_an_env_var(monkeypatch):
    """`RUNPOD_POD_ID` only exists on the pod itself; the orchestrator already
    holds the id directly and must not go looking for it in its own env."""
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    exp = make_exp()
    plans = make_plans(exp)
    destination = orchestrate.pod_log_destination(exp, plans, "pod123")
    assert destination == "results/native/qwen3_5_0_8b/pod123/pod.log.jsonl"


# --- orchestrate.capture_pod_log: fetch, then publish, never raise ----------


def test_capture_pod_log_fetches_and_uploads_before_returning(monkeypatch):
    exp = make_exp()
    plans = make_plans(exp)
    events = (
        {"source": "container", "line": "hello", "ts": "t1"},
        {"source": "system", "line": "sys", "ts": "t2"},
    )
    monkeypatch.setattr(
        pods, "fetch_log", lambda pod_id, **kw: pods.LogFetch(events=events, truncated=False)
    )
    uploads = hf_api_stub(monkeypatch)

    result = orchestrate.capture_pod_log(exp, plans, "pod123", "acct/results")

    assert result["path_in_repo"] == "results/native/qwen3_5_0_8b/pod123/pod.log.jsonl"
    assert result["container_lines"] == 1
    assert result["system_lines"] == 1
    assert result["truncated"] is False
    assert result["unreadable"] is None
    (path_in_repo, body, repo_id) = uploads[0]
    assert path_in_repo == result["path_in_repo"]
    assert repo_id == "acct/results"
    lines = body.decode().splitlines()
    assert json.loads(lines[0])["line"] == "hello"
    assert json.loads(lines[1])["line"] == "sys"


def test_capture_pod_log_records_a_fetch_failure_as_unreadable_with_no_path(monkeypatch):
    exp = make_exp()
    plans = make_plans(exp)

    def boom(pod_id, **kw):
        raise RuntimeError("RunPod REST GET ... -> 404: pod not found")

    monkeypatch.setattr(pods, "fetch_log", boom)
    uploads = hf_api_stub(monkeypatch)

    result = orchestrate.capture_pod_log(exp, plans, "pod123", "acct/results")

    assert result["path_in_repo"] is None
    assert result["unreadable"] and "404" in result["unreadable"]
    assert uploads == []


def test_capture_pod_log_records_an_upload_failure_as_unreadable_with_no_path(monkeypatch):
    exp = make_exp()
    plans = make_plans(exp)
    monkeypatch.setattr(
        pods, "fetch_log", lambda pod_id, **kw: pods.LogFetch(events=(), truncated=False)
    )
    hf_api_stub(monkeypatch, fail="upload_file")

    result = orchestrate.capture_pod_log(exp, plans, "pod123", "acct/results")

    assert result["path_in_repo"] is None
    assert result["unreadable"] and "upload" in result["unreadable"]


def test_an_empty_log_is_distinguishable_from_a_failed_fetch(monkeypatch):
    exp = make_exp()
    plans = make_plans(exp)

    monkeypatch.setattr(
        pods, "fetch_log", lambda pod_id, **kw: pods.LogFetch(events=(), truncated=False)
    )
    hf_api_stub(monkeypatch)
    empty = orchestrate.capture_pod_log(exp, plans, "pod-empty", "acct/results")

    def boom(pod_id, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(pods, "fetch_log", boom)
    failed = orchestrate.capture_pod_log(exp, plans, "pod-failed", "acct/results")

    assert empty["unreadable"] is None
    assert empty["path_in_repo"] is not None
    assert empty["container_lines"] == 0
    assert empty["system_lines"] == 0

    assert failed["unreadable"] is not None
    assert failed["path_in_repo"] is None
    assert failed["container_lines"] is None


def test_capture_pod_log_never_raises_a_secret_bearing_exception(monkeypatch):
    """`ERROR_BODY_CHARS` bounds every message this function returns; a raw
    exception str could in principle carry more, so nothing here is echoed
    unbounded."""
    exp = make_exp()
    plans = make_plans(exp)

    def boom(pod_id, **kw):
        raise RuntimeError("x" * 10_000)

    monkeypatch.setattr(pods, "fetch_log", boom)
    result = orchestrate.capture_pod_log(exp, plans, "pod123", "acct/results")
    assert len(result["unreadable"]) <= pods.ERROR_BODY_CHARS + len("RuntimeError: ")


# --- orchestrate.handle_pod_outcome: capture, then terminate, always --------


def outcome(pod_id="pod123", reason=pods.REASON_EXITED, status=pods.EXITED):
    return pods.PodOutcome(
        pod_id=pod_id, reason=reason, status=status, ever_ran=True, waited_seconds=12.0
    )


def test_a_log_is_captured_before_the_pod_is_terminated(monkeypatch):
    exp = make_exp()
    plans = make_plans(exp)
    calls: list[str] = []

    def fake_capture(*_args, **_kwargs):
        calls.append("capture")
        return {
            "path_in_repo": "results/native/qwen3_5_0_8b/pod123/pod.log.jsonl",
            "container_lines": 0,
            "system_lines": 0,
            "truncated": False,
            "unreadable": None,
        }

    def fake_terminate(pod_id):
        calls.append("terminate")

    monkeypatch.setattr(orchestrate, "capture_pod_log", fake_capture)
    monkeypatch.setattr(pods, "terminate", fake_terminate)

    entries = {exp.name: {"outcome": None, "unreadable": None, "log": None}}
    args = argparse.Namespace(result_repo="acct/results")
    orchestrate.handle_pod_outcome(outcome(), exp, plans, entries, args)

    assert calls == ["capture", "terminate"]
    assert entries[exp.name]["log"]["path_in_repo"].endswith("pod.log.jsonl")
    assert entries[exp.name]["outcome"]["reason"] == pods.REASON_EXITED


def test_a_capture_failure_still_lets_the_pod_be_terminated(monkeypatch):
    exp = make_exp()
    plans = make_plans(exp)
    terminated = []

    monkeypatch.setattr(
        orchestrate,
        "capture_pod_log",
        lambda *a, **k: {
            "path_in_repo": None,
            "container_lines": None,
            "system_lines": None,
            "truncated": False,
            "unreadable": "RunPod REST GET ... -> 404: pod not found",
        },
    )
    monkeypatch.setattr(pods, "terminate", lambda pod_id: terminated.append(pod_id))

    entries = {exp.name: {"outcome": None, "unreadable": None, "log": None}}
    args = argparse.Namespace(result_repo="acct/results")
    orchestrate.handle_pod_outcome(outcome(pod_id="pod123"), exp, plans, entries, args)

    assert terminated == ["pod123"]
    assert entries[exp.name]["log"]["unreadable"] is not None
    assert entries[exp.name]["log"]["path_in_repo"] is None


def test_the_ledger_entry_points_at_what_was_actually_written(monkeypatch):
    exp = make_exp()
    plans = make_plans(exp)
    events = ({"source": "container", "line": "hi", "ts": "t"},)
    monkeypatch.setattr(
        pods, "fetch_log", lambda pod_id, **kw: pods.LogFetch(events=events, truncated=False)
    )
    hf_api_stub(monkeypatch)
    monkeypatch.setattr(pods, "terminate", lambda pod_id: None)

    entries = {exp.name: {"outcome": None, "unreadable": None, "log": None}}
    args = argparse.Namespace(result_repo="acct/results")
    orchestrate.handle_pod_outcome(outcome(pod_id="pod999"), exp, plans, entries, args)

    expected = orchestrate.pod_log_destination(exp, plans, "pod999")
    assert entries[exp.name]["log"]["path_in_repo"] == expected


def test_terminate_runs_even_when_no_experiment_is_tracked_for_the_outcome(monkeypatch):
    terminated = []

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("capture_pod_log must not run without an experiment to file it under")

    monkeypatch.setattr(orchestrate, "capture_pod_log", fail_if_called)
    monkeypatch.setattr(pods, "terminate", lambda pod_id: terminated.append(pod_id))

    args = argparse.Namespace(result_repo="acct/results")
    orchestrate.handle_pod_outcome(outcome(pod_id="podXYZ"), None, {}, {}, args)

    assert terminated == ["podXYZ"]


def test_a_termination_failure_does_not_raise(monkeypatch):
    exp = make_exp()
    plans = make_plans(exp)

    def failing_terminate(pod_id):
        raise RuntimeError("network blip")

    monkeypatch.setattr(
        orchestrate,
        "capture_pod_log",
        lambda *a, **k: {
            "path_in_repo": None,
            "container_lines": 0,
            "system_lines": 0,
            "truncated": False,
            "unreadable": None,
        },
    )
    monkeypatch.setattr(pods, "terminate", failing_terminate)

    entries = {exp.name: {"outcome": None, "unreadable": None, "log": None}}
    args = argparse.Namespace(result_repo="acct/results")
    # Must not raise.
    orchestrate.handle_pod_outcome(outcome(), exp, plans, entries, args)
