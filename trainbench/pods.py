"""RunPod control, orchestrator side only.

Never imported on a pod: RUNPOD_API_KEY is an account-wide credential and a probe
pod has no business holding one.

Completion is a *transition*, not a snapshot. A pod that has not started yet and a
pod that has finished look almost identical over the API — both report a null
runtime — and the only thing separating them is whether a runtime was ever seen.
Reading one snapshot cannot tell them apart, so this module keeps per-pod state.

Two transports, on purpose. Creation goes over RunPod's REST API; reading and
terminating stay on the `runpod` SDK's GraphQL calls. Neither half is a preference:

- Creation cannot use the SDK. `runpod.api.mutations.pods` builds its mutation by
  f-string interpolation — `f'{{ key: "{k}", value: "{v}" }}'` for every env pair —
  so a value containing a quote closes the string and the rest of it becomes
  syntax. Our env carries `TRAINBENCH_CONFIG_JSON`, a whole JSON document. The
  first campaign lost all fifteen launches to `Syntax Error: Expected ":", found
  String ": {"` and created no pod. REST takes a JSON body, so the encoder escapes
  what the interpolator could not.
- Reading cannot use REST. The REST `Pod` object has no `runtime` field, and
  `runtime` is the whole basis of the pending/running distinction below. Measured
  against the live account: 50 REST pod objects, not one carried the key, while
  GraphQL `get_pod` returned a non-null runtime for a RUNNING pod and a null one
  for an EXITED pod. Moving `get` to REST would make every pod read as pending
  forever.

A restarting container is a third thing, and neither transport counts restarts.
Measured 2026-08-02 against the live account, because guessing the field is how
this repository has repeatedly ended up checking something other than the thing:

- REST publishes its `Pod` schema at `rest.runpod.io/v1/openapi.json`. It carries
  `desiredStatus`, `lastStartedAt` and `lastStatusChange`, and no restart count.
- GraphQL blocks introspection, so the candidates were probed one at a time and
  the server answered `Cannot query field "X" on type "Pod"`. `restartCount`,
  `restarts`, `restartPolicy`, `podStatus`, `status`, `containerStatus`,
  `exitCode` and `lastExitCode` do not exist on `Pod`; `restartCount`,
  `restarts`, `exitCode` and `status` do not exist on `PodRuntime`.
- `Pod.uptimeSeconds` exists and was 0 on all 50 pods, running ones included.

What is left is `runtime.uptimeInSeconds`, and it is a *container* clock rather
than a rental clock. Over the 21 running pods in the account, `now - uptime`
landed 10-140s after `createdAt` for fifteen of them — the image pull — and
hours to 8.9 days after it for six, while `lastStartedAt` equalled `createdAt`
for every single pod. So the field resets under a pod that the rest of the API
describes as continuously up, and that reset is the only restart this API will
ever report. `observe` reads it; nothing else can.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Models plus a framework venv land on pod-local disk; the default container disk
# is far too small for that.
DEFAULT_CONTAINER_DISK_GB = 120
POLL_SECONDS = 20

# https://rest.runpod.io/v1/openapi.json — `servers[0].url`, `POST /pods` with a
# `PodCreateInput` body, bearer auth. Field names below come from that document.
REST_BASE_URL = "https://rest.runpod.io/v1"
REST_TIMEOUT_SECONDS = 60

# How much of a rejection body to keep in the error. A launch failure is written
# to the orchestrator's ledger, so the message has to be long enough to name the
# offending field and short enough not to be a dump.
ERROR_BODY_CHARS = 500

# How many consecutive readings must agree before an *inferred* terminal state ends
# a pod's watch. A state the API states outright (`desiredStatus`) needs one. How
# often RunPod's API answers inconsistently is unmeasured here — the reason for the
# delay is that the two outcomes are not symmetric: waiting one extra poll costs
# POLL_SECONDS, acting on a wrong reading costs the run.
CONFIRMATIONS_REQUIRED = 2

# One observation of a pod.
PENDING = "pending"  # created, no runtime yet (provisioning or pulling the image)
RUNNING = "running"  # runtime present
RESTARTING = "restarting"  # runtime present, but not the container the last reading saw
EXITED = "exited"  # RunPod says the container will not run again
GONE = "gone"  # the API no longer knows this pod
UNKNOWN = "unknown"  # the API call itself failed; nothing was learned

# How far `runtime.uptimeInSeconds` may fall between two readings and still be the
# same container. The field is whole seconds derived from a start timestamp, so
# two API replicas rounding `now - startedAt` differently can put one second
# between them; a restart puts the entire life of the dead container between them.
# Anything in between is not a case this has evidence for, and the cost of the two
# mistakes is not symmetric — see REASON_RESTARTED.
UPTIME_JITTER_SECONDS = 1

# desiredStatus values that mean the container will not come back.
TERMINAL_DESIRED_STATUS = frozenset({"TERMINATED", "EXITED"})

# Why a pod stopped being watched. Recorded per pod, because "the work finished"
# and "we gave up waiting" are different results and only one of them is a datum.
REASON_EXITED = "exited"
REASON_STOPPED = "stopped"  # runtime went from present to absent
REASON_GONE = "gone"
REASON_TIMEOUT = "timeout"
# The container was replaced by another one. Not tolerated even once, and the
# reason is the entrypoint rather than the platform: it runs the pod's whole plan
# from the top, so a second container repeats settings the first already published
# and bills for them again. Waiting to see whether the next container fares better
# is what cost the first A100 canary ten minutes of a pod that restarted forty
# times while every reading said `running`.
REASON_RESTARTED = "restarted"


@dataclass(frozen=True)
class PodSpec:
    name: str
    image: str
    gpu_type_id: str
    env: dict[str, str]
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB
    data_center_id: str | None = None


@dataclass(frozen=True)
class Request:
    """Exactly what goes on the wire, minus authentication.

    `body` is bytes rather than a dict because the encoding *is* the fix: a test
    that inspects a dict proves nothing about how the value gets serialised, and
    serialisation is what broke the first campaign. The credential is deliberately
    absent — it is added inside `send` and never enters an object a caller can
    hold, log, or attach to a ledger entry.
    """

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None


Transport = Callable[[Request], dict[str, Any]]


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is not set; run under `infisical run --`")
    return key


def _client() -> Any:
    import runpod

    runpod.api_key = _api_key()
    return runpod


def create_body(spec: PodSpec) -> dict[str, Any]:
    """The `PodCreateInput` for one pod.

    Every field the previous GraphQL call relied on a default for is written out.
    The REST defaults are not the same defaults — `volumeInGb` defaults to 20 and
    `ports` to `8888/http,22/tcp` — and a benchmark that silently grows a disk or
    an ingress between runs has an uncontrolled variable in it.
    """
    body: dict[str, Any] = {
        "name": spec.name,
        "imageName": spec.image,
        "computeType": "GPU",
        "cloudType": "SECURE",
        "gpuTypeIds": [spec.gpu_type_id],
        # One GPU per pod. Pinned rather than defaulted: a multi-GPU pod would
        # change what every throughput number means, and it must come from a
        # PodSpec field somebody chose, not from a platform default that moved.
        "gpuCount": 1,
        "containerDiskInGb": spec.container_disk_gb,
        # No persistent volume of any kind, and no `networkVolumeId`: training data
        # must sit on pod-local NVMe or the dataloader axis measures the volume
        # instead of the pipeline. Not attaching one also frees the pod from a
        # single data centre.
        "volumeInGb": 0,
        # Nothing needs to reach the pod. The entrypoint runs the workload and
        # pushes the result out; the old call said the same thing with
        # `start_ssh=False`, which REST has no equivalent for.
        "ports": [],
        "env": dict(spec.env),
    }
    if spec.data_center_id is not None:
        body["dataCenterIds"] = [spec.data_center_id]
    return body


def create_request(spec: PodSpec) -> Request:
    return Request(
        method="POST",
        url=f"{REST_BASE_URL}/pods",
        headers={"Content-Type": "application/json"},
        body=json.dumps(create_body(spec)).encode(),
    )


def send(request: Request, urlopen: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    """Send one request and return its decoded payload.

    The key is read here, per call, and inlined into the header dict so that no
    named local holds it.
    """
    http = urllib.request.Request(
        request.url,
        data=request.body,
        headers={**request.headers, "Authorization": f"Bearer {_api_key()}"},
        method=request.method,
    )
    try:
        with urlopen(http, timeout=REST_TIMEOUT_SECONDS) as response:
            status = response.status
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:ERROR_BODY_CHARS].decode("utf-8", "replace")
        # `from None`: the chained HTTPError adds nothing the message lacks, and a
        # traceback renderer that prints frame locals would print the header dict.
        raise RuntimeError(
            f"RunPod REST {request.method} {request.url} -> {exc.code}: {detail}"
        ) from None
    if not payload:
        # A pod may exist and be billing with nobody holding its id. Say so.
        raise RuntimeError(
            f"RunPod REST {request.method} {request.url} -> {status} with an empty body; "
            "a pod may have been created without returning its id"
        )
    return json.loads(payload)


def create(spec: PodSpec, transport: Transport = send) -> dict[str, Any]:
    pod = transport(create_request(spec))
    if not isinstance(pod, dict) or not pod.get("id"):
        # The caller indexes `pod["id"]` to record what to terminate. Failing here
        # names the problem; failing there is a KeyError beside a billing pod.
        # The payload itself stays out of the message: a created pod echoes its
        # own env back, and that env carries the pod's Infisical token.
        raise RuntimeError(
            f"RunPod accepted the pod request but returned no id "
            f"(payload keys: {sorted(pod) if isinstance(pod, dict) else type(pod).__name__})"
        )
    return pod


def get(pod_id: str) -> dict[str, Any] | None:
    """Read one pod. Stays on GraphQL because REST does not report a runtime.

    The SDK interpolates `pod_id` into its query the same unescaped way it
    interpolates env values, so the defect is present here too — it is just not
    reachable: the only value that reaches it is an id RunPod generated and handed
    back from `create`, never anything this repository composes.
    """
    return _client().get_pod(pod_id)


def terminate(pod_id: str) -> None:
    """Same transport as `get`, for the same reason and with the same caveat."""
    _client().terminate_pod(pod_id)


@dataclass(frozen=True)
class Reading:
    """One observation: the pod's state, and the clock that dates its container.

    The uptime travels with the status because it is the input to the *next*
    reading's restart test and it costs an API call to fetch. A caller that kept
    only the status would have to ask again to learn what it already had.

    `uptime_seconds` is None whenever the reading did not come with a readable
    container clock — a pending pod, a failed call, or a runtime whose shape the
    API changed. None never reads as a restart: an absence is not evidence, and
    the alternative is terminating live pods over a field that moved.
    """

    status: str
    uptime_seconds: int | None = None


def container_uptime(runtime: Any) -> int | None:
    """Seconds the current container has been up, or None if it cannot be read.

    Tolerant on purpose. Before restarts were detected at all, any non-null
    runtime read as `running` whatever its shape, and a stricter reading here
    would turn a payload change into `unknown` for every pod at once.
    """
    if not isinstance(runtime, dict):
        return None
    uptime = runtime.get("uptimeInSeconds")
    return uptime if isinstance(uptime, int) and not isinstance(uptime, bool) else None


def restarted(previous: int | None, current: int | None) -> bool:
    """Whether these two container clocks belong to two different containers.

    A container's uptime only ever climbs while it lives, so a fall is a new
    container — the single fact this API offers about restarts, the module
    docstring records how that was established.
    """
    if previous is None or current is None:
        return False
    return previous - current > UPTIME_JITTER_SECONDS


def observe(
    pod_id: str,
    get_pod: Callable[[str], dict[str, Any] | None] = get,
    previous_uptime: int | None = None,
) -> Reading:
    """One reading of a pod, judged against the previous reading's container clock.

    `previous_uptime` is the only memory here, and it is the caller's: a crashloop
    is a transition, exactly like completion, and a snapshot cannot name one. Left
    out, this reads a container on its fortieth start as `running` — which is what
    the first A100 canary did for ten minutes while the orchestrator waited on a
    result that was never coming.

    A failed API call returns `unknown` rather than propagating: a transient 502
    from the control plane says nothing about the pod, and treating "we could not
    ask" as "it is done" would terminate a pod mid-run. `unknown` is not a
    completion state, so the pod stays watched until its own deadline.

    Reading the payload happens inside the same `try` as the call, and that is not
    defensive padding. A payload of an unexpected shape is the same non-answer as a
    502 — but outside the `try` it raised through `poll` and `wait_for_any` into a
    `main` with no handler, so the orchestrator died while its pods went on billing.
    Every way of failing to learn the pod's state has to end at the same sentinel.
    """
    try:
        pod = get_pod(pod_id)
        if pod is None:
            return Reading(GONE)
        if pod.get("desiredStatus") in TERMINAL_DESIRED_STATUS:
            return Reading(EXITED)
        runtime = pod.get("runtime")
        if runtime is None:
            return Reading(PENDING)
        uptime = container_uptime(runtime)
        if restarted(previous_uptime, uptime):
            return Reading(RESTARTING, uptime)
        return Reading(RUNNING, uptime)
    except Exception:  # noqa: BLE001 - any transport or shape failure is the same non-answer
        return Reading(UNKNOWN)


def is_finished(status: str, ever_ran: bool) -> bool:
    """Whether this observation means the pod's work is over.

    `ever_ran` is the whole point. A null runtime on a pod that never ran is a pod
    still pulling its image; a null runtime on a pod that was running is a pod
    whose container exited. Without the distinction an entire sweep is terminated
    seconds after launch, every combination recorded as producing no result.

    This answers "does this reading say the work is over", not "is the work over".
    Whether one reading is enough is `PodWatch.poll`'s question — see
    CONFIRMATIONS_REQUIRED.
    """
    if status in (EXITED, GONE, RESTARTING):
        return True
    if status == PENDING:
        return ever_ran
    return False


def is_stated(status: str) -> bool:
    """Whether the API stated the pod is finished rather than us inferring it.

    `EXITED` comes from `desiredStatus`: the control plane says the container will
    not run again. `GONE` and a vanished runtime are inferences from an absence,
    and an absence is exactly what a momentarily inconsistent read looks like.

    `RESTARTING` is here for a different reason: it cannot be confirmed by a second
    reading. The container that replaced the dead one is alive and its clock
    climbs, so the poll after a restart reads `running` again — demanding
    agreement would mean never acting on a slow crashloop at all, and on a fast one
    only by accident. It is also not an absence: two present, climbing clocks that
    disagree is a positive reading, and `UPTIME_JITTER_SECONDS` is what absorbs the
    noise a second reading would have.
    """
    return status in (EXITED, RESTARTING)


@dataclass
class PodOutcome:
    """Why one pod stopped being watched."""

    pod_id: str
    reason: str
    status: str
    ever_ran: bool
    waited_seconds: float
    # The container clock behind the last reading. In the ledger it separates the
    # two pods that both end as `restarted`: one whose container lived seventeen
    # seconds, and one that ran its plan for forty minutes and was bounced after.
    # A null here says the API stopped reporting the clock, which is the state in
    # which restarts cannot be seen at all.
    uptime_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "reason": self.reason,
            "last_status": self.status,
            "ever_ran": self.ever_ran,
            "waited_seconds": round(self.waited_seconds, 1),
            "uptime_seconds": self.uptime_seconds,
        }


@dataclass
class _Tracked:
    started: float
    deadline: float
    ever_ran: bool = False
    last_status: str = PENDING
    # Consecutive readings agreeing on an inferred terminal state.
    agreed: int = 0
    # The container clock of the last reading that carried one. Kept across polls
    # rather than per reading: it is what the next reading is compared against, and
    # it survives a `pending` or `unknown` in between so that a restart hidden
    # behind one failed call is still a fall when the next answer arrives.
    last_uptime: int | None = None


@dataclass
class PodWatch:
    """Watches pods, each against its own deadline.

    A shared deadline charges a pod launched an hour into the sweep for the time
    the sweep spent on everything before it, so the last pods in a queue are
    killed long before they have had their allowance. The clock starts when a pod
    is tracked.
    """

    timeout_seconds: float
    get_pod: Callable[[str], dict[str, Any] | None] = get
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    poll_seconds: float = POLL_SECONDS
    _tracked: dict[str, _Tracked] = field(default_factory=dict)

    def track(self, pod_id: str) -> None:
        now = self.clock()
        self._tracked[pod_id] = _Tracked(started=now, deadline=now + self.timeout_seconds)

    def forget(self, pod_id: str) -> None:
        self._tracked.pop(pod_id, None)

    @property
    def watching(self) -> list[str]:
        return list(self._tracked)

    def poll(self) -> list[PodOutcome]:
        """One sweep over every watched pod. Finished pods stop being watched."""
        outcomes = []
        for pod_id, state in list(self._tracked.items()):
            reading = observe(pod_id, self.get_pod, state.last_uptime)
            status = reading.status
            if reading.uptime_seconds is not None:
                state.last_uptime = reading.uptime_seconds
            if status in (RUNNING, RESTARTING):
                state.ever_ran = True
            finished = is_finished(status, state.ever_ran)
            if finished and not is_stated(status):
                # An inferred terminal state has to repeat before it counts. A
                # failed API call was already debounced by never being a completion
                # state at all, while a pod the API momentarily does not know, or a
                # runtime that momentarily reads null, ended the watch on a single
                # reading. Both are absences, and the same argument covers all
                # three: acting on one reading terminates a live pod and files the
                # combination as producing no result. Waiting costs one poll.
                state.agreed = state.agreed + 1 if state.last_status == status else 1
                finished = state.agreed >= CONFIRMATIONS_REQUIRED
            else:
                state.agreed = 0
            state.last_status = status
            now = self.clock()
            reason = None
            if finished:
                reason = {
                    EXITED: REASON_EXITED,
                    GONE: REASON_GONE,
                    RESTARTING: REASON_RESTARTED,
                }.get(status, REASON_STOPPED)
            elif now >= state.deadline:
                reason = REASON_TIMEOUT
            if reason is None:
                continue
            outcomes.append(
                PodOutcome(
                    pod_id=pod_id,
                    reason=reason,
                    status=status,
                    ever_ran=state.ever_ran,
                    waited_seconds=now - state.started,
                    uptime_seconds=reading.uptime_seconds,
                )
            )
            self.forget(pod_id)
        return outcomes

    def wait_for_any(self) -> list[PodOutcome]:
        """Block until at least one watched pod yields an outcome.

        Always terminates: every watched pod carries a deadline, so the longest
        this can block is the nearest one.
        """
        while self._tracked:
            outcomes = self.poll()
            if outcomes:
                return outcomes
            self.sleep(self.poll_seconds)
        return []
