"""RunPod control, orchestrator side only.

Never imported on a pod: RUNPOD_API_KEY is an account-wide credential and a probe
pod has no business holding one.

Completion is a *transition*, not a snapshot. A pod that has not started yet and a
pod that has finished look almost identical over the API — both report a null
runtime — and the only thing separating them is whether a runtime was ever seen.
Reading one snapshot cannot tell them apart, so this module keeps per-pod state.

Three transports, on purpose. Creation goes over RunPod's REST API; reading sends
its own GraphQL document; terminating stays on the `runpod` SDK. None of that is a
preference:

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
  GraphQL returned a non-null runtime for a RUNNING pod and a null one for an
  EXITED pod. Moving `get` to REST would make every pod read as pending forever.
- Reading cannot use the SDK either, and that is the correction the first real pod
  forced. `runpod.get_pod` selects `runtime { ports { … } }` and nothing else, so
  the container clock this module is built on never arrived — see POD_QUERY.

Writing the read by hand cost one thing the SDK gave away, and the third pod paid
for it: `requests` sends a User-Agent and urllib does not, and the GraphQL host
refuses urllib's default outright. See USER_AGENT.

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

It is also *coarse*, and it does not fall monotonically. Those two measurements
together decide how it may be read. Three healthy running pods, read five times
twenty-two seconds apart (2026-08-02): the value advanced in steps of exactly
29-31 seconds and stood still across a whole poll on every one of the three. It is
served from a cache about half a minute deep, so a *failure to climb* over one
poll is not evidence — a liveness test built on that would terminate live pods.

And read against a container that was actually cycling (2026-08-02, pod
xchraazlhvqt6y, ~12s apart while the container restarted):

    RUNNING 0   RUNNING -11   RUNNING -11   RUNNING -9   RUNNING -9

The clock goes *negative* — the cached `now` trails the `startedAt` the newest
container has just written — and between samples it rises as often as it falls. So
a test that compares consecutive readings is decided by where the poll happens to
land: `0 -> -11` fires, `-11 -> -9` does not, and over six minutes of that pod
nothing fired at all. Comparing against wall time instead takes the timing out of
it. A container's clock climbs one second per second, so from the first reading
that carried one, every later reading must report at least that value plus the
elapsed time, less one cache depth. `outran_its_clock` is that floor;
`restarted` stays for the fall too large to be cache noise, which fires before the
floor has had time to rise.

The floor is anchored on the first clock this module *saw*, not on the pod's rental
age, and that is deliberate. Between renting a pod and starting its container sits
an image pull — 10-140s across the account's other pods, never measured for this
repository's multi-gigabyte images — so a launch-anchored floor needs an allowance
nobody here has measured, and one wide enough to be safe would also detect later
than this one does.
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

# Every request this module sends identifies itself, and that is a transport
# requirement rather than politeness. urllib's default signature —
# `Python-urllib/3.13` — is refused at the edge by api.runpod.io/graphql with
# `403: error code: 1010`, a Cloudflare rule on the client signature and not on
# the credential or the document. Measured 2026-08-02 against the live API, same
# key, same body, same minute:
#
#     Python-urllib/3.13  -> 403, error code: 1010
#     curl/8.0            -> 200
#     trainbench/1.0      -> 200
#
# Reading used to go through the `runpod` SDK, which sends `requests`' own
# User-Agent; replacing it with a hand-written urllib request took the reads from
# succeeding-without-the-clock to not succeeding at all, and `observe` turned every
# one of them into `unknown` in silence.
#
# Set in `send`, so on every request, rather than only on the GraphQL ones.
# rest.runpod.io answered the default with 200 in that same measurement, so REST
# does not need it today — but which of a provider's hosts has an edge rule
# enabled is the provider's setting, and this is the module's one door to the
# network. A per-host header would be a second list to keep in step with the hosts.
USER_AGENT = "trainbench/1.0"

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

# How often `runtime.uptimeInSeconds` refreshes, measured 2026-08-02: three healthy
# pods, five reads twenty-two seconds apart, steps of 29-31s with a full poll of no
# movement on every one. Both thresholds below are written in terms of it, because
# both are really the same question — how far the reported clock can sit from the
# container's real age without anything being wrong.
CLOCK_REFRESH_SECONDS = 31

# How far `runtime.uptimeInSeconds` may fall between two readings and still be the
# same container. One refresh: two readings can be served from snapshots taken that
# far apart, so a fall of that size says nothing at all about the container, and
# the earlier one-second threshold would have read cache noise as a crashloop. A
# restart puts the entire life of the dead container between the two readings.
# Smaller falls are not conceded — they are left to the floor below, which does not
# depend on where the poll lands.
UPTIME_JITTER_SECONDS = CLOCK_REFRESH_SECONDS

# How far behind wall time a living container's reported clock may fall over a
# whole watch. Only the *current* reading's staleness can hurt: a stale anchor
# reports less than the container's real age and so lowers the floor. One refresh
# is therefore the worst legitimate case, and this is three of them. Unlike a stall
# test it does not accumulate — a healthy clock that stands still for one poll is
# still minutes clear of the floor several polls later.
CLOCK_LAG_ALLOWANCE_SECONDS = 3 * CLOCK_REFRESH_SECONDS

# desiredStatus values that mean the container will not come back.
TERMINAL_DESIRED_STATUS = frozenset({"TERMINATED", "EXITED"})

# How long a watch may learn nothing before it stops counting as a watch.
#
# `unknown` is deliberately not a completion state — a 502 from the control plane
# says nothing about the pod — but with no ceiling at all, "the reads are failing"
# and "the pod is busy" are the same thing to every caller. That is the third time
# this class has cost a pod: a Cloudflare rule answered 403 to every poll for ten
# minutes, `observe` returned `unknown` each time, the ledger recorded
# `outcome: null`, and the pod was deleted by hand.
#
# Wall time rather than a poll count, so the ceiling does not move when
# `poll_seconds` does. The value is a judgement and not a measurement — how long
# RunPod's control plane stays unreadable is not measured here — set far above any
# single-poll blip and far below a pod deadline (60 minutes by default), so a
# transport that is broken rather than busy is named while the pod still has most
# of its allowance left.
#
# Ending the watch terminates the pod, and that is the intended trade. Blind, we
# cannot tell a finished pod from a running one, so leaving it costs unbounded
# billing; the pod also carries its own self-kill deadline, so this is the earlier
# of two stops rather than the only one.
UNREADABLE_CEILING_SECONDS = 5 * 60

# Why a pod stopped being watched. Recorded per pod, because "the work finished"
# and "we gave up waiting" are different results and only one of them is a datum.
REASON_EXITED = "exited"
REASON_STOPPED = "stopped"  # runtime went from present to absent
REASON_GONE = "gone"
REASON_TIMEOUT = "timeout"
# Nothing was learned about this pod for long enough that the watch was not a
# watch. See UNREADABLE_CEILING_SECONDS.
REASON_UNREADABLE = "unreadable"
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
        headers={
            # First, so a Request may state its own; last would silently un-set the
            # one thing between these reads and a 403.
            "User-Agent": USER_AGENT,
            **request.headers,
            "Authorization": f"Bearer {_api_key()}",
        },
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


# The read, written out here rather than taken from the SDK, and the selection set
# is the whole point. `runpod.get_pod` asks for `runtime { ports { … } }` — so the
# runtime it returns is non-null, `container_uptime` finds no `uptimeInSeconds` in
# it, and every container reads as `running` on its fortieth start. That is what
# the first real pod did: nine probe runs in four minutes while the orchestrator
# waited, and `uptime_seconds: null` in its ledger. Measured 2026-08-02 against
# pod 0dw2kaljoo8pio in the same minute: the SDK's document returned `runtime`
# keys `['ports']`, this one returned `uptimeInSeconds: 1188465`.
#
# Narrow on purpose beyond that. The SDK's document also selects `env`, which on
# our pods carries an Infisical token, and `observe`'s payload is one traceback
# away from a log. What is not asked for cannot leak.
GRAPHQL_URL = "https://api.runpod.io/graphql"
POD_QUERY = """
query trainbenchPod($id: String!) {
  pod(input: {podId: $id}) {
    id
    desiredStatus
    runtime {
      uptimeInSeconds
    }
  }
}
"""


def read_request(pod_id: str) -> Request:
    """The read on the wire. The id travels as a GraphQL variable.

    The SDK interpolates `pod_id` into its query the same unescaped way it
    interpolates env values. That was never reachable here — the only value that
    reaches it is an id RunPod generated — but a variable costs nothing and closes
    the question rather than arguing it.
    """
    return Request(
        method="POST",
        url=GRAPHQL_URL,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"query": POD_QUERY, "variables": {"id": pod_id}}).encode(),
    )


def get(pod_id: str, transport: Transport = send) -> dict[str, Any] | None:
    """Read one pod, or None if the API no longer knows it.

    GraphQL answers a failed query with HTTP 200 and an `errors` key, so the
    envelope has to be unwrapped rather than returned: left alone, an error body
    would reach `observe` as a pod with no `desiredStatus` and no `runtime` and be
    read as `pending` — a pod that is still pulling its image, forever.
    """
    payload = transport(read_request(pod_id))
    if errors := payload.get("errors"):
        message = errors[0].get("message") if isinstance(errors[0], dict) else errors[0]
        raise RuntimeError(f"RunPod GraphQL read of {pod_id}: {str(message)[:ERROR_BODY_CHARS]}")
    return (payload.get("data") or {}).get("pod")


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

    `detail` is why an `unknown` is unknown, and it exists because the sentinel on
    its own is unactionable. Ten minutes of `unknown` looked identical whether the
    control plane was blinking or the edge was refusing every request outright —
    and it was the second, with `403: error code: 1010` in an exception nobody
    kept. It is None for every reading that learned something.
    """

    status: str
    uptime_seconds: int | None = None
    detail: str | None = None


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
    container. It is the fast test rather than the reliable one: a fall this size
    is unambiguous the moment it is seen, and whether it is ever seen depends on
    where the poll lands relative to the restart. `outran_its_clock` is what does
    not depend on that.
    """
    if previous is None or current is None:
        return False
    return previous - current > UPTIME_JITTER_SECONDS


def outran_its_clock(anchor: int | None, elapsed_seconds: float, current: int | None) -> bool:
    """Whether the pod has been watched longer than its container's clock accounts for.

    `anchor` is the first clock this watch saw and `elapsed_seconds` is the wall
    time since. A living container's clock climbs one second per second, so it owes
    the anchor plus everything that has passed, and the only thing that can put it
    legitimately below that is the cache — one refresh, allowed for three times
    over. Reporting less than that means the container being clocked is not the one
    the anchor was taken from.

    This is the rule the observer needed and did not have. Its inputs are one
    reading and a wall clock, so no poll can land between the restart and the
    evidence: a container clock reading -11 eight minutes after the clock it was
    anchored on is a restart on that single reading, and stays one on every reading
    after it.
    """
    if anchor is None or current is None:
        return False
    return current + CLOCK_LAG_ALLOWANCE_SECONDS < anchor + elapsed_seconds


def observe(
    pod_id: str,
    get_pod: Callable[[str], dict[str, Any] | None] = get,
    previous_uptime: int | None = None,
    anchor_uptime: int | None = None,
    since_anchor_seconds: float = 0.0,
) -> Reading:
    """One reading of a pod, judged against what earlier readings established.

    The memory is the caller's: a crashloop is a transition, exactly like
    completion, and a snapshot cannot name one. Left out, this reads a container on
    its fortieth start as `running` — which is what the first A100 canary did for
    ten minutes while the orchestrator waited on a result that was never coming.

    Two kinds of memory, because the two restart tests need different things.
    `previous_uptime` is the last reading's clock, for the fall. `anchor_uptime`
    with the wall time since it was taken is the floor, and it is the one that does
    not care when the poll lands.

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
        if restarted(previous_uptime, uptime) or outran_its_clock(
            anchor_uptime, since_anchor_seconds, uptime
        ):
            return Reading(RESTARTING, uptime)
        return Reading(RUNNING, uptime)
    except Exception as exc:  # noqa: BLE001 - any transport or shape failure is the same non-answer
        # The sentinel is the same for every way of not learning the state; the
        # reason is not, and losing it is how a 403 on every single request read as
        # a pod that was merely slow. `type` as well as the message, because a
        # shape failure raises something with an unhelpful text (`KeyError: 'pod'`).
        return Reading(UNKNOWN, detail=f"{type(exc).__name__}: {exc}"[:ERROR_BODY_CHARS])


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
    # Two container clocks, because a crashloop makes the last one uninformative:
    # it dates the container that ended the watch, which is seconds old or, when
    # the cache is behind the restart, negative. `peak_uptime_seconds` dates the
    # longest-lived container the watch saw, and that is what separates the two
    # pods that both end as `restarted` — one bouncing every seventeen seconds
    # without ever finishing, one that ran its plan for forty minutes and was
    # restarted after publishing it. Neither field says the plan finished; the
    # uploaded result says that. A null in both says the API stopped reporting the
    # clock, which is the state in which restarts cannot be seen at all.
    uptime_seconds: int | None = None
    peak_uptime_seconds: int | None = None
    # How much of this watch learned nothing, and the last reason it learned
    # nothing. Set on every outcome that had an unreadable poll, not only on
    # `REASON_UNREADABLE`: a `timeout` reached through ten minutes of 403s and a
    # `timeout` reached through ten minutes of clean reads are different results,
    # and the ledger recorded them identically. None when every read answered.
    unreadable: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "reason": self.reason,
            "last_status": self.status,
            "ever_ran": self.ever_ran,
            "waited_seconds": round(self.waited_seconds, 1),
            "uptime_seconds": self.uptime_seconds,
            "peak_uptime_seconds": self.peak_uptime_seconds,
            "unreadable": self.unreadable,
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
    # The first clock this watch saw and when, which is what `outran_its_clock`
    # measures every later reading against. Never re-anchored: the bound it gives
    # only tightens as the watch runs, and moving it forward onto a reading of the
    # replacement container is how a crashloop would keep resetting the evidence.
    anchor_uptime: int | None = None
    anchor_at: float | None = None
    # The longest-lived container seen, for the ledger. See PodOutcome.
    peak_uptime: int | None = None
    # The blindness. `unknown_since` is when the current unbroken run of
    # unreadable polls began and is cleared by any reading that learned
    # something, so it measures a spell rather than a total; `unknown_total`
    # counts every unreadable poll of the watch, so a pod read through repeated
    # short outages still says so in the ledger.
    unknown_since: float | None = None
    unknown_total: int = 0
    last_detail: str | None = None


def _unreadable_summary(state: _Tracked, now: float) -> str | None:
    """What this watch failed to learn, in one sentence for the ledger.

    None when every poll answered, so the field's presence is itself the signal.
    """
    if not state.unknown_total:
        return None
    unbroken = ""
    if state.unknown_since is not None:
        unbroken = f", the last {round(now - state.unknown_since)}s of it unbroken"
    return (
        f"{state.unknown_total} of this watch's polls learned nothing{unbroken}. "
        f"Last failure: {state.last_detail}"
    )


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
    # Called with `(pod_id, message)` on every poll that learned nothing, before
    # the ceiling has been reached. `wait_for_any` loops inside itself, so without
    # this the caller holds no thread of control between the launch and the
    # outcome, and a watch that was blind from its first poll had nowhere to say
    # so for ten minutes. Silence is the default because `PodWatch` has no console
    # of its own; the orchestrator supplies one.
    on_blind: Callable[[str, str], None] = lambda _pod_id, _message: None
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
            now = self.clock()
            since_anchor = now - state.anchor_at if state.anchor_at is not None else 0.0
            reading = observe(
                pod_id, self.get_pod, state.last_uptime, state.anchor_uptime, since_anchor
            )
            status = reading.status
            if status == UNKNOWN:
                state.unknown_total += 1
                state.last_detail = reading.detail
                if state.unknown_since is None:
                    state.unknown_since = now
                self.on_blind(
                    pod_id,
                    f"cannot see {pod_id}: {round(now - state.unknown_since)}s of "
                    f"unreadable polls ({state.unknown_total} in this watch), "
                    f"{round(UNREADABLE_CEILING_SECONDS - (now - state.unknown_since))}s "
                    f"before the watch gives up. Last failure: {reading.detail}",
                )
            else:
                state.unknown_since = None
            if reading.uptime_seconds is not None:
                state.last_uptime = reading.uptime_seconds
                if state.anchor_at is None:
                    state.anchor_uptime, state.anchor_at = reading.uptime_seconds, now
                if state.peak_uptime is None or reading.uptime_seconds > state.peak_uptime:
                    state.peak_uptime = reading.uptime_seconds
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
            reason = None
            if finished:
                reason = {
                    EXITED: REASON_EXITED,
                    GONE: REASON_GONE,
                    RESTARTING: REASON_RESTARTED,
                }.get(status, REASON_STOPPED)
            elif (
                state.unknown_since is not None
                and now - state.unknown_since >= UNREADABLE_CEILING_SECONDS
            ):
                # Checked before the deadline, so a watch that has learned nothing
                # is reported as unreadable rather than as a pod that ran out of
                # time — a distinction the ledger had no way to make.
                reason = REASON_UNREADABLE
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
                    peak_uptime_seconds=state.peak_uptime,
                    unreadable=_unreadable_summary(state, now),
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
