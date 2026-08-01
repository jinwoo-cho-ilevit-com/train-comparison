"""RunPod control, orchestrator side only.

Never imported on a pod: RUNPOD_API_KEY is an account-wide credential and a probe
pod has no business holding one.

Completion is a *transition*, not a snapshot. A pod that has not started yet and a
pod that has finished look almost identical over the API — both report a null
runtime — and the only thing separating them is whether a runtime was ever seen.
Reading one snapshot cannot tell them apart, so this module keeps per-pod state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Models plus a framework venv land on pod-local disk; the default container disk
# is far too small for that.
DEFAULT_CONTAINER_DISK_GB = 120
POLL_SECONDS = 20

# How many consecutive readings must agree before an *inferred* terminal state ends
# a pod's watch. A state the API states outright (`desiredStatus`) needs one. How
# often RunPod's API answers inconsistently is unmeasured here — the reason for the
# delay is that the two outcomes are not symmetric: waiting one extra poll costs
# POLL_SECONDS, acting on a wrong reading costs the run.
CONFIRMATIONS_REQUIRED = 2

# One observation of a pod.
PENDING = "pending"  # created, no runtime yet (provisioning or pulling the image)
RUNNING = "running"  # runtime present
EXITED = "exited"  # RunPod says the container will not run again
GONE = "gone"  # the API no longer knows this pod
UNKNOWN = "unknown"  # the API call itself failed; nothing was learned

# desiredStatus values that mean the container will not come back.
TERMINAL_DESIRED_STATUS = frozenset({"TERMINATED", "EXITED"})

# Why a pod stopped being watched. Recorded per pod, because "the work finished"
# and "we gave up waiting" are different results and only one of them is a datum.
REASON_EXITED = "exited"
REASON_STOPPED = "stopped"  # runtime went from present to absent
REASON_GONE = "gone"
REASON_TIMEOUT = "timeout"


@dataclass(frozen=True)
class PodSpec:
    name: str
    image: str
    gpu_type_id: str
    env: dict[str, str]
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB
    data_center_id: str | None = None


def _client() -> Any:
    import os

    import runpod

    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is not set; run under `infisical run --`")
    runpod.api_key = key
    return runpod


def create(spec: PodSpec) -> dict[str, Any]:
    runpod = _client()
    return runpod.create_pod(
        name=spec.name,
        image_name=spec.image,
        gpu_type_id=spec.gpu_type_id,
        cloud_type="SECURE",
        # No network volume by design: training data must sit on pod-local NVMe or
        # the dataloader axis measures the volume instead of the pipeline. Not
        # attaching one also frees the pod from a single data centre.
        network_volume_id=None,
        container_disk_in_gb=spec.container_disk_gb,
        data_center_id=spec.data_center_id,
        env=spec.env,
        start_ssh=False,
        support_public_ip=False,
    )


def get(pod_id: str) -> dict[str, Any] | None:
    return _client().get_pod(pod_id)


def terminate(pod_id: str) -> None:
    _client().terminate_pod(pod_id)


def observe(pod_id: str, get_pod: Callable[[str], dict[str, Any] | None] = get) -> str:
    """One reading of a pod, with no memory of earlier readings.

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
            return GONE
        if pod.get("desiredStatus") in TERMINAL_DESIRED_STATUS:
            return EXITED
        return RUNNING if pod.get("runtime") is not None else PENDING
    except Exception:  # noqa: BLE001 - any transport or shape failure is the same non-answer
        return UNKNOWN


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
    if status in (EXITED, GONE):
        return True
    if status == PENDING:
        return ever_ran
    return False


def is_stated(status: str) -> bool:
    """Whether the API stated the pod is finished rather than us inferring it.

    `EXITED` comes from `desiredStatus`: the control plane says the container will
    not run again. `GONE` and a vanished runtime are inferences from an absence,
    and an absence is exactly what a momentarily inconsistent read looks like.
    """
    return status == EXITED


@dataclass
class PodOutcome:
    """Why one pod stopped being watched."""

    pod_id: str
    reason: str
    status: str
    ever_ran: bool
    waited_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "reason": self.reason,
            "last_status": self.status,
            "ever_ran": self.ever_ran,
            "waited_seconds": round(self.waited_seconds, 1),
        }


@dataclass
class _Tracked:
    started: float
    deadline: float
    ever_ran: bool = False
    last_status: str = PENDING
    # Consecutive readings agreeing on an inferred terminal state.
    agreed: int = 0


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
            status = observe(pod_id, self.get_pod)
            if status == RUNNING:
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
                reason = {EXITED: REASON_EXITED, GONE: REASON_GONE}.get(status, REASON_STOPPED)
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
