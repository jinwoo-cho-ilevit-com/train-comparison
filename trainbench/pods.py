"""RunPod control, orchestrator side only.

Never imported on a pod: RUNPOD_API_KEY is an account-wide credential and a probe
pod has no business holding one.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

# Models plus a framework venv land on pod-local disk; the default container disk
# is far too small for that.
DEFAULT_CONTAINER_DISK_GB = 120
POLL_SECONDS = 20


@dataclass(frozen=True)
class PodSpec:
    name: str
    image: str
    gpu_type_id: str
    env: dict[str, str]
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB
    data_center_id: str | None = None


def _client() -> Any:
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


def is_finished(pod: dict[str, Any] | None) -> bool:
    """A pod is done when it has no running runtime left.

    The entrypoint always exits 0, so completion is inferred from the pod state
    rather than from an exit code.
    """
    if pod is None:
        return True
    if pod.get("desiredStatus") in ("TERMINATED", "EXITED"):
        return True
    return pod.get("runtime") is None and pod.get("lastStatusChange") is not None


def wait_for_any(pod_ids: list[str], timeout_seconds: int) -> list[str]:
    """Block until at least one pod finishes; return the finished ids.

    Returns everything still pending on timeout so a hung pod cannot stall a sweep
    forever — the caller terminates them and records the combination as untested.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        done = [pid for pid in pod_ids if is_finished(get(pid))]
        if done:
            return done
        time.sleep(POLL_SECONDS)
    return pod_ids
