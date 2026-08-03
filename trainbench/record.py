"""Run records: what environment produced a number.

Framework images ship different torch/transformers versions, so the resolved
versions are a confound that has to travel with every result rather than being
assumed uniform.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from trainbench.applied import AppliedState
from trainbench.config import git_state
from trainbench.config_schema import BenchConfig

# Recorded for every run so a pod's hardware is attributable after the fact. Pods
# differ in host CPU and memory bandwidth, which shows up as throughput differences.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "datasets",
    "unsloth",
    "axolotl",
    "ms-swift",
    "sentence-transformers",
    "tevatron",
    "liger-kernel",
    # `fla-core` is the distribution transformers actually imports for Gated
    # DeltaNet — `fla.ops`, `fla.modules` — while `flash-linear-attention` is the
    # thin wrapper (`fla.layers`, `fla.models`) that `==`-pins it
    # (trainbench/axes.py's `FLA_OPS_DISTRIBUTION`/`FLA_DISTRIBUTIONS`, verified
    # 2026-08-03 against `envs/native/uv.lock`: both resolve to 0.5.2). Both are
    # tracked rather than one replacing the other: the pin is what makes them
    # agree today, and the day a wheel breaks it, dropping either name would hide
    # exactly the split this field exists to show.
    "flash-linear-attention",
    "fla-core",
    # Axis-critical. flash-attn decides whether an fa2/3/4 request is real, and
    # causal-conv1d is required alongside fla for the Gated DeltaNet fast path —
    # recording only half of that pair records only half the evidence.
    "flash-attn",
    "causal-conv1d",
    # Also axis-critical, and both silent substitutions if left unrecorded.
    # `kernels` present without `flash-attn` makes transformers rewrite an
    # `attn=fa2` request onto a Hub kernel instead of refusing it — pinned at
    # `FLASH_ATTN_KERNEL_FALLBACK["flash_attention_2"] ==
    # "kernels-community/flash-attn2"`
    # (transformers/modeling_flash_attention_utils.py:65-66, verified against the
    # installed 5.14.1 wheel) — so a run labelled fa2 can be measuring a
    # different kernel depending only on whether this package is installed.
    # `triton` is what `compile.mode` actually compiles through (torch inductor's
    # CUDA backend); a version drift there is a `compile` axis confound the same
    # way a torch version drift is a framework confound.
    "kernels",
    "triton",
    "bitsandbytes",
    "deepspeed",
    "torchvision",
    "trl",
)


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that can shift a measurement."""
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _nvidia_smi() -> dict[str, str] | None:
    query = "name,uuid,memory.total,driver_version"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    first = out.stdout.strip().splitlines()
    if not first:
        return None
    fields = [f.strip() for f in first[0].split(",")]
    return dict(zip(query.split(","), fields, strict=False))


def _cpu_quota() -> float | None:
    """CPU allotted to this container, not to the host machine.

    os.cpu_count() reports the host's cores, so a pod pinned to 8 vCPU on a 128-core
    box reports 128. Dataloading is the axis most sensitive to this, and PLAN.md
    names host vCPU differences as the thing cross-pod deviation must be attributed
    to — recording the wrong number defeats that.

    Both cgroup versions are read. A v1 host is exactly where an unbounded
    os.cpu_count() is most misleading, so answering None there would leave the
    field blank in the case it was added for.
    """
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except OSError:
        quota = []
    if len(quota) == 2 and quota[0] != "max":
        return int(quota[0]) / int(quota[1])
    try:
        v1_quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        v1_period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
    except (OSError, ValueError):
        return None
    # -1 is cgroup v1's "no limit".
    if v1_quota <= 0 or v1_period <= 0:
        return None
    return v1_quota / v1_period


def _total_memory_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        return None
    return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def host_spec() -> dict[str, Any]:
    """Host identity and hardware. Needed to attribute cross-pod deviation."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        # Host cores, process affinity, and cgroup quota are three different
        # numbers; which one bounds the dataloader depends on the deployment.
        "cpu_count_host": os.cpu_count(),
        "cpu_count_process": getattr(os, "process_cpu_count", os.cpu_count)(),
        "cpu_quota": _cpu_quota(),
        "cpu_model": _cpu_model(),
        "memory_total_gb": _total_memory_gb(),
        "cuda_runtime": torch.version.cuda,
        "torch_device_count": torch.accelerator.device_count()
        if hasattr(torch, "accelerator") and torch.accelerator.is_available()
        else 0,
        "gpu": _nvidia_smi(),
        # Set by the orchestrator so a result can be traced back to its pod.
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "runpod_datacenter": os.environ.get("RUNPOD_DC_ID"),
    }


def build_record(
    config: BenchConfig,
    device: torch.device,
    applied: AppliedState | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Everything needed to interpret a number after the run is gone.

    `applied` is what separates "requested fa3" from "ran fa3". Without it the
    result JSON records the request and nothing else, and the whole point of
    trainbench/applied.py — that the two are not the same claim — never reaches
    the file anyone actually reads afterwards. None means the caller did not
    construct a model (an env report), not that the axes were verified.
    """
    git = git_state()
    return {
        # `time.time()`, not a monotonic clock: `scripts/report.py` sorts artifacts
        # from different pods against each other, and a monotonic reading is only
        # comparable within one process. The wall clock can step backwards under
        # NTP, so ordering is correct to within that correction and no finer.
        "recorded_at": time.time(),
        "git_commit": git["commit"],
        # A dirty tree means the recorded commit does not contain the measured code.
        "git_dirty": git["dirty"],
        "git_source": git["source"],
        # Which image produced this number. Without it a result cannot be traced
        # back to the code and package set that generated it.
        "image": os.environ.get("TRAINBENCH_IMAGE"),
        "image_digest": os.environ.get("TRAINBENCH_IMAGE_DIGEST"),
        "config": config.model_dump(mode="json"),
        "applied": applied.to_dict() if applied is not None else None,
        "device": str(device),
        "packages": package_versions(),
        "host": host_spec(),
        **extra,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Atomic write: temp file in the same directory, then replace.

    A pod can vanish mid-write; a half-written result that parses is worse than
    no result at all. The rename is only atomic with respect to data that already
    reached the disk, so the temp file is fsynced before it happens — otherwise
    the rename can land while the contents have not.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # default=str so one unserializable value degrades to its repr instead of
    # losing the entire result file — a probe must always produce output.
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    with open(tmp, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path
