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
from pathlib import Path
from typing import Any

import torch

from trainbench.config import git_commit
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
    "flash-linear-attention",
    # Axis-critical. flash-attn decides whether an fa2/3/4 request is real, and
    # causal-conv1d is required alongside fla for the Gated DeltaNet fast path —
    # recording only half of that pair records only half the evidence.
    "flash-attn",
    "causal-conv1d",
    "transformer-engine",
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
    """
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except OSError:
        return None
    if len(quota) != 2 or quota[0] == "max":
        return None
    return int(quota[0]) / int(quota[1])


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


def build_record(config: BenchConfig, device: torch.device, **extra: Any) -> dict[str, Any]:
    return {
        "git_commit": git_commit(),
        # Which image produced this number. Without it a result cannot be traced
        # back to the code and package set that generated it.
        "image": os.environ.get("TRAINBENCH_IMAGE"),
        "image_digest": os.environ.get("TRAINBENCH_IMAGE_DIGEST"),
        "config": config.model_dump(mode="json"),
        "device": str(device),
        "packages": package_versions(),
        "host": host_spec(),
        **extra,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Atomic write: temp file in the same directory, then replace.

    A pod can vanish mid-write; a half-written result that parses is worse than
    no result at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # default=str so one unserializable value degrades to its repr instead of
    # losing the entire result file — a probe must always produce output.
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path
