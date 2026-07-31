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


def host_spec() -> dict[str, Any]:
    """Host identity and hardware. Needed to attribute cross-pod deviation."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
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
