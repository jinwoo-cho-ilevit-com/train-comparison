"""Whether a timing run trained anything, and what it is when it did not.

A finite loss is not evidence that a step happened. Every framework here calls
something like `enable_input_require_grads`, which puts `requires_grad` on the
embedding *output* rather than on a parameter, so the graph stays differentiable
and `loss.backward()` returns normally with every parameter frozen. Three cells
of the 2026-08-02 campaign passed that way with `params_with_grad=0` and
`trainable_params=0`, and a reproduction elsewhere recorded a gradient norm of
zero behind a published 46,000 tokens/second.

`trainbench/probe/steps.py` refuses that state at probe time. This module is the
same question asked of a measured run, and the two must count the same things:

* `trainable_params`, `total_params` and `params_with_grad` are counts of
  parameter **tensors**, never of elements. The question is "did anything
  train", for which an element count is the wrong unit — one frozen 2B model and
  one trainable bias would both be "mostly zero" in elements.
* `grad_norm` is the global L2 norm over `p.grad` of every parameter carrying
  one, taken **after** the backward of the last measured step and before the
  optimizer zeroes them. It is not defined by the probe, which is why it is
  defined here in full: zero means the backward reached no parameter, and it is
  the only one of these counts that a partially-connected graph can fail.

OOM is a fifth outcome and not a slow one. A record that carries no metrics
falls through the report's cell logic to `launch_state` and renders as a
combination nobody attempted, so the memory ceiling has to be stamped on the
record as its own status.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

__all__ = [
    "GATE_FIELDS",
    "STATUS_OOM",
    "gradient_norm",
    "is_oom",
    "oom_status",
    "parameter_counts",
    "training_verdict",
]

# What a record must carry before the gate can be applied to it. Absence is a
# refusal rather than a pass: "we could not check" reading as "it is fine" is
# how the frozen cells were published in the first place.
GATE_FIELDS = (
    "grad_norm",
    "trainable_params",
    "total_params",
    "loss_first",
    "loss_last",
    "peak_memory_bytes",
)

# The record status for a run that hit the memory ceiling. The same string the
# `record-report` boundary pins, because the report distinguishes this from
# "never attempted" by reading it.
STATUS_OOM = "oom"

# Substrings CUDA and the CPU allocator use when they run out. Matched only as a
# fallback behind the typed exception: torch raises `torch.OutOfMemoryError` for
# the device path, and the host allocator raises a plain `RuntimeError`.
_OOM_MARKERS = ("out of memory", "cuda error: out of memory", "cannot allocate memory")

# How many elements `gradient_norm` promotes to float64 at a time, so the
# promotion costs 32 MiB rather than eight bytes per gradient element. Measured
# 2026-08-03 on this host over one 151936x2048 bf16 gradient: 4,980,031,488 bytes
# of max-RSS delta whole-tensor, 41,697,280 chunked.
_NORM_CHUNK_ELEMENTS = 1 << 22


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Parameter tensors: how many exist, how many train, how many got a gradient.

    Counted the same way `probe/steps.py::infonce_backward` counts them, so the
    measurement gate and the probe refusal cannot disagree about what they mean.
    """
    total = 0
    trainable = 0
    with_grad = 0
    for parameter in model.parameters():
        total += 1
        if parameter.requires_grad:
            trainable += 1
            if parameter.grad is not None:
                with_grad += 1
    return {
        "total_params": total,
        "trainable_params": trainable,
        "params_with_grad": with_grad,
    }


def _squared_norm(grad: torch.Tensor) -> torch.Tensor:
    """Sum of squares of one gradient, in float64, without a float64 copy of it.

    Chunked along the first dimension, which slices any strided tensor into views
    — flattening would copy a non-contiguous gradient whole, which is the
    allocation this function exists to avoid.
    """
    if grad.ndim == 0 or grad.numel() <= _NORM_CHUNK_ELEMENTS:
        return torch.linalg.vector_norm(grad, 2, dtype=torch.float64).pow(2)
    rows = max(1, _NORM_CHUNK_ELEMENTS // (grad.numel() // grad.shape[0]))
    total = torch.zeros((), dtype=torch.float64, device=grad.device)
    for chunk in torch.split(grad, rows):
        total = total + torch.linalg.vector_norm(chunk, 2, dtype=torch.float64).pow(2)
    return total


def gradient_norm(model: torch.nn.Module) -> float:
    """Global L2 norm of every gradient currently on the model.

    Read before `optimizer.zero_grad`, because after it there is nothing to
    norm and the answer would be a confident zero. Computed in float64 from
    detached tensors: a bf16 sum over thousands of tensors underflows toward
    zero, which is the exact value this number exists to distinguish.

    The float64 promotion is bounded (`_NORM_CHUNK_ELEMENTS`) rather than taken
    over a whole gradient at once. Its caller runs this between
    `reset_peak_memory` and `peak_memory_bytes`, and inside the block that files
    a failure as `status: oom` — so a promotion the size of the model's largest
    gradient is reported as the step's peak memory, and its own OOM is published
    as the hardware ceiling.
    """
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total = total + _squared_norm(parameter.grad.detach()).cpu()
    return float(total.sqrt())


def is_oom(error: BaseException) -> bool:
    """Whether this exception is the memory ceiling rather than a defect.

    The typed exception first; the message match is the fallback for the host
    allocator and for frameworks that re-raise as `RuntimeError`.
    """
    if isinstance(error, torch.OutOfMemoryError):
        return True
    if not isinstance(error, (RuntimeError, MemoryError)):
        return False
    text = str(error).lower()
    return any(marker in text for marker in _OOM_MARKERS)


def oom_status(error: BaseException, *, peak_bytes: int | None = None) -> dict[str, Any]:
    """The record fields that make an OOM readable as its own outcome.

    Kept apart from `metrics`: a record with a `metrics` block asserts that a
    measured window completed, and this one did not. What it carries instead is
    the ceiling it hit, which is a result — the combination is unmeasurable at
    this batch size on this device, and that is an answer to the study's
    question rather than an absence of one.
    """
    if not is_oom(error):
        raise ValueError(
            f"{type(error).__name__} is not an out-of-memory condition; filing it as one "
            "would publish a defect as a hardware limit"
        )
    return {
        "status": STATUS_OOM,
        "oom": {
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "peak_memory_bytes": peak_bytes,
        },
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _whole_number(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def training_verdict(
    metrics: Mapping[str, Any] | None,
    *,
    peft_mode: str,
    device: str,
) -> tuple[bool, list[str]]:
    """Whether this run's numbers are a speed result, and why not when they are not.

    Every reason is returned rather than the first, because a frozen graph fails
    several of these at once and a caller that saw only `grad_norm` would fix the
    norm and re-publish the same run.

    `peft_mode` decides what the parameter counts should look like: a full
    finetune in which some tensors are frozen is a different workload from the
    one its label claims, and a LoRA run in which everything trains has no
    adapter narrowing anything.
    """
    if not isinstance(metrics, Mapping):
        return False, ["no `metrics`: nothing was reported to check"]
    reasons: list[str] = []

    grad_norm = metrics.get("grad_norm")
    if not _finite_number(grad_norm):
        reasons.append(f"`grad_norm`={grad_norm!r} is absent or not finite")
    elif grad_norm <= 0:
        reasons.append(f"`grad_norm`={grad_norm}: the backward reached no parameter")

    trainable = metrics.get("trainable_params")
    total = metrics.get("total_params")
    if not _whole_number(trainable) or trainable <= 0:
        reasons.append(f"`trainable_params`={trainable!r}: this model cannot learn")
    if not _whole_number(total) or total <= 0:
        # `GATE_FIELDS` names this one, and absence there is a refusal rather
        # than a pass. Skipping the peft check instead is how a full finetune
        # that froze most of its tensors gets published as a full finetune.
        reasons.append(
            f"`total_params`={total!r}: the peft.mode check has nothing to compare "
            f"`trainable_params`={trainable!r} against"
        )
    elif _whole_number(trainable) and trainable > 0:
        if peft_mode == "full" and trainable != total:
            reasons.append(
                f"peft.mode=full but {trainable} of {total} parameter tensors train; "
                "a full finetune that froze part of the model is a different workload"
            )
        if peft_mode in ("lora", "qlora") and trainable >= total:
            reasons.append(
                f"peft.mode={peft_mode} but {trainable} of {total} parameter tensors train; "
                "the adapter did not narrow anything"
            )

    first, last = metrics.get("loss_first"), metrics.get("loss_last")
    if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
        reasons.append("`loss_first`/`loss_last` are absent")
    elif not (math.isfinite(first) and math.isfinite(last)):
        reasons.append(f"loss is not finite: {first} -> {last}")
    elif last >= first:
        reasons.append(f"loss did not fall: {first} -> {last}")

    if str(device).startswith("cuda"):
        peak = metrics.get("peak_memory_bytes")
        if not _whole_number(peak) or peak <= 0:
            reasons.append(f"`peak_memory_bytes`={peak!r} on a CUDA run")

    return not reasons, reasons
