"""What a timing run reports, and how it is measured.

Kept apart from `scripts/bench.py` so the arithmetic can be checked against hand
calculations without a model, a GPU, or a training loop. Everything here is pure
except `peak_memory_bytes` and `synchronize`, which are the two places the device
has to be asked.

Deliberately absent: MFU. The standard FLOP formula assumes dense attention over
a fixed hidden size, and all three models under test break it in different ways —
Qwen3.5 is 75% Gated DeltaNet (linear attention), gemma-4 does a per-layer
embedding lookup that is memory traffic rather than matmul, and both VLMs use
sliding-window layers. A single formula would be wrong by an unknown amount per
model, which is worse than no number. tokens/s is the primary figure until a
per-model formula exists with a unit test behind it (PLAN.md Phase 1).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence

import torch

__all__ = [
    "METRIC_DEFINITIONS",
    "percentile",
    "summarise",
    "peak_memory_bytes",
    "reset_peak_memory",
    "StepTimer",
]

# What each counted name means, carried in every summary. An unstated definition
# is how a number gets misread later: `rows` and `samples` differ by a factor of
# two here because a contrastive batch feeds a query and a positive per sample,
# and the two token counts differ by however much padding the batch carried.
# These strings are the answer to "counted how?" travelling with the count.
METRIC_DEFINITIONS: dict[str, str] = {
    "step": (
        "fetching the micro-batches, moving them to the device, forward, backward and the "
        "optimizer step — everything one training step costs, including the data pipeline"
    ),
    "samples": (
        "(query, positive) pairs consumed per step, summed over grad_accum micro-batches. "
        "This is PLAN.md's samples/s"
    ),
    "rows": ("sequences fed to the forward pass per step: queries + positives, so twice `samples`"),
    "tokens": (
        "non-padding positions (attention_mask.sum()). Not what the forward computed on — "
        "padding goes through it too, see `padded_tokens` — and not free of a per-model "
        "constant: the query side carries config.model.instruction_prompt, which is non-null "
        "for exactly one model (docs/CONTRACTS.md §5)"
    ),
    "padded_tokens": (
        "every position the forward computed on, padding included (input_ids.numel())"
    ),
    "images": "images actually handed to the processor per step",
    "images_dropped": (
        "images present in the rows that the processor could not take, per step. Non-zero "
        "means this model read a text-only view of an image corpus"
    ),
}


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation.

    These are step times read by a human comparing two configurations, not a
    distribution to be smoothed. An interpolated p95 reports a duration that no
    step took.
    """
    if not values:
        raise ValueError("no samples; a percentile over nothing is not zero")
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def reset_peak_memory(device: torch.device) -> None:
    """Start the device's high-water mark from here.

    Called after warmup so the reported peak belongs to the measured window.
    Allocations from model construction and autotuning are not what a step costs.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    """Device high-water mark since the last reset, or None off CUDA.

    A high-water mark rather than a sample. Sampling memory at intervals reported
    a *lower* figure at a later point in a monotone accumulation once already in
    this repository (`scripts/prepare_data.py`, `_peak_rss_bytes`), which is only
    possible if the samples were measuring allocator noise.

    None rather than 0 on CPU: zero is a measurement, and this is the absence of
    one. A caller that writes 0 into a result publishes a peak memory of nothing.
    """
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def synchronize(device: torch.device) -> None:
    """Wait for the device before reading a clock.

    CUDA kernels are queued asynchronously, so timing without this measures how
    fast Python got to the next line.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StepTimer:
    """Per-step wall clock with the device sync that makes it mean anything.

    Warmup steps are timed too, and discarded by `summarise` rather than skipped
    here — a discarded sample that was never taken cannot be reported, and the
    number of steps thrown away is part of how a figure was produced.

    `__enter__` synchronises *before* reading the clock, which drains anything the
    caller queued earlier. That only measures honestly if the caller queues no
    device work outside the window: a host-to-device copy issued before `with
    timer` would be drained by this sync and charged to nobody. The measured loop
    (`scripts/bench.py`) fetches and transfers its batches inside the window for
    exactly that reason.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.durations: list[float] = []

    def __enter__(self) -> StepTimer:
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        synchronize(self.device)
        self.durations.append(time.perf_counter() - self._start)


def summarise(
    durations: Sequence[float],
    *,
    discard: int,
    rows_per_step: float,
    tokens_per_step: float | None = None,
    peak_bytes: int | None = None,
    extra_counts: Mapping[str, float] | None = None,
    totals: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The reported figures, from the steps that were kept.

    `discard` is applied here and recorded, not assumed. `torch.compile` with
    autotuning spends its first steps benchmarking kernels, so the discard count
    differs per axis (PLAN.md) — a summary that does not say how many steps it
    threw away cannot be compared with another one.

    `tokens_per_step` is optional and separate from `rows_per_step` because the
    token count is what makes two models comparable and the row count is not:
    tokenisers differ, and image tokens differ again (Qwen scales with pixels,
    gemma-4 is fixed at 280). A run that could not count tokens reports None
    rather than a row-derived stand-in.

    The per-step counts are averages over the measured steps and are floats on
    purpose. Flooring them to int understated every rate by up to one row per
    step, which on a 32-row batch is 3% — the same size as the deviation that
    invalidates a pod baseline (AGENTS.md).

    `extra_counts` gets the same per-step/per-second treatment under its own
    names, and `totals` is merged verbatim for figures that are not rates.
    Both, and the fixed `METRIC_DEFINITIONS`, travel in the summary so a reader
    of the result JSON never has to infer what was counted.
    """
    if discard < 0:
        raise ValueError(f"discard must not be negative, got {discard}")
    kept = list(durations[discard:])
    if not kept:
        raise ValueError(
            f"{len(durations)} step(s) timed and {discard} discarded leaves nothing to "
            "report; a summary over zero steps is not a measurement"
        )
    total = sum(kept)
    summary: dict[str, object] = {
        "steps_timed": len(durations),
        "steps_discarded": discard,
        "steps_measured": len(kept),
        "step_seconds_p50": percentile(kept, 0.50),
        "step_seconds_p95": percentile(kept, 0.95),
        "step_seconds_mean": total / len(kept),
        "rows_per_step": rows_per_step,
        "rows_per_second": rows_per_step * len(kept) / total,
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": (tokens_per_step * len(kept) / total) if tokens_per_step else None,
        "peak_memory_bytes": peak_bytes,
        # Named so a reader does not have to know why it is missing. MFU is absent
        # by decision, not by oversight (see the module docstring).
        "mfu": None,
        "mfu_reason": "no per-model FLOP formula is validated for GDN / PLE / sliding window",
        "metric_definitions": METRIC_DEFINITIONS,
    }
    for name, per_step in (extra_counts or {}).items():
        summary[f"{name}_per_step"] = per_step
        summary[f"{name}_per_second"] = per_step * len(kept) / total
    summary.update(totals or {})
    return summary
