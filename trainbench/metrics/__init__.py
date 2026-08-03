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
from typing import TYPE_CHECKING

import torch

from trainbench.metrics.statistics import (
    AGGREGATORS,
    CudaEventTimer,
    aggregate,
    build_timer,
    repeat_seeds,
    stdev,
)
from trainbench.metrics.validity import (
    GATE_FIELDS,
    STATUS_OOM,
    gradient_norm,
    is_oom,
    oom_status,
    parameter_counts,
    training_verdict,
)

if TYPE_CHECKING:
    from trainbench.config_schema import BenchConfig

__all__ = [
    "AGGREGATORS",
    "GATE_FIELDS",
    "METRIC_DEFINITIONS",
    "STATUS_OOM",
    "CudaEventTimer",
    "StepTimer",
    "aggregate",
    "build_timer",
    "gradient_norm",
    "is_oom",
    "oom_status",
    "parameter_counts",
    "peak_memory_bytes",
    "percentile",
    "repeat_seeds",
    "reset_peak_memory",
    "stdev",
    "summarise",
    "training_verdict",
]

# Counter names `summarise` produces itself. An `extra_counts` entry that
# reuses one would overwrite the figure computed from the named parameter, so
# the two counts of the same thing would land under one key and the loser would
# be invisible.
_RESERVED_COUNTS = frozenset({"rows", "tokens", "padded_tokens"})

# Substrings that mark a name as a rate rather than a counter. `summarise` is
# the only place a rate is allowed to come into existence, because it is the
# only place that knows the time the counts were divided by. A rate arriving
# from a caller is a framework's own tokens/sec under a different name, which
# the `collate-metrics` boundary forbids from crossing and this forbids from
# landing.
_RATE_MARKERS = ("per_second", "per_sec", "_rate", "throughput", "tokens_per", "samples_per")

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
    # The validity gate's counts. Written here so the definition travels in the
    # result rather than living only in trainbench/metrics/validity.py, and so
    # the unit is legible beside the number: `trainbench/probe/steps.py` reports
    # the same names at probe time and the two must not drift apart.
    "grad_norm": (
        "global L2 norm over `p.grad` of every parameter carrying one, taken after the "
        "backward of the last measured step and before the optimizer zeroes them. Zero "
        "means the backward reached no parameter, which a finite loss does not rule out"
    ),
    "trainable_params": (
        "parameter *tensors* with requires_grad, not elements. The question is whether "
        "anything trained, for which an element count is the wrong unit"
    ),
    "total_params": "parameter tensors the model holds, trainable or not",
    "params_with_grad": (
        "trainable parameter tensors that actually received a gradient. Below "
        "`trainable_params` means part of the model is detached from the loss"
    ),
    "peak_memory_bytes": (
        "device high-water mark since the reset that follows warmup, so it belongs to the "
        "measured window rather than to model construction and autotuning. None off CUDA, "
        "because zero bytes is a measurement and this is the absence of one"
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

    An accelerator queues kernels asynchronously, so timing without this measures
    how fast Python got to the next line rather than how long the device took.
    `device.type == "cuda"` used to be the only branch here, which made every step
    time silently meaningless on any other accelerator: `trainbench/device.py`
    resolves `mps` via `torch.accelerator.current_accelerator()` on this laptop,
    and the old body no-opped on it exactly as it does on `cpu` — where a no-op is
    correct, because CPU ops are synchronous already.

    `torch.accelerator.synchronize` is the device-agnostic form (cuda, mps, xpu),
    so one call covers every accelerator this process could resolve to instead of
    one `torch.cuda`-shaped branch per kind. It raises if `device` is not the kind
    of accelerator this process actually has (`ValueError: ... doesn't match the
    current accelerator ...`), which this function turns into a named refusal
    rather than letting it read as an obscure crash: a `StepTimer` built for a
    device this process cannot synchronize must refuse to time it, not report a
    wall-clock number under a device-measurement label.
    """
    if device.type == "cpu":
        return
    current = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
    if current is None or current.type != device.type:
        raise RuntimeError(
            f"cannot synchronize device={device}: torch.accelerator reports "
            f"{current!r} as the current accelerator. Timing {device} here would "
            "either no-op (wrong: kernels queued on it would not be drained before "
            "the clock reads) or synchronize the wrong device — StepTimer must not "
            "be used to time a device this process cannot actually synchronize."
        )
    torch.accelerator.synchronize(device)


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


def _measurement_block(config: BenchConfig | None) -> dict[str, object]:
    """How this figure was produced, as the run declared it.

    A summary built without a config still carries a block, marked
    `declared: false` and filled with the schema's defaults. The alternative —
    omitting it — makes an undeclared run indistinguishable from one that
    declared the defaults, and the whole point of moving these off the harness
    was that a reader can tell.
    """
    from trainbench.config_schema import MeasurementConfig

    declared = config is not None
    measurement = config.measurement if config is not None else MeasurementConfig()
    return {
        "declared": declared,
        "repeats": measurement.repeats,
        "instrument": measurement.instrument,
        "aggregate": measurement.aggregate,
        "trim_fraction": measurement.trim_fraction,
        "seed_policy": measurement.seed_policy,
        "throughput_denominator": measurement.throughput_denominator,
        "baseline_tolerance": measurement.baseline_tolerance,
        # "uncalibrated" is the current answer and it is a finding, not a
        # placeholder: the 3% in AGENTS.md has no source, and contention alone
        # has moved a step-time standard deviation by 30x elsewhere.
        "baseline_tolerance_status": measurement.tolerance_status,
    }


def summarise(
    durations: Sequence[float],
    *,
    discard: int,
    rows_per_step: float,
    tokens_per_step: float | None = None,
    padded_tokens_per_step: float | None = None,
    peak_bytes: int | None = None,
    extra_counts: Mapping[str, float] | None = None,
    totals: Mapping[str, object] | None = None,
    config: BenchConfig | None = None,
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

    `padded_tokens_per_step` is a named parameter rather than one more entry in
    `extra_counts` because it is the other candidate denominator, not an
    incidental count: which of the two divides the step time reverses the
    ranking of the `dataloader.packing` axis, so the choice is declared in
    `config.measurement.throughput_denominator` and both counts are carried. The
    schema pins that field to `tokens` until a report renders the other rate, so
    the denominator named here is the one the published tables ranked on.

    `extra_counts` gets the same per-step/per-second treatment under its own
    names, and `totals` is merged verbatim for figures that are not rates.
    Neither may carry a name that reads as a rate: this function is the only
    place a rate comes into existence, because it is the only place that knows
    the time the counts were divided by, and a rate arriving from a caller is a
    framework's own tokens/sec by another name.

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
    extra_counts = dict(extra_counts or {})
    totals = dict(totals or {})
    _refuse_rates_and_collisions(extra_counts, totals, padded_tokens_per_step)

    measurement = _measurement_block(config)
    total = sum(kept)
    mean = total / len(kept)
    counts: dict[str, float | None] = {
        "tokens": tokens_per_step,
        "padded_tokens": padded_tokens_per_step
        if padded_tokens_per_step is not None
        else extra_counts.get("padded_tokens"),
    }
    denominator = str(measurement["throughput_denominator"])
    if config is not None and counts[denominator] is None:
        raise ValueError(
            f"measurement.throughput_denominator={denominator} but this run counted no "
            f"{denominator}; the declared denominator of every rate in the result was "
            "never measured"
        )

    summary: dict[str, object] = {
        "steps_timed": len(durations),
        "steps_discarded": discard,
        "steps_measured": len(kept),
        "step_seconds_p50": percentile(kept, 0.50),
        "step_seconds_p95": percentile(kept, 0.95),
        "step_seconds_mean": mean,
        # The declared statistic, beside the fixed-definition mean rather than in
        # place of it: every rate below divides by the mean, so replacing it would
        # make the rates and the headline step time disagree about the same run.
        "step_seconds_aggregate": aggregate(
            kept,
            method=str(measurement["aggregate"]),
            trim_fraction=float(measurement["trim_fraction"]),  # type: ignore[arg-type]
        ),
        "step_seconds_stdev": stdev(kept),
        "rows_per_step": rows_per_step,
        "rows_per_second": rows_per_step / mean,
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": (tokens_per_step / mean) if tokens_per_step else None,
        "peak_memory_bytes": peak_bytes,
        # Which of the two token counts is the headline throughput. A name, not a
        # third rate: both rates are already here, and a duplicated number is one
        # more thing that can disagree with itself.
        "throughput_denominator": denominator,
        # Whether the profiler was on. A profiled run's step time is not
        # reportable, and a record that cannot say which it was cannot be filtered.
        "profiled": config.run.profiler if config is not None else None,
        "measurement": measurement,
        # Named so a reader does not have to know why it is missing. MFU is absent
        # by decision, not by oversight (see the module docstring).
        "mfu": None,
        "mfu_reason": "no per-model FLOP formula is validated for GDN / PLE / sliding window",
        "metric_definitions": METRIC_DEFINITIONS,
    }
    if padded_tokens_per_step is not None:
        extra_counts["padded_tokens"] = padded_tokens_per_step
    for name, per_step in extra_counts.items():
        summary[f"{name}_per_step"] = per_step
        summary[f"{name}_per_second"] = per_step / mean
    summary.update(totals)
    return summary


def _refuse_rates_and_collisions(
    extra_counts: Mapping[str, float],
    totals: Mapping[str, object],
    padded_tokens_per_step: float | None,
) -> None:
    """Guard the two ways a caller can corrupt the summary without an error.

    A rate handed in is a framework's own figure entering through the door that
    exists to keep it out. A counter name this function already produces is
    worse than a duplicate: the loser of the collision leaves no trace, so a
    result carries one of two disagreeing counts and says nothing about the other.
    """
    offenders = sorted(
        name for name in (*extra_counts, *totals) if any(marker in name for marker in _RATE_MARKERS)
    )
    if offenders:
        raise ValueError(
            f"{offenders} name a rate. `summarise` divides the counters by the time it "
            "measured; a rate arriving here was computed against a time this side never "
            "read, which is a framework's own tokens/sec by another name."
        )
    if padded_tokens_per_step is not None and "padded_tokens" in extra_counts:
        raise ValueError(
            "padded_tokens arrived both as the named parameter and in extra_counts; one of "
            "the two counts would be silently overwritten"
        )
    # `padded_tokens` is exempt: it is a named parameter *and* a legal
    # `extra_counts` key, and the line above is what stops it being both at once.
    collisions = sorted((_RESERVED_COUNTS & set(extra_counts)) - {"padded_tokens"})
    if collisions:
        raise ValueError(
            f"extra_counts reuses {collisions}, which `summarise` computes from its own "
            "named parameters; the two counts would land under one key"
        )
