"""How many samples a figure is made of, and how they become one number.

Every choice here was a constant somewhere before it was a knob: the aggregation
was an arithmetic mean written into `summarise`, the instrument was
`time.perf_counter` written into `StepTimer`, and the repeat count was one
because nothing ever asked for a second run. A constant that differs between two
adapters is a difference the result attributes to the axis under test, so each
one is read off `config.measurement` and recorded beside the number it produced.

No default here is calibrated. What warmup, how many repeats and which
aggregation this study should use are answers a pod produces after measuring the
noise floor; until then the schema's job is to make the answer expressible and
recorded, not to guess it.
"""

from __future__ import annotations

import secrets
import statistics as _stdlib
from collections.abc import Sequence
from typing import Any

import torch

__all__ = [
    "AGGREGATORS",
    "CudaEventTimer",
    "aggregate",
    "build_timer",
    "repeat_seeds",
    "stdev",
]

# The largest seed a repeat may draw. 2**32 keeps it inside what numpy's legacy
# seeding accepts, which is the narrowest of the generators `set_seed` feeds.
SEED_BITS = 32


def _mean(samples: Sequence[float]) -> float:
    return sum(samples) / len(samples)


def _median(samples: Sequence[float]) -> float:
    """Nearest-rank, for the reason `percentile` gives: an interpolated middle
    reports a duration that no step took."""
    ordered = sorted(samples)
    return ordered[(len(ordered) - 1) // 2]


def _trimmed_mean(samples: Sequence[float], trim_fraction: float) -> float:
    """Mean after dropping `trim_fraction` of the samples from each end.

    Step-time distributions have a long right tail — one page fault, one
    allocator growth, one kernel that recompiled — and an arithmetic mean lets a
    single such step move the reported figure by more than the effect being
    measured.
    """
    ordered = sorted(samples)
    cut = int(len(ordered) * trim_fraction)
    kept = ordered[cut : len(ordered) - cut]
    if not kept:
        raise ValueError(
            f"trim_fraction={trim_fraction} removes all {len(ordered)} sample(s) from both "
            "ends; a mean over nothing is not a measurement"
        )
    return _mean(kept)


def _olympic(samples: Sequence[float]) -> float:
    """Drop the single fastest and the single slowest, average the rest.

    MLPerf's scoring for the Small-LLM finetuning benchmark. It differs from a
    trimmed mean in being defined by a count rather than a fraction, so it
    removes exactly one outlier at each end whatever the sample size.
    """
    if len(samples) < 3:
        raise ValueError(
            f"olympic scoring drops the fastest and the slowest sample, so it needs at "
            f"least 3; got {len(samples)}"
        )
    ordered = sorted(samples)
    return _mean(ordered[1:-1])


# The aggregations `measurement.aggregate` may name. Kept as a mapping rather
# than an if-chain so the schema's Literal and this table are checkable against
# each other — a value the schema offers and this does not implement would
# otherwise fail at the end of a measured run instead of before it starts.
AGGREGATORS = {
    "mean": lambda samples, trim: _mean(samples),
    "median": lambda samples, trim: _median(samples),
    "trimmed_mean": lambda samples, trim: _trimmed_mean(samples, trim),
    "olympic": lambda samples, trim: _olympic(samples),
}


def aggregate(samples: Sequence[float], *, method: str, trim_fraction: float = 0.0) -> float:
    """One number from many samples, by the named method.

    Refuses an unknown method rather than falling back to the mean: a run that
    silently averaged when it was asked for olympic scoring reports a figure
    under a label it did not produce, which is the shape of every defect this
    module's docstring lists.
    """
    if not samples:
        raise ValueError("no samples; an aggregate over nothing is not zero")
    if method not in AGGREGATORS:
        raise ValueError(f"unknown aggregate {method!r}; expected one of {sorted(AGGREGATORS)}")
    return float(AGGREGATORS[method](list(samples), trim_fraction))


class CudaEventTimer:
    """Per-step timing from CUDA events rather than a host clock.

    A host clock reads the time at which Python reached a line; the device
    reaches the same point later and by an amount that varies with queue depth.
    `StepTimer` closes that gap by synchronising at both ends, which is correct
    and also serialises the pipeline it is measuring. Events are recorded in the
    stream and read after the fact, so the measured window keeps its run-ahead.

    Elapsed times are read in `durations`, not in `__exit__`: `Event.elapsed_time`
    synchronises on the event, and doing that per step would reinstate exactly the
    stall this class exists to avoid.
    """

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError(
                f"measurement.instrument=cuda_event needs a CUDA device, got {device}. "
                "Falling back to the host clock would report a wall-clock number under "
                "the label of a device measurement."
            )
        self.device = device
        self._pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def __enter__(self) -> CudaEventTimer:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._pairs.append((start, end))
        return self

    def __exit__(self, *exc: object) -> None:
        self._pairs[-1][1].record()

    @property
    def durations(self) -> list[float]:
        """Seconds per step. `elapsed_time` is milliseconds; every other figure
        in this package is seconds, and a unit that changes with the instrument
        is a rate that changes with the instrument."""
        if self._pairs:
            self._pairs[-1][1].synchronize()
        return [start.elapsed_time(end) / 1000.0 for start, end in self._pairs]


def build_timer(device: torch.device, instrument: str) -> Any:  # noqa: ANN401 - two timer types
    """The timer `measurement.instrument` names, or a refusal.

    Never a fallback. `kernel=none` picking up an environment-provided `fla` is
    the same defect one layer down, and it is why this raises instead of quietly
    returning the wall clock on a machine with no GPU.
    """
    from trainbench.metrics import StepTimer

    if instrument == "wall_clock":
        return StepTimer(device)
    if instrument == "cuda_event":
        return CudaEventTimer(device)
    raise ValueError(
        f"unknown measurement.instrument {instrument!r}; expected 'wall_clock' or 'cuda_event'"
    )


def repeat_seeds(policy: str, repeats: int, base_seed: int) -> tuple[int, ...]:
    """The seed each repeat runs under, one per repeat, all recorded.

    MLPerf CLOSED draws its seeds from `/dev/urandom` and requires that no two
    runs log the same one, because repeating a fixed seed re-measures a single
    point instead of sampling the distribution the variance estimate claims to
    describe.

    Both policies are expressible and the study still runs `fixed`, which is a
    decision that outlives this function only until a pod has measured the noise
    floor. Making `per_repeat` producible now is what stops that change from
    being a schema change later.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if policy == "fixed":
        return (base_seed,) * repeats
    if policy != "per_repeat":
        raise ValueError(f"unknown measurement.seed_policy {policy!r}")
    drawn: list[int] = []
    seen: set[int] = set()
    while len(drawn) < repeats:
        candidate = secrets.randbits(SEED_BITS)
        if candidate in seen:
            continue
        seen.add(candidate)
        drawn.append(candidate)
    return tuple(drawn)


def stdev(samples: Sequence[float]) -> float | None:
    """Sample standard deviation, or None when one sample cannot have one.

    Reported beside the aggregate because the deviation threshold that
    invalidates a pod baseline is compared against this, and a threshold quoted
    without the spread it is applied to cannot be checked by a reader.
    """
    if len(samples) < 2:
        return None
    return _stdlib.stdev(samples)
