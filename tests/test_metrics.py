"""The arithmetic behind every reported number.

Checked against hand calculations rather than against itself. A metrics test that
compares the function to a second implementation of the same formula proves the
two agree, not that either is right.
"""

from __future__ import annotations

import pytest
import torch

from trainbench.metrics import (
    StepTimer,
    peak_memory_bytes,
    percentile,
    reset_peak_memory,
    summarise,
)

CPU = torch.device("cpu")


def test_percentile_is_nearest_rank_and_reports_a_step_that_happened():
    """An interpolated p95 reports a duration no step took."""
    values = [1.0, 2.0, 3.0, 4.0]

    # ceil(0.5*4) = 2nd smallest; ceil(0.95*4) = 4th.
    assert percentile(values, 0.50) == 2.0
    assert percentile(values, 0.95) == 4.0
    assert all(percentile(values, q) in values for q in (0.1, 0.5, 0.9, 0.95, 1.0))


def test_percentile_over_nothing_is_an_error_not_zero():
    """Zero seconds per step is a measurement; no samples is the absence of one."""
    with pytest.raises(ValueError, match="over nothing"):
        percentile([], 0.5)


def test_the_summary_arithmetic_matches_a_hand_calculation():
    """Six steps of a tenth of a second each, the first two thrown away."""
    summary = summarise(
        [9.0, 9.0, 0.1, 0.1, 0.1, 0.1],
        discard=2,
        rows_per_step=8,
        tokens_per_step=1024,
    )

    assert summary["steps_timed"] == 6
    assert summary["steps_discarded"] == 2
    assert summary["steps_measured"] == 4
    # 4 kept steps, 0.4s total: 8 rows x 4 / 0.4 = 80 rows/s, 1024 x 4 / 0.4 = 10240 tokens/s
    assert summary["rows_per_second"] == pytest.approx(80.0)
    assert summary["tokens_per_second"] == pytest.approx(10240.0)
    assert summary["step_seconds_mean"] == pytest.approx(0.1)
    # The discarded 9.0s steps must not reach the percentiles.
    assert summary["step_seconds_p50"] == pytest.approx(0.1)
    assert summary["step_seconds_p95"] == pytest.approx(0.1)


def test_the_discarded_steps_are_reported_not_just_dropped():
    """A figure produced after throwing away 20 steps and one produced after
    throwing away 2 are not comparable, and the summary is where that is said.
    `torch.compile` autotuning makes the count differ per axis (PLAN.md)."""
    summary = summarise([1.0] * 10, discard=7, rows_per_step=1)

    assert (summary["steps_timed"], summary["steps_discarded"]) == (10, 7)


def test_discarding_everything_is_an_error():
    """Otherwise a run whose warmup exceeded its step count reports a throughput
    computed from no steps at all."""
    with pytest.raises(ValueError, match="nothing to report"):
        summarise([1.0, 1.0], discard=2, rows_per_step=1)


def test_a_negative_discard_is_refused():
    with pytest.raises(ValueError, match="not be negative"):
        summarise([1.0], discard=-1, rows_per_step=1)


def test_tokens_are_not_invented_from_rows():
    """Row count is not a stand-in for token count: tokenisers differ per model and
    image tokens differ again. A run that could not count tokens says so."""
    summary = summarise([0.5], discard=0, rows_per_step=4)

    assert summary["rows_per_second"] == pytest.approx(8.0)
    assert summary["tokens_per_step"] is None
    assert summary["tokens_per_second"] is None


def test_mfu_is_absent_with_its_reason_attached():
    """Absent by decision, not by oversight. The standard FLOP formula is wrong by
    an unknown amount for all three models (GDN, PLE, sliding window)."""
    summary = summarise([1.0], discard=0, rows_per_step=1)

    assert summary["mfu"] is None
    assert "GDN" in str(summary["mfu_reason"])


def test_peak_memory_off_cuda_is_none_rather_than_zero():
    """Zero bytes is a measurement. Writing it into a result publishes a peak
    memory of nothing for every CPU run."""
    reset_peak_memory(CPU)

    assert peak_memory_bytes(CPU) is None


def test_the_timer_records_one_duration_per_step():
    timer = StepTimer(CPU)
    for _ in range(3):
        with timer:
            pass

    assert len(timer.durations) == 3
    assert all(d >= 0 for d in timer.durations)


def test_the_timer_times_warmup_steps_too():
    """Discarding happens in `summarise`, which reports how many were thrown away.
    A sample that was never taken cannot be reported as discarded."""
    timer = StepTimer(CPU)
    for _ in range(5):
        with timer:
            pass

    summary = summarise(timer.durations, discard=2, rows_per_step=1)

    assert summary["steps_timed"] == 5
    assert summary["steps_measured"] == 3
