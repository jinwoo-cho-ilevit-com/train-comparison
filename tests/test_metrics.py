"""The arithmetic behind every reported number.

Checked against hand calculations rather than against itself. A metrics test that
compares the function to a second implementation of the same formula proves the
two agree, not that either is right.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from trainbench.metrics import (
    GATE_FIELDS,
    METRIC_DEFINITIONS,
    STATUS_OOM,
    StepTimer,
    aggregate,
    build_timer,
    gradient_norm,
    is_oom,
    oom_status,
    parameter_counts,
    peak_memory_bytes,
    percentile,
    reset_peak_memory,
    stdev,
    summarise,
    synchronize,
    training_verdict,
)
from trainbench.metrics.validity import _NORM_CHUNK_ELEMENTS

from .test_config import compose_cfg as bench_config

CPU = torch.device("cpu")

# The `record-report` boundary's own payload. Read here so the measurement-side
# gate and the gate that contract executes are checked against one record rather
# than against two hand-written ones that can drift apart.
RECORD_SAMPLE = Path(__file__).resolve().parent / "fixtures" / "run_record.sample.json"


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


def test_synchronize_is_a_no_op_on_cpu():
    """CPU ops are synchronous already; there is nothing to wait for."""
    synchronize(CPU)


def test_synchronize_dispatches_to_the_matching_accelerator(monkeypatch):
    """`device.type == 'cuda'` used to be the only branch here, which silently
    no-opped on `mps` — this is the fix: any accelerator this process actually
    has gets a real wait, through the one device-agnostic call."""
    calls = []
    fake_device = torch.device("mps")
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: fake_device)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda d: calls.append(d))

    synchronize(fake_device)

    assert calls == [fake_device]


def test_synchronize_refuses_a_device_it_cannot_actually_wait_for(monkeypatch):
    """A mismatch between `device` and the process's real accelerator would
    otherwise either no-op or wait on the wrong device — both report a
    wall-clock number under a device-measurement label, which `StepTimer` must
    not do."""
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("mps"))

    with pytest.raises(RuntimeError, match="cannot synchronize"):
        synchronize(torch.device("cuda"))


def test_synchronize_refuses_a_device_when_no_accelerator_is_available(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="cannot synchronize"):
        synchronize(torch.device("cuda"))


# ---------------------------------------------------------------------------
# Token accounting: two denominators, and neither of them a framework's own rate
# ---------------------------------------------------------------------------


def test_token_accounting_keeps_the_two_denominators_apart():
    """Padding reaches 89% of a batch on GLUE-CoLA at seq128 and half of it on
    ordinary NLP corpora, and which count divides the step time decides which way
    the `dataloader.packing` axis ranks. Both counts are reported and neither is
    derived from the other."""
    summary = summarise(
        [0.5] * 4,
        discard=0,
        rows_per_step=8,
        tokens_per_step=1000,
        padded_tokens_per_step=4000,
    )

    assert summary["tokens_per_step"] == 1000
    assert summary["padded_tokens_per_step"] == 4000
    assert summary["tokens_per_second"] == pytest.approx(2000.0)
    assert summary["padded_tokens_per_second"] == pytest.approx(8000.0)
    # The definitions travel with the counts, and they are not the same sentence.
    definitions = summary["metric_definitions"]
    assert definitions["tokens"] != definitions["padded_tokens"]
    assert "padding included" in definitions["padded_tokens"]


def test_token_accounting_declares_which_count_is_the_throughput_denominator():
    """An undeclared denominator is how two adapters report incomparable
    throughputs under one column heading. It is a config knob, and the name of the
    chosen counter lands in the summary — not a third copy of one of the rates,
    which would be one more number able to disagree with itself.

    The knob is pinned to `tokens` until a report ranks on the other rate, so the
    name recorded here is the one the published tables used."""
    summary = summarise(
        [0.5] * 4,
        discard=0,
        rows_per_step=8,
        tokens_per_step=1000,
        padded_tokens_per_step=4000,
        config=bench_config(),
    )

    assert summary["throughput_denominator"] == "tokens"
    assert summary["measurement"]["throughput_denominator"] == "tokens"
    rates = [key for key in summary if key.endswith("_per_second")]
    assert sorted(rates) == ["padded_tokens_per_second", "rows_per_second", "tokens_per_second"]


def test_token_accounting_refuses_a_declared_denominator_nothing_counted():
    """A denominator the run never counted is a result whose headline rate has no
    numerator."""
    with pytest.raises(ValueError, match="never measured"):
        summarise([0.5], discard=0, rows_per_step=8, config=bench_config())


def test_token_accounting_refuses_a_rate_handed_in_by_a_caller():
    """The harness never uses a framework's own tokens/sec. `summarise` is the only
    place a rate is created, because it is the only place that knows the time the
    counters were divided by — so a rate arriving as a count is refused by name."""
    for offending in ({"tokens_per_second": 1.0}, {"samples_per_sec": 2.0}, {"throughput": 3.0}):
        with pytest.raises(ValueError, match="name a rate"):
            summarise([0.5], discard=0, rows_per_step=1, extra_counts=offending)
    with pytest.raises(ValueError, match="name a rate"):
        summarise([0.5], discard=0, rows_per_step=1, totals={"framework_tokens_per_second": 9.0})


def test_token_accounting_refuses_two_counts_of_the_same_thing():
    """The loser of a key collision leaves no trace, so the result would carry one
    of two disagreeing counts and say nothing about the other."""
    with pytest.raises(ValueError, match="silently overwritten"):
        summarise(
            [0.5],
            discard=0,
            rows_per_step=1,
            padded_tokens_per_step=10,
            extra_counts={"padded_tokens": 20},
        )
    with pytest.raises(ValueError, match="reuses"):
        summarise([0.5], discard=0, rows_per_step=1, extra_counts={"rows": 20})


# ---------------------------------------------------------------------------
# Measurement statistics: warmup, repeats, instrument, aggregation
# ---------------------------------------------------------------------------


def test_statistics_are_read_from_config_and_recorded_with_the_figure():
    """Warmup, repeat count, instrument and aggregation were four constants inside
    the harness. Two summaries produced under different values of any of them are
    not comparable, and the summary is where that is said."""
    config = bench_config(
        "+measurement.aggregate=olympic",
        "+measurement.instrument=wall_clock",
        "train.warmup_discard_steps=2",
    )
    summary = summarise(
        [9.0, 9.0, 1.0, 2.0, 3.0, 4.0],
        discard=2,
        rows_per_step=1,
        tokens_per_step=100,
        config=config,
    )

    assert summary["measurement"]["declared"] is True
    # One, because one measured window is what the harness runs — see
    # `tests/test_config.py` for the refusal that keeps any other value out.
    assert summary["measurement"]["repeats"] == 1
    assert summary["measurement"]["aggregate"] == "olympic"
    assert summary["measurement"]["instrument"] == "wall_clock"
    assert summary["steps_discarded"] == 2
    # olympic over the kept [1,2,3,4]: drop 1 and 4, mean of 2 and 3.
    assert summary["step_seconds_aggregate"] == pytest.approx(2.5)
    assert summary["step_seconds_mean"] == pytest.approx(2.5)
    assert summary["profiled"] is False


def test_statistics_left_undeclared_are_marked_undeclared_rather_than_assumed():
    """A summary built without a config still says so. Omitting the block would
    make a run that declared nothing indistinguishable from one that declared the
    defaults, which is the distinction moving these off the harness was for."""
    summary = summarise([1.0, 2.0], discard=0, rows_per_step=1)

    assert summary["measurement"]["declared"] is False
    assert summary["profiled"] is None
    assert summary["measurement"]["baseline_tolerance_status"] == "uncalibrated"


def test_statistics_aggregations_are_each_a_different_number():
    """Four names that collapse to one number would be a knob with one value.
    Hand-calculated on a sample with a fat right tail, which is the shape a step
    time distribution actually has."""
    samples = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 11.0]

    assert aggregate(samples, method="mean") == pytest.approx(2.0)
    assert aggregate(samples, method="median") == pytest.approx(1.0)
    assert aggregate(samples, method="olympic") == pytest.approx(1.0)
    # 10% off each end drops one 1.0 and the 11.0, leaving eight 1.0s.
    assert aggregate(samples, method="trimmed_mean", trim_fraction=0.1) == pytest.approx(1.0)
    assert stdev(samples) == pytest.approx(3.1622776601683795)
    assert stdev([1.0]) is None


def test_statistics_refuse_an_aggregate_they_cannot_compute():
    """Silence here is a figure produced by an aggregation the run did not ask
    for, published under the label of the one it did."""
    with pytest.raises(ValueError, match="unknown aggregate"):
        aggregate([1.0, 2.0], method="geometric_mean")
    with pytest.raises(ValueError, match="over nothing"):
        aggregate([], method="mean")
    with pytest.raises(ValueError, match="at least 3"):
        aggregate([1.0, 2.0], method="olympic")
    with pytest.raises(ValueError, match="removes all"):
        aggregate([1.0, 2.0], method="trimmed_mean", trim_fraction=0.5)


def test_statistics_instrument_is_selected_not_guessed():
    """`cuda_event` on a CPU is refused rather than served a host clock. A
    wall-clock number reported under a device-measurement label is the same defect
    as `kernel=none` picking up an environment-provided fla."""
    assert isinstance(build_timer(CPU, "wall_clock"), StepTimer)
    with pytest.raises(ValueError, match="needs a CUDA device"):
        build_timer(CPU, "cuda_event")
    with pytest.raises(ValueError, match="unknown measurement.instrument"):
        build_timer(CPU, "perf_counter")


# ---------------------------------------------------------------------------
# The training-validity gate, run in the shape that produced the defect
# ---------------------------------------------------------------------------


class _Encoder(torch.nn.Module):
    """An embedding and a projection. Small enough to hand-count its tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(ids))


def _frozen_like_unsloth() -> _Encoder:
    """Every parameter frozen, with the graph kept differentiable through the
    embedding output — what `enable_input_require_grads` does and why three cells
    backpropagated through a fully frozen model and were published as supported."""
    model = _Encoder()
    model.requires_grad_(False)
    model.embed.register_forward_hook(lambda _m, _a, out: out.requires_grad_(True))
    return model


def _one_backward(model: _Encoder) -> float:
    """One contrastive step, the way the measured loop takes it."""
    ids = torch.arange(8).reshape(4, 2)
    pooled = model(ids)[:, -1, :]
    half = pooled.shape[0] // 2
    scores = pooled[:half] @ pooled[half:].T
    loss = torch.nn.functional.cross_entropy(scores, torch.arange(half))
    loss.backward()
    return float(loss.detach())


def test_the_validity_gate_refuses_the_frozen_graph_that_backward_did_not_catch():
    """The reproduction, not a mock of it: the backward runs, returns a finite
    loss, and reaches no parameter at all."""
    model = _frozen_like_unsloth()
    loss = _one_backward(model)

    assert math.isfinite(loss), "the frozen graph must still produce a finite loss"
    counts = parameter_counts(model)
    # embed.weight, proj.weight, proj.bias — tensors, never elements.
    assert counts == {"total_params": 3, "trainable_params": 0, "params_with_grad": 0}
    assert gradient_norm(model) == 0.0

    ok, reasons = training_verdict(
        {**counts, "grad_norm": gradient_norm(model), "loss_first": 2.9, "loss_last": loss},
        peft_mode="full",
        device="cuda:0",
    )
    assert ok is False
    joined = " ".join(reasons)
    assert "grad_norm" in joined and "trainable_params" in joined


def test_the_validity_gate_passes_the_same_model_once_it_can_learn():
    """The refusal above must be about training and not about the harness: the
    identical step with the parameters unfrozen has to come out the other way, or
    the gate would refuse every run."""
    model = _Encoder()
    _one_backward(model)

    counts = parameter_counts(model)
    assert counts == {"total_params": 3, "trainable_params": 3, "params_with_grad": 3}
    norm = gradient_norm(model)
    assert norm > 0

    ok, reasons = training_verdict(
        {
            **counts,
            "grad_norm": norm,
            "loss_first": 2.9,
            "loss_last": 1.2,
            "peak_memory_bytes": 1024,
        },
        peft_mode="full",
        device="cuda:0",
    )
    assert (ok, reasons) == (True, [])


class _WidestFloat64Promotion(TorchDispatchMode):
    """The largest number of elements any single op promoted to float64.

    A float64 output means the op materialised the promotion, so the widest tensor
    it touched is what that promotion cost. Read below the dispatcher rather than
    from RSS because a high-water mark is shared with the rest of the session and
    would report whichever test ran before this one.
    """

    def __init__(self) -> None:
        self.widest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        leaves = [
            leaf
            for leaf in (*tree_leaves((args, kwargs)), *tree_leaves(out))
            if isinstance(leaf, torch.Tensor)
        ]
        if any(leaf.dtype == torch.float64 for leaf in leaves):
            self.widest = max(self.widest, *(leaf.numel() for leaf in leaves))
        return out


class _OneBigGradient(torch.nn.Module):
    """One parameter whose gradient is wider than a float64 promotion may be."""

    def __init__(self, rows: int, columns: int, *, transposed: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(rows, columns, dtype=torch.bfloat16))
        shape = (columns, rows) if transposed else (rows, columns)
        grad = torch.full(shape, 0.5, dtype=torch.bfloat16)
        self.weight.grad = grad.T if transposed else grad


def test_the_gradient_norm_does_not_promote_a_whole_gradient_to_float64():
    """`grad.to(torch.float64).pow(2).sum()` materialises eight bytes per element,
    twice. On one Qwen3.5 embedding gradient that is 5 GB, and `scripts/bench.py`
    calls this after `reset_peak_memory` and before `peak_memory_bytes`, inside the
    block that files a failure as `status: oom` — so the allocation is reported as
    the step's peak memory, and its own OOM as the hardware ceiling.

    The bound is the finding, not the speed: what is asserted is that no single
    operation promoted more than `_NORM_CHUNK_ELEMENTS` elements at once.
    """
    rows, columns = 4096, 2048
    model = _OneBigGradient(rows, columns)
    assert rows * columns > _NORM_CHUNK_ELEMENTS, "the gradient must exceed one chunk to bound it"

    watcher = _WidestFloat64Promotion()
    with watcher:
        norm = gradient_norm(model)

    # Every element is 0.5, exactly representable in bf16: sum of squares is
    # numel/4 and the norm is its root, hand-checkable rather than golden.
    assert norm == pytest.approx(math.sqrt(rows * columns * 0.25))
    assert watcher.widest <= _NORM_CHUNK_ELEMENTS, (
        f"one operation promoted {watcher.widest} elements to float64, over the "
        f"{_NORM_CHUNK_ELEMENTS}-element bound; a gradient-sized promotion lands in "
        "peak_memory_bytes and can OOM inside the block that files OOM as the ceiling"
    )


def test_the_gradient_norm_reads_a_non_contiguous_gradient_without_copying_it():
    """Chunking along the first dimension slices any strided tensor into views.
    Flattening first would copy a non-contiguous gradient whole — the same
    allocation, arrived at from the other side."""
    rows, columns = 4096, 2048
    model = _OneBigGradient(rows, columns, transposed=True)
    assert not model.weight.grad.is_contiguous()

    watcher = _WidestFloat64Promotion()
    with watcher:
        norm = gradient_norm(model)

    assert norm == pytest.approx(math.sqrt(rows * columns * 0.25))
    assert watcher.widest <= _NORM_CHUNK_ELEMENTS


def test_the_validity_gate_refuses_a_record_that_deleted_the_field_it_compares_against():
    """`GATE_FIELDS` names `total_params` and says absence is a refusal rather than
    a pass. Skipping the peft check when it is missing is the same silence: a
    `peft.mode=full` run that trained 12 of N tensors — the partial freeze HAZARDS
    records — would be published as a full finetune because nothing was there to
    compare 12 against.

    Deleted rather than set to a bad value: every other case in this file mutates a
    field, and absence is the one the gate used to pass.
    """
    healthy = {
        "grad_norm": 1.5,
        "trainable_params": 12,
        "total_params": 100,
        "loss_first": 2.9,
        "loss_last": 1.2,
        "peak_memory_bytes": 1024,
    }
    assert "total_params" in GATE_FIELDS

    for mode in ("full", "lora"):
        without = {key: value for key, value in healthy.items() if key != "total_params"}
        ok, reasons = training_verdict(without, peft_mode=mode, device="cuda:0")
        assert ok is False, f"peft.mode={mode} passed with no `total_params` to compare against"
        assert any("total_params" in reason for reason in reasons), reasons

    # The same record with the field present is the contrast: the gate has an
    # answer either way, and it is not "fine".
    assert training_verdict(healthy, peft_mode="full", device="cuda:0")[0] is False
    assert training_verdict({**healthy, "trainable_params": 100}, peft_mode="full", device="cuda:0")


@pytest.mark.parametrize(
    ("name", "over", "peft_mode", "expected"),
    [
        ("a loss that did not fall", {"loss_last": 3.1}, "full", "loss did not fall"),
        ("a loss that is not finite", {"loss_last": float("nan")}, "full", "not finite"),
        ("a full finetune that froze part of itself", {"trainable_params": 2}, "full", "full"),
        ("a LoRA in which everything trains", {}, "lora", "did not narrow"),
        ("a CUDA run with no peak memory", {"peak_memory_bytes": None}, "full", "peak_memory"),
        ("a total the gate cannot compare", {"total_params": None}, "full", "total_params"),
        ("an absent gradient norm", {"grad_norm": None}, "full", "grad_norm"),
        # On its own, with every other field healthy: this is the field-reported
        # shape — 46,000 tokens/second at a gradient norm of zero.
        ("a zero gradient norm alone", {"grad_norm": 0.0}, "full", "reached no parameter"),
    ],
)
def test_the_validity_gate_names_every_way_a_run_is_not_a_speed_result(
    name, over, peft_mode, expected
):
    metrics = {
        "grad_norm": 1.5,
        "trainable_params": 4,
        "total_params": 4,
        "loss_first": 2.9,
        "loss_last": 1.2,
        "peak_memory_bytes": 1024,
        **over,
    }
    ok, reasons = training_verdict(metrics, peft_mode=peft_mode, device="cuda:0")

    assert ok is False, f"{name} passed the gate"
    assert expected in " ".join(reasons), f"{name}: no reason mentions {expected}; got {reasons}"


def test_the_validity_gate_agrees_with_the_record_it_will_be_applied_to():
    """The frozen `record-report` sample, judged by this side's gate.

    The boundary carries its own copy of this decision so the report can be
    developed against it. Two copies drift; running both over one payload is what
    says they have not.
    """
    payload = json.loads(RECORD_SAMPLE.read_text())
    metrics = payload["metrics"]
    peft_mode = payload["config"]["peft"]["mode"]

    assert training_verdict(metrics, peft_mode=peft_mode, device=payload["device"]) == (True, [])
    frozen = {**metrics, "grad_norm": 0.0, "trainable_params": 0, "loss_last": 2.8734}
    assert training_verdict(frozen, peft_mode=peft_mode, device=payload["device"])[0] is False
    assert training_verdict(None, peft_mode=peft_mode, device=payload["device"]) == (
        False,
        ["no `metrics`: nothing was reported to check"],
    )


# ---------------------------------------------------------------------------
# Peak memory, and the memory ceiling as its own outcome
# ---------------------------------------------------------------------------


def test_peak_memory_travels_beside_the_speed_it_bought():
    """Frameworks buy speed with VRAM in different amounts — activation
    checkpointing defaults and optimizer state placement both differ — so a speed
    figure with no memory beside it is not a comparison."""
    summary = summarise([0.5] * 4, discard=0, rows_per_step=8, peak_bytes=34359738368)

    assert summary["peak_memory_bytes"] == 34359738368
    assert "peak_memory_bytes" in METRIC_DEFINITIONS


def test_peak_memory_absent_on_cpu_is_not_reported_as_zero():
    """Zero bytes is a measurement. Writing it into a result publishes a peak
    memory of nothing for every CPU run."""
    reset_peak_memory(CPU)
    summary = summarise([0.5], discard=0, rows_per_step=1, peak_bytes=peak_memory_bytes(CPU))

    assert summary["peak_memory_bytes"] is None


def test_peak_memory_ceiling_is_its_own_result_category():
    """An OOM record carries no metrics, so without a status of its own the report
    falls through to `launch_state` and renders a spent pod-hour as a combination
    nobody attempted. "Slow", "unsupported" and "never attempted" are three other
    things and none of them is this."""
    error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert is_oom(error)
    assert is_oom(torch.OutOfMemoryError("out of memory"))
    assert not is_oom(ValueError("pooling refused a packed batch"))

    status = oom_status(error, peak_bytes=34359738368)
    assert status["status"] == STATUS_OOM == "oom"
    assert status["oom"]["peak_memory_bytes"] == 34359738368
    assert "metrics" not in status


def test_peak_memory_ceiling_is_not_where_a_defect_gets_filed():
    """Filing a bug as a hardware limit publishes "this framework cannot do it" for
    a line of our own code."""
    with pytest.raises(ValueError, match="not an out-of-memory condition"):
        oom_status(ValueError("pooling refused a packed batch"))
