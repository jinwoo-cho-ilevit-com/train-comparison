"""The measured loop, run end to end on CPU.

Not `main()`: that downloads a checkpoint. What is exercised here is the part that
produces numbers — `bench.train`, with a real optimizer step, a real backward, the
`step_context` wrapper, warmup discard and the metrics it feeds — plus the collate
that decides what a step is fed and the capture probes that decide whether a run
is allowed to report.

Only `purpose=probe` is reachable on CPU. `optim.name=adamw_fused` resolves to
`adamw_unfused` without a CUDA device, which is a permanent mismatch, so a CPU
timing run is blocked by design (docs/CONTRACTS.md §6) — that blocking is asserted
in tests/test_axes.py rather than worked around here. What a CPU test *can* do, and
what §2 asks of Wave 3 G, is check that the block comes from the device and not
from this harness's own construction; `test_a_timing_run_is_blocked_by_the_device`
is that check.
"""

from __future__ import annotations

import importlib.util
import inspect
import pickle
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from trainbench import axes
from trainbench.applied import AppliedMismatch, Built, assert_matches, capture

from .conftest import REPO_ROOT
from .test_applied import bench, config_mapping  # noqa: F401

_spec = importlib.util.spec_from_file_location("bench_entry", REPO_ROOT / "scripts" / "bench.py")
assert _spec and _spec.loader
bench_entry = importlib.util.module_from_spec(_spec)
sys.modules["bench_entry"] = bench_entry
_spec.loader.exec_module(bench_entry)

CPU = torch.device("cpu")


class TinyEmbedder(torch.nn.Module):
    """Returns hidden states the way a transformers model does, so `steps.encode`
    reaches them by the same attribute it uses in production."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(64, width)
        self.proj = torch.nn.Linear(width, width)
        # `axes.assemble` and the capture probes read a transformers-shaped config.
        self.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    def forward(self, input_ids, attention_mask, **_):
        hidden = self.proj(self.embed(input_ids))
        return type("Output", (), {"last_hidden_state": hidden})()


class FakeProcessor:
    """Enough of a transformers processor to drive the collate on CPU.

    One token per character, so a batch's token counts are readable by hand. Pads
    on the right, which is `qwen3_vl_emb_2b`'s side and the one `last_token_pool`
    then accepts. `image_processor` is what `Collate` reads to decide whether this
    checkpoint can take pixels at all, so a text-only processor is built by leaving
    it None rather than by naming an architecture.
    """

    def __init__(self, *, accepts_images: bool = True) -> None:
        self.image_processor = SimpleNamespace() if accepts_images else None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        content = messages[0]["content"]
        images = "".join("<img>" for block in content if block["type"] == "image")
        text = "".join(block["text"] for block in content if block["type"] == "text")
        return f"{images}{text}{'<gen>' if add_generation_prompt else ''}"

    def __call__(
        self,
        text,
        images=None,
        return_tensors=None,
        padding=None,
        truncation=False,
        max_length=None,
    ):
        rows = [[1 + (ord(c) % 60) for c in one] for one in text]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        width = max(len(row) for row in rows)
        encoded = {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in rows]
            ),
        }
        if images:
            encoded["pixel_values"] = torch.zeros(len(images), 3, 4, 4)
        return encoded


def rows(count: int, *, qry_image: bool = False, pos_image: bool = False, text: str = "ab"):
    """Rows shaped like the pinned subset, MMEB placeholder markup included.

    `prepare_data.py` stores `qry` and `pos_text` verbatim, marker and all, and
    converting them is this loader's job — so the fixtures carry the marker.
    """
    return [
        {
            "qry": f"<|image_1|>\n{text}{index}",
            "pos_text": f"{text * 2}{index}",
            "qry_image": SimpleNamespace(name=f"q{index}") if qry_image else None,
            "pos_image": SimpleNamespace(name=f"p{index}") if pos_image else None,
        }
        for index in range(count)
    ]


def batches(samples: int, length: int = 6, count: int = 8):
    """`info_nce` splits the pooled embeddings down the middle, so each batch
    carries `samples` queries followed by `samples` positives."""
    for _ in range(count):
        yield micro_batch(samples, length)


def micro_batch(samples: int, length: int = 6, offset: int = 0) -> bench_entry.MicroBatch:
    ids = torch.randint(1, 60, (samples * 2, length)) + offset % 3
    return bench_entry.MicroBatch(
        tensors={"input_ids": ids, "attention_mask": torch.ones_like(ids)},
        tokens=int(ids.numel()),
        padded_tokens=int(ids.numel()),
        rows=samples * 2,
        samples=samples,
        images=0,
        images_dropped=0,
    )


@pytest.fixture
def probe_config(config_mapping):  # noqa: F811 - the imported fixture, requested once
    return bench(
        config_mapping,
        **{
            "run.purpose": "probe",
            "train.steps": 6,
            "train.warmup_discard_steps": 2,
            "train.grad_accum": 1,
            "train.batch_size": 2,
            "data.limit": 8,
        },
    )


def built_with(model, lr: float = 1e-3, optimizer=None) -> Built:
    return Built(
        model=model,
        optimizer=optimizer or torch.optim.AdamW(model.parameters(), lr=lr),
        loss_fn=None,
        dataloader=None,
        framework="native",
    )


# --- the measured loop -------------------------------------------------------


def test_the_measured_loop_runs_and_reports_what_it_measured(probe_config):
    summary = bench_entry.train(built_with(TinyEmbedder()), list(batches(2)), probe_config, CPU)

    assert summary["steps_timed"] == 6
    assert summary["steps_discarded"] == 2
    assert summary["steps_measured"] == 4
    assert summary["rows_per_second"] > 0
    assert summary["tokens_per_second"] > 0
    # samples/s is PLAN.md's figure; rows/s counts queries and positives, so it is
    # twice that, and the summary says which is which rather than leaving a reader
    # to guess from the name.
    assert summary["samples_per_step"] * 2 == summary["rows_per_step"]
    assert "instruction_prompt" in summary["metric_definitions"]["tokens"]
    # No CUDA, so there is no peak to report — and 0 would be a measurement.
    assert summary["peak_memory_bytes"] is None
    assert summary["loss_first"] is not None


def test_the_data_pipeline_is_inside_the_timed_window(probe_config):
    """The defect this replaced: `for batch in loader:` produced the batch in the
    loop header, so tokenisation, collate and the host-to-device copy happened
    before t0. Every value of `dataloader.backend` / `packing` / `pretokenize` then
    measures the same step and the ablation returns zero by construction."""
    delay = 0.05

    class SlowLoader:
        def __init__(self):
            self.ready = [micro_batch(2) for _ in range(8)]

        def __iter__(self):
            for batch in self.ready:
                time.sleep(delay)
                yield batch

    config = bench_config_of(probe_config, steps=4, discard=0)

    summary = bench_entry.train(built_with(TinyEmbedder()), SlowLoader(), config, CPU)

    assert summary["step_seconds_p50"] >= delay
    assert summary["samples_per_second"] <= summary["samples_per_step"] / delay


def test_grad_accum_consumes_distinct_micro_batches(config_mapping):  # noqa: F811
    """Accumulation over one batch repeated N times gives identical sequence
    lengths, identical padding and a warm cache, and nothing in the result JSON
    would say so."""
    config = bench(
        config_mapping,
        **{
            "run.purpose": "probe",
            "train.steps": 2,
            "train.warmup_discard_steps": 0,
            "train.grad_accum": 3,
            "train.batch_size": 2,
        },
    )
    served: list[int] = []

    class CountingLoader:
        def __init__(self):
            self.ready = [micro_batch(2, offset=index) for index in range(6)]

        def __iter__(self):
            for index, batch in enumerate(self.ready):
                served.append(index)
                yield batch

    bench_entry.train(built_with(TinyEmbedder()), CountingLoader(), config, CPU)

    assert served == [0, 1, 2, 3, 4, 5]


def test_the_loop_does_not_spin_on_a_loader_that_yields_nothing(probe_config):
    """`while step < total: for batch in loader:` had no progress guarantee: a
    loader yielding zero batches spun with no output and no exception until the pod
    deadline killed it."""
    with pytest.raises(RuntimeError, match="yielded no batches"):
        bench_entry.train(built_with(TinyEmbedder()), [], probe_config, CPU)


def test_the_optimizer_actually_steps_on_gradients_that_exist(probe_config):
    """Without this the loop measures a forward and a backward and calls it
    training, which is faster than the thing it claims to be timing.

    "The weights moved" is not enough on its own: `configs/optim/*.yaml` set
    `weight_decay: 0.01`, and AdamW's decoupled decay moves weights with the
    gradients forced to zero. What the name claims is that the step consumed
    gradients, so that is what is asserted — at the moment of the step, before
    `zero_grad` can erase the evidence."""

    class GradSpyAdamW(torch.optim.AdamW):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.grad_max_at_step: list[float] = []

        def step(self, *args, **kwargs):
            seen = [
                float(p.grad.abs().max())
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            self.grad_max_at_step.append(max(seen) if seen else 0.0)
            return super().step(*args, **kwargs)

    model = TinyEmbedder()
    before = model.proj.weight.detach().clone()
    optimizer = GradSpyAdamW(model.parameters(), lr=0.1)

    bench_entry.train(built_with(model, optimizer=optimizer), list(batches(2)), probe_config, CPU)

    assert len(optimizer.grad_max_at_step) == probe_config.train.steps
    assert all(seen > 0 for seen in optimizer.grad_max_at_step)
    assert not torch.equal(before, model.proj.weight.detach())


def test_the_reported_losses_exclude_the_warmup_steps(probe_config, monkeypatch):
    """`loss_first` used to be the first *warmup* step while every other figure in
    the summary was post-discard, so the two could not be read together."""
    seen: list[torch.Tensor] = []
    real = bench_entry.info_nce

    def spy(queries, documents, temperature):
        loss = real(queries, documents, temperature)
        seen.append(loss.detach().clone())
        return loss

    monkeypatch.setattr(bench_entry, "info_nce", spy)

    summary = bench_entry.train(built_with(TinyEmbedder()), list(batches(2)), probe_config, CPU)

    discard = probe_config.train.warmup_discard_steps
    assert len(seen) == probe_config.train.steps
    assert summary["loss_first"] == pytest.approx(float(seen[discard]))
    assert summary["loss_last"] == pytest.approx(float(seen[-1]))
    assert summary["loss_first"] != pytest.approx(float(seen[0]))


def test_a_warmup_longer_than_the_run_never_reaches_the_loop(config_mapping):  # noqa: F811
    """It would report a throughput computed from no steps at all. The schema
    refuses it first, which is the better place — the run never boots."""
    with pytest.raises(Exception, match="warmup_discard_steps"):
        bench(
            config_mapping,
            **{"run.purpose": "probe", "train.steps": 2, "train.warmup_discard_steps": 4},
        )


def test_a_declared_pooling_the_loop_does_not_implement_is_refused(probe_config, monkeypatch):
    """`model.pooling` is a config knob and `steps.encode` pools the last token
    unconditionally. A run whose config named something else would measure one
    thing under the name of another."""
    monkeypatch.setattr(type(probe_config.model), "model_config", {}, raising=False)
    object.__setattr__(probe_config.model, "pooling", "mean")

    with pytest.raises(RuntimeError, match="pools the last token"):
        bench_entry.train(built_with(TinyEmbedder()), list(batches(2)), probe_config, CPU)


# --- the collate -------------------------------------------------------------


def test_the_collate_puts_queries_first_and_positives_second(probe_config):
    """`info_nce` splits pooled embeddings at the midpoint; a collate that
    interleaved them would pair every query with the wrong positive and still
    produce a loss that goes down."""
    batch = bench_entry.Collate(FakeProcessor(accepts_images=False), probe_config)(
        rows(2, text="ab")
    )

    lengths = [int(mask.sum()) for mask in batch.tensors["attention_mask"]]
    prompt = len(probe_config.model.instruction_prompt or "")
    generation = len("<gen>") if probe_config.model.add_generation_prompt else 0
    # "ab0"/"ab1" for the queries, "abab0"/"abab1" for the positives. The MMEB
    # `<|image_1|>` marker is gone from both: it is MMEB's markup, and a model that
    # read it literally would be tokenising another framework's placeholder.
    assert lengths == [prompt + 3 + generation, prompt + 3 + generation] + [
        5 + generation,
        5 + generation,
    ]
    assert batch.samples == 2
    assert batch.rows == 4


def test_the_collate_returns_cpu_tensors_and_survives_a_worker(probe_config):
    """`configs/data/*.yaml` set `num_workers: 8`. `.to(device)` in a collate runs
    in a DataLoader worker: on fork, in a child of a process that has already
    initialised CUDA (`Cannot re-initialize CUDA in forked subprocess`); on spawn a
    local closure cannot even be pickled to get there.

    There is no CUDA here, so "the tensors are on the CPU" is a claim this machine
    cannot falsify. What it can check is the two things that make the transfer
    impossible to reintroduce: the collate is never handed a device, and it is a
    picklable module-level object rather than a closure."""
    collate = bench_entry.Collate(FakeProcessor(), probe_config)
    batch = collate(rows(2, qry_image=True))

    assert "device" not in inspect.signature(bench_entry.Collate.__init__).parameters
    assert all(value.device.type == "cpu" for value in batch.tensors.values())
    assert isinstance(pickle.loads(pickle.dumps(collate)), bench_entry.Collate)


def test_the_collate_builds_image_and_text_only_rows_in_one_batch(probe_config):
    """16 of the 20 pinned configs carry `qry_image` and 7 carry `pos_image`, so a
    batch drawn from the subset holds both kinds of row at once."""
    collate = bench_entry.Collate(FakeProcessor(), probe_config)

    batch = collate(rows(1, qry_image=True, pos_image=True) + rows(1))

    # One query image and one positive image, and the flat list follows the text
    # order — queries then positives — because processors consume images in the
    # order their placeholders appear across the batch.
    assert batch.images == 2
    assert batch.images_dropped == 0
    assert batch.tensors["pixel_values"].shape[0] == 2
    placeholder = len("<img>")
    lengths = [int(mask.sum()) for mask in batch.tensors["attention_mask"]]
    # Row 0's query and positive each carry a placeholder; row 1's carry none.
    assert lengths[0] - lengths[1] == placeholder
    assert lengths[2] - lengths[3] == placeholder


def test_a_processor_that_cannot_take_images_reports_how_many_it_dropped(probe_config):
    """A text-only checkpoint reading an image corpus is a real configuration in
    this study, and the number it did not read has to be in the result rather than
    inferred from the model name."""
    collate = bench_entry.Collate(FakeProcessor(accepts_images=False), probe_config)

    batch = collate(rows(2, qry_image=True, pos_image=True))

    assert batch.images == 0
    assert batch.images_dropped == 4
    assert "pixel_values" not in batch.tensors


def test_an_image_batch_over_max_seq_len_is_refused_rather_than_truncated(config_mapping):  # noqa: F811
    """Truncation cuts placeholder tokens away from the pixels they stand for, and
    the forward pass then dies on N image features against fewer image tokens. A
    text-only batch truncates normally; this one cannot."""
    config = bench(config_mapping, **{"run.purpose": "probe", "data.max_seq_len": 8})
    collate = bench_entry.Collate(FakeProcessor(), config)

    assert int(collate(rows(2, text="x" * 40)).tensors["input_ids"].shape[1]) == 8
    with pytest.raises(RuntimeError, match="over data.max_seq_len"):
        collate(rows(2, qry_image=True, text="x" * 40))


# --- what the capture probes are allowed to say about this harness -----------


def loader_this_harness_builds(config, processor=None):
    """Exactly what `main()` constructs: `axes.assemble`, then the collate assigned
    over the loader's default."""
    model = TinyEmbedder()
    built, _ = axes.assemble(
        model, config, CPU, framework="native", dataset=bench_entry.PairDataset(rows(4))
    )
    built.dataloader.collate_fn = bench_entry.Collate(processor or FakeProcessor(), config)
    return built


def test_this_harness_leaves_no_dataloader_axis_undetermined(probe_config):
    """Assigning a closure over `collate_fn` turned `dataloader.packing` from
    determined-False into undetermined, because `_capture_dataloader_packing` reads
    an `axis_packing` attribute the closure did not have; `PairDataset` declared no
    `column_names`, so `dataloader.pretokenize` was undetermined too. Undetermined
    blocks a timing run exactly like a mismatch, so the harness's own construction
    was what refused every run.

    The fix is on this side: the objects declare what the probes read. Weakening
    `assert_matches` or the probes would remove the safety device instead."""
    state = capture(loader_this_harness_builds(probe_config), probe_config)

    axes_by_name = {axis.axis: axis for axis in state.axes}
    for name in ("dataloader.backend", "dataloader.packing", "dataloader.pretokenize"):
        axis = axes_by_name[name]
        assert axis.determined, f"{name} undetermined: {axis.detail}"
        assert axis.matches, f"{name}: requested {axis.requested!r}, applied {axis.applied!r}"


def test_a_timing_run_is_blocked_by_the_device_and_not_by_this_harness(config_mapping):  # noqa: F811
    """`purpose=probe` returns from `assert_matches` immediately, so a suite that
    only ever runs probe never executes the enforcing branch at all — the
    capture -> assert_matches path had no coverage (docs/CONTRACTS.md §2).

    A CPU timing run must still be refused: `adamw_fused` resolves to
    `adamw_unfused` without CUDA (§6). What this pins is *what* refuses it. No
    `dataloader.*` axis may appear in the reasons, or the harness is blocking
    itself and the block says nothing about the machine."""
    timing = bench(config_mapping, **{"run.purpose": "timing", "train.batch_size": 2})
    assert timing.run.purpose == "timing"

    state = capture(loader_this_harness_builds(timing), timing)
    with pytest.raises(AppliedMismatch) as raised:
        assert_matches(state, timing)

    reasons = str(raised.value)
    assert "optim.name" in reasons
    assert "dataloader." not in reasons


def bench_config_of(config, *, steps: int, discard: int):
    """A copy of `config` with a different step budget.

    `BenchConfig` is frozen, and re-composing through Hydra for a two-field change
    would make this test depend on the config directory as well as on the loop."""
    train = config.train.model_copy(update={"steps": steps, "warmup_discard_steps": discard})
    return config.model_copy(update={"train": train})
