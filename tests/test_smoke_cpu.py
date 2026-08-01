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
import json
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

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

    # `attention_mask` defaults to None the way a transformers model's does: a
    # packed batch carries no mask, and a stub that required one would make the
    # packed path untestable for a reason no real model has.
    def forward(self, input_ids, attention_mask=None, **_):
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

    # `steps.pad_token_id` reads this, and `build_collate` refuses packing without
    # it: PackedCollate searches every sequence for the pad id. Never produced by
    # the character tokeniser below (`1 + ord(c) % 60` starts at 1), so a pad id
    # found in a packed sequence really did come from padding.
    pad_token_id = 0
    # A real processor declares one, and `embedding.align_padding_side` refuses a
    # processor that does not — it cannot establish which side would be pooled.
    # `main()` calls it, so a stub without this could not stand in for a checkpoint.
    padding_side = "right"

    def __init__(self, *, accepts_images: bool = True) -> None:
        self.image_processor = SimpleNamespace() if accepts_images else None
        # Every tokenising call, so a test can assert the pretokenize axis moved
        # the work out of the step rather than only relabelling it.
        self.tokenize_calls = 0

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
        self.tokenize_calls += 1
        texts = [text] if isinstance(text, str) else list(text)
        rows = [[1 + (ord(c) % 60) for c in one] for one in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        if not padding:
            # A real tokenizer with padding=False returns ragged python lists, which
            # is the one shape `axes.PackedCollate` accepts — it refuses a rectangle
            # because a rectangle is a padded batch.
            return {"input_ids": rows[0] if isinstance(text, str) else rows}
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


def padded_batches(samples: int, count: int, length: int = 6):
    """Batches whose mask actually contains padding, with rows of unequal length.

    `micro_batch` attends every position, and with nothing padded `last_token_pool`
    returns the same row whichever side it is told to expect — so a batch built
    from it cannot tell a harness that threaded `config.model.padding_side` through
    from one that passed a literal. These pad on the right, which is the declared
    side for the configs here; `last_token_pool` refuses a mask that disagrees.
    """
    for _ in range(count):
        base = micro_batch(samples, length)
        mask = base.tensors["attention_mask"].clone()
        mask[:, -1] = 0
        mask[0, -2] = 0
        yield base._replace(
            tensors={**base.tensors, "attention_mask": mask}, tokens=int(mask.sum())
        )


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


def text_only_rows(count: int, text: str = "ab") -> list[dict[str, Any]]:
    """Rows that declare no image column at all.

    Not `rows()` with its image columns set to None: `axes.image_columns` reads
    what the dataset *declares*, so a row carrying `qry_image=None` still declares
    an image column and `axes._gradcache_needs_splittable_data` refuses
    `loss=cached_mnrl` for it. Dropping the keys is what makes a dataset text-only,
    and text-only data is the only kind GradCache is applicable to here.
    """
    return [{"qry": f"{text}{index}", "pos_text": f"{text * 2}{index}"} for index in range(count)]


def assembled_loss(config, dataset=None) -> Any:
    """The callable `axes.assemble` puts in `Built.loss_fn`, for this config.

    Taken from `assemble` rather than rebuilt here. `axes._loss` is D's internal
    structure (docs/CONTRACTS.md §2 fixes the four call sites, not what is inside
    them), and a test that rebuilt the loss its own way would keep passing while
    the loss the harness actually runs changed underneath it.

    `dataset` is threaded through because `loss.name` is decided partly by it:
    `cached_mnrl` is refused outright for a dataset that declares image columns.
    """
    built, _ = axes.assemble(
        TinyEmbedder(),
        config,
        CPU,
        framework="native",
        dataset=bench_entry.PairDataset(rows(4)) if dataset is None else dataset,
    )
    return built.loss_fn


def built_with(
    model, config, lr: float = 1e-3, optimizer=None, loss_fn=None, dataset=None
) -> Built:
    """What `main()` hands `train`, minus the dataloader the caller supplies.

    `loss_fn` defaults to the assembled one instead of to None. It was None, which
    made every test here pass against a loop that computed `info_nce` inline — the
    field `capture` reads to certify `loss.name` was the one field no test could
    have noticed was unused.
    """
    return Built(
        model=model,
        optimizer=optimizer or torch.optim.AdamW(model.parameters(), lr=lr),
        loss_fn=assembled_loss(config, dataset) if loss_fn is None else loss_fn,
        dataloader=None,
        framework="native",
    )


def text_only(count: int = 4):
    """The dataset `loss=cached_mnrl` is applicable to."""
    return bench_entry.PairDataset(text_only_rows(count))


# --- the measured loop -------------------------------------------------------


def test_the_measured_loop_runs_and_reports_what_it_measured(probe_config):
    summary = bench_entry.train(
        built_with(TinyEmbedder(), probe_config), list(batches(2)), probe_config, CPU
    )

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

    summary = bench_entry.train(built_with(TinyEmbedder(), config), SlowLoader(), config, CPU)

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

    bench_entry.train(built_with(TinyEmbedder(), config), CountingLoader(), config, CPU)

    assert served == [0, 1, 2, 3, 4, 5]


def test_the_loop_does_not_spin_on_a_loader_that_yields_nothing(probe_config):
    """`while step < total: for batch in loader:` had no progress guarantee: a
    loader yielding zero batches spun with no output and no exception until the pod
    deadline killed it."""
    with pytest.raises(RuntimeError, match="yielded no batches"):
        bench_entry.train(built_with(TinyEmbedder(), probe_config), [], probe_config, CPU)


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

    bench_entry.train(
        built_with(model, probe_config, optimizer=optimizer), list(batches(2)), probe_config, CPU
    )

    assert len(optimizer.grad_max_at_step) == probe_config.train.steps
    assert all(seen > 0 for seen in optimizer.grad_max_at_step)
    assert not torch.equal(before, model.proj.weight.detach())


def test_the_reported_losses_exclude_the_warmup_steps(probe_config):
    """`loss_first` used to be the first *warmup* step while every other figure in
    the summary was post-discard, so the two could not be read together."""
    seen: list[torch.Tensor] = []
    real = assembled_loss(probe_config)

    def spy(queries, documents):
        loss = real(queries, documents)
        seen.append(loss.detach().clone())
        return loss

    summary = bench_entry.train(
        built_with(TinyEmbedder(), probe_config, loss_fn=spy),
        list(batches(2)),
        probe_config,
        CPU,
    )

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
        bench_entry.train(
            built_with(TinyEmbedder(), probe_config), list(batches(2)), probe_config, CPU
        )


# --- the loss the loop actually runs -----------------------------------------


def gradcache_config(config_mapping, **overrides):  # noqa: F811
    """A `loss=cached_mnrl` config. `mini_batch` may not exceed `train.batch_size`
    (`config_schema._gradcache_mini_batch_fits`), so both move together."""
    settings = {
        "run.purpose": "probe",
        "train.steps": 2,
        "train.warmup_discard_steps": 0,
        "train.grad_accum": 1,
        "train.batch_size": 2,
        "data.limit": 8,
        "loss.name": "cached_mnrl",
        "loss.mini_batch": 2,
    }
    settings.update(overrides)
    return bench(config_mapping, **settings)


class ForwardCountingSGD(torch.optim.SGD):
    """Forward calls per optimizer step, counted on the model the loop was handed.

    Wrapping the real object's `forward` is what makes the count an observation of
    what ran rather than of what a fixture arranged. `lr=0.0` because these tests
    read gradients and counts, not weights — and SGD rather than AdamW because
    AdamW's decoupled decay moves weights with the gradients at zero, the same
    confusion `test_the_optimizer_actually_steps` exists to keep out.
    """

    def __init__(self, model) -> None:
        super().__init__(model.parameters(), lr=0.0)
        self.per_step: list[int] = []
        self._seen = 0
        inner = model.forward

        def forward(*args: Any, **kwargs: Any):
            self._seen += 1
            return inner(*args, **kwargs)

        model.forward = forward

    def step(self, *args: Any, **kwargs: Any):
        self.per_step.append(self._seen)
        self._seen = 0
        return super().step(*args, **kwargs)


def test_the_loop_computes_the_loss_the_run_assembled(probe_config):
    """The loop called `info_nce` inline and never touched `built.loss_fn`.

    For `loss=mnrl` the two are the same arithmetic, so no number moved and nothing
    here could see it — and that is precisely why it survived: `capture` reads
    `built.loss_fn` to certify `loss.name`, so the certified object was the one
    object the measured loop did not run. The moment `axes._loss` learned a second
    loss, the label and the number came apart.

    The spy returns a number no InfoNCE would, so a loop that recomputed the loss
    its own way cannot report this one by coincidence.
    """
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    scored = assembled_loss(probe_config)

    def spy(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        calls.append((queries.detach().clone(), documents.detach().clone()))
        return scored(queries, documents) + 5.0

    summary = bench_entry.train(
        built_with(TinyEmbedder(), probe_config, loss_fn=spy),
        list(batches(2)),
        probe_config,
        CPU,
    )

    assert len(calls) == probe_config.train.steps * probe_config.train.grad_accum
    # Halves, not the whole pooled tensor: the loop is what splits queries from
    # positives, and handing the loss all four rows would score every row against
    # itself.
    assert [tuple(side.shape[0] for side in call) for call in calls] == [(2, 2)] * len(calls)
    discard = probe_config.train.warmup_discard_steps
    plain = float(scored(*calls[discard]))
    assert summary["loss_first"] == pytest.approx(plain + 5.0)
    # Stated against the same batch rather than against a magnitude: at
    # `loss.temperature=0.02` a fresh model already separates two pairs, so plain
    # InfoNCE here is ~0 and an absolute threshold would prove nothing.
    assert summary["loss_first"] != pytest.approx(plain)


def test_cached_mnrl_runs_gradcache_rather_than_plain_in_batch_negatives(config_mapping):  # noqa: F811
    """The failure this closes was strictly worse than a crash.

    Before `axes._loss` could build `cached_mnrl`, such a run died in `assemble` and
    produced no number. After, `assemble` and `assert_matches` both passed while the
    loop went on scoring ordinary in-batch negatives — and the result JSON carried
    `loss.name=cached_mnrl`. The repository's only three timing manifests
    (`configs/experiment/phase2-loss-*.yaml`) are on this axis.

    GradCache is visible in the forward count, which is what makes this an
    observation rather than an assertion about the code: every piece of the batch is
    encoded twice, once under `no_grad` for the cache and once with a graph to
    consume it. The plain path encodes once.
    """
    config = gradcache_config(config_mapping)
    model = TinyEmbedder()
    counter = ForwardCountingSGD(model)

    summary = bench_entry.train(
        built_with(model, config, optimizer=counter, dataset=text_only()),
        list(batches(2)),
        config,
        CPU,
    )

    # Every piece encoded twice: once under `no_grad` to build the cache, once with
    # a graph to consume it. The plain path encodes once, whatever the split.
    pieces = (2 * config.train.batch_size) // config.loss.mini_batch
    assert counter.per_step == [2 * pieces] * config.train.steps
    assert summary["loss_first"] is not None
    assert config.loss.name in summary["loss_definition"]


def test_the_plain_path_encodes_the_batch_once(probe_config):
    """The other half of the count above: without it, `2 * pieces` is a number with
    nothing to be larger than, and a loop that double-encoded every run would
    satisfy both tests."""
    model = TinyEmbedder()
    counter = ForwardCountingSGD(model)

    bench_entry.train(
        built_with(model, probe_config, optimizer=counter), list(batches(2)), probe_config, CPU
    )

    assert counter.per_step == [probe_config.train.grad_accum] * probe_config.train.steps


def test_gradcache_and_plain_backward_accumulate_the_same_gradient(config_mapping):  # noqa: F811
    """`scale` is the wiring that is wrong silently.

    `gradcache_backward` takes `scale=1/grad_accum` and multiplies the *gradient*;
    the plain path divides the *loss*. Drop the argument and GradCache accumulates
    `grad_accum` times the gradient the other path does — same config, same label,
    a different effective learning rate, and every number still printed. Nothing in
    the result JSON would say so.

    GradCache computes the exact gradient, so with the same weights and the same
    batches the two paths must agree to floating-point error. That also pins the
    rest of the call: a batch handed over unsplit lands here as a mismatch. The
    batches are padded, and unequally, so that `padding_side` is pinned too — with
    nothing padded, `last_token_pool` returns the same row for either side and a
    literal passed in place of `config.model.padding_side` goes unnoticed.
    """
    cached = gradcache_config(config_mapping, **{"train.grad_accum": 2, "train.steps": 1})
    plain = gradcache_config(
        config_mapping,
        **{"train.grad_accum": 2, "train.steps": 1, "loss.name": "mnrl", "loss.mini_batch": None},
    )

    def gradients(config):
        torch.manual_seed(0)
        model = TinyEmbedder()
        torch.manual_seed(1)
        data = list(padded_batches(2, count=config.train.grad_accum))
        spy = GradSpySGD(model.parameters(), lr=0.0)
        bench_entry.train(
            built_with(model, config, optimizer=spy, dataset=text_only()), data, config, CPU
        )
        return spy.seen

    from_cache, from_plain = gradients(cached), gradients(plain)

    assert len(from_cache) == len(from_plain) == 1
    assert from_plain[0], "no gradients reached the step; the comparison would be vacuous"
    for name, grad in from_plain[0].items():
        assert grad.abs().max() > 0, f"{name} arrived at the step as zeros"
        assert torch.allclose(from_cache[0][name], grad, atol=1e-6), name


class GradSpySGD(torch.optim.SGD):
    """The gradients as the step saw them, before `zero_grad` erases the evidence."""

    def __init__(self, params, **kwargs):
        super().__init__(list(params), **kwargs)
        self.seen: list[dict[str, torch.Tensor]] = []

    def step(self, *args, **kwargs):
        self.seen.append(
            {
                f"{group}.{index}": p.grad.detach().clone()
                for group, params in enumerate(self.param_groups)
                for index, p in enumerate(params["params"])
                if p.grad is not None
            }
        )
        return super().step(*args, **kwargs)


def test_gradcache_stops_the_run_on_a_batch_it_cannot_split_by_rows(config_mapping):  # noqa: F811
    """GradCache is text-only here: `pixel_values` counts patches (Qwen-VL) or
    images (gemma-4), and mapping those back to rows needs that model's own
    placeholder accounting, so `axes._split_rows` refuses rather than guess.

    `axes._gradcache_needs_splittable_data` refuses an image-carrying *dataset*
    before a batch exists, which is the layer that covers every configured run.
    This is the layer under it: the dataset here declares text only and a batch
    turns up with pixels anyway. The refusal has to reach the caller. Swallowed
    inside the timed window — by a `try` here, or by a context manager returning
    True from `__exit__` — the run would go on to report a step time for a step
    that computed nothing.
    """
    config = gradcache_config(config_mapping)
    with_pixels = micro_batch(2)
    with_pixels.tensors["pixel_values"] = torch.zeros(3, 3, 4, 4)

    with pytest.raises(RuntimeError, match="cannot split"):
        bench_entry.train(
            built_with(TinyEmbedder(), config, dataset=text_only()), [with_pixels], config, CPU
        )


def test_a_loss_that_refuses_the_pooled_signature_dies_on_the_first_step(probe_config):
    """The fail-closed half of the wiring, independent of GradCache existing.

    `axes._loss` returns a `cached_mnrl` whose `(queries, documents)` signature
    raises, precisely so that a harness reaching for the plain shape crashes instead
    of measuring in-batch negatives under the wrong label. That only works if the
    loop lets the exception out — and it has to be out of the *first* step, before
    `summarise` can turn timings into a throughput figure.
    """
    steps_entered = []

    def refuses(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        steps_entered.append(len(steps_entered))
        raise RuntimeError("cannot be computed from pooled embeddings")

    with pytest.raises(RuntimeError, match="pooled embeddings"):
        bench_entry.train(
            built_with(TinyEmbedder(), probe_config, loss_fn=refuses),
            list(batches(2)),
            probe_config,
            CPU,
        )

    assert steps_entered == [0]


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


# --- dataloader.packing and dataloader.pretokenize ---------------------------
# Both axes were unreachable from this harness until now: the collate assignment in
# `main()` overwrote `axes.PackedCollate` unconditionally, so `torch_packed` was
# certified False against a True request and refused at `assert_matches`; and
# nothing called `axes.pretokenize`, so `torch_pretokenized` died in `assemble`.


def axis_config(config_mapping, **dataloader):  # noqa: F811
    """A probe config with the dataloader axis set and workers off.

    `num_workers: 0` because these tests count tokenising calls on the processor,
    and a forked worker's copy of that counter never comes back.
    """
    return bench(
        config_mapping,
        **{
            "run.purpose": "probe",
            "train.steps": 4,
            "train.warmup_discard_steps": 0,
            "train.grad_accum": 1,
            "train.batch_size": 2,
            "data.limit": 8,
            "data.num_workers": 0,
            **{f"dataloader.{key}": value for key, value in dataloader.items()},
        },
    )


def harness_loader(config, processor, source=None):
    """`main()`'s construction for whatever this config's dataloader axes say."""
    dataset = bench_entry.PairDataset(rows(8, qry_image=True)) if source is None else source
    if config.dataloader.pretokenize:
        dataset = axes.pretokenize(dataset, bench_entry.Encode(processor, config))
    built, _ = axes.assemble(TinyEmbedder(), config, CPU, framework="native", dataset=dataset)
    built.dataloader.collate_fn = bench_entry.build_collate(processor, config)
    return built


def test_a_packed_run_certifies_packing_from_the_class_that_owns_it(config_mapping):  # noqa: F811
    """The wrapper carries the accounting the step needs; it must not carry a
    second opinion about whether the batch is packed. `axis_packing` is forwarded
    from `axes.PackedCollate`, and the capture probe reads the forwarded value."""
    config = axis_config(config_mapping, packing=True)
    built = harness_loader(config, FakeProcessor())

    collate = built.dataloader.collate_fn
    assert isinstance(collate.packed, axes.PackedCollate)
    # Not a class attribute of the wrapper: two declarations is how one of them
    # drifts into a label the run did not earn.
    assert "axis_packing" not in vars(bench_entry.PackedBatches)
    assert collate.axis_packing is axes.PackedCollate.axis_packing

    axis = {a.axis: a for a in capture(built, config).axes}["dataloader.packing"]
    assert axis.determined, axis.detail
    assert (axis.requested, axis.applied) == ("True", "True")


def test_a_packed_batch_keeps_queries_first_so_the_pairing_survives(config_mapping):  # noqa: F811
    """`info_nce` splits the pooled embeddings at the midpoint and
    `packed_last_token_pool` returns them in packing order, so the order the
    sequences are packed in *is* the pairing."""
    config = axis_config(config_mapping, packing=True)
    processor = FakeProcessor()
    pairs = rows(2, qry_image=True)

    micro = bench_entry.build_collate(processor, config)(pairs)

    texts = bench_entry.Collate(processor, config).pair_texts(pairs, with_images=False).texts
    expected = [len(text) for text in texts]
    lengths = (micro.cu_seqlens[1:] - micro.cu_seqlens[:-1]).tolist()
    assert lengths == expected
    assert micro.samples == 2
    assert micro.rows == 4
    # No padding at all is the whole of what packing claims to save.
    assert micro.tokens == micro.padded_tokens == sum(expected)
    assert tuple(micro.tensors["input_ids"].shape) == (1, sum(expected))
    # The pixels cannot ride along in a pack, so they are counted, not forgotten.
    assert micro.images == 0
    assert micro.images_dropped == 2


def test_a_packed_batch_pools_each_sequence_at_its_own_last_token(config_mapping):  # noqa: F811
    """A packed batch is one row with no attention_mask, which is exactly the
    contract `last_token_pool` refuses to weaken — so it gets its own pooling and
    the boundaries have to be carried. Hand calculation: hidden state = position,
    so sequence i must pool to `cu_seqlens[i + 1] - 1`."""

    class Positions(torch.nn.Module):
        def forward(self, input_ids, **_):
            total = int(input_ids.shape[1])
            hidden = torch.arange(total, dtype=torch.float32).reshape(1, total, 1)
            return type("Output", (), {"last_hidden_state": hidden})()

    cu_seqlens = torch.tensor([0, 3, 7, 9], dtype=torch.int32)
    tensors = {"input_ids": torch.ones(1, 9, dtype=torch.long)}

    pooled = bench_entry.pooled_embeddings(Positions(), tensors, "right", cu_seqlens)

    assert pooled.flatten().tolist() == [2.0, 6.0, 8.0]


def test_pretokenize_moves_the_tokenisation_out_of_the_measured_step(config_mapping):  # noqa: F811
    """The axis is not a label on unchanged work: after `axes.pretokenize` the
    measured loop must tokenise nothing at all. The control is the same loop on the
    same rows with the axis off, which tokenises on every batch it draws."""
    processor = FakeProcessor()
    config = axis_config(config_mapping, pretokenize=True)
    built = harness_loader(config, processor)
    processor.tokenize_calls = 0

    bench_entry.train(built, built.dataloader, config, CPU)

    assert processor.tokenize_calls == 0

    plain = axis_config(config_mapping)
    other = FakeProcessor()
    plain_built = harness_loader(plain, other)
    other.tokenize_calls = 0
    bench_entry.train(plain_built, plain_built.dataloader, plain, CPU)
    assert other.tokenize_calls > 0


def test_packing_and_pretokenize_together_need_no_tokenizer_in_the_step(config_mapping):  # noqa: F811
    """The one combination `axes.PackedCollate` was written to serve. The rows
    already carry unpadded ids, so the pack is assembled out of them and nothing is
    tokenised inside the timed window."""
    processor = FakeProcessor()
    config = axis_config(config_mapping, packing=True, pretokenize=True)
    built = harness_loader(config, processor)
    processor.tokenize_calls = 0

    summary = bench_entry.train(built, built.dataloader, config, CPU)

    assert processor.tokenize_calls == 0
    assert summary["steps_measured"] == 4
    assert summary["tokens_per_step"] == summary["padded_tokens_per_step"]
    state = capture(built, config)
    determined = {a.axis: (a.requested, a.applied) for a in state.axes}
    assert determined["dataloader.packing"] == ("True", "True")
    assert determined["dataloader.pretokenize"] == ("True", "True")


def test_packing_with_gradcache_is_refused_rather_than_mispooled(config_mapping):  # noqa: F811
    """GradCache pools row-wise pieces the padded way; a packed batch is one row
    whose boundaries live in `cu_seqlens`. Together they would pool the wrong
    positions and still report both axes as applied."""
    config = axis_config(config_mapping, packing=True)

    def loss_fn(queries, documents):
        raise AssertionError("the run should have stopped before any loss was computed")

    loss_fn.gradcache_backward = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="Measure the two axes separately"):
        bench_entry.train(built_with(TinyEmbedder(), config, loss_fn=loss_fn), [], config, CPU)


def test_a_cross_device_loss_without_a_world_stops_the_run_on_the_first_step(config_mapping):  # noqa: F811
    """`parallel.cross_device_negatives=true` builds a loss that all-gathers, and
    that gather refuses to run without a process group. The refusal is fail-closed
    only if something calls the loss: a harness that computed its own InfoNCE would
    measure ordinary local negatives on a single-process pod and report them under
    the cross-device label, and nothing in the result JSON would say so.

    The real assembled loss, not a stand-in. A synthetic raiser proves the loop
    propagates an exception; it would go on passing if `axes._loss` stopped
    attaching the gather, which is the half this pins.
    """
    config = bench(
        config_mapping,
        **{
            "run.purpose": "probe",
            "train.steps": 4,
            "train.warmup_discard_steps": 0,
            "train.batch_size": 2,
            "parallel.strategy": "single",
            "parallel.cross_device_negatives": True,
        },
    )
    built = built_with(TinyEmbedder(), config)
    # The axis is applied and read back, so what stops the run below is the missing
    # world and not an unverified axis.
    axis = {a.axis: a for a in capture(built, config).axes}["parallel.cross_device_negatives"]
    assert (axis.requested, axis.applied) == ("True", "True")

    with pytest.raises(RuntimeError, match="needs an initialised process group"):
        bench_entry.train(built, list(batches(2)), config, CPU)


# --- a refused setting is a result ------------------------------------------
# `main()` end to end, with the checkpoint stubbed. The refusals below are the real
# ones raised by `trainbench/axes.py` and `trainbench/applied.py` for these configs,
# not stand-ins: what is under test is that the reason reaches `--out` instead of
# only the exit code reaching the pod log.


@pytest.fixture
def stub_checkpoint(monkeypatch):
    """A checkpoint `main()` can load without a network or a 2B download.

    Patched on the auto classes' `from_pretrained` rather than by rebinding
    `transformers.AutoModel` on the module: transformers is a lazy module, and a
    rebind there does not survive `from transformers import AutoModel` inside
    `build_run` — the suite then quietly pulled the real checkpoint out of whatever
    HF cache the machine happened to have, which is exactly the "the check ran and
    examined something else" shape this repository keeps producing. Run the tests
    below with an empty `HF_HOME` and they still pass; with the rebind they did not.
    """
    import transformers

    processor = FakeProcessor()
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: processor)
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", staticmethod(lambda *a, **k: TinyEmbedder())
    )
    return processor


@pytest.fixture
def pod_setting(tmp_path, monkeypatch, stub_checkpoint):
    """Run `main()` the way docker/entrypoint.sh does: a config file and an `--out`.

    Returns `(exit code, parsed record or None)`. The subset is stubbed at
    `load_pairs` because the pinned corpus is a Hub download; every axis under test
    here is decided before a row is read.
    """

    out = tmp_path / "result.json"

    def run(config, rows_for_run):
        monkeypatch.setattr(
            bench_entry, "load_pairs", lambda _: bench_entry.PairDataset(rows_for_run)
        )
        config_path = tmp_path / "resolved_config.json"
        config_path.write_text(json.dumps(config.model_dump(mode="json")))
        code = bench_entry.main(["--config", str(config_path), "--out", str(out)])
        return code, (json.loads(out.read_text()) if out.exists() else None)

    # So a test that expects the run to raise can still ask whether anything was
    # filed. "Nothing was written" is the assertion, and it needs the path.
    run.out = out
    return run


def timing_config(config_mapping, **overrides):  # noqa: F811
    settings = {
        "run.purpose": "timing",
        "train.steps": 2,
        "train.warmup_discard_steps": 0,
        "train.grad_accum": 1,
        "train.batch_size": 2,
        "data.limit": 8,
        "data.num_workers": 0,
    }
    settings.update(overrides)
    return bench(config_mapping, **settings)


def test_an_axis_this_data_cannot_apply_is_written_to_the_result_file(pod_setting, config_mapping):  # noqa: F811
    """`loss=cached_mnrl` over image-carrying rows, which is a configured pod run.

    `axes._gradcache_needs_splittable_data` refuses it in `assemble`. Before this,
    the process died with no `--out`, docker/entrypoint.sh filed a fallback record
    saying `exit 1`, and the reason — the sentence naming `pixel_values` and why a
    row cannot be recovered from it — stayed in the pod log. The report then said
    only that a pod-hour had been spent.
    """
    config = timing_config(config_mapping, **{"loss.name": "cached_mnrl", "loss.mini_batch": 2})

    code, record = pod_setting(config, rows(8, qry_image=True))

    assert code != 0, "a refusal is not a success; the sweep counts this setting as failed"
    assert record is not None
    # The body, not a category. This sentence is what the report has to be able to
    # say, and nothing here paraphrases it.
    assert "pixel_values" in record["refusal"]["reason"]
    assert "_split_rows refuses every such batch" in record["refusal"]["reason"]
    assert record["refusal"]["kind"] == "UnappliedAxis"
    assert record["refusal"]["stage"] == "assemble"
    # The setting, so the row is attributable without decoding the whole config.
    assert record["refusal"]["requested_axes"]["loss.name"] == "cached_mnrl"
    # Never. report.py renders a record carrying this as a measurement.
    assert "metrics" not in record


def test_an_axis_this_environment_cannot_apply_is_written_before_the_timer(
    pod_setting,
    config_mapping,  # noqa: F811
):
    """`precision=mxfp8` has no recipe, and `axes.step_context` is what says so.

    That call site is inside the measured loop, after `timer.__enter__`. `build_run`
    calls it once up front so the refusal lands here rather than on step 0 inside
    the timed window, where the only choices are catching inside the loop or
    crashing on something that was knowable before it started.
    """
    config = timing_config(config_mapping, **{"precision.name": "mxfp8"})

    code, record = pod_setting(config, rows(8))

    assert code != 0
    assert record["refusal"]["stage"] == "step_context"
    assert "Transformer Engine recipe" in record["refusal"]["reason"]
    assert "metrics" not in record


def test_a_mismatched_axis_is_recorded_with_the_axes_that_disagreed(pod_setting, config_mapping):  # noqa: F811
    """`AppliedMismatch` is not the same finding as `UnappliedAxis` and the record
    says so.

    A CPU timing run is refused because `adamw_fused` resolves to `adamw_unfused`
    without CUDA (docs/CONTRACTS.md §6) — a fact about the machine. But the same
    exception is raised when this harness's own construction leaves an axis
    undetermined, which has happened here twice. Which axes disagreed is the only
    thing that tells them apart, so the record keeps the whole `AppliedState`.
    """
    code, record = pod_setting(timing_config(config_mapping), text_only_rows(8))

    assert code != 0
    assert record["refusal"]["kind"] == "AppliedMismatch"
    assert record["refusal"]["stage"] == "assert_matches"
    assert "optim.name" in record["refusal"]["reason"]
    disagreed = {a["axis"] for a in record["applied"]["axes"] if not a["matches"]}
    assert "optim.name" in disagreed
    assert "metrics" not in record


def test_a_failure_inside_the_measured_loop_is_not_recorded_as_a_refusal(
    pod_setting,
    config_mapping,  # noqa: F811
    monkeypatch,
):
    """The boundary. Catching a refusal and swallowing one differ only in where the
    `try` ends, and a loop whose failures were recorded as refusals would file a
    half-run measurement as a setting that was declined.

    `axes.step_context` is called per step, so the loop really can raise this type;
    the assertion is that `main` does not catch it and writes nothing.
    """
    config = timing_config(config_mapping, **{"run.purpose": "probe"})

    def raise_mid_loop(*_args, **_kwargs):
        raise axes.UnappliedAxis("raised after the loop had started")

    monkeypatch.setattr(bench_entry, "train", raise_mid_loop)

    with pytest.raises(axes.UnappliedAxis, match="after the loop had started"):
        pod_setting(config, text_only_rows(8))
    assert not pod_setting.out.exists()


def test_a_crash_that_is_not_a_refusal_still_leaves_no_result(
    pod_setting,
    config_mapping,  # noqa: F811
    monkeypatch,
):
    """Only the two refusal types are caught, and the catch sits around the same
    region an OOM comes out of.

    A checkpoint that will not download, a CUDA OOM, a corpus that came back short:
    those leave no `--out`, which is what makes docker/entrypoint.sh publish a
    fallback record instead of a result. A `try` widened to `Exception` would turn
    every one of them into a tidy record claiming an axis could not be applied —
    a run that died would be filed as a run that was declined.
    """
    config = timing_config(config_mapping)

    def out_of_memory(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(axes, "assemble", out_of_memory)

    with pytest.raises(RuntimeError, match="out of memory"):
        pod_setting(config, text_only_rows(8))
    assert not pod_setting.out.exists()


# --- what scripts/report.py makes of a refusal record ------------------------


def merged_document(tmp_path, artifacts: dict[str, dict[str, Any]]) -> str:
    """`scripts/report.py` run for real over the given `path -> record` artifacts.

    A subprocess rather than an import: report.py imports its sibling
    `publish_result` off `scripts/`, and the question here is what the tool a human
    runs produces, not what a re-wired copy of it would.
    """
    results = tmp_path / "downloaded"
    for relative, payload in artifacts.items():
        path = results / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str))
    matrix = tmp_path / "matrix.md"
    matrix.write_text("# hand-written head\n")
    done = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "report.py"),
            "--results",
            str(results),
            "--matrix",
            str(matrix),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return matrix.read_text()


def test_report_renders_the_refusal_as_a_reason_and_not_as_a_measurement(
    tmp_path,
    pod_setting,
    config_mapping,  # noqa: F811
):
    """Passed through the real report.py, not asserted about in the abstract.

    The control is the same record with a `metrics` block added: it is the one key
    that decides the lane, so without the control this test would also pass against
    a report that renders no measurement table at all.
    """
    config = timing_config(config_mapping, **{"loss.name": "cached_mnrl", "loss.mini_batch": 2})
    _, refused = pod_setting(config, rows(8, qry_image=True))

    pod = "results/native/qwen3_vl_emb_2b/pod-a"
    measured = {
        key: value for key, value in refused.items() if key not in ("status", "refusal")
    } | {
        "metrics": {
            "step_seconds_p50": 0.5,
            "step_seconds_p95": 0.6,
            "steps_measured": 4,
            "samples_per_second": 8.0,
        }
    }
    document = merged_document(
        tmp_path,
        {
            f"{pod}/loss-cached_mnrl/result.json": refused,
            f"{pod}/loss-mnrl/result.json": measured,
        },
    )

    rows_in_tables = [line for line in document.splitlines() if line.startswith("| loss-")]
    # The control landed in a table, so "not in a table" below is a distinction the
    # document can actually draw.
    assert any(line.startswith("| loss-mnrl |") for line in rows_in_tables)
    assert not any(line.startswith("| loss-cached_mnrl |") for line in rows_in_tables)
    # And the reason is in the document, in full.
    assert "pixel_values" in document
    assert "_split_rows refuses every such batch" in document
    assert "지표 없음" in document
