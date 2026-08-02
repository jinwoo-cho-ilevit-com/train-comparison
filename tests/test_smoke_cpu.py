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
import io
import json
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from trainbench import axes, collate
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
    # The two Qwen checkpoints this stub stands in for ship one, and
    # trainbench/prompt.py refuses `prompt_format=chat_template` from a processor
    # that has none. `raw` is exercised against `RawProcessor` below.
    chat_template = "{{ messages }}"

    def __init__(self, *, accepts_images: bool = True) -> None:
        self.image_processor = SimpleNamespace() if accepts_images else None
        # Every tokenising call, so a test can assert the pretokenize axis moved
        # the work out of the step rather than only relabelling it.
        self.tokenize_calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Every turn, not just the first: `model.instruction_prompt` reaches the
        # template as a system turn ahead of the user one, and a stub that rendered
        # `messages[0]` alone would drop whichever of the two came second.
        rendered = ""
        for message in messages:
            content = message["content"]
            rendered += "".join("<img>" for block in content if block["type"] == "image")
            rendered += "".join(block["text"] for block in content if block["type"] == "text")
        return f"{rendered}{'<gen>' if add_generation_prompt else ''}"

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
            # One sublist per row, which is the shape all three real processors
            # take and the only shape `Gemma4Processor` takes at all (measured
            # 2026-08-02). Checked rather than flattened silently: the equality
            # below is `Gemma4Processor.validate_inputs`'s, and a collate that went
            # back to a flat list would otherwise pass here and fail on a pod.
            if len(images) != len(texts) or not all(isinstance(row, list) for row in images):
                raise ValueError(
                    f"images must be one list per row: got {len(images)} entr(ies) for "
                    f"{len(texts)} text(s)"
                )
            encoded["pixel_values"] = torch.zeros(sum(len(row) for row in images), 3, 4, 4)
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


def micro_batch(samples: int, length: int = 6, offset: int = 0) -> collate.MicroBatch:
    ids = torch.randint(1, 60, (samples * 2, length)) + offset % 3
    return collate.MicroBatch(
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
        dataset=collate.PairDataset(rows(4)) if dataset is None else dataset,
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
    return collate.PairDataset(text_only_rows(count))


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
    """`pixel_values` counts patches (Qwen-VL) or images (gemma-4), never rows, so
    `axes._split_rows` cuts it from the per-row image counts the collate recorded —
    and refuses when there are none rather than guessing.

    This is the batch that has pixels and no counts: `micro_batch` leaves
    `images_per_row` at None, which is what every collate that drops images
    produces. The refusal has to reach the caller. Swallowed inside the timed
    window — by a `try` here, or by a context manager returning True from
    `__exit__` — the run would go on to report a step time for a step that computed
    nothing.
    """
    config = gradcache_config(config_mapping)
    with_pixels = micro_batch(2)
    with_pixels.tensors["pixel_values"] = torch.zeros(3, 3, 4, 4)

    with pytest.raises(RuntimeError, match="cannot attribute"):
        bench_entry.train(
            built_with(TinyEmbedder(), config, dataset=text_only()), [with_pixels], config, CPU
        )


def test_the_collate_counts_every_row_s_images_against_the_placeholders_it_wrote(
    config_mapping,  # noqa: F811
):
    """`images_per_row` is the map GradCache cuts `pixel_values` with, and this is
    the only place it can be checked against what it claims to describe.

    The processor consumes the flat image list in the order the placeholders appear
    across the batch, so the count recorded for a row has to equal the number of
    placeholders that row's own text carries. Anything else — an order swapped
    between the two halves, a `None` image counted, a dropped image still counted —
    puts the batch's row boundaries somewhere the pixels are not, and
    `_split_rows` would cut there without complaint because the vector is
    self-consistent.

    Rows are asymmetric on purpose: only some carry a positive image, so a
    `images_per_row` built from the query side alone, or interleaved rather than
    concatenated, is a different vector from the right one.
    """
    processor = FakeProcessor()
    config = axis_config(config_mapping)
    pairs = rows(2, qry_image=True) + rows(2, qry_image=True, pos_image=True, text="cd")

    micro = collate.Collate(processor, config)(pairs)
    texts = collate.Collate(processor, config).pair_texts(pairs).texts

    assert micro.images_per_row == tuple(text.count("<img>") for text in texts)
    assert len(micro.images_per_row) == micro.rows
    assert sum(micro.images_per_row) == micro.images == int(micro.tensors["pixel_values"].shape[0])
    # Queries first, positives after — not interleaved. Two of the four positives
    # carry an image, and both are in the second half.
    assert micro.images_per_row == (1, 1, 1, 1, 0, 0, 1, 1)


def test_the_processor_is_handed_one_image_list_per_row(config_mapping):  # noqa: F811
    """The other consumer of the same map, and the reason there is only one of it.

    Measured 2026-08-02 against the real processors: `Gemma4Processor` reads a flat
    list as a single row's images and raises "Received inconsistently sized batches
    of images (1) and text (4)", so no gemma-4 batch carrying images could be built
    at all — for any loss, not just this one. Both Qwen processors take either form
    and return byte-identical tensors for one image per row, so the grouped form is
    the one shape all three accept.

    Grouping it with `images_per_row` rather than with a second walk of the rows is
    what keeps the processor's view and `_split_rows`' view from drifting: a wrong
    vector is now a batch the processor itself refuses.
    """

    class Recording(FakeProcessor):
        def __call__(self, text, images=None, **kwargs):
            self.images = images
            return super().__call__(text, images=images, **kwargs)

    processor = Recording()
    config = axis_config(config_mapping)
    pairs = rows(2, qry_image=True) + rows(2, qry_image=True, pos_image=True, text="cd")

    micro = collate.Collate(processor, config)(pairs)

    assert [len(group) for group in processor.images] == list(micro.images_per_row)
    assert len(processor.images) == micro.rows
    # The two text-only positives get an empty list, not a missing entry: the count
    # of sublists has to equal the count of texts.
    assert processor.images[4] == [] and processor.images[5] == []


class RawProcessor(FakeProcessor):
    """A pre-trained checkpoint: no chat template, an image token, and an
    `apply_chat_template` that cannot be called. `google/gemma-4-E2B` is this shape
    (measured 2026-08-02, transformers 5.14.1)."""

    chat_template = None
    image_token = "<|image|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        raise AssertionError("prompt_format=raw must not reach apply_chat_template")


def test_the_collate_builds_rows_for_a_checkpoint_with_no_chat_template(config_mapping):  # noqa: F811
    """The measured harness reads the same `model.prompt_format` the probe does.

    A raw row is the placeholder and the text and nothing else — no role or turn
    markers — which is why the format is declared per model and recorded in the
    result rather than inferred here (docs/model-spec.md).
    """
    processor = RawProcessor()
    config = bench(
        config_mapping,
        **{
            "model.prompt_format": "raw",
            "model.add_generation_prompt": False,
            "model.instruction_prompt": None,
            "data.num_workers": 0,
        },
    )

    built = collate.Collate(processor, config).pair_texts(rows(1, qry_image=True))

    # The MMEB marker is gone, the model's own placeholder leads the query, and the
    # text-only positive carries none.
    assert built.texts == ["<|image|>ab0", "abab0"]


def test_a_dropped_image_is_not_counted_as_one_the_row_still_carries(config_mapping):  # noqa: F811
    """A text-only checkpoint takes no pixels, and `Collate` counts those images as
    dropped. Counting them in `images_per_row` anyway would build boundaries for a
    `pixel_values` that is not in the batch, and `_split_rows` would then refuse
    every batch of a run that is otherwise fine."""
    config = axis_config(config_mapping)
    pairs = rows(2, qry_image=True, pos_image=True)

    micro = collate.Collate(FakeProcessor(accepts_images=False), config)(pairs)

    assert micro.images_dropped == 4
    assert micro.images_per_row == (0, 0, 0, 0)
    assert "pixel_values" not in micro.tensors


def test_the_pieces_a_real_batch_splits_into_carry_their_own_rows_pixels(config_mapping):  # noqa: F811
    """The join between the two halves of this: a batch built by the real collate,
    cut by the real split.

    Everything either side of this seam is checked on its own — the collate's counts
    against the placeholders it wrote, and `_split_rows` against a stand-in that
    consumes pixels the way a VL model does — and neither notices if the two stop
    describing the same batch. `TinyEmbedder` ignores `pixel_values` entirely, so
    the measured loop below cannot notice either.

    Half the rows carry a positive image and half do not, so the vector is not its
    own reverse and not its own interleaving: a collate that built the counts in
    the wrong order lands here as a different cut.
    """
    config = gradcache_config(config_mapping, **{"train.batch_size": 4, "loss.mini_batch": 4})
    pairs = rows(2, qry_image=True) + rows(2, qry_image=True, pos_image=True, text="cd")

    micro = collate.build_collate(FakeProcessor(), config)(pairs)
    pieces = axes._split_rows(micro.tensors, config.loss.mini_batch, micro.images_per_row)

    assert micro.images_per_row == (1, 1, 1, 1, 0, 0, 1, 1)
    # Four query rows with an image each, then two text-only positives and two with
    # one. Cut down the middle, that is four images and then two.
    assert [int(piece["pixel_values"].shape[0]) for piece in pieces] == [4, 2]
    assert sum(int(piece["pixel_values"].shape[0]) for piece in pieces) == micro.images


def test_gradcache_runs_a_measured_loop_over_image_carrying_rows(config_mapping):  # noqa: F811
    """The whole chain in one run, on the data this study measures.

    `configs/data`'s two subsets are MMEB draws with an image on nearly every row,
    and this axis was refused for all of them until `_split_rows` could attribute
    `pixel_values`. What has to hold now is every link at once: the collate records
    how many images each row put in, `MicroBatch` carries that out of the worker,
    the loop hands it to `gradcache_backward`, and the split cuts the pixels at a
    row boundary. A break anywhere in it arrives here as a raise, because each
    piece is checked against its own rows on the way through.

    `built_with` is not used: its loader is supplied by the caller, and the point
    here is the collate `build_collate` really returns.
    """
    config = gradcache_config(
        config_mapping,
        **{"train.batch_size": 2, "loss.mini_batch": 2, "data.num_workers": 0},
    )
    processor = FakeProcessor()
    built = harness_loader(config, processor)

    summary = bench_entry.train(built, built.dataloader, config, CPU)

    assert summary["steps_measured"] == config.train.steps
    # The pixels were in the batches this measured. A run that had quietly read a
    # text-only view of an image corpus reports these the other way round.
    assert summary["images_read_total"] > 0
    assert summary["images_dropped_total"] == 0


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
    batch = collate.Collate(FakeProcessor(accepts_images=False), probe_config)(rows(2, text="ab"))

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
    collate_fn = collate.Collate(FakeProcessor(), probe_config)
    batch = collate_fn(rows(2, qry_image=True))

    assert "device" not in inspect.signature(collate.Collate.__init__).parameters
    assert all(value.device.type == "cpu" for value in batch.tensors.values())
    assert isinstance(pickle.loads(pickle.dumps(collate_fn)), collate.Collate)


def test_the_collate_builds_image_and_text_only_rows_in_one_batch(probe_config):
    """16 of the 20 pinned configs carry `qry_image` and 7 carry `pos_image`, so a
    batch drawn from the subset holds both kinds of row at once."""
    collate_fn = collate.Collate(FakeProcessor(), probe_config)

    batch = collate_fn(rows(1, qry_image=True, pos_image=True) + rows(1))

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
    collate_fn = collate.Collate(FakeProcessor(accepts_images=False), probe_config)

    batch = collate_fn(rows(2, qry_image=True, pos_image=True))

    assert batch.images == 0
    assert batch.images_dropped == 4
    assert "pixel_values" not in batch.tensors


def test_an_image_batch_over_max_seq_len_is_refused_rather_than_truncated(config_mapping):  # noqa: F811
    """Truncation cuts placeholder tokens away from the pixels they stand for, and
    the forward pass then dies on N image features against fewer image tokens. A
    text-only batch truncates normally; this one cannot."""
    config = bench(config_mapping, **{"run.purpose": "probe", "data.max_seq_len": 8})
    collate_fn = collate.Collate(FakeProcessor(), config)

    assert int(collate_fn(rows(2, text="x" * 40)).tensors["input_ids"].shape[1]) == 8
    with pytest.raises(RuntimeError, match="over data.max_seq_len"):
        collate_fn(rows(2, qry_image=True, text="x" * 40))


# --- what the capture probes are allowed to say about this harness -----------


def loader_this_harness_builds(config, processor=None):
    """Exactly what `main()` constructs: `axes.assemble`, then the collate assigned
    over the loader's default."""
    model = TinyEmbedder()
    built, _ = axes.assemble(
        model, config, CPU, framework="native", dataset=collate.PairDataset(rows(4))
    )
    built.dataloader.collate_fn = collate.Collate(processor or FakeProcessor(), config)
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
    dataset = collate.PairDataset(rows(8, qry_image=True)) if source is None else source
    if config.dataloader.pretokenize:
        dataset = axes.pretokenize(dataset, collate.Encode(processor, config))
    built, _ = axes.assemble(TinyEmbedder(), config, CPU, framework="native", dataset=dataset)
    built.dataloader.collate_fn = collate.build_collate(processor, config)
    return built


def test_a_packed_run_certifies_packing_from_the_class_that_owns_it(config_mapping):  # noqa: F811
    """The wrapper carries the accounting the step needs; it must not carry a
    second opinion about whether the batch is packed. `axis_packing` is forwarded
    from `axes.PackedCollate`, and the capture probe reads the forwarded value."""
    config = axis_config(config_mapping, packing=True)
    built = harness_loader(config, FakeProcessor())

    collate_fn = built.dataloader.collate_fn
    assert isinstance(collate_fn.packed, axes.PackedCollate)
    # Not a class attribute of the wrapper: two declarations is how one of them
    # drifts into a label the run did not earn.
    assert "axis_packing" not in vars(collate.PackedBatches)
    assert collate_fn.axis_packing is axes.PackedCollate.axis_packing

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

    micro = collate.build_collate(processor, config)(pairs)

    texts = collate.Collate(processor, config).pair_texts(pairs, with_images=False).texts
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
        monkeypatch.setattr(bench_entry, "load_pairs", lambda _: collate.PairDataset(rows_for_run))
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


def test_an_axis_this_environment_cannot_apply_is_written_before_the_timer(
    pod_setting,
    config_mapping,  # noqa: F811
):
    """`precision=mxfp8` has no recipe, and `axes.step_context` is what says so.

    That call site is inside the measured loop, after `timer.__enter__`. `build_run`
    calls it once up front so the refusal lands here rather than on step 0 inside
    the timed window, where the only choices are catching inside the loop or
    crashing on something that was knowable before it started.

    This is also the check that a refused setting reaches `--out` at all. Its
    subject used to be `loss=cached_mnrl` over image-carrying rows, which was every
    configured pod run until `_split_rows` learned to attribute `pixel_values` to
    rows; that refusal is gone, and the record path is the same one either way.
    Before it existed, such a process died with no `--out`, docker/entrypoint.sh
    filed a fallback record saying `exit 1`, and the reason stayed in the pod log.
    """
    config = timing_config(config_mapping, **{"precision.name": "mxfp8"})

    code, record = pod_setting(config, rows(8))

    assert code != 0, "a refusal is not a success; the sweep counts this setting as failed"
    assert record["refusal"]["stage"] == "step_context"
    assert record["refusal"]["kind"] == "UnappliedAxis"
    # The body, not a category. This sentence is what the report has to be able to
    # say, and nothing here paraphrases it.
    assert "Transformer Engine recipe" in record["refusal"]["reason"]
    # The setting, so the row is attributable without decoding the whole config.
    assert record["refusal"]["requested_axes"]["precision.name"] == "mxfp8"
    # Never. report.py renders a record carrying this as a measurement.
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
    config = timing_config(config_mapping, **{"precision.name": "mxfp8"})
    _, refused = pod_setting(config, rows(8))

    pod = "results/native/qwen3_vl_emb_2b/pod-a"
    measured = {
        key: value for key, value in refused.items() if key not in ("status", "refusal")
    } | {
        "metrics": {
            "step_seconds_p50": 0.5,
            "step_seconds_p95": 0.6,
            "steps_measured": 4,
            "samples_per_second": 8.0,
            # The training-validity gate `report.py` applies before it ranks a run.
            # Without these the control is a record that cannot say it trained, and
            # the report keeps it out of the table for that reason instead of the
            # refusal this test is about.
            "grad_norm": 1.5,
            "trainable_params": 219,
            "total_params": 219,
            "loss_first": 2.9,
            "loss_last": 2.4,
        }
    }
    document = merged_document(
        tmp_path,
        {
            f"{pod}/precision-mxfp8/result.json": refused,
            f"{pod}/precision-bf16/result.json": measured,
        },
    )

    rows_in_tables = [line for line in document.splitlines() if line.startswith("| precision-")]
    # The control landed in a table, so "not in a table" below is a distinction the
    # document can actually draw.
    assert any(line.startswith("| precision-bf16 |") for line in rows_in_tables)
    assert not any(line.startswith("| precision-mxfp8 |") for line in rows_in_tables)
    # And the reason is in the document, in full — both halves of the sentence, so
    # a renderer that printed a category would fail here. The second half used to be
    # "which is not implemented"; the recipe is implemented now and what is missing
    # is the package, which is what the refusal says on this host.
    assert "Transformer Engine recipe" in document
    assert "is not importable here" in document
    assert "지표 없음" in document


# --- the preflight: the whole plan, before the first setting ----------------------


@pytest.fixture
def pod_gpu(monkeypatch):
    """A pod on an A100 with an image that says which GPUs it covers.

    Both halves are stubbed for a reason a CPU host cannot avoid: there is no GPU
    to read a capability from, and no Docker ENV to carry the arch list. The
    stubbing is at the seam the pod uses — `current_gpu_arch` reads
    `torch.cuda.get_device_capability`, the variable is read from the environment —
    so a test about the plan is never also a test about the hardware.
    """
    monkeypatch.setenv(bench_entry.CUDA_ARCHS_ENV, "80;90;100")
    monkeypatch.setattr(bench_entry, "current_gpu_arch", lambda: "80")


def plan_file(tmp_path, items):
    path = tmp_path / "resolved_plan.json"
    path.write_text(json.dumps(items))
    return path


def plan_item(name, config):
    """A plan item as `orchestrate.Run.summary()` writes one."""
    return {"name": name, "role": "experiment", "overrides": [], "config": config}


def test_a_plan_whose_settings_can_all_be_applied_passes(tmp_path, config_mapping, pod_gpu):  # noqa: F811
    config = bench(config_mapping).model_dump(mode="json")
    path = plan_file(tmp_path, [plan_item("a", config), plan_item("b", config)])
    assert bench_entry.preflight(path) == 0


def test_one_unapplicable_setting_refuses_the_whole_plan(tmp_path, config_mapping, pod_gpu):  # noqa: F811
    """`precision=mxfp8` is refused on every host this study runs on: the recipe
    needs Transformer Engine, which is absent here, and needs compute capability
    10.x, which the A100 pods do not have. `preflight` reaches it through
    `axes.step_context`."""
    good = bench(config_mapping).model_dump(mode="json")
    bad = bench(config_mapping, **{"precision.name": "mxfp8"}).model_dump(mode="json")
    path = plan_file(tmp_path, [plan_item("a", good), plan_item("b", bad)])
    assert bench_entry.preflight(path) == bench_entry.PREFLIGHT_EXIT


def test_an_empty_plan_is_a_refusal_and_not_a_clean_bill(tmp_path, pod_gpu):
    """Zero settings checked is the state this exists to catch, not the state it
    reports as fine (docs/CONTRACTS.md §6: an empty input is a failure)."""
    assert bench_entry.preflight(plan_file(tmp_path, [])) == bench_entry.PREFLIGHT_EXIT


def test_a_plan_with_nothing_composable_in_it_is_a_refusal(tmp_path, pod_gpu):
    path = plan_file(tmp_path, [{"name": "a"}, {"name": "b", "config": {}}])
    assert bench_entry.preflight(path) == bench_entry.PREFLIGHT_EXIT


def test_a_plan_that_cannot_be_read_is_a_refusal(tmp_path, pod_gpu):
    assert bench_entry.preflight(tmp_path / "absent.json") == bench_entry.PREFLIGHT_EXIT
    unparseable = tmp_path / "plan.json"
    unparseable.write_text("{ not json")
    assert bench_entry.preflight(unparseable) == bench_entry.PREFLIGHT_EXIT


def test_one_malformed_item_does_not_take_the_settings_beside_it_down(
    tmp_path,
    config_mapping,  # noqa: F811
    pod_gpu,
):
    """`docker/entrypoint.sh` stops that setting alone and publishes a record naming
    it; refusing the pod here would overturn that from the other side."""
    config = bench(config_mapping).model_dump(mode="json")
    path = plan_file(tmp_path, [plan_item("a", config), {"name": "b"}])
    assert bench_entry.preflight(path) == 0


# --- the preflight: is this pod's GPU one the image compiled kernels for ---------


@pytest.mark.parametrize(
    ("capability", "arch"),
    [((8, 0), "80"), ((9, 0), "90"), ((10, 0), "100"), ((8, 9), "89"), ((12, 0), "120")],
)
def test_a_capability_is_spelled_the_way_every_arch_list_spells_it(capability, arch):
    """torch builds its own `-gencode` this way (`_get_cuda_arch_flags`): the
    capability becomes `f'{major}.{minor}'` and then `num = f"{major}{minor}"`.
    `89` is why the reading is unambiguous — Ada is capability 8.9."""
    assert bench_entry.device_arch(capability) == arch


@pytest.mark.parametrize(
    ("declared", "archs"),
    [
        ("80;90;100", ["80", "90", "100"]),
        (" 80 ; 90 ", ["80", "90"]),
        ("80,90", ["80", "90"]),
        # Architecture-specific SASS is still that device's arch, not a fourth kind
        # of number. Nothing in the image uses one today.
        ("90a;100", ["90", "100"]),
        ("", []),
        (None, []),
    ],
)
def test_the_images_arch_list_is_read_the_way_the_dockerfile_writes_it(declared, archs):
    assert bench_entry.declared_archs(declared) == archs


def test_a_gpu_the_image_compiled_for_is_not_refused():
    assert bench_entry.gpu_refusal("80;90;100", "80") is None
    assert bench_entry.gpu_refusal("80;90;100", "100") is None


def test_a_gpu_outside_the_images_arch_list_is_refused():
    reason = bench_entry.gpu_refusal("80;90;100", "120")
    assert "sm_120" in reason and "sm_80/sm_90/sm_100" in reason


def test_an_image_that_declares_no_archs_is_refused_rather_than_assumed():
    """The absence is not evidence that the image is wide. `Dockerfile.framework`
    sets this variable in the same file that copies the entrypoint which calls
    this, so an image with the check and without the variable is not a state this
    repository builds — and passing on nothing to compare against would pass
    exactly the pods the check exists for."""
    assert bench_entry.CUDA_ARCHS_ENV in bench_entry.gpu_refusal(None, "80")
    assert bench_entry.CUDA_ARCHS_ENV in bench_entry.gpu_refusal("", "80")


def test_a_pod_booted_to_measure_with_no_visible_gpu_is_refused():
    assert "no CUDA device is visible" in bench_entry.gpu_refusal("80;90;100", None)


def test_the_arch_check_reads_the_current_device_and_names_none(monkeypatch):
    """No device string is constructed, so this is not a second device resolver
    beside `trainbench/device.py` — and with no CUDA there is nothing to read."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bench_entry.current_gpu_arch() is None


def test_a_runnable_plan_on_the_wrong_gpu_measures_nothing(tmp_path, config_mapping, monkeypatch):  # noqa: F811
    monkeypatch.setenv(bench_entry.CUDA_ARCHS_ENV, "80;90;100")
    monkeypatch.setattr(bench_entry, "current_gpu_arch", lambda: "120")
    config = bench(config_mapping).model_dump(mode="json")
    path = plan_file(tmp_path, [plan_item("a", config)])
    assert bench_entry.preflight(path) == bench_entry.PREFLIGHT_EXIT


def test_a_pod_wrong_in_both_ways_is_told_both(tmp_path, config_mapping, monkeypatch):  # noqa: F811
    """One pod log naming the GPU and the setting is worth more than two relaunches."""
    monkeypatch.setenv(bench_entry.CUDA_ARCHS_ENV, "80;90;100")
    monkeypatch.setattr(bench_entry, "current_gpu_arch", lambda: "120")
    bad = bench(config_mapping, **{"precision.name": "mxfp8"}).model_dump(mode="json")
    path = plan_file(tmp_path, [plan_item("a", bad)])
    log = io.StringIO()
    assert bench_entry.preflight(path, stream=log) == bench_entry.PREFLIGHT_EXIT
    printed = log.getvalue()
    assert "this pod's GPU" in printed
    assert "mxfp8" in printed


# --- what the adapter says, reaching the run ---------------------------------
#
# `trainbench/loader.py` declares two things `scripts/bench.py` has to carry
# across: the axes the framework computes inside its own step, and the numeric
# regime it trains in. Both were declared on one side and read by nobody, so a
# tevatron cell had `loss.name` undetermined and every axolotl step ran outside
# autocast. These exercise the wiring end to end through `main()`, not the two
# ends separately — that is how it stayed open through three lanes.


def adapter_binding(monkeypatch, **declared):
    """Run `main()` against a binding that declares what a real adapter would.

    Built on top of the real `load_framework` so the model and processor are the
    stubbed ones and only the declarations differ; the point under test is what
    `build_run` does with the fields, not what any framework loads.
    """
    original = bench_entry.load_framework

    def patched(config, device):
        loaded = original(config, device)
        # By field name, which is also an assertion: `Binding` and `AdapterOut`
        # carry the same eight, and the contract fixes them.
        carried = {name: getattr(loaded, name) for name in bench_entry.Binding._fields}
        return bench_entry.Binding(**carried | declared)

    monkeypatch.setattr(bench_entry, "load_framework", patched)


def test_a_probe_record_carries_the_gradient_norm_and_the_parameter_counts(
    pod_setting,
    config_mapping,  # noqa: F811
):
    """The validity gate the `record-report` boundary applies has to be true of a
    run, not only of a fixture.

    A finite loss and a full set of step times are exactly what a fully frozen
    graph produces — three cells of the 2026-08-02 campaign were published that
    way. `metrics.gradient_norm` and `metrics.parameter_counts` are the reads that
    tell the two apart, and until this wave nothing called them from the measured
    loop, so every real record failed the gate its own contract declares.

    A positive norm is the assertion. Reading the gradients after
    `optimizer.zero_grad` gives a confident zero, which is the one value this
    number exists to distinguish.

    `batch_size=4` and not the fixture's 2: a two-row batch is one query and one
    document, InfoNCE over a 1x1 logit matrix is exactly zero, and so is its
    gradient. Measured here — the first version of this test read 0.0 and it was
    the data, not the wiring. A gate asserting a positive norm on that batch would
    be unsatisfiable for a reason nothing in the record could name.
    """
    config = timing_config(config_mapping, **{"run.purpose": "probe", "train.batch_size": 4})

    code, record = pod_setting(config, text_only_rows(8))

    assert code == 0
    metrics_block = record["metrics"]
    assert metrics_block["grad_norm"] > 0
    assert metrics_block["trainable_params"] == metrics_block["total_params"] > 0
    assert metrics_block["params_with_grad"] > 0
    # The declared denominator, and the block that says how the figure was made.
    # `summarise` refuses a config whose denominator this run never counted.
    assert metrics_block["measurement"]["declared"] is True
    assert metrics_block["padded_tokens_per_step"] > 0
    assert metrics_block["profiled"] is False


def test_the_axes_an_adapter_owns_reach_the_record_instead_of_being_undetermined(
    pod_setting,
    config_mapping,  # noqa: F811
    monkeypatch,
):
    """tevatron's `DenseModel.forward` computes the loss and the cross-device
    gather itself (decision 5). The adapter declares that on `AdapterOut`; nothing
    passed the declaration to `axes.assemble`, so `Built.owned_axes` stayed empty,
    `loss.name` came back undetermined and `assert_matches` refused every timing
    run of that cell.
    """
    adapter_binding(
        monkeypatch,
        framework="tevatron",
        owned_axes={
            "loss.name": "DenseModel.forward computes it",
            "parallel.cross_device_negatives": "DenseModel.forward gathers",
        },
    )
    # The config still requests `framework=native`, and that is deliberate: the
    # binding says tevatron, and ownership has to follow the object the adapter
    # returned rather than the request. If it followed the request, writing
    # `framework=tevatron` in a config would exempt a cell from the axes it was
    # least likely to apply.
    #
    # `purpose=probe`: this CPU host mismatches `optim.name` and `precision.name`
    # for reasons that have nothing to do with ownership, and an enforced purpose
    # would refuse the run before the question here could be asked. What the
    # ownership has to survive is `capture`, which runs either way.
    config = timing_config(config_mapping, **{"run.purpose": "probe"})

    code, record = pod_setting(config, text_only_rows(8))

    assert code == 0, record
    assert record["applied"]["framework_owned"] == [
        "loss.name",
        "parallel.cross_device_negatives",
    ]
    owned = {a["axis"]: a for a in record["applied"]["axes"]}
    assert owned["loss.name"]["owner"] == "tevatron"
    # Ownership is a declaration that nobody looked, never a certification.
    assert owned["loss.name"]["applied"] is None
    assert owned["loss.name"]["matches"] is False


def test_the_context_an_adapter_requires_is_entered_by_the_measured_step(
    pod_setting,
    config_mapping,  # noqa: F811
    monkeypatch,
):
    """axolotl loads `embed_tokens`/`lm_head` in fp32 beside a bf16 body, so its
    step runs under `torch.autocast` upstream (decision 1). The contract forbids
    the adapter from opening its own `with`, and `axes.step_context` had no
    parameter to receive the requirement — so the declaration existed and no step
    ever entered the region.

    Asserted from inside the model's forward: that is the only place that can say
    the region was live when the work happened.
    """
    seen = []

    class AutocastWatchingEmbedder(TinyEmbedder):
        def forward(self, input_ids, attention_mask=None, **kwargs):
            seen.append(torch.is_autocast_enabled("cpu"))
            return super().forward(input_ids, attention_mask, **kwargs)

    adapter_binding(
        monkeypatch,
        model=AutocastWatchingEmbedder(),
        required_step_context=SimpleNamespace(
            kind="autocast",
            device_type="cpu",
            dtype="bfloat16",
            reason="axolotl trains under torch.autocast(bfloat16)",
        ),
    )
    config = timing_config(config_mapping, **{"run.purpose": "probe"})

    code, _ = pod_setting(config, text_only_rows(8))

    assert code == 0
    assert seen and all(seen), "every measured forward has to run inside the region"


def test_the_memory_ceiling_in_the_measured_loop_is_a_result_and_not_a_crash(
    pod_setting,
    config_mapping,  # noqa: F811
    monkeypatch,
):
    """OOM is a fifth outcome, not a slow one.

    "This combination does not fit at this batch size on this device" answers the
    study's question; a process that died with no `--out` does not, and
    docker/entrypoint.sh files it as a combination nobody attempted. So the record
    is written, stamped `oom`, and carries **no** `metrics` block — a metrics block
    asserts that a measured window completed, and this one did not.

    Narrow on purpose: only around the measured loop. An OOM during construction
    still leaves no result (the test above pins that), because that is a different
    finding and belongs to a different lane's design.
    """
    config = timing_config(config_mapping, **{"run.purpose": "probe"})

    def out_of_memory(*_args, **_kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(bench_entry, "train", out_of_memory)

    code, record = pod_setting(config, text_only_rows(8))

    assert code == bench_entry.OOM_EXIT
    assert code != bench_entry.REFUSED_EXIT, "declined and ran-out-of-memory are read apart"
    assert record["status"] == "oom"
    assert record["oom"]["error_type"] == "OutOfMemoryError"
    assert "metrics" not in record
