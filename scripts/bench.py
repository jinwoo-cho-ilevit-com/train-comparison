"""Measure one setting: one model, one axis configuration, one number.

Runs inside every framework image, so it takes a resolved config JSON rather than
composing with Hydra (Hydra's antlr4 pin is incompatible with axolotl), the same
interface `scripts/verify_env.py` uses.

    python scripts/bench.py --config resolved.json --out result.json

**One setting per process.** A sweep is the pod re-running this file once per
entry in its plan (docker/entrypoint.sh), not a loop in here. A process that has
already run a setting carries that setting's autotune cache, compiled graphs and
allocator fragmentation into the next one, and `kernel`/`attn` cannot be changed
after the model exists at all — `axes.patch` runs before construction. Reusing the
process would trade the thing being measured for the time it takes to load a model.

The five calls to `trainbench/axes.py` and `trainbench/applied.py` are what make a
number reportable, and `audit_plan.py`'s `assert-called` requires this file to make
all five. `assert_matches` is called here directly rather than through
`trainbench/probe/steps.py::verify_axes`, which wraps it in `report.run(...)` and
therefore *swallows* the raise — a harness built on that would satisfy the audit
while a mismatched axis went on to produce a number.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import torch

from trainbench import axes, metrics
from trainbench.applied import assert_matches, capture
from trainbench.config import load_bench_config
from trainbench.config_schema import BenchConfig
from trainbench.device import get_device
from trainbench.embedding import align_padding_side, info_nce
from trainbench.probe import steps
from trainbench.record import build_record, write_json
from trainbench.seed import set_seed

# MMEB stores its own placeholder markup inside `qry` / `pos_text` verbatim
# (`scripts/prepare_data.py`): `"<|image_1|>\nRepresent the given image.\n"`. It is
# MMEB's markup, not any model's, and this is the loader that converts — each model
# has different image tokens, which `apply_chat_template` is what inserts. Leaving
# the marker in would feed one model literal text where another model's placeholder
# belongs.
MMEB_IMAGE_MARKER = re.compile(r"<\|image_\d+\|>")


class MicroBatch(NamedTuple):
    """One collated micro-batch, plus what it contained.

    The counts are computed in the collate — which is to say in the DataLoader
    worker, alongside tokenisation — rather than in the training step. Counting in
    the step would put a reduction over the mask inside the timed window, and once
    the batch is on the device that reduction is a synchronisation.

    `tensors` is what goes to `model(**tensors)`; nothing else in here may reach it.
    """

    tensors: dict[str, torch.Tensor]
    tokens: int
    padded_tokens: int
    rows: int
    samples: int
    images: int
    images_dropped: int


class PairDataset(torch.utils.data.Dataset):
    """Rows of the pinned subset, untouched.

    Yields the raw row so the collate can reach `qry_image` / `pos_image` as well
    as the text. Tokenisation and image processing happen in the collate, inside
    the timed window, so that `dataloader.pretokenize` and `dataloader.packing`
    stay real axes: pre-tokenising moves that work out of the measured step, which
    is the very difference those axes exist to measure.

    `column_names` is declared because `applied._capture_dataloader_pretokenize`
    reads it to decide whether the rows arrive already tokenised. A dataset that
    declares nothing leaves that axis undetermined, and an undetermined axis blocks
    a timing run exactly like a mismatched one (docs/CONTRACTS.md §2).
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("PairDataset over zero rows; there is nothing to measure")
        self.rows = rows
        self.column_names = sorted({key for row in rows for key in row})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def load_pairs(config: BenchConfig) -> PairDataset:
    """Rows from the pinned subset revision.

    Reads `config.data.repo_id` / `config.data.revision` / `config.data.effective_rows`
    through the config object. The pin is the point: a run that streamed a branch
    would report a number measured on a corpus nobody can name afterwards, which is
    what `config_schema.py`'s revision validators exist to prevent.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        config.data.repo_id,
        revision=config.data.revision,
        split="train",
        streaming=True,
    )
    wanted = config.data.effective_rows
    rows = list(dataset.take(wanted))
    if len(rows) < wanted:
        raise RuntimeError(
            f"{config.data.repo_id}@{config.data.revision} yielded {len(rows)} rows, "
            f"asked for {wanted}; a short corpus makes every throughput figure optimistic"
        )
    return PairDataset(rows)


class Collate:
    """Queries and positives in one batch, queries first, images included.

    `info_nce` splits the pooled embeddings down the middle, so the halves have to
    line up: every query text comes before every positive text, and the flat image
    list follows that same order because processors consume images in the order
    their placeholders appear across the batch.

    A class rather than a closure for two reasons, both of which are how the
    previous version broke a real run:

    * `applied._capture_dataloader_packing` reads `axis_packing` off the collate.
      A closure carries no such attribute, so assigning one to the loader turned
      `dataloader.packing` from determined-False into undetermined, and
      `assert_matches` then refused every timing run
    * `configs/data/*.yaml` set `num_workers: 8`, and a local closure cannot be
      pickled to a spawned worker

    Nothing here touches the device. The previous version moved tensors inside the
    collate, which on fork runs `.to(cuda)` in a child of a process that had
    already initialised CUDA (`RuntimeError: Cannot re-initialize CUDA in forked
    subprocess`). The transfer belongs in the step anyway — that is where its cost
    is part of what a step costs.
    """

    # Read back by applied._capture_dataloader_packing. This collate pads a batch
    # to its longest row; it does not concatenate sequences, so the answer is False
    # and it is declared rather than inferred.
    axis_packing = False

    def __init__(self, processor: Any, config: BenchConfig) -> None:
        self.processor = processor
        self.config = config
        # The query side carries the model's official instruction prompt; the
        # positive side never does (docs/model-spec.md). It is a constant on one
        # side of the pair, which is also why `tokens` is not a clean comparison
        # across models — METRIC_DEFINITIONS says so in the result.
        self.prompt = config.model.instruction_prompt or ""
        # Whether this processor can take pixels at all, read off the processor
        # rather than branched on `model.arch`: a text-only checkpoint returns a
        # bare tokeniser from AutoProcessor. Rows still carry their images; they
        # are counted as dropped so the result says plainly that this model read a
        # text-only view of an image corpus.
        self.accepts_images = getattr(processor, "image_processor", None) is not None

    def _text(self, raw: str | None, with_image: bool) -> str:
        """One side of a pair, in this model's own chat format.

        `add_generation_prompt` is `config.model.add_generation_prompt`, which is
        exactly this argument (docs/CONTRACTS.md §5) — with last-token pooling it
        decides which token becomes the embedding, so it cannot be defaulted here.
        """
        text = MMEB_IMAGE_MARKER.sub("", raw or "").strip()
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if with_image:
            content.insert(0, {"type": "image"})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=self.config.model.add_generation_prompt,
        )

    def __call__(self, rows: list[dict[str, Any]]) -> MicroBatch:
        queries: list[str] = []
        positives: list[str] = []
        query_images: list[Any] = []
        positive_images: list[Any] = []
        dropped = 0

        for row in rows:
            side_images = []
            for column, bucket in (("qry_image", query_images), ("pos_image", positive_images)):
                image = row.get(column)
                if image is None:
                    side_images.append(False)
                elif self.accepts_images:
                    bucket.append(image)
                    side_images.append(True)
                else:
                    dropped += 1
                    side_images.append(False)
            # Rows differ in which image columns they carry — 4 of the 20 pinned
            # configs have no `qry_image` and 13 no `pos_image`
            # (docs/evidence/data-subset-mmeb-subset.json). A row without one gets
            # no placeholder, which is what lets text-only and image rows share a
            # batch: the flat image list below then has exactly as many entries as
            # there are placeholders, in the same order.
            queries.append(self.prompt + self._text(row.get("qry"), side_images[0]))
            positives.append(self._text(row.get("pos_text"), side_images[1]))

        images = query_images + positive_images
        kwargs: dict[str, Any] = {
            "text": queries + positives,
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            kwargs["images"] = images
        else:
            # Truncation is safe only with no pixels in the batch. `max_seq_len` is
            # enforced against the image case below instead of by cutting it: the
            # placeholders expand into one token per image feature, and truncating
            # them away while keeping `pixel_values` is the N-features-vs-M-tokens
            # mismatch the forward pass dies on.
            kwargs["truncation"] = True
            kwargs["max_length"] = self.config.data.max_seq_len

        encoded = self.processor(**kwargs)
        input_ids = encoded["input_ids"]
        length = int(input_ids.shape[1])
        if images and length > self.config.data.max_seq_len:
            raise RuntimeError(
                f"a batch of {len(images)} image(s) expanded to {length} tokens, over "
                f"data.max_seq_len={self.config.data.max_seq_len}. Truncating it would cut "
                "image placeholders away from the pixels they stand for and the forward pass "
                "would fail on the count mismatch; raise data.max_seq_len (or lower the "
                "processor's pixel budget) rather than measuring a truncated multimodal batch."
            )

        return MicroBatch(
            tensors=dict(encoded),
            tokens=int(encoded["attention_mask"].sum()),
            padded_tokens=int(input_ids.numel()),
            rows=int(input_ids.shape[0]),
            samples=len(rows),
            images=len(images),
            images_dropped=dropped,
        )


def micro_batches(loader: Iterable[Any]) -> Iterator[Any]:
    """The loader, restarted as often as the step count needs, but never silently.

    A step now draws `grad_accum` batches, so the loop cannot be `for batch in
    loader` any more. Restarting has to be explicit, and so does the failure: the
    previous `while step < total: for batch in loader:` had no progress guarantee,
    and a loader yielding nothing spun with no output and no exception until the
    pod deadline killed it.
    """
    while True:
        produced = 0
        for batch in loader:
            produced += 1
            yield batch
        if produced == 0:
            raise RuntimeError(
                f"{type(loader).__name__} yielded no batches, so the measured loop cannot "
                "advance; it would otherwise spin until the pod deadline killed it"
            )


def to_device(tensors: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Host-to-device transfer, called inside the timed window.

    Floating-point inputs are cast to the model's dtype: `pixel_values` comes off
    the image processor in fp32 while the model is loaded in bf16, and the vision
    tower would raise on the mismatch. Integer inputs (`input_ids`,
    `attention_mask`, `image_grid_thw`) keep their dtype.
    """
    moved: dict[str, Any] = {}
    for key, value in tensors.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device)
    return moved


def train(built: Any, loader: Any, config: BenchConfig, device: torch.device) -> dict[str, Any]:
    """The measured loop.

    **One step = fetch + transfer + forward + backward + optimizer step, all inside
    the timer.** The previous shape (`for batch in loader:` outside `with timer`)
    produced the batch in the loop header, so tokenisation, collate and the
    host-to-device copy all happened before t0. That does not merely inflate the
    number: `dataloader.backend`, `dataloader.packing` and `dataloader.pretokenize`
    are ablation axes of this study, and with the data pipeline outside the window
    every value of them measures the same step. The ablation returns zero by
    construction, and PLAN.md's dataloader-bottleneck check — a prerequisite for
    Phase 2 — cannot be performed at all.

    Every step goes inside `axes.step_context(config)`. The fp8 recipes wrap the
    forward pass, so a loop that applied them anywhere else would report an fp8
    number for a step that never entered the recipe — and the capture probe would
    still find the swapped modules and call it a match.

    Nothing inside the window reads a device tensor into Python. `float(loss)` is
    `.item()`, a blocking device-to-host copy, and putting it after `backward()`
    forfeits the CPU run-ahead that would have overlapped with GPU execution. It
    biases along the axis being measured: `compile` and `kernel=liger` exist to cut
    kernel launches, so they forfeit less and their speedup reads as larger than it
    is. The losses are kept as detached tensors and converted after the loop. How
    much this cost is unmeasured — no GPU here (docs/methodology.md).

    Warmup steps are timed and then discarded by `metrics.summarise`, which records
    how many. The peak memory counter is reset after warmup so the figure belongs
    to the measured window rather than to model construction and autotuning; the
    losses and the counts are gathered post-discard for the same reason — the
    reported `loss_first` used to be the first *warmup* step while every other
    figure in the summary was post-discard.
    """
    if config.model.pooling != "lasttoken":
        raise RuntimeError(
            f"model.pooling={config.model.pooling} is declared but "
            "trainbench/probe/steps.py::encode pools the last token unconditionally; "
            "the knob would name one thing while the run measured another."
        )
    built.model.train()
    timer = metrics.StepTimer(device)
    total = config.train.steps
    discard = config.train.warmup_discard_steps
    grad_accum = config.train.grad_accum
    side = config.model.padding_side
    dtype = steps.dtype_for(device)
    stream = micro_batches(loader)
    counted = dict.fromkeys(
        ("tokens", "padded_tokens", "rows", "samples", "images", "images_dropped"), 0
    )
    first_loss: torch.Tensor | None = None
    last_loss: torch.Tensor | None = None

    for step in range(total):
        if step == discard:
            metrics.reset_peak_memory(device)
        measured = step >= discard
        with timer, axes.step_context(config):
            for _ in range(grad_accum):
                # `grad_accum` distinct micro-batches, not the same one N times.
                # Feeding one batch repeatedly gives identical sequence lengths,
                # identical padding and a warm cache, and nothing in the result
                # would say so.
                micro = next(stream)
                tensors = to_device(micro.tensors, device, dtype)
                # Not `steps.infonce_backward`: it ends with
                # `model.zero_grad(set_to_none=True)`, which is right for a probe
                # (one step, nothing after it) and wrong here — the optimizer
                # would step on gradients that had just been wiped, and the loop
                # would time a forward and a backward while calling it training.
                # Caught by tests/test_smoke_cpu.py, which asserts the gradients
                # are non-zero at the moment of the step. `encode` and `info_nce`
                # are reused unchanged.
                pooled = steps.encode(built.model, tensors, side)
                half = pooled.shape[0] // 2
                loss = info_nce(pooled[:half], pooled[half:], config.loss.temperature)
                # Scaled so N micro-batches accumulate to one batch's gradient
                # rather than N times it. The recorded loss is the unscaled one, so
                # it stays comparable across grad_accum settings.
                (loss / grad_accum).backward()
                if measured:
                    detached = loss.detach()
                    first_loss = detached if first_loss is None else first_loss
                    last_loss = detached
                    # Counted from the micro-batch that was actually fed, not from
                    # the config: with distinct micro-batches the padding differs
                    # per batch, so multiplying one batch's token count by
                    # grad_accum is wrong.
                    for name in counted:
                        counted[name] += getattr(micro, name)
            built.optimizer.step()
            built.optimizer.zero_grad(set_to_none=True)

    kept = max(1, len(timer.durations) - discard)
    summary = metrics.summarise(
        timer.durations,
        discard=discard,
        rows_per_step=counted["rows"] / kept,
        tokens_per_step=counted["tokens"] / kept,
        peak_bytes=metrics.peak_memory_bytes(device),
        extra_counts={
            name: counted[name] / kept
            for name in ("samples", "padded_tokens", "images", "images_dropped")
        },
        totals={
            "images_read_total": counted["images"],
            "images_dropped_total": counted["images_dropped"],
        },
    )
    # Converted here, outside the timed window: this is the device sync the loop
    # exists to keep out of the measurement.
    summary["loss_first"] = float(first_loss) if first_loss is not None else None
    summary["loss_last"] = float(last_loss) if last_loss is not None else None
    summary["loss_definition"] = (
        "unscaled InfoNCE of one micro-batch. loss_first is the first micro-batch of the "
        "first measured step, loss_last the last micro-batch of the final step; warmup steps "
        "are excluded, as they are from every other figure here"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="resolved config JSON")
    parser.add_argument("--out", required=True, type=Path, help="where to write the result")
    args = parser.parse_args(argv)

    config = load_bench_config(args.config)
    device = get_device(config.device)
    # Follows the config rather than forcing determinism the way the probe does.
    # Deterministic mode disables kernel autotuning, which is part of what a timing
    # run measures, and the schema already refuses `deterministic=true` for
    # `purpose=timing` — hardcoding it here would override that silently.
    set_seed(config.train.seed, deterministic=config.train.deterministic, warn_only=True)

    from transformers import AutoModel, AutoProcessor

    axes.patch(config)
    processor = AutoProcessor.from_pretrained(config.model.hf_id, revision=config.model.revision)
    align_padding_side(processor, config.model.padding_side)
    model = AutoModel.from_pretrained(
        config.model.hf_id,
        revision=config.model.revision,
        dtype=steps.dtype_for(device),
        **axes.load_kwargs(config),
    )
    model.to(device)

    dataset = load_pairs(config)
    built, applied = axes.assemble(model, config, device, framework="native", dataset=dataset)
    # `assemble` has no collate argument, so the loader it builds carries torch's
    # default one and this is the only place to replace it. The replacement declares
    # `axis_packing`, which is what `applied._capture_dataloader_packing` reads —
    # without it this assignment would turn a determined axis into an undetermined
    # one and `assert_matches` below would refuse the run.
    built.dataloader.collate_fn = Collate(processor, config)

    state = capture(built, config)
    # Directly, and before a single step runs. Not inside a try, and not through
    # `steps.verify_axes`: the whole value of this call is that it stops the run.
    assert_matches(state, config)

    # Timing and profiling are separate runs (AGENTS.md). The schema already
    # refuses `run.profiler=true` for `purpose=timing`; this is where the other
    # purposes act on it, and the trace is written next to the result rather than
    # merged into it so no reported number can come from a profiled step.
    if config.run.profiler:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            if device.type == "cuda"
            else [torch.profiler.ProfilerActivity.CPU],
        ) as profile:
            summary = train(built, built.dataloader, config, device)
        trace = args.out.with_suffix(".trace.json")
        profile.export_chrome_trace(str(trace))
        summary["profiled"] = True
        summary["trace_path"] = str(trace)
        print(f"wrote {trace}")
    else:
        summary = train(built, built.dataloader, config, device)
        summary["profiled"] = False
    record = build_record(config, device, applied=state, metrics=summary, applied_axes=applied)
    write_json(args.out, record)

    print(f"{config.model.name} x {config.framework.name}: {summary['steps_measured']} steps")
    print(f"  step p50 {summary['step_seconds_p50']:.4f}s  p95 {summary['step_seconds_p95']:.4f}s")
    # samples/s is PLAN.md's figure and rows/s is twice it, so both are named.
    print(
        f"  samples/s {summary['samples_per_second']:.2f}  "
        f"rows/s {summary['rows_per_second']:.2f}  "
        f"tokens/s {summary['tokens_per_second']}"
    )
    print(
        f"  images read {summary['images_read_total']}  dropped {summary['images_dropped_total']}"
    )
    print(f"  peak memory {summary['peak_memory_bytes']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
