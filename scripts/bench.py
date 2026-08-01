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
from trainbench.embedding import align_padding_side, packed_last_token_pool
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
    # Sequence boundaries, present only when `dataloader.packing=true`. They are
    # kept out of `tensors` because `model(**tensors)` would reject them: they
    # belong to pooling, not to the forward pass (axes.PACKED_BOUNDARY_KEYS).
    # None is what tells the step to pool the padded way.
    cu_seqlens: torch.Tensor | None = None


class PairTexts(NamedTuple):
    """One batch's 2N templated strings, queries first, and the images for them."""

    texts: list[str]
    images: list[Any]
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

    def pair_texts(self, rows: list[dict[str, Any]], *, with_images: bool = True) -> PairTexts:
        """The 2N templated strings and the images that go with them.

        Split out of `__call__` because the packing and pretokenize paths need the
        same strings from the same rows and must not grow a second spelling of how
        a pair becomes text — which model's placeholder, which side carries the
        instruction prompt, where the MMEB marker went.

        `with_images=False` is the packed path: `axes.PackedCollate` returns token
        ids alone, so pixels cannot ride along and every image in the rows is
        counted as dropped rather than quietly forgotten.
        """
        queries: list[str] = []
        positives: list[str] = []
        query_images: list[Any] = []
        positive_images: list[Any] = []
        dropped = 0
        take_images = with_images and self.accepts_images

        for row in rows:
            side_images = []
            for column, bucket in (("qry_image", query_images), ("pos_image", positive_images)):
                image = row.get(column)
                if image is None:
                    side_images.append(False)
                elif take_images:
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

        return PairTexts(
            texts=queries + positives,
            images=query_images + positive_images,
            images_dropped=dropped + sum(int(row.get("images_dropped", 0) or 0) for row in rows),
        )

    def __call__(self, rows: list[dict[str, Any]]) -> MicroBatch:
        built = self.pair_texts(rows)
        images = built.images
        dropped = built.images_dropped
        kwargs: dict[str, Any] = {
            "text": built.texts,
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


class Encode:
    """One row, tokenised before the timed window opens.

    Handed to `axes.pretokenize`, which is what `dataloader.pretokenize=true`
    means: the tokenisation does not change, it moves out of the measured step.
    Each side is tokenised **alone and unpadded**, which is also the only form
    `axes.PackedCollate` will accept — a row tokenised as part of a padded batch
    carries PAD that packing would count as real tokens.

    Images cannot survive this: the row that comes back is token ids, and pixels
    tokenised now would have to be carried as tensors through the dataset. How many
    were left behind travels in the row so the result can report it rather than
    quietly measuring a text-only run on an image corpus.
    """

    def __init__(self, processor: Any, config: BenchConfig) -> None:
        self.collate = Collate(processor, config)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = config.data.max_seq_len

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        built = self.collate.pair_texts([row], with_images=False)
        # One call for the pair, `padding=False`: the two sides of a row differ in
        # length and padding them against each other here would write PAD into the
        # dataset, where nothing downstream could tell it from content.
        query, positive = self.tokenizer(
            built.texts, padding=False, truncation=True, max_length=self.max_length
        )["input_ids"]
        return {
            "input_ids": list(query),
            "positive_input_ids": list(positive),
            "images_dropped": built.images_dropped,
        }


class PackedPairs:
    """`axes.PackedCollate`'s `tokenize` hook: one 1-D id tensor per sequence.

    Queries first, then positives, because `info_nce` splits the pooled embeddings
    at the midpoint — `PackedCollate` pools in packing order, so the order this
    returns *is* the pairing.

    Rows that already carry `input_ids` (`dataloader.pretokenize=true`) are read
    rather than tokenised again; the rest are tokenised here with `padding=False`,
    which is what `PackedCollate` requires and checks. It is the same hook either
    way because what `PackedCollate` asks for is the sequences in loss order, and
    only the source of the ids differs.
    """

    def __init__(self, processor: Any, config: BenchConfig) -> None:
        self.collate = Collate(processor, config)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = config.data.max_seq_len

    def __call__(self, rows: list[dict[str, Any]]) -> list[torch.Tensor]:
        if all("input_ids" in row for row in rows):
            queries = [torch.as_tensor(row["input_ids"]) for row in rows]
            positives = [torch.as_tensor(row["positive_input_ids"]) for row in rows]
            return queries + positives
        built = self.collate.pair_texts(rows, with_images=False)
        encoded = self.tokenizer(
            built.texts, padding=False, truncation=True, max_length=self.max_length
        )
        return [torch.as_tensor(ids) for ids in encoded["input_ids"]]


class PackedBatches:
    """`axes.PackedCollate`, wrapped so the step gets the same `MicroBatch` shape.

    The wrapper exists for the accounting, not for the packing: `PackedCollate`
    returns bare tensors, and the measured loop needs to know how many samples,
    tokens and images that batch stood for. The boundary vectors are lifted out of
    the batch here rather than in the step, so what stays in `tensors` is exactly
    what `model(**tensors)` takes.

    `axis_packing` is **read off the wrapped collate**, never declared again. A
    second declaration is a second answer to the question
    `applied._capture_dataloader_packing` asks, and two answers is how one of them
    drifts into a label the run did not earn.

    A packed batch has no padding by construction, so `tokens` and
    `padded_tokens` are the same number — which is the whole of what packing
    claims to save.
    """

    def __init__(self, packed: Any) -> None:
        self.packed = packed
        self.axis_packing = packed.axis_packing

    def __call__(self, rows: list[dict[str, Any]]) -> MicroBatch:
        # Counted before delegating: `PackedCollate` returns ids alone, so this is
        # the last point at which the rows still say what was left behind.
        dropped = sum(
            int(row.get("images_dropped", 0) or 0)
            if "images_dropped" in row
            else sum(1 for column in axes.IMAGE_COLUMNS if row.get(column) is not None)
            for row in rows
        )
        batch = self.packed(rows)
        boundaries = {key: batch.pop(key) for key in axes.PACKED_BOUNDARY_KEYS}
        total = int(batch["input_ids"].numel())
        return MicroBatch(
            tensors=batch,
            tokens=total,
            padded_tokens=total,
            rows=int(boundaries["seq_lengths"].numel()),
            samples=len(rows),
            images=0,
            images_dropped=dropped,
            cu_seqlens=boundaries["cu_seqlens"],
        )


class PretokenizedCollate:
    """Pre-tokenised rows padded back into a rectangle, queries first.

    `dataloader.pretokenize=true` without packing still needs a padded batch, and
    torch's default collate cannot build one out of variable-length id lists. The
    padding goes on `config.model.padding_side` because `last_token_pool` checks
    the mask against that side and refuses to pool a PAD.
    """

    # Read back by applied._capture_dataloader_packing. This pads; it does not
    # concatenate sequences.
    axis_packing = False

    def __init__(self, processor: Any, config: BenchConfig) -> None:
        self.pad_id = steps.pad_token_id(processor) or 0
        self.padding_side = config.model.padding_side

    def __call__(self, rows: list[dict[str, Any]]) -> MicroBatch:
        sequences = [torch.as_tensor(row["input_ids"]) for row in rows]
        sequences += [torch.as_tensor(row["positive_input_ids"]) for row in rows]
        width = max(int(sequence.numel()) for sequence in sequences)
        input_ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for index, sequence in enumerate(sequences):
            length = int(sequence.numel())
            span = (
                slice(0, length) if self.padding_side == "right" else slice(width - length, width)
            )
            input_ids[index, span] = sequence.to(torch.long)
            attention_mask[index, span] = 1
        return MicroBatch(
            tensors={"input_ids": input_ids, "attention_mask": attention_mask},
            tokens=int(attention_mask.sum()),
            padded_tokens=int(input_ids.numel()),
            rows=len(sequences),
            samples=len(rows),
            images=0,
            images_dropped=sum(int(row.get("images_dropped", 0) or 0) for row in rows),
        )


def build_collate(processor: Any, config: BenchConfig) -> Any:
    """The collate this run's `dataloader.*` axes call for.

    Assigned over whatever `axes.assemble` built, which is the only way in — that
    function takes no collate argument. The packed branch wraps `axes.PackedCollate`
    rather than replacing it, so `dataloader.packing` is still applied and read back
    from the class that owns it; overwriting it outright is what left `torch_packed`
    certified False against a True request and refused at `assert_matches`.
    """
    if not config.dataloader.packing:
        return (
            PretokenizedCollate(processor, config)
            if config.dataloader.pretokenize
            else Collate(processor, config)
        )
    pad_id = steps.pad_token_id(processor)
    if pad_id is None:
        raise RuntimeError(
            "dataloader.packing=true needs the processor's pad token id: PackedCollate "
            "searches every sequence for it, because a PAD packed as a real token inflates "
            "tokens/s and becomes some sequence's pooled embedding while the run still "
            "certifies packing as applied. This processor declares no pad token."
        )
    return PackedBatches(axes.PackedCollate(tokenize=PackedPairs(processor, config), pad_id=pad_id))


def pooled_embeddings(
    model: Any, tensors: dict[str, Any], padding_side: str, cu_seqlens: Any = None
) -> torch.Tensor:
    """The batch's embeddings, pooled the way this batch is shaped.

    The padded case is `steps.encode` unchanged. The packed case cannot use it:
    a packed batch has no `attention_mask` and one row holds every sequence end to
    end, which is precisely the contract `last_token_pool` refuses to weaken. The
    hidden-state lookup is the one thing duplicated from `steps.encode`; that
    function pools unconditionally, so there is nothing there to reuse that stops
    short of pooling. It belongs in `trainbench/probe/steps.py` next to its twin
    the moment that lane wants it.
    """
    if cu_seqlens is None:
        return steps.encode(model, tensors, padding_side)
    output = model(**tensors, output_hidden_states=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(output, "hidden_states", None)
        hidden = hidden[-1] if hidden else output[0]
    return packed_last_token_pool(hidden, cu_seqlens)


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

    **The loss comes from `built.loss_fn`, never from `info_nce` directly.** The
    previous version called `info_nce` inline, which was correct for `loss=mnrl`
    and silently wrong for everything else: once `axes._loss` learned to build
    `cached_mnrl`, `assemble` and `assert_matches` both passed while the loop went
    on measuring ordinary in-batch negatives and the result carried the label
    `cached_mnrl`. A crash had become a mislabelled number, which is worse — and
    `capture` reads `built.loss_fn` to certify `loss.name`, so the object that was
    certified was the one object the loop did not use.

    Two shapes of loss are supported, and which one a run takes is read off the
    callable rather than branched on `config.loss.name` — the config is the request
    and this loop runs what was built (docs/CONTRACTS.md §2):

    * a plain `(queries, documents) -> loss`, whose backward this loop issues;
    * one carrying `gradcache_backward`, which encodes the batch twice and issues
      its own backward, returning the loss already detached.

    Both stay entirely inside the timer. GradCache's second forward pass is a real
    cost of that setting and has to land in the step time it is compared on.
    `grad_accum` scales both the same way — `loss / grad_accum` on one path and
    `scale=1 / grad_accum` on the other, which multiplies the gradient and not the
    returned loss — so the two paths accumulate to one batch's gradient rather than
    to different effective learning rates.
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
    # Read once, outside the loop: a getattr per micro-batch would be a dictionary
    # lookup inside the timed window on every step of every run.
    gradcache_backward = getattr(built.loss_fn, "gradcache_backward", None)
    if gradcache_backward is not None and config.dataloader.packing:
        raise RuntimeError(
            "loss=cached_mnrl with dataloader.packing=true: GradCache splits the batch into "
            "row-wise pieces and pools each with the padded convention, and a packed batch is "
            "one row whose boundaries live in cu_seqlens. It would pool the wrong positions "
            "and still report both axes as applied. Measure the two axes separately."
        )
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
                # are non-zero at the moment of the step. `encode` is reused
                # unchanged.
                if gradcache_backward is not None:
                    # GradCache owns the whole forward/backward: it encodes the
                    # batch in pieces under no_grad, scores every row at once, then
                    # re-encodes each piece with a graph and seeds its backward
                    # from the cache. `scale` multiplies the gradient rather than
                    # the returned loss, so the loss recorded below is the unscaled
                    # one on this path too.
                    loss = gradcache_backward(
                        built.model, tensors, padding_side=side, scale=1.0 / grad_accum
                    )
                else:
                    # `micro.cu_seqlens` is None unless `dataloader.packing=true`,
                    # and it is what selects the pooling: a packed batch is one row
                    # with no attention_mask, which `last_token_pool` refuses.
                    pooled = pooled_embeddings(built.model, tensors, side, micro.cu_seqlens)
                    half = pooled.shape[0] // 2
                    # `built.loss_fn`, not `info_nce`: the temperature and any
                    # cross-rank gather are already closed over by the callable
                    # `axes.assemble` built, and that callable is the one
                    # `applied.capture` reads to certify `loss.name`.
                    loss = built.loss_fn(pooled[:half], pooled[half:])
                    # Scaled so N micro-batches accumulate to one batch's gradient
                    # rather than N times it. The recorded loss is the unscaled one,
                    # so it stays comparable across grad_accum settings.
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
        f"unscaled {config.loss.name} over one micro-batch, as computed by the loss "
        "built.loss_fn holds — not a separately recomputed InfoNCE. loss_first is the first "
        "micro-batch of the first measured step, loss_last the last micro-batch of the final "
        "step; warmup steps are excluded, as they are from every other figure here"
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
    if config.dataloader.pretokenize:
        # Before `assemble`, which is the whole of the axis: `_dataloader` refuses
        # `pretokenize=true` over rows that do not already carry token ids, because
        # building the loader anyway would leave the tokenisation inside the timed
        # step under a pretokenized label.
        dataset = axes.pretokenize(dataset, Encode(processor, config))
    built, applied = axes.assemble(model, config, device, framework="native", dataset=dataset)
    # `assemble` has no collate argument, so the loader it builds carries either
    # torch's default one or `axes.PackedCollate`, and this is the only place to
    # replace it. What goes in declares `axis_packing`, which is what
    # `applied._capture_dataloader_packing` reads — an assignment that did not would
    # turn a determined axis into an undetermined one and `assert_matches` below
    # would refuse the run.
    built.dataloader.collate_fn = build_collate(processor, config)

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
