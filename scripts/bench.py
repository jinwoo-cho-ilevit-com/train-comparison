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

A setting those calls refuse still writes `--out` (`refusal_record`) and still
exits non-zero. "Ran, and this data or this image cannot do it" is a result of
this study and has to reach the report; before, only the exit code survived the
pod and the reason stayed in a log nobody reads afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import torch

from trainbench import axes, metrics
from trainbench.applied import AppliedMismatch, AppliedState, assert_matches, capture
from trainbench.config import load_bench_config, to_bench_config
from trainbench.config_schema import BenchConfig, axis_knobs
from trainbench.device import get_device
from trainbench.embedding import align_padding_side, packed_last_token_pool
from trainbench.probe import steps
from trainbench.record import build_record, write_json
from trainbench.seed import set_seed

# Exit code for a setting refused before it measured anything. Distinct from 1,
# which is what an unhandled exception exits with and which leaves no result file
# at all; distinct from `timeout`'s 124 and from docker/entrypoint.sh's 125 and
# 127. The pod log is the only place an exit code is read, and "refused" and
# "crashed" are different findings there.
REFUSED_EXIT = 3

# Exit code for a plan whose settings cannot all run. Distinct from REFUSED_EXIT
# because they are different findings: that one is a setting that was attempted
# and declined, this one is a pod that measured nothing at all, and the pod log is
# where both are read.
PREFLIGHT_EXIT = 4

# The GPUs this image's compiled kernels cover, written into the image by
# `docker/Dockerfile.framework` beside the `FLASH_ATTN_CUDA_ARCHS` /
# `NVTE_CUDA_ARCHS` it mirrors. flash-attn emits `code=sm_XX` and no PTX, so a pod
# on a GPU outside the list fails with "no kernel image is available for execution
# on the device" — after the model is loaded and the first kernel launches.
CUDA_ARCHS_ENV = "TRAINBENCH_CUDA_ARCHS"

# `status` prefix on a refusal record. Not `no_result`: that value belongs to
# `publish_result.fallback_record` and means no result file existed, which is the
# case this exists to stop producing.
REFUSED_STATUS = "axis-refused"

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
    # How many images each row of `tensors` contributed, in batch-row order. The
    # processor flattens every row's images into one `pixel_values` and keeps no
    # record of where the boundaries were, so this collate is the last place that
    # knows — and `loss=cached_mnrl` cannot split a multimodal batch without it
    # (axes._split_rows). Kept out of `tensors` for the same reason as the
    # boundaries above: `model(**tensors)` would reject it.
    #
    # None means no batch of this shape carries pixels (the packed and pretokenized
    # collates drop images), not that the counts are zero — and `_split_rows`
    # refuses rather than assumes if pixels turn up anyway.
    images_per_row: tuple[int, ...] | None = None


class PairTexts(NamedTuple):
    """One batch's 2N templated strings, queries first, and the images for them.

    `images_per_row` counts, for each of those 2N strings and in the same order,
    how many of `images` belong to it. That is the map `loss=cached_mnrl` needs to
    cut `pixel_values` at the right place, and it is recoverable here and nowhere
    later: the processor consumes the flat list in placeholder order and returns
    one concatenated tensor.
    """

    texts: list[str]
    images: list[Any]
    images_dropped: int
    images_per_row: tuple[int, ...]


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


def _group_by_row(images: list[Any], images_per_row: tuple[int, ...]) -> list[list[Any]]:
    """The flat image list cut into one sublist per batch row.

    The same vector that tells `axes._split_rows` where a row's pixels begin also
    tells the processor which row each image belongs to, so there is one map and
    not two. A row that carries none gets `[]`, which is what keeps the sublist
    count equal to the text count — the equality `Gemma4Processor.validate_inputs`
    checks.
    """
    grouped: list[list[Any]] = []
    cursor = 0
    for count in images_per_row:
        grouped.append(images[cursor : cursor + count])
        cursor += count
    if cursor != len(images):
        raise RuntimeError(
            f"images_per_row accounts for {cursor} image(s) and the batch carries {len(images)}; "
            "the two are built in the same loop, so a disagreement means one row's images would "
            "be handed to another row's placeholders"
        )
    return grouped


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
        query_counts: list[int] = []
        positive_counts: list[int] = []
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
            query_counts.append(int(side_images[0]))
            positive_counts.append(int(side_images[1]))

        # Concatenated the same way as the texts and the images: every query row
        # first, then every positive. One order for all three, so a count belongs
        # to the string above it and to that string's slice of the flat image list.
        return PairTexts(
            texts=queries + positives,
            images=query_images + positive_images,
            images_dropped=dropped + sum(int(row.get("images_dropped", 0) or 0) for row in rows),
            images_per_row=tuple(query_counts + positive_counts),
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
            # Grouped per row, not one flat list. Measured 2026-08-02 against the
            # real processors: `Gemma4Processor` rejects a flat list outright —
            # `make_nested_list_of_images` reads it as one row's images and
            # `validate_inputs` raises "Received inconsistently sized batches of
            # images (1) and text (4)" — so no gemma-4 batch carrying images could
            # be built at all, for any loss. Both Qwen processors accept either and
            # return byte-identical tensors for the one-image-per-row case, so the
            # grouped form is the one shape all three take.
            kwargs["images"] = _group_by_row(images, built.images_per_row)
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
            images_per_row=built.images_per_row,
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
                    #
                    # `images_per_row` is what lets it cut `pixel_values` at a row
                    # boundary instead of refusing the batch. It is passed even
                    # when it is None — a text-only batch needs no map, and a
                    # multimodal one whose collate recorded none is refused rather
                    # than split by position.
                    loss = gradcache_backward(
                        built.model,
                        tensors,
                        padding_side=side,
                        scale=1.0 / grad_accum,
                        images_per_row=micro.images_per_row,
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


class RefusedSetting(RuntimeError):
    """A setting this pod cannot measure, tagged with where the refusal fired.

    Wraps the two refusals construction can end on. They are not the same finding,
    and the record says which:

    * `axes.UnappliedAxis` — nothing here can put the requested value into effect,
      because of what the image ships or what the data looks like. That is a
      property of the setting: the same pod re-running it gets the same answer, so
      it is a result and belongs in the report.
    * `applied.AppliedMismatch` — request and reality disagree, or an axis could
      not be read back at all. Sometimes that is the same kind of fact
      (`adamw_fused` resolves to `adamw_unfused` without CUDA, docs/CONTRACTS.md
      §6) and sometimes it is a defect in this harness — assigning a closure over
      `collate_fn` once made the harness refuse every one of its own runs. The two
      are told apart by *which* axes disagreed, so a mismatch record carries the
      whole `AppliedState` and a reader who cannot tell must not read it as a
      property of the hardware.

    The stage matters for the same reason: `patch` fires before the model exists,
    `assemble` during construction, `assert_matches` after. Only the last had a
    model to read back, so the stage is what says whether `applied` in the record
    means anything.
    """

    def __init__(self, stage: str, cause: Exception, state: AppliedState | None = None) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause
        self.state = state


@contextmanager
def refusing(stage: str, state: AppliedState | None = None) -> Iterator[None]:
    """Tag whichever refusal comes out of this region with the call site it fired at.

    Only the two refusal types are caught. Everything else — a checkpoint that will
    not download, an OOM, a collate that cannot find a pad id — passes through
    untouched and leaves no result file, which is what makes docker/entrypoint.sh
    publish a fallback record rather than a result for it. Widening this to
    `Exception` would turn every crash into a tidy record saying the axis could not
    be applied, which is a different and false claim.
    """
    try:
        yield
    except (axes.UnappliedAxis, AppliedMismatch) as exc:
        raise RefusedSetting(stage, exc, state) from exc


def refusal_record(
    config: BenchConfig, device: torch.device, refused: RefusedSetting
) -> dict[str, Any]:
    """The result file for a setting refused before it measured anything.

    **No `metrics` key, ever.** `scripts/report.py` renders any record carrying one
    as a measurement, so a refusal with a `metrics` block — even zeroed, even empty
    — would be a fabricated figure in the results table. Without it the record
    lands in that file's `지표 없음` list, whose whole subject is a pod-hour spent
    for no number.

    `status` is where the reason travels, because that is the one field report.py
    prints verbatim for such a record. Collapsed to a single line for that reason;
    the message is also in `refusal.reason` unwrapped, next to the structured
    fields.

    Every axis knob's requested value is recorded, not the one axis refused. Which
    axis an `UnappliedAxis` is about lives only in its prose, and recovering it by
    matching knob names against that prose is the regex-over-prose guess this
    repository has already had to take out of `config-consumed`. The reason
    sentence names the axis; the map says what the whole setting asked for.

    No `probe` block, though `publish_result.fallback_record` builds one for the
    same shape of event. It would make the artifact `report.Artifact.graded_here`,
    and a refused timing setting would then outrank the probe that graded the same
    (framework, model) cell and take the cell over.
    """
    reason = " ".join(str(refused.cause).split())
    return build_record(
        config,
        device,
        applied=refused.state,
        status=f"{REFUSED_STATUS} ({refused.stage}, {type(refused.cause).__name__}) — {reason}",
        refusal={
            "kind": type(refused.cause).__name__,
            "stage": refused.stage,
            "reason": str(refused.cause),
            "requested_axes": {knob: str(read(config)) for knob, read in axis_knobs().items()},
        },
    )


def build_run(config: BenchConfig, device: torch.device) -> tuple[Any, list[str], AppliedState]:
    """Everything between the resolved config and the first measured step.

    Split out of `main` so that the refusals can be caught around a region that
    *stops* at `assert_matches`. The measured loop is deliberately outside it: an
    exception raised in there is a failure partway through a measurement, and
    filing it as a clean refusal would say a setting was declined when a loop had
    already run. `tests/test_smoke_cpu.py` pins that boundary.
    """
    from transformers import AutoModel, AutoProcessor

    with refusing("patch"):
        axes.patch(config)
    processor = AutoProcessor.from_pretrained(config.model.hf_id, revision=config.model.revision)
    align_padding_side(processor, config.model.padding_side)
    with refusing("load_kwargs"):
        load_kwargs = axes.load_kwargs(config)
    model = AutoModel.from_pretrained(
        config.model.hf_id,
        revision=config.model.revision,
        dtype=steps.dtype_for(device),
        **load_kwargs,
    )
    model.to(device)

    dataset = load_pairs(config)
    with refusing("assemble"):
        if config.dataloader.pretokenize:
            # Before `assemble`, which is the whole of the axis: `_dataloader`
            # refuses `pretokenize=true` over rows that do not already carry token
            # ids, because building the loader anyway would leave the tokenisation
            # inside the timed step under a pretokenized label.
            dataset = axes.pretokenize(dataset, Encode(processor, config))
        built, applied = axes.assemble(model, config, device, framework="native", dataset=dataset)
    with refusing("step_context"):
        # The fifth call site, and the only one the measured loop enters. Called
        # once here and the result dropped, so a precision value with no recipe is
        # refused before the timer starts. Left to the loop it would raise on step
        # 0 *after* `timer.__enter__`, where the choice is between catching inside
        # the timed window and crashing on something that was knowable here. Safe
        # to call twice because it is a factory: it either raises or returns a
        # fresh context manager. A future recipe that made construction expensive
        # or stateful would have to move this to a cheaper precondition check.
        axes.step_context(config)
    # `assemble` has no collate argument, so the loader it builds carries either
    # torch's default one or `axes.PackedCollate`, and this is the only place to
    # replace it. What goes in declares `axis_packing`, which is what
    # `applied._capture_dataloader_packing` reads — an assignment that did not would
    # turn a determined axis into an undetermined one and `assert_matches` below
    # would refuse the run.
    built.dataloader.collate_fn = build_collate(processor, config)

    state = capture(built, config)
    # Directly, and before a single step runs. Not through `steps.verify_axes`,
    # which wraps it in `report.run(...)` and swallows the raise. The `refusing`
    # block does not swallow it either: it re-raises a tagged exception that
    # `main` writes to the result file and then exits non-zero on.
    with refusing("assert_matches", state):
        assert_matches(state, config)
    return built, applied, state


def device_arch(capability: tuple[int, int]) -> str:
    """`(8, 0)` -> `"80"`. The one place this project spells a capability as an arch.

    Not a convention chosen here. torch builds its own `-gencode` flags this way —
    `capability = torch.cuda.get_device_capability(i)` becomes
    `arch = f'{major}.{minor}'`, then `num = f"{major}{minor}"` and
    `-gencode=arch=compute_{num},code=sm_{num}` (`torch/utils/cpp_extension.py`,
    `_get_cuda_arch_flags`, torch 2.13.0). The arch lists in
    `docker/Dockerfile.framework` are in that same spelling, which is why
    transformer-engine's own default can contain `89`: Ada is capability 8.9, and
    no other reading of that number exists.

    Concatenation rather than `major * 10 + minor`, to stay identical to the line
    above rather than merely equal to it for the capabilities that exist today.
    """
    major, minor = capability
    return f"{major}{minor}"


def declared_archs(value: str | None) -> list[str]:
    """The image's arch list, as `Dockerfile.framework` writes it: `80;90;100`.

    A trailing letter is dropped — `90a` is architecture-specific SASS for
    capability 9.0, so it is that device's arch and not a fourth kind of number.
    Nothing in the current image uses one; the alternative is that such an entry
    silently matches no GPU at all.
    """
    if not value:
        return []
    archs = []
    for entry in value.replace(",", ";").split(";"):
        digits = "".join(ch for ch in entry.strip() if ch.isdigit())
        if digits:
            archs.append(digits)
    return archs


def current_gpu_arch() -> str | None:
    """The arch of the GPU this process would run on, or None if there is no GPU.

    Reads the current device rather than naming one: no device string is
    constructed here, so this is not a second device resolver beside
    `trainbench/device.py` (AGENTS.md).
    """
    if not torch.cuda.is_available():
        return None
    return device_arch(torch.cuda.get_device_capability())


def gpu_refusal(declared: str | None, arch: str | None) -> str | None:
    """Why this pod's GPU cannot run this image's kernels, or None if it can.

    **An image that declares nothing is refused.**
    `docker/Dockerfile.framework` sets `TRAINBENCH_CUDA_ARCHS` unconditionally,
    for every framework, in the same file that copies `docker/entrypoint.sh` — the
    only thing that calls this. So an image carrying this check and not the
    variable is not a state this repository can build; it is an image from before
    the narrowing, or one whose env was overridden, and in either case what its
    kernels cover is unknown. Passing on "nothing to compare against" is the shape
    this repository has shipped ten times, and here it would pass exactly the pods
    the check exists for. The cost of being wrong is one loud relaunch with the
    variable set; the cost the other way is a pod that dies after loading a model.

    A pod with no visible GPU is refused for the same reason and not the same one:
    the plan reached here only through the timing/profile/quality branch, so this
    pod was booted to measure on a GPU, and it has none.
    """
    archs = declared_archs(declared)
    if not archs:
        return (
            f"{CUDA_ARCHS_ENV} is not set, so this image does not say which GPUs its "
            "kernels were compiled for. Every image built by "
            "docker/Dockerfile.framework sets it; an image without it is older than "
            "that or had its environment overridden, and flash-attn ships no PTX to "
            "fall back on."
        )
    if arch is None:
        return (
            f"no CUDA device is visible, but this pod was launched to measure on one "
            f"(the image compiled kernels for sm_{'/sm_'.join(archs)})."
        )
    if arch not in archs:
        return (
            f"this GPU is sm_{arch} and the image compiled kernels for "
            f"sm_{'/sm_'.join(archs)} only. flash-attn emits code=sm_XX with no PTX, "
            "so the run would die with 'no kernel image is available for execution on "
            "the device' once the first kernel launched."
        )
    return None


def preflight(plan_path: Path, stream: Any = None) -> int:
    """Put every setting of this pod's plan through the refusals, before any of them run.

    The pod is the only place this question can be answered. Whether `axes.patch`
    accepts a setting depends on what the image contains — fla, causal-conv1d, a
    CUDA runtime — and the audit host has none of them, so the same check run on a
    laptop inverts: it rejects the `kernel=fla` baseline that every pod is about to
    run correctly and passes the `kernel=none` setting that dies on a Qwen3.5
    image. That measurement is why this is not a gate in `scripts/audit_plan.py`.

    `bench.py` already refuses a setting at `axes.patch`, so what this adds is
    *when*. A sweep learns about its second setting only after the first has
    finished, and a pod whose whole plan is unrunnable finds that out after it has
    booted a B200, pulled an image and downloaded a checkpoint. This costs seconds
    and it costs them before the model exists.

    Only the three call sites that need no model can run here — `patch`,
    `load_kwargs`, `step_context`. `assemble` and `assert_matches` are what
    `main` does per setting, and nothing here replaces them: a plan that passes
    preflight can still be refused for what the built model turns out to be.

    An empty plan is a refusal, and so is a plan with nothing composable in it. A
    pod that measures nothing is the failure this exists to catch, and reading zero
    settings as "none refused" would make the check quietest exactly where it has
    seen the least.

    A plan item carrying no resolved config is reported and *not* counted against
    the plan. It is a malformed plan rather than an axis this image cannot apply,
    `docker/entrypoint.sh` already stops that setting alone and publishes a record
    naming it, and taking the pod down over it here would silently overturn that —
    the rest of the axis is still worth the pod that was booted for it.

    The GPU is checked too, and before the settings, because it is a property of
    the pod rather than of any one of them (`gpu_refusal`). Both are reported even
    when the first has already decided the answer: one pod log that names the wrong
    GPU *and* the unrunnable setting is worth more than two relaunches.
    """
    # Resolved per call, not in the signature: a default argument binds the
    # `sys.stdout` that existed at import, which is not the one a caller replacing
    # it is reading.
    stream = sys.stdout if stream is None else stream
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"preflight: cannot read the plan at {plan_path}: {exc}", file=stream)
        return PREFLIGHT_EXIT
    if not isinstance(plan, list) or not plan:
        print(
            f"preflight: {plan_path} carries no settings; this pod would measure nothing",
            file=stream,
        )
        return PREFLIGHT_EXIT

    arch = current_gpu_arch()
    gpu = gpu_refusal(os.environ.get(CUDA_ARCHS_ENV), arch)
    if gpu is None:
        print(f"preflight: this pod's GPU is sm_{arch}, which the image covers OK", file=stream)

    refused, checked = [], 0
    for index, item in enumerate(plan):
        name = (isinstance(item, dict) and item.get("name")) or f"setting-{index}"
        resolved = item.get("config") if isinstance(item, dict) else None
        if not isinstance(resolved, dict) or not resolved:
            print(f"preflight: {name} carries no resolved config; not checked", file=stream)
            continue
        checked += 1
        try:
            config = to_bench_config(resolved)
            axes.patch(config)
            axes.load_kwargs(config)
            with axes.step_context(config):
                pass
        except Exception as exc:  # noqa: BLE001 - anything that stops a setting stops the pod
            refused.append(f"{name}: {type(exc).__name__}: {' '.join(str(exc).split())}")
            continue
        print(f"preflight: {name} OK", file=stream)
    if gpu is not None:
        print(f"preflight REFUSED this pod's GPU: {gpu}", file=stream)
    for line in refused:
        print(f"preflight REFUSED {line}", file=stream)
    if refused or gpu is not None:
        counted = f"{len(refused)} of the {checked} setting(s) it could compose"
        cause = counted if refused else "this pod's GPU"
        if refused and gpu is not None:
            cause = f"{counted}, and this pod's GPU,"
        print(f"preflight: {cause} cannot run in this image; nothing is measured", file=stream)
        return PREFLIGHT_EXIT
    if not checked:
        print(
            f"preflight: none of the {len(plan)} plan item(s) carried a config to check; "
            "this pod would measure nothing",
            file=stream,
        )
        return PREFLIGHT_EXIT
    print(f"preflight: all {checked} setting(s) can run", file=stream)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="resolved config JSON")
    parser.add_argument("--out", type=Path, help="where to write the result")
    parser.add_argument(
        "--preflight",
        type=Path,
        metavar="PLAN",
        help="check every setting of this plan and measure nothing",
    )
    args = parser.parse_args(argv)

    if args.preflight is not None:
        return preflight(args.preflight)
    if args.config is None or args.out is None:
        parser.error("--config and --out are required unless --preflight is given")

    config = load_bench_config(args.config)
    device = get_device(config.device)
    # Follows the config rather than forcing determinism the way the probe does.
    # Deterministic mode disables kernel autotuning, which is part of what a timing
    # run measures, and the schema already refuses `deterministic=true` for
    # `purpose=timing` — hardcoding it here would override that silently.
    set_seed(config.train.seed, deterministic=config.train.deterministic, warn_only=True)

    try:
        built, applied, state = build_run(config, device)
    except RefusedSetting as refused:
        # A result file, and a non-zero exit. The sweep in docker/entrypoint.sh
        # publishes whatever `--out` holds with `--mode result` and counts this
        # setting as failed, so the axis keeps running and the reason reaches the
        # report instead of dying in the pod log with the exit code.
        record = refusal_record(config, device, refused)
        write_json(args.out, record)
        print(record["status"], file=sys.stderr)
        print(f"wrote {args.out}")
        return REFUSED_EXIT

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
