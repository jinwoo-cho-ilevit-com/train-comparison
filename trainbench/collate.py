"""Rows of the pinned subset, turned into the micro-batches the timed step reads.

The `collate-metrics` boundary lives here: `MicroBatch` is what a collate returns
and what every counter in the result is derived from, and
`tests/contract/test_collate_metrics.py` looks for it in this module by name.

Nothing here touches the device or divides by a time. The counts are produced in
the DataLoader worker so that no reduction over the mask lands inside the timed
window, and a rate computed here would be a throughput this side never measured.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import torch

from trainbench import axes
from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.prompt import format_prompt

# MMEB stores its own placeholder markup inside `qry` / `pos_text` verbatim
# (`scripts/prepare_data.py`): `"<|image_1|>\nRepresent the given image.\n"`. It is
# MMEB's markup, not any model's, and this is the loader that converts — each model
# has different image tokens, which `trainbench/prompt.py` is what inserts. Leaving
# the marker in would feed one model literal text where another model's placeholder
# belongs.
MMEB_IMAGE_MARKER = re.compile(r"<\|image_\d+\|>")

# The pack's boundaries under the names `model(**tensors)` reads them by. All of
# these are `TransformersKwargs` members (`transformers/utils/generic.py:800-839`),
# and together they are the keys a collate may add to `tensors`
# (`tests/fixtures/microbatch.sample.json:tensors_may_add`). The two spellings in
# `axes.PACKED_BOUNDARY_KEYS` carry the same boundaries and the model does reject
# those, which is why they are lifted out and these put in.
#
# Two groups, because they gate two different kernels and are not all-or-nothing
# with each other: the varlen four reach attention, `seq_idx` reaches the causal
# conv of `arch=qwen3_5`'s linear-attention layers.
VARLEN_KWARGS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
SEQ_IDX_KWARGS = ("seq_idx",)


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
        # across models — METRIC_DEFINITIONS says so in the result. It is handed to
        # `format_prompt`, never prefixed to its output: see `_text`.
        self.prompt = config.model.instruction_prompt or ""
        # Whether this processor can take pixels at all, read off the processor
        # rather than branched on `model.arch`: a text-only checkpoint returns a
        # bare tokeniser from AutoProcessor. Rows still carry their images; they
        # are counted as dropped so the result says plainly that this model read a
        # text-only view of an image corpus.
        self.accepts_images = getattr(processor, "image_processor", None) is not None

    def _text(self, raw: str | None, with_image: bool, instruction: str = "") -> str:
        """One side of a pair, in this model's own prompt format.

        Both `add_generation_prompt` and `prompt_format` are the config's
        (docs/CONTRACTS.md §5) — with last-token pooling the first decides which
        token becomes the embedding, and the second decides whether there is a chat
        template to pass it to at all. Neither can be defaulted here.

        `instruction` goes through `format_prompt` rather than being concatenated
        onto its result, which is the whole of the fix for
        `qwen3-vl-query-prompt-may-go-in-twice`: the Qwen template inserts the same
        instruction itself when the row carries no system turn.
        """
        text = MMEB_IMAGE_MARKER.sub("", raw or "").strip()
        return format_prompt(
            self.processor,
            text,
            with_image=with_image,
            prompt_format=self.config.model.prompt_format,
            add_generation_prompt=self.config.model.add_generation_prompt,
            instruction_prompt=instruction,
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
            queries.append(self._text(row.get("qry"), side_images[0], self.prompt))
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
    what `model(**tensors)` takes — and put back under the names the model does
    read them by (`varlen_kwargs` for attention, `seq_idx_kwargs` for the causal
    conv), which is the only place a pack's boundaries reach either kernel.

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
        batch.update(varlen_kwargs(boundaries["cu_seqlens"], boundaries["seq_lengths"]))
        batch.update(seq_idx_kwargs(boundaries["seq_lengths"]))
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


def varlen_kwargs(cu_seqlens: torch.Tensor, seq_lengths: torch.Tensor) -> dict[str, Any]:
    """The pack's boundaries, in the four names transformers takes varlen on.

    `modeling_flash_attention_utils.py:761-763` is the gate:

        is_fa_with_varlen_kwargs = all(
            kwarg is not None
            for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
        )

    All four or none — a partial set leaves that `all(...)` false, the pack runs as
    one dense sequence, and the run still reports `dataloader.packing=True`. That is
    why they are built together here rather than assigned key by key at the call
    site, and why the boundary's fixture states the same rule as an invariant.

    **`arch=qwen3_5` is what this is for.** `qwen3_vl` and `gemma4` get their
    isolation from `position_ids` alone: `masking_utils.py:858-868` derives the
    packed-sequence mask from it whenever no 2D mask and no cache are passed, which
    a packed batch satisfies. Qwen3.5 alternates full_attention with
    linear_attention, and `Qwen3_5GatedDeltaNet.forward` never receives
    `position_ids` — it reads `kwargs.get("cu_seq_lens_q")`
    (`modeling_qwen3_5.py:538-550`) and scans the whole pack as a single sequence
    when that is None, silently. Emitted for every packed batch and not for that
    arch alone, because the boundaries are the same either way and an arch switch
    here would be a second place a pack's isolation could be lost by editing a
    config field that names no attention.

    The conv half of the same arch is `seq_idx_kwargs` below, on its own gate.

    `max_length_*` stay Python ints. `to_device` passes non-tensors through
    untouched, so a 0-dim tensor here would be a host-to-device round trip inside
    the timed window for a number that only bounds a kernel launch.
    """
    return {
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": int(seq_lengths.max()),
        "max_length_k": int(seq_lengths.max()),
    }


def seq_idx_kwargs(seq_lengths: torch.Tensor) -> dict[str, Any]:
    """Which sequence of the pack each token belongs to, for the causal conv.

    The varlen four never reach `arch=qwen3_5`'s conv. `Qwen3_5GatedDeltaNet` calls
    `causal_conv1d_fn(..., seq_idx=kwargs.get("seq_idx"))`
    (`modeling_qwen3_5.py:492-499`) and that argument alone is what stops the
    convolution's receptive field from running over a sequence boundary — a
    different kernel from the one `varlen_kwargs` gates, so the two are not
    all-or-nothing with each other and the fixture groups them apart.

    One int32 index per token, shaped like `input_ids`. That is the form the only
    pinned implementation of the semantics reads: `Lfm2ShortConv.slow_forward`
    takes `seq_idx[0]`, cuts at `si[1:] != si[:-1]` and convolves each segment on
    its own (`modeling_lfm2.py:383-396`).

    Emitted for every packed batch rather than for `arch=qwen3_5` alone, for the
    reason `varlen_kwargs` gives: an arch switch here is a second place a pack's
    isolation can be lost by editing a config field that names no attention.
    Measured 2026-08-03 on CPU, this checkout — `Qwen3_5TextModel`,
    `Qwen3VLTextModel` and `Gemma4TextModel` all accept it on a packed forward and
    return byte-identical hidden states with and without it. Identical is what a
    host without `causal-conv1d` can show: the fallback branch
    (`modeling_qwen3_5.py:500-501`) has no `seq_idx` argument at all, so whether
    the real kernel honours the boundaries is a pod question.
    """
    counts = seq_lengths.to(torch.long)
    return {
        "seq_idx": torch.repeat_interleave(
            torch.arange(counts.numel(), dtype=torch.int32), counts
        ).unsqueeze(0)
    }


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
