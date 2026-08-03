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
import importlib
import inspect
import json
import os
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import torch

from trainbench import axes, kernels, metrics
from trainbench.applied import (
    ENFORCED_PURPOSES,
    AppliedMismatch,
    AppliedState,
    assert_matches,
    capture,
)
from trainbench.collate import Encode, build_collate, load_pairs
from trainbench.config import load_bench_config, to_bench_config
from trainbench.config_schema import BenchConfig, axis_knobs
from trainbench.device import get_device
from trainbench.embedding import packed_last_token_pool, padding_preverified
from trainbench.probe import steps
from trainbench.probe.types import MAX_TRACEBACK_CHARS
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

# Exit code for a run the memory ceiling stopped partway through. Distinct from
# REFUSED_EXIT because the two are read differently in the pod log: that one is a
# setting declined before it measured anything, this one is a setting that was
# measured until the device ran out. Both write a result file.
OOM_EXIT = 5

# Exit code for a run a diagnosable failure stopped. Distinct from OOM_EXIT
# because that one is the device's limit and this one is a defect, and from 1,
# which leaves no result file at all — a pod died on a `RuntimeError` from
# `trainbench/collate.py` and the only trace anywhere was the entrypoint's
# "produced no result (exit 1)", which cost two launches to read a message that
# was already in the container log. Distinct from `timeout`'s 124 and from
# docker/entrypoint.sh's 125 and 127.
FAILED_EXIT = 6

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

# `status` prefix on a record for a run a failure stopped. Not `no_result` for
# `REFUSED_STATUS`'s reason, and not `REFUSED_STATUS` either: a refusal is a
# setting this pod declined to measure and this is a setting it tried to measure
# and could not, which are different findings in the report.
FAILED_STATUS = "run-failed"


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

    **`use_cache=False` is what makes the pack isolated.** `position_ids` alone do
    not: `masking_utils._preprocess_mask_arguments` calls
    `find_packed_sequence_indices` only when `past_key_values is None`, and every
    model here ships `config.use_cache=True`, so the default forward builds a cache
    and the mask comes back as one causal triangle over the whole pack — every
    sequence reading the ones before it, with no exception and no warning.
    Qwen3.5's linear-attention cache raises instead, so a packed run of that
    architecture never starts. There is no cache to keep in a training step, which
    is why this is a fix and not a trade.
    """
    if cu_seqlens is None:
        return steps.encode(model, tensors, padding_side)
    output = model(**tensors, output_hidden_states=False, use_cache=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(output, "hidden_states", None)
        hidden = hidden[-1] if hidden else output[0]
    return packed_last_token_pool(hidden, cu_seqlens)


def _tevatron_forward_loss(
    model: Any, tensors: dict[str, Any], config: BenchConfig, built: Any
) -> torch.Tensor:
    """tevatron's own training step, not `pooled_embeddings` + `built.loss_fn`.

    `DenseModel.forward(query=, passage=)` encodes, pools, normalises, scores and
    computes InfoNCE itself (pinned dd06310 retriever/modeling/encoder.py:52-87,
    dense.py:18-46), which is why `loss.name` is framework-owned for tevatron
    (`trainbench/loader.py`'s tevatron `Adapter`). Mirrors
    `trainbench/probe/tevatron.py::_backward`: the collate lays every query before
    every positive, so the batch's first half is the query side.
    """
    half = tensors["input_ids"].shape[0] // 2
    query = {key: value[:half] for key, value in tensors.items()}
    passage = {key: value[half:] for key, value in tensors.items()}
    output = model(query=query, passage=passage)
    if output.loss is None:
        raise RuntimeError(
            "tevatron's forward returned loss=None: an empty query or passage dict took "
            "its inference branch instead of training (dd06310 encoder.py:53-61)"
        )
    return output.loss


def _sentence_transformers_pooled(model: Any, tensors: dict[str, Any]) -> torch.Tensor:
    """sentence-transformers' forward takes one positional mapping, not `**tensors`.

    Pinned 5.6.1 `SentenceTransformer.forward(self, input: dict, **kwargs)` — the
    signature `refuse_a_forward_this_harness_cannot_call` refuses for every other
    adapter. Pooling happens inside the module
    (`trainbench/probe/sentence_transformers.py`), so this reads
    `sentence_embedding` off the result rather than calling `pooled_embeddings`.

    Returns pooled embeddings, not a loss: sentence_transformers does not own
    `loss.name` (docs/CONTRACTS.md §2, `applied.FRAMEWORK_OWNABLE`), so `train()`
    computes it from `built.loss_fn` itself — `scripts/audit_plan.py`'s
    `assert-called` reads that call literally, and a loss returned from here
    instead would be one hop too far for it to see.
    """
    return model(tensors)["sentence_embedding"]


StepRunner = Callable[[Any, dict[str, Any], BenchConfig, Any], torch.Tensor]
Pooler = Callable[[Any, dict[str, Any]], torch.Tensor]

# tevatron's forward computes pooling, scoring and the loss itself — the
# `owner=framework` half of `Step` (docs/CONTRACTS.md §2). Read by
# `refuse_a_step_this_harness_cannot_drive` and `train()`, and only reached when
# the binding's `step.owner` actually says so: a binding that merely names this
# framework without declaring the owned step still runs the generic loop, which
# is what lets `test_the_axes_an_adapter_owns_reach_the_record_instead_of_being_undetermined`
# (tests/test_smoke_cpu.py) isolate axis ownership from step ownership.
FRAMEWORK_OWNED_STEP_RUNNERS: dict[str, StepRunner] = {
    "tevatron": _tevatron_forward_loss,
}

# sentence-transformers still leaves the step to the harness (it does not own
# `loss.name`) but its forward will not bind as `model(**batch)` — a calling
# convention `Step` has no field for. Read by
# `refuse_a_forward_this_harness_cannot_call` and `train()` regardless of
# `step.owner`. Maps to a pooler, not a full step runner, so `train()` is the one
# place that calls `built.loss_fn` (see `_sentence_transformers_pooled`).
NONSTANDARD_FORWARD_POOLERS: dict[str, Pooler] = {
    "sentence_transformers": _sentence_transformers_pooled,
}


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


def train(
    built: Any,
    loader: Any,
    config: BenchConfig,
    device: torch.device,
    required_context: Any = None,
    adapter_step: Any = None,
) -> dict[str, Any]:
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

    Every step goes inside `axes.step_context(config, required_context)`. The fp8
    recipes wrap the forward pass, so a loop that applied them anywhere else would
    report an fp8 number for a step that never entered the recipe — and the capture
    probe would still find the swapped modules and call it a match.
    `required_context` is the framework's own regime (axolotl trains under
    `torch.autocast(bfloat16)`); it enters at the same site and never at the
    adapter's, so there is one place a precision context is established.

    **`optimizer.zero_grad` opens the step rather than closing it.** Each timed
    window still holds exactly one zero_grad, one accumulation and one optimizer
    step, so the window is unchanged in content; what changes is that the final
    step's gradients survive the loop. `metrics.gradient_norm` and
    `metrics.parameter_counts` are read there, outside the timer and before
    anything wipes them — read after a zero_grad they are a confident zero, which
    is the exact value they exist to distinguish, and read inside the timer they
    would put a float64 reduction over every parameter into the measurement.

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

    A third shape exists for a framework this generic call cannot drive at all:
    `adapter_step` (the binding's `AdapterOut.step`, `None` for every direct caller
    that predates it) says whether `built.framework` owns the step, and
    `FRAMEWORK_OWNED_STEP_RUNNERS`/`NONSTANDARD_FORWARD_POOLERS` name the two ways
    that happens — tevatron's own forward computes the loss, sentence-transformers'
    forward takes one positional mapping and pools inside its own module (its loss
    still comes from `built.loss_fn`, read directly below rather than through a
    step runner, so `scripts/audit_plan.py`'s `assert-called` can see it). Neither
    combines with GradCache or `dataloader.packing`, refused below before the loop
    opens.
    """
    if config.model.pooling != "lasttoken":
        raise RuntimeError(
            f"model.pooling={config.model.pooling} is declared but "
            "trainbench/probe/steps.py::encode pools the last token unconditionally; "
            "the knob would name one thing while the run measured another."
        )
    built.model.train()
    # The instrument the run declared, not this file's choice of clock. It refuses
    # rather than falls back: `cuda_event` off CUDA would report a wall-clock number
    # under a device-measurement label.
    timer = metrics.build_timer(device, config.measurement.instrument)
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
    # `adapter_step.owner == "framework"` only when the binding actually declared
    # it — `adapter_step=None` (every pre-existing caller) and a harness-owned step
    # both leave this False, so a run that merely names `built.framework` without
    # the declaration still takes the generic path below.
    framework_owns_step = False
    if adapter_step is not None:
        # Not `loader`: that name is this function's own DataLoader parameter
        # (see its docstring), and reassigning it here would only be harmless by
        # accident — `stream = micro_batches(loader)` above already captured the
        # real one, so nothing downstream reads the shadowed name back. A second
        # caller added later that reads `loader` after this line would not get
        # that accident twice.
        loader_module = importlib.import_module("trainbench.loader")
        framework_owns_step = adapter_step.owner != loader_module.HARNESS_STEP.owner
    step_runner = FRAMEWORK_OWNED_STEP_RUNNERS.get(built.framework) if framework_owns_step else None
    pooler = None if step_runner is not None else NONSTANDARD_FORWARD_POOLERS.get(built.framework)
    if step_runner is not None or pooler is not None:
        if gradcache_backward is not None:
            raise RuntimeError(
                f"loss=cached_mnrl with framework={built.framework}: its own forward computes "
                "pooling itself (and, for a framework-owned step, the loss too), so GradCache's "
                "row-wise re-encoding would target a call this framework never makes. Measure "
                "the two axes separately."
            )
        if config.dataloader.packing:
            raise RuntimeError(
                f"dataloader.packing=true with framework={built.framework}: its own forward "
                "expects the padded batch this collate builds (queries then positives, split "
                "in half); a packed batch is one row with no such split. Measure the two axes "
                "separately."
            )
    counted = dict.fromkeys(
        ("tokens", "padded_tokens", "rows", "samples", "images", "images_dropped"), 0
    )
    first_loss: torch.Tensor | None = None
    last_loss: torch.Tensor | None = None

    # True exactly when the loop below reaches `last_token_pool` at all: the
    # generic path and GradCache both pool the padded convention through
    # `pooled_embeddings`/`encode`; `step_runner`/`pooler` pool inside the
    # framework's own forward, and `dataloader.packing=true` has no
    # `attention_mask` to check in the first place (`packed_last_token_pool`
    # is a separate function with its own, still-synchronous checks — out of
    # scope here). Only in this case is there a per-call host-sync guard worth
    # skipping.
    #
    # Nothing here re-proves the mask conforms: `trainbench/collate.py`'s
    # `Collate`/`PretokenizedCollate` already called `assert_padding_conforms` on
    # every micro-batch's `attention_mask` while building it, on CPU, in the
    # DataLoader worker — before this batch ever reached `loader`. That is what
    # earns `padding_preverified()` here: not a separate pass over this run's
    # data, but the batch this loop is about to pool having already been through
    # the one place its mask was constructed.
    uses_last_token_pool = step_runner is None and pooler is None and not config.dataloader.packing
    padding_guard: Any = padding_preverified() if uses_last_token_pool else nullcontext()

    with padding_guard:
        for step in range(total):
            if step == discard:
                metrics.reset_peak_memory(device)
            measured = step >= discard
            with timer, axes.step_context(config, required_context):
                # Opens the step: see the docstring. The gradients this clears are the
                # previous step's, and the last step's are left standing for the
                # validity read below.
                built.optimizer.zero_grad(set_to_none=True)
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
                    if step_runner is not None:
                        # tevatron: its own forward computes pooling, scoring and
                        # the loss itself. See `_tevatron_forward_loss` above.
                        loss = step_runner(built.model, tensors, config, built)
                        (loss / grad_accum).backward()
                    elif pooler is not None:
                        # sentence_transformers: a nonstandard forward call, but the
                        # harness's own loss still certifies `loss.name` — read
                        # directly here, not through a step runner, for the reason
                        # `_sentence_transformers_pooled`'s docstring gives.
                        pooled = pooler(built.model, tensors)
                        half = pooled.shape[0] // 2
                        loss = built.loss_fn(pooled[:half], pooled[half:])
                        (loss / grad_accum).backward()
                    elif gradcache_backward is not None:
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

    # Outside the timer, before anything clears the gradients. Without these the
    # `record-report` validity gate is true of the fixture and of nothing else: a
    # run over a fully frozen graph produces a finite loss and a full set of step
    # times, and three cells of the 2026-08-02 campaign were published that way.
    validity = metrics.parameter_counts(built.model)
    validity["grad_norm"] = metrics.gradient_norm(built.model)

    durations = timer.durations
    kept = max(1, len(durations) - discard)
    summary = metrics.summarise(
        durations,
        discard=discard,
        rows_per_step=counted["rows"] / kept,
        tokens_per_step=counted["tokens"] / kept,
        # The other candidate denominator, named rather than left in
        # `extra_counts`: which of the two divides the step time reverses the
        # ranking of the `dataloader.packing` axis, and `summarise` refuses a
        # config whose declared denominator this run never counted.
        padded_tokens_per_step=counted["padded_tokens"] / kept,
        peak_bytes=metrics.peak_memory_bytes(device),
        extra_counts={
            name: counted[name] / kept for name in ("samples", "images", "images_dropped")
        },
        totals={
            "images_read_total": counted["images"],
            "images_dropped_total": counted["images_dropped"],
        },
        # Carries `measurement` and `profiled` into the summary from the run's own
        # declaration instead of leaving the harness's former constants implicit.
        config=config,
    )
    summary.update(validity)
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

    Wraps the refusals construction can end on. They are not the same finding,
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
    * `loader.AdapterRefusal` — the framework built something this loop must not
      measure: a fully frozen graph, two training dtypes with no declared regime,
      or a step this loop cannot drive. unsloth returning `trainable_params=0` is
      the state three cells of the 2026-08-02 campaign published, so it is a
      result of this study and has to reach the report as one.
    * `kernels.KernelProvenanceError` — the kernel that would run cannot be named,
      would still arrive over the network, or builds no isolation mask for a
      packed batch. Each makes the number untraceable or wrong rather than absent.

    The stage matters for the same reason: `patch` fires before the model exists,
    `binding` and `assemble` during construction, `assert_matches` after. Only the
    last had a model to read back, so the stage is what says whether `applied` in
    the record means anything. `fingerprint` is carried for the same reason the
    measured record carries it: a refusal that happened after the model was built
    can still say which kernel and which dtypes it was refusing.
    """

    def __init__(
        self,
        stage: str,
        cause: Exception,
        state: AppliedState | None = None,
        fingerprint: Any = None,
    ) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause
        self.state = state
        self.fingerprint = fingerprint


def refusal_types() -> tuple[type[BaseException], ...]:
    """Every exception class that means "this setting is a result, not a crash".

    `trainbench.loader` is resolved on use rather than imported at module scope,
    for the reason `load_framework` gives. Read as a function so a refusal type
    added there cannot be one this file forgets to catch — the alternative is a
    literal tuple that goes stale silently, which is how `AdapterRefusal` came to
    leave `main` uncaught and publish nothing at all.
    """
    loader = importlib.import_module("trainbench.loader")
    return (
        axes.UnappliedAxis,
        AppliedMismatch,
        loader.AdapterRefusal,
        kernels.KernelProvenanceError,
    )


@contextmanager
def refusing(
    stage: str, state: AppliedState | None = None, fingerprint: Any = None
) -> Iterator[None]:
    """Tag whichever refusal comes out of this region with the call site it fired at.

    Only the `refusal_types` are caught. Everything else — a checkpoint that will
    not download, an OOM, a collate that cannot find a pad id — passes through
    untouched and leaves no result file, which is what makes docker/entrypoint.sh
    publish a fallback record rather than a result for it. Widening this to
    `Exception` would turn every crash into a tidy record saying the axis could not
    be applied, which is a different and false claim.
    """
    try:
        yield
    except refusal_types() as exc:
        raise RefusedSetting(stage, exc, state, fingerprint) from exc


def fingerprint_payload(fingerprint: Any) -> dict[str, Any]:
    """The build fingerprint under the key the `kernel-provenance` boundary names.

    `loader.describe` computes it on every run — which kernel repo and revision
    the build actually bound, which dtypes the framework changed behind us, which
    parameters it left trainable — and until this reached `build_record` it died
    with the pod. Two pods binding different revisions of the same fa2 request
    produced indistinguishable result files, and `loader.fingerprint_diff`, whose
    whole subject is that difference, had no input to read.

    Empty when there is none: a refusal that fired before the model existed has
    nothing to fingerprint, and a `null` under this key would claim it looked.
    """
    return {} if fingerprint is None else {kernels.RUN_RECORD_KEY: fingerprint}


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
        **fingerprint_payload(refused.fingerprint),
    )


def failure_status(error: BaseException) -> dict[str, Any]:
    """The record fields that make a diagnosable failure readable as its own outcome.

    Shaped like `metrics.oom_status` and refusing for the same reason in reverse:
    an out-of-memory condition filed here would publish the device's ceiling as a
    defect, so the two cannot be confused in either direction.

    **No `metrics` key, ever**, for `refusal_record`'s reason: a record carrying one
    asserts that a measured window completed. What travels instead is the exception
    and the tail of its traceback, which is what puts the reason in
    `scripts/report.py`'s `지표 없음` list — that file prints `status` verbatim for a
    record with no metrics, so the message reaches a reader who has no container log.

    The traceback is kept from the end: the innermost frame and the message name the
    defect, while the outermost frames are the same for every run.
    """
    if metrics.is_oom(error):
        raise ValueError(
            f"{type(error).__name__} is an out-of-memory condition; metrics.oom_status files "
            "it as the ceiling it is, and filing it here would publish a hardware limit as a "
            "defect in this harness"
        )
    reason = " ".join(str(error).split())
    return {
        "status": f"{FAILED_STATUS} ({type(error).__name__}) — {reason}",
        "failure": {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-MAX_TRACEBACK_CHARS:],
        },
    }


class Binding(NamedTuple):
    """What built the model, and what it says about itself.

    The field names are `trainbench/loader.py`'s `AdapterOut` verbatim
    (`tests/contract/test_loader_bench.py` fixes the eight), so the adapters lane's
    object substitutes for this one without another edit here. Only the first three
    are answerable without an adapter; the rest belong to the framework that built
    the model and are left None rather than guessed at.

    A NamedTuple and not a dataclass: this file is loaded by path with no
    `sys.modules` entry (`docker/entrypoint.sh`'s preflight stand-in does exactly
    that), and `@dataclass` looks its own module up in `sys.modules` and dies on
    the None it gets back.
    """

    framework: str
    model: Any
    processor: Any
    step: Any = None
    owned_axes: dict[str, str] | None = None
    required_step_context: Any = None
    fingerprint: Any = None
    documented_entry_point: Any = None


def load_framework(config: BenchConfig, device: torch.device) -> Binding:
    """The model and processor `config.framework.name` asks for.

    `trainbench.loader` is the single entry point every framework is served
    through, and it is imported by name rather than at module scope so that this
    file stays loadable by path (docker/entrypoint.sh's preflight stand-in).

    There is no transformers fallback here any more. There used to be one, from
    when `loader.py` did not exist; keeping it left two definitions of the native
    build — `trainbench/probe/native.py::load` is the other and the maintained one
    — and turned a `loader.py` missing from an image into a quiet change of which
    code path produced the number, rather than the packaging failure it is.

    The name that reaches `assemble` comes from the binding, so a run under an
    adapter is labelled by what built it rather than by this file's opinion.
    """
    loader = importlib.import_module("trainbench.loader")
    return loader.load(config, device)


def close_kernel_fetch_doors(config: BenchConfig, stream: Any = None) -> list[str]:
    """Shut every path a kernel could still arrive on, and name the ones that were open.

    Called before the model exists, because that is when transformers rewrites a
    `flash_attention_2` request into a Hub repo id and downloads it
    (`modeling_utils.py:1997-2003`, gated on `is_kernels_available()` alone). A
    kernel that arrives mid-run is not the kernel the image digest pins, and the
    fetch itself lands inside the run the same way reading data off a network
    volume does.

    Only for the purposes whose axes are enforced. A probe exists to find out
    whether a combination loads at all and is the branch that populates the cache
    the measured run then reads; closing the door there would turn "this framework
    cannot build this model" into "this pod had no cache", which is a different
    finding. `open_fetch_doors` says the same in its own words: an empty list is
    what a timing run has to start in.
    """
    if config.run.purpose not in ENFORCED_PURPOSES:
        return []
    # Resolved per call, not in the signature, for the reason `preflight` gives.
    stream = sys.stdout if stream is None else stream
    was_open = kernels.forbid_runtime_kernel_fetch()
    for door in was_open:
        print(f"kernel fetch door closed: {door}", file=stream)
    return was_open


def prefetch_checkpoint(config: BenchConfig) -> str:
    """Materialise the checkpoint in the local cache and return the sha it resolved to.

    `from_pretrained` downloads and builds in one call, so the doors cannot be shut
    between those two — and they have to be, because the `flash_attention_2` ->
    Hub-repo rewrite fires while the model is being built. Pulling the snapshot
    first is what makes both true: the weights arrive over an open network, and the
    build that follows reads them off disk with the network shut.

    The returned identity is the resolved commit, not the ref that was asked for.
    `configs/model/*.yaml` all pin `revision: null`, so what a run loaded is
    otherwise recoverable only from whatever `main` happened to point at that day.
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(config.model.hf_id, revision=config.model.revision))
    return f"{config.model.hf_id}@{snapshot.name}"


def fetch_the_pinned_inputs(config: BenchConfig, stream: Any = None) -> tuple[Any, dict[str, str]]:
    """The corpus and the checkpoint, fetched while the network is still open.

    Both used to be read *after* `close_kernel_fetch_doors`, and offline mode has a
    different answer for each of them. `huggingface_hub` refuses: a cold pod died
    with `OSError("We couldn't connect to ...")` inside `from_pretrained` and
    `ConnectionError(OfflineModeIsEnabled)` inside `load_pairs`, neither of them a
    `refusal_types()`, so every timing and quality setting exited 1 with no result
    file and `docker/entrypoint.sh` filed the pod as having produced nothing.
    `datasets` does something worse: an offline `load_dataset(..., revision=...)`
    answers with whatever build is already cached and only logs it — measured on
    this host, that was the D1-corrupt subset `CORRUPT_DATA_REVISIONS` exists to
    refuse — while `build_record` still writes the pinned revision beside the
    number. The corpus is therefore read here rather than merely warmed: a
    substitution cannot happen on a request that was answered before the door shut.

    The order this establishes is the whole of the fix — pinned inputs, then the
    doors, then the model — and `tests/test_smoke_cpu.py` asserts both ends of it,
    because a later round finding the other end reopened is how this file got here.
    """
    stream = sys.stdout if stream is None else stream
    dataset = load_pairs(config)
    identities = {"corpus": f"{config.data.repo_id}@{config.data.revision}"}
    if config.run.purpose in ENFORCED_PURPOSES:
        # Same gate as `close_kernel_fetch_doors`: a probe leaves the doors open, so
        # it has nothing to get ahead of and stays on the plain `from_pretrained`
        # path that populates the cache this branch later reads.
        identities["checkpoint"] = prefetch_checkpoint(config)
    for name, identity in sorted(identities.items()):
        print(f"pinned input fetched before the doors closed: {name}={identity}", file=stream)
    return dataset, identities


def refuse_a_forward_this_harness_cannot_call(step: Any, model: Any, framework: str) -> None:
    """Refuse a model whose `forward` will not take the batch as keyword arguments.

    `train()` calls `model(**batch, ...)`. `Step` says who owns the step and which
    keys the batch needs; it has no way to say that a framework's forward takes the
    batch as one positional mapping, so an adapter can declare the harness step
    truthfully by both of those fields and still be undrivable. sentence-transformers
    is that case — pinned 5.6.1 `SentenceTransformer.forward(self, input: dict,
    **kwargs)` — and it died with `TypeError` on step 0, after the checkpoint was
    loaded and the timer was open, where nothing catches it and no result file is
    written. The ST cell then rendered as a combination nobody tried rather than one
    that was refused.

    Skipped for a `framework` in `FRAMEWORK_OWNED_STEP_RUNNERS` or
    `NONSTANDARD_FORWARD_POOLERS`: that entry calls the model its own way
    (`query=`/`passage=` for tevatron, one positional mapping for
    sentence_transformers) and is what `train()` actually drives it through, so
    this generic `model(**batch)` check does not apply to it.

    Read off the live object with `inspect`, not from a declaration: the declaration
    is what was wrong. A `*args`/`**kwargs` forward binds anything and is left alone,
    so this refuses only what would actually raise. Only the declared `batch_keys`
    are bound — the collate adds more (`pixel_values`, `cu_seq_lens_*`) and which of
    those a model takes is per checkpoint, not per harness.
    """
    if framework in FRAMEWORK_OWNED_STEP_RUNNERS or framework in NONSTANDARD_FORWARD_POOLERS:
        return
    loader = importlib.import_module("trainbench.loader")
    forward = getattr(model, "forward", None)
    if forward is None:
        return
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        # A builtin or a C-level callable exposes no signature. Refusing here would
        # decline a model on the grounds that this check cannot read it.
        return
    try:
        signature.bind(**dict.fromkeys(step.batch_keys))
    except TypeError as exc:
        raise loader.AdapterRefusal(
            f"{type(model).__name__}.forward{signature} cannot be called as "
            f"model(**batch) with the declared batch keys {sorted(step.batch_keys)}: {exc}. "
            "The harness loop drives exactly that call, so this build would raise on the "
            "first micro-batch with the checkpoint loaded and the timer already open, and "
            "the setting would be published as a pod that produced nothing"
        ) from exc


def refuse_a_step_this_harness_cannot_drive(step: Any, framework: str) -> None:
    """Refuse an adapter whose declared step is not one `train()` can run.

    `train()` runs the generic step — this file's forward through
    `pooled_embeddings`, `built.loss_fn` over the two halves, and
    `built.optimizer.step()`, fed the keys `build_collate` builds — for any
    `owner=framework` step whose `framework` is not in
    `FRAMEWORK_OWNED_STEP_RUNNERS`, and that entry's own runner for one that is.
    `AdapterOut` declares what the framework's own step is, and nothing used to
    read that declaration.

    Both halves of it decide something. `owner=framework` means the framework's
    call computes the step, and `applied._owned` has already exempted the axes it
    owns from the capture — so running the generic loop anyway on an
    unregistered framework would publish a harness number labelled
    `framework_owned`, which is the `loss=cached_mnrl` shape: a mislabelled number
    rather than a crash. A registered `framework` is exempted: its runner is what
    calls the model, and `unbuilt` below has nothing to check because that runner
    never goes through `build_collate`'s `harness.batch_keys` at all — tevatron's
    runner derives `query`/`passage` from the batch itself
    (`_tevatron_forward_loss`).
    """
    loader = importlib.import_module("trainbench.loader")
    harness = loader.HARNESS_STEP
    if step is None:
        raise loader.AdapterRefusal(
            "the binding declares no step, so nothing says whether this harness loop or the "
            "framework runs it; measuring it either way would be a guess"
        )
    if step.owner != harness.owner:
        if framework in FRAMEWORK_OWNED_STEP_RUNNERS:
            return
        raise loader.AdapterRefusal(
            f"the adapter declares a {step.owner}-owned step ({step.callable}) for "
            f"{framework!r}, and no FRAMEWORK_OWNED_STEP_RUNNERS entry drives it. Running the "
            f"harness's {harness.owner}-owned loop anyway would time this file's forward and "
            "loss and file the result under the axes the framework was exempted from "
            "certifying"
        )
    unbuilt = sorted(set(step.batch_keys) - set(harness.batch_keys))
    if unbuilt:
        raise loader.AdapterRefusal(
            f"the adapter's step needs batch keys {unbuilt} and trainbench/collate.py builds "
            f"{sorted(harness.batch_keys)}; the forward would raise on the first micro-batch, "
            "with the checkpoint loaded and the timer already open"
        )


def refuse_packing_the_mask_registry_cannot_isolate(binding: Binding, config: BenchConfig) -> None:
    """Refuse `dataloader.packing` on a build whose implementation makes no mask.

    The fingerprint says whether the resolved implementation is in
    `AttentionMaskInterface`, and `kernels.assert_packing_is_isolated` is what
    turns that into a refusal — it had no caller, so the one place holding both
    the fingerprint and `config.dataloader.packing` never asked. An unregistered
    implementation makes transformers skip mask creation entirely and pass
    `attention_mask=None` down, with no exception and no warning, and the pack's
    sequences become each other's context while the record certifies
    `dataloader.packing=True` beside an ordinary throughput number.
    """
    if not config.dataloader.packing:
        return
    kernels.assert_packing_is_isolated(binding.fingerprint[kernels.BUILD_FINGERPRINT_KEY])


def build_run(
    config: BenchConfig, device: torch.device
) -> tuple[Any, list[str], AppliedState, Binding]:
    """Everything between the resolved config and the first measured step.

    The binding comes back out because the measured loop needs one thing off it
    that `Built` cannot carry: `required_step_context` is what the *framework*
    demands of the step, not what this harness constructed, and `axes.step_context`
    has to be handed it on every step.

    Split out of `main` so that the refusals can be caught around a region that
    *stops* at `assert_matches`. The measured loop is deliberately outside it: an
    exception raised in there is a failure partway through a measurement, and
    filing it as a clean refusal would say a setting was declined when a loop had
    already run. `tests/test_smoke_cpu.py` pins that boundary.
    """
    dataset, _ = fetch_the_pinned_inputs(config)
    close_kernel_fetch_doors(config)
    with refusing("patch"):
        axes.patch(config)
    with refusing("load_kwargs"):
        # `native_binding` tags its own `axes.load_kwargs` call, but an adapter's
        # does not go through it — `trainbench.loader` calls `axes.load_kwargs`
        # itself, outside that block. Without this an `UnappliedAxis` a refused
        # load-time axis raises would leave `main`'s broad `except` instead of
        # `refusal_record`, and the setting would produce no result file at all.
        binding = load_framework(config, device)
    with refusing("binding", fingerprint=binding.fingerprint):
        # Everything that can only be asked once the model exists, and all of it
        # before a timer opens. The fetch doors are re-read rather than merely
        # closed above: building the model imports whatever the framework needs,
        # and a module that arrives after the close carries its own default —
        # closing and never looking again is the shape of check this repository
        # keeps shipping.
        if config.run.purpose in ENFORCED_PURPOSES:
            kernels.assert_no_runtime_kernel_fetch()
        refuse_a_step_this_harness_cannot_drive(binding.step, binding.framework)
        refuse_a_forward_this_harness_cannot_call(binding.step, binding.model, binding.framework)
        refuse_packing_the_mask_registry_cannot_isolate(binding, config)
    model, processor = binding.model, binding.processor

    with refusing("assemble", fingerprint=binding.fingerprint):
        if config.dataloader.pretokenize:
            # Before `assemble`, which is the whole of the axis: `_dataloader`
            # refuses `pretokenize=true` over rows that do not already carry token
            # ids, because building the loader anyway would leave the tokenisation
            # inside the timed step under a pretokenized label.
            dataset = axes.pretokenize(dataset, Encode(processor, config))
        built, applied = axes.assemble(
            model,
            config,
            device,
            framework=binding.framework,
            dataset=dataset,
            # The adapter's declaration, not the config's request: `applied._owned`
            # exempts these axes from the capture, and letting `framework=tevatron`
            # grant that exemption would let a request excuse the cell least likely
            # to have applied the axis.
            owned_axes=binding.owned_axes or {},
        )
    with refusing("step_context", fingerprint=binding.fingerprint):
        # The fifth call site, and the only one the measured loop enters. Called
        # once here and the result dropped, so a precision value with no recipe is
        # refused before the timer starts. Left to the loop it would raise on step
        # 0 *after* `timer.__enter__`, where the choice is between catching inside
        # the timed window and crashing on something that was knowable here. Safe
        # to call twice because it is a factory: it either raises or returns a
        # fresh context manager. A future recipe that made construction expensive
        # or stateful would have to move this to a cheaper precondition check.
        axes.step_context(config, binding.required_step_context)
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
    with refusing("assert_matches", state, fingerprint=binding.fingerprint):
        assert_matches(state, config)
    return built, applied, state, binding


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

    An axis refusal only stands the pod down for a purpose `applied` enforces.
    `probe` is not one: "does this framework take this axis" is the question a
    probe pod was launched to answer, and refusing to start it turns a deliberate
    answer into `결과 없음(기동됨)` — indistinguishable from a pod that booted and
    died. A config the image's own schema cannot parse is still a refusal for
    every purpose, because that setting cannot be answered at all.

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

    refused, declined, checked = [], [], 0
    for index, item in enumerate(plan):
        name = (isinstance(item, dict) and item.get("name")) or f"setting-{index}"
        resolved = item.get("config") if isinstance(item, dict) else None
        if not isinstance(resolved, dict) or not resolved:
            print(f"preflight: {name} carries no resolved config; not checked", file=stream)
            continue
        checked += 1
        try:
            config = to_bench_config(resolved)
        except Exception as exc:  # noqa: BLE001 - the image's schema rejected the config
            refused.append(f"{name}: {type(exc).__name__}: {' '.join(str(exc).split())}")
            continue
        described = f"{name}: {{}}"
        try:
            axes.patch(config)
            axes.load_kwargs(config)
            with axes.step_context(config):
                pass
        except refusal_types() as exc:
            line = described.format(f"{type(exc).__name__}: {' '.join(str(exc).split())}")
            target = declined if config.run.purpose not in ENFORCED_PURPOSES else refused
            target.append(line)
            continue
        except Exception as exc:  # noqa: BLE001 - anything else that stops a setting stops the pod
            refused.append(described.format(f"{type(exc).__name__}: {' '.join(str(exc).split())}"))
            continue
        print(f"preflight: {name} OK", file=stream)
    if gpu is not None:
        print(f"preflight REFUSED this pod's GPU: {gpu}", file=stream)
    for line in declined:
        print(f"preflight: {line} — declined, and the run is what files that", file=stream)
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
    if declined:
        print(
            f"preflight: {len(declined)} of the {checked} setting(s) decline an axis this "
            "image cannot apply, and the run publishes that as its answer",
            file=stream,
        )
        return 0
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
        built, applied, state, binding = build_run(config, device)
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
    required = binding.required_step_context
    try:
        if config.run.profiler:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                if device.type == "cuda"
                else [torch.profiler.ProfilerActivity.CPU],
            ) as profile:
                summary = train(built, built.dataloader, config, device, required, binding.step)
            trace = args.out.with_suffix(".trace.json")
            profile.export_chrome_trace(str(trace))
            summary["trace_path"] = str(trace)
            print(f"wrote {trace}")
        else:
            summary = train(built, built.dataloader, config, device, required, binding.step)
    except (KeyboardInterrupt, SystemExit):
        # Not a property of the setting: the pod was stopped from outside, and a
        # record would publish an operator's action as this combination's result.
        raise
    except BaseException as exc:  # noqa: BLE001 - every outcome here is a result file
        # A result file either way, and no `metrics` block in either. The memory
        # ceiling is an answer to this study's question — this combination does not
        # fit at this batch size on this device. A `RuntimeError` from the collate
        # or a framework is not an answer, but it is diagnosable, and a record
        # carrying its traceback is the difference between a named defect and a
        # pod-hour that reads as a combination nobody attempted. Without one, both
        # fall through report.py's cell logic identically.
        if metrics.is_oom(exc):
            outcome, code = (
                metrics.oom_status(exc, peak_bytes=metrics.peak_memory_bytes(device)),
                OOM_EXIT,
            )
        else:
            outcome, code = failure_status(exc), FAILED_EXIT
        record = build_record(
            config,
            device,
            applied=state,
            applied_axes=applied,
            **outcome,
            **fingerprint_payload(binding.fingerprint),
        )
        write_json(args.out, record)
        print(record["status"], file=sys.stderr)
        print(f"wrote {args.out}")
        return code
    record = build_record(
        config,
        device,
        applied=state,
        metrics=summary,
        applied_axes=applied,
        # What the build turned out to be, beside what it was asked to be. Without
        # it a result file cannot say which kernel revision produced the number.
        **fingerprint_payload(binding.fingerprint),
    )
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
