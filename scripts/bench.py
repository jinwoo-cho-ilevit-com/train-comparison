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
import json
import os
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import torch

from trainbench import axes, metrics
from trainbench.applied import AppliedMismatch, AppliedState, assert_matches, capture
from trainbench.collate import Encode, build_collate, load_pairs
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


def native_binding(config: BenchConfig, device: torch.device) -> Binding:
    """The transformers path, which is what every framework fell back to before.

    Kept byte-for-byte as it ran inline in `build_run` so that the six frameworks
    opening up cannot change what `framework=native` measures.
    """
    from transformers import AutoModel, AutoProcessor

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
    return Binding(framework="native", model=model, processor=processor)


def load_framework(config: BenchConfig, device: torch.device) -> Binding:
    """The model and processor `config.framework.name` asks for.

    `trainbench.loader` is the adapters lane's single entry point and does not
    exist yet; until it does, every framework is served by the native path, which
    is the state `_capture_framework` already reports honestly. The name that
    reaches `assemble` comes from the binding either way, so a run under an adapter
    is labelled by what built it rather than by this file's opinion.
    """
    try:
        loader = importlib.import_module("trainbench.loader")
    except ModuleNotFoundError:
        return native_binding(config, device)
    return loader.load(config, device)


def build_run(config: BenchConfig, device: torch.device) -> tuple[Any, list[str], AppliedState]:
    """Everything between the resolved config and the first measured step.

    Split out of `main` so that the refusals can be caught around a region that
    *stops* at `assert_matches`. The measured loop is deliberately outside it: an
    exception raised in there is a failure partway through a measurement, and
    filing it as a clean refusal would say a setting was declined when a loop had
    already run. `tests/test_smoke_cpu.py` pins that boundary.
    """
    with refusing("patch"):
        axes.patch(config)
    binding = load_framework(config, device)
    model, processor = binding.model, binding.processor

    dataset = load_pairs(config)
    with refusing("assemble"):
        if config.dataloader.pretokenize:
            # Before `assemble`, which is the whole of the axis: `_dataloader`
            # refuses `pretokenize=true` over rows that do not already carry token
            # ids, because building the loader anyway would leave the tokenisation
            # inside the timed step under a pretokenized label.
            dataset = axes.pretokenize(dataset, Encode(processor, config))
        built, applied = axes.assemble(
            model, config, device, framework=binding.framework, dataset=dataset
        )
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
