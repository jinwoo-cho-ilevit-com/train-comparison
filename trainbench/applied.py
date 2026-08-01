"""Requested optimisation vs what the model actually ended up running.

The failure this exists to prevent has already happened once in this project: with
`fla`/`causal-conv1d` absent, transformers silently falls back to a slow torch
implementation for Qwen3.5's Gated DeltaNet layers and says so in a single log
line — no exception, no return value (docs/support-matrix.md). The same shape of
failure applies to every axis: FlashAttention falls back to sdpa when unbuilt,
torch.compile falls back to eager on a graph break, fp8 recipes no-op on
unsupported hardware. Each produces a plausible number under a wrong label, and
the result JSON alone cannot tell them apart afterwards.

Axes that have no capture probe yet report `applied=None` (undetermined), and an
undetermined axis fails a timing run exactly like a mismatched one. That is the
fail-safe direction: an axis is only ever certified once someone writes a probe
that actually looks.

The set of axes is derived from the schema, not listed here. A hand-written list
fails open — an axis missing from it is not "undetermined", it does not exist,
and deleting a line would disable a check with nothing to notice. Marking a field
`axis=True` in trainbench/config_schema.py is what puts it under this guarantee.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_args

from trainbench.config_schema import BenchConfig, RunConfig, axis_knobs

# Purposes whose numbers get reported. A probe or profile run may proceed with
# undetermined axes; a timing run may not.
ENFORCED_PURPOSES = ("timing", "quality")
# Taken from the schema so a purpose can never be added without deciding whether
# it is enforced.
KNOWN_PURPOSES = get_args(RunConfig.model_fields["purpose"].annotation)


@dataclass(frozen=True)
class AxisState:
    """One axis: what was asked for, and what is actually in effect."""

    axis: str
    requested: str
    applied: str | None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def determined(self) -> bool:
        return self.applied is not None

    @property
    def matches(self) -> bool:
        return self.determined and self.applied == self.requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "requested": self.requested,
            "applied": self.applied,
            "determined": self.determined,
            "matches": self.matches,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AppliedState:
    axes: tuple[AxisState, ...]

    def mismatched(self) -> list[AxisState]:
        return [a for a in self.axes if a.determined and not a.matches]

    def undetermined(self) -> list[AxisState]:
        return [a for a in self.axes if not a.determined]

    def missing(self) -> list[str]:
        """Axis knobs the schema declares that this state says nothing about."""
        return sorted(set(axis_knobs()) - {a.axis for a in self.axes})

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": [a.to_dict() for a in self.axes],
            "all_determined": not self.undetermined(),
            "all_matched": not self.mismatched(),
            "missing": self.missing(),
        }


class AppliedMismatch(RuntimeError):
    """A reportable run was about to measure something other than what it claims."""


@dataclass(frozen=True)
class Built:
    """What a run actually constructed.

    Half the axes are not properties of the model: the optimizer decides
    `optim.name` and `train.offload`, the dataloader decides `dataloader.*`, the
    loss decides `loss.name` and `parallel.cross_device_negatives`. A capture that
    only ever sees the model can never verify those, so it would certify a run on
    the strength of the half it can reach.

    A field left None means the run did not build that piece. Its axes come back
    undetermined, which blocks a reportable run — never "fine by absence".
    """

    model: Any = None
    optimizer: Any = None
    dataloader: Any = None
    loss_fn: Any = None
    # Which adapter actually ran. Set by that adapter with a literal, never copied
    # from the config: the config is the request, and telling the two apart is the
    # entire job of this module.
    framework: str | None = None


# --- per-axis capture probes -------------------------------------------------
# A probe returns the applied value, or None when it cannot determine it. Adding
# an axis here is what makes that axis measurable; until then it blocks timing
# runs by design. Keys are the dotted knob names from the schema.

CaptureFn = Callable[["Built", BenchConfig], tuple[str | None, dict[str, Any]]]


def _attn_per_module(model: Any) -> dict[str, str]:
    """Every place transformers records an attention implementation.

    Reading only `model.config._attn_implementation` is not enough. In
    `set_attn_implementation`, transformers updates the top-level config first and
    then walks submodules; any submodule that does not follow the AttentionInterface
    approach gets a log warning and keeps its previous implementation, and a
    submodule that cannot support the request has it downgraded individually. Vision
    towers are the common case. All three models here are multimodal, so a top-level
    'flash_attention_2' over an sdpa vision tower is the realistic outcome, and it
    would read as a clean match.
    """
    found: dict[str, str] = {}
    root = getattr(model, "config", None)
    top = getattr(root, "_attn_implementation", None)
    if top is not None:
        found["model"] = str(top)

    # Any submodule carrying its own config, rather than PreTrainedModel
    # instances specifically: what matters is that something in the tree records
    # a second answer, not what class recorded it.
    for name, module in getattr(model, "named_modules", lambda: [])():
        if module is model:
            continue
        impl = getattr(getattr(module, "config", None), "_attn_implementation", None)
        if impl is not None:
            found[name or type(module).__name__] = str(impl)

    for key in getattr(root, "sub_configs", ()) or ():
        sub = getattr(root, key, None)
        impl = getattr(sub, "_attn_implementation", None)
        if impl is not None:
            found.setdefault(f"config.{key}", str(impl))
    return found


def _capture_attn(built: Built, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    per_module = _attn_per_module(built.model)
    if not per_module:
        return None, {"reason": "no _attn_implementation found on the model or its subconfigs"}
    # Counts rather than the whole map: a real VLM records this in one place per
    # layer, so the map runs to hundreds of entries on a 5B model and would be
    # carried by every result file. What matters is how many answers there are
    # and which modules gave the minority one.
    counts: dict[str, int] = {}
    for impl in per_module.values():
        counts[impl] = counts.get(impl, 0) + 1
    detail = {"implementations": counts, "modules_checked": len(per_module)}
    if len(counts) == 1:
        return next(iter(counts)), detail
    top = per_module.get("model")
    dissenting = sorted(name for name, impl in per_module.items() if impl != top)
    # Deliberately not equal to any requested value, so a partial application is a
    # mismatch rather than a match on whatever the majority happened to be.
    return "mixed(" + ",".join(sorted(counts)) + ")", {**detail, "dissenting": dissenting[:8]}


def _capture_freeze_ple(built: Built, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """Whether the per-layer embeddings are actually frozen.

    Counting matches is half the check. `_ple_report` used to report ok=True on a
    zero match, so a renamed parameter upstream would have read as a successful
    freeze of nothing — 2.39B parameters, 46.8% of gemma-4-E2B, quietly still
    training. Zero matches on gemma4 is undetermined, not False.
    """
    # Imported here rather than at module scope: axes.py imports Built from this
    # module, and the two would otherwise form a cycle.
    from trainbench import axes

    if built.model is None:
        return None, {"reason": "no model was built"}
    params = axes.ple_parameters(built.model)
    if not params and config.model.arch == "gemma4":
        return None, {
            "reason": f"no parameter name contains {axes.PLE_PARAM_MARKER!r} on a gemma4 model",
            "matched": 0,
        }
    frozen = [n for n, p in params if not p.requires_grad]
    detail = {"matched": len(params), "frozen": len(frozen)}
    if not params:
        # No per-layer embeddings exist, so none are frozen. True by absence, and
        # the schema already refuses freeze.ple on a non-gemma4 model.
        return "False", detail
    if len(frozen) == len(params):
        return "True", detail
    if not frozen:
        return "False", detail
    unfrozen = sorted({n for n, _ in params} - set(frozen))
    return "partial", {**detail, "unfrozen_sample": unfrozen[:5]}


def _capture_optim(built: Built, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """Which optimizer a run is actually stepping with.

    The fused AdamW kernel is CUDA-only, so an unfused AdamW is reported as a
    different applied value: `adamw_fused` is the name of a kernel, and a run that
    did not use it must not carry that label.
    """
    if built.optimizer is None:
        return None, {"reason": "no optimizer was built"}
    kind = type(built.optimizer).__name__
    fused = any(group.get("fused") for group in getattr(built.optimizer, "param_groups", []))
    detail = {
        "class": kind,
        "fused": bool(fused),
        "param_groups": len(built.optimizer.param_groups),
    }
    if kind != "AdamW":
        return kind.lower(), detail
    return ("adamw_fused" if fused else "adamw_unfused"), detail


def _capture_loss(built: Built, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """Which loss a run actually computes.

    Read off an attribute the builder sets rather than inferred from the config,
    which is the request. GradCache is the case that matters: it must not be
    possible to report its speedup for a run that computed plain in-batch
    negatives.
    """
    if built.loss_fn is None:
        return None, {"reason": "no loss function was built"}
    value = getattr(built.loss_fn, "axis_value", None)
    if value is None:
        return None, {"reason": f"{type(built.loss_fn).__name__} declares no axis_value"}
    return str(value), {"callable": getattr(built.loss_fn, "__name__", repr(built.loss_fn))}


def _capture_framework(built: Built, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """Which adapter produced the run.

    A registry that routed `framework=unsloth` to the native path would otherwise
    publish native numbers in the unsloth row, and nothing in the result would
    say so.
    """
    if built.framework is None:
        return None, {"reason": "no adapter declared itself"}
    return str(built.framework), {}


def _capture_freeze_vision_tower(
    built: Built, config: BenchConfig
) -> tuple[str | None, dict[str, Any]]:
    """Whether the vision tower is actually frozen.

    Zero matches is undetermined rather than False for the reason the PLE probe
    gives: every model in this study has a vision tower, so a marker matching
    nothing means the marker is wrong, and "nothing is frozen" would be the same
    answer a correct marker gives for an unfrozen tower.
    """
    from trainbench import axes

    if built.model is None:
        return None, {"reason": "no model was built"}
    arch = config.model.arch
    try:
        params = axes.vision_parameters(built.model, arch)
    except axes.UnappliedAxis as exc:
        return None, {"reason": str(exc)}
    if not params:
        return None, {
            "reason": f"no parameter name matches {list(axes.VISION_PARAM_MARKERS[arch])} "
            f"on an arch={arch} model, so the marker no longer fits this checkpoint",
            "matched": 0,
        }
    frozen = [name for name, param in params if not param.requires_grad]
    detail = {"matched": len(params), "frozen": len(frozen), "arch": arch}
    if len(frozen) == len(params):
        return "True", detail
    if not frozen:
        return "False", detail
    unfrozen = sorted({name for name, _ in params} - set(frozen))
    return "partial", {**detail, "unfrozen_sample": unfrozen[:5]}


def _module_roots(model: Any) -> dict[str, int]:
    """Which package each module's class was defined in, counted.

    The class is what a kernel library replaces. Liger's documented sequence
    patches the transformers module and then instantiates, so its modules carry
    `liger_kernel.*` as their defining package; a model built without the patch
    carries only `torch.*` and `transformers.*`.
    """
    roots: dict[str, int] = {}
    for _, module in model.named_modules():
        root = type(module).__module__.split(".")[0]
        roots[root] = roots.get(root, 0) + 1
    return roots


def _capture_kernel(built: Built, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """Which kernel library, if any, is inside the model that was built.

    A negative answer here is load-bearing rather than trivial: the framework
    adapters patch transformers themselves — that is what Unsloth is — so
    `kernel=none` is a claim about the whole built model, not only about whether
    `patch()` did something.
    """
    from trainbench import axes

    if built.model is None:
        return None, {"reason": "no model was built"}
    roots = _module_roots(built.model)
    found = sorted({axes.KERNEL_MODULE_ROOTS[r] for r in roots if r in axes.KERNEL_MODULE_ROOTS})
    # Packages that are neither torch nor a kernel we name. Reported rather than
    # judged: a framework wrapper is not a kernel axis value, but a reader of the
    # result should be able to see that something else defined these modules.
    foreign = sorted(r for r in roots if r not in ("torch", "transformers", "builtins"))
    detail = {"modules_checked": sum(roots.values()), "packages": foreign[:8]}
    if not found:
        return "none", detail
    if len(found) == 1:
        return found[0], detail
    return "mixed(" + ",".join(found) + ")", detail


def _compiled_wrappers(model: Any) -> list[str]:
    """Names of the modules `torch.compile` replaced, root first as ''."""
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "dynamo_ctx") and hasattr(module, "_orig_mod")
    ]


def _capture_compile(built: Built, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """Which compilation a run is actually carrying.

    Read off the wrapper `torch.compile` returns and the inductor configuration it
    derived from the mode, not off the argument that was passed in. Three things
    that all look like a compiled run from the config make different answers here:
    a model nothing wrapped, a model wrapped at a different mode, and a wrapper
    built while dynamo is disabled process-wide (`TORCHDYNAMO_DISABLE=1` still
    returns an OptimizedModule).

    What it cannot see is a graph break: compilation is lazy, so at construction
    time no graph exists yet. This certifies the request reached torch, not that
    torch stayed in the compiled path.
    """
    import torch

    if built.model is None:
        return None, {"reason": "no model was built"}
    disabled = bool(getattr(torch._dynamo.config, "disable", False))
    wrappers = _compiled_wrappers(built.model)
    detail: dict[str, Any] = {"compiled_modules": len(wrappers), "dynamo_disabled": disabled}
    if disabled:
        return "none", {**detail, "reason": "torch._dynamo is disabled process-wide"}
    if not wrappers:
        return "none", detail
    if wrappers[0] != "":
        return "regional", {**detail, "wrapped_sample": wrappers[:5]}
    inductor = getattr(built.model.dynamo_ctx, "compiler_config", None)
    if not isinstance(inductor, dict):
        return None, {
            **detail,
            "reason": f"the compiled wrapper carries no inductor config "
            f"({type(inductor).__name__}), so its mode cannot be read back",
        }
    # `mode=max-autotune` is the only one of our values that changes inductor's
    # configuration; `default` leaves it identical to no mode at all, which is
    # what makes the two the same request.
    autotune = bool(inductor.get("max_autotune"))
    return ("max-autotune" if autotune else "default"), {**detail, "max_autotune": autotune}


def _capture_gradient_checkpointing(
    built: Built, config: BenchConfig
) -> tuple[str | None, dict[str, Any]]:
    """Whether activation checkpointing is on, read off the modules that do it.

    Scanned rather than taken from `model.is_gradient_checkpointing`, which is
    `any(...)`: a partial enable and a full one give it the same answer, and a
    partial enable is a model whose measured memory belongs to neither setting.
    """
    if built.model is None:
        return None, {"reason": "no model was built"}
    flags = [
        bool(module.gradient_checkpointing)
        for _, module in built.model.named_modules()
        if hasattr(module, "gradient_checkpointing")
    ]
    if not flags:
        return None, {"reason": "no module in this model exposes a gradient_checkpointing flag"}
    detail = {"modules_with_flag": len(flags), "enabled": sum(flags)}
    if all(flags):
        return "full", detail
    if not any(flags):
        return "none", detail
    return "partial", detail


def _capture_peft(built: Built, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """Which adapter, if any, the built model carries.

    `full` here means no adapter was attached, which is what the axis selects — it
    is not a claim that every parameter is trainable, because the freeze axes are
    free to turn parameters off inside a full finetune.
    """
    if built.model is None:
        return None, {"reason": "no model was built"}
    peft_config = getattr(built.model, "peft_config", None) or {}
    kinds = sorted(
        {
            str(getattr(entry, "peft_type", "unknown")).lower().rsplit(".", 1)[-1]
            for entry in getattr(peft_config, "values", lambda: [])()
        }
    )
    wrapper = type(built.model).__module__.split(".")[0]
    detail: dict[str, Any] = {"wrapper_package": wrapper, "adapters": kinds}
    if not kinds and wrapper != "peft":
        return "full", detail
    if kinds == ["lora"]:
        # qlora is lora over a quantised base, so the adapter type alone cannot
        # tell the two apart; the base's quantisation is what does.
        quantised = bool(
            getattr(built.model, "is_loaded_in_4bit", False)
            or getattr(getattr(built.model, "config", None), "quantization_config", None)
        )
        return ("qlora" if quantised else "lora"), {**detail, "base_quantised": quantised}
    # Deliberately not equal to any configurable value: an adapter we cannot name
    # must be a mismatch rather than fall into the nearest one.
    return "peft(" + ",".join(kinds or ["unknown"]) + ")", detail


# How each parallelism wrapper announces itself. Matched on the class name because
# the packages that define them are not installed in every environment, so
# `isinstance` would need an import that is allowed to fail.
PARALLEL_WRAPPERS = {
    "DistributedDataParallel": "ddp",
    "FullyShardedDataParallel": "fsdp2",
    "DeepSpeedEngine": "deepspeed",
}


def _capture_parallel_strategy(
    built: Built, config: BenchConfig
) -> tuple[str | None, dict[str, Any]]:
    """What the run is actually distributed as.

    `single` is checked against the process group, not only against the absence of
    a wrapper. A job launched under `torchrun` with `parallel=single` runs one
    unsynchronised replica per rank and finishes faster than one process would;
    that number is not a single-GPU number and must not be recorded as one.
    """
    import torch.distributed as dist

    if built.model is None:
        return None, {"reason": "no model was built"}
    initialised = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if initialised else 1
    detail: dict[str, Any] = {"world_size": world_size, "process_group": initialised}
    for name, module in built.model.named_modules():
        strategy = PARALLEL_WRAPPERS.get(type(module).__name__)
        if strategy is not None:
            return strategy, {**detail, "wrapper": type(module).__name__, "at": name or "model"}
    if world_size > 1:
        return f"unwrapped(world_size={world_size})", detail
    return "single", detail


def _capture_dataloader_backend(
    built: Built, config: BenchConfig
) -> tuple[str | None, dict[str, Any]]:
    """Which iterator feeds the run.

    DALI replaces the DataLoader rather than configuring one, so the type of the
    object is the whole answer.
    """
    import torch

    if built.dataloader is None:
        return None, {"reason": "no dataloader was built"}
    kind = type(built.dataloader)
    detail = {"class": kind.__name__, "package": kind.__module__.split(".")[0]}
    if isinstance(built.dataloader, torch.utils.data.DataLoader):
        return "torch", detail
    if detail["package"] == "nvidia":
        return "dali", detail
    return kind.__name__.lower(), detail


def _capture_dataloader_packing(
    built: Built, config: BenchConfig
) -> tuple[str | None, dict[str, Any]]:
    """Whether sequences are packed, read off the collate function.

    Packing is a property of how a batch is assembled, so torch's own collate is
    positive evidence that nothing packs: it concatenates along a new dimension
    and cannot produce a packed batch. A collate we do not recognise is
    undetermined rather than False — that is where a packing implementation would
    live, and guessing False about it is how an unpacked run gets a packed label.
    """
    from torch.utils.data import _utils

    if built.dataloader is None:
        return None, {"reason": "no dataloader was built"}
    collate = getattr(built.dataloader, "collate_fn", None)
    detail = {"collate": getattr(collate, "__name__", type(collate).__name__)}
    if collate in (_utils.collate.default_collate, _utils.collate.default_convert):
        return "False", detail
    declared = getattr(collate, "axis_packing", None)
    if declared is None:
        return None, {
            **detail,
            "reason": f"collate {detail['collate']} declares no axis_packing, so whether it "
            "packs cannot be read back",
        }
    return str(bool(declared)), detail


def _capture_dataloader_pretokenize(
    built: Built, config: BenchConfig
) -> tuple[str | None, dict[str, Any]]:
    """Whether the rows arrive already tokenised.

    Read off the dataset's declared columns, which is where the work moves to: a
    pretokenised run tokenises outside the timed window, so the loader looks
    identical either way and only its dataset differs. A dataset that declares no
    columns is undetermined; peeking at a row instead would make a capture do I/O
    on the failure path.
    """
    if built.dataloader is None:
        return None, {"reason": "no dataloader was built"}
    dataset = getattr(built.dataloader, "dataset", None)
    columns = getattr(dataset, "column_names", None)
    if columns is None:
        features = getattr(dataset, "features", None)
        columns = list(features) if features is not None else None
    if columns is None:
        return None, {
            "reason": f"{type(dataset).__name__} declares no column_names or features, so "
            "whether its rows are tokenised cannot be read back"
        }
    from trainbench import axes

    tokenised = sorted(set(columns) & set(axes.TOKENIZED_COLUMNS))
    return str(bool(tokenised)), {"columns": sorted(columns)[:12], "tokenised_columns": tokenised}


def _capture_cross_device_negatives(
    built: Built, config: BenchConfig
) -> tuple[str | None, dict[str, Any]]:
    """Whether the loss gathers embeddings across ranks.

    Read off the loss the run built, for the same reason `loss.name` is: this axis
    changes how many negatives each row is scored against, and reporting a
    world-sized batch's speed for a local-batch loss is the failure case.
    """
    if built.loss_fn is None:
        return None, {"reason": "no loss function was built"}
    declared = getattr(built.loss_fn, "axis_cross_device_negatives", None)
    if declared is None:
        return None, {
            "reason": f"{getattr(built.loss_fn, '__name__', type(built.loss_fn).__name__)} "
            "declares no axis_cross_device_negatives"
        }
    return str(bool(declared)), {"callable": getattr(built.loss_fn, "__name__", "?")}


_CAPTURES: dict[str, CaptureFn] = {
    "attn.name": _capture_attn,
    "compile.mode": _capture_compile,
    "dataloader.backend": _capture_dataloader_backend,
    "dataloader.packing": _capture_dataloader_packing,
    "dataloader.pretokenize": _capture_dataloader_pretokenize,
    "framework.name": _capture_framework,
    "freeze.ple": _capture_freeze_ple,
    "freeze.vision_tower": _capture_freeze_vision_tower,
    "kernel.name": _capture_kernel,
    "loss.name": _capture_loss,
    "optim.name": _capture_optim,
    "parallel.cross_device_negatives": _capture_cross_device_negatives,
    "parallel.strategy": _capture_parallel_strategy,
    "peft.mode": _capture_peft,
    "train.gradient_checkpointing": _capture_gradient_checkpointing,
}

# Where the applied value is expressed in different words from the config value.
# transformers reports 'flash_attention_3', never 'fa3', so the request has to be
# translated before the two are compared.
_REQUESTED_OVERRIDES: dict[str, Callable[[BenchConfig], Any]] = {
    "attn.name": lambda config: config.attn.impl,
}


def capture(built: Built, config: BenchConfig) -> AppliedState:
    """Read back the state of every axis from what the run actually constructed.

    Never raises: an axis that cannot be read is undetermined, which blocks a
    reportable run just as a mismatch does.
    """
    states = []
    for axis, default_reader in axis_knobs().items():
        read_requested = _REQUESTED_OVERRIDES.get(axis, default_reader)
        try:
            requested = str(read_requested(config))
        except BaseException as exc:  # noqa: BLE001 - an unreadable request is still an axis
            reason = f"{type(exc).__name__}: {exc}"[:300]
            states.append(AxisState(axis, "unreadable", None, {"reason": reason}))
            continue
        probe = _CAPTURES.get(axis)
        if probe is None:
            states.append(
                AxisState(axis, requested, None, {"reason": "no capture probe implemented"})
            )
            continue
        try:
            applied, detail = probe(built, config)
        except BaseException as exc:  # noqa: BLE001 - an unreadable axis is undetermined
            applied, detail = None, {"reason": f"{type(exc).__name__}: {exc}"[:300]}
        states.append(AxisState(axis, requested, applied, detail))
    return AppliedState(tuple(states))


def assert_matches(state: AppliedState, config: BenchConfig) -> None:
    """Refuse to let a reportable run proceed on unverified or wrong settings.

    Undetermined is treated exactly like mismatched: "we could not check" must not
    read as "it is fine", or the whole mechanism becomes decorative. An empty or
    partial state is the same failure one level up — it means capture never ran
    for some axis, not that the axis was fine.

    Takes the config rather than a purpose string so the value cannot be a typo:
    `"Timing"` would otherwise pass every check silently.
    """
    purpose = config.run.purpose
    if purpose not in KNOWN_PURPOSES:
        raise ValueError(f"unknown run purpose {purpose!r}; expected one of {KNOWN_PURPOSES}")
    if purpose not in ENFORCED_PURPOSES:
        return

    problems = [f"{axis}: never captured" for axis in state.missing()]
    for axis in state.mismatched():
        problems.append(f"{axis.axis}: requested {axis.requested!r}, applied {axis.applied!r}")
    for axis in state.undetermined():
        reason = axis.detail.get("reason", "unknown")
        problems.append(f"{axis.axis}: requested {axis.requested!r}, undetermined ({reason})")

    if problems:
        raise AppliedMismatch(
            f"purpose={purpose} requires every axis to be verified as applied. "
            + "; ".join(problems)
        )
