"""Where an axis is turned on.

`trainbench/applied.py` reads back what a run ended up with; this module is the
other half — what asks for it. They are separated because they are checked
against each other: an axis is certified only when the code that applies it and
the code that verifies it are both present, and `IMPLEMENTED` here is required to
equal the set of capture probes over there.

The four functions below are the complete set of places a measured run turns an
axis on. Their number and shape are fixed here, in Wave 0, because the harness
that calls them (`scripts/bench.py`) is built later by a different lane: without
an agreed set of call sites, the lane that adds axes and the lane that adds the
harness each invent one.

    patch(config)                           -> applied      BEFORE construction
    load_kwargs(config)                     -> from_pretrained kwargs
    assemble(model, config, device, ...)    -> (Built, applied)
    step_context(config)                    -> context manager around the step

`patch` comes first because some kernels replace classes rather than instances.
Liger's documented sequence is `apply_liger_kernel_to_llama()` and then
"# 2. Instantiate patched model" (Liger README), so a model built before the
patch is a model the axis never reached.

`assemble` returns everything at once rather than exposing a builder per piece,
because some axes do not let the pieces be built separately:
`deepspeed.initialize(model=..., model_parameters=..., training_data=...)`
returns the engine, the optimizer and the dataloader from a single call
(DeepSpeed docs, cifar-10 and bert-pretraining tutorials), so `zero2`/`zero3` and
the offload axis cannot be split across three independent hooks. It returns the
model for the same reason `torch.compile`, `get_peft_model` and FSDP do: they
replace the model rather than mutate it.

`step_context` exists because precision is not only a construction-time choice —
fp8 recipes wrap the forward pass — and an axis with nowhere to live is an axis
that gets applied somewhere unverified.

An axis value with no implementation raises `UnappliedAxis` rather than falling
back to the default. A silent substitution is the failure this whole mechanism
exists to prevent, and it is not less of one for happening in our own code.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from trainbench.applied import Built
from trainbench.config_schema import BenchConfig
from trainbench.embedding import info_nce

# gemma-4's per-layer embeddings. Every one of the 108 PLE tensors carries this
# in its name — `language_model.embed_tokens_per_layer.weight`,
# `layers.N.per_layer_input_gate`, `per_layer_model_projection`, and so on
# (docs/model-spec.md, read off model.safetensors.index.json). Substring rather
# than an enumerated list because the layer index is part of the name.
PLE_PARAM_MARKER = "per_layer"

# Vision-tower parameters, per architecture. Read off each checkpoint rather than
# guessed: docs/model-spec.md says in as many words that a guessed marker freezes
# zero tensors and records that as success, which is the failure `_ple_report`
# had already shipped once with `altup`.
#
# Measured 2026-08-01 from each repo's safetensors header on `main`:
#
#   Qwen/Qwen3-VL-Embedding-2B   model.visual.*         315 of  625 tensors
#   Qwen/Qwen3.5-0.8B            model.visual.*         153 of  488 tensors
#   google/gemma-4-E2B           model.vision_tower.*   658 of 2011 tensors
#                                model.embed_vision.*     1
#
# Both Qwen models keep the projector inside the tower (`visual.merger`), while
# gemma-4 keeps it outside as `embed_vision.embedding_projection`; it is included
# so the axis means the same thing on all three. gemma-4's `audio_tower` is
# deliberately not here — it is a third tower, not the vision one.
VISION_PARAM_MARKERS = {
    "qwen3_vl": ("visual.",),
    "qwen3_5": ("visual.",),
    "gemma4": ("vision_tower.", "embed_vision."),
}

# Class definitions from these packages inside a built model are the evidence
# that a kernel library patched it. Liger replaces the transformers classes
# (`apply_liger_kernel_to_llama()` then "# 2. Instantiate patched model"), so a
# model built afterwards carries their modules; a model built before it does not.
KERNEL_MODULE_ROOTS = {
    "liger_kernel": "liger",
    "fla": "fla",
    "kernels": "kernels_hub",
}

# Columns whose presence means the rows arrived already tokenised. Read off the
# dataset the loader was built around, because that is where `pretokenize` moves
# the work to — the loader itself looks the same either way.
TOKENIZED_COLUMNS = ("input_ids",)

# Axis knobs this module can actually put into effect. Required to equal
# `applied._CAPTURES`: `audit_plan.py`'s `axis-wired` check enforces it, and
# `tests/test_applied.py::test_applied_and_verified_sets_agree` pins it.
#
# `precision.name` and `train.offload` are absent on purpose. Neither has a site
# here: the load dtype is chosen by `probe/steps.py::dtype_for` from the device,
# and offload is inseparable from `deepspeed.initialize`, which builds the model,
# optimizer and dataloader together. The module docstring of tests/test_axes.py
# carries the long form.
IMPLEMENTED = frozenset(
    {
        "attn.name",
        "compile.mode",
        "dataloader.backend",
        "dataloader.packing",
        "dataloader.pretokenize",
        "framework.name",
        "freeze.ple",
        "freeze.vision_tower",
        "kernel.name",
        "loss.name",
        "optim.name",
        "parallel.cross_device_negatives",
        "parallel.strategy",
        "peft.mode",
        "train.gradient_checkpointing",
    }
)


class UnappliedAxis(RuntimeError):
    """An axis value that nothing here can put into effect.

    Raised instead of returning the default, because the default would then be
    measured under the requested value's name.
    """


def patch(config: BenchConfig) -> list[str]:
    """Axes that must be applied before the model exists.

    Kernel libraries monkey-patch the transformers classes, so patching after
    construction leaves the already-built modules untouched: the run would report
    a kernel that never ran.
    """
    if config.kernel.name != "none":
        raise UnappliedAxis(
            f"kernel={config.kernel.name} patches transformers classes before the model "
            "is built; not implemented."
        )
    return []


def load_kwargs(config: BenchConfig) -> dict[str, Any]:
    """Keyword arguments for `from_pretrained`.

    Attention is set here rather than afterwards because transformers validates
    and may downgrade the request during construction; setting it later would mean
    the model was built once with the wrong one.
    """
    return {"attn_implementation": config.attn.impl}


def ple_parameters(model: Any) -> list[tuple[str, Any]]:
    """The per-layer embedding tensors, by name."""
    return [(n, p) for n, p in model.named_parameters() if PLE_PARAM_MARKER in n]


def vision_parameters(model: Any, arch: str) -> list[tuple[str, Any]]:
    """The vision tower's tensors, by name, for this architecture.

    Raises for an architecture with no measured marker rather than returning
    nothing: an empty list would freeze nothing and read as a tower that happens
    to be small, and `freeze.vision_tower` would then be reported as applied.
    """
    markers = VISION_PARAM_MARKERS.get(arch)
    if markers is None:
        raise UnappliedAxis(
            f"no vision-tower parameter marker is recorded for arch={arch!r}; "
            f"known: {sorted(VISION_PARAM_MARKERS)}. Read it off the checkpoint's "
            "safetensors header before adding one (docs/model-spec.md)."
        )
    return [
        (name, param)
        for name, param in model.named_parameters()
        if any(marker in name for marker in markers)
    ]


def assemble(
    model: Any,
    config: BenchConfig,
    device: torch.device,
    framework: str,
    dataset: Any = None,
) -> tuple[Built, list[str]]:
    """Build everything a run needs, and report which axes that put into effect.

    `framework` is passed in by the adapter that is running rather than read from
    the config: the config says which framework was requested, and the whole point
    of the capture side is that the request is not evidence of what ran.

    Does not report success beyond naming the axes it applied — `applied.capture`
    decides that by inspecting the result. A function that both acts and certifies
    its own action cannot catch the case where the action did not take.
    """
    applied: list[str] = []
    if config.parallel.strategy in ("zero2", "zero3") or config.train.offload != "none":
        raise UnappliedAxis(
            f"parallel={config.parallel.strategy} / offload={config.train.offload} needs "
            "deepspeed.initialize, which returns the model, optimizer and dataloader "
            "together; it has to be built here rather than by the pieces below."
        )
    if config.parallel.strategy != "single":
        raise UnappliedAxis(
            f"parallel.strategy={config.parallel.strategy} wraps the model (DDP, FSDP2) "
            "and needs an initialised process group; not implemented."
        )

    model, names = _apply_to_model(model, config)
    applied += names
    optimizer, names = _optimizer(model.parameters(), config, device)
    applied += names
    loss, names = _loss(config)
    applied += names
    loader, names = _dataloader(dataset, config)
    applied += names

    built = Built(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        loss_fn=loss,
        framework=framework,
    )
    return built, [*applied, "framework.name"]


def step_context(config: BenchConfig) -> contextlib.AbstractContextManager:
    """Context wrapping one training step.

    bf16 needs none: the model is already loaded in that dtype, so an autocast
    region would be a second, different answer to the same question. The fp8
    recipes do need one, and refusing here is what keeps a bf16 step from being
    measured under their name.
    """
    if config.precision.name != "bf16":
        raise UnappliedAxis(
            f"precision={config.precision.name} needs a Transformer Engine recipe "
            "around the forward pass, which is not implemented."
        )
    return contextlib.nullcontext()


def _apply_to_model(model: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """Axes that change the model itself. May return a different object.

    One ordering constraint is real and one is a structuring choice, and they are
    labelled here because an invented reason is worse than none.

    Real: freezing runs before peft, because peft freezes every base parameter and
    the freeze axes would have nothing left to decide afterwards; and all of it
    runs before the optimizer is built (docs/CONTRACTS.md §2 fixes this — FSDP2
    needs the optimizer built over sharded parameters), so the optimizer holds the
    parameters the run actually trains.

    A choice: `_compile` is last because it is the only site that replaces the
    object rather than mutating it, so keeping it last means every other site
    receives the model it was handed. This is not a correctness requirement. An
    earlier version of this docstring claimed checkpointing had to precede compile
    because "the compiled wrapper is not that model"; that is false —
    `OptimizedModule.__getattr__` delegates to `_orig_mod`, so the hook reaches
    through, the flags get set on the inner modules, and `named_modules()` still
    finds them. Reversing the two leaves the suite green. Whether the reverse
    order costs anything at run time is unmeasured.

    An axis whose configured value is the inert one — `compile=none`,
    `peft=full`, `freeze.*=false` — applies nothing and so is not named here.
    What it means is read back by `applied.capture`, which looks at the object
    rather than at this list.
    """
    applied: list[str] = []
    applied += _freeze(model, config)
    model, names = _peft(model, config)
    applied += names
    applied += _gradient_checkpointing(model, config)
    model, names = _compile(model, config)
    applied += names
    return model, applied


def _freeze(model: Any, config: BenchConfig) -> list[str]:
    """Turn off gradients for the tensors each freeze axis names.

    A marker that matches nothing is left to the capture probe rather than raised
    on here: `applied._capture_freeze_ple` reports zero matches as undetermined,
    which blocks a reportable run without stopping a probe whose job is to find
    out what the checkpoint actually contains.
    """
    applied: list[str] = []
    if config.freeze.ple:
        for _, param in ple_parameters(model):
            param.requires_grad_(False)
        applied.append("freeze.ple")
    if config.freeze.vision_tower:
        for _, param in vision_parameters(model, config.model.arch):
            param.requires_grad_(False)
        applied.append("freeze.vision_tower")
    return applied


def _peft(model: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """Adapter attachment. `full` attaches nothing, which is the whole of it.

    LoRA is refused rather than attached because `get_peft_model` freezes every
    base parameter, so a `freeze.ple=false` LoRA run would read back as frozen and
    every LoRA timing run would be blocked the moment this axis starts reporting
    (docs/CONTRACTS.md §2). What `freeze.*` should mean under an adapter — frozen,
    or frozen on top of what peft froze — is a decision that has to be made and
    tested against a real peft model, and this lane has no environment with peft
    in it to test either answer against.
    """
    if config.peft.mode != "full":
        raise UnappliedAxis(
            f"peft.mode={config.peft.mode} needs get_peft_model, which rewrites the model "
            "in place and freezes every base parameter; how that combines with the freeze "
            "axes is undecided (docs/CONTRACTS.md §2), so it is refused rather than run as "
            "full finetuning under a LoRA label."
        )
    return model, []


def _gradient_checkpointing(model: Any, config: BenchConfig) -> list[str]:
    """Trade compute for activation memory.

    `use_reentrant=False` is not a tuning choice here: the reentrant variant skips
    the recomputation entirely when no input to a checkpointed block requires
    grad, which is exactly what a frozen vision tower produces — and `freeze.*`
    and this axis are crossed in the ablation.
    """
    mode = config.train.gradient_checkpointing
    if mode == "none":
        return []
    if mode != "full":
        raise UnappliedAxis(
            f"train.gradient_checkpointing={mode} needs a policy for "
            "torch.utils.checkpoint.create_selective_checkpoint_contexts; not implemented."
        )
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise UnappliedAxis(
            f"{type(model).__name__} has no gradient_checkpointing_enable, so nothing here "
            "can turn train.gradient_checkpointing=full on for it."
        )
    enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return ["train.gradient_checkpointing"]


def _compile(model: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """`torch.compile`, whole-model or per repeated block.

    The schema's values other than `none` are torch's own mode spellings, so they
    are passed through rather than translated; a translation table would be a
    second place for `max-autotune` to be spelled. `none` is spelled that way
    because YAML reads a bare `off` as boolean False.

    Regional compilation goes through the model's own `compile_repeated_blocks`
    rather than a walk of the module tree: which blocks repeat is a property of
    the architecture, and guessing it would compile the wrong thing under the
    right name.
    """
    mode = config.compile.mode
    if mode == "none":
        return model, []
    if mode == "regional":
        compile_blocks = getattr(model, "compile_repeated_blocks", None)
        if not callable(compile_blocks):
            raise UnappliedAxis(
                f"compile=regional needs {type(model).__name__}.compile_repeated_blocks, "
                "which this model does not have."
            )
        compile_blocks()
        return model, ["compile.mode"]
    return torch.compile(model, mode=mode), ["compile.mode"]


def _optimizer(params: Any, config: BenchConfig, device: torch.device) -> tuple[Any, list[str]]:
    """`fused` follows the device: the fused AdamW kernel is CUDA-only and asking
    for it on CPU raises. The capture probe reports the unfused case as a
    different applied value, so a CPU run cannot report a fused number."""
    if config.optim.name != "adamw_fused":
        raise UnappliedAxis(
            f"optim={config.optim.name} has no implementation here; adamw_8bit needs "
            "bitsandbytes and muon needs a Muon implementation, neither of which is in "
            "any environment yet (audit: axis-packages)."
        )
    built = torch.optim.AdamW(
        params,
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        fused=device.type == "cuda",
    )
    return built, ["optim.name"]


def _dataloader(dataset: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """DALI replaces the DataLoader with its own iterator rather than configuring
    one (DALI docs: "replacing the standard DataLoader with DALIClassificationIterator"),
    which is why this builds the loader instead of returning kwargs for it."""
    if config.dataloader.backend != "torch":
        raise UnappliedAxis(
            f"dataloader.backend={config.dataloader.backend} builds its own iterator; "
            "not implemented."
        )
    if config.dataloader.packing or config.dataloader.pretokenize:
        raise UnappliedAxis("dataloader packing/pretokenize are not implemented.")
    if dataset is None:
        return None, []
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
    )
    # The backend axis is only applied once a loader exists; without a dataset
    # nothing was decided, and the capture probe reports it undetermined. packing
    # and pretokenize are not named: both are false here, and false is the absence
    # of an application rather than one.
    return loader, ["dataloader.backend"]


def _loss(config: BenchConfig) -> tuple[Any, list[str]]:
    """GradCache is refused rather than approximated: it changes the gradient
    computation, and measuring plain in-batch negatives under its name would
    report a speedup for work that was never done."""
    if config.loss.name == "cached_mnrl":
        raise UnappliedAxis(
            "loss=cached_mnrl needs a GradCache implementation and its own numerical "
            "equivalence test before it can be measured (PLAN.md, Wave 3)."
        )
    if config.loss.name != "mnrl":
        raise UnappliedAxis(f"loss={config.loss.name} has no implementation here.")
    if config.parallel.cross_device_negatives:
        raise UnappliedAxis(
            "parallel.cross_device_negatives needs an all-gather inside the loss; not implemented."
        )
    temperature = config.loss.temperature

    def mnrl(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        return info_nce(queries, documents, temperature)

    # Read back by applied._capture_loss: the function's identity is the only
    # evidence of which loss a run actually computed.
    mnrl.axis_value = "mnrl"
    # A literal set by the branch that built the non-gathering closure, not a copy
    # of the config. `mnrl` above computes similarities over the local batch only,
    # so this records what the function does, and a loss built anywhere else
    # declares nothing and comes back undetermined.
    mnrl.axis_cross_device_negatives = False
    return mnrl, ["loss.name"]
