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

# Axis knobs this module can actually put into effect. Required to equal
# `applied._CAPTURES`: `audit_plan.py`'s `axis-wired` check enforces it, and
# `tests/test_applied.py::test_applied_and_verified_sets_agree` pins it.
IMPLEMENTED = frozenset({"attn.name", "freeze.ple", "optim.name", "loss.name", "framework.name"})


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
    """Axes that change the model itself. May return a different object."""
    applied = []
    if config.freeze.ple:
        for _, param in ple_parameters(model):
            param.requires_grad_(False)
        applied.append("freeze.ple")
    return model, applied


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
    # nothing was decided, and the capture probe reports it undetermined.
    return loader, []


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
    return mnrl, ["loss.name"]
