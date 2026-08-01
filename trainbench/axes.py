"""Where an axis is turned on.

`trainbench/applied.py` reads back what a run ended up with; this module is the
other half — what asks for it. They are separated because they are checked
against each other: an axis is certified only when the code that applies it and
the code that verifies it are both present, and `IMPLEMENTED` here is required to
equal the set of capture probes over there.

The hooks below are the complete set of places a measured run turns an axis on.
Their number and shape are fixed here, in Wave 0, because the harness that calls
them (`scripts/bench.py`) is built later by a different lane: without an agreed
set of call sites, the lane that adds axes and the lane that adds the harness
each invent one, and the contract's whole purpose is to stop that.

    load_kwargs(config)                -> from_pretrained kwargs
    apply(model, config)               -> (model, applied)      may wrap or replace
    optimizer(params, config, device)  -> (optimizer, applied)
    dataloader_kwargs(config)          -> DataLoader kwargs
    loss_fn(config)                    -> (callable, applied)

`apply` returns the model because half the axes replace it rather than mutate it:
`torch.compile` and `get_peft_model` both hand back a new object, and FSDP/DeepSpeed
wrap it. A signature that only mutates in place cannot express them.

An axis value with no implementation raises `UnappliedAxis` rather than falling
back to the default. A silent substitution is the failure this whole mechanism
exists to prevent, and it is not less of one for happening in our own code.
"""

from __future__ import annotations

from typing import Any

import torch

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
IMPLEMENTED = frozenset({"attn.name", "freeze.ple", "optim.name", "loss.name"})


class UnappliedAxis(RuntimeError):
    """An axis value that nothing here can put into effect.

    Raised instead of returning the default, because the default would then be
    measured under the requested value's name.
    """


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


def apply(model: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """Put the post-construction axes into effect.

    Returns the model as well as the axes applied: an axis that wraps or replaces
    the model has no other way to hand the result back, and a caller that ignored
    it would keep measuring the unwrapped one.

    Does not report success — `applied.capture` does that by looking at the result
    afterwards. A function that both acts and certifies its own action cannot
    catch the case where the action did not take.
    """
    applied = []
    if config.freeze.ple:
        for _, param in ple_parameters(model):
            param.requires_grad_(False)
        applied.append("freeze.ple")
    return model, applied


def optimizer(params: Any, config: BenchConfig, device: torch.device) -> tuple[Any, list[str]]:
    """Build the optimizer for the requested axis value.

    `fused` follows the device: the fused AdamW kernel is CUDA-only and asking for
    it on CPU raises. The capture probe reports the unfused case as a different
    applied value rather than the same one, so a CPU run cannot report a fused
    number.
    """
    if config.optim.name != "adamw_fused":
        raise UnappliedAxis(
            f"optim={config.optim.name} has no implementation here; "
            "adamw_8bit needs bitsandbytes and muon needs a Muon implementation, "
            "neither of which is in any environment yet (audit: axis-packages)."
        )
    built = torch.optim.AdamW(
        params,
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        fused=device.type == "cuda",
    )
    return built, ["optim.name"]


def dataloader_kwargs(config: BenchConfig) -> dict[str, Any]:
    """Keyword arguments for the DataLoader.

    The backend axis itself is not implemented — DALI builds a pipeline rather
    than taking kwargs — so a DALI request is refused here rather than quietly
    served by torch.
    """
    if config.dataloader.backend != "torch":
        raise UnappliedAxis(
            f"dataloader.backend={config.dataloader.backend} has no implementation here."
        )
    return {"num_workers": config.data.num_workers}


def loss_fn(config: BenchConfig) -> tuple[Any, list[str]]:
    """The contrastive loss for the requested axis value.

    GradCache (`cached_mnrl`) is refused rather than approximated: it changes the
    gradient computation, and measuring plain in-batch negatives under its name
    would report a speedup for work that was never done.
    """
    if config.loss.name == "cached_mnrl":
        raise UnappliedAxis(
            "loss=cached_mnrl needs a GradCache implementation and its own numerical "
            "equivalence test before it can be measured (PLAN.md, Wave 3)."
        )
    if config.loss.name != "mnrl":
        raise UnappliedAxis(f"loss={config.loss.name} has no implementation here.")
    temperature = config.loss.temperature

    def mnrl(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        return info_nce(queries, documents, temperature)

    # Read back by applied._capture_loss: the function's identity is the only
    # evidence of which loss a run actually computed.
    mnrl.axis_value = "mnrl"
    return mnrl, ["loss.name"]
