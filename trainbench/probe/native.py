"""Probe for the native harness: plain transformers.

This is the reference path. When a framework probe fails but this one succeeds for
the same model, the failure belongs to the framework rather than the model.
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench import axes
from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    """Fills `report` in place. The registry owns it so that whatever was recorded
    before a crash survives the crash (see trainbench/probe/registry.py)."""
    from transformers import AutoModel, AutoProcessor

    hf_id = config.model.hf_id
    revision = config.model.revision

    steps.patch_axes(config, report)
    # Before the load and not inside it: a refused load-time axis is an answer
    # about the axis, not about the checkpoint (see steps.load_kwargs).
    load_kwargs = steps.load_kwargs(config, report)

    ok, processor = report.run(
        "processor_load",
        lambda: AutoProcessor.from_pretrained(hf_id, revision=revision),
    )
    if not ok:
        report.skip("model_load", "processor did not load")
        return

    ok, model = report.run(
        "model_load",
        # AutoModel rather than the generative head: an embedding model never
        # materialises the LM head, which for gemma4 is 262144 x 1536.
        # Axis settings come from trainbench/axes.py so that there is one place
        # that asks for them and one place that reads them back.
        lambda: AutoModel.from_pretrained(
            hf_id,
            revision=revision,
            dtype=steps.dtype_for(device),
            **load_kwargs,
        ),
    )
    if not ok:
        return
    model.to(device)

    # "native" is a literal: the config says which framework was asked for, and
    # this file is the evidence of which one ran.
    model = steps.verify_axes(model, config, device, "native", report)

    # Whatever a check returns is recorded as its detail, so the tensors stay in
    # this dict and only shapes come back out.
    tokenized: dict[str, torch.Tensor] = {}
    side = config.model.padding_side

    report.run("padding_side_alignment", lambda: steps.padding_side_alignment(processor, side))

    text_ok = report.run(
        "text_tokenize", lambda: steps.tokenize_text(processor, device, tokenized, side)
    )[0]

    report.run(
        "visual_tokens",
        lambda: steps.visual_token_count(
            processor,
            model,
            device,
            side,
            config.model.max_tokens_per_image,
            config.model.prompt_format,
        ),
    )

    if text_ok:
        report.run("text_embed_forward", lambda: _embed(model, tokenized, side))
        report.run(
            "infonce_backward",
            lambda: steps.infonce_backward(model, tokenized, config.loss.temperature, side),
        )
    else:
        report.skip("text_embed_forward", "tokenization failed")
        report.skip("infonce_backward", "tokenization failed")

    report.run(
        "multimodal_embed_forward",
        lambda: _multimodal_embed(model, processor, device, side, config.model.prompt_format),
    )

    if config.model.arch == "gemma4":
        report.run("ple_parameters", lambda: _ple_report(model))

    if config.peft.mode in ("lora", "qlora"):
        report.run("lora_attach", lambda: _lora_attach(model, config))


def _embed(model: Any, batch: dict[str, torch.Tensor], padding_side: str) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        pooled = steps.encode(model, batch, padding_side)
    return {"embedding_shape": list(pooled.shape)}


def _multimodal_embed(
    model: Any, processor: Any, device: torch.device, padding_side: str, prompt_format: str
) -> dict[str, Any]:
    model.eval()
    batch = steps.image_batch(processor, device, padding_side, prompt_format)
    with torch.no_grad():
        pooled = steps.encode(model, batch, padding_side)
    return {"embedding_shape": list(pooled.shape), "seq_len": int(batch["input_ids"].shape[1])}


def _ple_report(model: Any) -> dict[str, Any]:
    """Locate gemma4 per-layer embeddings and confirm they can be frozen.

    Roughly half of gemma-4-E2B's 5.1B parameters live in these tables, so whether
    they are trainable dominates the optimizer-memory picture for this model.
    """
    # Which parameters are PLE is defined once, in axes.py. A second definition
    # here drifted already: it also matched "altup", which the measured weight map
    # shows matches nothing (docs/model-spec.md), and it would have gone on
    # disagreeing with the freeze axis that this check is about.
    matches = [(name, param.numel()) for name, param in axes.ple_parameters(model)]
    # Zero matches is a failure, not a finding. Roughly half of gemma-4-E2B lives in
    # these tables, so an empty match means the name marker no longer fits this
    # checkpoint — and `freeze.ple` would then freeze nothing while reporting that
    # it froze what it was asked to. Reporting ok on a zero match is how the second,
    # already-drifted PLE definition went unnoticed.
    if not matches:
        raise ValueError(
            f"no parameter name contains {axes.PLE_PARAM_MARKER!r} in this gemma4 "
            f"checkpoint ({len(list(model.named_parameters()))} parameters); "
            "freeze.ple would silently freeze nothing."
        )
    total = sum(p.numel() for p in model.parameters())
    ple_total = sum(n for _, n in matches)

    before = {name: param.requires_grad for name, param in model.named_parameters()}
    for _, param in axes.ple_parameters(model):
        param.requires_grad_(False)
    frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    # Restore what was there, not everything to True: forcing True would undo the
    # freeze axis that axes.apply() just set, and the run would measure a model
    # training 2.39B parameters it was told to freeze.
    for name, param in model.named_parameters():
        param.requires_grad_(before[name])
    return {
        "matched_parameter_names": [name for name, _ in matches][:20],
        "matched_count": len(matches),
        "ple_parameters": ple_total,
        "total_parameters": total,
        "ple_fraction": round(ple_total / total, 4) if total else None,
        "froze_successfully": frozen,
    }


def _lora_attach(model: Any, config: BenchConfig) -> dict[str, Any]:
    """Whether peft accepts this architecture. Terminal: no check may follow it.

    `get_peft_model` rewrites the model's modules in place, so this leaves behind
    a model that is no longer the one the axes were captured from. It runs last
    and its result is discarded for that reason.

    It is not an axis application site — `peft.mode` has no implementation in
    axes.py yet, because freezing under LoRA collides with `freeze.ple`: peft
    freezes every base parameter, so a `freeze.ple=false` run would read back as
    frozen. Deciding whether freeze axes mean "frozen" or "frozen in addition to
    what peft does" belongs to the lane that implements the axis, not here.
    """
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=config.peft.r,
            lora_alpha=config.peft.alpha,
            lora_dropout=config.peft.dropout,
            target_modules="all-linear",
        ),
    )
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": round(trainable / total, 6) if total else None,
    }
