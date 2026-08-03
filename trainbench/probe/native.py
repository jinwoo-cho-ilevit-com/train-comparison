"""Probe for the native harness: plain transformers.

This is the reference path. When a framework probe fails but this one succeeds for
the same model, the failure belongs to the framework rather than the model.
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def load_processor(config: BenchConfig) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        config.model.hf_id,
        revision=config.model.revision,
        **steps.pixel_budget_kwargs(config),
    )


def load_model(config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]) -> Any:
    """AutoModel rather than the generative head: an embedding model never
    materialises the LM head, which is a large tensor no run here needs."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        config.model.hf_id,
        revision=config.model.revision,
        dtype=steps.dtype_for(device),
        **load_kwargs,
    )


def load(config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """The reference build, for `trainbench/loader.py`.

    The probe below takes the same two calls as separate checks: a processor that
    does not load and a model that does not load are different answers, and the
    harness has no report to record the difference in.
    """
    processor = load_processor(config)
    model = load_model(config, device, load_kwargs)
    model.to(device)
    return model, processor


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    """Fills `report` in place. The registry owns it so that whatever was recorded
    before a crash survives the crash (see trainbench/probe/registry.py)."""
    hf_id = config.model.hf_id
    revision = config.model.revision

    steps.patch_axes(config, report)
    # Before the load and not inside it: a refused load-time axis is an answer
    # about the axis, not about the checkpoint (see steps.load_kwargs).
    load_kwargs = steps.load_kwargs(config, report)

    ok, processor = report.run("processor_load", lambda: load_processor(config))
    if not ok:
        report.skip("model_load", "processor did not load")
        return

    ok, model = report.run("model_load", lambda: load_model(config, device, load_kwargs))
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

    report.run(
        "padding_side_alignment",
        lambda: steps.padding_side_alignment(processor, side, hf_id, revision),
    )

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

    if config.peft.mode == "lora":
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


def _lora_attach(model: Any, config: BenchConfig) -> dict[str, Any]:
    """Whether peft accepts this architecture. Terminal: no check may follow it.

    `get_peft_model` rewrites the model's modules in place, so this leaves behind
    a model that is no longer the one the axes were captured from. It runs last
    and its result is discarded for that reason.

    It is not an axis application site: freezing under LoRA collides with
    `freeze.vision_tower`, because peft freezes every base parameter, so a
    `freeze.vision_tower=false` run would read back as frozen regardless
    (`config_schema.py`'s `_freeze_axes_mean_nothing_under_an_adapter` refuses
    the combination outright).
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
