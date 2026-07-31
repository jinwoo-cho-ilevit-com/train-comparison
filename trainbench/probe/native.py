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


def run(config: BenchConfig, device: torch.device) -> ProbeReport:
    report = ProbeReport(framework="native", model=config.model.name)
    from transformers import AutoModel, AutoProcessor

    hf_id = config.model.hf_id
    revision = config.model.revision

    ok, processor = report.run(
        "processor_load",
        lambda: AutoProcessor.from_pretrained(hf_id, revision=revision),
    )
    if not ok:
        report.skip("model_load", "processor did not load")
        return report

    ok, model = report.run(
        "model_load",
        # AutoModel rather than the generative head: an embedding model never
        # materialises the LM head, which for gemma4 is 262144 x 1536.
        lambda: AutoModel.from_pretrained(hf_id, revision=revision, dtype=steps.dtype_for(device)),
    )
    if not ok:
        return report
    model.to(device)

    # Whatever a check returns is recorded as its detail, so tensors are kept in a
    # closure rather than returned.
    tokenized: dict[str, torch.Tensor] = {}

    def _tokenize() -> dict[str, Any]:
        tokenized.update(steps.text_batch(processor, device))
        return {
            "keys": sorted(tokenized),
            "input_ids_shape": list(tokenized["input_ids"].shape),
        }

    text_ok = report.run("text_tokenize", _tokenize)[0]

    report.run("visual_tokens", lambda: steps.visual_token_count(processor, model, device))

    if text_ok:
        report.run("text_embed_forward", lambda: _embed(model, tokenized))
        report.run(
            "infonce_backward",
            lambda: steps.infonce_backward(model, tokenized, config.loss.temperature),
        )
    else:
        report.skip("text_embed_forward", "tokenization failed")
        report.skip("infonce_backward", "tokenization failed")

    report.run("multimodal_embed_forward", lambda: _multimodal_embed(model, processor, device))

    if config.model.arch == "gemma4":
        report.run("ple_parameters", lambda: _ple_report(model))

    if config.peft.mode in ("lora", "qlora"):
        report.run("lora_attach", lambda: _lora_attach(model, config))

    return report


def _embed(model: Any, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        pooled = steps.encode(model, batch)
    return {"embedding_shape": list(pooled.shape)}


def _multimodal_embed(model: Any, processor: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    batch = steps.image_batch(processor, device)
    with torch.no_grad():
        pooled = steps.encode(model, batch)
    return {"embedding_shape": list(pooled.shape), "seq_len": int(batch["input_ids"].shape[1])}


def _ple_report(model: Any) -> dict[str, Any]:
    """Locate gemma4 per-layer embeddings and confirm they can be frozen.

    Roughly half of gemma-4-E2B's 5.1B parameters live in these tables, so whether
    they are trainable dominates the optimizer-memory picture for this model.
    """
    matches = [
        (name, param.numel())
        for name, param in model.named_parameters()
        if "per_layer" in name or "altup" in name
    ]
    total = sum(p.numel() for p in model.parameters())
    ple_total = sum(n for _, n in matches)
    for name, param in model.named_parameters():
        if "per_layer" in name or "altup" in name:
            param.requires_grad_(False)
    frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    # Restore, so later checks see the model as loaded.
    for _, param in model.named_parameters():
        param.requires_grad_(True)
    return {
        "matched_parameter_names": [name for name, _ in matches][:20],
        "matched_count": len(matches),
        "ple_parameters": ple_total,
        "total_parameters": total,
        "ple_fraction": round(ple_total / total, 4) if total else None,
        "froze_successfully": frozen,
    }


def _lora_attach(model: Any, config: BenchConfig) -> dict[str, Any]:
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
