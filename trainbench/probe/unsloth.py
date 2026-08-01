"""Probe for Unsloth.

Answers three questions the documentation leaves open:
  1. does FastVisionModel accept these VLM checkpoints
  2. does Unsloth's patching survive a contrastive loss, which has no LM head and
     therefore bypasses the fused cross-entropy its speedups largely come from
  3. does FastSentenceTransformer actually refuse a VLM checkpoint (its docs only
     ever mention encoder-only models, which is not the same as refusing)
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import unsloth
    from unsloth import FastVisionModel

    report.add_version(unsloth)

    hf_id = config.model.hf_id
    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        model, processor = FastVisionModel.from_pretrained(
            hf_id,
            load_in_4bit=config.peft.mode == "qlora",
            dtype=steps.dtype_for(device),
        )
        loaded["model"] = model
        loaded["processor"] = processor
        return {"model_class": type(model).__name__, "processor_class": type(processor).__name__}

    if not report.run("fast_vision_model_load", _load)[0]:
        report.skip("infonce_backward", "model did not load")
        report.run("fast_sentence_transformer_accepts_vlm", lambda: _try_fast_st(hf_id))
        return

    model, processor = loaded["model"], loaded["processor"]

    if config.peft.mode in ("lora", "qlora"):
        report.run("get_peft_model", lambda: _peft(model, config))

    tokenized: dict[str, torch.Tensor] = {}
    side = config.model.padding_side

    if report.run("text_tokenize", lambda: steps.tokenize_text(processor, device, tokenized))[0]:
        report.run(
            "infonce_backward",
            lambda: steps.infonce_backward(model, tokenized, config.loss.temperature, side),
        )
    else:
        report.skip("infonce_backward", "tokenization failed")

    report.run("visual_tokens", lambda: steps.visual_token_count(processor, model, device))
    report.run("fast_sentence_transformer_accepts_vlm", lambda: _try_fast_st(hf_id))


def _peft(model: Any, config: BenchConfig) -> dict[str, Any]:
    from unsloth import FastVisionModel

    patched = FastVisionModel.get_peft_model(
        model,
        r=config.peft.r,
        lora_alpha=config.peft.alpha,
        lora_dropout=config.peft.dropout,
        target_modules="all-linear",
    )
    trainable = sum(p.numel() for p in patched.parameters() if p.requires_grad)
    total = sum(p.numel() for p in patched.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": round(trainable / total, 6) if total else None,
    }


def _try_fast_st(hf_id: str) -> dict[str, Any]:
    """Whether Unsloth's fast embedding path takes a VLM checkpoint.

    A refusal here is the expected outcome and is itself the finding: it would mean
    the 1.8-3.3x embedding speedup does not apply to any model in this study.
    """
    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(hf_id, for_inference=True)
    return {"accepted": True, "model_class": type(model).__name__}
