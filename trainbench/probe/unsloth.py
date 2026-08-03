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


def load(config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """`full_finetuning` is not optional here.

    Its default is False, and with 4bit, 8bit and full all off unsloth switches to
    "16bit LoRA" and exports UNSLOTH_ENABLE_FULL_FINETUNING=0; `post_patch_model`
    then hands that to unsloth_zoo's `prepare_model_for_training`, which freezes
    every parameter whose name has no LoRA marker. Under `peft.mode=full` nothing
    attaches LoRA, so the whole model comes back frozen (unsloth 2026.7.6
    models/vision.py:1164-1187, 2094-2129; unsloth_zoo 2026.7.7
    training_utils.py:383).

    `load_kwargs` has nowhere to go: `FastVisionModel.from_pretrained` owns the
    call and takes its own keywords, so the axis is left unapplied and the capture
    side reports the mismatch rather than a guessed keyword succeeding.

    `revision` is not one of `FastBaseModel.from_pretrained`'s named parameters
    (unsloth 2026.7.6 models/vision.py:810-841), but the call reads it out of
    `**kwargs` explicitly (`kwargs.get("revision")` at models/vision.py:1103,
    :1316, :1469), so passing it here does reach the checkpoint fetch rather than
    being silently dropped.
    """
    from unsloth import FastVisionModel

    return FastVisionModel.from_pretrained(
        config.model.hf_id,
        load_in_4bit=config.peft.mode == "qlora",
        full_finetuning=config.peft.mode == "full",
        dtype=steps.dtype_for(device),
        revision=config.model.revision,
    )


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import unsloth

    report.add_version(unsloth)
    steps.patch_axes(config, report)

    hf_id = config.model.hf_id
    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        # `load` above, which is also what a timing run takes: two copies of
        # `full_finetuning=` would drift, and a campaign that trained zero
        # parameters is what that costs.
        model, processor = load(config, device, {})
        loaded["model"] = model
        loaded["processor"] = processor
        return {
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "full_finetuning": config.peft.mode == "full",
        }

    if not report.run("fast_vision_model_load", _load)[0]:
        report.skip("infonce_backward", "model did not load")
        report.run(
            "fast_sentence_transformer_accepts_vlm",
            lambda: _try_fast_st(hf_id),
            expected_failure=True,
        )
        return

    processor = loaded["processor"]

    if config.peft.mode in ("lora", "qlora"):
        report.run("get_peft_model", lambda: _peft(loaded, config))

    # Before the optimizer, which docs/CONTRACTS.md §2 fixes as the order: an
    # optimizer built over the pre-peft parameters holds tensors the run no longer
    # trains through. `_peft` writes back into `loaded` for the same reason.
    model = steps.verify_axes(loaded["model"], config, device, "unsloth", report)

    tokenized: dict[str, torch.Tensor] = {}
    side = config.model.padding_side

    report.run(
        "padding_side_alignment",
        lambda: steps.padding_side_alignment(processor, side, hf_id, config.model.revision),
    )

    if report.run("text_tokenize", lambda: steps.tokenize_text(processor, device, tokenized, side))[
        0
    ]:
        report.run(
            "infonce_backward",
            lambda: steps.infonce_backward(model, tokenized, config.loss.temperature, side),
        )
    else:
        report.skip("infonce_backward", "tokenization failed")

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
    report.run(
        "fast_sentence_transformer_accepts_vlm",
        lambda: _try_fast_st(hf_id),
        expected_failure=True,
    )


def _peft(loaded: dict[str, Any], config: BenchConfig) -> dict[str, Any]:
    """Attach LoRA and keep whatever `get_peft_model` handed back.

    The return value used to be dropped, and every later check went on using the
    pre-peft `model`. That is harmless if Unsloth patches in place and wrong if it
    returns a wrapper, and which one it does cannot be established here — unsloth
    installs only inside its own image. Writing the result back into `loaded` is
    correct under both, so the question does not have to be answered to be safe.
    """
    from unsloth import FastVisionModel

    patched = FastVisionModel.get_peft_model(
        loaded["model"],
        r=config.peft.r,
        lora_alpha=config.peft.alpha,
        lora_dropout=config.peft.dropout,
        target_modules="all-linear",
    )
    replaced = patched is not loaded["model"]
    loaded["model"] = patched
    trainable = sum(p.numel() for p in patched.parameters() if p.requires_grad)
    total = sum(p.numel() for p in patched.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": round(trainable / total, 6) if total else None,
        # Which of the two behaviours this version has, recorded rather than assumed.
        "returned_a_new_object": replaced,
        "model_class": type(patched).__name__,
    }


def _try_fast_st(hf_id: str) -> dict[str, Any]:
    """Whether Unsloth's fast embedding path takes a VLM checkpoint.

    A refusal here is the expected outcome and is itself the finding: it would mean
    the 1.8-3.3x embedding speedup does not apply to any model in this study. Its
    caller therefore passes `expected_failure=True`, which is what keeps the
    answer from rendering as a broken cell — and what makes an acceptance show up
    in `ProbeReport.unexpected_passes`, i.e. as the support matrix being wrong.
    """
    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(hf_id, for_inference=True)
    return {"accepted": True, "model_class": type(model).__name__}
