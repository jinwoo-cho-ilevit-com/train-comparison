"""Probe for Axolotl.

Axolotl is config-driven: models are loaded through a normalised cfg rather than a
direct loader call, so the probe builds a minimal cfg and drives ModelLoader the
way the project's own docs do. Axolotl's Qwen3-VL support is undocumented, which
is the main thing this answers.

`axes.load_kwargs` has nowhere to go on this path: axolotl normalises its own cfg
into the loader call and takes attention as cfg keys of its own, not as
`from_pretrained` keywords. Rather than invent a mapping, the axis is left
unapplied and the capture side reads back what the model was actually built with,
so an unhonoured request reads as a mismatch instead of as a success.
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import axolotl

    report.add_version(axolotl)
    steps.patch_axes(config, report)

    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        from axolotl.loaders.model import ModelLoader
        from axolotl.loaders.tokenizer import load_tokenizer
        from axolotl.utils.config import normalize_config, prepare_plugins
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault(
            {
                "base_model": config.model.hf_id,
                "sequence_len": config.data.max_seq_len,
                "bf16": True,
                "load_in_4bit": config.peft.mode == "qlora",
                "adapter": None if config.peft.mode == "full" else config.peft.mode,
                "lora_r": config.peft.r or None,
                "lora_alpha": config.peft.alpha or None,
                "lora_dropout": config.peft.dropout,
                "lora_target_linear": config.peft.mode != "full",
                "datasets": [],
            }
        )
        normalize_config(cfg)
        prepare_plugins(cfg)
        tokenizer = load_tokenizer(cfg)
        model, _ = ModelLoader(cfg, tokenizer).load()
        loaded["model"] = model
        loaded["tokenizer"] = tokenizer
        return {
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
        }

    if not report.run("model_loader_load", _load)[0]:
        report.skip("infonce_backward", "model did not load")
        return

    model, tokenizer = loaded["model"], loaded["tokenizer"]
    model.to(device)
    model = steps.verify_axes(model, config, device, "axolotl", report)

    tokenized: dict[str, torch.Tensor] = {}
    side = config.model.padding_side

    report.run("padding_side_alignment", lambda: steps.padding_side_alignment(tokenizer, side))

    if report.run("text_tokenize", lambda: steps.tokenize_text(tokenizer, device, tokenized, side))[
        0
    ]:
        report.run(
            "infonce_backward",
            lambda: steps.infonce_backward(model, tokenized, config.loss.temperature, side),
        )
    else:
        report.skip("infonce_backward", "tokenization failed")
