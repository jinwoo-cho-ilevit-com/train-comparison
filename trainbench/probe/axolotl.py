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

from trainbench import axes
from trainbench.config_schema import BenchConfig
from trainbench.loader import ADAPTERS
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def load(config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """axolotl's own order, which this probe used to invert.

    `normalize_config` is written against a cfg that `validate_config` has already
    filled in: run first it divides `batch_size // micro_batch_size` with both
    still None, because `DictDefault.__missing__` returns None instead of raising
    (axolotl 0.18.0 cli/config.py:303-322). Filling those two keys by hand does not
    fix it — the next unfilled key is `context_parallel_size`, compared against 1
    in loaders/patch_manager.py. Only validation puts defaults on all of them.

    `revision_of_model` is the cfg key, not `revision`: `ModelLoader` reads
    `cfg.revision_of_model` into its own `model_kwargs["revision"]`
    (axolotl 0.18.0 loaders/model.py:223-224), and `load_tokenizer` /
    `loaders/processor.py` read the same key for the tokenizer and processor
    (loaders/tokenizer.py:144-177, loaders/processor.py:24-25) — one field pins
    all three loads.
    """
    from axolotl.loaders.model import ModelLoader
    from axolotl.loaders.tokenizer import load_tokenizer
    from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "base_model": config.model.hf_id,
            "revision_of_model": config.model.revision,
            "sequence_len": config.data.max_seq_len,
            "bf16": True,
            "load_in_4bit": False,
            "adapter": None if config.peft.mode == "full" else config.peft.mode,
            "lora_r": config.peft.r or None,
            "lora_alpha": config.peft.alpha or None,
            "lora_dropout": config.peft.dropout,
            "lora_target_linear": config.peft.mode != "full",
            # The four keys axolotl's schema requires before a cfg is a cfg, every
            # one of them read from this study's own config rather than invented.
            # `validate_config` fetches nothing, so naming the real repo costs no
            # download.
            "micro_batch_size": config.train.batch_size,
            "gradient_accumulation_steps": config.train.grad_accum,
            "learning_rate": config.optim.lr,
            "datasets": [{"path": config.data.repo_id, "type": "completion"}],
        }
    )
    prepare_plugins(cfg)
    cfg = validate_config(cfg)
    normalize_config(cfg)
    tokenizer = load_tokenizer(cfg)
    model, _ = ModelLoader(cfg, tokenizer).load()
    model.to(device)
    return model, tokenizer


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import axolotl

    report.add_version(axolotl)
    steps.patch_axes(config, report)

    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        # The cfg and the validate/normalize order live in `load` above, which is
        # also what a timing run takes. Two copies of that order is how the
        # inverted one survived a whole campaign.
        model, tokenizer = load(config, device, {})
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
    model = steps.verify_axes(model, config, device, "axolotl", report)

    tokenized: dict[str, torch.Tensor] = {}
    side = config.model.padding_side

    report.run(
        "padding_side_alignment",
        lambda: steps.padding_side_alignment(
            tokenizer, side, config.model.hf_id, config.model.revision
        ),
    )

    def _infonce_backward() -> dict[str, Any]:
        # ADAPTERS["axolotl"].required_step_context is the one declaration of this
        # regime; a timing run enters it the same way at scripts/bench.py:278. On a
        # non-CUDA host this raises UnappliedAxis, recorded as an ordinary failed check.
        required = ADAPTERS["axolotl"].required_step_context
        with axes.step_context(config, required):
            return steps.infonce_backward(model, tokenized, config.loss.temperature, side)

    if report.run("text_tokenize", lambda: steps.tokenize_text(tokenizer, device, tokenized, side))[
        0
    ]:
        report.run("infonce_backward", _infonce_backward)
    else:
        report.skip("infonce_backward", "tokenization failed")
