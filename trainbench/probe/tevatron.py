"""Probe for Tevatron.

The installed distribution reports version 0.0.1 from git HEAD, which does not
match the 2.0 described in the paper, so the first question is what this package
actually is. The module layout is recorded rather than assumed: a wrong guess at
the API would be recorded as "unsupported" when the real answer is "probed wrong".

For the same reason `axes.load_kwargs` is not forced through `DenseModel.load`:
what that call forwards to `from_pretrained` is exactly the sort of thing this
probe exists to find out, and a wrong keyword would be recorded as tevatron
refusing the model. The attention axis is left to the capture side, which reads
it off the built model and reports the mismatch.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

import torch

from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def plant_pad_token_id(hf_config: Any) -> dict[str, Any]:
    """Put `pad_token_id` on the top-level config, because tevatron reads it there.

    `DenseModel.load` does `base_model.config.pad_token_id` directly rather than
    through `getattr` (tevatron dd063104 retriever/modeling/encoder.py:167), and
    transformers 5.14.1 declares that field only on the text sub-config: none of
    `Qwen3VLConfig`, `Qwen3_5Config` or `Gemma4Config` carries it at the top level
    and none of them has an `attribute_map` that would forward it, so the read
    raises `AttributeError` before its `is None` comparison. That is where all
    three tevatron cells of the second campaign died. Upstream cannot be fixed
    from here and `@strict` blocks reads, not writes, so the attribute is planted
    before the load.

    This copy is a shim, not the model's value — the model reads the text
    sub-config — so nothing should read it back.

    A `None` is planted as `None` rather than filled in: upstream's next line is
    `if ... is None: ... = 0`, and letting its own default land keeps this out of
    a decision that belongs to tevatron. A config that already declares the field
    is left alone.
    """
    text = hf_config
    get_text_config = getattr(hf_config, "get_text_config", None)
    if callable(get_text_config):
        text = get_text_config()
    if hasattr(hf_config, "pad_token_id"):
        return {
            "pad_token_id_planted": False,
            "pad_token_id": hf_config.pad_token_id,
            "pad_token_id_source": type(hf_config).__name__,
        }
    value = getattr(text, "pad_token_id", None)
    hf_config.pad_token_id = value
    return {
        "pad_token_id_planted": True,
        "pad_token_id": value,
        "pad_token_id_source": type(text).__name__,
    }


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import tevatron

    report.add_version(tevatron)
    steps.patch_axes(config, report)

    def _layout() -> dict[str, Any]:
        submodules = sorted(
            m.name for m in pkgutil.iter_modules(tevatron.__path__, prefix="tevatron.")
        )
        return {"submodules": submodules, "path": list(tevatron.__path__)}

    report.run("module_layout", _layout)

    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        from transformers import AutoConfig

        modeling = importlib.import_module("tevatron.retriever.modeling")
        dense = modeling.DenseModel
        hf_config = AutoConfig.from_pretrained(config.model.hf_id, revision=config.model.revision)
        shim = plant_pad_token_id(hf_config)
        model = dense.load(
            config.model.hf_id,
            pooling="last",
            normalize=True,
            config=hf_config,
            revision=config.model.revision,
        )
        loaded["model"] = model
        return {"model_class": type(model).__name__, **shim}

    if not report.run("dense_model_load", _load)[0]:
        report.skip("infonce_backward", "model did not load")
        return

    model = loaded["model"]
    model.to(device)
    model = steps.verify_axes(model, config, device, "tevatron", report)

    def _tokenizer() -> dict[str, Any]:
        from transformers import AutoProcessor

        loaded["processor"] = AutoProcessor.from_pretrained(
            config.model.hf_id, revision=config.model.revision
        )
        return {"processor_class": type(loaded["processor"]).__name__}

    if not report.run("processor_load", _tokenizer)[0]:
        report.skip("infonce_backward", "processor did not load")
        return

    tokenized: dict[str, torch.Tensor] = {}
    side = config.model.padding_side

    report.run(
        "padding_side_alignment",
        lambda: steps.padding_side_alignment(
            loaded["processor"], side, config.model.hf_id, config.model.revision
        ),
    )

    if report.run(
        "text_tokenize", lambda: steps.tokenize_text(loaded["processor"], device, tokenized, side)
    )[0]:
        report.run(
            "infonce_backward",
            lambda: steps.infonce_backward(model, tokenized, config.loss.temperature, side),
        )
    else:
        report.skip("infonce_backward", "tokenization failed")
