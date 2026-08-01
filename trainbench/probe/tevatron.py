"""Probe for Tevatron.

The installed distribution reports version 0.0.1 from git HEAD, which does not
match the 2.0 described in the paper, so the first question is what this package
actually is. The module layout is recorded rather than assumed: a wrong guess at
the API would be recorded as "unsupported" when the real answer is "probed wrong".
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

import torch

from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import tevatron

    report.add_version(tevatron)

    def _layout() -> dict[str, Any]:
        submodules = sorted(
            m.name for m in pkgutil.iter_modules(tevatron.__path__, prefix="tevatron.")
        )
        return {"submodules": submodules, "path": list(tevatron.__path__)}

    report.run("module_layout", _layout)

    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        modeling = importlib.import_module("tevatron.retriever.modeling")
        dense = modeling.DenseModel
        model = dense.load(config.model.hf_id, pooling="last", normalize=True)
        loaded["model"] = model
        return {"model_class": type(model).__name__}

    if not report.run("dense_model_load", _load)[0]:
        report.skip("infonce_backward", "model did not load")
        return

    model = loaded["model"]
    model.to(device)

    def _tokenizer() -> dict[str, Any]:
        from transformers import AutoProcessor

        loaded["processor"] = AutoProcessor.from_pretrained(config.model.hf_id)
        return {"processor_class": type(loaded["processor"]).__name__}

    if not report.run("processor_load", _tokenizer)[0]:
        report.skip("infonce_backward", "processor did not load")
        return

    tokenized: dict[str, torch.Tensor] = {}
    side = config.model.padding_side

    if report.run(
        "text_tokenize", lambda: steps.tokenize_text(loaded["processor"], device, tokenized)
    )[0]:
        report.run(
            "infonce_backward",
            lambda: steps.infonce_backward(model, tokenized, config.loss.temperature, side),
        )
    else:
        report.skip("infonce_backward", "tokenization failed")
