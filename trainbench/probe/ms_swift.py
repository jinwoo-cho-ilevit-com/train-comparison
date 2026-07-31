"""Probe for ms-swift.

ms-swift advertises infonce embedding training and Qwen3-VL/Gemma4 support, so
this probe checks its loader accepts each model and that a contrastive step runs
through the model it returns.
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def run(config: BenchConfig, device: torch.device) -> ProbeReport:
    report = ProbeReport(framework="ms_swift", model=config.model.name)
    import swift

    report.add_version(swift)

    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        from swift import get_model_processor

        model, processor = get_model_processor(config.model.hf_id)
        loaded["model"] = model
        loaded["processor"] = processor
        return {"model_class": type(model).__name__, "processor_class": type(processor).__name__}

    if not report.run("get_model_processor", _load)[0]:
        report.skip("infonce_backward", "model did not load")
        return report

    model, processor = loaded["model"], loaded["processor"]
    model.to(device)

    def _template() -> dict[str, Any]:
        from swift import get_template

        template = get_template(processor)
        return {"template_class": type(template).__name__}

    report.run("get_template", _template)

    tokenized: dict[str, torch.Tensor] = {}

    def _tokenize() -> dict[str, Any]:
        tokenized.update(steps.text_batch(processor, device))
        return {"input_ids_shape": list(tokenized["input_ids"].shape)}

    if report.run("text_tokenize", _tokenize)[0]:
        report.run(
            "infonce_backward",
            lambda: steps.infonce_backward(model, tokenized, config.loss.temperature),
        )
    else:
        report.skip("infonce_backward", "tokenization failed")

    report.run("visual_tokens", lambda: steps.visual_token_count(processor, model, device))
    return report
