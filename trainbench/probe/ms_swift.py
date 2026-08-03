"""Probe for ms-swift.

ms-swift advertises infonce embedding training and Qwen3-VL/Gemma4 support, so
this probe checks its loader accepts each model and that a contrastive step runs
through the model it returns.

`get_model_processor` owns the `from_pretrained` call, so `axes.load_kwargs`
(attention implementation, quantisation config) has nowhere to go here. It is
left unapplied rather than guessed at a keyword: the capture side then reads
whatever attention the model was really built with and reports the mismatch,
which is the correct outcome for an axis this path cannot honour.
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench.config_schema import BenchConfig
from trainbench.probe import steps
from trainbench.probe.types import ProbeReport


def load(config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """`get_model_processor` owns the `from_pretrained` call, so `load_kwargs` has
    nowhere to go here and is left unapplied rather than guessed at a keyword.

    `revision` is a named parameter of `get_model_processor` itself (ms-swift
    4.4.2 swift/model/register.py:525), threaded into `get_model_info_meta(...,
    revision=revision, ...)` at :603, which is what resolves the local snapshot
    both the model and the processor load from — so passing it here pins both.
    """
    from swift import get_model_processor

    model, processor = get_model_processor(config.model.hf_id, revision=config.model.revision)
    model.to(device)
    return model, processor


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import swift

    report.add_version(swift)
    steps.patch_axes(config, report)

    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        # `load` above is the one definition of this call, so a timing run and this
        # probe cannot end up loading two different models.
        model, processor = load(config, device, {})
        loaded["model"] = model
        loaded["processor"] = processor
        return {"model_class": type(model).__name__, "processor_class": type(processor).__name__}

    if not report.run("get_model_processor", _load)[0]:
        report.skip("infonce_backward", "model did not load")
        return

    model, processor = loaded["model"], loaded["processor"]
    model = steps.verify_axes(model, config, device, "ms_swift", report)

    def _template() -> dict[str, Any]:
        from swift import get_template

        template = get_template(processor)
        return {"template_class": type(template).__name__}

    report.run("get_template", _template)

    tokenized: dict[str, torch.Tensor] = {}
    side = config.model.padding_side

    report.run(
        "padding_side_alignment",
        lambda: steps.padding_side_alignment(
            processor, side, config.model.hf_id, config.model.revision
        ),
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
            config.model.prompt_format,
        ),
    )
