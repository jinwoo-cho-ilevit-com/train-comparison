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


def apply_temperature(model: Any, config: BenchConfig) -> dict[str, Any]:
    """Fill the temperature `DenseModel.load` has no keyword for.

    `EncoderModel.__init__`'s default is 1.0 (dd06310 encoder.py:38) and `load`
    (encoder.py:159-165) forwards no `temperature`, so a loaded model scores at
    1.0 regardless of `config.loss.temperature` until something sets it. The
    override is recorded rather than silent: a tevatron cell measured at a
    temperature no other framework's cell used would corrupt the comparison if
    nothing said so.
    """
    before = model.temperature
    model.temperature = config.loss.temperature
    return {"temperature": model.temperature, "temperature_before_override": before}


def load_dense_model(config: BenchConfig) -> tuple[Any, dict[str, Any]]:
    """`DenseModel.load` with the shim planted first, and what the shim did."""
    from transformers import AutoConfig

    modeling = importlib.import_module("tevatron.retriever.modeling")
    hf_config = AutoConfig.from_pretrained(config.model.hf_id, revision=config.model.revision)
    shim = plant_pad_token_id(hf_config)
    model = modeling.DenseModel.load(
        config.model.hf_id,
        pooling="last",
        normalize=True,
        config=hf_config,
        revision=config.model.revision,
    )
    shim.update(apply_temperature(model, config))
    return model, shim


def load(config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """The build `trainbench/loader.py` takes for a timing run.

    `load_kwargs` is not forced through `DenseModel.load`: the same dict reaches
    `LoraConfig.from_pretrained` on the LoRA path (tevatron dd06310
    retriever/modeling/encoder.py:131, :170), so a transformers-only keyword would
    be refused there rather than here.
    """
    from transformers import AutoProcessor

    model, _ = load_dense_model(config)
    model.to(device)
    processor = AutoProcessor.from_pretrained(
        config.model.hf_id,
        revision=config.model.revision,
        **steps.pixel_budget_kwargs(config),
    )
    return model, processor


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
        model, shim = load_dense_model(config)
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
            config.model.hf_id,
            revision=config.model.revision,
            **steps.pixel_budget_kwargs(config),
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

    def _backward() -> dict[str, Any]:
        """tevatron's own training step, not the harness's.

        `EncoderModel.forward(query=, passage=)` encodes, pools, normalises,
        scores and computes InfoNCE itself (dd06310 encoder.py:52-87,
        dense.py:18-46), so `steps.encode`/`embedding.info_nce` never run here —
        `EncoderOutput.loss` is the whole step. The split mirrors
        `steps.infonce_backward`'s: the collate lays every query before every
        positive, so the first half of the tokenised batch is the query side.

        An empty `query` or `passage` dict is falsy and silently takes the eval
        branch (encoder.py:53-61), which leaves `loss=None`; refused here rather
        than left to surface as an `AttributeError` on `.backward()`.
        """
        model.train()
        half = tokenized["input_ids"].shape[0] // 2
        query = {k: v[:half] for k, v in tokenized.items()}
        passage = {k: v[half:] for k, v in tokenized.items()}
        output = model(query=query, passage=passage)
        if output.loss is None:
            raise ValueError(
                "EncoderModel.forward returned loss=None; query or passage was empty "
                "and the call took its inference branch instead of training"
            )
        output.loss.backward()
        detail = steps.training_step_evidence(model, output.loss)
        detail["temperature"] = model.temperature
        return detail

    if report.run(
        "text_tokenize", lambda: steps.tokenize_text(loaded["processor"], device, tokenized, side)
    )[0]:
        report.run("infonce_backward", _backward)
    else:
        report.skip("infonce_backward", "tokenization failed")
