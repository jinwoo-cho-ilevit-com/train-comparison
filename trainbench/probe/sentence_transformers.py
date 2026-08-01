"""Probe for Sentence Transformers.

Qwen3-VL-Embedding-2B ships with a sentence-transformers config, so this path is
expected to work for it; the open question is the two generative VLMs, which have
no ST module layout and must fall back to default pooling.

No padding-side alignment here, unlike every other adapter: ST pools inside its
own module rather than through `last_token_pool`, so `config.model.padding_side`
is not the assumption in play, and forcing it onto ST's tokeniser would change an
input this probe is meant to observe untouched. What pooling ST actually chose is
already recorded as `has_module_layout`.
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench.config_schema import BenchConfig
from trainbench.embedding import info_nce
from trainbench.probe import steps
from trainbench.probe.fixtures import PROBE_PAIRS
from trainbench.probe.types import ProbeReport


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    report.add_version(sentence_transformers)
    steps.patch_axes(config, report)
    # Outside `_load` for the reason steps.load_kwargs gives: a refused axis here
    # used to be recorded as a framework that cannot load this model, and this
    # path has the extra way of getting there — `BitsAndBytesConfig` is imported
    # on it, so an image without bitsandbytes failed the load with an ImportError.
    load_kwargs = steps.load_kwargs(config, report)

    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        # `model_kwargs` is forwarded to `AutoModel.from_pretrained` on the torch
        # backend (SentenceTransformer.__init__), so this is the one framework
        # path where the load-time axes can be honoured rather than left to read
        # back as a mismatch.
        model = SentenceTransformer(
            config.model.hf_id,
            device=str(device),
            revision=config.model.revision,
            model_kwargs=load_kwargs,
        )
        loaded["model"] = model
        return {
            "modules": [type(m).__name__ for m in model],
            "embedding_dim": model.get_sentence_embedding_dimension(),
            # Present only when the checkpoint carries an ST module layout;
            # otherwise ST fell back to default pooling and the embedding may not
            # match what the model was trained to produce.
            "has_module_layout": len(list(model)) > 1,
        }

    if not report.run("sentence_transformer_load", _load)[0]:
        report.skip("encode", "model did not load")
        report.skip("mnrl_backward", "model did not load")
        return

    model = steps.verify_axes(loaded["model"], config, device, "sentence_transformers", report)
    texts = [q for q, _ in PROBE_PAIRS] + [d for _, d in PROBE_PAIRS]

    report.run(
        "encode",
        lambda: {"shape": list(model.encode(texts, convert_to_tensor=True).shape)},
    )

    def _backward() -> dict[str, Any]:
        # Uses the shared info_nce rather than ST's loss class so the loss is
        # identical across frameworks; comparing frameworks under different loss
        # implementations would not be a framework comparison.
        features = model.tokenize(texts)
        features = {k: v.to(device) if hasattr(v, "to") else v for k, v in features.items()}
        pooled = model(features)["sentence_embedding"]
        half = pooled.shape[0] // 2
        loss = info_nce(pooled[:half], pooled[half:], config.loss.temperature)
        loss.backward()
        with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        model.zero_grad(set_to_none=True)
        return {"loss": float(loss.detach()), "params_with_grad": with_grad}

    report.run("mnrl_backward", _backward)

    def _cached_loss_available() -> dict[str, Any]:
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

        CachedMultipleNegativesRankingLoss(model, mini_batch_size=config.loss.mini_batch or 8)
        return {"available": True}

    # GradCache is the axis where reported overhead disagrees (20% vs 2-2.4x), so
    # its availability per model matters.
    report.run("cached_mnrl_constructs", _cached_loss_available)
