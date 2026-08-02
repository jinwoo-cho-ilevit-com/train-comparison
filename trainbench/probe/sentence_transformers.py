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
from trainbench.loader import AdapterRefusal
from trainbench.probe import steps
from trainbench.probe.fixtures import PROBE_PAIRS
from trainbench.probe.types import ProbeReport


def processor_of(model: Any) -> Any:
    """The HF processor ST loaded, which is not the `SentenceTransformer` itself.

    `AdapterOut.processor` is called as an HF processor — `trainbench/collate.py`
    does `processor(text=..., return_tensors=...)` and reads `chat_template` off
    it — and `SentenceTransformer.forward(input, **kwargs)` answers neither.
    ST holds the real one inside its first module and republishes it as
    `.processor`/`.tokenizer` (sentence-transformers 5.6.1 base/model.py:1505,
    :1524 -> base/modules/transformer.py:671, :902).
    """
    for name in ("processor", "tokenizer"):
        found = getattr(model, name, None)
        if found is not None:
            return found
    raise AdapterRefusal(
        f"sentence_transformers built a {type(model).__name__} that publishes neither "
        "'processor' nor 'tokenizer', so this harness has nothing to tokenise a batch "
        "with; handing back the model itself would fail inside the first collate instead"
    )


def load(config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """The build `trainbench/loader.py` takes for a timing run.

    `model_kwargs` is forwarded to `AutoModel.from_pretrained` on the torch
    backend, so this is one of the two paths where a load-time axis can be
    honoured.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.model.hf_id,
        device=str(device),
        revision=config.model.revision,
        model_kwargs=load_kwargs,
    )
    return model, processor_of(model)


def run(config: BenchConfig, device: torch.device, report: ProbeReport) -> None:
    import sentence_transformers

    report.add_version(sentence_transformers)
    steps.patch_axes(config, report)
    # Outside `_load` for the reason steps.load_kwargs gives: a refused axis here
    # used to be recorded as a framework that cannot load this model, and this
    # path has the extra way of getting there — `BitsAndBytesConfig` is imported
    # on it, so an image without bitsandbytes failed the load with an ImportError.
    load_kwargs = steps.load_kwargs(config, report)

    loaded: dict[str, Any] = {}

    def _load() -> dict[str, Any]:
        # `load` above, so the probe and a timing run cannot build two different
        # models out of the same config.
        model, _ = load(config, device, load_kwargs)
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
        #
        # The step is ST's own — it pools inside its module, so `steps.encode` is
        # not the path here — but the evidence it has to produce is the same one,
        # which is why the counting and the refusal are `steps.training_step_evidence`.
        # This path used to return `params_with_grad` alone: the guard that caught
        # three frozen unsloth cells was not on it, and a fully frozen model here
        # would have passed. `BaseModel` is an `nn.Sequential` (sentence-transformers
        # 5.6.1 base/model.py:50), so `model.parameters()` reaches the backbone and
        # any adapter inside it, which is what those counts are of.
        features = model.tokenize(texts)
        features = {k: v.to(device) if hasattr(v, "to") else v for k, v in features.items()}
        pooled = model(features)["sentence_embedding"]
        half = pooled.shape[0] // 2
        loss = info_nce(pooled[:half], pooled[half:], config.loss.temperature)
        loss.backward()
        return steps.training_step_evidence(model, loss)

    report.run("mnrl_backward", _backward)

    def _cached_loss_available() -> dict[str, Any]:
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

        CachedMultipleNegativesRankingLoss(model, mini_batch_size=config.loss.mini_batch or 8)
        return {"available": True}

    # GradCache is the axis where reported overhead disagrees (20% vs 2-2.4x), so
    # its availability per model matters.
    report.run("cached_mnrl_constructs", _cached_loss_available)
