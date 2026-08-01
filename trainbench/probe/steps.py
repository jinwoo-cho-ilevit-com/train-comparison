"""Check bodies shared by every framework probe.

Each framework loads models differently, but the questions asked of the loaded
model are the same, and asking them identically is what makes the resulting
matrix comparable.
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench import applied, axes
from trainbench.config_schema import BenchConfig
from trainbench.embedding import align_padding_side, info_nce, last_token_pool
from trainbench.probe.fixtures import PROBE_IMAGE_SIZE, PROBE_PAIRS, probe_image
from trainbench.probe.types import ProbeReport


def dtype_for(device: torch.device) -> torch.dtype:
    # bf16 is the training dtype but is not dependable on CPU/MPS, and a probe only
    # answers "does it run".
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def patch_axes(config: BenchConfig, report: ProbeReport) -> None:
    """Axes that have to be applied before any model exists.

    Kernel libraries replace transformers classes, so a model built first is a
    model the patch never reached (docs/CONTRACTS.md §2). Every adapter goes
    through here, not just the native one — an adapter that skips it would report
    a kernel axis it never asked for.
    """
    report.run("axes_patch", lambda: {"applied": axes.patch(config)})


def verify_axes(
    model: Any,
    config: BenchConfig,
    device: torch.device,
    framework: str,
    report: ProbeReport,
) -> Any:
    """Build the rest of the run around `model`, then read back what took effect.

    Returns the model to use afterwards: `assemble` may hand back a different
    object, because peft, `torch.compile` and FSDP all replace the model rather
    than mutate it.

    `framework` is a literal passed in by the calling adapter and never read from
    the config. The config records what was requested; this literal is the
    evidence of which code path actually ran, which is the entire reason
    applied.py exists (docs/CONTRACTS.md §2).

    A failure inside `assemble` leaves `built` holding the model alone, so the
    axes it would have covered come back undetermined rather than unexamined.
    """
    built = applied.Built(model=model)

    def _assemble() -> dict[str, Any]:
        nonlocal built
        built, names = axes.assemble(model, config, device, framework=framework)
        return {"applied": names}

    report.run("axes_assemble", _assemble)
    report.applied = applied.capture(built, config)
    # Records the verdict rather than aborting: a probe answers "does it run", and
    # purpose=probe is not enforced. A reportable purpose raises here, which is the
    # point — the same call in the measurement harness stops the run.
    report.run("axes_verified", lambda: _verified(report.applied, config))
    return built.model if built.model is not None else model


def _verified(state: applied.AppliedState, config: BenchConfig) -> dict[str, Any]:
    applied.assert_matches(state, config)
    return state.to_dict()


def padding_side_alignment(processor: Any, padding_side: str) -> dict[str, Any]:
    """Force the tokeniser onto the configured padding side, and fail if it had to.

    Alignment happens first so that whatever runs after this check pools a real
    token either way; the raise is what makes the disagreement loud. A checkpoint
    that pads differently from `docs/model-spec.yaml` is a spec that has gone
    stale, and the run is the only place that can notice.
    """
    detail = align_padding_side(processor, padding_side)
    if detail["disagreed"]:
        raise ValueError(
            f"{detail['disagreed']} declared padding_side {detail['declared_before']} but "
            f"config.model.padding_side is {padding_side!r}; it has been forced onto the "
            "configured side, and docs/model-spec.yaml no longer matches this checkpoint."
        )
    return detail


def encode(model: Any, batch: dict[str, torch.Tensor], padding_side: str) -> torch.Tensor:
    """Pooled embedding from whatever hidden states the model exposes.

    `padding_side` is threaded through from `config.model.padding_side` rather than
    read off the processor: the config is what the audit compares against
    docs/model-spec.yaml. It is not merely a claim any more — every batch built
    here goes through `align_padding_side` first, and `last_token_pool` rejects a
    mask that disagrees with the declared side.
    """
    output = model(**batch, output_hidden_states=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(output, "hidden_states", None)
        hidden = hidden[-1] if hidden else output[0]
    return last_token_pool(hidden, batch["attention_mask"], padding_side=padding_side)


def text_batch(processor: Any, device: torch.device, padding_side: str) -> dict[str, torch.Tensor]:
    align_padding_side(processor, padding_side)
    texts = [q for q, _ in PROBE_PAIRS] + [d for _, d in PROBE_PAIRS]
    batch = processor(text=texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in batch.items()}


def tokenize_text(
    processor: Any, device: torch.device, into: dict[str, torch.Tensor], padding_side: str
) -> dict[str, Any]:
    """Tokenise the probe pairs into `into` and return only JSON-safe detail.

    Every adapter needs the tensors afterwards but whatever a check returns becomes
    its `detail` and is serialised, so the tensors go into the caller's dict and
    only shapes come back. Written once here because five adapters had the same
    closure, and the copies had already drifted in what they reported.
    """
    into.update(text_batch(processor, device, padding_side))
    return {
        "keys": sorted(into),
        "input_ids_shape": list(into["input_ids"].shape),
        "padding_side": padding_side,
    }


def image_batch(processor: Any, device: torch.device, padding_side: str) -> dict[str, Any]:
    """Multimodal batch.

    The text must carry the model's image placeholder tokens; passing raw text
    alongside images silently produces zero image tokens against N image features,
    and the forward pass then fails on the mismatch. `apply_chat_template` is what
    inserts the placeholders, so it is not optional here.
    """
    align_padding_side(processor, padding_side)
    image = probe_image()
    texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for q, _ in PROBE_PAIRS
    ]
    batch = processor(text=texts, images=[image] * len(texts), return_tensors="pt", padding=True)
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


# Where the image placeholder id is declared, newest name first. transformers
# renamed `image_token_index` to `image_token_id`, and some VLM configs keep it on
# the text sub-config instead of the top level. Reading only the first name makes a
# model whose config uses another one look like it has no image tokens at all.
IMAGE_TOKEN_ID_FIELDS = ("image_token_id", "image_token_index")


def image_token_id(model: Any) -> tuple[int, str]:
    """The image placeholder token id and where it was found."""
    configs: list[tuple[str, Any]] = [("config", model.config)]
    get_text_config = getattr(model.config, "get_text_config", None)
    if callable(get_text_config):
        try:
            configs.append(("text_config", get_text_config()))
        except Exception:  # noqa: BLE001 - framework wrappers expose odd configs
            # Swallowed on purpose: the answer we owe the caller is whether an image
            # token id exists, and letting an accessor's failure surface instead
            # would report a missing id as a broken config.
            pass

    for where, config in configs:
        for field in IMAGE_TOKEN_ID_FIELDS:
            value = getattr(config, field, None)
            if value is not None:
                return int(value), f"{where}.{field}"
    raise ValueError(
        f"no image token id on this model config; looked for {list(IMAGE_TOKEN_ID_FIELDS)} "
        f"on {[where for where, _ in configs]}"
    )


def pad_token_id(processor: Any) -> int | None:
    """The processor's pad token id, wherever it keeps it. None if it has none."""
    for holder in (getattr(processor, "tokenizer", None), processor):
        value = getattr(holder, "pad_token_id", None)
        if value is not None:
            return int(value)
    return None


def visual_token_count(
    processor: Any,
    model: Any,
    device: torch.device,
    padding_side: str,
    tokens_per_image: int | None,
) -> dict[str, Any]:
    """How many tokens one fixed image costs on this model.

    All three models use patch_size 16 but differ in spatial merge and pooling, so
    the same image is not the same cost. Speed comparisons are meaningless until
    this is pinned per model.

    Every row of the batch carries the same image, so the four gates below are the
    ways a wrong id can still produce a plausible-looking number:

    * an id equal to the pad token counts padding. `0 < n < seq_len` accepts that
      happily, and the resulting count is a property of the batch shape rather
      than of the model
    * counts that differ per sample mean the id matched something the rows do not
      share; grading `per_sample[0]` alone accepted `[280, 279]`
    * a count of 0, or one filling the sequence, means the id or the chat template
      is wrong — `apply_chat_template` always emits role and text tokens around
      the placeholders
    * `config.model.tokens_per_image`, where the model declares one, is the
      answer this is supposed to reproduce. gemma4 fixes it at 280 regardless of
      resolution, so a different measurement is a disagreement to resolve, not a
      measurement to publish. The Qwen models are pixel-proportional and declare
      None, which is why this is a comparison and not a lookup.

    A wrong number here silently rescales every tokens/s figure that divides by it.
    """
    batch = image_batch(processor, device, padding_side)
    token_id, source = image_token_id(model)
    input_ids = batch["input_ids"]
    per_sample = (input_ids == token_id).sum(dim=1).tolist()
    total_seq_len = int(input_ids.shape[1])
    if not per_sample:
        raise ValueError("the probe batch is empty, so nothing was counted")
    count = per_sample[0]

    pad_id = pad_token_id(processor)
    if pad_id is not None and token_id == pad_id:
        raise ValueError(
            f"image token id {token_id} from {source} is this processor's pad token id; "
            f"the {count} tokens counted are padding, not image placeholders."
        )
    if len(set(per_sample)) != 1:
        raise ValueError(
            f"visual token counts disagree across samples: {per_sample}. Every row of this "
            f"batch carries the same image, so token id {token_id} from {source} is matching "
            "something the rows do not share."
        )
    if not 0 < count < total_seq_len:
        raise ValueError(
            f"visual token count {count} is outside 0 < n < {total_seq_len} for token id "
            f"{token_id} from {source}; the placeholder id or the chat template is wrong."
        )
    if tokens_per_image is not None and count != tokens_per_image:
        raise ValueError(
            f"measured {count} visual tokens but config.model.tokens_per_image declares "
            f"{tokens_per_image}; one of the two is wrong and every tokens/s figure divides "
            "by this number."
        )

    return {
        "image_size": list(PROBE_IMAGE_SIZE),
        "image_token_id": token_id,
        "image_token_id_source": source,
        "pad_token_id": pad_id,
        "visual_tokens_per_image": count,
        "visual_tokens_per_sample": per_sample,
        "declared_tokens_per_image": tokens_per_image,
        "total_seq_len": total_seq_len,
    }


def infonce_backward(
    model: Any, batch: dict[str, torch.Tensor], temperature: float, padding_side: str
) -> dict:
    """One contrastive training step.

    This is the check that matters for framework support: patching that works for
    a language-modelling loss can still break when the loss is contrastive over
    pooled embeddings, because no LM head is involved.
    """
    model.train()
    pooled = encode(model, batch, padding_side)
    half = pooled.shape[0] // 2
    loss = info_nce(pooled[:half], pooled[half:], temperature)
    loss.backward()
    with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach()),
        "params_with_grad": with_grad,
        "trainable_params": trainable,
    }
