"""Check bodies shared by every framework probe.

Each framework loads models differently, but the questions asked of the loaded
model are the same, and asking them identically is what makes the resulting
matrix comparable.
"""

from __future__ import annotations

from typing import Any

import torch

from trainbench.embedding import info_nce, last_token_pool
from trainbench.probe.fixtures import PROBE_IMAGE_SIZE, PROBE_PAIRS, probe_image


def dtype_for(device: torch.device) -> torch.dtype:
    # bf16 is the training dtype but is not dependable on CPU/MPS, and a probe only
    # answers "does it run".
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def encode(model: Any, batch: dict[str, torch.Tensor], padding_side: str) -> torch.Tensor:
    """Pooled embedding from whatever hidden states the model exposes.

    `padding_side` is threaded through from `config.model.padding_side` rather than
    read off the processor: the config is what the audit compares against
    docs/model-spec.yaml, and pooling the wrong index is invisible in the output.
    """
    output = model(**batch, output_hidden_states=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(output, "hidden_states", None)
        hidden = hidden[-1] if hidden else output[0]
    return last_token_pool(hidden, batch["attention_mask"], padding_side=padding_side)


def text_batch(processor: Any, device: torch.device) -> dict[str, torch.Tensor]:
    texts = [q for q, _ in PROBE_PAIRS] + [d for _, d in PROBE_PAIRS]
    batch = processor(text=texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in batch.items()}


def tokenize_text(
    processor: Any, device: torch.device, into: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Tokenise the probe pairs into `into` and return only JSON-safe detail.

    Every adapter needs the tensors afterwards but whatever a check returns becomes
    its `detail` and is serialised, so the tensors go into the caller's dict and
    only shapes come back. Written once here because five adapters had the same
    closure, and the copies had already drifted in what they reported.
    """
    into.update(text_batch(processor, device))
    return {
        "keys": sorted(into),
        "input_ids_shape": list(into["input_ids"].shape),
    }


def image_batch(processor: Any, device: torch.device) -> dict[str, Any]:
    """Multimodal batch.

    The text must carry the model's image placeholder tokens; passing raw text
    alongside images silently produces zero image tokens against N image features,
    and the forward pass then fails on the mismatch. `apply_chat_template` is what
    inserts the placeholders, so it is not optional here.
    """
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


def visual_token_count(processor: Any, model: Any, device: torch.device) -> dict[str, Any]:
    """How many tokens one fixed image costs on this model.

    All three models use patch_size 16 but differ in spatial merge and pooling, so
    the same image is not the same cost. Speed comparisons are meaningless until
    this is pinned per model.
    """
    batch = image_batch(processor, device)
    token_id, source = image_token_id(model)
    input_ids = batch["input_ids"]
    per_sample = (input_ids == token_id).sum(dim=1).tolist()
    total_seq_len = int(input_ids.shape[1])
    count = per_sample[0] if per_sample else 0

    # A count outside this range means the id is wrong, not that the model is
    # cheap or expensive — and a wrong number here silently rescales every
    # tokens/s figure that divides by it. `apply_chat_template` always emits role
    # and text tokens around the placeholders, so a batch that is all image tokens
    # means the id matched something else (padding is the usual culprit).
    if not 0 < count < total_seq_len:
        raise ValueError(
            f"visual token count {count} is outside 0 < n < {total_seq_len} for token id "
            f"{token_id} from {source}; the placeholder id or the chat template is wrong."
        )

    return {
        "image_size": list(PROBE_IMAGE_SIZE),
        "image_token_id": token_id,
        "image_token_id_source": source,
        "visual_tokens_per_image": count,
        "visual_tokens_per_sample": per_sample,
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
