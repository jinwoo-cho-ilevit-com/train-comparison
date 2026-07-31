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


def encode(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pooled embedding from whatever hidden states the model exposes."""
    output = model(**batch, output_hidden_states=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(output, "hidden_states", None)
        hidden = hidden[-1] if hidden else output[0]
    return last_token_pool(hidden, batch["attention_mask"])


def text_batch(processor: Any, device: torch.device) -> dict[str, torch.Tensor]:
    texts = [q for q, _ in PROBE_PAIRS] + [d for _, d in PROBE_PAIRS]
    batch = processor(text=texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in batch.items()}


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


def visual_token_count(processor: Any, model: Any, device: torch.device) -> dict[str, Any]:
    """How many tokens one fixed image costs on this model.

    All three models use patch_size 16 but differ in spatial merge and pooling, so
    the same image is not the same cost. Speed comparisons are meaningless until
    this is pinned per model.
    """
    batch = image_batch(processor, device)
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("model config has no image_token_id")
    per_sample = (batch["input_ids"] == image_token_id).sum(dim=1).tolist()
    return {
        "image_size": list(PROBE_IMAGE_SIZE),
        "image_token_id": int(image_token_id),
        "visual_tokens_per_image": per_sample[0] if per_sample else None,
        "total_seq_len": int(batch["input_ids"].shape[1]),
    }


def infonce_backward(model: Any, batch: dict[str, torch.Tensor], temperature: float) -> dict:
    """One contrastive training step.

    This is the check that matters for framework support: patching that works for
    a language-modelling loss can still break when the loss is contrastive over
    pooled embeddings, because no LM head is involved.
    """
    model.train()
    pooled = encode(model, batch)
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
