"""Pooling and contrastive loss.

Defined once and imported by both the probe and the trainer. Re-implementing
pooling "similarly" for a second code path is the usual source of train/serve
skew (convention 07).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    padding_side: str,
) -> torch.Tensor:
    """Embedding = last non-padded position, the convention these decoder-based
    embedding models are trained with.

    hidden_states: (batch, seq, dim), attention_mask: (batch, seq).

    `padding_side` comes from `model.padding_side` (config), not from `arch`: only
    gemma-4 pads left, and branching on the architecture would hide that fact from
    whoever reads this function. It has no default — the earlier version assumed
    right padding while claiming to handle both, and returned position 1 instead of
    3 for the mask [0,0,1,1]. A default would let that mistake back in silently.
    """
    if padding_side == "right":
        # Padding sits after the content, so the last attended index is length - 1.
        index = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_index, index]
    if padding_side == "left":
        # Padding sits before the content, so every row ends on real content.
        return hidden_states[:, -1]
    raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")


def info_nce(
    queries: torch.Tensor,
    documents: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """In-batch-negatives contrastive loss (MultipleNegativesRankingLoss).

    Row i of `queries` is positive with row i of `documents`; every other document
    in the batch is a negative. This is why batch size dominates embedding
    training quality — and why it dominates the optimization picture here.
    """
    queries = F.normalize(queries, dim=-1)
    documents = F.normalize(documents, dim=-1)
    logits = queries @ documents.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
