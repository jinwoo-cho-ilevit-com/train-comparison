"""Pooling and contrastive loss.

Defined once and imported by both the probe and the trainer. Re-implementing
pooling "similarly" for a second code path is the usual source of train/serve
skew (convention 07).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Embedding = last non-padded position, the convention these decoder-based
    embedding models are trained with.

    hidden_states: (batch, seq, dim), attention_mask: (batch, seq).
    """
    # Works for both left- and right-padded batches: pick the last attended index.
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_index, lengths]


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
