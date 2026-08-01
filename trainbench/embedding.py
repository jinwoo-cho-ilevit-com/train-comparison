"""Pooling and contrastive loss.

Defined once and imported by both the probe and the trainer. Re-implementing
pooling "similarly" for a second code path is the usual source of train/serve
skew (convention 07).

`align_padding_side` lives here rather than with the probe steps for the same
reason: it is the tokeniser-side half of the assumption `last_token_pool` makes,
and the measurement harness needs it as much as the probe does.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

PADDING_SIDES = ("left", "right")


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

    The mask is checked against the declared side rather than trusted. Both
    branches used to pool a PAD position, silently and without an exception, when
    the tokeniser padded the other way: right pooled `sum-1`, which for [0,0,1,1]
    is a PAD, and left read the last column, which for [1,1,0,0] is a PAD. Nothing
    downstream can tell a PAD embedding from a real one, so the disagreement has
    to stop the run here.
    """
    if padding_side not in PADDING_SIDES:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")

    mask = attention_mask.bool()
    lengths = mask.sum(dim=1)
    if not bool((lengths > 0).all()):
        empty = (lengths == 0).nonzero().flatten().tolist()
        raise ValueError(
            f"rows {empty[:8]} have no attended token, so they have no last token to pool; "
            "an all-padding row would otherwise be pooled at index 0, which is a PAD."
        )

    positions = torch.arange(mask.size(1), device=mask.device)
    if padding_side == "right":
        expected = positions < lengths[:, None]
    else:
        expected = positions >= (mask.size(1) - lengths)[:, None]
    if not torch.equal(mask, expected):
        wrong = (mask != expected).any(dim=1).nonzero().flatten().tolist()
        raise ValueError(
            f"attention_mask is not {padding_side}-padded on rows {wrong[:8]}; "
            f"padding_side={padding_side!r} came from config.model.padding_side while the "
            "tokeniser padded the other way, and pooling would have returned a PAD embedding."
        )

    if padding_side == "right":
        # Padding sits after the content, so the last attended index is length - 1.
        batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_index, lengths - 1]
    # Padding sits before the content, so every row ends on real content — which
    # the check above is what makes true rather than assumed.
    return hidden_states[:, -1]


def packed_last_token_pool(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Embedding per sequence when the batch is packed into one row.

    `last_token_pool` asserts one sequence per row with contiguous padding, and a
    packed batch breaks that contract head-on: there is no padding, and one row
    holds every sequence end to end. That assertion is not weakened to let packing
    through — it is what stops a PAD embedding from being pooled — so packing gets
    its own entry point and the boundaries have to be carried explicitly.

    `cu_seqlens` is the cumulative-sequence-length vector varlen attention kernels
    already take (`[0, len_0, len_0+len_1, ...]`), rather than a second, private
    description of the same boundaries: two spellings of where a sequence ends is
    how a packed batch gets pooled at the wrong token while both halves look
    internally consistent.

    hidden_states: (total, dim), or (1, total, dim) as the model returns it.
    Returns (sequences, dim), in packing order.
    """
    if hidden_states.dim() == 3:
        if hidden_states.size(0) != 1:
            raise ValueError(
                f"a packed batch is one row, got {hidden_states.size(0)}; "
                "hidden_states shaped (batch>1, seq, dim) is a padded batch and belongs in "
                "last_token_pool, where the padding side is checked."
            )
        hidden_states = hidden_states[0]
    if hidden_states.dim() != 2:
        raise ValueError(
            "hidden_states must be (total, dim) or (1, total, dim), got "
            f"{tuple(hidden_states.shape)}"
        )
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(
            f"cu_seqlens must be a 1-D vector of at least two offsets, got "
            f"{tuple(cu_seqlens.shape)}; without a start and an end no sequence is delimited."
        )
    offsets = cu_seqlens.to(dtype=torch.long, device=hidden_states.device)
    if int(offsets[0]) != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {int(offsets[0])}")
    lengths = offsets[1:] - offsets[:-1]
    if not bool((lengths > 0).all()):
        empty = (lengths <= 0).nonzero().flatten().tolist()
        raise ValueError(
            f"sequences {empty[:8]} are empty or out of order in cu_seqlens; an empty "
            "sequence has no last token to pool, and pooling it would read the previous "
            "sequence's last token under its name."
        )
    total = hidden_states.size(0)
    if int(offsets[-1]) != total:
        raise ValueError(
            f"cu_seqlens ends at {int(offsets[-1])} but the packed batch holds {total} "
            "tokens; the boundaries do not describe this batch, so every pooled position "
            "after the first disagreement would belong to the wrong sequence."
        )
    # The last token of sequence i sits at (start of i+1) - 1, which is what makes
    # the whole of this function the same convention as `last_token_pool`.
    return hidden_states[offsets[1:] - 1]


def _padding_side_holders(processor: Any) -> list[tuple[str, Any]]:
    """Every object on a processor that declares a padding side.

    A processor keeps it on `.tokenizer`; a bare tokenizer (axolotl hands one
    back) keeps it on itself; some processors carry their own copy as well, and a
    copy left on the old value is a copy that can win at call time.
    """
    holders = []
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) is not None:
        holders.append(("tokenizer", tokenizer))
    if getattr(processor, "padding_side", None) is not None:
        holders.append(("processor", processor))
    return holders


def align_padding_side(processor: Any, padding_side: str) -> dict[str, Any]:
    """Make the tokeniser pad on the side `last_token_pool` is told to expect.

    Nothing used to do this. `config.model.padding_side` reached the pooling
    function while the processor kept whatever its checkpoint declared, so the
    config value was a claim rather than a constraint — and gemma-4-E2B, the only
    model that pads left, is the one model no probe has run against.

    The config wins because it is the audited value: `audit_plan.py`'s
    `model-spec` check compares it against `docs/model-spec.yaml` value for value,
    which is what makes it reviewable. What each holder declared beforehand is
    returned rather than dropped, so a checkpoint that has drifted away from the
    spec stays visible in the result instead of being quietly overwritten.
    """
    if padding_side not in PADDING_SIDES:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    holders = _padding_side_holders(processor)
    if not holders:
        raise ValueError(
            f"{type(processor).__name__} declares no padding_side, so which side it pads "
            "cannot be established; last-token pooling would read a PAD position without "
            "anything saying so."
        )

    declared = {}
    for name, holder in holders:
        declared[name] = str(holder.padding_side)
        holder.padding_side = padding_side
    # Read back: a processor that stores the attribute somewhere else, or refuses
    # the write, would otherwise leave this reporting a change that never happened.
    stuck = sorted(name for name, holder in holders if str(holder.padding_side) != padding_side)
    if stuck:
        raise ValueError(
            f"padding_side stayed {[declared[n] for n in stuck]} on {stuck} after being set to "
            f"{padding_side!r}; this processor does not take the setting."
        )
    return {
        "padding_side": padding_side,
        "declared_before": declared,
        "disagreed": sorted(name for name, value in declared.items() if value != padding_side),
    }


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
