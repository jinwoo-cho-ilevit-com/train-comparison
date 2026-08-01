"""Pooling and loss.

The left-padding cases exist because their absence is what let the defect ship:
`last_token_pool` claimed in a comment to handle both padding sides while only
ever being tested with right padding, and gemma-4-E2B — the one model that pads
left — is also the one model with no probe run against it.
"""

from __future__ import annotations

import pytest
import torch

from trainbench.embedding import info_nce, last_token_pool

# Positions 0..3 are distinguishable, so a wrong index is visible in the value.
HIDDEN = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])


def test_right_padding_pools_the_last_content_position():
    mask = torch.tensor([[1, 1, 0, 0]])

    pooled = last_token_pool(HIDDEN, mask, padding_side="right")

    assert torch.equal(pooled, torch.tensor([[2.0, 2.0]]))


def test_left_padding_pools_the_last_content_position():
    """The regression. Summing the mask gives index 1 here; the content ends at 3."""
    mask = torch.tensor([[0, 0, 1, 1]])

    pooled = last_token_pool(HIDDEN, mask, padding_side="left")

    assert torch.equal(pooled, torch.tensor([[4.0, 4.0]]))


def test_left_padding_pools_per_row_not_per_batch():
    """Rows of different length must each pool their own last position, which for
    left padding is the same index — the check is that the values differ by row."""
    hidden = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    mask = torch.tensor([[0, 1, 1], [0, 0, 1]])

    pooled = last_token_pool(hidden, mask, padding_side="left")

    assert torch.equal(pooled, torch.tensor([[3.0], [6.0]]))


def test_right_padding_pools_per_row():
    hidden = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    pooled = last_token_pool(hidden, mask, padding_side="right")

    assert torch.equal(pooled, torch.tensor([[2.0], [4.0]]))


def test_unpadded_batch_agrees_across_both_sides():
    """With nothing padded the two branches must land on the same position. A
    disagreement here would mean one of them is off by one."""
    mask = torch.ones(1, 4, dtype=torch.long)

    left = last_token_pool(HIDDEN, mask, padding_side="left")
    right = last_token_pool(HIDDEN, mask, padding_side="right")

    assert torch.equal(left, right)
    assert torch.equal(left, torch.tensor([[4.0, 4.0]]))


def test_padding_side_is_required():
    """No default: defaulting to right is exactly the assumption that shipped."""
    with pytest.raises(TypeError):
        last_token_pool(HIDDEN, torch.ones(1, 4, dtype=torch.long))


def test_unknown_padding_side_is_refused_not_guessed():
    with pytest.raises(ValueError, match="padding_side"):
        last_token_pool(HIDDEN, torch.ones(1, 4, dtype=torch.long), padding_side="middle")


def test_info_nce_is_lower_when_pairs_align():
    aligned = torch.eye(4)
    shuffled = torch.eye(4).flip(0)

    assert info_nce(aligned, aligned, 0.02) < info_nce(aligned, shuffled, 0.02)
