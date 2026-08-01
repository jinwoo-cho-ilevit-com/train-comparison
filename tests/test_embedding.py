"""Pooling and loss.

The left-padding cases exist because their absence is what let the defect ship:
`last_token_pool` claimed in a comment to handle both padding sides while only
ever being tested with right padding, and gemma-4-E2B — the one model that pads
left — is also the one model with no probe run against it.
"""

from __future__ import annotations

import pytest
import torch

from trainbench.embedding import align_padding_side, info_nce, last_token_pool

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


def test_right_padding_refuses_a_left_padded_mask():
    """The reproduced defect. `sum-1` is index 1 here, which is a PAD: the config
    said right, the tokeniser padded left, and the pooled embedding was padding
    with no exception and no warning."""
    mask = torch.tensor([[0, 0, 1, 1]])

    with pytest.raises(ValueError, match="not right-padded"):
        last_token_pool(HIDDEN, mask, padding_side="right")


def test_left_padding_refuses_a_right_padded_mask():
    """The mirror case. The left branch reads the last column and never looks at the
    mask, so both existing left tests — whose masks end in 1 — could not catch it."""
    mask = torch.tensor([[1, 1, 0, 0]])

    with pytest.raises(ValueError, match="not left-padded"):
        last_token_pool(HIDDEN, mask, padding_side="left")


def test_a_row_with_no_attended_token_is_refused():
    """`clamp(min=0)` turned an empty row into index 0, which is a PAD."""
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    hidden = HIDDEN.expand(2, 4, 2)

    with pytest.raises(ValueError, match="no attended token"):
        last_token_pool(hidden, mask, padding_side="right")


class _Tokenizer:
    def __init__(self, padding_side):
        self.padding_side = padding_side


class _Processor:
    def __init__(self, tokenizer_side, own_side=None):
        self.tokenizer = _Tokenizer(tokenizer_side)
        if own_side is not None:
            self.padding_side = own_side


def test_align_padding_side_forces_the_configured_side_and_says_what_it_was():
    processor = _Processor("left", own_side="left")

    detail = align_padding_side(processor, "right")

    assert processor.tokenizer.padding_side == "right"
    # The processor's own copy counts: one left on the old value can win at call
    # time, and then the batch is padded the way the check said it was not.
    assert processor.padding_side == "right"
    assert detail["declared_before"] == {"tokenizer": "left", "processor": "left"}
    assert detail["disagreed"] == ["processor", "tokenizer"]


def test_align_padding_side_refuses_a_processor_that_declares_nothing():
    """Nothing here could then establish which side it pads, and last-token pooling
    would read a PAD position without anything saying so."""
    with pytest.raises(ValueError, match="declares no padding_side"):
        align_padding_side(object(), "right")


def test_align_padding_side_refuses_a_setting_that_does_not_take():
    class _Frozen:
        padding_side = "left"

        def __setattr__(self, name, value):
            pass

    with pytest.raises(ValueError, match="does not take the setting"):
        align_padding_side(_Frozen(), "right")


def test_info_nce_is_lower_when_pairs_align():
    aligned = torch.eye(4)
    shuffled = torch.eye(4).flip(0)

    assert info_nce(aligned, aligned, 0.02) < info_nce(aligned, shuffled, 0.02)
