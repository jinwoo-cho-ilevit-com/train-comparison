"""What a packed batch tells attention about where its sequences end.

`docs/methodology.md` used to conclude that a pack has no isolation under
`attn=sdpa` because this repository builds no block-diagonal mask. transformers
5.14.1 builds one itself from `position_ids` alone
(`masking_utils.py:735-764` and `:858-868`), and `PackedCollate` already satisfies
its three preconditions — no 2D mask, no cache, positions restarting at 0. The
first half of this file measures that on CPU instead of arguing about it.

The half that is **not** automatic is `arch=qwen3_5`: three of every four of its
layers are `linear_attention`, and `Qwen3_5GatedDeltaNet.forward` never receives
`position_ids`. It reads `kwargs.get("cu_seq_lens_q")` and, when that is None,
runs the whole pack through one recurrent scan without raising. The second half
pins that the four varlen kwargs leave the collate together.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from trainbench.collate import VARLEN_KWARGS, build_collate

PAD_ID = 0

# Word lengths chosen so the three packed sequences have different lengths: a pack
# of equal-length sequences hides an off-by-one in the boundaries.
ROWS = [
    {"qry": "alpha beta gamma", "pos_text": "delta"},
    {"qry": "epsilon zeta", "pos_text": "eta theta iota kappa"},
]


class _Tokenizer:
    """Word-level ids, no model behind them, and no padding unless asked."""

    pad_token_id = PAD_ID

    def __call__(
        self,
        text: list[str],
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        rows = [[3 + sum(map(ord, word)) % 97 for word in one.split()] for one in text]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        if not padding:
            return {"input_ids": rows}
        width = max(len(row) for row in rows)
        return {
            "input_ids": torch.tensor(
                [row + [PAD_ID] * (width - len(row)) for row in rows], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in rows], dtype=torch.long
            ),
        }


class _Processor:
    """`AutoProcessor`'s shape for a text-only checkpoint."""

    chat_template = None
    image_processor = None

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.image_token = "<image>"

    def __call__(self, text: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        return self.tokenizer(
            text or [], **{k: v for k, v in kwargs.items() if k != "return_tensors"}
        )


def _config(packing: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            instruction_prompt=None,
            prompt_format="raw",
            add_generation_prompt=False,
            padding_side="right",
        ),
        data=SimpleNamespace(max_seq_len=64),
        dataloader=SimpleNamespace(packing=packing, pretokenize=False),
    )


@pytest.fixture
def packed() -> Any:
    return build_collate(_Processor(), _config(packing=True))(ROWS)


def test_a_packed_batch_carries_the_varlen_kwargs_in_full(packed: Any):
    """All four or none. `modeling_flash_attention_utils.py:761-763` gates the
    varlen path on `all(kwarg is not None for kwarg in (...))`, so three of four is
    a dense forward pass over the whole pack reported as a packed one."""
    present = [name for name in VARLEN_KWARGS if packed.tensors.get(name) is not None]

    assert present == list(VARLEN_KWARGS), (
        f"the packed batch carries {present} of {list(VARLEN_KWARGS)}. transformers takes "
        "the varlen path only on all four; a partial set attends across every sequence "
        "boundary in the pack and nothing raises."
    )


def test_the_varlen_gate_in_the_pinned_source_still_needs_all_four():
    """The rule above is transformers', not this repository's.

    Read out of the installed wheel rather than restated, so that a version bump
    which loosens or renames the gate fails here instead of being assumed.
    """
    import transformers.modeling_flash_attention_utils as fa

    source = Path(fa.__file__).read_text()
    gate = (
        "is_fa_with_varlen_kwargs = all(\n"
        "        kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, "
        "max_length_q, max_length_k)\n"
        "    )"
    )

    assert gate in source, (
        f"{fa.__file__} no longer contains the all-four gate this collate is built "
        "against. Re-read it before trusting anything else in this file."
    )


def test_the_varlen_kwargs_describe_this_pack_and_not_some_other(packed: Any):
    """Offsets and the longest sequence, both derived from the same lengths the
    boundary vector reports — a `max_length_q` too small silently truncates the
    kernel's view of the longest sequence."""
    lengths = packed.cu_seqlens[1:] - packed.cu_seqlens[:-1]

    assert torch.equal(packed.tensors["cu_seq_lens_q"], packed.cu_seqlens)
    assert torch.equal(packed.tensors["cu_seq_lens_k"], packed.cu_seqlens)
    assert packed.tensors["max_length_q"] == int(lengths.max())
    assert packed.tensors["max_length_k"] == int(lengths.max())
    assert len(set(lengths.tolist())) > 1, "equal-length sequences would hide an offset error"


def test_the_varlen_lengths_stay_on_the_host(packed: Any):
    """`to_device` moves tensors and passes everything else through. A 0-dim tensor
    here would be a host-to-device round trip inside the timed window for a number
    that only bounds a kernel launch."""
    assert isinstance(packed.tensors["max_length_q"], int)
    assert not torch.is_tensor(packed.tensors["max_length_q"])


def test_a_padded_batch_carries_none_of_the_varlen_kwargs():
    """There is nothing to isolate in a padded rectangle, and a batch that named
    boundaries it does not have would take the varlen path over its padding."""
    padded = build_collate(_Processor(), _config(packing=False))(ROWS)

    assert [name for name in VARLEN_KWARGS if name in padded.tensors] == []


@pytest.mark.parametrize(
    "arch, config_class",
    [
        ("qwen3_5", "transformers.models.qwen3_5.configuration_qwen3_5:Qwen3_5TextConfig"),
        ("qwen3_vl", "transformers.models.qwen3_vl.configuration_qwen3_vl:Qwen3VLTextConfig"),
    ],
)
def test_packing_isolation_is_block_diagonal_for_this_pack(arch: str, config_class: str, packed):
    """`create_causal_mask` over the collate's own output, on CPU.

    The mask is not this repository's to build: `position_ids` restarting at 0 per
    sequence is the whole input transformers needs. What is checked is the result —
    a token may attend to earlier tokens of its own sequence and to nothing else.
    """
    import importlib

    from transformers.masking_utils import create_causal_mask

    module_name, _, name = config_class.partition(":")
    config = getattr(importlib.import_module(module_name), name)(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=128,
    )
    config._attn_implementation = "sdpa"

    position_ids = packed.tensors["position_ids"]
    total = int(position_ids.shape[1])
    mask = create_causal_mask(
        config=config,
        inputs_embeds=torch.zeros(1, total, config.hidden_size),
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    assert mask is not None, (
        f"{arch}: create_causal_mask returned None, which is the un-isolated pack — every "
        "sequence reads the ones packed before it and the loss still converges to something."
    )
    offsets = packed.cu_seqlens.tolist()
    sequence_of = torch.zeros(total, dtype=torch.long)
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        sequence_of[start:end] = index
    query = torch.arange(total).unsqueeze(1)
    key = torch.arange(total).unsqueeze(0)
    expected = (sequence_of[query] == sequence_of[key]) & (query >= key)

    assert torch.equal(mask[0, 0], expected), (
        f"{arch}: the mask is not block-diagonal causal.\n{mask[0, 0].int()}\nexpected\n"
        f"{expected.int()}"
    )


def test_the_isolation_comes_from_the_restarting_positions(packed):
    """The control for the test above.

    A pack whose positions run straight through gets one causal triangle, which is
    what `docs/methodology.md` described. Without this, a mask function that ignored
    `position_ids` entirely would pass the block-diagonal check on any pack whose
    sequences happened to be laid out that way.
    """
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=128,
    )
    config._attn_implementation = "sdpa"
    total = int(packed.tensors["position_ids"].shape[1])

    mask = create_causal_mask(
        config=config,
        inputs_embeds=torch.zeros(1, total, config.hidden_size),
        attention_mask=None,
        past_key_values=None,
        position_ids=torch.arange(total).unsqueeze(0),
    )

    # None is `sdpa`'s "no mask needed, use is_causal" — one triangle over the pack.
    assert mask is None or torch.equal(
        mask[0, 0], torch.ones(total, total, dtype=torch.bool).tril()
    ), "straight-through positions must not produce isolation; nothing else supplies it"
