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

That layer isolates on two kwargs, not one: the recurrent scan takes
`cu_seq_lens_q` and the causal convolution ahead of it takes `seq_idx`
(`modeling_qwen3_5.py:492-499`). The third part of this file pins the second one.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from trainbench.collate import SEQ_IDX_KWARGS, VARLEN_KWARGS, build_collate

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


def test_a_packed_batch_carries_seq_idx_shaped_like_its_tokens(packed: Any):
    """One int32 index per token slot.

    `TransformersKwargs` declares `seq_idx: torch.IntTensor`
    (`transformers/utils/generic.py:839`) and describes it as the sequence index of
    each token in a flattened packed batch. A wrong dtype or a wrong length is an
    index into tokens the pack has not got.
    """
    marks = packed.tensors[SEQ_IDX_KWARGS[0]]

    assert marks.dtype == torch.int32
    assert marks.shape == packed.tensors["input_ids"].shape


def test_seq_idx_marks_exactly_the_packs_sequences(packed: Any):
    """Its change points are the pack's boundaries and nothing else.

    A vector of the right dtype and length that cuts anywhere else isolates a
    segmentation no sequence has, which the conv would apply without raising.
    """
    row = packed.tensors[SEQ_IDX_KWARGS[0]][0]
    changes = (row[1:] != row[:-1]).nonzero().flatten() + 1

    assert int(row[0]) == 0
    assert int(row[-1]) == packed.rows - 1
    assert changes.tolist() == packed.cu_seqlens[1:-1].tolist(), (
        f"seq_idx cuts at {changes.tolist()} and the pack's sequences begin at "
        f"{packed.cu_seqlens[1:-1].tolist()}"
    )


def test_a_padded_batch_carries_no_seq_idx():
    """A rectangle has no packed segments, and naming some would make the conv
    reset its state at positions that are the middle of a row."""
    padded = build_collate(_Processor(), _config(packing=False))(ROWS)

    assert [name for name in SEQ_IDX_KWARGS if name in padded.tensors] == []


def test_the_conv_in_the_pinned_source_still_reads_seq_idx_and_nothing_else():
    """Read out of the installed wheel rather than restated.

    Both branches are pinned: the fused call that takes `seq_idx`, and the fallback
    that has no argument for it — which is why a pod without `causal-conv1d` gets no
    conv isolation however this collate fills the batch.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5

    source = Path(qwen3_5.__file__).read_text()
    conv = (
        "            if self.causal_conv1d_fn is not None:\n"
        "                mixed_qkv = self.causal_conv1d_fn(\n"
        "                    x=mixed_qkv,\n"
        "                    weight=self.conv1d.weight.squeeze(1),\n"
        "                    bias=self.conv1d.bias,\n"
        "                    activation=self.activation,\n"
        '                    seq_idx=kwargs.get("seq_idx"),\n'
        "                )\n"
        "            else:\n"
        "                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])"
    )

    assert conv in source, (
        f"{qwen3_5.__file__} no longer reaches its causal conv the way this collate is built "
        "against. Re-read it before trusting anything else about seq_idx here."
    )


def test_the_pinned_source_defines_seq_idx_as_one_index_per_token():
    """The shape and the meaning, taken from the only pure-torch implementation.

    `causal_conv1d` is a CUDA package and is not installed here, so what `seq_idx`
    is expected to look like cannot be read off the kernel. `Lfm2ShortConv` carries
    the same argument through a torch fallback and that fallback is the definition:
    `seq_idx[0]` is per-token, and a segment runs between value changes.
    """
    import transformers.models.lfm2.modeling_lfm2 as lfm2

    source = Path(lfm2.__file__).read_text()
    semantics = (
        "            si = seq_idx[0]\n"
        "            change = (si[1:] != si[:-1]).nonzero(as_tuple=True)[0] + 1\n"
    )

    assert semantics in source, (
        f"{lfm2.__file__} no longer cuts segments at seq_idx's value changes. The shape this "
        "collate emits is derived from that reading."
    )


@pytest.mark.parametrize(
    "arch, config_class, model_class",
    [
        ("qwen3_5", "Qwen3_5TextConfig", "Qwen3_5TextModel"),
        ("qwen3_vl", "Qwen3VLTextConfig", "Qwen3VLTextModel"),
        ("gemma4", "Gemma4TextConfig", "Gemma4TextModel"),
    ],
)
def test_every_arch_takes_the_packed_batch_this_collate_builds(
    arch: str, config_class: str, model_class: str, packed: Any
):
    """`seq_idx` rides on every packed batch, so every arch has to accept one.

    Only `arch=qwen3_5` has a causal conv to isolate; the collate emits the key for
    all three rather than switching on a config field that names no attention, and
    that is only safe if the other two take it. A forward on CPU is what says so.

    What this rules out is a refusal, and that is all — measured here, these models
    accept a kwarg nobody declared just as quietly
    (`test_an_undeclared_kwarg_is_swallowed_just_as_quietly`). The two tests below
    carry the part that is not vacuous: who reads the key, and who declares it.

    The hidden states are compared with and without it. On this host they are equal
    for all three, `qwen3_5` included: `causal_conv1d_fn` is None without the CUDA
    package, and the fallback branch has no `seq_idx` argument. What the real kernel
    does with it is a pod question and is not measured here.
    """
    import transformers

    config = getattr(transformers, config_class)(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=64,
        vocab_size=128,
    )
    config._attn_implementation = "sdpa"
    model = getattr(transformers, model_class)(config).eval()
    ids = {key: packed.tensors[key] for key in ("input_ids", "position_ids")}

    with torch.no_grad():
        plain = model(**ids, use_cache=False).last_hidden_state
        marked = model(
            **ids, seq_idx=packed.tensors[SEQ_IDX_KWARGS[0]], use_cache=False
        ).last_hidden_state

    assert plain.shape == marked.shape
    assert torch.equal(plain, marked), (
        f"{arch}: seq_idx changed the hidden states on a host with no causal-conv1d, where "
        "the fallback branch cannot read it. Something else is consuming the key."
    )


def test_an_undeclared_kwarg_is_swallowed_just_as_quietly(packed: Any):
    """The control for the test above.

    A forward that does not raise says nothing on its own: these models take
    `**kwargs` and never look at what they were not built to read. Without this,
    the arch test would read as evidence that `seq_idx` is understood, when all it
    shows is that it is not refused.
    """
    import transformers

    config = transformers.Qwen3VLTextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=64,
        vocab_size=128,
    )
    config._attn_implementation = "sdpa"
    model = transformers.Qwen3VLTextModel(config).eval()
    ids = {key: packed.tensors[key] for key in ("input_ids", "position_ids")}

    with torch.no_grad():
        plain = model(**ids, use_cache=False).last_hidden_state
        nonsense = model(**ids, use_cache=False, not_a_transformers_kwarg=ids["input_ids"])

    assert torch.equal(plain, nonsense.last_hidden_state)


def test_only_qwen3_5_reads_seq_idx_among_the_arches_this_study_measures():
    """Why the collate emits it for every pack instead of switching on `model.arch`.

    The key is inert for `qwen3_vl` and `gemma4` because their modeling code has no
    reader for it, not because the collate withheld it. If a version bump gives one
    of them a `seq_idx` consumer, that changes what a packed batch means for that
    arch and this is where it surfaces.
    """
    import transformers.models.gemma4.modeling_gemma4 as gemma4
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl

    readers = {
        module.__name__: "seq_idx" in Path(module.__file__).read_text()
        for module in (qwen3_5, qwen3_vl, gemma4)
    }

    assert readers == {
        qwen3_5.__name__: True,
        qwen3_vl.__name__: False,
        gemma4.__name__: False,
    }, f"which arches read seq_idx has changed: {readers}"


def test_transformers_still_declares_seq_idx_as_a_forward_kwarg():
    """`TransformersKwargs` is the list of keys a caller may put in `tensors`.

    A key that leaves that list is one the models are free to stop threading down
    to their layers, and it would keep passing every forward above in silence.
    """
    from transformers.utils.generic import TransformersKwargs

    declared = TransformersKwargs.__annotations__

    assert "seq_idx" in declared, (
        f"TransformersKwargs no longer declares seq_idx; it holds {sorted(declared)}"
    )
    assert "not_a_transformers_kwarg" not in declared
