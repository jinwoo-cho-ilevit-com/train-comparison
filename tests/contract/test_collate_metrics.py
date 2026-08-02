"""The `collate-metrics` boundary: what the collate hands the measurement side.

The collate side (lane-f) and the measurement side (lane-d) work in isolation on
either side of this payload, so neither owns this file. What crosses is one
`MicroBatch` per micro-batch, and `tests/fixtures/microbatch.sample.json` is that
payload written down: four real batches, field by field, with each field's unit
and meaning. Two lanes can read a prose description differently; they have a much
harder time reading the same JSON differently, so the fixture is the contract and
this module is what makes disagreeing with it fail.

What it pins, and why each one is a way to be wrong silently:

* `tokens` and `padded_tokens` are separate and neither may stand in for the
  other. They are the two candidate denominators of throughput, and the
  `dataloader.packing` axis reverses rank depending on which is used.
* `rows` and `samples` are separate. Packing and pairing both make them differ,
  and with in-batch negatives the sequence count is part of the objective rather
  than a batching detail.
* `images` and `images_dropped` are separate, so a text-only view of an image
  corpus cannot be reported as an image run.
* `cu_seqlens` is present exactly when the batch is packed. It is what selects the
  pooling; absent-but-packed pools the wrong positions and still reports a number.
* A pack's boundaries reach two kernels by two different names, and each is judged
  on its own gate: the varlen four all together for attention, `seq_idx` alone for
  the causal conv. One rule over both would let either group's absence pass as the
  other group's shape.
* Units hold: a field named `tokens` is always a token count, never a row count.
* `tensors` is exactly `model(**tensors)` and carries no accounting number and no
  packing boundary.
* The payload carries counters, never a rate. The harness recomputes tokens/sec
  from these; it never takes a framework's own figure.

The payload is resolved by name rather than by import path — `trainbench/collate.py`
first, `scripts/bench.py` after it — because lane-d moves this code and the
contract is about what crosses the boundary, not where it is defined.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "microbatch.sample.json"
SPEC = json.loads(FIXTURE_PATH.read_text())

FIELDS = SPEC["fields"]
FIELD_NAMES = tuple(field["name"] for field in FIELDS)
COUNTER_NAMES = tuple(
    field["name"] for field in FIELDS if field["type"] == "int" and field["required"]
)
BOUNDARY_KEYS = tuple(SPEC["packing_boundary_keys"])
# Two groups, one gate each. `varlen_attention` is all-or-nothing because
# transformers reads the four together; `packed_conv` is judged on its own
# invariants, and holding it to the varlen rule is what would make a `qwen3_vl`
# pack — four kwargs, no conv to isolate — read as a partial varlen set.
VARLEN_KWARGS = tuple(SPEC["tensors_may_add"]["varlen_attention"])
PACKED_CONV_KWARGS = tuple(SPEC["tensors_may_add"]["packed_conv"])
MAY_ADD = VARLEN_KWARGS + PACKED_CONV_KWARGS
PAD_ID = SPEC["pad_id"]

# Where the payload may be defined, in the order lane-d moves it. `scripts/bench.py`
# is last because it is where the code is today and the first place it leaves.
CANDIDATE_MODULES = ("trainbench.collate", "trainbench.metrics", "trainbench.axes")
ENTRY_POINT = REPO_ROOT / "scripts" / "bench.py"

# A rate here would mean the collate had already divided by a time this side never
# measured — which is the framework-reported tokens/sec the harness must not use.
RATE_MARKERS = ("per_second", "per_sec", "_rate", "throughput", "tokens_per", "samples_per")


@functools.cache
def _modules() -> tuple[Any, ...]:
    """The importable candidates, in the order lane-d moves the payload into them."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    found = []
    for name in CANDIDATE_MODULES:
        try:
            found.append(importlib.import_module(name))
        except ImportError:
            continue
    return tuple(found)


@functools.cache
def _entry_point() -> Any:
    """`scripts/bench.py`, executed as a module — the fallback, loaded only if needed.

    Loaded last and lazily so that once lane-d has moved the payload out, a bench.py
    that fails to import cannot mask a `trainbench/collate.py` that works.
    """
    spec = importlib.util.spec_from_file_location("bench_entry_contract", ENTRY_POINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve(name: str) -> Any:
    for module in _modules():
        found = getattr(module, name, None)
        if found is not None:
            return found
    fallback = getattr(_entry_point(), name, None)
    if fallback is not None:
        return fallback
    searched = ", ".join((*CANDIDATE_MODULES, ENTRY_POINT.name))
    pytest.fail(
        f"the collate-metrics boundary needs {name!r} and none of {searched} defines it. "
        f"Moving it is fine — add its new module to CANDIDATE_MODULES; renaming it is a "
        f"change to the boundary and belongs in {FIXTURE_PATH.name} first."
    )


def _word_id(word: str) -> int:
    """The stand-in processor's vocabulary, stated in the fixture."""
    return 3 + sum(ord(character) for character in word) % 997


class StubTokenizer:
    """A tokenizer with no model behind it.

    The processor is the external dependency at this boundary, so it is the one
    thing stubbed: the collates under test are real, and the fixture records ids
    this function produces so the payload can be recomputed anywhere.
    """

    pad_token_id = PAD_ID

    def __call__(
        self,
        text: list[str],
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        rows = [[_word_id(word) for word in one.split()] for one in text]
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


class StubProcessor:
    """`AutoProcessor`'s shape, minus the model.

    `image_processor` is what `Collate` reads to decide whether this checkpoint can
    take pixels at all, so `accepts_images=False` is a text-only checkpoint and
    every image in the rows is counted as dropped.
    """

    chat_template = None

    def __init__(self, accepts_images: bool) -> None:
        self.tokenizer = StubTokenizer()
        self.image_token = "<image>"
        self.image_processor = SimpleNamespace() if accepts_images else None

    def __call__(
        self, text: list[str] | None = None, images: list[list[Any]] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        encoded = self.tokenizer(
            text or [], **{key: value for key, value in kwargs.items() if key != "return_tensors"}
        )
        if images:
            flat = [image for row in images for image in row]
            encoded["pixel_values"] = torch.tensor(
                [[float(len(str(image))), 1.0] for image in flat], dtype=torch.float32
            )
        return encoded


def _config(packing: bool = False, pretokenize: bool = False) -> SimpleNamespace:
    """The fields the collates read off `BenchConfig`, and nothing else.

    `prompt_format='raw'` with no instruction prompt keeps this test out of
    lane-f's prompt decision: whichever way the Qwen instruction prompt ends up
    being passed, the payload's shape is what is frozen here.
    """
    return SimpleNamespace(
        model=SimpleNamespace(
            instruction_prompt=None,
            prompt_format="raw",
            add_generation_prompt=False,
            padding_side="left",
        ),
        data=SimpleNamespace(max_seq_len=SPEC["max_seq_len"]),
        dataloader=SimpleNamespace(packing=packing, pretokenize=pretokenize),
    )


def _micro_batch(case: str) -> Any:
    """One real micro-batch, built the way `build_collate` builds it for that case."""
    build_collate = _resolve("build_collate")
    entry = SPEC["payloads"][case]
    if case == "padded_multimodal":
        return build_collate(StubProcessor(accepts_images=True), _config())(SPEC["rows"])
    if case == "padded_text_only":
        return build_collate(StubProcessor(accepts_images=False), _config())(SPEC["rows"])
    if case == "packed":
        collate = build_collate(StubProcessor(accepts_images=False), _config(packing=True))
        return collate(SPEC["rows"])
    collate = build_collate(StubProcessor(accepts_images=False), _config(pretokenize=True))
    return collate(entry["encoded_rows"])


def _as_json(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "dtype": str(value.dtype).replace("torch.", ""),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    return list(value) if isinstance(value, tuple) else value


CASES = tuple(SPEC["payloads"])


@pytest.fixture(params=CASES)
def case(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_the_payload_carries_exactly_the_frozen_fields_and_no_rate():
    """Renaming, adding, dropping or reordering a field is a change to the boundary.

    The order matters as well as the set: `MicroBatch` is a NamedTuple, so two
    fields that swap places keep every name and change every value.
    """
    micro_batch = _resolve("MicroBatch")

    assert micro_batch._fields == FIELD_NAMES, (
        f"the payload's fields are {micro_batch._fields} and {FIXTURE_PATH.name} freezes "
        f"{FIELD_NAMES}. Both lanes read that file; change it there first."
    )
    optional = {field["name"]: field.get("default") for field in FIELDS if not field["required"]}
    assert micro_batch._field_defaults == optional
    offenders = [name for name in FIELD_NAMES if any(m in name for m in RATE_MARKERS)]
    assert not offenders, (
        f"{offenders} name a rate. The collate hands over raw counters and the measurement "
        "side divides by the time it measured; a rate crossing here is a framework's own "
        "tokens/sec by another name."
    )


def test_the_collates_reproduce_the_frozen_payload(case: str):
    """The four collates, run on the fixture's rows, produce the fixture's payload.

    Every accounting field is compared exactly, so a rename, a merge of `tokens`
    into `padded_tokens`, or a swap of two counts fails here. `tensors` is compared
    key by key, and the only keys a lane may add without touching the fixture are
    the varlen kwargs named in `tensors_may_add` — anything else is a change to
    what `model(**tensors)` receives and belongs in the fixture first.
    """
    expected = SPEC["payloads"][case]["payload"]
    produced = _micro_batch(case)

    for name in FIELD_NAMES:
        if name == "tensors":
            continue
        assert _as_json(getattr(produced, name)) == expected[name], (
            f"{case}.{name} is {getattr(produced, name)!r}; {FIXTURE_PATH.name} freezes "
            f"{expected[name]!r} ({[f for f in FIELDS if f['name'] == name][0]['meaning']})"
        )

    tensors = produced.tensors
    extra = set(tensors) - set(expected["tensors"])
    assert extra <= set(MAY_ADD), (
        f"{case}.tensors gained {sorted(extra - set(MAY_ADD))}, which the fixture does not "
        f"declare. Only {list(MAY_ADD)} may appear without updating it."
    )
    assert set(expected["tensors"]) <= set(tensors), (
        f"{case}.tensors lost {sorted(set(expected['tensors']) - set(tensors))}"
    )
    for key, frozen in expected["tensors"].items():
        actual = _as_json(tensors[key])
        assert (actual["dtype"], actual["shape"]) == (frozen["dtype"], frozen["shape"]), (
            f"{case}.tensors[{key!r}] is {actual['dtype']}{actual['shape']}, frozen as "
            f"{frozen['dtype']}{frozen['shape']}"
        )
        if actual["dtype"].startswith("float"):
            # Pixels get a tolerance band; ids are compared exactly, because an id
            # that is nearly right is a different token.
            assert torch.allclose(
                tensors[key], torch.tensor(frozen["values"], dtype=tensors[key].dtype)
            )
        else:
            assert actual["values"] == frozen["values"]


def test_every_invariant_holds_for_a_real_batch(case: str):
    """The relations between the fields, checked on the batch rather than in prose.

    A prose invariant two lanes read differently is what this whole file exists to
    prevent, so each line of the fixture's `invariants` is decided here.
    """
    entry = SPEC["payloads"][case]
    produced = _micro_batch(case)
    tensors = produced.tensors
    input_ids = tensors["input_ids"]
    mask = tensors.get("attention_mask")

    # `tensors` is `model(**tensors)` and nothing else.
    forbidden = (set(tensors) & set(FIELD_NAMES)) | (set(tensors) & set(BOUNDARY_KEYS))
    assert not forbidden, (
        f"{case}.tensors carries {sorted(forbidden)}, which are accounting or packing-boundary "
        "names. model(**tensors) would reject them, and a count sent to the device would be "
        "read back inside the timed window."
    )
    present = [name for name in VARLEN_KWARGS if tensors.get(name) is not None]
    assert len(present) in (0, len(VARLEN_KWARGS)), (
        f"{case}.tensors carries {present} of the varlen kwargs. transformers takes the varlen "
        f"path only when all of {list(VARLEN_KWARGS)} are non-None, so a partial set is a padded "
        "forward pass reported as a packed one."
    )

    # Packing: cu_seqlens is present exactly when the batch is packed, and a packed
    # batch is the one with no attention_mask.
    packed = bool(entry["packed"])
    assert (mask is None) is packed, f"{case} declares packed={packed} but mask presence disagrees"
    assert (produced.cu_seqlens is not None) is packed, (
        f"{case}: cu_seqlens is present exactly when the batch is packed. Absent-but-packed is "
        "pooled the padded way, which reads a position no sequence ends at."
    )

    # Token accounting: two counts, computed two different ways, neither derived
    # from the other.
    assert produced.padded_tokens == int(input_ids.numel())
    if mask is not None:
        assert produced.tokens == int(mask.sum())
        assert produced.tokens < produced.padded_tokens, (
            f"{case} has ragged rows, so real tokens must come out below padded slots. Equal "
            "counts here mean one field is being computed from the other."
        )
    else:
        assert produced.tokens == produced.padded_tokens

    # Sequences vs dataset rows.
    assert produced.rows == 2 * produced.samples
    if packed:
        offsets = produced.cu_seqlens
        assert produced.rows == int(offsets.numel()) - 1
        assert int(offsets[0]) == 0
        assert int(offsets[-1]) == produced.tokens
        assert bool((offsets[1:] > offsets[:-1]).all()), "cu_seqlens must strictly increase"
    else:
        assert produced.rows == int(input_ids.shape[0])

    # The conv half of a pack's isolation, on its own gate. `Qwen3_5GatedDeltaNet`
    # hands `causal_conv1d_fn` nothing but `seq_idx`, so none of the rules above
    # say anything about whether the convolution stays inside a sequence.
    marks = tensors.get("seq_idx")
    assert (marks is not None) is packed, (
        f"{case}: seq_idx is present exactly when the batch is packed. Absent-but-packed lets "
        "the causal conv of arch=qwen3_5 run its receptive field across every sequence "
        "boundary in the pack, and present-but-padded names segments the rectangle has not got."
    )
    if packed:
        assert marks.dtype == torch.int32, (
            f"{case}: seq_idx is {marks.dtype}; TransformersKwargs declares torch.IntTensor "
            "(transformers/utils/generic.py:839)"
        )
        assert marks.shape == input_ids.shape, (
            f"{case}: seq_idx is {list(marks.shape)} and input_ids is {list(input_ids.shape)}. "
            "It is one index per token slot; any other shape indexes tokens it does not have."
        )
        row = marks[0]
        assert int(row[0]) == 0
        assert bool((row[1:] >= row[:-1]).all()), "seq_idx must not decrease along the pack"
        assert int(row[-1]) == produced.rows - 1, (
            f"{case}: seq_idx ends at {int(row[-1])} and the pack holds {produced.rows} "
            "sequences; the last sequence would be merged into the one before it."
        )
        changes = (row[1:] != row[:-1]).nonzero().flatten() + 1
        assert changes.tolist() == produced.cu_seqlens[1:-1].tolist(), (
            f"{case}: seq_idx changes at {changes.tolist()} and the pack's boundaries are "
            f"{produced.cu_seqlens[1:-1].tolist()}. The conv would be cut somewhere no "
            "sequence begins, which isolates the wrong thing and still raises nothing."
        )

    # Images.
    if produced.images_per_row is None:
        assert produced.images == 0
    else:
        assert len(produced.images_per_row) == produced.rows
        assert sum(produced.images_per_row) == produced.images


def test_real_and_padded_tokens_cannot_stand_in_for_each_other():
    """The padding fraction the two fields exist to expose.

    If either field were computed from the other, the padded batches would report
    no padding and the packed one would report some. Both directions are checked
    because a merge of the two fields passes half of this on its own.
    """
    padded = _micro_batch("padded_text_only")
    packed = _micro_batch("packed")

    assert padded.tokens < padded.padded_tokens
    assert padded.tokens == packed.tokens, (
        "the same rows hold the same real tokens whether padded or packed; only the slots "
        "differ. A disagreement means one of the two counts is measuring the other thing."
    )
    assert packed.padded_tokens == packed.tokens < padded.padded_tokens


def test_the_fixture_keeps_the_counters_distinguishable():
    """A guard on the fixture, not on the code.

    Comparing a produced payload against a frozen one catches a swapped pair of
    fields only if the two frozen values differ. This keeps the padded samples in
    that state, so `tokens` cannot quietly start carrying `rows`. The packed sample
    is exempt: `tokens == padded_tokens` there is the property packing has.
    """
    for name in ("padded_multimodal", "padded_text_only"):
        payload = SPEC["payloads"][name]["payload"]
        counters = {field: payload[field] for field in COUNTER_NAMES}
        assert len(set(counters.values())) == len(counters), (
            f"{name} has two counters with the same value ({counters}), so swapping them "
            f"would pass. Change the fixture's rows until every counter differs."
        )
