"""Contract `kernel-provenance`: what identifies the attention kernel that ran.

Frozen before lane-e (`trainbench/kernels.py`) and lane-g (`trainbench/loader.py`
and the probe adapters) start, and owned by neither. Lane-g builds the model that
binds a kernel; lane-e reads the binding back and refuses `dataloader.packing`
where it is unsafe. Two lanes can each pass their own tests while holding opposite
assumptions about what crosses between them, so the payload lives in
`tests/fixtures/kernel_fingerprint.sample.json` and the rules live here.

Three properties of transformers 5.14.1 are why this payload has the fields it has.
Each is asserted below against the installed package rather than quoted:

1. A version string does not identify the kernel. Requesting `flash_attention_2`
   without the `flash-attn` package rewrites the request to a Hub repository id
   (`modeling_utils.py:1997-2003`, `modeling_flash_attention_utils.py:65-71`),
   which is fetched at runtime and may carry a revision (`hub_kernels.py:484-505`).
   Identity is therefore repo + revision, never `transformers==5.14.1`.
2. Registration in the mask registry is a second, separate fact. Mask creation is
   skipped and `None` is passed to the attention layers when the implementation is
   absent from `AttentionMaskInterface._global_mapping`
   (`masking_utils.py:826-827`, returned at `:939-940`), and registering an
   attention function does not register a mask function. Packed-sequence isolation
   is exactly what disappears, so this flag is lane-e's refusal input.
3. Coverage is per sub-config. `attn_implementation` may be a dict and each
   sub-config takes its own value (`configuration_utils.py:401-417`), so one string
   cannot say which towers of a multimodal model got the kernel.

확인 안 함: whether the fa2/fa3/fa4 Hub kernels load, register, and bind at all.
`flash-attn` and `kernels` are absent from this host (macOS, no CUDA) and only
`envs/native` carries them, on a linux image. What settles it is the first GPU pod
run recording a fingerprint for `attn=fa2` and `attn=fa3`: the values below are
placeholders, the shape is what is frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "kernel_fingerprint.sample.json"

# Where the payload travels. Lane-g returns it inside the build fingerprint under
# BUILD_FINGERPRINT_KEY; the run record carries that fingerprint under
# RUN_RECORD_KEY. Both are string keys in a JSON document, which is the only thing
# that survives the pod the run happened on.
BUILD_FINGERPRINT_KEY = "kernel"
RUN_RECORD_KEY = "build_fingerprint"

# The registry whose membership decides whether a mask gets built. Named in the
# payload because there are two registries and only one of them answers this
# question: `AttentionInterface` says the kernel is callable, `AttentionMaskInterface`
# says a mask will be made for it. Reading the wrong one reports safety that is not there.
MASK_REGISTRY = "transformers.masking_utils.AttentionMaskInterface"

FINGERPRINT_KEYS = frozenset({"requested", "resolved", "backbones"})
REQUESTED_KEYS = frozenset({"axis", "value", "attn_implementation"})
RESOLVED_KEYS = frozenset({"attn_implementation", "identity", "mask_registered", "mask_registry"})
IDENTITY_KEYS = frozenset(
    {"source", "package", "package_version", "repo_id", "requested_ref", "revision"}
)
BACKBONE_KEYS = frozenset({"attn_implementation", "mask_registered", "requested"})

# `hub` is resolved from the Hub at runtime and is identified by repo + revision.
# `package` is an installed wheel (flash-attn) and `builtin` ships inside
# transformers; both are identified by distribution name + version, and neither may
# claim a repo.
IDENTITY_SOURCES = frozenset({"hub", "package", "builtin"})


class ContractViolation(AssertionError):
    """The payload crossing `kernel-provenance` is not the shape both lanes agreed on."""


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractViolation(f"{where} must be a mapping, got {type(value).__name__}: {value!r}")
    return value


def _exact_keys(value: dict[str, Any], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ContractViolation(
            f"{where} keys are frozen by this contract; missing={missing} unexpected={unexpected}"
        )


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ContractViolation(f"{where} must be a bool, got {type(value).__name__}: {value!r}")
    return value


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(f"{where} must be a non-empty string, got {value!r}")
    return value


def _validate_identity(identity: Any) -> None:
    ident = _mapping(identity, "resolved.identity")
    _exact_keys(ident, IDENTITY_KEYS, "resolved.identity")
    source = ident["source"]
    if source not in IDENTITY_SOURCES:
        raise ContractViolation(
            f"resolved.identity.source must be one of {sorted(IDENTITY_SOURCES)}, got {source!r}"
        )
    if source == "hub":
        for field in ("repo_id", "revision"):
            if not isinstance(ident[field], str) or not ident[field].strip():
                raise ContractViolation(
                    "a Hub kernel is fetched at runtime and is identified by repo_id and "
                    f"revision; resolved.identity.{field}={ident[field]!r}. A package version "
                    "identifies the library that went looking, not the kernel that bound."
                )
        return
    for field in ("package", "package_version"):
        if not isinstance(ident[field], str) or not ident[field].strip():
            raise ContractViolation(
                f"resolved.identity.source={source!r} is identified by its distribution, so "
                f"{field} is required; got {ident[field]!r}"
            )
    for field in ("repo_id", "revision"):
        if ident[field] is not None:
            raise ContractViolation(
                f"resolved.identity.source={source!r} did not come from the Hub, so {field} "
                f"must be null; got {ident[field]!r}"
            )


def validate_kernel_fingerprint(payload: Any) -> None:
    """Raise `ContractViolation` unless `payload` is a kernel fingerprint.

    Import-free on purpose: a checkout without the `native` extra can still decide
    whether a stored payload is well formed.
    """
    fingerprint = _mapping(payload, "fingerprint")
    _exact_keys(fingerprint, FINGERPRINT_KEYS, "fingerprint")

    requested = _mapping(fingerprint["requested"], "requested")
    _exact_keys(requested, REQUESTED_KEYS, "requested")
    for field in REQUESTED_KEYS:
        _name(requested[field], f"requested.{field}")

    resolved = _mapping(fingerprint["resolved"], "resolved")
    _exact_keys(resolved, RESOLVED_KEYS, "resolved")
    resolved_impl = _name(resolved["attn_implementation"], "resolved.attn_implementation")
    _validate_identity(resolved["identity"])
    _flag(resolved["mask_registered"], "resolved.mask_registered")
    if resolved["mask_registry"] != MASK_REGISTRY:
        raise ContractViolation(
            f"resolved.mask_registry must name the registry that decides mask creation, "
            f"{MASK_REGISTRY!r}; got {resolved['mask_registry']!r}"
        )
    identity = resolved["identity"]
    if identity["source"] == "hub" and not resolved_impl.startswith(identity["repo_id"]):
        raise ContractViolation(
            f"resolved.attn_implementation={resolved_impl!r} does not name "
            f"resolved.identity.repo_id={identity['repo_id']!r}; the recorded provenance "
            "belongs to a different kernel than the one the model is running."
        )

    backbones = fingerprint["backbones"]
    if not isinstance(backbones, dict) or not backbones:
        raise ContractViolation(
            "backbones must be a non-empty mapping of sub-config name -> coverage. "
            "attn_implementation is set per sub-config and a backbone left out of the request "
            "keeps SDPA, so a single value cannot say which towers got the kernel; got "
            f"{backbones!r}"
        )

    covered = []
    for name, entry in sorted(backbones.items()):
        where = f"backbones[{name!r}]"
        coverage = _mapping(entry, where)
        _exact_keys(coverage, BACKBONE_KEYS, where)
        impl = _name(coverage["attn_implementation"], f"{where}.attn_implementation")
        registered = _flag(coverage["mask_registered"], f"{where}.mask_registered")
        if _flag(coverage["requested"], f"{where}.requested"):
            covered.append(name)
            if impl != resolved_impl:
                raise ContractViolation(
                    f"{where} is marked as having received the request but runs {impl!r}, "
                    f"not resolved.attn_implementation={resolved_impl!r}"
                )
            if registered != resolved["mask_registered"]:
                raise ContractViolation(
                    f"{where}.mask_registered={registered} contradicts "
                    f"resolved.mask_registered={resolved['mask_registered']} for the same "
                    "implementation"
                )
        elif impl == resolved_impl:
            raise ContractViolation(
                f"{where} runs the requested implementation {impl!r} but is marked "
                "requested=false; coverage would read as narrower than it is"
            )
    if not covered:
        raise ContractViolation(
            "no backbone is marked as having received the requested implementation, so the "
            "request reached nothing and the fingerprint records a kernel that is not running"
        )


def packing_isolation_holds(fingerprint: dict[str, Any]) -> bool:
    """Whether packed sequences stay isolated under this fingerprint.

    The rule lane-e's refusal reads. Any backbone whose implementation is absent
    from the mask registry gets `attention_mask=None`, and a packed batch then has
    every sequence in it visible to the ones that follow. One such backbone is
    enough, which is why this is `all` and not a check of the text tower alone.
    """
    return all(entry["mask_registered"] for entry in fingerprint["backbones"].values())


@pytest.fixture(scope="module")
def samples() -> dict[str, dict[str, Any]]:
    document = json.loads(FIXTURE.read_text())
    assert document["contract"] == "kernel-provenance"
    assert document["schema_version"] == 1
    return document["samples"]


def test_the_stored_samples_are_the_frozen_shape(samples):
    """The file is the specification. Both lanes read this, not a prose summary."""
    assert set(samples) == {
        "sdpa_builtin_gemma4",
        "fa2_hub_fallback_qwen3_vl",
        "fa3_hub_kernel_mask_unregistered_qwen3_vl",
    }
    for fingerprint in samples.values():
        validate_kernel_fingerprint(fingerprint)

    # Both outcomes of the refusal input are present, so a sample set that stops
    # covering the unsafe case is a visible edit rather than a quiet one.
    holds = {name: packing_isolation_holds(fp) for name, fp in samples.items()}
    assert holds == {
        "sdpa_builtin_gemma4": True,
        "fa2_hub_fallback_qwen3_vl": True,
        "fa3_hub_kernel_mask_unregistered_qwen3_vl": False,
    }


def test_a_hub_kernel_named_by_a_version_string_is_refused(samples):
    """Drift mutation 1: repo + revision replaced by a version.

    An image digest pins the library that goes looking for a kernel. It does not pin
    the kernel, because that is resolved from the Hub while the run is starting.
    """
    fingerprint = json.loads(json.dumps(samples["fa2_hub_fallback_qwen3_vl"]))
    fingerprint["resolved"]["identity"].update(
        {"repo_id": None, "revision": None, "package": "transformers", "package_version": "5.14.1"}
    )

    with pytest.raises(ContractViolation, match="identified by repo_id and revision"):
        validate_kernel_fingerprint(fingerprint)


def test_a_hub_revision_that_went_missing_is_refused(samples):
    """A repository without a revision names a moving target, not a kernel."""
    fingerprint = json.loads(json.dumps(samples["fa2_hub_fallback_qwen3_vl"]))
    fingerprint["resolved"]["identity"]["revision"] = ""

    with pytest.raises(ContractViolation, match="resolved.identity.revision=''"):
        validate_kernel_fingerprint(fingerprint)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("resolved",), "resolved keys are frozen"),
        (("backbones", "text_config"), "backbones\\['text_config'\\] keys are frozen"),
    ],
    ids=["resolved", "backbone"],
)
def test_dropping_the_mask_registration_flag_is_refused(samples, path, expected):
    """Drift mutation 2: the flag lane-e's refusal reads is gone.

    Without it, `dataloader.packing=true` on an implementation that builds no mask
    produces a throughput number for a batch whose sequences read each other.
    """
    fingerprint = json.loads(json.dumps(samples["fa3_hub_kernel_mask_unregistered_qwen3_vl"]))
    target = fingerprint
    for key in path:
        target = target[key]
    del target["mask_registered"]

    with pytest.raises(ContractViolation, match=expected):
        validate_kernel_fingerprint(fingerprint)


def test_a_mask_registration_flag_that_stopped_being_a_flag_is_refused(samples):
    """`"false"` is truthy, and `packing_isolation_holds` would pass on it."""
    fingerprint = json.loads(json.dumps(samples["fa3_hub_kernel_mask_unregistered_qwen3_vl"]))
    fingerprint["backbones"]["text_config"]["mask_registered"] = "false"

    with pytest.raises(ContractViolation, match="must be a bool"):
        validate_kernel_fingerprint(fingerprint)


@pytest.mark.parametrize(
    "collapsed",
    [
        "kernels-community/vllm-flash-attn3",
        {},
        {"text_config": "kernels-community/vllm-flash-attn3"},
    ],
    ids=["one_string", "empty", "string_values"],
)
def test_per_backbone_coverage_cannot_collapse_to_one_value(samples, collapsed):
    """Drift mutation 3: the per-sub-config map becomes a single answer.

    Both VL models under test take `attn_implementation` per backbone, and a
    backbone left out keeps SDPA. Collapsing the map reports a kernel on towers
    that never received it.
    """
    fingerprint = json.loads(json.dumps(samples["fa3_hub_kernel_mask_unregistered_qwen3_vl"]))
    fingerprint["backbones"] = collapsed

    with pytest.raises(ContractViolation, match="backbones"):
        validate_kernel_fingerprint(fingerprint)


def test_a_backbone_mislabelled_as_covered_is_refused(samples):
    """The map is only worth keeping if the labels cannot disagree with it."""
    fingerprint = json.loads(json.dumps(samples["fa2_hub_fallback_qwen3_vl"]))
    fingerprint["backbones"]["vision_config"]["requested"] = True

    with pytest.raises(ContractViolation, match="not resolved.attn_implementation"):
        validate_kernel_fingerprint(fingerprint)


def test_a_request_that_reached_no_backbone_is_refused(samples):
    fingerprint = json.loads(json.dumps(samples["fa2_hub_fallback_qwen3_vl"]))
    fingerprint["backbones"]["text_config"] = {
        "attn_implementation": "sdpa",
        "mask_registered": True,
        "requested": False,
    }

    with pytest.raises(ContractViolation, match="request reached nothing"):
        validate_kernel_fingerprint(fingerprint)


# --- what the payload's fields are claims about ------------------------------
# Everything below runs against the installed transformers. The contract's fields
# only mean something if transformers still behaves the way they assume.


def test_the_named_mask_registry_is_the_one_that_decides_mask_creation():
    """`mask_registry` names `AttentionMaskInterface`, and `masking_utils` consults
    that class's `_global_mapping` (`masking_utils.py:826`)."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    module, _, attribute = MASK_REGISTRY.rpartition(".")
    assert module == "transformers.masking_utils"
    assert attribute == AttentionMaskInterface.__name__
    assert ALL_MASK_ATTENTION_FUNCTIONS._global_mapping is AttentionMaskInterface._global_mapping
    assert set(AttentionMaskInterface._global_mapping) == {
        "eager",
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
        "flex_attention",
        "sdpa",
    }


def test_registering_an_attention_kernel_does_not_register_its_mask():
    """The two registries are independent, which is why the payload carries the flag.

    A kernel bound through `AttentionInterface.register` is callable and produces
    no mask. Nothing warns.
    """
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = "trainbench_contract_probe"

    def _unused_attention(*args, **kwargs):
        raise NotImplementedError

    AttentionInterface.register(name, _unused_attention)
    try:
        assert name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        assert name not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    finally:
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)


def test_an_unregistered_implementation_drops_the_packed_sequence_mask(samples):
    """The mechanism `packing_isolation_holds` exists to refuse.

    Same packed `position_ids` under both samples: the registered one gets a
    block-diagonal mask that isolates the three sequences, the unregistered one gets
    `None` and every token in the pack can attend to the ones before it.
    """
    from transformers import PreTrainedConfig
    from transformers.masking_utils import create_causal_mask

    embeds = torch.zeros(1, 6, 4)
    # Three packed sequences of 2, 3 and 1 tokens, positions restarting at 0.
    position_ids = torch.tensor([[0, 1, 0, 1, 2, 0]])

    def mask_for(implementation: str) -> torch.Tensor | None:
        config = PreTrainedConfig()
        config._attn_implementation = implementation
        return create_causal_mask(
            config=config,
            inputs_embeds=embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )

    safe = samples["sdpa_builtin_gemma4"]
    unsafe = samples["fa3_hub_kernel_mask_unregistered_qwen3_vl"]
    assert packing_isolation_holds(safe) and not packing_isolation_holds(unsafe)

    built = mask_for(safe["resolved"]["attn_implementation"])
    assert built is not None
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(built[0, 0], expected)

    assert mask_for(unsafe["resolved"]["attn_implementation"]) is None


def test_mask_registration_is_a_read_back_not_a_property_of_the_requested_name(samples):
    """Why the flag cannot be derived from the config.

    The fa2 sample records `mask_registered: true`, and its implementation is absent
    from the registry until the kernel has been fetched and bound. The same string
    is safe or unsafe depending on whether that happened, so only the built model
    can answer.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    fingerprint = samples["fa2_hub_fallback_qwen3_vl"]
    assert fingerprint["resolved"]["mask_registered"] is True
    assert fingerprint["resolved"]["attn_implementation"] not in (
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    )


def test_requesting_flash_attention_by_name_resolves_to_a_hub_repository(samples):
    """Why identity is repo + revision. `flash_attention_2` is not a kernel name; it
    is rewritten to a repository that is fetched while the run starts, and the
    repository string itself carries an optional `@revision`."""
    from transformers.integrations.hub_kernels import is_kernel
    from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_FALLBACK

    fingerprint = samples["fa2_hub_fallback_qwen3_vl"]
    requested = fingerprint["requested"]["attn_implementation"]
    repo_id = fingerprint["resolved"]["identity"]["repo_id"]

    assert not is_kernel(requested)
    assert FLASH_ATTN_KERNEL_FALLBACK[requested] == repo_id
    assert is_kernel(repo_id)
    assert is_kernel(f"{repo_id}@{fingerprint['resolved']['identity']['revision']}")


def test_a_backbone_omitted_from_the_request_keeps_sdpa(samples):
    """Why coverage is a map. Qwen3-VL takes the implementation per sub-config, and
    the top-level string keeps describing something else entirely."""
    from transformers import Qwen3VLConfig

    config = Qwen3VLConfig()
    assert set(config.sub_configs) == {"text_config", "vision_config"}

    config._attn_implementation = "sdpa"
    config._attn_implementation = {"text_config": "flash_attention_2"}

    assert config.text_config._attn_implementation == "flash_attention_2"
    assert config.vision_config._attn_implementation == "sdpa"
    # The one string a run would otherwise record names neither tower's kernel.
    assert config._attn_implementation == "sdpa"

    covered = samples["fa2_hub_fallback_qwen3_vl"]["backbones"]
    assert set(covered) == set(config.sub_configs)
    assert [name for name, entry in covered.items() if entry["requested"]] == ["text_config"]


# --- where the payload travels -----------------------------------------------


def test_the_fingerprint_survives_the_run_record_writer(tmp_path, samples):
    """Lane-g returns it inside the build fingerprint; the record carries that.

    `record.write_json` serialises with `default=str`, so a value that is not JSON
    native is not rejected — it degrades to its repr and the provenance silently
    becomes unusable. The round-trip is what catches that.
    """
    from trainbench.record import write_json

    record = {RUN_RECORD_KEY: {BUILD_FINGERPRINT_KEY: samples["fa2_hub_fallback_qwen3_vl"]}}
    path = write_json(tmp_path / "result.json", record)
    reloaded = json.loads(path.read_text())

    assert reloaded == record
    validate_kernel_fingerprint(reloaded[RUN_RECORD_KEY][BUILD_FINGERPRINT_KEY])
