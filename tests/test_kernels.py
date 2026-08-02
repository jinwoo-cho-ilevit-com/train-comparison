"""The gate for `trainbench/kernels.py`.

`tests/contract/test_kernel_provenance.py` carries no xfail and passes against an
empty `trainbench/kernels.py`, because it defines its own validator and never
imports the module. That is the shape this repository has been caught by nine
times: a green check with nothing under it. So the runtime module needs a gate of
its own, and the first thing that gate has to prove is that the two validators
reach the same verdict — otherwise the contract stays green while the runtime
admits a payload it refuses.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from trainbench import kernels

CONTRACT_PATH = Path(__file__).resolve().parent / "contract" / "test_kernel_provenance.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kernel_fingerprint.sample.json"


def _load_contract() -> Any:
    """Import the frozen contract by path; `tests/contract` is not a package."""
    spec = importlib.util.spec_from_file_location("_frozen_kernel_provenance", CONTRACT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load_contract()
SAMPLES: dict[str, dict[str, Any]] = json.loads(FIXTURE.read_text())["samples"]


def _sample(name: str) -> dict[str, Any]:
    return copy.deepcopy(SAMPLES[name])


def _mutate(name: str, edit) -> dict[str, Any]:
    fingerprint = _sample(name)
    edit(fingerprint)
    return fingerprint


def _drop(path: tuple[str, ...], key: str):
    def edit(fingerprint: dict[str, Any]) -> None:
        target = fingerprint
        for step in path:
            target = target[step]
        del target[key]

    return edit


def _set(path: tuple[str, ...], key: str, value: Any):
    def edit(fingerprint: dict[str, Any]) -> None:
        target = fingerprint
        for step in path:
            target = target[step]
        target[key] = value

    return edit


HUB = "fa2_hub_fallback_qwen3_vl"
UNREGISTERED = "fa3_hub_kernel_mask_unregistered_qwen3_vl"

# (id, payload, expected message fragment or None when the payload is well formed).
# The first block is every mutation the frozen contract performs, with the regex
# that contract asserts on; the second block is coverage the contract does not
# reach, where agreement is still the requirement.
CASES: list[tuple[str, Any, str | None]] = [
    ("sample_sdpa_builtin", _sample("sdpa_builtin_gemma4"), None),
    ("sample_fa2_hub", _sample(HUB), None),
    ("sample_fa3_unregistered", _sample(UNREGISTERED), None),
    (
        "hub_named_by_a_version_string",
        _mutate(
            HUB,
            _set(
                ("resolved", "identity"),
                "repo_id",
                None,
            ),
        ),
        "identified by repo_id and revision",
    ),
    (
        "hub_revision_went_missing",
        _mutate(HUB, _set(("resolved", "identity"), "revision", "")),
        "resolved.identity.revision=''",
    ),
    (
        "resolved_mask_flag_dropped",
        _mutate(UNREGISTERED, _drop(("resolved",), "mask_registered")),
        "resolved keys are frozen",
    ),
    (
        "backbone_mask_flag_dropped",
        _mutate(UNREGISTERED, _drop(("backbones", "text_config"), "mask_registered")),
        r"backbones\['text_config'\] keys are frozen",
    ),
    (
        "mask_flag_stopped_being_a_flag",
        _mutate(UNREGISTERED, _set(("backbones", "text_config"), "mask_registered", "false")),
        "must be a bool",
    ),
    (
        "backbones_collapsed_to_one_string",
        _mutate(UNREGISTERED, _set((), "backbones", "kernels-community/vllm-flash-attn3")),
        "backbones",
    ),
    ("backbones_emptied", _mutate(UNREGISTERED, _set((), "backbones", {})), "backbones"),
    (
        "backbones_values_are_strings",
        _mutate(
            UNREGISTERED,
            _set((), "backbones", {"text_config": "kernels-community/vllm-flash-attn3"}),
        ),
        "backbones",
    ),
    (
        "backbone_mislabelled_as_covered",
        _mutate(HUB, _set(("backbones", "vision_config"), "requested", True)),
        "not resolved.attn_implementation",
    ),
    (
        "request_reached_no_backbone",
        _mutate(
            HUB,
            _set(
                ("backbones",),
                "text_config",
                {"attn_implementation": "sdpa", "mask_registered": True, "requested": False},
            ),
        ),
        "request reached nothing",
    ),
    # --- beyond the contract's own mutations -------------------------------
    ("not_a_mapping", ["fingerprint"], "must be a mapping"),
    (
        "unexpected_top_level_key",
        _mutate(HUB, _set((), "kernel", "liger")),
        "keys are frozen",
    ),
    (
        "mask_registry_renamed",
        _mutate(HUB, _set(("resolved",), "mask_registry", "transformers.AttentionInterface")),
        "registry that decides mask creation",
    ),
    (
        "identity_source_invented",
        _mutate(HUB, _set(("resolved", "identity"), "source", "wheel")),
        "resolved.identity.source must be one of",
    ),
    (
        "builtin_claiming_a_repo",
        _mutate(
            "sdpa_builtin_gemma4",
            _set(("resolved", "identity"), "repo_id", "kernels-community/flash-attn2"),
        ),
        "did not come from the Hub",
    ),
    (
        "package_source_without_a_version",
        _mutate("sdpa_builtin_gemma4", _set(("resolved", "identity"), "package_version", "")),
        "identified by its distribution",
    ),
    (
        "provenance_of_a_different_kernel",
        _mutate(
            HUB, _set(("resolved",), "attn_implementation", "kernels-community/vllm-flash-attn3")
        ),
        "belongs to a different kernel",
    ),
    (
        "requested_axis_blanked",
        _mutate(HUB, _set(("requested",), "axis", "   ")),
        "must be a non-empty string",
    ),
    (
        "covered_backbone_contradicts_the_resolved_flag",
        _mutate(UNREGISTERED, _set(("backbones", "text_config"), "mask_registered", True)),
        "contradicts",
    ),
    (
        "resolved_implementation_blanked",
        _mutate(HUB, _set(("resolved",), "attn_implementation", "")),
        "must be a non-empty string",
    ),
]

CASE_IDS = [case[0] for case in CASES]


def _verdict(validator, payload: Any) -> str | None:
    try:
        validator(payload)
    except Exception as error:  # noqa: BLE001 - the verdict is what is compared
        return f"{type(error).__mro__[1].__name__}: {error}"
    return None


@pytest.mark.parametrize(("case_id", "payload", "expected"), CASES, ids=CASE_IDS)
def test_the_runtime_validator_agrees_with_contract(case_id, payload, expected):
    """Completion condition 2: same verdict on every case the contract exercises.

    A divergence here is invisible from either side — the contract stays green
    because it judges stored samples with its own copy of the rule, and the
    runtime stays green because nothing compares it to anything.
    """
    contract_refused = _verdict(contract.validate_kernel_fingerprint, payload)
    runtime_refused = _verdict(kernels.validate_fingerprint, payload)

    assert (contract_refused is None) == (runtime_refused is None), (
        f"{case_id}: contract said {contract_refused!r}, runtime said {runtime_refused!r}"
    )
    if expected is None:
        return
    # Agreeing on "refused" is not enough: the two must refuse it for the same
    # stated reason, or the runtime is rejecting the payload by accident.
    with pytest.raises(contract.ContractViolation, match=expected):
        contract.validate_kernel_fingerprint(copy.deepcopy(payload))
    with pytest.raises(kernels.UnidentifiedKernel, match=expected):
        kernels.validate_fingerprint(copy.deepcopy(payload))


def test_the_case_table_agrees_with_contract_about_what_is_well_formed():
    """Guards the guard: a table of only-invalid payloads would agree vacuously."""
    accepted = [case_id for case_id, _, expected in CASES if expected is None]
    refused = [case_id for case_id, _, expected in CASES if expected is not None]
    assert len(accepted) == len(SAMPLES) == 3
    assert len(refused) >= 20

    contract_mutations = sum(
        1
        for name, member in vars(contract).items()
        if name.startswith("test_") and callable(member)
    )
    assert contract_mutations >= 11


def test_the_two_modules_freeze_the_same_payload_names():
    """The keys travel between lanes as strings; a rename on one side is silent."""
    assert kernels.BUILD_FINGERPRINT_KEY == contract.BUILD_FINGERPRINT_KEY == "attention"
    assert kernels.RUN_RECORD_KEY == contract.RUN_RECORD_KEY
    assert kernels.MASK_REGISTRY == contract.MASK_REGISTRY
    assert kernels.FINGERPRINT_KEYS == contract.FINGERPRINT_KEYS
    assert kernels.REQUESTED_KEYS == contract.REQUESTED_KEYS
    assert kernels.RESOLVED_KEYS == contract.RESOLVED_KEYS
    assert kernels.IDENTITY_KEYS == contract.IDENTITY_KEYS
    assert kernels.BACKBONE_KEYS == contract.BACKBONE_KEYS
    assert kernels.IDENTITY_SOURCES == contract.IDENTITY_SOURCES


def test_isolation_agrees_with_contract_on_every_stored_sample():
    for name, fingerprint in SAMPLES.items():
        assert kernels.packing_isolation_holds(fingerprint) == contract.packing_isolation_holds(
            fingerprint
        ), name


# --- completion condition 3: packing is refused where no mask is built --------


def test_packing_is_refused_when_mask_registered_is_false():
    with pytest.raises(kernels.UnsafePacking) as refusal:
        kernels.assert_packing_is_isolated(_sample(UNREGISTERED))

    message = str(refusal.value)
    # The refusal has to carry the mechanism. "unsupported combination" would send
    # an operator looking for a missing package instead of a dropped mask.
    assert "silently disappears" in message
    assert "attention_mask=None" in message
    assert kernels.MASK_REGISTRY in message
    assert "text_config" in message
    # Naming the safe tower too would send the reader to the wrong one.
    assert "vision_config" not in message


def test_packing_is_allowed_when_every_backbone_is_mask_registered():
    for name in ("sdpa_builtin_gemma4", HUB):
        kernels.assert_packing_is_isolated(_sample(name))


def test_packing_refusal_reads_mask_registered_per_backbone():
    """One unregistered tower loses isolation for the whole forward.

    `resolved.mask_registered` describes the towers that got the request. A
    backbone that kept sdpa is registered, and a backbone that kept something
    else is not; reading the summary flag would pass this build.
    """
    fingerprint = _sample(HUB)
    fingerprint["backbones"]["vision_config"]["mask_registered"] = False
    assert fingerprint["resolved"]["mask_registered"] is True

    with pytest.raises(kernels.UnsafePacking, match="vision_config"):
        kernels.assert_packing_is_isolated(fingerprint)


def test_mask_registered_is_read_from_the_registry_that_decides_mask_creation():
    from transformers.masking_utils import AttentionMaskInterface

    module, _, attribute = kernels.MASK_REGISTRY.rpartition(".")
    assert module == "transformers.masking_utils"
    assert attribute == AttentionMaskInterface.__name__
    assert kernels._mask_registry() is AttentionMaskInterface._global_mapping
    assert "sdpa" in kernels._mask_registry()
    assert "kernels-community/flash-attn2" not in kernels._mask_registry()


# --- completion condition 4: no runtime kernel fetch on a pod -----------------


def _closed_env() -> dict[str, str]:
    return dict(kernels.RUNTIME_FETCH_ENV)


def test_no_runtime_fetch_reports_an_unset_environment():
    doors = kernels.open_fetch_doors({}, {})
    assert [door.split("=")[0] for door in doors] == ["$HF_HUB_OFFLINE", "$USE_HUB_KERNELS"]

    with pytest.raises(kernels.RuntimeKernelFetch, match="fetched over the network"):
        kernels.assert_no_runtime_kernel_fetch({}, {})


def test_no_runtime_fetch_accepts_a_closed_environment():
    assert kernels.open_fetch_doors(_closed_env(), {}) == []
    kernels.assert_no_runtime_kernel_fetch(_closed_env(), {})


def test_no_runtime_fetch_sees_past_an_environment_set_too_late():
    """The variables are read once, at import, into a module global.

    Setting `USE_HUB_KERNELS=NO` after transformers is imported changes nothing,
    and an environment-only check would call that closed. This is the case the
    research brief flagged and it is real: `hub_kernels.py:57-58` reads it at
    module level, `huggingface_hub/constants.py:202` likewise.
    """
    stale = types.ModuleType("transformers.integrations.hub_kernels")
    stale._TRANSFORMERS_USE_HUB_KERNELS = "YES"
    stale._kernels_enabled = True
    modules = {"transformers.integrations.hub_kernels": stale}

    doors = kernels.open_fetch_doors(_closed_env(), modules)
    assert len(doors) == 2
    assert all("cached at import" in door for door in doors)

    with pytest.raises(kernels.RuntimeKernelFetch):
        kernels.assert_no_runtime_kernel_fetch(_closed_env(), modules)


def test_no_runtime_fetch_closes_the_cached_globals_as_well_as_the_environment():
    stale = types.ModuleType("transformers.integrations.hub_kernels")
    stale._TRANSFORMERS_USE_HUB_KERNELS = "YES"
    stale._kernels_enabled = True
    hub = types.ModuleType("huggingface_hub.constants")
    hub.HF_HUB_OFFLINE = False
    layer = types.ModuleType("kernels.layer.globals")
    layer._DISABLE_KERNEL_MAPPING = False
    modules = {
        "transformers.integrations.hub_kernels": stale,
        "huggingface_hub.constants": hub,
        "kernels.layer.globals": layer,
    }
    env: dict[str, str] = {}

    was_open = kernels.forbid_runtime_kernel_fetch(env, modules)

    assert len(was_open) == 6
    assert env == kernels.RUNTIME_FETCH_ENV
    assert stale._kernels_enabled is False
    assert stale._TRANSFORMERS_USE_HUB_KERNELS == "NO"
    assert hub.HF_HUB_OFFLINE is True
    assert layer._DISABLE_KERNEL_MAPPING is True
    kernels.assert_no_runtime_kernel_fetch(env, modules)


def test_no_runtime_fetch_ignores_modules_that_were_never_imported():
    """A pod without `kernels` installed has no global to close, not an open door."""
    assert kernels.open_fetch_doors(_closed_env(), {}) == []


def test_no_runtime_fetch_by_environment_alone_does_not_close_the_flash_attention_rewrite():
    """`USE_HUB_KERNELS` is not on the path that rewrites fa2 to a repo id.

    `modeling_utils.py:1997-2003` gates the rewrite on `is_kernels_available()`
    only. On this host `kernels` is absent so the answer is False; what is asserted
    is that the predicate consults the package rather than the variable.
    """
    from transformers.utils.import_utils import is_kernels_available

    assert kernels.flash_attention_falls_back_to_the_hub("sdpa") is False
    assert kernels.flash_attention_falls_back_to_the_hub("flash_attention_2") is (
        bool(is_kernels_available()) and importlib.util.find_spec("flash_attn") is None
    )
    assert importlib.util.find_spec("kernels") is None
    assert is_kernels_available() is False


# --- reading the fingerprint off a built model --------------------------------


class _Model:
    def __init__(self, config: Any) -> None:
        self.config = config


def _qwen3_vl_config(implementation: Any) -> Any:
    """A built config: every sub-config carries a value before the request lands.

    `from_pretrained` leaves the model in this state. Constructing the config and
    assigning a dict straight onto it does not — the setter falls back to the
    sub-config's current value, and a fresh `Qwen3VLConfig` has `None` there.
    """
    from transformers import Qwen3VLConfig

    config = Qwen3VLConfig()
    config._attn_implementation = "sdpa"
    config._attn_implementation = implementation
    return config


def test_the_fingerprint_is_read_back_from_the_built_model():
    model = _Model(_qwen3_vl_config("sdpa"))

    fingerprint = kernels.read_fingerprint(model, axis="attn.name", value="sdpa", requested="sdpa")

    contract.validate_kernel_fingerprint(fingerprint)
    assert fingerprint["resolved"]["attn_implementation"] == "sdpa"
    assert fingerprint["resolved"]["identity"]["source"] == "builtin"
    assert fingerprint["resolved"]["identity"]["package"] == "transformers"
    assert fingerprint["resolved"]["mask_registered"] is True
    assert set(fingerprint["backbones"]) == {"text_config", "vision_config"}
    assert kernels.packing_isolation_holds(fingerprint)


def test_a_backbone_left_out_of_the_request_is_not_reported_as_covered():
    """The failure the per-sub-config map exists to prevent."""
    repo = "kernels-community/flash-attn2"
    model = _Model(_qwen3_vl_config({"text_config": repo}))

    fingerprint = kernels.read_fingerprint(
        model,
        axis="attn.name",
        value="fa2",
        requested={"text_config": repo},
        requested_ref="version=1",
        revision_resolver=lambda repo_id: "0" * 40,
    )

    contract.validate_kernel_fingerprint(fingerprint)
    assert fingerprint["resolved"]["identity"] == {
        "source": "hub",
        "package": None,
        "package_version": None,
        "repo_id": repo,
        "requested_ref": "version=1",
        "revision": "0" * 40,
    }
    assert fingerprint["backbones"]["text_config"]["requested"] is True
    assert fingerprint["backbones"]["vision_config"]["requested"] is False
    assert fingerprint["backbones"]["vision_config"]["attn_implementation"] == "sdpa"


def test_a_hub_kernel_whose_revision_cannot_be_read_is_refused():
    """Completion condition 1: the version of the library is not a substitute."""
    repo = "kernels-community/flash-attn2"
    model = _Model(_qwen3_vl_config({"text_config": repo}))

    with pytest.raises(kernels.UnidentifiedKernel, match="no revision came back"):
        kernels.read_fingerprint(
            model, axis="attn.name", value="fa2", requested={"text_config": repo}
        )


def test_an_inline_revision_on_the_implementation_string_identifies_the_kernel():
    repo = "kernels-community/vllm-flash-attn3"
    model = _Model(_qwen3_vl_config({"text_config": f"{repo}@{'1' * 40}"}))

    fingerprint = kernels.read_fingerprint(
        model, axis="attn.name", value="fa3", requested={"text_config": repo}
    )

    contract.validate_kernel_fingerprint(fingerprint)
    assert fingerprint["resolved"]["identity"]["repo_id"] == repo
    assert fingerprint["resolved"]["identity"]["revision"] == "1" * 40
    assert fingerprint["resolved"]["mask_registered"] is False
    with pytest.raises(kernels.UnsafePacking):
        kernels.assert_packing_is_isolated(fingerprint)


def test_a_request_that_bound_two_different_kernels_is_refused():
    model = _Model(_qwen3_vl_config({"text_config": "flash_attention_2", "vision_config": "sdpa"}))

    with pytest.raises(kernels.UnidentifiedKernel, match="do not agree on"):
        kernels.read_fingerprint(
            model,
            axis="attn.name",
            value="fa2",
            requested={"text_config": "flash_attention_2", "vision_config": "sdpa"},
        )


def test_the_top_level_key_does_not_reach_the_sub_configs():
    """`{"": impl}` asks the parent, and the parent does not pass it on.

    transformers reads each sub-config out of the dict by its own key —
    `value.get(subconfig_key, current_subconfig_attn)`, `configuration_utils.py`
    :401-417 — so the towers keep what they had. Reading this as "the parent
    dispatches to everything" put the request's identity on backbones that stayed on
    SDPA, and then died on an empty list with a message about two kernels.
    """
    config = _qwen3_vl_config({"": "eager"})
    assert config._attn_implementation == "eager"
    assert config.text_config._attn_implementation == "sdpa"
    assert config.vision_config._attn_implementation == "sdpa"

    with pytest.raises(kernels.UnidentifiedKernel, match="none of those keys names a backbone"):
        kernels.read_fingerprint(
            _Model(config), axis="attn.name", value="eager", requested={"": "eager"}
        )


def test_a_dict_naming_no_backbone_says_the_request_bound_nothing():
    """A multimodal dict handed to a text-only checkpoint.

    Qwen3 has no sub-configs, so its only backbone is the parent and the setter
    drops `text_config` entirely. The refusal has to name that, not report the
    backbones as disagreeing about a request none of them received.
    """
    from transformers import Qwen3Config

    config = Qwen3Config()
    assert config.sub_configs == {}
    config._attn_implementation = "sdpa"
    config._attn_implementation = {"text_config": "eager"}
    assert config._attn_implementation == "sdpa"

    with pytest.raises(kernels.UnidentifiedKernel, match="none of those keys names a backbone"):
        kernels.read_fingerprint(
            _Model(config), axis="attn.name", value="eager", requested={"text_config": "eager"}
        )


def test_the_top_level_key_lands_on_a_model_that_has_no_sub_configs():
    """The other half of the same setter: with no sub-configs the parent is the only
    backbone and `""` is the only key that reaches it."""
    from transformers import Qwen3Config

    config = Qwen3Config()
    config._attn_implementation = "sdpa"
    config._attn_implementation = {"": "eager"}
    assert config._attn_implementation == "eager"

    fingerprint = kernels.read_fingerprint(
        _Model(config), axis="attn.name", value="eager", requested={"": "eager"}
    )

    contract.validate_kernel_fingerprint(fingerprint)
    assert set(fingerprint["backbones"]) == {"qwen3"}
    assert fingerprint["backbones"]["qwen3"]["requested"] is True
    assert fingerprint["requested"]["attn_implementation"] == "eager"


def test_an_implementation_whose_package_is_absent_cannot_be_identified():
    """`flash_attention_2` with no `flash-attn` wheel names no kernel."""
    assert importlib.util.find_spec("flash_attn") is None
    model = _Model(_qwen3_vl_config("flash_attention_2"))

    with pytest.raises(kernels.UnidentifiedKernel, match="flash-attn"):
        kernels.read_fingerprint(
            model, axis="attn.name", value="fa2", requested="flash_attention_2"
        )


def test_a_tower_that_records_no_implementation_is_refused_not_dropped():
    """A sub-config left at `None` would otherwise vanish from the coverage map."""
    from transformers import Qwen3VLConfig

    config = Qwen3VLConfig()
    config.text_config._attn_implementation = "sdpa"
    assert config.vision_config._attn_implementation is None

    with pytest.raises(kernels.UnidentifiedKernel, match="'vision_config' records no"):
        kernels.read_fingerprint(_Model(config), axis="attn.name", value="sdpa", requested="sdpa")


def test_a_model_that_records_no_implementation_anywhere_is_refused():
    class _Bare:
        sub_configs: dict[str, Any] = {}

    with pytest.raises(kernels.UnidentifiedKernel, match="records no _attn_implementation"):
        kernels.read_fingerprint(_Model(_Bare()), axis="attn.name", value="sdpa", requested="sdpa")


def test_the_fingerprint_survives_the_run_record_writer(tmp_path):
    """Where the payload travels: `record.write_json` serialises with `default=str`."""
    from trainbench.record import write_json

    model = _Model(_qwen3_vl_config("sdpa"))
    fingerprint = kernels.read_fingerprint(model, axis="attn.name", value="sdpa", requested="sdpa")
    record = {kernels.RUN_RECORD_KEY: {kernels.BUILD_FINGERPRINT_KEY: fingerprint}}

    reloaded = json.loads(write_json(tmp_path / "result.json", record).read_text())

    assert reloaded == record
    kernels.validate_fingerprint(reloaded[kernels.RUN_RECORD_KEY][kernels.BUILD_FINGERPRINT_KEY])


def test_the_documented_selectors_reach_the_refusal_they_name():
    """`.plans/remaining-code/kernels.md` runs this file through three `-k` filters.

    Caught in the act: `-k mask_registered` originally matched only
    `..._is_allowed_when_every_backbone_is_mask_registered`, so disabling the
    refusal outright left that selector green. A selector that reaches only the
    permissive half of a rule is an empty check with a name.
    """
    names = [name for name in globals() if name.startswith("test_")]
    for selector, must_include in (
        ("agrees_with_contract", "test_the_runtime_validator_agrees_with_contract"),
        ("mask_registered", "test_packing_is_refused_when_mask_registered_is_false"),
        ("no_runtime_fetch", "test_no_runtime_fetch_reports_an_unset_environment"),
    ):
        selected = [name for name in names if selector in name]
        assert len(selected) >= 2, (selector, selected)
        assert must_include in selected, (selector, selected)


def test_this_gate_is_looking_at_the_module_it_claims_to():
    """The empty-input lock. A `trainbench/kernels.py` that lost these names would
    otherwise take this file down with an import error nobody reads as a finding."""
    assert kernels.__file__.endswith("trainbench/kernels.py")
    assert sys.modules["trainbench.kernels"] is kernels
    for name in (
        "validate_fingerprint",
        "packing_isolation_holds",
        "assert_packing_is_isolated",
        "read_fingerprint",
        "open_fetch_doors",
        "forbid_runtime_kernel_fetch",
        "assert_no_runtime_kernel_fetch",
    ):
        assert callable(getattr(kernels, name)), name
