"""The adapter registry: six frameworks through one entry point.

No checkpoint is downloaded here. What can be established on this host is the
shape of what an adapter hands back — the registry, the declarations, the build
fingerprint read off a module tree, and the refusals derived from it. Whether a
framework actually loads a checkpoint is a pod question and is recorded as one in
`.plans/notes/adapters.md`.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch

from trainbench import axes, loader
from trainbench.applied import FRAMEWORK_OWNABLE, FRAMEWORK_OWNED, AxisState
from trainbench.config import to_bench_config
from trainbench.config_schema import axis_knobs
from trainbench.device import get_device

from .conftest import CONFIG_DIR, REPO_ROOT


def compose_bench(*overrides):
    from hydra import compose, initialize_config_dir

    from trainbench.compose import resolve

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["run=probe", "device=cpu", *overrides])
        return resolve(cfg)[1]


@pytest.fixture(scope="module")
def config():
    return to_bench_config(compose_bench())


@pytest.fixture(scope="module")
def contract():
    """The frozen `loader-bench` validator, loaded by path.

    `tests/contract/` is not a package, and the point of reusing it is that the
    live object is judged by the same rules as the fixture rather than by a second
    opinion written here.
    """
    path = REPO_ROOT / "tests" / "contract" / "test_loader_bench.py"
    spec = importlib.util.spec_from_file_location("_contract_loader_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- a stand-in for a built checkpoint ----------------------------------------


def _qwen3_vl_config(implementation="sdpa"):
    """A built config: every sub-config carries a value before the request lands."""
    from transformers import Qwen3VLConfig

    config = Qwen3VLConfig()
    config._attn_implementation = "sdpa"
    config._attn_implementation = implementation
    return config


class _Tower(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2).to(dtype)


class _Build(torch.nn.Module):
    """Two towers and a rotary buffer, which is the shape every adapter returns."""

    def __init__(self, visual="bfloat16", language="bfloat16", implementation="sdpa"):
        super().__init__()
        self.config = _qwen3_vl_config(implementation)
        self.visual = _Tower(getattr(torch, visual))
        self.language_model = _Tower(getattr(torch, language))
        self.register_buffer("inv_freq", torch.zeros(2))


def _fingerprint(config, **kwargs):
    return loader.build_fingerprint(_Build(**kwargs), config)


# --- the registry -------------------------------------------------------------


def test_every_framework_the_schema_names_has_an_adapter():
    """Five of six were reachable by config and unable to produce a number."""
    assert set(loader.ADAPTERS) == loader.FRAMEWORKS
    assert all(name == adapter.name for name, adapter in loader.ADAPTERS.items())
    assert all(callable(adapter.load) for adapter in loader.ADAPTERS.values())


def test_the_registry_is_derived_from_the_schema_not_from_a_second_list():
    from typing import get_args

    from trainbench.config_schema import FrameworkConfig

    assert loader.FRAMEWORKS == frozenset(get_args(FrameworkConfig.model_fields["name"].annotation))


def test_load_routes_by_the_configured_framework(monkeypatch):
    """And the name on the way out is the adapter's literal, not the request."""
    seen = {}

    def _fake_load(cfg, device, load_kwargs):
        seen["load_kwargs"] = load_kwargs
        return _Build(), object()

    monkeypatch.setattr("trainbench.probe.native.load", _fake_load)
    monkeypatch.setitem(
        loader.ADAPTERS, "native", _adapter(honours_load_kwargs=True, aligns_padding_side=False)
    )
    mapping = json.loads(json.dumps(compose_bench()))
    mapping["framework"]["name"] = "native"
    bench_config = to_bench_config(mapping)

    out = loader.load(bench_config, get_device("cpu"))

    assert out.framework == "native"
    assert seen["load_kwargs"] == axes.load_kwargs(bench_config)


def test_only_the_paths_that_own_from_pretrained_are_handed_the_load_axes():
    """Where the framework owns the call the axis is left unapplied rather than
    guessed at a keyword, and the capture side reports the mismatch."""
    honoured = {n for n, a in loader.ADAPTERS.items() if a.honours_load_kwargs}

    assert honoured == {"native", "sentence_transformers"}


# --- the build fingerprint ----------------------------------------------------


def test_fingerprint_carries_the_five_blocks_the_contract_names(config):
    fingerprint = _fingerprint(config)

    assert set(fingerprint) == {
        "module_classes",
        "parameter_dtypes",
        "buffer_dtypes",
        "trainable_parameter_names",
        loader.BUILD_FINGERPRINT_KEY,
    }
    assert loader.BUILD_FINGERPRINT_KEY == "attention"


def test_fingerprint_reads_the_module_tree_off_the_built_object(config):
    fingerprint = _fingerprint(config, language="float32")

    assert fingerprint["module_classes"]["model"] == "_Build"
    assert fingerprint["module_classes"]["visual"] == "_Tower"
    # Grandchildren, which is where a wrapper stops looking like the backbone.
    assert fingerprint["module_classes"]["visual.proj"] == "Linear"
    assert fingerprint["parameter_dtypes"]["visual.proj.weight"] == "bfloat16"
    assert fingerprint["parameter_dtypes"]["language_model.proj.weight"] == "float32"
    assert fingerprint["buffer_dtypes"] == {"inv_freq": "float32"}


def test_fingerprint_stops_before_the_per_layer_depth(config):
    """A map that grew with the checkpoint would be a diff nobody can read."""
    deep = _Build()
    deep.visual.proj = torch.nn.Sequential(torch.nn.Linear(2, 2))

    classes = loader.module_classes(deep)

    assert "visual.proj" in classes
    assert not [name for name in classes if name.count(".") >= loader.MODULE_CLASS_DEPTH]


def test_fingerprint_asks_kernels_for_the_attention_block(config):
    fingerprint = _fingerprint(config)
    block = fingerprint[loader.BUILD_FINGERPRINT_KEY]

    assert set(block) == {"requested", "resolved", "backbones"}
    assert block["requested"]["axis"] == "attn.name"
    assert block["requested"]["value"] == config.attn.name
    assert block["resolved"]["attn_implementation"] == config.attn.impl


# --- the three builds that already happened -----------------------------------


def test_fingerprint_catches_a_fully_frozen_build(config):
    """unsloth froze every parameter and `infonce_backward` passed anyway.

    A step over a frozen graph still produces a number; that number is the speed of
    a model that learns nothing.
    """
    frozen = _Build()
    frozen.requires_grad_(False)

    assert loader.build_fingerprint(frozen, config)["trainable_parameter_names"] == []
    with pytest.raises(loader.AdapterRefusal, match="no trainable parameter"):
        loader.describe(loader.ADAPTERS["native"], frozen, object(), config)


def test_a_frozen_build_is_the_state_lora_has_not_attached_to_yet(config):
    """Every unsloth LoRA run died on the intermediate state the harness makes.

    `FastVisionModel.from_pretrained(full_finetuning=False)` freezes every
    parameter without a LoRA marker, and `axes._peft` attaches the LoRA inside
    `axes.assemble` — after this module. Refusing the frozen build at load time
    took out half the full-vs-LoRA comparison for one of six frameworks.
    """
    lora = to_bench_config(compose_bench("peft=lora"))
    frozen = _Build()
    frozen.requires_grad_(False)

    assert loader.attaches_an_adapter_later(lora)
    assert not loader.attaches_an_adapter_later(config)
    out = loader.describe(loader.ADAPTERS["unsloth"], frozen, object(), lora)
    assert out.fingerprint["trainable_parameter_names"] == []
    # Deferred, not dropped: the same check condemns the same build once nothing
    # is left to attach, which is the call the assembled model gets.
    with pytest.raises(loader.AdapterRefusal, match="no trainable parameter"):
        loader.refuse_a_build_the_fingerprint_condemns("unsloth", out.fingerprint, None)


def test_fingerprint_catches_two_modules_left_in_fp32(config):
    """axolotl leaves embed_tokens and lm_head in fp32 while the rest goes bf16.

    Measured without the context it trains in, it is a second numeric regime
    reported under one label — so the mixed build is refused unless the adapter
    declared one.
    """
    mixed = _Build(language="float32")
    fingerprint = loader.build_fingerprint(mixed, config)
    trainable = fingerprint["trainable_parameter_names"]

    assert len({fingerprint["parameter_dtypes"][n] for n in trainable}) > 1
    with pytest.raises(loader.AdapterRefusal, match="more than one dtype"):
        loader.describe(loader.ADAPTERS["native"], mixed, object(), config)
    # The same build under the adapter that declares the autocast is accepted.
    out = loader.describe(loader.ADAPTERS["axolotl"], mixed, object(), config)
    assert out.required_step_context.kind == loader.AUTOCAST


def test_fingerprint_catches_extra_tensors(config):
    """unsloth's gemma-4 build carried more tensors than native's.

    Counted over parameters and buffers together: a framework that adds tensors has
    no obligation to add parameters.
    """
    plain = _Build()
    padded = _Build()
    for i in range(60):
        padded.register_buffer(f"extra_{i}", torch.zeros(1))

    left = loader.build_fingerprint(plain, config)
    right = loader.build_fingerprint(padded, config)

    assert loader.tensor_count(right) - loader.tensor_count(left) == 60
    assert loader.fingerprint_diff(left, right)["tensor_count"] == (
        loader.tensor_count(left),
        loader.tensor_count(right),
    )


def test_fingerprint_diff_is_empty_between_two_identical_builds(config):
    """Otherwise every pair would look like a confound and none would be read."""
    assert loader.fingerprint_diff(_fingerprint(config), _fingerprint(config)) == {}


def test_fingerprint_diff_names_a_dtype_regime_difference(config):
    diff = loader.fingerprint_diff(_fingerprint(config), _fingerprint(config, language="float32"))

    assert diff["parameter_dtypes"] == (["bfloat16"], ["bfloat16", "float32"])
    assert "module_classes" not in diff


# --- tevatron owns its loss ---------------------------------------------------


def test_tevatron_owns_the_axes_its_forward_subsumes():
    """`EncoderModel.forward` scores, divides by temperature and computes
    cross-entropy itself, and gathers across devices when `is_ddp`."""
    adapter = loader.ADAPTERS["tevatron"]

    assert set(adapter.owned_axes) == {"loss.name", "parallel.cross_device_negatives"}
    assert adapter.step.owner == loader.FRAMEWORK
    assert adapter.step.callable == "tevatron.retriever.modeling.DenseModel.forward"
    assert adapter.step.batch_keys == ("query", "passage")


def test_tevatron_owns_only_axes_capture_lets_an_adapter_disclaim():
    """`applied.FRAMEWORK_OWNABLE` is the bound, and it is re-checked there.

    An adapter naming an axis outside it would have that axis filed as applied by
    this harness — the state the third one exists to prevent.
    """
    owned = loader.ADAPTERS["tevatron"].owned_axes

    assert set(owned) <= set(FRAMEWORK_OWNABLE)
    assert set(owned) <= set(axis_knobs())
    for axis in owned:
        state = AxisState(axis=axis, requested="x", applied=None, owner="tevatron")
        assert state.state == FRAMEWORK_OWNED
        assert not state.matches


def test_tevatron_owns_nothing_the_harness_still_runs():
    """The other five run the harness step, so none of them may disclaim an axis."""
    harness = [a for a in loader.ADAPTERS.values() if a.step.owner == loader.HARNESS]

    assert len(harness) == 5
    assert all(not adapter.owned_axes for adapter in harness)


# --- axolotl's autocast -------------------------------------------------------


def test_axolotl_autocast_is_required_by_the_adapter_and_established_elsewhere():
    """The adapter states the requirement; `axes.step_context` is what enters it.

    docs/CONTRACTS.md §2 fixes that as the only site, so an adapter that opened its
    own `with torch.autocast(...)` would put the precision regime somewhere the
    capture side does not read.
    """
    context = loader.ADAPTERS["axolotl"].required_step_context

    assert context.kind == loader.AUTOCAST
    assert (context.device_type, context.dtype) == ("cuda", "bfloat16")
    assert context.established_by == loader.ESTABLISHED_BY == "axes.step_context"
    assert callable(getattr(axes, loader.ESTABLISHED_BY.rpartition(".")[2]))


def test_axolotl_autocast_names_a_dtype_torch_has():
    context = loader.ADAPTERS["axolotl"].required_step_context

    assert isinstance(getattr(torch, context.dtype), torch.dtype)


def test_only_axolotl_declares_a_step_context():
    """native loads in pure bf16 and needs none; declaring one would be a second,
    different answer to the same question."""
    declared = {n for n, a in loader.ADAPTERS.items() if a.required_step_context is not None}

    assert declared == {"axolotl"}


def test_a_context_axes_step_context_cannot_establish_is_refused():
    with pytest.raises(loader.AdapterRefusal, match="not in \\['autocast'\\]"):
        loader.StepContext(kind="grad_scaler", device_type="cuda", dtype="bfloat16", reason="r")


def test_a_context_the_adapter_establishes_itself_is_refused():
    with pytest.raises(loader.AdapterRefusal, match="only place a precision context"):
        loader.StepContext(
            kind="autocast",
            device_type="cuda",
            dtype="bfloat16",
            reason="r",
            established_by="adapter",
        )


# --- documented entry points --------------------------------------------------

# A citation has to name a file. Prose about what a framework "usually" does is
# what did not survive contact with the pinned versions.
CITES_A_FILE = re.compile(r"[\w./-]+\.(py|md|txt)")


def test_documented_entry_point_is_declared_for_every_framework():
    for name, adapter in sorted(loader.ADAPTERS.items()):
        entry = adapter.documented_entry_point
        assert entry.framework.strip(), name
        assert entry.harness_uses.strip(), name
        assert CITES_A_FILE.search(entry.source), (name, entry.source)


def test_documented_entry_point_differs_wherever_the_two_strings_do():
    differing = {n for n, a in loader.ADAPTERS.items() if a.documented_entry_point.differs}

    # native is the reference path: it is this harness, so nothing differs from it.
    assert differing == loader.FRAMEWORKS - {"native"}
    assert loader.ADAPTERS["native"].documented_entry_point.differs is False


def test_an_unverified_entry_point_cannot_carry_a_verdict():
    with pytest.raises(loader.AdapterRefusal, match="cannot also carry a verdict"):
        loader.EntryPoint(
            framework="a", harness_uses="b", differs=True, source=f"{loader.UNVERIFIED} — open"
        )


def test_an_unknown_entry_point_has_to_be_written_as_one():
    with pytest.raises(loader.AdapterRefusal, match="has to be written as one"):
        loader.EntryPoint(framework="a", harness_uses="b", differs=None, source="cited.py:1")


def test_a_differing_entry_point_cannot_be_filed_as_identical():
    with pytest.raises(loader.AdapterRefusal, match="say otherwise"):
        loader.EntryPoint(framework="a", harness_uses="b", differs=False, source="cited.py:1")


IN_REPO = re.compile(r"(?:trainbench|tests|\.plans|scripts)/[\w./-]+\.(?:py|md)")


def test_every_citation_that_points_into_this_repo_points_at_something():
    """A citation pointing at nothing is the docstring that claimed to follow
    axolotl's own order while inverting it."""
    checked = []
    for name, adapter in sorted(loader.ADAPTERS.items()):
        for path in IN_REPO.findall(adapter.documented_entry_point.source):
            checked.append(path)
            assert (REPO_ROOT / path).exists(), (name, path)

    assert len(checked) >= len(loader.ADAPTERS)


# --- the live object satisfies the frozen contract ----------------------------


def test_a_live_adapter_out_passes_the_frozen_contract_validator(config, contract):
    """Same rules as the sample, applied to what an adapter really hands back."""
    problems = {}
    for name, adapter in sorted(loader.ADAPTERS.items()):
        build = _Build(language="float32") if adapter.required_step_context else _Build()
        out = loader.describe(adapter, build, _Build(), config)
        payload = out.to_payload()
        assert set(payload) == set(contract.PAYLOAD_KEYS)
        problems[name] = contract.validate(name, payload)

    assert problems == dict.fromkeys(loader.ADAPTERS, [])


def test_adapter_out_carries_the_eight_fields_the_boundary_names(contract):
    assert loader.ADAPTER_OUT_FIELDS == contract.ADAPTER_OUT_FIELDS


# --- declarations that would file an axis as ours -----------------------------


def _adapter(**overrides):
    base = {
        "name": "native",
        "step": loader.HARNESS_STEP,
        "documented_entry_point": loader.ADAPTERS["native"].documented_entry_point,
    }
    return loader.Adapter(**{**base, **overrides})


def test_an_adapter_cannot_disclaim_an_axis_that_is_not_one():
    with pytest.raises(loader.AdapterRefusal, match="non-axes"):
        _adapter(step=loader.ADAPTERS["tevatron"].step, owned_axes={"loss.everything": "invented"})


def test_an_adapter_cannot_disclaim_an_axis_capture_will_not_let_it():
    with pytest.raises(loader.AdapterRefusal, match="FRAMEWORK_OWNABLE"):
        _adapter(step=loader.ADAPTERS["tevatron"].step, owned_axes={"framework.name": "mine"})


def test_a_framework_owned_step_that_disclaims_nothing_is_refused():
    with pytest.raises(loader.AdapterRefusal, match="declaring none files it as applied"):
        _adapter(step=loader.ADAPTERS["tevatron"].step)


def test_a_harness_step_that_disclaims_an_axis_is_refused():
    with pytest.raises(loader.AdapterRefusal, match="cannot hand an axis to a step it runs"):
        _adapter(owned_axes={"loss.name": "not ours"})


def test_a_framework_owned_step_has_to_name_its_callable():
    with pytest.raises(loader.AdapterRefusal, match="name the callable"):
        loader.Step(owner=loader.FRAMEWORK, callable=None, batch_keys=("query",))


def test_a_harness_step_names_no_framework_callable():
    with pytest.raises(loader.AdapterRefusal, match="names no framework callable"):
        loader.Step(owner=loader.HARNESS, callable="x.y", batch_keys=("input_ids",))


# --- what bench.py reads off the result ---------------------------------------


def test_bench_reads_the_binding_fields_this_module_produces():
    """`scripts/bench.py` substitutes `AdapterOut` for its own `Binding`, so the
    two field lists moving apart would leave the seam calling native forever."""
    source = (REPO_ROOT / "scripts" / "bench.py").read_text(encoding="utf-8")
    binding = source.partition("class Binding(NamedTuple):")[2].partition("\ndef ")[0]
    declared = set(re.findall(r"^    (\w+):", binding, flags=re.MULTILINE))

    assert declared == loader.ADAPTER_OUT_FIELDS
    assert "trainbench.loader" in source


def test_an_adapter_refusal_is_caught_by_the_block_that_wraps_the_load():
    """`bench.build_run` loads inside `refusing('load_kwargs')`, whose catch list is
    the two refusal types — everything else leaves no result file at all.

    A refusal that escapes it is published as a pod that crashed rather than as a
    setting this harness declined, which is the difference the block exists for.
    """
    spec = importlib.util.spec_from_file_location(
        "_bench_for_refusal", REPO_ROOT / "scripts" / "bench.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    with pytest.raises(bench.RefusedSetting) as refused:
        with bench.refusing("load_kwargs"):
            raise loader.AdapterRefusal("a build this harness must not put a number on")

    assert type(refused.value.cause) is loader.AdapterRefusal


def test_the_fingerprint_is_the_shape_a_run_record_carries(config, contract):
    """The lane's axis-G deliverable has to survive the trip into a result file.

    `RUN_RECORD_KEY` is named here rather than spelled again by the record writer,
    and the payload is judged by the frozen validator rather than by a second
    opinion written in this file.
    """
    out = loader.describe(loader.ADAPTERS["native"], _Build(), object(), config)
    carried = json.loads(json.dumps({loader.RUN_RECORD_KEY: dict(out.fingerprint)}))

    assert loader.RUN_RECORD_KEY == "build_fingerprint"
    payload = {**out.to_payload(), "fingerprint": carried[loader.RUN_RECORD_KEY]}
    assert contract.validate("native", payload) == []


FRAMEWORK_CALL = ("from_pretrained", "ModelLoader", "get_model_processor", "SentenceTransformer")


@pytest.mark.parametrize("name", sorted(loader.ADAPTERS))
def test_every_adapter_load_is_a_real_function_not_a_stub(name):
    """A registry of six lambdas would satisfy the contract and load nothing.

    Read off the source because reaching the call needs the framework installed,
    and each one installs only inside its own image.
    """
    module = importlib.import_module(loader.ADAPTERS[name].module)
    source = Path(module.__file__).read_text(encoding="utf-8")
    body = source.partition("\ndef load(")[2].partition("\ndef ")[0]
    helpers = set(re.findall(r"^def (\w+)\(", source, re.M)) - {"load", "run"}

    assert callable(module.load)
    assert any(call in source for call in FRAMEWORK_CALL), name
    # Either it makes the call or it hands off to a helper in the same module;
    # what it may not be is a body that reaches neither.
    assert any(call in body for call in FRAMEWORK_CALL) or any(
        f"{helper}(" in body for helper in helpers
    ), body


def test_the_framework_imports_stay_out_of_this_module():
    """`doc-commands` demands every third-party import under `trainbench/` of the
    root lock and exempts exactly the six per-image adapters.

    Six frameworks in one lockfile is the resolution `envs/` exists because nobody
    can satisfy, so a `from unsloth import ...` here would turn the documented
    `uv sync` into a command that cannot run.
    """
    source = (REPO_ROOT / "trainbench" / "loader.py").read_text(encoding="utf-8")
    frameworks = ("unsloth", "swift", "axolotl", "tevatron", "sentence_transformers")

    assert not [
        name for name in frameworks if re.search(rf"^\s*(from|import) {name}\b", source, re.M)
    ]
    assert loader.PROBE_PACKAGE == "trainbench.probe"


def _module_level_functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _plain_calls(node):
    """Every `name(...)` under `node`, nested functions and lambdas included.

    Bare names only. `steps.load_kwargs(config, report)` is an attribute call on an
    imported module and is exactly what this must not count: it satisfied the
    regex this test used to be, so the check stayed green over a `run` that had
    stopped calling the shared build at all.
    """
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


@pytest.mark.parametrize("name", sorted(loader.ADAPTERS))
def test_the_probe_and_the_harness_take_the_same_load(name):
    """Two definitions of `full_finetuning=` or of axolotl's validate/normalize
    order is what a whole campaign cost.

    `loader.Adapter.load` calls this module's `load`, so `run` has to reach that
    same function — either directly, or through every module-level helper `load`
    itself calls, which is how native takes the processor and the model as two
    checks.
    """
    module = importlib.import_module(loader.ADAPTERS[name].module)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    functions = _module_level_functions(tree)

    entry = functions["load"]
    helpers = _plain_calls(entry) & set(functions)
    reached = _plain_calls(functions["run"]) & ({"load"} | helpers)

    assert reached, f"{name}.run calls neither load nor any helper of it: {sorted(helpers)}"
    if "load" not in reached:
        assert reached >= helpers, f"{name}.run takes {sorted(reached)} of {sorted(helpers)}"
