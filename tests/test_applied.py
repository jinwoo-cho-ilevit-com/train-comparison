"""An axis is certified only when something actually looked at it."""

from __future__ import annotations

import json
from enum import Enum
from types import SimpleNamespace
from typing import get_args

import pytest
import torch

from trainbench import applied as applied_module
from trainbench import axes
from trainbench.applied import (
    ADAPTER_PARAM_MARKER,
    AppliedMismatch,
    AppliedState,
    AxisState,
    Built,
    assert_matches,
    capture,
)
from trainbench.config import to_bench_config
from trainbench.config_schema import BenchConfig, axis_knobs
from trainbench.probe.types import Check, ProbeReport

CPU = torch.device("cpu")


def torch_model():
    """A real nn.Module: assemble() builds an optimizer from its parameters."""
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    return model


@pytest.fixture
def config_mapping():
    from hydra import compose, initialize_config_dir

    from trainbench.compose import resolve

    from .conftest import CONFIG_DIR

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return resolve(compose(config_name="config", overrides=["device=cpu"]))[1]


def bench(config_mapping, **overrides):
    mapping = json.loads(json.dumps(config_mapping))
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        mapping[section][key] = value
    return to_bench_config(mapping)


def fake_model(attn_impl: str | None, vision_impl: str | None = None, params=()):
    """A model shaped like the ones under test: a tower with its own config.

    The previous fixture was one flat namespace, which could not represent a
    submodule disagreeing with the top level — so the check that missed exactly
    that could not have been caught by the tests either.
    """
    modules = []
    if vision_impl is not None:
        tower = SimpleNamespace(config=SimpleNamespace(_attn_implementation=vision_impl))
        modules.append(("visual", tower))
    return SimpleNamespace(
        config=SimpleNamespace(_attn_implementation=attn_impl, sub_configs=()),
        named_modules=lambda: modules,
        named_parameters=lambda: iter(params),
    )


def built(attn_impl="sdpa", vision_impl=None, params=(), **pieces):
    """What a run constructed. Axes live on more than the model — the optimizer
    decides optim.name, the loss decides loss.name — so capture takes all of it."""
    return Built(model=fake_model(attn_impl, vision_impl, params), **pieces)


def axis(state, name):
    return next(a for a in state.axes if a.axis == name)


def test_attn_is_read_back_from_the_model(config_mapping):
    state = capture(built(), bench(config_mapping))

    assert axis(state, "attn.name").applied == "sdpa"
    assert axis(state, "attn.name").matches


def test_silent_fallback_is_caught(config_mapping):
    """Asking for flash_attention_3 and getting sdpa is the exact failure this
    module exists for: a plausible number under a wrong label."""
    config = bench(config_mapping, **{"attn.name": "fa3"})

    state = capture(built(), config)

    assert not axis(state, "attn.name").matches
    with pytest.raises(AppliedMismatch, match="requested 'flash_attention_3', applied 'sdpa'"):
        assert_matches(state, config)


def test_vision_tower_left_on_sdpa_is_a_mismatch(config_mapping):
    """transformers updates the top-level config, then warns and moves on for any
    submodule that cannot take the implementation. Every model here is multimodal,
    so a flash-attention language model over an sdpa vision tower is the realistic
    partial application — and reading only the top level calls it a match."""
    config = bench(config_mapping, **{"attn.name": "fa2"})

    state = capture(built("flash_attention_2", vision_impl="sdpa"), config)

    attn = axis(state, "attn.name")
    assert attn.applied == "mixed(flash_attention_2,sdpa)"
    assert not attn.matches
    assert attn.detail["implementations"] == {"flash_attention_2": 1, "sdpa": 1}
    assert attn.detail["dissenting"] == ["visual"]


def test_uniform_submodules_still_match(config_mapping):
    config = bench(config_mapping, **{"attn.name": "fa2"})

    state = capture(built("flash_attention_2", vision_impl="flash_attention_2"), config)

    assert axis(state, "attn.name").matches


# Every knob that selects an optimisation, pinned by name. Comparing the captured
# set against `axis_knobs()` instead would be a tautology: `capture` iterates that
# very function and appends in all three of its branches, so the assertion holds
# however the schema changes. Spelling them out makes adding or dropping a marker
# show up as a diff in this file.
AXES = (
    "attn.name",
    "compile.mode",
    "dataloader.backend",
    "dataloader.packing",
    "dataloader.pretokenize",
    "framework.name",
    "freeze.ple",
    "freeze.vision_tower",
    "kernel.name",
    "loss.name",
    "optim.name",
    "parallel.cross_device_negatives",
    "parallel.strategy",
    "peft.mode",
    "precision.name",
    "train.gradient_checkpointing",
    "train.offload",
)


def test_the_axis_set_is_exactly_this(config_mapping):
    """A hand-written list in the source fails open: an axis missing from it is
    not undetermined, it does not exist. The set is derived from the schema, and
    this is where a change to it has to be argued for."""
    assert sorted(axis_knobs()) == sorted(AXES)

    state = capture(built(), bench(config_mapping))

    assert sorted(a.axis for a in state.axes) == sorted(AXES)


def test_an_axis_without_a_probe_blocks_a_timing_run(config_mapping):
    """Undetermined must not read as fine, or the mechanism is decorative.

    The undetermined axis is synthetic rather than whichever real axis happens to
    be unwired today. Naming a real one couples this test to the axis inventory,
    which `audit_plan.py`'s `axis-wired` already tracks — and it would break here
    every time a lane does its job, until someone stops reading the failure and
    just edits the name. A synthetic axis also cannot go vacuous: deriving the
    subject from `axis_knobs() - _CAPTURES` would silently check nothing once
    every axis is wired.
    """
    config = bench(config_mapping)
    state = capture(built(), config)
    state = AppliedState(
        axes=(*state.axes, AxisState(axis="synthetic.unwired", requested="x", applied=None))
    )

    assert "synthetic.unwired" in {a.axis for a in state.undetermined()}
    with pytest.raises(AppliedMismatch, match="synthetic.unwired"):
        assert_matches(state, config)


def test_probe_runs_are_not_blocked(config_mapping):
    """A probe answers 'does it run', so unverified axes are acceptable there.

    Paired with the timing case above: on its own this passes even if
    assert_matches were an empty function.
    """
    state = capture(built(), bench(config_mapping))

    assert_matches(state, bench(config_mapping, **{"run.purpose": "probe"}))
    assert_matches(state, bench(config_mapping, **{"run.purpose": "profile", "run.profiler": True}))


def test_a_state_that_says_nothing_does_not_pass(config_mapping):
    """An empty state has no mismatches and no undetermined axes. Reading that as
    success would let a failed capture through as a clean run."""
    config = bench(config_mapping)

    with pytest.raises(AppliedMismatch, match="never captured"):
        assert_matches(AppliedState(()), config)


def test_a_partial_state_does_not_pass(config_mapping):
    config = bench(config_mapping)
    only_attn = AppliedState((AxisState("attn.name", "sdpa", "sdpa"),))

    with pytest.raises(AppliedMismatch, match="never captured"):
        assert_matches(only_attn, config)


def test_unknown_purpose_is_an_error_not_a_pass(config_mapping):
    """The enforced set is a membership test, so a value outside it is silently
    exempt. Anything not in the schema's own list has to be rejected loudly."""
    config = bench(config_mapping)
    forged = SimpleNamespace(run=SimpleNamespace(purpose="Timing"))

    with pytest.raises(ValueError, match="unknown run purpose"):
        assert_matches(capture(built(), config), forged)


def test_unreadable_model_is_undetermined_not_crash(config_mapping):
    state = capture(Built(model=object()), bench(config_mapping))

    attn = axis(state, "attn.name")
    assert attn.applied is None
    assert "reason" in attn.detail


def test_a_probe_that_raises_is_undetermined_not_crash(config_mapping, monkeypatch):
    from trainbench import applied

    def explode(model, config):
        raise RuntimeError("cuda is on fire")

    monkeypatch.setitem(applied._CAPTURES, "attn.name", explode)
    state = capture(built(), bench(config_mapping))

    assert axis(state, "attn.name").applied is None
    assert "cuda is on fire" in axis(state, "attn.name").detail["reason"]


def test_a_config_of_the_wrong_shape_is_undetermined_not_crash():
    """capture must survive anything: it runs on the failure path, where the
    thing that is wrong may well be the config itself."""
    state = capture(built(), SimpleNamespace())

    assert state.axes, "axes are still enumerated"
    assert all(a.applied is None for a in state.axes)


def test_applied_state_serialises_for_the_record(config_mapping):
    """This dict is what a result JSON carries; without it the file records the
    request and nothing about what ran."""
    payload = capture(built(), bench(config_mapping)).to_dict()

    assert json.loads(json.dumps(payload))["all_matched"] is True
    entry = next(a for a in payload["axes"] if a["axis"] == "attn.name")
    assert entry == {
        "axis": "attn.name",
        "requested": "sdpa",
        "applied": "sdpa",
        "owner": None,
        "state": "applied",
        "determined": True,
        "matches": True,
        "detail": {"implementations": {"sdpa": 1}, "modules_checked": 1},
    }
    assert payload["framework_owned"] == []


def test_the_attention_detail_does_not_grow_with_the_model():
    """A real VLM records the implementation once per layer — twelve places on a
    tiny test model, hundreds on a 5B one. Carrying that map in every result file
    would make the evidence the largest thing in it."""
    from trainbench.applied import _capture_attn

    wide = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa", sub_configs=()),
        named_modules=lambda: [
            (
                f"layers.{i}.attn",
                SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa")),
            )
            for i in range(500)
        ],
    )

    _, detail = _capture_attn(Built(model=wide), None)

    assert detail == {"implementations": {"sdpa": 501}, "modules_checked": 501}
    assert len(json.dumps(detail)) < 200


def param(name: str, requires_grad: bool):
    return name, SimpleNamespace(requires_grad=requires_grad, requires_grad_=lambda flag: None)


def gemma(config_mapping, **overrides):
    return bench(
        config_mapping,
        **{
            "model.arch": "gemma4",
            "model.padding_side": "left",
            "model.max_tokens_per_image": 280,
            "model.instruction_prompt": None,
        },
        **overrides,
    )


PLE_FROZEN = param("embed_tokens_per_layer.weight", False)
PLE_TRAINING = param("layers.0.per_layer_input_gate", True)


def test_freezing_nothing_is_not_a_successful_freeze(config_mapping):
    """`_ple_report` reported ok=True on zero matches, so an upstream rename would
    have read as a freeze of 2.39B parameters that were in fact still training."""
    state = capture(built(params=[param("model.layers.0.mlp.weight", True)]), gemma(config_mapping))

    ple = axis(state, "freeze.ple")

    assert ple.applied is None
    assert ple.detail["matched"] == 0


def test_a_half_applied_freeze_is_a_mismatch(config_mapping):
    config = gemma(config_mapping, **{"freeze.ple": True})

    state = capture(built(params=[PLE_FROZEN, PLE_TRAINING]), config)

    assert axis(state, "freeze.ple").applied == "partial"
    with pytest.raises(AppliedMismatch, match="freeze.ple"):
        assert_matches(state, config)


def test_a_freeze_that_took_matches(config_mapping):
    config = gemma(config_mapping, **{"freeze.ple": True})
    frozen = [PLE_FROZEN, param("layers.0.per_layer_input_gate", False)]

    state = capture(built(params=frozen), config)

    assert axis(state, "freeze.ple").matches


def test_expected_failure_does_not_condemn_the_report():
    report = ProbeReport(framework="unsloth", model="m")
    report.add(Check(name="load", ok=True))
    report.add(Check(name="fast_st_accepts_vlm", ok=False, expected_failure=True))

    assert report.all_ok

    report.add(Check(name="real_problem", ok=False))
    assert not report.all_ok


def test_a_documented_limitation_that_starts_working_is_surfaced():
    """all_ok cannot say this: a passing expected-failure is not a failure. But it
    means the support matrix is now wrong, and only the run knows."""
    report = ProbeReport(framework="unsloth", model="m")
    report.run("fast_st_accepts_vlm", lambda: {"loaded": True}, expected_failure=True)

    assert report.all_ok
    assert report.unexpected_passes == ["fast_st_accepts_vlm"]
    assert report.to_dict()["unexpected_passes"] == ["fast_st_accepts_vlm"]


def test_applied_and_verified_sets_agree():
    """An axis that can be applied but not read back is the failure applied.py
    exists for; one that can be read back but not applied certifies a default."""
    from trainbench.applied import _CAPTURES

    assert set(_CAPTURES) == set(axes.IMPLEMENTED)
    assert set(_CAPTURES) <= set(axis_knobs())


def test_an_axis_on_the_optimizer_is_verified_from_the_optimizer(config_mapping):
    """Half the axes are not properties of the model. A capture that only saw the
    model would report them undetermined forever, or worse, be widened to guess."""
    config = bench(config_mapping)
    params = [torch.nn.Parameter(torch.zeros(2))]
    fused = torch.optim.AdamW(params, lr=1e-5)
    fused.param_groups[0]["fused"] = True

    state = capture(built(optimizer=fused), config)

    assert axis(state, "optim.name").applied == "adamw_fused"
    assert axis(state, "optim.name").matches


def test_an_unfused_adamw_is_not_the_fused_axis(config_mapping):
    """`adamw_fused` names a CUDA-only kernel. A CPU run builds the same class
    without it, and reporting that under the same name would put an unfused
    number in the fused row."""
    config = bench(config_mapping)

    state = capture(
        built(optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))])), config
    )

    assert axis(state, "optim.name").applied == "adamw_unfused"
    assert not axis(state, "optim.name").matches


def test_a_run_that_built_no_optimizer_is_undetermined(config_mapping):
    state = capture(built(), bench(config_mapping))

    assert axis(state, "optim.name").applied is None
    assert axis(state, "optim.name").detail["reason"] == "no optimizer was built"


def optimizer_named(class_name: str, groups):
    """An optimizer of a class with this exact name. The class name is how capture
    recognises an optimizer from a package that does not install on every host."""
    instance = type(class_name, (), {})()
    instance.param_groups = list(groups)
    instance.state = {}
    return instance


def test_an_optimizer_class_the_table_does_not_name_is_undetermined(config_mapping):
    """Deriving the value from the class name is a second mapping into this axis's
    vocabulary, and it reaches past the Newton-Schulz reading, which knows one exact
    spelling: a class named `MUON` spelt the requested value without a single param
    group being looked at, so an optimizer whose orthogonalised side is empty could
    publish its number as Muon's."""
    config = bench(config_mapping, **{"optim.name": "muon"})
    optimizer = optimizer_named("MUON", [{"params": [torch.nn.Parameter(torch.zeros(2, 2))]}])

    state = capture(built(optimizer=optimizer), config)

    assert axis(state, "optim.name").applied is None
    assert not axis(state, "optim.name").matches
    assert "MUON" in axis(state, "optim.name").detail["reason"]
    with pytest.raises(AppliedMismatch, match="optim.name"):
        assert_matches(state, config)


def test_a_muon_that_orthogonalises_is_named_by_the_table(config_mapping):
    """The other half of the same guard: `OPTIM_CLASS_AXIS['Muon']` is the only way
    to this value, so deleting that row has to stop certifying a real Muon run."""
    config = bench(config_mapping, **{"optim.name": "muon"})
    groups = [{"params": [torch.nn.Parameter(torch.zeros(2, 2))], "use_muon": True}]

    state = capture(built(optimizer=optimizer_named("Muon", groups)), config)

    assert axis(state, "optim.name").applied == "muon"
    assert axis(state, "optim.name").matches
    assert axis(state, "optim.name").detail["newton_schulz_tensors"] == 1


def test_the_loss_axis_is_read_off_the_loss_that_was_built(config_mapping):
    """GradCache is the case this exists for: plain in-batch negatives must not be
    able to report a cached_mnrl speedup for work that was never done."""
    config = bench(config_mapping)
    loss, names = axes._loss(config)

    state = capture(built(loss_fn=loss), config)

    assert names == ["loss.name"]
    assert axis(state, "loss.name").matches


def test_a_loss_that_declares_nothing_is_undetermined(config_mapping):
    state = capture(built(loss_fn=lambda q, d: q), bench(config_mapping))

    assert axis(state, "loss.name").applied is None
    assert "axis_value" in axis(state, "loss.name").detail["reason"]


UNIMPLEMENTED_AXES = (
    # `optim.name: muon` was here. It is applied now (trainbench/axes.py::_optimizer,
    # pytorch-optimizer's Muon) and its cases moved to tests/test_axes.py, where the
    # refusal that remains is conditional — a model with no >=2D parameter, for which
    # every tensor would take the internal AdamW path under Muon's name.
    {"optim.name": "adamw_8bit"},
    # `loss.name: cached_mnrl` was here. It is applied now
    # (trainbench/axes.py::_loss, `gradcache_backward`) and its cases moved to
    # tests/test_axes.py, where what remains is conditional — a batch whose
    # tensors cannot be attributed to rows, and the plain `(queries, documents)`
    # signature, which GradCache cannot answer at all.
    {"dataloader.backend": "dali"},
    # `dataloader.packing` and `dataloader.pretokenize` were here. Both are applied
    # now (trainbench/axes.py::PackedCollate, ::pretokenize) and their refusal cases
    # moved to tests/test_axes.py, where they are conditional on what the dataset
    # carries rather than unconditional.
    {"parallel.strategy": "zero3"},
    {"train.offload": "optimizer"},
    # `parallel.cross_device_negatives: True` was here. It is applied now
    # (trainbench/axes.py::_gather_with_grad) and its cases moved to
    # tests/test_axes.py. The refusal that remains is not `assemble`'s: building
    # the gathering closure is what applies the axis, and a process with no world
    # to gather from raises at the first step instead — before a number.
)


@pytest.mark.parametrize("override", UNIMPLEMENTED_AXES)
def test_an_axis_value_with_no_implementation_is_refused_not_substituted(config_mapping, override):
    """Returning the default under the requested name is the failure the whole
    module exists to prevent, and it is not less of one in our own code."""
    config = bench(config_mapping, **override)

    with pytest.raises(axes.UnappliedAxis):
        axes.assemble(torch_model(), config, CPU, framework="native")


def test_assemble_hands_back_the_model(config_mapping):
    """peft, torch.compile and FSDP all replace the model rather than mutate it,
    and deepspeed.initialize returns the model, optimizer and dataloader from one
    call. A hook that only mutated in place could not express any of them."""
    model = torch_model()

    result, applied_names = axes.assemble(model, bench(config_mapping), CPU, framework="native")

    assert result.model is model
    assert result.optimizer is not None
    assert result.loss_fn is not None
    assert set(applied_names) == {"optim.name", "loss.name", "framework.name"}


def test_the_framework_axis_is_the_adapter_that_ran_not_the_one_requested(config_mapping):
    """A registry that routed framework=unsloth to the native path would publish
    native numbers in the unsloth row, and nothing in the result would say so."""
    config = bench(config_mapping, **{"framework.name": "unsloth"})
    misrouted, _ = axes.assemble(torch_model(), config, CPU, framework="native")

    state = capture(misrouted, config)

    assert axis(state, "framework.name").applied == "native"
    assert not axis(state, "framework.name").matches
    with pytest.raises(AppliedMismatch, match="framework.name"):
        assert_matches(state, config)


def test_an_adapter_that_declares_nothing_is_undetermined(config_mapping):
    state = capture(built(), bench(config_mapping))

    assert axis(state, "framework.name").applied is None


# --- precision.name -----------------------------------------------------------
# The load dtype is chosen by `probe/steps.py::dtype_for`, so unlike every other
# axis the request is not applied by `axes.py` at all. That is what makes reading
# it back off the weights the whole of the check.


def bf16_model(dtype=torch.bfloat16, vision_dtype=None):
    """A model whose parameters are in a dtype, with a tower to disagree with."""
    model = torch.nn.Module()
    model.language_model = torch.nn.Linear(2, 2).to(dtype)
    model.visual = torch.nn.Linear(2, 2).to(vision_dtype or dtype)
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    return model


def test_precision_is_read_off_the_weights_not_off_the_request(config_mapping):
    config = bench(config_mapping)

    state = capture(Built(model=bf16_model()), config)

    assert axis(state, "precision.name").applied == "bf16"
    assert axis(state, "precision.name").matches
    assert axis(state, "precision.name").detail["base"] == {"bf16": 4}


def test_a_model_loaded_in_fp32_does_not_pass_as_bf16(config_mapping):
    """`dtype_for` returns fp32 on anything that is not CUDA, so a bf16 request
    measured on the wrong machine is the ordinary way this goes wrong — and the
    config says bf16 either way."""
    config = bench(config_mapping)

    state = capture(Built(model=bf16_model(torch.float32)), config)

    assert axis(state, "precision.name").applied == "fp32"
    with pytest.raises(AppliedMismatch, match="precision.name: requested 'bf16', applied 'fp32'"):
        assert_matches(state, config)


def test_a_half_converted_model_is_neither_precision(config_mapping):
    """A tower left in fp32 under a bf16 language model is the same partial
    application the attention probe exists for, and it must not match the dtype
    that happens to be commonest."""
    config = bench(config_mapping)

    state = capture(Built(model=bf16_model(vision_dtype=torch.float32)), config)

    assert axis(state, "precision.name").applied == "mixed(bf16,fp32)"
    with pytest.raises(AppliedMismatch, match="precision.name"):
        assert_matches(state, config)


def test_lora_adapter_weights_do_not_make_a_bf16_run_mixed(config_mapping):
    """peft holds adapters in fp32 over a bf16 base, so counting every parameter
    together would report `mixed(bf16,fp32)` for every LoRA run and block the half
    of this study that LoRA is — the freeze x peft collision again, one axis over.

    A real `get_peft_model` rather than a stand-in: what peft does to the dtype of
    the weights it adds is the fact under test, and a fake would do whatever this
    test told it to.
    """
    from peft import LoraConfig, get_peft_model

    config = bench(config_mapping, **{"peft.mode": "lora", "peft.r": 4})
    adapted = get_peft_model(bf16_model(), LoraConfig(r=4, target_modules="all-linear"))

    precision = axis(capture(Built(model=adapted), config), "precision.name")

    assert precision.detail["adapter"] == {"fp32": 4}, "peft no longer keeps adapters in fp32"
    assert precision.applied == "bf16"
    assert precision.matches


def test_an_adapter_whose_parameters_stop_matching_is_undetermined(config_mapping):
    """The marker is a string, and a string drifts. `_ple_report` shipped with one
    that matched nothing and reported success; zero matches under a declared
    adapter has to be undetermined rather than a base dtype that quietly includes
    the adapter's."""
    model = bf16_model()
    model.peft_config = {"default": SimpleNamespace(peft_type="LORA")}

    precision = axis(capture(Built(model=model), bench(config_mapping)), "precision.name")

    assert precision.applied is None
    assert ADAPTER_PARAM_MARKER in precision.detail["reason"]


class _RecipeLinear(torch.nn.Linear):
    """A module a recipe library defined, which is how one announces itself.

    Subclassed rather than restamped onto `torch.nn.Linear`: writing `__module__`
    onto torch's own class renames it for the whole session, and the tests that
    then read `transformer_engine` off an ordinary model fail somewhere else.
    """


_RecipeLinear.__module__ = "transformer_engine.pytorch.module.linear"


def test_a_module_swapped_by_a_recipe_package_is_not_read_as_bf16(config_mapping):
    """A low-precision recipe package keeps bf16 parameters and casts inside the
    step it wraps. Reading the weights straight off such a model would report
    bf16, and a bf16 request over it would then be certified as a match on
    nothing but the swapped-in module's own dtype choice."""
    model = bf16_model()
    model.language_model = _RecipeLinear(2, 2).to(torch.bfloat16)

    precision = axis(capture(Built(model=model), bench(config_mapping)), "precision.name")

    assert precision.applied is None
    assert "transformer_engine" in precision.detail["reason"]


def test_a_model_with_no_floating_point_weights_is_undetermined(config_mapping):
    model = torch.nn.Module()
    model.counter = torch.nn.Parameter(torch.zeros(2, dtype=torch.int64), requires_grad=False)

    precision = axis(capture(Built(model=model), bench(config_mapping)), "precision.name")

    assert precision.applied is None
    assert precision.detail["base"] == {}


# --- train.offload ------------------------------------------------------------
# `axes.assemble` refuses every value but `none`, so `none` is the only one that
# can reach a run and the only one this can prove. The rest are deepspeed's, and
# a probe that cannot tell them apart says so.


def optimizer_over(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-5)


def test_nothing_offloaded_is_read_off_where_the_tensors_are(config_mapping):
    model = bf16_model()
    config = bench(config_mapping)

    state = capture(Built(model=model, optimizer=optimizer_over(model)), config)

    assert axis(state, "train.offload").applied == "none"
    assert axis(state, "train.offload").matches
    assert axis(state, "train.offload").detail["compute"] == ["cpu"]


def test_parameters_held_off_the_compute_device_are_not_none(config_mapping):
    """Parameters somewhere the model does not compute is what `offload=param`
    produces. Nothing here can request it, so the run that shows up in this state
    got there by some route no one declared, and it must not report `none`."""
    config = bench(config_mapping)
    elsewhere = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2, device="meta"))], lr=1e-5)

    state = capture(Built(model=bf16_model(), optimizer=elsewhere), config)

    assert axis(state, "train.offload").applied == "offloaded(meta)"
    with pytest.raises(AppliedMismatch, match="train.offload"):
        assert_matches(state, config)


def test_optimizer_state_held_off_the_compute_device_is_not_none(config_mapping):
    """The state is what `offload=optimizer` moves, and it is empty until the
    first step — so a probe reading only the parameters would report `none` for an
    offloaded run and would have nothing to examine besides."""
    model = bf16_model()
    optimizer = optimizer_over(model)
    held = optimizer.param_groups[0]["params"][0]
    optimizer.state[held] = {"exp_avg": torch.zeros(2, device="meta")}
    config = bench(config_mapping)

    state = capture(Built(model=model, optimizer=optimizer), config)

    assert axis(state, "train.offload").applied == "offloaded(meta)"
    assert axis(state, "train.offload").detail["state_tensors"] == 1
    with pytest.raises(AppliedMismatch, match="train.offload"):
        assert_matches(state, config)


def deepspeed_optimizer():
    class DeepSpeedZeroOptimizer:
        param_groups = ()
        state: dict = {}

    DeepSpeedZeroOptimizer.__module__ = "deepspeed.runtime.zero.stage_1_and_2"
    return DeepSpeedZeroOptimizer()


def test_deepspeed_is_undetermined_rather_than_none(config_mapping):
    """Offload lives in deepspeed's own config, not on any object reachable from
    here. `none` would then be a claim about a setting nothing read."""
    model = bf16_model()
    config = bench(config_mapping)

    state = capture(Built(model=model, optimizer=deepspeed_optimizer()), config)
    offload = axis(state, "train.offload")

    assert offload.applied is None
    assert "deepspeed" in offload.detail["reason"]


class DeepSpeedEngine(torch.nn.Module):
    """Named, not renamed: `PARALLEL_WRAPPERS` matches on the class name because
    the packages that define these are not installed in every environment."""


def test_a_deepspeed_engine_around_the_model_is_undetermined_too(config_mapping):
    """The engine is where the optimizer would be built, so a run can reach this
    probe with a torch optimizer and still be a deepspeed run."""
    model = bf16_model()
    model.wrapped = DeepSpeedEngine()

    offload = axis(
        capture(Built(model=model, optimizer=optimizer_over(model)), bench(config_mapping)),
        "train.offload",
    )

    assert offload.applied is None
    assert offload.detail["engine"] == "DeepSpeedEngine"


def test_an_optimizer_holding_nothing_is_undetermined(config_mapping):
    """Where a run keeps no parameters says nothing about offload, and `none` from
    an empty scan is the shape of check this repository keeps shipping."""
    empty = torch.optim.AdamW([{"params": []}], lr=1e-5)

    offload = axis(
        capture(Built(model=bf16_model(), optimizer=empty), bench(config_mapping)), "train.offload"
    )

    assert offload.applied is None
    assert "no parameters" in offload.detail["reason"]


def test_offload_without_a_model_is_undetermined(config_mapping):
    """Nothing to be off: without the model there is no compute device to compare
    the optimizer's tensors against."""
    offload = axis(
        capture(Built(optimizer=optimizer_over(bf16_model())), bench(config_mapping)),
        "train.offload",
    )

    assert offload.applied is None
    assert "device" in offload.detail["reason"]


# --- the run these two axes were blocking -------------------------------------


class _Block(torch.nn.Module):
    """A submodule carrying the flag `train.gradient_checkpointing` is read off."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2).to(dtype)
        self.gradient_checkpointing = False


class _Rows(torch.utils.data.Dataset):
    """A dataset that declares its columns, which is where `pretokenize` is read."""

    column_names = ("query", "document")

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.zeros(2)


def whole_run(config) -> Built:
    """Everything a bf16 run on an accelerator constructs, in the state it leaves.

    Assembled here rather than through `axes.assemble` because the optimizer has
    to come back fused, and the fused AdamW kernel is CUDA-only — the one axis
    docs/CONTRACTS.md §6 says a CPU cannot satisfy. Everything else is the real
    object each probe reads.
    """
    model = torch.nn.Module()
    model.language_model = _Block()
    model.visual = _Block()
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.param_groups[0]["fused"] = True
    loss, _ = axes._loss(config)
    return Built(
        model=model,
        optimizer=optimizer,
        dataloader=torch.utils.data.DataLoader(_Rows()),
        loss_fn=loss,
        framework="native",
    )


def test_a_timing_run_is_no_longer_refused_for_want_of_a_probe(config_mapping):
    """The state this whole file is about. Before `precision.name` and
    `train.offload` had probes, every axis of every timing run could be verified
    and the run was still refused — `assert_matches` requires all of them, and two
    were undetermined by construction on any device. Nothing recorded that; the
    audit reported it as two unwired axes, which reads as work outstanding rather
    than as a measurement path that cannot run.
    """
    config = bench(config_mapping, **{"run.purpose": "timing"})

    state = capture(whole_run(config), config)

    assert [a.axis for a in state.undetermined()] == []
    assert [a.axis for a in state.mismatched()] == []
    assert_matches(state, config)


def test_a_timing_run_is_still_refused_when_one_axis_disagrees(config_mapping):
    """Paired with the test above, which on its own would also pass if
    `assert_matches` had been loosened to make the two new axes fit."""
    config = bench(config_mapping, **{"run.purpose": "timing"})
    run = whole_run(config)
    run.optimizer.param_groups[0]["fused"] = False

    with pytest.raises(AppliedMismatch, match="optim.name"):
        assert_matches(capture(run, config), config)


def test_capture_reports_an_axis_with_no_probe_as_undetermined(config_mapping, monkeypatch):
    """The producing half of the no-probe guarantee.

    It used to be driven by whichever axis happened to be unwired, and it was
    written to be deleted once none were — which would have taken the coverage
    with it. It is worth keeping: mutating `capture`'s no-probe branch to
    `AxisState(axis, requested, requested, ...)` is one word, and it left 325
    tests green while certifying two axes nothing had looked at.

    A synthetic knob keeps the branch reachable without waiting for someone to add
    an axis and forget the probe, which is the case it is there to catch.
    """
    from trainbench import applied

    knobs = {**axis_knobs(), "synthetic.unwired": lambda config: "x"}
    monkeypatch.setattr(applied, "axis_knobs", lambda: knobs)
    config = bench(config_mapping)

    state = capture(built(), config)

    unwired = axis(state, "synthetic.unwired")
    assert unwired.applied is None
    assert unwired.detail["reason"] == "no capture probe implemented"
    with pytest.raises(AppliedMismatch, match="synthetic.unwired"):
        assert_matches(state, config)


# --- the deepspeed engine readers ---------------------------------------------
# `tests/contract/test_applied_axes.py` patches `zero_stage` and `offload_targets`
# and pins that capture consults them. What it deliberately does not pin is what
# they read, because that is a pod question. These are that half: an engine shaped
# like the one `deepspeed/runtime/engine.py:1125-1154` documents, so the readers
# are exercised against a shape rather than only against a patch.
#
# 확인 안 함: that a real DeepSpeedEngine answers this way. deepspeed does not
# import on this host, and the package is sdist-only in the env locks.


class _Stage(int):
    """An int subclass, as `ZeroStageEnum` is."""


class _Device(str, Enum):  # noqa: UP042 - `OffloadDeviceEnum` is spelled this way
    """A str subclass, as `OffloadDeviceEnum` is."""

    none = "none"
    cpu = "cpu"


class _Offload:
    def __init__(self, device):
        self.device = device


class DeepSpeedEngineWithReaders(torch.nn.Module):
    """The engine, renamed at construction to what `PARALLEL_WRAPPERS` matches on."""


STAGE_3 = _Stage(3)


def zero_engine(stage=STAGE_3, optimizer_to=None, param_to=None, without=()):
    engine = DeepSpeedEngineWithReaders()
    engine.__class__ = type("DeepSpeedEngine", (DeepSpeedEngineWithReaders,), {})
    if "zero_optimization_stage" not in without:
        engine.zero_optimization_stage = lambda: stage
    if "zero_offload_optimizer" not in without:
        engine.zero_offload_optimizer = lambda: optimizer_to and _Offload(optimizer_to)
    if "zero_offload_param" not in without:
        engine.zero_offload_param = lambda: param_to and _Offload(param_to)
    return engine


def test_the_stage_comes_off_the_engines_own_reader():
    assert applied_module.zero_stage(zero_engine(stage=_Stage(2))) == 2
    assert applied_module.zero_stage(zero_engine(stage=_Stage(3))) == 3


def test_an_engine_that_answers_nothing_usable_reports_no_stage():
    """Each of these is a run whose stage nothing read, and `zero2` is not the
    nearest guess — it is a different partitioning with a different speed."""

    def raises():
        raise RuntimeError("uninitialised")

    angry = zero_engine()
    angry.zero_optimization_stage = raises

    assert applied_module.zero_stage(zero_engine(without=("zero_optimization_stage",))) is None
    assert applied_module.zero_stage(angry) is None
    assert applied_module.zero_stage(zero_engine(stage=True)) is None, "a bool is not a stage"
    assert applied_module.zero_stage(zero_engine(stage="3")) is None


def test_a_stage_outside_the_two_the_config_offers_is_named_not_rounded(config_mapping):
    """ZeRO has more stages than this study asks for. A run on stage 1 belongs to
    no setting here, and rounding it into one publishes its speed as another's."""
    config = bench(config_mapping, **{"parallel.strategy": "zero2"})
    model = bf16_model()
    model.engine = zero_engine(stage=_Stage(1))

    strategy = axis(capture(Built(model=model), config), "parallel.strategy")

    assert strategy.applied == "deepspeed(stage=1)"
    assert not strategy.matches


def test_the_offload_pair_comes_off_the_engines_own_readers():
    """`OffloadDeviceEnum` subclasses `str`, so `device == "none"` is the reading
    that works and `str(device)` is not — that spells the member's own repr and
    would report every un-offloaded engine as offloaded."""
    targets = applied_module.offload_targets

    assert targets(zero_engine()) == (False, False)
    assert targets(zero_engine(optimizer_to=_Device.cpu)) == (True, False)
    assert targets(zero_engine(param_to=_Device.cpu)) == (False, True)
    assert targets(zero_engine(optimizer_to=_Device.cpu, param_to=_Device.cpu)) == (True, True)
    assert targets(zero_engine(optimizer_to=_Device.none)) == (False, False)


def test_an_engine_that_does_not_expose_the_readers_says_nothing():
    assert applied_module.offload_targets(zero_engine(without=("zero_offload_param",))) is None
    assert applied_module.offload_targets(object()) is None


def test_a_real_engine_in_the_tree_decides_both_axes(config_mapping):
    """End to end, without a patch: the engine is where both of these live, and
    until it was read `zero2`/`zero3` and every offload but `none` could not be
    certified at all, so `assert_matches` refused every run that asked for them."""
    config = bench(config_mapping, **{"parallel.strategy": "zero3", "train.offload": "optimizer"})
    model = bf16_model()
    model.engine = zero_engine(stage=_Stage(3), optimizer_to=_Device.cpu)

    state = capture(Built(model=model, optimizer=optimizer_over(model)), config)

    assert axis(state, "parallel.strategy").applied == "zero3"
    assert axis(state, "parallel.strategy").matches
    assert axis(state, "train.offload").applied == "optimizer"
    assert axis(state, "train.offload").matches
    assert axis(state, "train.offload").detail["offload_optimizer"] is True


# --- the class table -----------------------------------------------------


def literal_values(section: str, name: str) -> set[str]:
    annotation = BenchConfig.model_fields[section].annotation.model_fields[name].annotation
    return {str(option) for option in get_args(annotation)}


def test_the_table_maps_from_what_ran_not_from_what_was_asked_for():
    """The shortest way to make the contested axis match is a table keyed by the
    request, and that certifies every run. The table is keyed by the class the
    run built, so no key of it may be a value the config can ask for."""
    assert not set(applied_module.OPTIM_CLASS_AXIS) & literal_values("optim", "name")


# --- the third state ----------------------------------------------------------


def owned_run(config, framework="tevatron", owned=("loss.name",)) -> Built:
    """A whole run whose loss the adapter computed itself, as tevatron's does."""
    run = whole_run(config)
    return Built(
        model=run.model,
        optimizer=run.optimizer,
        dataloader=run.dataloader,
        loss_fn=None,
        framework=framework,
        owned_axes=owned,
    )


def test_a_disclaimed_axis_lets_the_run_measure_and_says_who_has_it(config_mapping):
    """The tevatron cell. Its loss is computed inside `DenseModel.forward`, so with
    two states the run is refused over an axis that was never ours."""
    config = bench(config_mapping, **{"run.purpose": "timing", "framework.name": "tevatron"})

    state = capture(owned_run(config), config)
    loss = axis(state, "loss.name")

    assert loss.state == "framework_owned"
    assert loss.owner == "tevatron"
    assert loss.applied is None
    assert "tevatron" in loss.detail["reason"]
    assert [a.axis for a in state.framework_owned()] == ["loss.name"]
    assert loss not in state.undetermined()
    assert state.to_dict()["framework_owned"] == ["loss.name"]


def test_the_same_run_without_the_declaration_is_still_refused(config_mapping):
    """Paired with the test above, which on its own passes even if `assert_matches`
    stopped enforcing anything: the only difference here is the declaration."""
    config = bench(config_mapping, **{"run.purpose": "timing", "framework.name": "tevatron"})

    state = capture(owned_run(config, owned=()), config)

    assert axis(state, "loss.name").state == "undetermined"
    with pytest.raises(AppliedMismatch, match="loss.name"):
        assert_matches(state, config)


def test_a_disclaimed_axis_that_carries_a_value_is_a_mismatch():
    """Ownership is a declaration that nobody looked. An entry with both an owner
    and an applied value is an adapter certifying itself for having declined to
    look, and it has to stop the run rather than pass as somebody else's."""
    forged = AxisState("loss.name", "mnrl", "mnrl", owner="tevatron")

    assert forged.state == "framework_owned"
    assert not forged.matches
    assert forged in AppliedState((forged,)).mismatched()


def test_an_owner_on_an_axis_the_boundary_does_not_allow_is_not_ownership(config_mapping):
    """Re-checked on the state rather than only where capture grants it: hand-built
    states are what the record round trip and `report.py` produce, and an adapter
    able to disclaim `framework.name` could disclaim everything under it."""
    config = bench(config_mapping, **{"run.purpose": "timing"})
    forged = AxisState("framework.name", "native", None, owner="tevatron")

    assert forged.state == "undetermined"
    state = AppliedState(
        tuple(
            forged if a.axis == "framework.name" else a
            for a in capture(whole_run(config), config).axes
        )
    )

    assert state.to_dict()["framework_owned"] == []
    with pytest.raises(AppliedMismatch, match="framework.name"):
        assert_matches(state, config)
