"""An axis is certified only when something actually looked at it."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest
import torch

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
from trainbench.config_schema import axis_knobs
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
        "determined": True,
        "matches": True,
        "detail": {"implementations": {"sdpa": 1}, "modules_checked": 1},
    }


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


def test_an_fp8_recipe_is_refused_rather_than_run_as_bf16(config_mapping):
    """Precision is not only a construction-time choice — fp8 wraps the forward
    pass — so an axis with nowhere to live would get applied somewhere unverified."""
    assert isinstance(axes.step_context(bench(config_mapping)), contextlib.AbstractContextManager)

    with pytest.raises(axes.UnappliedAxis):
        axes.step_context(bench(config_mapping, **{"precision.name": "mxfp8"}))


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


def test_an_fp8_recipe_is_not_read_off_bf16_weights(config_mapping):
    """transformer-engine keeps bf16 parameters and casts inside the recipe that
    wraps the step. Reading the weights of an mxfp8 run would report bf16, and a
    bf16 request over that model would then be certified as a match."""
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
