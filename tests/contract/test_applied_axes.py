"""Contract: the `applied-axes` boundary.

Between lane-c (`trainbench/applied.py`, which reads back what actually got
applied) and lane-h (`trainbench/axes.py` + `configs/`, which applies it). Owned by
neither, so neither can move the interface while working against it.

Three things are pinned here, and each exists because a lane could otherwise ship a
run that measures one thing under another thing's name.

1. **Three axis states, not two.** Today an axis is either applied or undetermined,
   and undetermined blocks a reportable run. That is the right fail-safe for an axis
   nobody looked at, and the wrong answer for one that is not ours to look at:
   tevatron's `DenseModel.forward` computes encoding, pooling, normalisation,
   scoring, InfoNCE and the distributed gather itself (`encoder.py:52-87`, pinned
   `dd06310`), and the decision taken is to measure each framework's own training
   step. In a tevatron cell `loss.name` and `parallel.cross_device_negatives` are
   the framework's, and with only two states requesting them there reads as our loss
   failing to apply. `framework_owned` is that third state.

2. **A capture may not become a mirror.** The module's principle is that it reads
   back, not that it repeats the request. Four axes currently return values that can
   never equal their config value, and the shortest fix for each — return what was
   asked for — turns the whole mechanism into decoration. The mirror tests below are
   the ones that must die if that shortcut is taken anywhere.

3. **What travels.** The payload `trainbench/record.py::build_record` writes under
   `applied`, pinned by `tests/fixtures/axis_state.sample.json` rather than
   described in prose: two lanes read a written specification differently and have a
   much harder time reading the same file differently.

The hydra/bench helpers below are deliberately a second copy of the ones in
`tests/test_applied.py`. That file sits next to lane-c's module and a lane may
rewrite it; a contract must not depend on a file either lane owns.

Not checked here, and not checkable on this host: what these capture paths read off
a real bitsandbytes optimizer, a real deepspeed engine or a real Transformer Engine
recipe. None of the three installs here — 확인 안 함. That is why the two foreign
class-name tables below are pinned as *tables*: the spellings are a pod question,
and a table keeps them in one reviewable place instead of inside a branch.

**Twenty-one of these tests are deferred, not passing.** `trainbench/applied.py`
carries none of the third state yet, and nine lanes fan out from the commit that
adds this file: leaving those tests red would make every lane's own gate report a
failure belonging to none of them. They are `xfail(strict=True)` instead, each
naming what lane-c has not landed. `strict` is what stops lane-c from implementing
the boundary and leaving the markers in place — a deferred test that starts passing
is an error, not a quiet pass.

The command that decides whether the deferral is over, which belongs in lane-c's
completion criteria:

    infisical run --env=dev -- uv run pytest tests/contract/test_applied_axes.py \\
        --runxfail -k the_contract_defers_nothing

Exit 0 means every marker is gone. `--runxfail` appears there for one reason only:
it makes that single test report its real result instead of its expected one. It
must never be used across the whole file to judge the contract — over the file it
strips the deferral from all twenty-one and reports failures that are already
known, which decides nothing. The `-k` is what keeps it honest.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pytest
import torch

from trainbench import applied as applied_module
from trainbench.applied import (
    AppliedMismatch,
    AppliedState,
    AxisState,
    Built,
    assert_matches,
    capture,
)
from trainbench.config import to_bench_config
from trainbench.config_schema import BenchConfig, axis_knobs

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
SAMPLE = REPO_ROOT / "tests" / "fixtures" / "axis_state.sample.json"

# The three states, and which of them a reportable run may carry.
APPLIED, FRAMEWORK_OWNED, UNDETERMINED = "applied", "framework_owned", "undetermined"

# The axes lane-c must be able to hand to a framework. Both are inside tevatron's
# forward pass; `framework.name` is deliberately not here, because it is the
# evidence of *which* framework ran and an adapter that could disclaim it could
# disclaim everything downstream of it.
OWNED_BY_TEVATRON = ("loss.name", "parallel.cross_device_negatives")

# Axes whose applied value cannot today equal any value the config offers, which is
# what makes implementing them impossible: `assert_matches` refuses the run.
CONTESTED = ("optim.name", "parallel.strategy", "precision.name", "train.offload")


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def config_mapping():
    from hydra import compose, initialize_config_dir

    from trainbench.compose import resolve

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return resolve(compose(config_name="config", overrides=["device=cpu"]))[1]


def bench(config_mapping, **overrides) -> BenchConfig:
    mapping = json.loads(json.dumps(config_mapping))
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        mapping[section][key] = value
    return to_bench_config(mapping)


def sample() -> dict[str, Any]:
    """The stored boundary payload, without the note that explains it to a reader."""
    loaded = json.loads(SAMPLE.read_text())
    return {key: value for key, value in loaded.items() if not key.startswith("_")}


def literal_values(dotted: str) -> set[str]:
    """Everything the config can ask for on this axis, as capture spells it."""
    section, name = dotted.split(".")
    annotation = BenchConfig.model_fields[section].annotation.model_fields[name].annotation
    options = get_args(annotation)
    return {str(option) for option in options} or {"True", "False"}


def axis(state: AppliedState, name: str) -> AxisState:
    return next(entry for entry in state.axes if entry.axis == name)


def model(attn="sdpa", modules=(), params=(), **attrs):
    """A model shaped like the ones under test: a tree whose submodules may
    disagree with the top-level config."""
    return SimpleNamespace(
        config=SimpleNamespace(_attn_implementation=attn, sub_configs=()),
        named_modules=lambda: list(modules),
        named_parameters=lambda: iter(params),
        **attrs,
    )


def tower(attn: str):
    """A submodule carrying its own attention implementation, as a vision tower does."""
    return SimpleNamespace(config=SimpleNamespace(_attn_implementation=attn))


def instance(class_name: str, **attrs):
    """An object of a class with this exact name. The class name is how
    `applied.py` recognises a foreign object without importing a package that is
    not installed in every environment."""
    return type(class_name, (), dict(attrs))()


def optimizer(class_name: str = "AdamW", fused: bool = False, params=()):
    return instance(
        class_name,
        param_groups=[{"fused": fused, "params": list(params)}],
        state={},
    )


def tensor(dtype=torch.bfloat16):
    return torch.zeros(2, dtype=dtype)


def weights(count: int = 2, dtype=torch.bfloat16):
    return [(f"layer.{i}.weight", tensor(dtype)) for i in range(count)]


def full_state(config: BenchConfig, **replace: AxisState) -> AppliedState:
    """A state where every axis was captured and agrees with the request, except
    the ones named here.

    `assert_matches` refuses a state that is missing any axis, so a test about one
    axis has to supply the other sixteen. Building them from `axis_knobs()` keeps
    that from becoming a second hand-written axis list that goes stale.
    """
    entries = []
    for name, read in axis_knobs().items():
        if name in replace:
            entries.append(replace[name])
            continue
        requested = str(read(config))
        entries.append(AxisState(name, requested, requested))
    return AppliedState(tuple(entries))


# --- 1. the vocabulary: three states, not two --------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.AXIS_STATES; remove this marker when it does",
)
def test_the_state_vocabulary_names_three_states():
    """A named vocabulary rather than a trichotomy each reader re-derives from two
    nullable fields. `report.py` has to show a tevatron cell as having no loss axis
    at all, and the third consumer that re-derives it is where the distinction gets
    collapsed back into two."""
    assert set(applied_module.AXIS_STATES) == {APPLIED, FRAMEWORK_OWNED, UNDETERMINED}


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.FRAMEWORK_OWNABLE; remove this marker when it does",
)
def test_ownership_is_bounded_by_a_declared_set():
    """An adapter that could own any axis could disclaim all of them, and capture
    would certify a run by declining to look at it."""
    ownable = set(applied_module.FRAMEWORK_OWNABLE)

    assert ownable <= set(axis_knobs()), "an ownable axis that is not an axis"
    assert set(OWNED_BY_TEVATRON) <= ownable
    assert "framework.name" not in ownable, "an adapter may not disclaim being itself"


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed AxisState.owner; remove this marker when it does",
)
def test_a_framework_owned_axis_does_not_block_a_reportable_run(config_mapping):
    config = bench(config_mapping, **{"framework.name": "tevatron"})
    owned = AxisState(
        "loss.name",
        "mnrl",
        None,
        owner="tevatron",
        detail={"reason": "DenseModel.forward computes InfoNCE itself"},
    )

    state = full_state(config, **{"loss.name": owned})

    assert owned.state == FRAMEWORK_OWNED
    # Ownership is a declaration that we did not read the axis, not a certification
    # that it was applied. An owned axis carrying an applied value would let an
    # adapter certify itself.
    assert owned.applied is None
    assert owned.matches is False
    assert [entry.axis for entry in state.framework_owned()] == ["loss.name"]
    assert owned not in state.undetermined()

    assert_matches(state, config)


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed AxisState.state; remove this marker when it does",
)
def test_the_same_axis_unowned_and_unread_still_blocks(config_mapping):
    """Paired with the test above: on its own that one passes even if
    `assert_matches` stopped enforcing anything at all."""
    config = bench(config_mapping, **{"framework.name": "tevatron"})
    unread = AxisState("loss.name", "mnrl", None, detail={"reason": "no loss was built"})

    state = full_state(config, **{"loss.name": unread})

    assert unread.state == UNDETERMINED
    with pytest.raises(AppliedMismatch, match="loss.name"):
        assert_matches(state, config)


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed Built.owned_axes; remove this marker when it does",
)
def test_ownership_comes_from_the_adapter_not_from_the_config(config_mapping):
    """The mirror failure in its new-state form.

    `framework=tevatron` in the config is the request. If that alone made the loss
    axis somebody else's, every framework cell would exempt itself from the axes it
    was least likely to apply, which is the opposite of what this module is for. The
    declaration has to come off `Built`, which only the adapter that actually ran
    constructs — the same reason `Built.framework` is not read from the config.
    """
    config = bench(config_mapping, **{"framework.name": "tevatron"})

    silent = capture(Built(model=model(), framework="tevatron"), config)
    assert axis(silent, "loss.name").state == UNDETERMINED
    assert axis(silent, "loss.name").owner is None

    declaring = capture(
        Built(model=model(), framework="tevatron", owned_axes=("loss.name",)), config
    )
    owned = axis(declaring, "loss.name")
    assert owned.state == FRAMEWORK_OWNED
    assert owned.owner == "tevatron"
    # Disclaiming an axis is not the same as certifying it. An owned axis that also
    # carried the requested value would be the mirror wearing the new state's name:
    # every framework cell would pass every axis it declined to look at.
    assert owned.applied is None
    assert not owned.matches
    assert owned.detail.get("reason"), "a disclaimed axis has to say whose it is and why"


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed Built.owned_axes; remove this marker when it does",
)
def test_an_axis_outside_the_declared_set_cannot_be_disclaimed(config_mapping):
    config = bench(config_mapping, **{"framework.name": "tevatron"})

    state = capture(Built(model=model(), framework="tevatron", owned_axes=("kernel.name",)), config)

    assert axis(state, "kernel.name").owner is None
    assert axis(state, "kernel.name").state != FRAMEWORK_OWNED


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed Built.owned_axes; remove this marker when it does",
)
def test_ownership_without_a_declared_adapter_is_not_ownership(config_mapping):
    """`owner` names who computes the axis instead of us. A run where no adapter
    declared itself has nobody to name, and 'owned by nobody' must fall back to
    undetermined rather than to exempt."""
    config = bench(config_mapping, **{"framework.name": "tevatron"})

    state = capture(Built(model=model(), owned_axes=("loss.name",)), config)

    assert axis(state, "loss.name").state == UNDETERMINED


# --- 2. a capture may not become a mirror ------------------------------------


def lora_model(dtype=torch.bfloat16):
    entry = SimpleNamespace(peft_type="LORA")
    return model(
        params=weights(dtype=dtype),
        peft_config=SimpleNamespace(values=lambda: [entry]),
    )


MIRRORS = (
    # (axis, config overrides, Built, what capture must say instead of the request)
    ("attn.name", {"attn.name": "fa3"}, lambda: Built(model=model("sdpa")), "sdpa"),
    ("kernel.name", {"kernel.name": "liger"}, lambda: Built(model=model()), "none"),
    (
        "peft.mode",
        {"peft.mode": "qlora", "peft.r": 8},
        lambda: Built(model=model(params=weights())),
        "full",
    ),
    (
        "precision.name",
        {"precision.name": "bf16"},
        lambda: Built(model=model(params=weights(dtype=torch.float32))),
        "fp32",
    ),
    (
        "parallel.strategy",
        {"parallel.strategy": "ddp"},
        lambda: Built(model=model(params=weights())),
        "single",
    ),
    (
        "optim.name",
        {"optim.name": "adamw_8bit"},
        lambda: Built(model=model(params=weights()), optimizer=optimizer("AdamW")),
        "adamw_unfused",
    ),
    (
        "dataloader.backend",
        {"dataloader.backend": "dali"},
        lambda: Built(dataloader=torch.utils.data.DataLoader([0, 1], batch_size=1)),
        "torch",
    ),
    (
        "loss.name",
        {"loss.name": "cached_mnrl", "loss.mini_batch": 2},
        lambda: Built(loss_fn=SimpleNamespace(axis_value="mnrl", __name__="mnrl")),
        "mnrl",
    ),
)


@pytest.mark.parametrize("name,overrides,build,expected", MIRRORS, ids=[m[0] for m in MIRRORS])
def test_capture_reads_the_run_back_rather_than_repeating_the_request(
    config_mapping, name, overrides, build, expected
):
    """Every one of these runs asked for one thing and built another.

    This is the whole boundary in one test: make any capture return the requested
    value and the axis it belongs to fails here. A mapping added so that
    `optim=adamw_8bit` can match must map from what the run built, never from what
    it asked for — a shortcut through `config` passes every other test in this file.

    Asserted through `determined`/`matches` rather than through the new `state`
    field on purpose: this is a standing guard, and it has to hold at the base
    commit as well as after the boundary lands.
    """
    config = bench(config_mapping, **overrides)

    entry = axis(capture(build(), config), name)

    assert entry.applied == expected
    assert entry.applied != entry.requested
    assert entry.determined
    assert not entry.matches


# Values capture produces on purpose that no config can ask for. Each names a run
# belonging to no setting, and each is what makes `assert_matches` refuse rather
# than round the run to the nearest label. A lane adding mappings so the contested
# axes can match must not sweep these away with them.
UNNAMEABLE = (
    (
        "attn.name",
        lambda: Built(
            model=model(
                "flash_attention_2",
                modules=[("visual", tower("sdpa"))],
            )
        ),
        "mixed(flash_attention_2,sdpa)",
    ),
    (
        "precision.name",
        lambda: Built(model=model(params=[*weights(1), ("head.weight", tensor(torch.float32))])),
        "mixed(bf16,fp32)",
    ),
    (
        "optim.name",
        lambda: Built(model=model(params=weights()), optimizer=optimizer("AdamW")),
        "adamw_unfused",
    ),
    (
        "peft.mode",
        lambda: Built(
            model=SimpleNamespace(
                **vars(lora_model()),
                is_loaded_in_4bit=True,
            )
        ),
        None,  # any qlora(...) qualification; asserted by prefix below
    ),
)


@pytest.mark.parametrize("name,build,expected", UNNAMEABLE, ids=[u[0] for u in UNNAMEABLE])
def test_deliberately_unnameable_values_survive(config_mapping, name, build, expected):
    config = bench(config_mapping)

    entry = axis(capture(build(), config), name)

    assert entry.applied is not None, "an unnameable run must be named, not left unread"
    if expected is not None:
        assert entry.applied == expected
    else:
        assert entry.applied.startswith("qlora(")
    assert entry.applied not in literal_values(name), (
        f"{entry.applied!r} became a value the config can ask for, so a run that "
        "belongs to no setting now matches one"
    )


# --- 3. the four axes that cannot currently equal their config value ---------


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.OPTIM_CLASS_AXIS; remove this marker when it does",
)
def test_the_optimizer_class_table_only_maps_to_values_the_config_offers():
    """`kind.lower()` cannot produce an underscore, so `adamw_8bit` was unreachable
    and a run that requested it could never be certified. The fix is a table from
    the built optimizer's class to the axis value — not a translation of the
    request, which is the mirror.

    확인 안 함: whether the class names in this table are the ones bitsandbytes
    actually constructs. bitsandbytes does not install on this host; a pod must
    print `type(optimizer).__name__` for each value.
    """
    table = applied_module.OPTIM_CLASS_AXIS
    offered = literal_values("optim.name")

    assert set(table.values()) <= offered, sorted(set(table.values()) - offered)
    assert "adamw_8bit" in set(table.values())


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.OPTIM_CLASS_AXIS; remove this marker when it does",
)
def test_the_optimizer_axis_is_decided_by_the_table(config_mapping):
    config = bench(config_mapping, **{"optim.name": "adamw_8bit"})
    class_name = next(
        name for name, value in applied_module.OPTIM_CLASS_AXIS.items() if value == "adamw_8bit"
    )

    entry = axis(
        capture(Built(model=model(params=weights()), optimizer=optimizer(class_name)), config),
        "optim.name",
    )

    assert entry.applied == "adamw_8bit"
    assert entry.matches


def test_an_optimizer_the_table_does_not_name_earns_no_config_value(config_mapping):
    """The table is what certifies; being an optimizer is not."""
    config = bench(config_mapping, **{"optim.name": "adamw_8bit"})

    entry = axis(
        capture(
            Built(model=model(params=weights()), optimizer=optimizer("SomeVendorAdamW8Bit")),
            config,
        ),
        "optim.name",
    )

    assert entry.applied not in literal_values("optim.name")
    assert not entry.matches


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.PRECISION_RECIPE_AXIS; remove this marker when it does",
)
def test_the_precision_recipe_table_only_maps_to_values_the_config_offers():
    """`_capture_precision` refuses to read an fp8 run's precision off bf16 weights,
    which is correct and is why the axis is permanently undetermined. The recipe is
    the thing that decides it, so the recipe has to travel on `Built`.

    확인 안 함: the Transformer Engine class names. transformer-engine does not
    install on this host.
    """
    table = applied_module.PRECISION_RECIPE_AXIS
    offered = literal_values("precision.name")

    assert set(table.values()) <= offered, sorted(set(table.values()) - offered)
    assert {"mxfp8", "nvfp4"} <= set(table.values())


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed Built.precision_recipe; remove this marker when it does",
)
@pytest.mark.parametrize("wanted", ["mxfp8", "nvfp4"])
def test_precision_is_read_off_the_recipe_the_step_actually_wrapped_with(config_mapping, wanted):
    config = bench(config_mapping, **{"precision.name": wanted})
    class_name = next(
        name for name, value in applied_module.PRECISION_RECIPE_AXIS.items() if value == wanted
    )

    entry = axis(
        capture(
            Built(model=model(params=weights()), precision_recipe=instance(class_name)),
            config,
        ),
        "precision.name",
    )

    assert entry.applied == wanted
    assert entry.matches


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed Built.precision_recipe; remove this marker when it does",
)
def test_an_fp8_request_with_no_recipe_is_never_certified_from_the_weights(config_mapping):
    """The refusal that must survive the change: bf16 parameters are what an fp8 run
    has, so reading the dtype and calling it mxfp8 is exactly the failure this
    module exists to prevent."""
    config = bench(config_mapping, **{"precision.name": "mxfp8"})

    entry = axis(
        capture(Built(model=model(params=weights()), precision_recipe=None), config),
        "precision.name",
    )

    assert entry.applied != "mxfp8"
    assert not entry.matches


def engine_model(engine_class: str = "DeepSpeedEngine", params=()):
    return model(modules=[("module", instance(engine_class))], params=params)


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.zero_stage; remove this marker when it does",
)
@pytest.mark.parametrize("stage,expected", [(2, "zero2"), (3, "zero3")])
def test_the_zero_stage_is_read_off_the_engine_not_off_the_config(
    config_mapping, monkeypatch, stage, expected
):
    """One engine class stands for both stages, so the class name alone can only say
    `deepspeed` — a value no config offers, which blocked both settings. The stage
    has to come from the engine.

    `zero_stage` is monkeypatched rather than fed a hand-built deepspeed config,
    because what a real `DeepSpeedEngine` exposes is 확인 안 함 on this host. What
    the contract pins is that capture asks it, and answers with the config's own
    vocabulary — patching the seam and watching the answer move is the evidence
    that it is consulted at all.
    """
    config = bench(config_mapping, **{"parallel.strategy": expected})
    monkeypatch.setattr(applied_module, "zero_stage", lambda engine: stage)

    entry = axis(capture(Built(model=engine_model()), config), "parallel.strategy")

    assert entry.applied == expected
    assert entry.matches


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.zero_stage; remove this marker when it does",
)
def test_an_engine_whose_stage_cannot_be_read_is_undetermined(config_mapping, monkeypatch):
    config = bench(config_mapping, **{"parallel.strategy": "zero3"})
    monkeypatch.setattr(applied_module, "zero_stage", lambda engine: None)

    entry = axis(capture(Built(model=engine_model()), config), "parallel.strategy")

    assert entry.applied is None
    assert entry.state == UNDETERMINED


def test_a_model_with_no_engine_is_unaffected_by_the_stage_reader(config_mapping):
    """`parallel=single` and `ddp`/`fsdp2` still answer from the wrapper. A stage
    reader that started deciding every run would be a second, contradictory source
    for this axis."""
    config = bench(config_mapping)

    entry = axis(capture(Built(model=model(params=weights())), config), "parallel.strategy")

    assert entry.applied == "single"


OFFLOAD_TARGETS = (
    ((False, False), "none"),
    ((True, False), "optimizer"),
    ((False, True), "param"),
    ((True, True), "both"),
)


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.offload_targets; remove this marker when it does",
)
@pytest.mark.parametrize("targets,expected", OFFLOAD_TARGETS, ids=[o[1] for o in OFFLOAD_TARGETS])
def test_offload_is_read_off_the_engine_config(config_mapping, monkeypatch, targets, expected):
    """The docstring on `_capture_offload` concedes it: under deepspeed the setting
    lives in the engine's own config and nothing reachable from the optimizer says
    what it is. So the engine's config is what has to be read, and `(optimizer,
    param)` is the pair every value of this axis is made of.

    확인 안 함: where deepspeed keeps it. deepspeed does not install on this host.
    """
    config = bench(config_mapping, **{"train.offload": expected})
    monkeypatch.setattr(applied_module, "offload_targets", lambda engine: targets)

    entry = axis(
        capture(
            Built(
                model=engine_model(params=weights()),
                optimizer=optimizer("DeepSpeedZeroOptimizer"),
            ),
            config,
        ),
        "train.offload",
    )

    assert entry.applied == expected
    assert entry.matches


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.offload_targets; remove this marker when it does",
)
def test_an_engine_that_does_not_say_where_it_offloads_is_undetermined(config_mapping, monkeypatch):
    """The direction that must not flip: reporting `none` because nothing looked is
    worse than reporting nothing."""
    config = bench(config_mapping, **{"train.offload": "optimizer"})
    monkeypatch.setattr(applied_module, "offload_targets", lambda engine: None)

    entry = axis(
        capture(
            Built(
                model=engine_model(params=weights()),
                optimizer=optimizer("DeepSpeedZeroOptimizer"),
            ),
            config,
        ),
        "train.offload",
    )

    assert entry.applied is None
    assert entry.state == UNDETERMINED


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.OPTIM_CLASS_AXIS; remove this marker when it does",
)
@pytest.mark.parametrize("name", CONTESTED)
def test_every_contested_axis_can_reach_a_state_that_matches(config_mapping, name):
    """The reason this boundary exists at all: lane-h cannot implement an axis whose
    applied value can never equal its request, because `assert_matches` refuses the
    run. Read as a whole-axis statement rather than per value — the per-value
    evidence is the tests above."""
    reachable = {
        "optim.name": set(applied_module.OPTIM_CLASS_AXIS.values()),
        "precision.name": set(applied_module.PRECISION_RECIPE_AXIS.values()),
        "parallel.strategy": {"zero2", "zero3"},
        "train.offload": {value for _, value in OFFLOAD_TARGETS},
    }[name]
    blocked = {
        "optim.name": {"adamw_8bit"},
        "precision.name": {"mxfp8", "nvfp4"},
        "parallel.strategy": {"zero2", "zero3"},
        "train.offload": {"optimizer", "param", "both"},
    }[name]

    assert blocked <= reachable
    assert reachable <= literal_values(name)


# --- 4. what travels ---------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed applied.FRAMEWORK_OWNABLE; remove this marker when it does",
)
def test_the_stored_sample_is_itself_a_valid_instance_of_the_contract():
    """The fixture is the thing two lanes look at instead of reading prose the same
    way. A sample that has drifted out of the contract is worse than none."""
    payload = sample()
    knobs = set(axis_knobs())

    assert set(payload) == {"axes", "all_determined", "all_matched", "framework_owned", "missing"}
    seen = set()
    for entry in payload["axes"]:
        assert entry["axis"] in knobs, entry["axis"]
        assert entry["state"] in {APPLIED, FRAMEWORK_OWNED, UNDETERMINED}
        assert entry["determined"] is (entry["applied"] is not None)
        if entry["state"] == FRAMEWORK_OWNED:
            assert entry["owner"] is not None
            assert entry["applied"] is None
            assert entry["axis"] in set(applied_module.FRAMEWORK_OWNABLE)
        else:
            assert entry["owner"] is None
        seen.add(entry["state"])

    assert seen == {APPLIED, FRAMEWORK_OWNED, UNDETERMINED}, (
        "the sample must carry all three states, or it cannot pin the one that is new"
    )
    assert payload["framework_owned"] == sorted(
        entry["axis"] for entry in payload["axes"] if entry["state"] == FRAMEWORK_OWNED
    )
    # The two summary flags are about the axes that are ours. An owned axis is not
    # an undetermined one, so it must not drag `all_determined` down with it.
    assert payload["all_determined"] is not any(
        entry["state"] == UNDETERMINED for entry in payload["axes"]
    )
    assert payload["all_matched"] is not any(
        entry["determined"] and not entry["matches"] for entry in payload["axes"]
    )
    assert set(CONTESTED) <= {entry["axis"] for entry in payload["axes"]}


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed the owner/state/framework_owned record keys; "
    "remove this marker when it does",
)
def test_the_record_has_the_shape_the_sample_pins(config_mapping):
    """`build_record` writes this dict under `applied`, and it is the only thing a
    result file says about what actually ran. A key dropped here is a state that
    stops travelling."""
    payload = capture(Built(model=model(params=weights())), bench(config_mapping)).to_dict()
    stored = sample()

    assert set(payload) == set(stored)
    assert set(payload["axes"][0]) == set(stored["axes"][0])
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed AppliedState.framework_owned; remove this marker when it does",
)
def test_a_framework_owned_axis_survives_the_round_trip():
    """Two runs that differ only in whether the loss was somebody else's must not
    serialise to the same record.

    `all_determined` cannot be the field that separates them. Forcing it false for
    an owned axis makes a tevatron cell indistinguishable from a run whose loss
    nobody read — which is the collapse the third state exists to prevent. The
    top-level `framework_owned` list is what carries the difference, so it has to be
    there and it has to be non-empty exactly when an axis was disclaimed.
    """
    owned = AxisState(
        "loss.name", "mnrl", None, detail={"reason": "tevatron computes it"}, owner="tevatron"
    )
    unread = AxisState("loss.name", "mnrl", None, detail={"reason": "no loss was built"})

    entry = json.loads(json.dumps(AppliedState((owned,)).to_dict()))
    other = json.loads(json.dumps(AppliedState((unread,)).to_dict()))

    assert entry["axes"][0]["owner"] == "tevatron"
    assert entry["axes"][0]["state"] == FRAMEWORK_OWNED
    assert entry["axes"][0]["applied"] is None
    assert entry["framework_owned"] == ["loss.name"]
    assert entry["all_matched"] is True

    assert other["framework_owned"] == []
    assert other["axes"][0]["state"] == UNDETERMINED
    assert other["axes"][0]["owner"] is None
    assert entry != other


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed AxisState.state; remove this marker when it does",
)
def test_the_three_states_are_distinguishable_from_the_record_alone(config_mapping):
    """report.py has to show a tevatron cell as having no loss axis rather than as
    having an unverified one, and it reads the file, not the objects."""
    config = bench(config_mapping, **{"framework.name": "tevatron"})
    state = full_state(
        config,
        **{
            "loss.name": AxisState("loss.name", "mnrl", None, owner="tevatron", detail={}),
            "kernel.name": AxisState("kernel.name", "liger", None, detail={"reason": "unread"}),
        },
    )

    by_axis = {entry["axis"]: entry for entry in json.loads(json.dumps(state.to_dict()))["axes"]}

    assert by_axis["loss.name"]["state"] == FRAMEWORK_OWNED
    assert by_axis["kernel.name"]["state"] == UNDETERMINED
    assert by_axis["framework.name"]["state"] == APPLIED
    three = {by_axis[name]["state"] for name in ("loss.name", "kernel.name", "framework.name")}
    assert three == {FRAMEWORK_OWNED, UNDETERMINED, APPLIED}


# --- 5. the deferral itself, and the check that keeps it from going quiet -----


def deferred_expectations(source: str, guard: str) -> list[str]:
    """Tests in this file still carrying `pytest.mark.xfail`, except `guard`.

    Read out of the syntax tree rather than by grepping, because the word appears
    in the prose above and a check that counts its own explanation is not counting
    markers. Every `pytest.mark.xfail` expression is found wherever it sits — a
    decorator, or `marks=` inside a `pytest.param` — and attributed to the test it
    encloses, so deferring one parametrisation is caught the same way as deferring
    a whole test.

    `guard` is excluded on purpose, and that exclusion is what makes the state
    machine terminate. Counting its own marker would leave a green state in which
    every other marker is gone and this one is not: the assertion would still fail,
    the marker would still be expected, and nothing would ever say so. Excluded, the
    last marker removal makes this test pass under a strict xfail — an XPASS, which
    is an error — and the only way out of that error is to delete this marker too.
    """
    tree = ast.parse(source)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            functions.append((start, node.end_lineno, node.name))

    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or _dotted(node) != "pytest.mark.xfail":
            continue
        enclosing = [
            name for start, end, name in functions if start <= node.lineno <= (end or node.lineno)
        ]
        found.update(name for name in enclosing if name != guard)
    return sorted(found)


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


@pytest.mark.xfail(
    strict=True,
    reason="lane-c has not landed the boundary, so this file still defers 21 tests; "
    "remove this marker last, once every other one is gone",
)
def test_the_contract_defers_nothing():
    """The check that stops the deferral from becoming the vacuum it was meant to
    avoid.

    Every expectation in this file that lane-c has not met yet is marked
    `xfail(strict=True)` rather than left red, because nine lanes fan out from this
    commit and a base that is red for reasons belonging to none of them makes every
    lane's own gate unreadable.

    The cost of that choice is that `pytest tests/contract/test_applied_axes.py`
    goes green while the contract is unmet, and green would then mean nothing. Three
    things close that, and all three are needed:

    - `strict=True` turns a deferred test that starts passing into an error, so
      lane-c cannot land the boundary and leave the markers behind.
    - This test names what is still deferred, so the debt is enumerable rather than
      spread across twenty-one decorators. `-rx` prints it.
    - It excludes its own marker from the count, so removing the last of the others
      makes it XPASS — an error — and the only fix is to remove this marker too.
      There is no green state with a marker left in the file.

    What none of that can do is make the plain command decide it, because the plain
    command has to be green today and the contract is unmet today. The command whose
    exit code decides it is in the module docstring, and it is what belongs in
    lane-c's completion criteria.
    """
    still_deferred = deferred_expectations(
        Path(__file__).read_text(), "test_the_contract_defers_nothing"
    )

    assert not still_deferred, (
        f"{len(still_deferred)} contract expectations are still deferred to lane-c: "
        + ", ".join(still_deferred)
    )
