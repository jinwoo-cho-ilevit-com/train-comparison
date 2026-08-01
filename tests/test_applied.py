"""An axis is certified only when something actually looked at it."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from trainbench import axes
from trainbench.applied import (
    AppliedMismatch,
    AppliedState,
    AxisState,
    assert_matches,
    capture,
)
from trainbench.config import to_bench_config
from trainbench.config_schema import axis_knobs
from trainbench.probe.types import Check, ProbeReport


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


def axis(state, name):
    return next(a for a in state.axes if a.axis == name)


def test_attn_is_read_back_from_the_model(config_mapping):
    state = capture(fake_model("sdpa"), bench(config_mapping))

    assert axis(state, "attn.name").applied == "sdpa"
    assert axis(state, "attn.name").matches


def test_silent_fallback_is_caught(config_mapping):
    """Asking for flash_attention_3 and getting sdpa is the exact failure this
    module exists for: a plausible number under a wrong label."""
    config = bench(config_mapping, **{"attn.name": "fa3"})

    state = capture(fake_model("sdpa"), config)

    assert not axis(state, "attn.name").matches
    with pytest.raises(AppliedMismatch, match="requested 'flash_attention_3', applied 'sdpa'"):
        assert_matches(state, config)


def test_vision_tower_left_on_sdpa_is_a_mismatch(config_mapping):
    """transformers updates the top-level config, then warns and moves on for any
    submodule that cannot take the implementation. Every model here is multimodal,
    so a flash-attention language model over an sdpa vision tower is the realistic
    partial application — and reading only the top level calls it a match."""
    config = bench(config_mapping, **{"attn.name": "fa2"})

    state = capture(fake_model("flash_attention_2", vision_impl="sdpa"), config)

    attn = axis(state, "attn.name")
    assert attn.applied == "mixed(flash_attention_2,sdpa)"
    assert not attn.matches
    assert attn.detail["implementations"] == {"flash_attention_2": 1, "sdpa": 1}
    assert attn.detail["dissenting"] == ["visual"]


def test_uniform_submodules_still_match(config_mapping):
    config = bench(config_mapping, **{"attn.name": "fa2"})

    state = capture(fake_model("flash_attention_2", vision_impl="flash_attention_2"), config)

    assert axis(state, "attn.name").matches


def test_every_axis_in_the_schema_is_reported(config_mapping):
    """The axis set is derived, so an axis cannot be dropped by deleting a line.

    A hand-written list fails open: a missing entry is not an undetermined axis,
    it is no axis at all, and nothing notices.
    """
    state = capture(fake_model("sdpa"), bench(config_mapping))

    assert {a.axis for a in state.axes} == set(axis_knobs())
    assert state.missing() == []
    for name in ("peft.mode", "loss.name", "train.offload", "parallel.cross_device_negatives"):
        assert name in axis_knobs(), f"{name} is a knob that changes the measurement"


def test_an_axis_without_a_probe_blocks_a_timing_run(config_mapping):
    """Undetermined must not read as fine, or the mechanism is decorative."""
    config = bench(config_mapping)
    state = capture(fake_model("sdpa"), config)

    unverified = {a.axis for a in state.undetermined()}
    assert "compile.mode" in unverified, "an axis with no capture probe must be undetermined"
    with pytest.raises(AppliedMismatch, match="compile.mode"):
        assert_matches(state, config)


def test_probe_runs_are_not_blocked(config_mapping):
    """A probe answers 'does it run', so unverified axes are acceptable there.

    Paired with the timing case above: on its own this passes even if
    assert_matches were an empty function.
    """
    state = capture(fake_model("sdpa"), bench(config_mapping))

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
        assert_matches(capture(fake_model("sdpa"), config), forged)


def test_unreadable_model_is_undetermined_not_crash(config_mapping):
    state = capture(object(), bench(config_mapping))

    attn = axis(state, "attn.name")
    assert attn.applied is None
    assert "reason" in attn.detail


def test_a_probe_that_raises_is_undetermined_not_crash(config_mapping, monkeypatch):
    from trainbench import applied

    def explode(model, config):
        raise RuntimeError("cuda is on fire")

    monkeypatch.setitem(applied._CAPTURES, "attn.name", explode)
    state = capture(fake_model("sdpa"), bench(config_mapping))

    assert axis(state, "attn.name").applied is None
    assert "cuda is on fire" in axis(state, "attn.name").detail["reason"]


def test_a_config_of_the_wrong_shape_is_undetermined_not_crash():
    """capture must survive anything: it runs on the failure path, where the
    thing that is wrong may well be the config itself."""
    state = capture(fake_model("sdpa"), SimpleNamespace())

    assert state.axes, "axes are still enumerated"
    assert all(a.applied is None for a in state.axes)


def test_applied_state_serialises_for_the_record(config_mapping):
    """This dict is what a result JSON carries; without it the file records the
    request and nothing about what ran."""
    payload = capture(fake_model("sdpa"), bench(config_mapping)).to_dict()

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

    _, detail = _capture_attn(wide, None)

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
            "model.tokens_per_image": 280,
            "model.instruction_prompt": None,
        },
        **overrides,
    )


PLE_FROZEN = param("embed_tokens_per_layer.weight", False)
PLE_TRAINING = param("layers.0.per_layer_input_gate", True)


def test_freezing_nothing_is_not_a_successful_freeze(config_mapping):
    """`_ple_report` reported ok=True on zero matches, so an upstream rename would
    have read as a freeze of 2.39B parameters that were in fact still training."""
    model = fake_model("sdpa", params=[param("model.layers.0.mlp.weight", True)])

    ple = axis(capture(model, gemma(config_mapping)), "freeze.ple")

    assert ple.applied is None
    assert ple.detail["matched"] == 0


def test_a_half_applied_freeze_is_a_mismatch(config_mapping):
    config = gemma(config_mapping, **{"freeze.ple": True})

    state = capture(fake_model("sdpa", params=[PLE_FROZEN, PLE_TRAINING]), config)

    assert axis(state, "freeze.ple").applied == "partial"
    with pytest.raises(AppliedMismatch, match="freeze.ple"):
        assert_matches(state, config)


def test_a_freeze_that_took_matches(config_mapping):
    config = gemma(config_mapping, **{"freeze.ple": True})
    frozen = [PLE_FROZEN, param("layers.0.per_layer_input_gate", False)]

    state = capture(fake_model("sdpa", params=frozen), config)

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
