"""An axis is certified only when something actually looked at it."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from trainbench.applied import AppliedMismatch, assert_matches, capture
from trainbench.config import to_bench_config
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


def fake_model(attn_impl: str | None):
    return SimpleNamespace(config=SimpleNamespace(_attn_implementation=attn_impl))


def test_attn_is_read_back_from_the_model(config_mapping):
    state = capture(fake_model("sdpa"), bench(config_mapping))

    attn = next(a for a in state.axes if a.axis == "attn")
    assert attn.applied == "sdpa"
    assert attn.matches


def test_silent_fallback_is_caught(config_mapping):
    """Asking for flash_attention_3 and getting sdpa is the exact failure this
    module exists for: a plausible number under a wrong label."""
    config = bench(config_mapping, **{"attn.name": "fa3", "attn.impl": "flash_attention_3"})

    state = capture(fake_model("sdpa"), config)

    attn = next(a for a in state.axes if a.axis == "attn")
    assert not attn.matches
    with pytest.raises(AppliedMismatch, match="requested 'flash_attention_3', applied 'sdpa'"):
        assert_matches(state, "timing")


def test_unwired_axis_blocks_a_timing_run(config_mapping):
    """Undetermined must not read as fine, or the mechanism is decorative."""
    state = capture(fake_model("sdpa"), bench(config_mapping))

    assert state.undetermined(), "axes without a capture probe should be undetermined"
    with pytest.raises(AppliedMismatch, match="undetermined"):
        assert_matches(state, "timing")


def test_probe_runs_are_not_blocked(config_mapping):
    """A probe answers 'does it run', so unverified axes are acceptable there."""
    state = capture(fake_model("sdpa"), bench(config_mapping))

    assert_matches(state, "probe")
    assert_matches(state, "profile")


def test_unreadable_model_is_undetermined_not_crash(config_mapping):
    state = capture(object(), bench(config_mapping))

    attn = next(a for a in state.axes if a.axis == "attn")
    assert attn.applied is None
    assert "reason" in attn.detail


def test_expected_failure_does_not_condemn_the_report():
    report = ProbeReport(framework="unsloth", model="m")
    report.add(Check(name="load", ok=True))
    report.add(Check(name="fast_st_accepts_vlm", ok=False, expected_failure=True))

    assert report.all_ok

    report.add(Check(name="real_problem", ok=False))
    assert not report.all_ok
