"""Invalid config combinations must fail before a run starts, not after."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from trainbench.compose import resolve
from trainbench.config_schema import BenchConfig

from .conftest import CONFIG_DIR


def compose_cfg(*overrides: str) -> BenchConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
        return resolve(cfg)[0]


def test_default_composition_is_valid():
    bench = compose_cfg()
    assert bench.model.arch == "qwen3_vl"
    assert bench.framework.name == "native"
    assert bench.run.purpose == "timing"


def test_every_model_group_composes():
    for name in ("qwen3_vl_emb_2b", "qwen3_5_0_8b", "gemma4_e2b"):
        assert compose_cfg(f"model={name}").model.name == name


def test_timing_run_rejects_profiler():
    with pytest.raises(ValidationError, match="forbids profiler"):
        compose_cfg("run=timing", "run.profiler=true")


def test_timing_run_rejects_deterministic():
    with pytest.raises(ValidationError, match="forbids deterministic"):
        compose_cfg("run=timing", "train.deterministic=true")


def test_profile_run_allows_profiler():
    assert compose_cfg("run=profile").run.profiler is True


def test_max_autotune_requires_enough_warmup():
    with pytest.raises(ValidationError, match="warmup_discard_steps"):
        compose_cfg("compile=max_autotune", "train.warmup_discard_steps=5")

    assert compose_cfg("compile=max_autotune", "train.warmup_discard_steps=25")


def test_gradcache_mini_batch_cannot_exceed_batch():
    with pytest.raises(ValidationError, match="exceeds train.batch_size"):
        compose_cfg("loss=cached_mnrl", "train.batch_size=4", "loss.mini_batch=8")


def test_gradcache_requires_mini_batch():
    with pytest.raises(ValidationError, match="requires loss.mini_batch"):
        compose_cfg("loss=cached_mnrl", "loss.mini_batch=null")


def test_ple_freeze_is_gemma4_only():
    with pytest.raises(ValidationError, match="only exist in gemma4"):
        compose_cfg("model=qwen3_5_0_8b", "freeze=ple")

    assert compose_cfg("model=gemma4_e2b", "freeze=ple").freeze.ple is True


def test_lora_requires_positive_rank():
    with pytest.raises(ValidationError, match="requires peft.r"):
        compose_cfg("peft=lora", "peft.r=0")


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        compose_cfg("+train.nonexistent_option=1")


def test_limit_must_be_positive():
    with pytest.raises(ValidationError):
        compose_cfg("data.limit=0")


def test_warmup_cannot_consume_the_whole_run():
    with pytest.raises(ValidationError, match="nothing would be measured"):
        compose_cfg("train.steps=10", "train.warmup_discard_steps=10")


def test_profile_run_requires_profiler():
    with pytest.raises(ValidationError, match="requires profiler=true"):
        compose_cfg("run=profile", "run.profiler=false")


def test_batch_cannot_exceed_the_sample():
    """A batch larger than the dataset makes InfoNCE compare a row against itself."""
    with pytest.raises(ValidationError, match="rows would repeat inside a batch"):
        compose_cfg("data.limit=8", "train.batch_size=16")


def test_measured_runs_require_pinned_data():
    with pytest.raises(ValidationError, match="requires data.revision"):
        compose_cfg("run=timing", "data.revision=null")

    # A probe answers "does it run", so it does not need a pinned corpus.
    assert compose_cfg("run=probe", "data.revision=null")


def test_instruction_prompt_only_for_the_official_embedding_model():
    with pytest.raises(ValidationError, match="no official embedding prompt"):
        compose_cfg("model=qwen3_5_0_8b", "model.instruction_prompt='Represent this.'")


def test_per_model_usage_spec_matches_documented_decisions():
    """docs/model-spec.md decisions 1 and 2 live in config, not code."""
    vl = compose_cfg("model=qwen3_vl_emb_2b").model
    assert vl.add_generation_prompt is True
    assert vl.instruction_prompt == "Represent the user's input."

    for name in ("qwen3_5_0_8b", "gemma4_e2b"):
        generative = compose_cfg(f"model={name}").model
        assert generative.add_generation_prompt is False
        assert generative.instruction_prompt is None
