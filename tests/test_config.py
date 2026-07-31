"""Invalid config combinations must fail before a run starts, not after."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from trainbench.config import to_bench_config
from trainbench.config_schema import BenchConfig

from .conftest import CONFIG_DIR


def compose_cfg(*overrides: str) -> BenchConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
        return to_bench_config(cfg)


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
