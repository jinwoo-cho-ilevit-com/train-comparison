"""Invalid config combinations must fail before a run starts, not after."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from trainbench.compose import resolve
from trainbench.config_schema import CORRUPT_DATA_REVISIONS, BenchConfig

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


def test_batch_cannot_exceed_the_subset_without_a_limit():
    """`quality.yaml` sets limit to null, so checking only the small-sample knob
    leaves the full-size configs open to the same self-negative destruction."""
    with pytest.raises(ValidationError, match="exceeds data.subset_rows"):
        compose_cfg("data.limit=null", "data.subset_rows=8", "train.batch_size=16")


def test_measured_runs_require_pinned_data():
    with pytest.raises(ValidationError, match="requires data.revision"):
        compose_cfg("run=timing", "data.revision=null")

    # A probe answers "does it run", so it does not need a pinned corpus.
    assert compose_cfg("run=probe", "data.revision=null")


def test_a_branch_name_does_not_count_as_pinned():
    """`revision: main` reads as pinned and is not — the Hub re-resolves it on
    every pull, so two runs a week apart can train on different corpora."""
    with pytest.raises(ValidationError, match="a branch or tag moves under the run"):
        compose_cfg("run=timing", "data.revision=main")


def test_every_attention_axis_value_has_an_implementation_name():
    """`config.attn.impl` is what applied.py compares against. A Literal value with
    no entry in the map would raise on read, which capture turns into undetermined
    and therefore a blocked timing run — fail-safe, but the axis would be dead."""
    from typing import get_args

    from trainbench.config_schema import ATTN_IMPL, AttnConfig

    assert set(get_args(AttnConfig.model_fields["name"].annotation)) == set(ATTN_IMPL)


def test_attention_impl_cannot_disagree_with_its_label():
    """A config free to name the axis fa3 while asking transformers for sdpa
    would be labelled fa3 and certified as a match by applied.py."""
    assert compose_cfg("attn=fa3").attn.impl == "flash_attention_3"
    with pytest.raises(ValidationError, match="attn"):
        compose_cfg("attn=fa3", "+attn.impl=sdpa")


def test_a_fixed_image_token_count_belongs_only_to_gemma4():
    """gemma4 expands every image to 280 soft tokens; both Qwen models are
    pixel-proportional, so a declared count there is an assumption, not a fact."""
    assert compose_cfg("model=gemma4_e2b").model.tokens_per_image == 280
    for name in ("qwen3_vl_emb_2b", "qwen3_5_0_8b"):
        assert compose_cfg(f"model={name}").model.tokens_per_image is None
        with pytest.raises(ValidationError, match="must be measured, not declared"):
            compose_cfg(f"model={name}", "model.tokens_per_image=280")

    with pytest.raises(ValidationError, match="must declare it"):
        compose_cfg("model=gemma4_e2b", "model.tokens_per_image=null")


def test_padding_side_is_declared_per_model():
    """The only model that pads left is the one whose pooling branch was wrong,
    so the value has to be visible to the code that pools rather than assumed."""
    assert compose_cfg("model=gemma4_e2b").model.padding_side == "left"
    for name in ("qwen3_vl_emb_2b", "qwen3_5_0_8b"):
        assert compose_cfg(f"model={name}").model.padding_side == "right"


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


def test_a_corrupt_subset_revision_is_refused_for_every_purpose():
    """D1 shipped because a probe against the damaged corpus reported that the
    pipeline worked. Restricting this to measured runs would leave that path open,
    so the refusal is not conditioned on `run.purpose`."""
    corrupt = next(iter(CORRUPT_DATA_REVISIONS))
    for purpose in ("probe", "timing"):
        with pytest.raises(ValidationError, match="known-corrupt subset"):
            compose_cfg(f"run={purpose}", f"data.revision={corrupt}")


def test_the_corrupt_check_accepts_the_short_shas_the_pin_check_allows():
    """`data.revision` may be a 7-character sha, and a denylist that only matched
    full ones would wave through the same corpus written a shorter way."""
    corrupt = next(iter(CORRUPT_DATA_REVISIONS))
    with pytest.raises(ValidationError, match="known-corrupt subset"):
        compose_cfg(f"data.revision={corrupt[:7]}")


def test_the_pinned_subsets_are_not_on_the_denylist():
    """The configs that ship must compose. This is what caught the denylist being
    landed before the regeneration it depends on."""
    for name in ("speed", "quality"):
        assert compose_cfg(f"data={name}").data.revision not in CORRUPT_DATA_REVISIONS
