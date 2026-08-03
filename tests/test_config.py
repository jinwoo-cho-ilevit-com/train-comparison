"""Invalid config combinations must fail before a run starts, not after."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from trainbench.compose import resolve
from trainbench.config_schema import (
    BASELINE_DEVIATION_LIMIT,
    CORRUPT_DATA_REVISIONS,
    BenchConfig,
)
from trainbench.metrics import repeat_seeds

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
    for name in ("qwen3_vl_emb_2b", "qwen3_5_0_8b"):
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


def test_padding_side_is_declared_per_model():
    """Declared per model rather than branched on `arch` in code, so the value is
    visible to the code that pools rather than assumed. (gemma-4-E2B, the one
    model that padded left, left the campaign 2026-08-03 — PLAN.md "gemma-4-E2B
    제외" — so both surviving models pad right; the field stays per-model because
    a future model could differ.)"""
    for name in ("qwen3_vl_emb_2b", "qwen3_5_0_8b"):
        assert compose_cfg(f"model={name}").model.padding_side == "right"


def test_prompt_format_is_declared_per_model():
    """Measured 2026-08-02 against the Hub: both surviving repositories ship
    chat_template.jinja. (`google/gemma-4-E2B` did not — it was the pre-trained
    checkpoint, and only `google/gemma-4-E2B-it` ships one; calling
    `apply_chat_template` regardless is what failed it on three frameworks. It
    left the campaign 2026-08-03 for an unrelated reason — full finetuning does
    not fit one A100 80GB — and `prompt_format=raw` is exercised directly by
    `test_a_raw_prompt_format_refuses_a_generation_prompt` below instead.)"""
    for name in ("qwen3_vl_emb_2b", "qwen3_5_0_8b"):
        assert compose_cfg(f"model={name}").model.prompt_format == "chat_template"


def test_a_raw_prompt_format_refuses_a_generation_prompt():
    """`add_generation_prompt` is an argument to `apply_chat_template`, and raw has
    no template to pass it to. With last-token pooling the pair would otherwise
    report a value that decided nothing.

    No surviving model declares `prompt_format=raw` (gemma-4-E2B did, and left
    the campaign 2026-08-03), so the value is asserted directly over a real
    model rather than skipped: the validator is `model.prompt_format`-scoped, not
    arch-scoped, and remains reachable by any future raw-format model.
    """
    with pytest.raises(ValidationError, match="no chat template"):
        compose_cfg(
            "model=qwen3_5_0_8b", "model.prompt_format=raw", "model.add_generation_prompt=true"
        )


def test_instruction_prompt_only_for_the_official_embedding_model():
    with pytest.raises(ValidationError, match="no official embedding prompt"):
        compose_cfg("model=qwen3_5_0_8b", "model.instruction_prompt='Represent this.'")


def test_per_model_usage_spec_matches_documented_decisions():
    """docs/model-spec.md decisions 1 and 2 live in config, not code."""
    vl = compose_cfg("model=qwen3_vl_emb_2b").model
    assert vl.add_generation_prompt is True
    assert vl.instruction_prompt == "Represent the user's input."

    generative = compose_cfg("model=qwen3_5_0_8b").model
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


def test_an_adapter_run_cannot_also_request_a_freeze_axis():
    """Measured, not assumed (tests/test_axes.py): `get_peft_model` freezes every
    base parameter regardless of what a freeze axis did first, so freezing it
    again builds the same model under an adapter. Two identical models must not
    occupy two rows of the ablation table under different labels."""
    with pytest.raises(ValidationError, match="freezes every base parameter"):
        compose_cfg("peft=lora", "freeze.vision_tower=true")


def test_the_freeze_axes_stay_available_to_a_full_finetune():
    """The refusal above is about adapters, not about freezing. A check that
    refused both would have removed the axis this study measures."""
    assert compose_cfg("peft=full", "freeze=vision_tower").freeze.vision_tower is True


def test_an_adapter_with_no_rank_is_refused():
    """`r=0` builds an adapter with no trainable parameter: a LoRA run that trains
    nothing while reporting itself as one."""
    with pytest.raises(ValidationError, match="requires peft.r"):
        compose_cfg("peft=lora", "peft.r=0")


# ---------------------------------------------------------------------------
# measurement: seed policy, the deviation threshold, and the statistics knobs
# ---------------------------------------------------------------------------


def test_a_repeat_count_no_loop_repeats_is_refused_before_it_reaches_the_record():
    """`scripts/bench.py` runs one measured window per process. A declared
    `repeats=10` therefore changes nothing and still lands in the `measurement`
    block of every record, beside a `step_seconds_stdev` that is the spread of the
    steps within one run — a reader takes it for the spread over ten.

    This is the shape `_the_trim_and_the_aggregate_agree` already refuses one field
    over. The refusal names the missing loop, so landing the loop and deleting the
    refusal are the same change.
    """
    with pytest.raises(ValidationError, match="no repeat loop"):
        compose_cfg("+measurement.repeats=10")
    assert compose_cfg().measurement.repeats == 1


def test_a_seed_policy_that_samples_nothing_is_refused_with_the_repeat_loop_absent():
    """MLPerf CLOSED draws each run's seed from `/dev/urandom` and requires that no
    two runs log the same one. With one measured window there is nothing to draw
    for, so `per_repeat` in a record would name a sampling policy nothing sampled.

    The schema keeps the literal and `metrics.repeat_seeds` keeps the behaviour, so
    what is missing is the loop and not the ability to express the policy.
    """
    with pytest.raises(ValidationError, match="repeats > 1"):
        compose_cfg("+measurement.seed_policy=per_repeat")
    assert compose_cfg().measurement.seed_policy == "fixed"

    seeds = repeat_seeds("per_repeat", 4, 1234)
    assert len(set(seeds)) == 4, f"two repeats share a seed: {seeds}"
    assert seeds != repeat_seeds("per_repeat", 4, 1234), (
        "two draws produced the same four seeds, so the policy is deriving them from "
        "the base seed rather than sampling"
    )
    assert repeat_seeds("fixed", 4, 1234) == (1234,) * 4


def test_an_unknown_seed_policy_is_refused_rather_than_defaulted():
    with pytest.raises(ValidationError):
        compose_cfg("+measurement.seed_policy=urandom")
    with pytest.raises(ValueError, match="unknown measurement.seed_policy"):
        repeat_seeds("urandom", 2, 1234)


def test_a_deviation_threshold_no_report_reads_cannot_be_declared_calibrated():
    """AGENTS.md's 3% has no source, and GPU contention alone has moved a step-time
    standard deviation by 30x elsewhere. Pod validity is nevertheless decided by
    `scripts/report.py`'s own `BASELINE_DEVIATION_LIMIT`, which reads nothing from
    here — so a pod that measured its noise floor and declared 8.1% would publish
    records claiming a calibrated threshold beside a table that used 3%.

    Until the report reads this field the schema pins it, and
    `config_schema.BASELINE_DEVIATION_LIMIT` is the one value for the report to
    read when it does.
    """
    config = compose_cfg()
    assert config.measurement.baseline_tolerance == BASELINE_DEVIATION_LIMIT
    assert config.measurement.baseline_tolerance_calibrated is False
    assert config.measurement.tolerance_status == "uncalibrated"

    with pytest.raises(ValidationError, match="BASELINE_DEVIATION_LIMIT"):
        compose_cfg("+measurement.baseline_tolerance=0.081")
    with pytest.raises(ValidationError, match="BASELINE_DEVIATION_LIMIT"):
        compose_cfg("+measurement.baseline_tolerance_calibrated=true")


def test_a_throughput_denominator_no_table_ranks_on_is_refused():
    """Padding reaches 89% of a batch on some corpora, and which token count
    divides the step time reverses the `dataloader.packing` ranking — which is why
    the field exists. `scripts/report.py` ranks on `tokens_per_second`
    unconditionally and renders no padded-token rate, so a declared
    `padded_tokens` would put a denominator in the record that no published figure
    used, invisibly to anyone reading the report."""
    with pytest.raises(ValidationError, match="padded-token rate"):
        compose_cfg("+measurement.throughput_denominator=padded_tokens")
    assert compose_cfg().measurement.throughput_denominator == "tokens"


def test_a_trim_fraction_and_an_aggregate_that_ignores_it_cannot_coexist():
    """Either direction is a knob recorded as applied while changing nothing:
    `trimmed_mean` with nothing trimmed is the arithmetic mean under another name,
    and a trim fraction under `mean` is a setting the result reports and the run
    never used."""
    with pytest.raises(ValidationError, match="requires trim_fraction"):
        compose_cfg("+measurement.aggregate=trimmed_mean")
    with pytest.raises(ValidationError, match="does not trim"):
        compose_cfg("+measurement.trim_fraction=0.1")

    assert compose_cfg(
        "+measurement.aggregate=trimmed_mean", "+measurement.trim_fraction=0.1"
    ).measurement.trim_fraction == pytest.approx(0.1)


def test_an_aggregate_wider_than_the_measured_window_is_refused_before_the_run():
    """Olympic scoring drops the fastest and the slowest step; a two-step window
    leaves nothing to average. Caught before the steps are paid for."""
    with pytest.raises(ValidationError, match="at least 3 measured steps"):
        compose_cfg(
            "+measurement.aggregate=olympic", "train.steps=4", "train.warmup_discard_steps=2"
        )

    assert (
        compose_cfg(
            "+measurement.aggregate=olympic", "train.steps=6", "train.warmup_discard_steps=2"
        ).measurement.aggregate
        == "olympic"
    )


def test_cuda_event_timing_is_refused_on_a_device_that_has_no_events():
    """A silent fall back to the host clock would report a wall-clock number under
    a device-measurement label — the same shape as `kernel=none` picking up an
    environment-provided fla."""
    with pytest.raises(ValidationError, match="CUDA events do not exist"):
        compose_cfg("device=cpu", "+measurement.instrument=cuda_event")
