"""Typed config with fail-fast validation.

Hydra composes the config; Pydantic rejects invalid combinations before a run
starts. Several validators encode measurement rules rather than type constraints —
a run that would silently produce a misleading number must not start at all.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# torch.compile with autotuning spends its first steps benchmarking kernels. Below
# this many discarded steps the warmup leaks into the measured window.
MAX_AUTOTUNE_MIN_WARMUP_STEPS = 20

# A Hub revision is only pinned if it is a commit sha. Short shas are allowed
# because `git rev-parse --short` output is what gets pasted in practice.
COMMIT_SHA = re.compile(r"[0-9a-f]{7,40}")

# Deviation from the canonical baseline that invalidates a pod, and the default
# of `measurement.baseline_tolerance`. Uncalibrated: AGENTS.md carries this number
# without a source. `scripts/report.py` decides pod validity with its own copy;
# until it reads this one, the schema refuses any other value.
BASELINE_DEVIATION_LIMIT = 0.03

# One visual token under `Qwen2VLImageProcessor` covers patch_size**2 * merge_size**2
# pixels (transformers 5.14.1 image_processing_qwen2_vl.py: `_preprocess` divides the
# resized grid by `patch_size` then folds `merge_size**2` patches into one feature).
# Both currently-active models declare patch_size=16, merge_size=2 in their own
# `preprocessor_config.json` (qwen3_vl, qwen3_5) -> 16**2 * 2**2 = 1024. Scoped to
# those two archs by `_the_pixel_cap_fits_the_sequence_budget` below: gemma4 uses a
# different image processor and a different budget declaration
# (`model.max_tokens_per_image`), not this pixel arithmetic.
QWEN_PIXELS_PER_VISUAL_TOKEN = 16**2 * 2**2

# Subset revisions that must never be trained on again, with the reason a run
# that recorded one has to be discarded. Named rather than deleted from the Hub:
# results already exist that record this revision, and which corpus they were
# measured on has to stay decidable after the fact.
#
# Enforced here rather than only in scripts/prepare_data.py because that script
# is not in the path of a training run. A pin is copied into a config once and
# read on every run afterwards, so the refusal belongs where every run passes.
CORRUPT_DATA_REVISIONS = {
    "b750b9c3263e9ef5dce225fd50aa25d7c58f1d5f": (
        "defect D1: pos_image was dropped, so 466 rows share one placeholder "
        "positive and 644 rows lost their query image"
    ),
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def Axis(*args: Any, **kwargs: Any) -> Any:  # noqa: N802 - mirrors pydantic's Field
    """Mark a field as an ablation axis: a knob that changes what gets measured.

    Axes carry an obligation. `trainbench/applied.py` refuses to let a reportable
    run start until every one of them has been read back off the constructed model
    and matched against the request, because every axis here has a silent-fallback
    path. Marking is done on the schema rather than in a list elsewhere so that a
    new axis inherits the obligation instead of having to be remembered into it.
    """
    extra = {**(kwargs.pop("json_schema_extra", None) or {}), "axis": True}
    return Field(*args, json_schema_extra=extra, **kwargs)


def axis_knobs() -> dict[str, Callable[[BenchConfig], Any]]:
    """Every marked axis as `section.field` -> a reader for its configured value."""
    knobs: dict[str, Callable[[BenchConfig], Any]] = {}
    for section, info in BenchConfig.model_fields.items():
        sub = info.annotation
        if not (isinstance(sub, type) and issubclass(sub, Strict)):
            continue
        for name, spec in sub.model_fields.items():
            if (spec.json_schema_extra or {}).get("axis"):
                knobs[f"{section}.{name}"] = _reader(section, name)
    return knobs


def _reader(section: str, name: str) -> Callable[[BenchConfig], Any]:
    return lambda config: getattr(getattr(config, section), name)


class ModelConfig(Strict):
    name: str
    hf_id: str
    arch: Literal["qwen3_vl", "qwen3_5", "gemma4"]
    revision: str | None = None
    # How this model is meant to be used, per docs/model-spec.md. These live in
    # config rather than code because they differ per model and a wrong value is
    # invisible: with last-token pooling, add_generation_prompt changes *which*
    # token becomes the embedding.
    add_generation_prompt: bool
    # Official instruction prompt, where the model has one. None for the
    # generative models, which have no embedding spec — inventing one would add
    # an unvalidated confound.
    instruction_prompt: str | None = None
    # From tokenizer_config.json. gemma4 pads left and the other two right, which
    # decides which index last-token pooling must read. Declared here rather than
    # branched on `arch` in code so that a lane reading the pooling code sees the
    # value it has to handle instead of assuming one.
    padding_side: Literal["left", "right"]
    # How the embedding is formed. Only qwen3_vl has an official answer
    # (1_Pooling/config.json); for the generative models this is our choice, and
    # writing it down is what makes it reviewable.
    pooling: Literal["lasttoken"]
    # Whether this checkpoint ships a chat template. `raw` is the pre-trained
    # checkpoint that does not, and its rows carry no role or turn markers — a
    # difference in what is measured, not an implementation detail, which is why
    # it is declared here and read only by trainbench/prompt.py.
    prompt_format: Literal["chat_template", "raw"]
    # The most visual tokens one image can become, where the model declares a cap.
    # No model here has a fixed per-image count: gemma4's processor computes the
    # count from the aspect ratio and stops at max_soft_tokens=280, and the Qwen
    # models are pixel-proportional with no declared token cap at all (None). The
    # count itself is always measured, never assumed — fixing a single budget
    # across models was abandoned (docs/model-spec.md decision 3).
    max_tokens_per_image: int | None = Field(default=None, gt=0)


class DataConfig(Strict):
    # The fixed subset we train on, pushed by scripts/prepare_data.py. Its revision
    # is the data version recorded with every run (convention 07).
    repo_id: str
    revision: str | None = None
    # Upstream the subset is drawn from. A community mirror of MMEB-train that
    # embeds images; the authoritative TIGER-Lab repo stores paths only.
    source_repo: str
    # The mirror's commit. Required, and not defaulted to None: the sampler
    # streamed `source_repo`'s HEAD while this file's docstring and PLAN.md both
    # said the commit was pinned, so a subset could not be traced to the upstream
    # it came from. It is also part of the shard cache key — without it a cached
    # draw outlives the upstream change that would have invalidated it.
    source_revision: str
    subset_rows: int = Field(gt=0)
    sample_seed: int
    # Pushing creates/overwrites a Hub repo, so it is opt-in per invocation.
    push_subset: bool = False
    # Small-sample runs (convention 04). None means the full split.
    limit: int | None = Field(default=None, gt=0)
    max_seq_len: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    # The processor's pixel budget, applied identically to every model in the
    # campaign so the workload stops being a function of which images got sampled.
    # Lives here rather than in `configs/model/` because the sameness across models
    # is the point: two copies in two model files would drift. 1310720 is
    # `Qwen/Qwen3-VL-Embedding-2B`'s own shipped `max_pixels`
    # (`preprocessor_config.json`), not our arithmetic — the validator at
    # `config_schema.py:624-643` already refuses a Qwen model presenting derived
    # arithmetic as its own spec, and giving qwen3_5 a bigger number here would be
    # exactly that in reverse.
    max_pixels: int = Field(gt=0)
    min_pixels: int = Field(gt=0)
    # The whole corpus's text side (query + positive, no image, no truncation),
    # tokenised with each active model's own prompt format, worst case across both:
    # measured 2026-08-03 against this revision (see the field's value in each
    # `configs/data/*.yaml` for the numbers). A corpus regenerated under a new
    # revision invalidates this number; re-measure it, do not carry it forward.
    text_token_ceiling: int = Field(gt=0)

    @property
    def effective_rows(self) -> int:
        """Rows a run actually draws from: the small-sample limit, else the subset."""
        return self.limit if self.limit is not None else self.subset_rows

    @model_validator(mode="after")
    def _pixel_budget_is_ordered(self) -> DataConfig:
        if self.min_pixels > self.max_pixels:
            raise ValueError(
                f"data.min_pixels ({self.min_pixels}) exceeds data.max_pixels "
                f"({self.max_pixels}); the processor's own backward-compat merge "
                "(transformers 5.14.1 image_processing_qwen2_vl.py __init__) would "
                "build a size dict with shortest_edge > longest_edge."
            )
        return self

    @model_validator(mode="after")
    def _upstream_is_pinned_to_a_commit(self) -> DataConfig:
        """A branch name here would put the mirror's HEAD in the sample and leave no
        record of which HEAD, which is the state this field was added to end."""
        if not COMMIT_SHA.fullmatch(self.source_revision):
            raise ValueError(
                f"data.source_revision must be a commit sha, got {self.source_revision!r}; "
                "a branch or tag moves under the draw and cannot be recorded."
            )
        return self


# transformers' own name for each attention axis value. Derived from `name`
# rather than configured alongside it: a config free to say `name: fa3` with
# `impl: sdpa` would label the run fa3 while trainbench/applied.py certified
# sdpa-requested-sdpa-applied as a match, which is the exact failure applied.py
# exists to prevent.
ATTN_IMPL = {
    "sdpa": "sdpa",
    "fa2": "flash_attention_2",
    "fa3": "flash_attention_3",
    "fa4": "flash_attention_4",
    "flex": "flex_attention",
}


class AttnConfig(Strict):
    name: Literal["sdpa", "fa2", "fa3", "fa4", "flex"] = Axis()

    @property
    def impl(self) -> str:
        return ATTN_IMPL[self.name]


class KernelConfig(Strict):
    name: Literal["none", "liger", "fla"] = Axis()


class PrecisionConfig(Strict):
    name: Literal["bf16"] = Axis()


class CompileConfig(Strict):
    # "none" rather than "off": YAML parses a bare `off` as boolean False.
    mode: Literal["none", "default", "max-autotune", "regional"] = Axis()


class OptimConfig(Strict):
    name: Literal["adamw_fused", "adamw_8bit", "muon"] = Axis()
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class LossConfig(Strict):
    name: Literal["mnrl", "cached_mnrl"] = Axis()
    temperature: float = Field(gt=0)
    # GradCache mini-batch. Only meaningful for cached_mnrl.
    mini_batch: int | None = Field(default=None, gt=0)


class PeftConfig(Strict):
    mode: Literal["full", "lora", "qlora"] = Axis()
    r: int = Field(default=0, ge=0)
    alpha: int = Field(default=0, ge=0)
    dropout: float = Field(default=0.0, ge=0, le=1)


class FreezeConfig(Strict):
    vision_tower: bool = Axis(False)
    # Per-layer embeddings, gemma4 only.
    ple: bool = Axis(False)


class DataloaderConfig(Strict):
    backend: Literal["torch", "dali"] = Axis()
    packing: bool = Axis(False)
    # Tokenise ahead of the timed window instead of inside the dataloader.
    pretokenize: bool = Axis(False)


class ParallelConfig(Strict):
    strategy: Literal["single", "ddp", "fsdp2", "zero2", "zero3"] = Axis()
    # All-gather embeddings across ranks so in-batch negatives span the whole
    # world. This is the only parallelism axis specific to contrastive training,
    # and the project's central claim is that batch size dominates here.
    cross_device_negatives: bool = Axis(False)


class FrameworkConfig(Strict):
    name: Literal[
        "native", "unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"
    ] = Axis()


class TrainConfig(Strict):
    batch_size: int = Field(gt=0)
    grad_accum: int = Field(default=1, gt=0)
    steps: int = Field(gt=0)
    warmup_discard_steps: int = Field(ge=0)
    gradient_checkpointing: Literal["none", "full", "selective"] = Axis("none")
    offload: Literal["none", "optimizer", "param", "both"] = Axis("none")
    seed: int
    deterministic: bool = False


class MeasurementConfig(Strict):
    """How a reported figure is produced from the steps and the runs behind it.

    Each field replaced a constant that was written into the harness. A constant
    an adapter is free to choose differently is a difference the result
    attributes to the axis under test, and every one of these was measured to
    matter: padding reaches 89% of a batch on some corpora and decides which way
    the packing axis ranks, GPU contention alone has produced a 30x change in
    standard deviation, and `torch.compile` autotuning spends its first steps
    benchmarking kernels.

    **No default here is calibrated.** They are the values the harness already
    behaved as if it had, written down so a run records them; what this study
    should use is a question a pod answers after measuring the noise floor.
    `baseline_tolerance_calibrated` is how a reader tells the two apart.

    A field whose consumer is not written yet is pinned to its default by
    `_no_knob_is_declared_ahead_of_the_code_that_would_apply_it`, so that the
    block this config lands in never reads as applied while nothing applied it.
    """

    # MLPerf repeats its Small-LLM finetuning benchmark at least ten times. One
    # is what this harness does today, and one repeat has no spread at all.
    repeats: int = Field(default=1, gt=0)
    # Where the clock is read. `cuda_event` records events in the stream instead
    # of synchronising around a host clock; it is refused off CUDA rather than
    # falling back, because a wall-clock number under a device-measurement label
    # is the failure this whole module exists to prevent.
    instrument: Literal["wall_clock", "cuda_event"] = "wall_clock"
    aggregate: Literal["mean", "median", "trimmed_mean", "olympic"] = "mean"
    # Fraction removed from each end for `trimmed_mean`, and meaningless for the
    # others - a validator keeps the two from disagreeing.
    trim_fraction: float = Field(default=0.0, ge=0.0, lt=0.5)
    # `fixed` re-measures one point; `per_repeat` samples the distribution the
    # spread claims to describe. MLPerf CLOSED requires the latter and requires
    # every drawn seed to be logged.
    seed_policy: Literal["fixed", "per_repeat"] = "fixed"
    # Which token count divides the step time. Padding is not free - the forward
    # computes on it - so both answers are defensible and they rank the
    # `dataloader.packing` axis differently. Declared rather than assumed.
    throughput_denominator: Literal["tokens", "padded_tokens"] = "tokens"
    # Deviation from the canonical baseline that invalidates a pod.
    baseline_tolerance: float = Field(default=BASELINE_DEVIATION_LIMIT, gt=0.0)
    # False means the number above is the one AGENTS.md carries without a source.
    # It travels into the result so a reader is not left to assume it was derived.
    baseline_tolerance_calibrated: bool = False

    @property
    def tolerance_status(self) -> str:
        return "calibrated" if self.baseline_tolerance_calibrated else "uncalibrated"

    @model_validator(mode="after")
    def _the_trim_and_the_aggregate_agree(self) -> MeasurementConfig:
        """A `trim_fraction` under an aggregate that ignores it is a knob that
        reads as applied and changes nothing, and `trimmed_mean` with nothing
        trimmed is an arithmetic mean wearing another name in the result."""
        if self.aggregate == "trimmed_mean" and self.trim_fraction <= 0:
            raise ValueError(
                "measurement.aggregate=trimmed_mean requires trim_fraction > 0; with "
                "nothing trimmed it is the arithmetic mean reported under another name."
            )
        if self.aggregate != "trimmed_mean" and self.trim_fraction > 0:
            raise ValueError(
                f"measurement.trim_fraction={self.trim_fraction} is set under "
                f"aggregate={self.aggregate}, which does not trim; the knob would be "
                "recorded as applied while changing nothing."
            )
        return self

    @model_validator(mode="after")
    def _no_knob_is_declared_ahead_of_the_code_that_would_apply_it(self) -> MeasurementConfig:
        """The rule above, applied to the fields whose consumer does not exist yet.

        Every field here lands in `metrics.summarise`'s `measurement` block, which
        a reader takes as how the figure beside it was produced. Four of them are
        read by nothing, so any value but the one the harness already behaves as if
        it had would be recorded as applied while changing nothing. Each refusal
        names what has to land before the value becomes declarable; deleting the
        refusal is then a one-line part of landing it.
        """
        if self.repeats != 1:
            raise ValueError(
                f"measurement.repeats={self.repeats} but scripts/bench.py runs one "
                "measured window per process and has no repeat loop; the record would "
                "carry a repeat count nothing repeated, beside a `step_seconds_stdev` "
                "that is the spread within one run."
            )
        if self.seed_policy != "fixed":
            raise ValueError(
                f"measurement.seed_policy={self.seed_policy} needs repeats > 1 to mean "
                "anything, and no repeat loop draws the seeds `metrics.repeat_seeds` "
                "would produce; the record would name a sampling policy nothing sampled."
            )
        if self.throughput_denominator != "tokens":
            raise ValueError(
                f"measurement.throughput_denominator={self.throughput_denominator} but "
                "scripts/report.py ranks on `tokens_per_second` unconditionally and "
                "renders no padded-token rate; the declared denominator and the "
                "published one would differ with nothing in the report saying so."
            )
        if (
            self.baseline_tolerance != BASELINE_DEVIATION_LIMIT
            or self.baseline_tolerance_calibrated
        ):
            raise ValueError(
                f"measurement.baseline_tolerance={self.baseline_tolerance} "
                f"(calibrated={self.baseline_tolerance_calibrated}) but pod validity is "
                "decided by scripts/report.py's own BASELINE_DEVIATION_LIMIT; a record "
                "claiming a calibrated threshold would sit beside a table that used "
                f"{BASELINE_DEVIATION_LIMIT}."
            )
        return self


class RunConfig(Strict):
    # timing  : the numbers we report. Profiler and deterministic mode are banned.
    # profile : kernel breakdown for diagnosis. Its timings are not reportable.
    # quality : long run for loss curve and retrieval metrics.
    # probe   : Phase 0 load-and-one-step check.
    purpose: Literal["timing", "profile", "quality", "probe"]
    profiler: bool = False
    # No experiment-tracking fields. Results are JSON written by
    # `trainbench/record.py`, and a tracker that reaches the network during a
    # measured step perturbs the thing being measured (PLAN.md decision 3).
    # The output directory is Hydra's (`hydra.run.dir`); duplicating it here would
    # give two sources of truth. Read it at runtime via config.output_dir().


class BenchConfig(Strict):
    device: str | None = None
    run: RunConfig
    model: ModelConfig
    data: DataConfig
    attn: AttnConfig
    kernel: KernelConfig
    precision: PrecisionConfig
    compile: CompileConfig
    optim: OptimConfig
    loss: LossConfig
    peft: PeftConfig
    freeze: FreezeConfig
    dataloader: DataloaderConfig
    parallel: ParallelConfig
    framework: FrameworkConfig
    train: TrainConfig
    # Defaulted rather than composed: there is no `configs/measurement/` group
    # yet, and creating one needs `configs/config.yaml` and the audit's group
    # tables, both of which belong to the integration wave (.plans/notes/measure.md).
    # Every field with a consumer is overridable as `+measurement.<field>=...` in
    # the meantime; the rest are pinned to their defaults by the schema.
    measurement: MeasurementConfig = MeasurementConfig()

    @model_validator(mode="after")
    def _timing_runs_are_uncontaminated(self) -> BenchConfig:
        """The profiler inflates iteration time and deterministic mode disables the
        kernel autotuning under measurement. Neither may be on for a run whose
        numbers get reported.

        No percentage is quoted here on purpose: the figure this docstring used to
        carry had no source, and the overhead varies with hardware and with which
        profiler options are on. docs/methodology.md records the gap and how it is
        closed."""
        if self.run.purpose == "timing":
            if self.run.profiler:
                raise ValueError(
                    "purpose=timing forbids profiler=true: profiling inflates step time. "
                    "Run a separate purpose=profile job for kernel breakdown."
                )
            if self.train.deterministic:
                raise ValueError(
                    "purpose=timing forbids deterministic=true: it disables the kernel "
                    "autotuning being measured. See docs/methodology.md."
                )
        return self

    @model_validator(mode="after")
    def _autotune_needs_warmup(self) -> BenchConfig:
        if (
            self.compile.mode == "max-autotune"
            and self.train.warmup_discard_steps < MAX_AUTOTUNE_MIN_WARMUP_STEPS
        ):
            raise ValueError(
                f"compile=max-autotune needs warmup_discard_steps >= "
                f"{MAX_AUTOTUNE_MIN_WARMUP_STEPS}, got {self.train.warmup_discard_steps}: "
                "autotuning would leak into the measured window."
            )
        return self

    @model_validator(mode="after")
    def _gradcache_mini_batch_fits(self) -> BenchConfig:
        if self.loss.name == "cached_mnrl":
            if self.loss.mini_batch is None:
                raise ValueError("loss=cached_mnrl requires loss.mini_batch")
            if self.loss.mini_batch > self.train.batch_size:
                raise ValueError(
                    f"loss.mini_batch ({self.loss.mini_batch}) exceeds train.batch_size "
                    f"({self.train.batch_size}); GradCache splits the batch, it cannot grow it."
                )
        return self

    @model_validator(mode="after")
    def _ple_is_gemma4_only(self) -> BenchConfig:
        if self.freeze.ple and self.model.arch != "gemma4":
            raise ValueError(
                f"freeze.ple applies to per-layer embeddings, which only exist in gemma4; "
                f"model arch is {self.model.arch}."
            )
        return self

    @model_validator(mode="after")
    def _warmup_leaves_a_measured_window(self) -> BenchConfig:
        """Discarding more steps than exist measures nothing at all. The
        max-autotune rule below sets a floor on warmup; this sets the ceiling."""
        if self.train.warmup_discard_steps >= self.train.steps:
            raise ValueError(
                f"train.warmup_discard_steps ({self.train.warmup_discard_steps}) must be "
                f"less than train.steps ({self.train.steps}); nothing would be measured."
            )
        return self

    @model_validator(mode="after")
    def _the_aggregate_has_samples_to_aggregate(self) -> BenchConfig:
        """The measured window has to be wide enough for the statistic chosen over it.

        Olympic scoring discards the fastest and the slowest step, so a two-step
        window leaves nothing; a trim fraction can remove every sample the same
        way. Refused before the run rather than after it, because both failures
        surface only once the steps have already been paid for.
        """
        measured = self.train.steps - self.train.warmup_discard_steps
        method = self.measurement.aggregate
        if method == "olympic" and measured < 3:
            raise ValueError(
                f"measurement.aggregate=olympic drops the fastest and the slowest step, so "
                f"it needs at least 3 measured steps; train.steps ({self.train.steps}) minus "
                f"train.warmup_discard_steps ({self.train.warmup_discard_steps}) leaves "
                f"{measured}."
            )
        if method == "trimmed_mean" and int(measured * self.measurement.trim_fraction) * 2 >= (
            measured
        ):
            raise ValueError(
                f"measurement.trim_fraction={self.measurement.trim_fraction} removes every "
                f"one of the {measured} measured step(s) from both ends; a mean over nothing "
                "is not a measurement."
            )
        return self

    @model_validator(mode="after")
    def _the_instrument_exists_on_the_requested_device(self) -> BenchConfig:
        """CUDA events cannot time a CPU run. Refused here when the device is
        named, and again in `metrics.build_timer` when it is resolved at runtime —
        `device: null` lets `trainbench.device` pick, so this validator alone
        would leave the case it was written for unguarded."""
        if self.measurement.instrument == "cuda_event" and self.device is not None:
            if not str(self.device).startswith("cuda"):
                raise ValueError(
                    f"measurement.instrument=cuda_event with device={self.device!r}: CUDA "
                    "events do not exist there, and falling back to the host clock would "
                    "report a wall-clock number under a device-measurement label."
                )
        return self

    @model_validator(mode="after")
    def _profile_runs_actually_profile(self) -> BenchConfig:
        if self.run.purpose == "profile" and not self.run.profiler:
            raise ValueError("purpose=profile requires profiler=true")
        return self

    @model_validator(mode="after")
    def _batch_fits_the_sample(self) -> BenchConfig:
        """A batch larger than the dataset reuses rows within one batch, which makes
        InfoNCE compare a row against itself and quietly destroys the loss.

        Checked against `effective_rows`, not `limit`: a config with `limit: null`
        still draws from a finite subset, and checking only the small-sample knob
        would leave the full-size configs unguarded.
        """
        rows = self.data.effective_rows
        if self.train.batch_size > rows:
            source = "data.limit" if self.data.limit is not None else "data.subset_rows"
            raise ValueError(
                f"train.batch_size ({self.train.batch_size}) exceeds {source} ({rows}); "
                "rows would repeat inside a batch and in-batch negatives would include "
                "the positive itself."
            )
        return self

    @model_validator(mode="after")
    def _measured_runs_pin_their_data(self) -> BenchConfig:
        """A number is not reproducible if the data it came from is a moving branch
        (convention 07 requires the data version on every run).

        A branch name is rejected as firmly as null. `revision: main` reads as
        pinned and is not: the Hub resolves it again on every pull.
        """
        if self.run.purpose not in ("timing", "quality"):
            return self
        revision = self.data.revision
        if revision is None:
            raise ValueError(
                f"purpose={self.run.purpose} requires data.revision to be pinned; "
                "'null' tracks a branch and makes the run unreproducible."
            )
        if not COMMIT_SHA.fullmatch(revision):
            raise ValueError(
                f"purpose={self.run.purpose} requires data.revision to be a commit sha, "
                f"got {revision!r}; a branch or tag moves under the run."
            )
        return self

    @model_validator(mode="after")
    def _no_run_reads_a_corrupt_subset(self) -> BenchConfig:
        """Refuse a pin that is known to name damaged data, for any purpose.

        Not restricted to measured runs like the check above: a probe or a smoke
        run against a corpus whose positives collapsed reports that the pipeline
        works, which is how the corruption survived its first review. The
        prefix match accepts the short shas COMMIT_SHA allows.
        """
        revision = self.data.revision
        if revision is None:
            return self
        for corrupt, reason in CORRUPT_DATA_REVISIONS.items():
            if corrupt.startswith(revision.lower()):
                raise ValueError(
                    f"data.revision={revision!r} is a known-corrupt subset of "
                    f"{self.data.repo_id}: {reason}. Regenerate with "
                    "scripts/prepare_data.py and pin the new revision."
                )
        return self

    @model_validator(mode="after")
    def _freeze_axes_mean_nothing_under_an_adapter(self) -> BenchConfig:
        """An adapter run cannot also be a freeze-axis run.

        Measured (peft 0.20.0): `get_peft_model` sets `requires_grad=False` on every
        base parameter, and the result is identical whether or not a freeze axis ran
        first. `freeze.ple=true` and `freeze.ple=false` therefore build the same
        model under LoRA — the axis has no state to be in.

        Refused here rather than at the axis, because the two settings are not a
        mismatch to be caught at capture time; they are a request for a comparison
        that does not exist. Leaving it legal would put two rows in the ablation
        table whose only difference is their labels.
        """
        frozen = [name for name in ("vision_tower", "ple") if getattr(self.freeze, name)]
        if self.peft.mode != "full" and frozen:
            raise ValueError(
                f"peft.mode={self.peft.mode} freezes every base parameter, so "
                f"freeze.{'/freeze.'.join(frozen)}=true selects nothing; use freeze=none "
                "with an adapter, or peft=full to measure the freeze axes."
            )
        return self

    @model_validator(mode="after")
    def _adapter_rank_is_set_when_an_adapter_is_used(self) -> BenchConfig:
        """`r=0` builds a LoRA with no trainable adapter parameters, which trains
        nothing at all while reporting itself as a LoRA run."""
        if self.peft.mode != "full" and self.peft.r <= 0:
            raise ValueError(f"peft.mode={self.peft.mode} requires peft.r > 0, got {self.peft.r}")
        return self

    @model_validator(mode="after")
    def _instruction_prompt_is_official_only(self) -> BenchConfig:
        """Only qwen3_vl ships an official embedding prompt. Inventing one for the
        generative models would add an unvalidated confound (docs/model-spec.md)."""
        if self.model.instruction_prompt is not None and self.model.arch != "qwen3_vl":
            raise ValueError(
                f"model.instruction_prompt is set for arch={self.model.arch}, which has "
                "no official embedding prompt. See docs/model-spec.md decision 1."
            )
        return self

    @model_validator(mode="after")
    def _an_image_token_cap_is_gemma4_only(self) -> BenchConfig:
        """gemma4's processor declares max_soft_tokens=280 and computes each image's
        count from its aspect ratio, so 280 bounds the count without being it. The
        Qwen models declare a pixel range and no token cap, so a number there would
        be our arithmetic presented as the model's spec, and it would reintroduce the
        cross-model token budget that decision 3 abandoned (docs/model-spec.md)."""
        capped = self.model.max_tokens_per_image is not None
        if capped and self.model.arch != "gemma4":
            raise ValueError(
                f"model.max_tokens_per_image is set for arch={self.model.arch}, which "
                "declares a pixel range and no token cap; the count must be measured."
            )
        if not capped and self.model.arch == "gemma4":
            raise ValueError(
                "arch=gemma4's processor declares max_soft_tokens; "
                "model.max_tokens_per_image must carry it so a measured count above the "
                "cap is caught instead of published."
            )
        return self

    @model_validator(mode="after")
    def _the_pixel_cap_fits_the_sequence_budget(self) -> BenchConfig:
        """A batch of 1 can already overflow `data.max_seq_len` — measured: a single
        image-bearing row reached 16281 tokens under Qwen3.5-0.8B's shipped
        `max_pixels` before `data.max_pixels` capped it. Refused here rather than
        four minutes into a pod, which is what it cost twice on 2026-08-03.

        The inequality: `data.text_token_ceiling` (one side of a pair, no image) +
        `ceil(data.max_pixels / QWEN_PIXELS_PER_VISUAL_TOKEN)` (one image, the most
        a row carries — query and positive are separate rows) must not exceed
        `data.max_seq_len`.

        Scoped to qwen3_vl/qwen3_5: gemma4 uses a different image processor and a
        different budget declaration (`model.max_tokens_per_image`), not this pixel
        arithmetic, and it is being dropped from the campaign regardless
        (`_an_image_token_cap_is_gemma4_only` above is its still-standing check).

        Scoped to `timing`/`quality` the same way `_measured_runs_pin_their_data`
        is: a probe answers "does it run" over whatever `data.max_seq_len` a test
        constructs to reach a different check, and is not a claim about a pod's
        real throughput capacity.
        """
        if self.run.purpose not in ("timing", "quality"):
            return self
        if self.model.arch not in ("qwen3_vl", "qwen3_5"):
            return self
        image_tokens = -(-self.data.max_pixels // QWEN_PIXELS_PER_VISUAL_TOKEN)
        worst_case = self.data.text_token_ceiling + image_tokens
        if worst_case > self.data.max_seq_len:
            raise ValueError(
                f"data.text_token_ceiling ({self.data.text_token_ceiling}) + "
                f"ceil(data.max_pixels / {QWEN_PIXELS_PER_VISUAL_TOKEN}) ({image_tokens}) "
                f"= {worst_case} exceeds data.max_seq_len ({self.data.max_seq_len}); "
                "lower data.max_pixels or raise data.max_seq_len before this reaches a pod."
            )
        return self

    @model_validator(mode="after")
    def _a_raw_prompt_has_no_generation_prompt(self) -> BenchConfig:
        """`add_generation_prompt` is an argument to `apply_chat_template`, and
        prompt_format=raw is the case where there is no template to pass it to. With
        last-token pooling it decides which token becomes the embedding, so the two
        cannot be left to disagree — the run would report a value it never applied."""
        if self.model.prompt_format == "raw" and self.model.add_generation_prompt:
            raise ValueError(
                "model.add_generation_prompt is true under model.prompt_format=raw, which "
                "has no chat template to append a generation prompt to."
            )
        return self

    @model_validator(mode="after")
    def _lora_needs_rank(self) -> BenchConfig:
        if self.peft.mode in ("lora", "qlora") and self.peft.r <= 0:
            raise ValueError(f"peft.mode={self.peft.mode} requires peft.r > 0")
        return self
