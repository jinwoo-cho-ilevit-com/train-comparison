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
    # Visual tokens one image becomes. gemma4 is fixed at 280; the Qwen models are
    # pixel-proportional, so None means "measure it, do not assume it". Fixing a
    # single budget across models was abandoned (docs/model-spec.md decision 3).
    tokens_per_image: int | None = Field(default=None, gt=0)


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

    @property
    def effective_rows(self) -> int:
        """Rows a run actually draws from: the small-sample limit, else the subset."""
        return self.limit if self.limit is not None else self.subset_rows

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
    name: Literal["none", "liger", "fla", "kernels_hub"] = Axis()


class PrecisionConfig(Strict):
    name: Literal["bf16", "mxfp8", "nvfp4"] = Axis()


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


class RunConfig(Strict):
    # timing  : the numbers we report. Profiler and deterministic mode are banned.
    # profile : kernel breakdown for diagnosis. Its timings are not reportable.
    # quality : long run for loss curve and retrieval metrics.
    # probe   : Phase 0 load-and-one-step check.
    purpose: Literal["timing", "profile", "quality", "probe"]
    profiler: bool = False
    trackio_project: str
    trackio_space_id: str | None = None
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
    def _fixed_image_tokens_are_gemma4_only(self) -> BenchConfig:
        """gemma4 expands every image to a fixed 280 soft tokens; both Qwen models
        are pixel-proportional. Declaring a fixed count for a dynamic model would
        reintroduce the cross-model token budget that decision 3 abandoned as
        impossible (docs/model-spec.md)."""
        fixed = self.model.tokens_per_image is not None
        if fixed and self.model.arch != "gemma4":
            raise ValueError(
                f"model.tokens_per_image is set for arch={self.model.arch}, whose image "
                "token count is pixel-proportional and must be measured, not declared."
            )
        if not fixed and self.model.arch == "gemma4":
            raise ValueError(
                "arch=gemma4 has a fixed image_seq_length; model.tokens_per_image must "
                "declare it so token accounting does not silently assume a dynamic one."
            )
        return self

    @model_validator(mode="after")
    def _lora_needs_rank(self) -> BenchConfig:
        if self.peft.mode in ("lora", "qlora") and self.peft.r <= 0:
            raise ValueError(f"peft.mode={self.peft.mode} requires peft.r > 0")
        return self
