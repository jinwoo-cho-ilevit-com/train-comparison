"""Typed config with fail-fast validation.

Hydra composes the config; Pydantic rejects invalid combinations before a run
starts. Several validators encode measurement rules rather than type constraints —
a run that would silently produce a misleading number must not start at all.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# torch.compile with autotuning spends its first steps benchmarking kernels. Below
# this many discarded steps the warmup leaks into the measured window.
MAX_AUTOTUNE_MIN_WARMUP_STEPS = 20


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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


class DataConfig(Strict):
    # The fixed subset we train on, pushed by scripts/prepare_data.py. Its revision
    # is the data version recorded with every run (convention 07).
    repo_id: str
    revision: str | None = None
    # Upstream the subset is drawn from. A community mirror of MMEB-train that
    # embeds images; the authoritative TIGER-Lab repo stores paths only.
    source_repo: str
    subset_rows: int = Field(gt=0)
    sample_seed: int
    # Pushing creates/overwrites a Hub repo, so it is opt-in per invocation.
    push_subset: bool = False
    # Small-sample runs (convention 04). None means the full split.
    limit: int | None = Field(default=None, gt=0)
    # Visual token count differs per model for the same image (merge/pooling differ),
    # so it is pinned explicitly and corrected per model after measurement.
    image_token_budget: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    num_workers: int = Field(ge=0)


class AttnConfig(Strict):
    name: Literal["sdpa", "fa2", "fa3", "fa4", "flex"]
    impl: str


class KernelConfig(Strict):
    name: Literal["none", "liger", "fla", "kernels_hub"]


class PrecisionConfig(Strict):
    name: Literal["bf16", "mxfp8", "nvfp4"]


class CompileConfig(Strict):
    # "none" rather than "off": YAML parses a bare `off` as boolean False.
    mode: Literal["none", "default", "max-autotune", "regional"]


class OptimConfig(Strict):
    name: Literal["adamw_fused", "adamw_8bit", "muon"]
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class LossConfig(Strict):
    name: Literal["mnrl", "cached_mnrl"]
    temperature: float = Field(gt=0)
    # GradCache mini-batch. Only meaningful for cached_mnrl.
    mini_batch: int | None = Field(default=None, gt=0)


class PeftConfig(Strict):
    mode: Literal["full", "lora", "qlora"]
    r: int = Field(default=0, ge=0)
    alpha: int = Field(default=0, ge=0)
    dropout: float = Field(default=0.0, ge=0, le=1)


class FreezeConfig(Strict):
    vision_tower: bool = False
    # Per-layer embeddings, gemma4 only.
    ple: bool = False


class DataloaderConfig(Strict):
    backend: Literal["torch", "dali"]
    packing: bool = False
    # Tokenise ahead of the timed window instead of inside the dataloader.
    pretokenize: bool = False


class ParallelConfig(Strict):
    strategy: Literal["single", "ddp", "fsdp2", "zero2", "zero3"]
    # All-gather embeddings across ranks so in-batch negatives span the whole
    # world. This is the only parallelism axis specific to contrastive training,
    # and the project's central claim is that batch size dominates here.
    cross_device_negatives: bool = False


class FrameworkConfig(Strict):
    name: Literal["native", "unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"]


class TrainConfig(Strict):
    batch_size: int = Field(gt=0)
    grad_accum: int = Field(default=1, gt=0)
    steps: int = Field(gt=0)
    warmup_discard_steps: int = Field(ge=0)
    gradient_checkpointing: Literal["none", "full", "selective"] = "none"
    offload: Literal["none", "optimizer", "param", "both"] = "none"
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
        """torch.profiler inflates iteration time by 20-44%, and deterministic mode
        disables the kernel autotuning under measurement. Neither may be on for a run
        whose numbers get reported."""
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
        InfoNCE compare a row against itself and quietly destroys the loss."""
        if self.data.limit is not None and self.train.batch_size > self.data.limit:
            raise ValueError(
                f"train.batch_size ({self.train.batch_size}) exceeds data.limit "
                f"({self.data.limit}); rows would repeat inside a batch and in-batch "
                "negatives would include the positive itself."
            )
        return self

    @model_validator(mode="after")
    def _measured_runs_pin_their_data(self) -> BenchConfig:
        """A number is not reproducible if the data it came from is a moving branch
        (convention 07 requires the data version on every run)."""
        if self.run.purpose in ("timing", "quality") and self.data.revision is None:
            raise ValueError(
                f"purpose={self.run.purpose} requires data.revision to be pinned; "
                "'null' tracks a branch and makes the run unreproducible."
            )
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
    def _lora_needs_rank(self) -> BenchConfig:
        if self.peft.mode in ("lora", "qlora") and self.peft.r <= 0:
            raise ValueError(f"peft.mode={self.peft.mode} requires peft.r > 0")
        return self
