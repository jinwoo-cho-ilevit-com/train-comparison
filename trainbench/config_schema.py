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


class DataConfig(Strict):
    repo_id: str
    revision: str | None = None
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


class ParallelConfig(Strict):
    strategy: Literal["single", "ddp", "fsdp2", "zero2", "zero3"]


class FrameworkConfig(Strict):
    name: Literal["native", "unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"]


class TrainConfig(Strict):
    batch_size: int = Field(gt=0)
    grad_accum: int = Field(default=1, gt=0)
    steps: int = Field(gt=0)
    warmup_discard_steps: int = Field(ge=0)
    gradient_checkpointing: Literal["none", "full", "selective"] = "none"
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
    def _lora_needs_rank(self) -> BenchConfig:
        if self.peft.mode in ("lora", "qlora") and self.peft.r <= 0:
            raise ValueError(f"peft.mode={self.peft.mode} requires peft.r > 0")
        return self
