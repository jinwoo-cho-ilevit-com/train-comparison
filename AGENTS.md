# AGENTS.md

Instructions for coding agents working in this repository.

## Project

Benchmark of embedding-model training speed across Qwen3-VL-Embedding-2B, Qwen3.5-0.8B,
and gemma-4-E2B — full finetuning vs LoRA, measured over an ablation of optimization
techniques on RunPod GPUs. Research design lives in `PLAN.md`.

## Commands

Secrets come from Infisical. Wrap every command:

- Test: `infisical run --env=dev -- uv run pytest`
- lint/format: `uv run ruff check && uv run ruff format --check`
- Small-sample smoke: `infisical run --env=dev -- uv run python scripts/env_report.py device=cpu model=qwen3_5_0_8b framework=native data.limit=4`

`scripts/env_report.py` walks the whole harness path (compose -> validate -> device ->
seed -> atomic write) without loading a model. Model x framework probing
(`scripts/verify_env.py`) and the measurement entry point (`scripts/bench.py`) are
not written yet.

## Conventions

Full conventions: `~/Codes/develop-convention` — follow the "Core Rules" of the doc
matching the type of work (doc map is that repo's README.md).

Especially keep in this project:

- No hardcoding, all values in central config (02). This repo is an ablation study:
  new experiment variants come from Hydra config composition, never code changes.
  Adding an axis by editing code is a convention violation.
- Every pipeline stage supports small-sample runs via `data.limit` + intermediate
  save/resume (04). `data.limit` is a config field, not an argparse flag.
- Scan for duplicates/dead code before completion, no `_v2`/`_new` naming (01)
- Unified seeding helper + device via `trainbench/device.py` only — inline `.cuda()`
  or `"cuda:0"` strings are forbidden, they break CPU fallback (03/07)

### Measurement rules specific to this repo

These exist because the deliverable is a speed benchmark; violating them silently
corrupts results.

- **Timing runs and profiling runs are separate.** `torch.profiler` inflates
  iteration time by 20-44%. Never report a number measured with the profiler on.
- **Deterministic mode is off during measurement.** It disables kernel autotuning,
  which is part of what we measure. `set_seed(deterministic=...)` supports both;
  tests and CPU smoke keep it on. The on/off cost is measured once and recorded in
  `docs/methodology.md` — this is the evidence convention 07 requires for turning
  it off.
- **Never read training data from a network volume.** Everything must be on
  pod-local NVMe during measurement, or the dataloader axis measures the volume
  instead of the pipeline.
- **Same axis, same pod.** Settings within one ablation axis are never split across
  pods — host CPU/memory-bandwidth differences would show up as throughput
  differences. Every pod runs the canonical baseline; >3% deviation invalidates it.
- Record the resolved torch/framework versions per run. Framework images bring
  their own stacks, so version is a confound that must be visible in results.

## Verification

Before completion: run `uv run pytest` + the smoke command above, check full output.
No completion claims without execution evidence. TODOs/stubs/`test.skip` are
blockers, not completion. Never fabricate a measured number — write "측정 안 함".
