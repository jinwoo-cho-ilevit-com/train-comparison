# AGENTS.md

Instructions for coding agents working in this repository.

## Project

Benchmark of embedding-model training speed across Qwen3-VL-Embedding-2B, Qwen3.5-0.8B,
and gemma-4-E2B — full finetuning vs LoRA, measured over an ablation of optimization
techniques on RunPod GPUs. Research design lives in `PLAN.md`.

## Commands

Secrets come from Infisical. Wrap every command:

- Setup: `uv sync --extra compose` (the extra carries Hydra; without it pytest
  fails at collection on a missing hydra)
- Test: `infisical run --env=dev -- uv run pytest`
- lint/format: `uv run ruff check && uv run ruff format --check`
- Config-path check: `infisical run --env=dev -- uv run python scripts/env_report.py device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4`

`train.batch_size` moves with `data.limit`: a batch wider than the sample makes
InfoNCE compare a row against itself, and the schema refuses to start such a run.

**`scripts/env_report.py` is not a training smoke.** It loads no model, reads no
data, and runs no step. It exercises the config path only — Hydra composition ->
schema validation -> device resolution -> seeding -> atomic JSON write — and prints
the resolved environment. It catches a broken config group, a validator that rejects
a documented command, a device helper that cannot fall back to CPU, and a record
write that does not land. It cannot catch anything about a model, a kernel, or a
throughput number.

The entry points that do load models are `scripts/verify_env.py` (framework x model
probe) and `scripts/bench.py` (measurement) — and `bench.py` is not written yet, so
no end-to-end training path exists in this repository today.

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

- **Timing runs and profiling runs are separate.** Never report a number measured
  with the profiler on. How much the profiler inflates iteration time here is
  unmeasured and uncited — see `docs/methodology.md`. Do not repeat a percentage
  for it.
- **Deterministic mode is off during measurement.** It disables kernel autotuning,
  which is part of what we measure. `set_seed(deterministic=...)` supports both;
  tests keep it on. The on/off cost is the evidence convention 07 requires for
  turning it off, and it is likewise unmeasured — `docs/methodology.md` records
  both the gap and how it gets closed.
- **Never read training data from a network volume.** Everything must be on
  pod-local NVMe during measurement, or the dataloader axis measures the volume
  instead of the pipeline.
- **Same axis, same pod.** Settings within one ablation axis are never split across
  pods — host CPU/memory-bandwidth differences would show up as throughput
  differences. Every pod runs the canonical baseline; >3% deviation invalidates it.
- Record the resolved torch/framework versions per run. Framework images bring
  their own stacks, so version is a confound that must be visible in results.

## Verification

Before completion: run `uv run pytest`, the config-path check above, and
`uv run python scripts/audit_plan.py`; read the full output. No completion claims
without execution evidence. TODOs/stubs/`test.skip` are blockers, not completion.
Never fabricate a measured number — write "측정 안 함".

Green does not mean reproducible. Wave 0's gate reported 102 passed while
`configs/data/` was untracked, so it held only in the checkout that happened to
have those files on disk; a clean clone could not compose a config at all. When a
check passes, confirm it had something to examine.
