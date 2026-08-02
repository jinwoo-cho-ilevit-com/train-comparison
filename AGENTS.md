# AGENTS.md

Instructions for coding agents working in this repository.

## Project

Benchmark of embedding-model training speed across Qwen3-VL-Embedding-2B, Qwen3.5-0.8B,
and gemma-4-E2B — full finetuning vs LoRA, measured over an ablation of optimization
techniques on RunPod GPUs. Research design lives in `PLAN.md`.

## Commands

Secrets come from Infisical. Wrap every command:

- Setup: `uv sync --extra compose --extra native` (`compose` carries Hydra, without
  which pytest fails at collection; `native` carries transformers, datasets, peft
  and pytorch-optimizer, which tests import inside the functions that need them —
  so a `compose`-only checkout collects the whole suite and then fails at runtime
  in exactly the tests that exercise a real model, a real adapter or a real
  optimizer. `doc-commands` is the check that keeps this command honest.)
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
probe) and `scripts/bench.py` (measurement). Both exist. Neither has run on a GPU
from this repository — every number in `docs/` is either CPU or unmeasured, and
"the code path exists" is not "the measurement happened".

## Launching pods — the token's scope is checked, not its contents

`scripts/orchestrate.py` refuses to start any pod unless the Infisical token it
would hand over reaches **exactly `ALLOWED_ON_POD` (`HF_TOKEN`)**. The check asks
the token what it can see and compares names — never values — because
`docker/entrypoint.sh` runs the workload under `infisical run`, which injects
everything that token can read. Handing the pod a short-lived token does nothing
if the token's scope is the whole project.

**It is an allowlist, and that is a measurement rather than a preference.** The
`dev` environment injects 27 names (measured 2026-08-02 by diffing `os.environ`
under `infisical run --env=dev` against a plain shell). A deny list of the four
account-wide credentials passed the other 22 — cloud, database and
model-provider keys nobody had enumerated. Adding names to `FORBIDDEN_ON_POD`
cannot fix this: a deny list holds what someone remembered, and the environment
it guards grows without asking it.

Two ways to satisfy the check:

- keep the pod's secrets in their own Infisical environment — this is the setup,
  and `--infisical-env` already defaults to `pod`, so nothing has to be passed
- or scope the machine identity itself to `HF_TOKEN`

`--infisical-env` names **the environment the pod reads**, never the one the
orchestrator reads. The orchestrator's own secrets come from the `infisical run
--env=dev` wrapping the script, which is why the pod's environment is the
default. It is also what the pre-launch check asks about, so the two cannot drift
apart. Pass it only to point a pod somewhere other than `pod`.

**The pod's token is minted, not inherited.** `infisical_token()` mints one from
the universal-auth identity and ignores the caller's own `INFISICAL_TOKEN`. That
matters because the documented way to run the orchestrator is itself
`infisical run --env=dev -- python scripts/orchestrate.py`, which puts a
dev-stored service token in the environment — and that token is bound to `dev`
and ignores `--env` entirely (measured: same 26 names for `dev`, for `pod`, and
for an environment that does not exist). Inheriting it handed every pod a
dev-wide token whichever environment was selected. To supply one deliberately,
use `TRAINBENCH_POD_INFISICAL_TOKEN`; nothing puts a token there by accident.

The check measures that property rather than assuming it: before reading the
scope it asks the token for an environment that cannot exist, and a token that
answers is bound to one environment and is refused. Without that, a dev-bound
token would start passing the moment `dev` was tidied up.

Current state (2026-08-02, real Infisical, counts only):

| 토큰 | `pod` (기본값) | `dev` |
|---|---|---|
| 머신 아이덴티티 (기본) | 통과, 스코프 `HF_TOKEN` | 거부, 초과 25개 |
| `dev` 저장 서비스 토큰 | 거부, `--env` 무시 | 거부, `--env` 무시 |

So the default path launches, and pointing a pod at `dev` is refused.

The check refuses in both directions. A token that **cannot** read `HF_TOKEN` is
also refused: without it every gated checkpoint answers 401 and the combination
gets filed as unsupported by a pod that was never equipped to load it — a spent
pod-hour producing a wrong result rather than no result.

`FORBIDDEN_ON_POD` still exists for two narrower jobs: `pod_env` refuses to build
an env dict containing one, and the scope refusal names them separately so an
operator knows which extras are urgent.

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

**Read the pinned source before asserting framework behaviour.** Every probe
failure the first Phase 0 campaign found (2026-08-02, 18 A100 pods) had its
answer sitting in an already-locked wheel nobody had opened. axolotl's own
order is `prepare_plugins -> validate_config -> normalize_config`
(`axolotl/cli/config.py`), while `trainbench/probe/axolotl.py` skipped
validation and called `normalize_config` first — its docstring claimed to
follow the project's own docs — and produced `TypeError: unsupported operand
type(s) for //: 'NoneType' and 'NoneType'` on all three models.
`FastVisionModel.from_pretrained` defaults `full_finetuning=False`
(`unsloth/models/loader.py`) and the chain ends in `requires_grad_(False)` on
every non-LoRA parameter; three cells backpropagated through a fully frozen
graph — `params_with_grad=0`, `trainable_params=0` — and `infonce_backward`
passed anyway, because `enable_input_require_grads()` keeps the graph
differentiable through the embedding output. `google/gemma-4-E2B` is a base
checkpoint with no `chat_template.jinja` — only the `-it` variant has one —
and three frameworks failed identically on `apply_chat_template`, a fact one
Hub file listing would have shown. Probes written from what usually works do
not survive contact with pinned versions; the fixes researched this way
landed on the first pod run, ms_swift went 0/3 to 3/3 and native's gemma-4
cell opened.

**Do not relay a number you did not produce.** A lane reported the audit at
`12/15` while the tree was actually at `11/15` (`plan-files` was red on two
undeclared files), and that number went straight into a status report
unchecked. The same shape produced a quoted test count read off a tree that
held another lane's uncommitted work. Re-run the gate yourself before quoting
it — if a claim cannot be checked in the session that makes it, write
"확인 안 함" instead of the number, the same rule this file already applies to
"측정 안 함" for measurements.
