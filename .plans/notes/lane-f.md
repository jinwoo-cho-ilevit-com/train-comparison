# Lane F — framework-owned step path

Base `ff14c08` — Lane P (precision removal) and Lane X (exit-code fidelity) both
merged. `git merge main` into this worktree was a clean fast-forward, twice
(`88007ee` -> `a4b312a` -> `ff14c08`), with no conflict either time. Baselines this
lane is measured against are main's post-P-post-X numbers: `pytest -q` 1202,
`pytest tests/contract -q` 117, `audit_plan.py` 13/15 exit 0.

**Lane X is complementary to this lane's failure record, not a collision.** It found
that `infisical run` flattens any non-zero child exit to a flat `1`, and made
`run_with_secrets` persist the real code to `${RESULT_DIR}/.last-exit` so the pod
log keeps it. Without that fix `FAILED_EXIT = 6` would have reached the log as `1`
— the same value it is meant to be told apart from. Neither lane touched the
other's file: Lane X changed `docker/entrypoint.sh` and `scripts/verify_env.py`,
this lane changed `scripts/bench.py`. Enumerated on the merged tree, the exit codes
in play are 0, 2, 3, 4, 5, 125 (`BUDGET_EXHAUSTED`), 127, `timeout`'s 124 and
Lane X's 128+signal; 6 collides with none.

## What this lane changed

`scripts/bench.py` now drives the `owner=framework` half of `Step`, which
`trainbench/loader.py` declared and nothing read. Two registries, keyed by
`Built.framework`/`AdapterOut.framework` — the adapter's own literal, never
`config.framework.name`, per CONTRACTS.md §2:

- `FRAMEWORK_OWNED_STEP_RUNNERS = {"tevatron": _tevatron_forward_loss}`, used only
  when the binding's `step.owner == "framework"`. Splits the padded batch in half
  (queries first, `trainbench/collate.py`'s invariant), calls
  `model(query=.., passage=..)`, returns `output.loss`.
- `NONSTANDARD_FORWARD_POOLERS = {"sentence_transformers": _sentence_transformers_pooled}`,
  used regardless of `step.owner` — ST's step stays harness-owned because it does
  not own `loss.name`. Calls `model(tensors)["sentence_embedding"]` (one positional
  mapping) and `train()` computes the loss from `built.loss_fn` inline.

The two call conventions are kept apart deliberately. Collapsing them into one
`StepRunner` would hide that ST's loss is still the harness's: written literally as
`loss = built.loss_fn(...)`, it stays visible to `audit_plan.py`'s `assert-called`,
which is a check about this file's text.

`train()` gained `adapter_step: Any = None`; `main()` passes `binding.step`. The two
refusals gained a `framework: str` and return early for a registered framework;
`build_run` passes `binding.framework`.

**No `Step`/`AdapterOut` field was added.** `PAYLOAD_KEYS` and `ADAPTER_OUT_FIELDS`
are untouched, `trainbench/loader.py` is unmodified, contract holds at 117. No
boundary request was needed.

## Recorded failures

`except BaseException` around the measured loop re-raised anything `metrics.is_oom`
returned False for — every ordinary `RuntimeError` — so a diagnosable failure wrote
no artifact and `docker/entrypoint.sh`'s fallback filed the pod as "produced no
result (exit 1)". A pod died that way on `trainbench/collate.py`'s
image-token-expansion refusal, and the message was in the container log all along.

- `failure_status(exc)` mirrors `metrics.oom_status` and refuses in the opposite
  direction: an OOM filed here would publish the device's ceiling as a defect, so
  neither outcome can be mistaken for the other.
- `FAILED_EXIT = 6` collides with none of 3 (`REFUSED_EXIT`), 4 (`PREFLIGHT_EXIT`),
  5 (`OOM_EXIT`), `timeout`'s 124, or entrypoint's 125/127.
- `FAILED_STATUS = "run-failed"`, kept clear of `no_result`, which
  `publish_result.fallback_record` owns and which means no file existed at all, and
  clear of `REFUSED_STATUS`, which means the setting was declined before measuring.
- **No `metrics` block**, the rule the OOM and refusal records already keep: a
  `metrics` block asserts a measured window completed.
- The traceback is truncated with `MAX_TRACEBACK_CHARS` (`trainbench/probe/types.py`),
  the constant the probe reports already use, kept from the end because the
  innermost frame and the message name the defect.
- `KeyboardInterrupt` and `SystemExit` are re-raised before any record is built. A
  pod stopped from outside is not a property of the combination.

**`scripts/report.py` and `docker/entrypoint.sh` need no change, verified by
reading.** `report.py` selects `barren = [a for a in rest if not a.metrics]` and
prints that record's `status` verbatim under `지표 없음`, which is where a failed run
belongs. `entrypoint.sh` chooses `--mode result` over `--mode fallback` on
`[[ -s "${setting_out}" ]]` — whether a non-empty result file exists, never on the
exit code — so the file existing is the whole fix, and `note="exit ${status}"`
records 6 with no list of codes to extend.

## Pinned sources read (not assumed)

- tevatron `dd06310` (`~/.cache/uv/git-v0/checkouts/af8e1386372d71f4/dd06310/src/tevatron`),
  confirmed against `envs/tevatron/uv.lock`. `retriever/modeling/encoder.py:52-87`
  — `EncoderModel.forward(self, query: Dict = None, passage: Dict = None)`;
  `if query:`/`if passage:` are dict truthiness, so an empty dict takes the eval
  branch and returns `loss=None`; `if self.training:` gates the training branch.
  `dense.py:19` — `encode_query` does `self.encoder(**qry, return_dict=True)` and
  reads `qry['attention_mask']`. `scripts/bench.py` calls `built.model.train()`
  before the loop, so the training branch is reached.
- The temperature and pooling confounds are closed, and not by this lane:
  `trainbench/probe/tevatron.py:92-93` passes `pooling="last", normalize=True` and
  `apply_temperature` fills the temperature `DenseModel.load` has no keyword for.
  The timing path goes through the same `load_dense_model`, so no tevatron run is
  measured at the class defaults (`pooling='cls'`, `temperature=1.0`).
- sentence-transformers 5.6.1 is **not installed on this host**. The signature used
  here (`forward(self, input: dict, **kwargs) -> dict`) comes from the citation
  already committed in `trainbench/loader.py`'s ST `Adapter` and from the working
  probe (`trainbench/probe/sentence_transformers.py:120-126`). **확인 안 함**
  against the installed package — see "Pod-only questions".

## The two boundary conflicts, as the integrator decided them

### (a) `tests/test_smoke_cpu.py` — rewritten, not deleted

`test_a_framework_owned_step_is_refused_instead_of_measured_by_this_loop` asserted
that an `owner=framework` step is *always* refused, which is the premise this lane
ends. The invariant underneath survives, so the scenario points at a framework
genuinely absent from `FRAMEWORK_OWNED_STEP_RUNNERS` (`ms_swift`) and the test is
renamed to state the narrowed rule:
`test_a_framework_owned_step_with_no_registered_runner_is_refused_not_measured`.
All four original assertions are kept verbatim, including `"metrics" not in
record` — no number at all is the point. It asserts `ms_swift` is absent from the
registry first, so registering it later fails this test instead of quietly
emptying it.

A second test in that file had the same shape of expired premise and was not
predicted by the review: `test_a_failure_inside_the_measured_loop_is_not_recorded_as_a_refusal`
pinned two things, and only one expired. "A mid-loop failure is not filed as a
refusal" is still true and still matters; "and `main` writes nothing" is what (c)
reverses. It now asserts the surviving half directly — `FAILED_EXIT` rather than
`REFUSED_EXIT`, a `run-failed` status, no `refusal` key, the exception type on the
record, no `metrics` — which is stronger than the absence it used to check, because
the two outcomes are now told apart by what the record says.

### (b) `scripts/audit_plan.py` — the exemption is derived, not registry-keyed

`assert-called`'s premise was that every `loss = ...` in `scripts/bench.py` must
read `built.loss_fn` syntactically. That held only because
`refuse_a_step_this_harness_cannot_drive` refused every `owner=framework` step, so
no framework-owned loss could execute in this file.

The exemption requires **two facts to agree**: the entry point's
`FRAMEWORK_OWNED_STEP_RUNNERS` names the framework, *and* that framework's
`Adapter` in `trainbench/loader.py` declares `loss.name` among its `owned_axes`
(`_frameworks_declaring_loss_ownership`, read off the `Adapter(...)` calls rather
than a list here, for `applied.py`'s fail-open reason). Registry membership is an
edit; the declaration is what `applied._owned` acts on when it exempts the axis
from the capture. A framework registered without it leaves the binding flagged.

Kept out of `derived`: an exempt binding suppresses `stray` but is not evidence
that `built.loss_fn` was consumed, so the "binds no loss from built.loss_fn" half
is still earned separately.

Renaming the local `loss` to dodge the AST walk was considered and rejected: the
value is genuinely not certified against `built.loss_fn`, so a rename would change
what the checker sees without changing what is true — the "check passes, nothing
was looked at" pattern `HAZARDS.md` §3 already catalogues.

## Mutation evidence

Each mutation was applied, observed to fail with the quoted output, then restored;
all four files verified byte-identical by sha256 after every restore.

| Mutation | Observed failure |
|---|---|
| `adapter_step.owner != HARNESS.owner` -> `==` | `TypeError: TevatronLike.forward() got an unexpected keyword argument 'input_ids'` — 3 failed, 6 passed |
| `_tevatron_forward_loss`'s `half = shape[0] // 2` -> `shape[0]` | `IndexError: Target 0 is out of bounds.` |
| ST pooler `model(tensors)` -> `model(**tensors)` | `TypeError: SentenceTransformerLike.forward() missing 1 required positional argument: 'input'` |
| `assert-called`: dropped the `owned_axes` conjunct, leaving registry membership alone | `AssertionError: assert not True` in the new audit test — the shallow exemption waves through a framework that declared nothing |
| `failure_status(exc), FAILED_EXIT` -> `raise` | `RuntimeError: a batch of 4 image(s) expanded to 9001 tokens...` escapes `main()`; no record written |
| removed the `KeyboardInterrupt`/`SystemExit` re-raise | a record was written and stderr read `run-failed (KeyboardInterrupt) — ` |
| `FAILED_STATUS = REFUSED_STATUS` | stderr read `axis-refused (UnappliedAxis) — raised after the loop had started`; the mid-loop test caught it |
| `ms_swift` added to `FRAMEWORK_OWNED_STEP_RUNNERS` | the (a) test's own guard fired: `AssertionError: assert 'ms_swift' not in {...}` |

## Known findings this lane did not fix

Raised by the review pass; each is a decision rather than a defect in what was
asked for, and none is in this lane's granted files except where noted.

1. `refuse_a_forward_this_harness_cannot_call` exempts by framework **name** and
   never looks at `step`. A tevatron binding declaring `owner=harness` therefore
   passes the pre-flight and reaches the generic `model(**batch)` — the
   `TypeError`-on-step-0-with-the-timer-open failure that function exists to
   prevent. Reproduced. The fix is to compute `framework_owns_step` once and have
   all three sites read it.
2. Nothing asserts inside `scripts/bench.py` that every key of
   `FRAMEWORK_OWNED_STEP_RUNNERS` names an adapter declaring `loss.name`.
   `audit_plan.py` now checks it for the audit's own exemption, which is a
   different question from the run being refused. `tests/contract/` is the durable
   place.
3. `_tevatron_forward_loss` slices **every** key of `tensors` at a row-derived
   index. `trainbench/axes.py:2123` states the opposite rule for the same split: an
   `IMAGE_PAYLOAD_KEYS` entry "is never fitted to the row rule even if its leading
   dimension happens to equal the row count". `pixel_values` and `image_grid_thw`
   count patches and images, not rows. Unreachable while `collate.py`'s expansion
   refusal fires first, and **live the moment that is fixed**.
   `MicroBatch.images_per_row` and `axes._split_rows` already hold the right split.
4. `train()` rebinds its `loader` parameter (the dataloader) to the
   `trainbench.loader` module. Harmless today — `stream = micro_batches(loader)`
   runs above it and nothing reads `loader` after it in that function — and a trap
   for the next edit.
5. `_tevatron_forward_loss` takes `config` and `built` and uses neither.

## Pod-only questions — 확인 안 함

- Whether sentence-transformers 5.6.1's `SentenceTransformer.forward` signature in
  the built image still matches the citation this lane relied on. The code
  docstrings state it flatly; only a pod can confirm it.
- Which pooling a real ST checkpoint performs. `train()` requires
  `config.model.pooling == "lasttoken"`, but the ST branch returns
  `sentence_embedding` from ST's own Pooling module, whose mode comes from the
  checkpoint's `1_Pooling/config.json`. `model.pooling` is not an axis, so nothing
  in the record is mislabelled, but cross-framework comparability depends on it.
- Whether `DenseModel.forward(query=, passage=)` receives the exact dtypes and
  shapes this collate produces. This lane's tests use stubs, not a checkpoint.
- Whether tevatron's `is_ddp` gather path behaves under single-GPU pods — out of
  scope, `parallel.*` sweeps are excluded from this campaign.
- Whether GradCache would ever need its own tevatron/ST integration. This lane
  refuses the combination outright rather than guessing.

## Gate output (this worktree, this session)

```
uv run ruff check && uv run ruff format --check
  [] / 116 files already formatted

infisical run --env=dev -- uv run pytest -q
  1214 passed, 14 warnings in 120.78s        (main 1202 + 12 new tests, 0 failed)

infisical run --env=dev -- uv run pytest tests/contract -q
  117 passed                                 (main 117, unmoved)

infisical run --env=dev -- uv run python scripts/audit_plan.py
  13/15 passing, 0 new failure(s), 0 newly fixed, 0 grew, 0 shrank, 0 unreadable
  exit 0                                     (main 13/15 exit 0)

infisical run --env=dev -- uv run python scripts/env_report.py device=cpu \
  model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4
  wrote outputs/qwen3_5_0_8b-native-.../env_report.json
```

The 12 new tests: 9 in `tests/test_bench_framework_step.py`, 2 in
`tests/test_smoke_cpu.py` (a recorded failure, an interruption that records
nothing), 1 in `tests/test_audit.py` (the derived exemption). Two tests were
rewritten rather than added, so they move no count.

## Sequencing note for the qlora-removal lane

`tests/test_smoke_cpu.py` is the one file that lane and this one share. The three
tests Lane P re-pointed at `peft.mode=qlora` —
`test_one_unapplicable_setting_refuses_the_whole_plan`,
`test_a_probe_that_declines_an_axis_is_still_started`,
`test_a_pod_wrong_in_both_ways_is_told_both` — go through `bench_entry.preflight`
with `plan_item`/`plan_file`/`pod_gpu`, and share no helper with anything this lane
touched, which uses `pod_setting`/`adapter_binding`/`timing_config`/`text_only_rows`
through `main()`. The only common names are `config_mapping` and `monkeypatch`, and
neither is re-pointed by either lane. The edits are also in different regions of the
file. This lane did not re-point them.
