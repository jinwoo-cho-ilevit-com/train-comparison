"""Measure one setting: one model, one axis configuration, one number.

Runs inside every framework image, so it takes a resolved config JSON rather than
composing with Hydra (Hydra's antlr4 pin is incompatible with axolotl), the same
interface `scripts/verify_env.py` uses.

    python scripts/bench.py --config resolved.json --out result.json

**One setting per process.** A sweep is the pod re-running this file once per
entry in its plan (docker/entrypoint.sh), not a loop in here. A process that has
already run a setting carries that setting's autotune cache, compiled graphs and
allocator fragmentation into the next one, and `kernel`/`attn` cannot be changed
after the model exists at all — `axes.patch` runs before construction. Reusing the
process would trade the thing being measured for the time it takes to load a model.

The five calls to `trainbench/axes.py` and `trainbench/applied.py` are what make a
number reportable, and `audit_plan.py`'s `assert-called` requires this file to make
all five. `assert_matches` is called here directly rather than through
`trainbench/probe/steps.py::verify_axes`, which wraps it in `report.run(...)` and
therefore *swallows* the raise — a harness built on that would satisfy the audit
while a mismatched axis went on to produce a number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from trainbench import axes, metrics
from trainbench.applied import assert_matches, capture
from trainbench.config import load_bench_config
from trainbench.config_schema import BenchConfig
from trainbench.device import get_device
from trainbench.embedding import align_padding_side, info_nce
from trainbench.probe import steps
from trainbench.record import build_record, write_json
from trainbench.seed import set_seed


class PairDataset(torch.utils.data.Dataset):
    """Query/positive text pairs from the pinned subset, as raw strings.

    Tokenisation happens in the collate function so that `dataloader.pretokenize`
    stays a real axis: pre-tokenising would move that work out of the timed window
    permanently, which is the very difference the axis is supposed to measure.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, str]:
        row = self.rows[index]
        return row["qry"] or "", row["pos_text"] or ""


def load_pairs(config: BenchConfig) -> PairDataset:
    """Rows from the pinned subset revision.

    Reads `config.data.repo_id` / `config.data.revision` / `config.data.subset_rows`
    through the config object. The pin is the point: a run that streamed a branch
    would report a number measured on a corpus nobody can name afterwards, which is
    what `config_schema.py`'s revision validators exist to prevent.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        config.data.repo_id,
        revision=config.data.revision,
        split="train",
        streaming=True,
    )
    wanted = config.data.effective_rows
    rows = list(dataset.take(wanted))
    if len(rows) < wanted:
        raise RuntimeError(
            f"{config.data.repo_id}@{config.data.revision} yielded {len(rows)} rows, "
            f"asked for {wanted}; a short corpus makes every throughput figure optimistic"
        )
    return PairDataset(rows)


def collate(processor: Any, device: torch.device, config: BenchConfig):
    """Queries and positives in one batch, queries first.

    `infonce_backward` splits the pooled embeddings down the middle, so the halves
    have to line up. `model.instruction_prompt` is prepended to the query side only
    — it is the official prompt for the embedding model and belongs on the side it
    was written for (docs/model-spec.md).
    """
    prompt = config.model.instruction_prompt or ""

    def build(batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        queries = [prompt + q for q, _ in batch]
        positives = [p for _, p in batch]
        tokenizer = getattr(processor, "tokenizer", processor)
        encoded = tokenizer(
            queries + positives,
            padding=True,
            truncation=True,
            max_length=config.data.max_seq_len,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in encoded.items()}

    return build


def train(built: Any, loader: Any, config: BenchConfig, device: torch.device) -> dict[str, Any]:
    """The measured loop.

    Every step goes inside `axes.step_context(config)`. The fp8 recipes wrap the
    forward pass, so a loop that applied them anywhere else would report an fp8
    number for a step that never entered the recipe — and the capture probe would
    still find the swapped modules and call it a match.

    Warmup steps are timed and then discarded by `metrics.summarise`, which records
    how many. The peak memory counter is reset after warmup so the figure belongs
    to the measured window rather than to model construction and autotuning.
    """
    if config.model.pooling != "lasttoken":
        raise RuntimeError(
            f"model.pooling={config.model.pooling} is declared but "
            "trainbench/probe/steps.py::encode pools the last token unconditionally; "
            "the knob would name one thing while the run measured another."
        )
    built.model.train()
    timer = metrics.StepTimer(device)
    total = config.train.steps
    discard = config.train.warmup_discard_steps
    side = config.model.padding_side
    losses: list[float] = []
    tokens = 0
    rows = 0
    step = 0

    while step < total:
        for batch in loader:
            if step == discard:
                metrics.reset_peak_memory(device)
            with timer, axes.step_context(config):
                for _ in range(config.train.grad_accum):
                    # Not `steps.infonce_backward`: it ends with
                    # `model.zero_grad(set_to_none=True)`, which is right for a probe
                    # (one step, nothing after it) and wrong here — the optimizer
                    # would step on gradients that had just been wiped, and the loop
                    # would time a forward and a backward while calling it training.
                    # Caught by tests/test_smoke_cpu.py, which asserts the weights
                    # move. `encode` and `info_nce` are reused unchanged.
                    pooled = steps.encode(built.model, batch, side)
                    half = pooled.shape[0] // 2
                    loss = info_nce(pooled[:half], pooled[half:], config.loss.temperature)
                    loss.backward()
                    losses.append(float(loss.detach()))
                built.optimizer.step()
                built.optimizer.zero_grad(set_to_none=True)
            # Counted from the batch that was actually fed, not from the config: a
            # short final batch would otherwise inflate every per-second figure.
            if step >= discard:
                tokens += int(batch["attention_mask"].sum()) * config.train.grad_accum
                rows += int(batch["input_ids"].shape[0]) * config.train.grad_accum
            step += 1
            if step >= total:
                break

    measured = max(1, total - discard)
    summary = metrics.summarise(
        timer.durations,
        discard=discard,
        rows_per_step=rows // measured,
        tokens_per_step=tokens // measured,
        peak_bytes=metrics.peak_memory_bytes(device),
    )
    summary["loss_first"] = losses[0] if losses else None
    summary["loss_last"] = losses[-1] if losses else None
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="resolved config JSON")
    parser.add_argument("--out", required=True, type=Path, help="where to write the result")
    args = parser.parse_args(argv)

    config = load_bench_config(args.config)
    device = get_device(config.device)
    # Follows the config rather than forcing determinism the way the probe does.
    # Deterministic mode disables kernel autotuning, which is part of what a timing
    # run measures, and the schema already refuses `deterministic=true` for
    # `purpose=timing` — hardcoding it here would override that silently.
    set_seed(config.train.seed, deterministic=config.train.deterministic, warn_only=True)

    from transformers import AutoModel, AutoProcessor

    axes.patch(config)
    processor = AutoProcessor.from_pretrained(config.model.hf_id, revision=config.model.revision)
    align_padding_side(processor, config.model.padding_side)
    model = AutoModel.from_pretrained(
        config.model.hf_id,
        revision=config.model.revision,
        dtype=steps.dtype_for(device),
        **axes.load_kwargs(config),
    )
    model.to(device)

    dataset = load_pairs(config)
    built, applied = axes.assemble(model, config, device, framework="native", dataset=dataset)
    built.dataloader.collate_fn = collate(processor, device, config)

    state = capture(built, config)
    # Directly, and before a single step runs. Not inside a try, and not through
    # `steps.verify_axes`: the whole value of this call is that it stops the run.
    assert_matches(state, config)

    # Timing and profiling are separate runs (AGENTS.md). The schema already
    # refuses `run.profiler=true` for `purpose=timing`; this is where the other
    # purposes act on it, and the trace is written next to the result rather than
    # merged into it so no reported number can come from a profiled step.
    if config.run.profiler:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            if device.type == "cuda"
            else [torch.profiler.ProfilerActivity.CPU],
        ) as profile:
            summary = train(built, built.dataloader, config, device)
        trace = args.out.with_suffix(".trace.json")
        profile.export_chrome_trace(str(trace))
        summary["profiled"] = True
        summary["trace_path"] = str(trace)
        print(f"wrote {trace}")
    else:
        summary = train(built, built.dataloader, config, device)
        summary["profiled"] = False
    record = build_record(config, device, applied=state, metrics=summary, applied_axes=applied)
    write_json(args.out, record)

    print(f"{config.model.name} x {config.framework.name}: {summary['steps_measured']} steps")
    print(f"  step p50 {summary['step_seconds_p50']:.4f}s  p95 {summary['step_seconds_p95']:.4f}s")
    print(f"  rows/s {summary['rows_per_second']:.2f}  tokens/s {summary['tokens_per_second']}")
    print(f"  peak memory {summary['peak_memory_bytes']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
