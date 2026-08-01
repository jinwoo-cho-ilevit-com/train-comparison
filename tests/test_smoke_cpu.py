"""The measured loop, run end to end on CPU.

Not `main()`: that downloads a checkpoint. What is exercised here is the part that
produces numbers — `bench.train`, with a real optimizer step, a real backward, the
`step_context` wrapper, warmup discard and the metrics it feeds.

Only `purpose=probe` is reachable on CPU. `optim.name=adamw_fused` resolves to
`adamw_unfused` without a CUDA device, which is a permanent mismatch, so a CPU
timing run is blocked by design (docs/CONTRACTS.md §6) — that blocking is asserted
in tests/test_axes.py rather than worked around here.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

from trainbench.applied import Built

from .conftest import REPO_ROOT
from .test_applied import bench, config_mapping  # noqa: F401

_spec = importlib.util.spec_from_file_location("bench_entry", REPO_ROOT / "scripts" / "bench.py")
assert _spec and _spec.loader
bench_entry = importlib.util.module_from_spec(_spec)
sys.modules["bench_entry"] = bench_entry
_spec.loader.exec_module(bench_entry)

CPU = torch.device("cpu")


class TinyEmbedder(torch.nn.Module):
    """Returns hidden states the way a transformers model does, so `steps.encode`
    reaches them by the same attribute it uses in production."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(64, width)
        self.proj = torch.nn.Linear(width, width)

    def forward(self, input_ids, attention_mask, **_):
        hidden = self.proj(self.embed(input_ids))
        return type("Output", (), {"last_hidden_state": hidden})()


def batches(rows: int, length: int = 6, count: int = 8):
    """`infonce_backward` splits the pooled embeddings down the middle, so each
    batch carries `rows` queries followed by `rows` positives."""
    for _ in range(count):
        yield {
            "input_ids": torch.randint(1, 60, (rows * 2, length)),
            "attention_mask": torch.ones(rows * 2, length, dtype=torch.long),
        }


@pytest.fixture
def probe_config(config_mapping):  # noqa: F811 - the imported fixture, requested once
    return bench(
        config_mapping,
        **{
            "run.purpose": "probe",
            "train.steps": 6,
            "train.warmup_discard_steps": 2,
            "train.grad_accum": 1,
            "train.batch_size": 2,
            "data.limit": 8,
        },
    )


def test_the_measured_loop_runs_and_reports_what_it_measured(probe_config):
    model = TinyEmbedder()
    built = Built(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn=None,
        dataloader=None,
        framework="native",
    )

    summary = bench_entry.train(built, list(batches(rows=2)), probe_config, CPU)

    assert summary["steps_timed"] == 6
    assert summary["steps_discarded"] == 2
    assert summary["steps_measured"] == 4
    assert summary["rows_per_second"] > 0
    assert summary["tokens_per_second"] > 0
    # No CUDA, so there is no peak to report — and 0 would be a measurement.
    assert summary["peak_memory_bytes"] is None
    assert summary["loss_first"] is not None


def test_the_optimizer_actually_steps(probe_config):
    """Without this the loop measures a forward and a backward and calls it
    training, which is faster than the thing it claims to be timing."""
    model = TinyEmbedder()
    before = model.proj.weight.detach().clone()
    built = Built(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        loss_fn=None,
        dataloader=None,
        framework="native",
    )

    bench_entry.train(built, list(batches(rows=2)), probe_config, CPU)

    assert not torch.equal(before, model.proj.weight.detach())


def test_a_warmup_longer_than_the_run_never_reaches_the_loop(config_mapping):  # noqa: F811
    """It would report a throughput computed from no steps at all. The schema
    refuses it first, which is the better place — the run never boots."""
    with pytest.raises(Exception, match="warmup_discard_steps"):
        bench(
            config_mapping,
            **{"run.purpose": "probe", "train.steps": 2, "train.warmup_discard_steps": 4},
        )


def test_a_declared_pooling_the_loop_does_not_implement_is_refused(probe_config, monkeypatch):
    """`model.pooling` is a config knob and `steps.encode` pools the last token
    unconditionally. A run whose config named something else would measure one
    thing under the name of another."""
    monkeypatch.setattr(type(probe_config.model), "model_config", {}, raising=False)
    object.__setattr__(probe_config.model, "pooling", "mean")
    model = TinyEmbedder()
    built = Built(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        loss_fn=None,
        dataloader=None,
        framework="native",
    )

    with pytest.raises(RuntimeError, match="pools the last token"):
        bench_entry.train(built, list(batches(rows=2)), probe_config, CPU)


def test_the_collate_puts_queries_first_and_positives_second(probe_config):
    """`infonce_backward` splits pooled embeddings at the midpoint; a collate that
    interleaved them would pair every query with the wrong positive and still
    produce a loss that goes down."""

    class Tokenizer:
        def __call__(self, texts, **_):
            return type(
                "Encoded",
                (),
                {
                    "items": lambda self: iter(
                        [("input_ids", torch.tensor([[len(t)] for t in texts]))]
                    ),
                    "keys": lambda self: ["input_ids"],
                    "__getitem__": lambda self, k: torch.tensor([[len(t)] for t in texts]),
                },
            )()

    build = bench_entry.collate(type("P", (), {"tokenizer": Tokenizer()})(), CPU, probe_config)
    batch = build([("qq", "pppp"), ("q", "pp")])

    # Lengths identify the strings. The queries carry `model.instruction_prompt`
    # in front, which is the other half of what this asserts: the official prompt
    # goes on the query side only, never on the positive.
    prompt = len(probe_config.model.instruction_prompt or "")
    assert [int(v) for v in batch["input_ids"].flatten()] == [prompt + 2, prompt + 1, 4, 2]
