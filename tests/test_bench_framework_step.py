"""The framework-owned step path in `scripts/bench.py`.

`refuse_a_step_this_harness_cannot_drive` and `refuse_a_forward_this_harness_cannot_call`
used to refuse every `owner=framework` step and every nonstandard forward
unconditionally — correctly, because nothing drove either. `FRAMEWORK_OWNED_STEP_RUNNERS`
(tevatron) and `NONSTANDARD_FORWARD_POOLERS` (sentence_transformers) are what now
drives them, and these tests pin that both refusals step aside for a registered
framework while still refusing an unregistered one, and that `train()` actually
calls the model the way each convention requires.

Stubs, not the real packages: this host has neither tevatron nor
sentence-transformers installed (AGENTS.md). Each stub restates only the calling
convention and return shape `trainbench/loader.py`'s `Adapter` declares and the
pinned sources name (dd06310 retriever/modeling/encoder.py:52-87 for tevatron,
sentence-transformers 5.6.1 `SentenceTransformer.forward` for ST) — not the
framework's own algorithm, which is out of this harness's scope to re-verify.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from trainbench.applied import Built

from .conftest import REPO_ROOT
from .test_applied import bench, config_mapping  # noqa: F401

_spec = importlib.util.spec_from_file_location(
    "bench_entry_lane_f", REPO_ROOT / "scripts" / "bench.py"
)
assert _spec and _spec.loader
bench_entry = importlib.util.module_from_spec(_spec)
sys.modules["bench_entry_lane_f"] = bench_entry
_spec.loader.exec_module(bench_entry)

CPU = torch.device("cpu")


class TevatronLike(torch.nn.Module):
    """`DenseModel.forward(query=, passage=) -> EncoderOutput(loss=...)`.

    Only the calling convention and the `.loss` attribute the pinned source
    returns (dd06310 encoder.py:52-87) — not tevatron's own pooling/scoring.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(64, 4)

    def forward(self, query: dict | None = None, passage: dict | None = None) -> Any:
        if not query or not passage:
            return SimpleNamespace(loss=None)
        q = self.embed(query["input_ids"]).mean(dim=1)
        p = self.embed(passage["input_ids"]).mean(dim=1)
        scores = q @ p.T
        target = torch.arange(scores.shape[0])
        return SimpleNamespace(loss=torch.nn.functional.cross_entropy(scores, target))


class SentenceTransformerLike(torch.nn.Module):
    """The pinned `SentenceTransformer.forward` signature, and nothing else.

    sentence-transformers 5.6.1: `forward(self, input: dict, **kwargs) -> dict` —
    one positional mapping in, `{"sentence_embedding": ...}` out.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(64, 4)

    def forward(self, input: dict, **kwargs: Any) -> dict:  # noqa: A002
        return {"sentence_embedding": self.embed(input["input_ids"]).mean(dim=1)}


def micro_batch(samples: int, length: int = 6) -> Any:
    from trainbench import collate

    ids = torch.randint(1, 60, (samples * 2, length))
    return collate.MicroBatch(
        tensors={"input_ids": ids, "attention_mask": torch.ones_like(ids)},
        tokens=int(ids.numel()),
        padded_tokens=int(ids.numel()),
        rows=samples * 2,
        samples=samples,
        images=0,
        images_dropped=0,
    )


def batches(samples: int, count: int = 4) -> list[Any]:
    return [micro_batch(samples) for _ in range(count)]


@pytest.fixture
def probe_config(config_mapping):  # noqa: F811
    return bench(
        config_mapping,
        **{
            "run.purpose": "probe",
            "train.steps": 4,
            "train.warmup_discard_steps": 1,
            "train.grad_accum": 1,
            "train.batch_size": 2,
            "data.limit": 8,
        },
    )


def _unreachable_loss(*_args: Any, **_kwargs: Any) -> torch.Tensor:
    raise AssertionError("tevatron's runner must never touch built.loss_fn")


def tevatron_built(loss_fn: Any = _unreachable_loss) -> Built:
    model = TevatronLike()
    return Built(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn=loss_fn,
        dataloader=None,
        framework="tevatron",
    )


def st_built(config: Any) -> Built:
    from trainbench.embedding import info_nce

    model = SentenceTransformerLike()
    return Built(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn=lambda q, d: info_nce(q, d, config.loss.temperature),
        dataloader=None,
        framework="sentence_transformers",
    )


# --- train() actually drives the two conventions -----------------------------


def test_tevatron_owned_step_is_driven_and_measured(probe_config):
    loader = importlib.import_module("trainbench.loader")
    step = loader.Step(
        owner=loader.FRAMEWORK,
        callable="tevatron.retriever.modeling.DenseModel.forward",
        batch_keys=("query", "passage"),
    )

    summary = bench_entry.train(tevatron_built(), batches(2), probe_config, CPU, adapter_step=step)

    assert summary["steps_measured"] == 3
    assert summary["loss_first"] is not None


def test_sentence_transformers_forward_is_driven_and_measured(probe_config):
    loader = importlib.import_module("trainbench.loader")

    summary = bench_entry.train(
        st_built(probe_config), batches(2), probe_config, CPU, adapter_step=loader.HARNESS_STEP
    )

    assert summary["steps_measured"] == 3
    assert summary["loss_first"] is not None


def test_a_binding_naming_the_framework_without_declaring_the_step_still_runs_generic(
    probe_config,
):
    """The regression `test_the_axes_an_adapter_owns_reach_the_record_instead_of_being_undetermined`
    (tests/test_smoke_cpu.py) needs: `built.framework == "tevatron"` alone must not
    route through the tevatron runner unless the step actually declares
    `owner=framework` — otherwise a model that merely gets labelled tevatron would
    be called `query=`/`passage=` and crash.
    """
    from trainbench.embedding import info_nce

    class Generic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(64, 4)

        def forward(self, input_ids: torch.Tensor, attention_mask: Any = None, **_: Any) -> Any:
            hidden = self.embed(input_ids)
            return SimpleNamespace(last_hidden_state=hidden)

    model = Generic()
    built = Built(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn=lambda q, d: info_nce(q, d, probe_config.loss.temperature),
        dataloader=None,
        framework="tevatron",
    )

    summary = bench_entry.train(built, batches(2), probe_config, CPU, adapter_step=None)

    assert summary["steps_measured"] == 3


# --- the two refusal gates: registered frameworks pass through, others do not --


def test_a_step_owner_mismatch_without_a_registered_runner_is_refused():
    """A framework the registries do not name still gets the original refusal —
    the shape `tests/test_smoke_cpu.py`'s pre-Lane-F test pinned, kept alive here
    for a framework this harness genuinely cannot drive."""
    loader = importlib.import_module("trainbench.loader")
    step = loader.Step(owner=loader.FRAMEWORK, callable="made_up.Model.forward", batch_keys=("x",))

    with pytest.raises(loader.AdapterRefusal, match="no FRAMEWORK_OWNED_STEP_RUNNERS"):
        bench_entry.refuse_a_step_this_harness_cannot_drive(step, "made_up_framework")


def test_a_registered_framework_owned_step_is_not_refused():
    loader = importlib.import_module("trainbench.loader")
    step = loader.Step(
        owner=loader.FRAMEWORK,
        callable="tevatron.retriever.modeling.DenseModel.forward",
        batch_keys=("query", "passage"),
    )
    bench_entry.refuse_a_step_this_harness_cannot_drive(step, "tevatron")  # does not raise


def test_a_nonstandard_forward_without_a_registered_pooler_is_still_refused():
    """The control: the forward-binding refusal still fires for an unregistered
    framework whose model cannot be called as `model(**batch)`."""
    loader = importlib.import_module("trainbench.loader")
    with pytest.raises(loader.AdapterRefusal, match=r"cannot be called as model\(\*\*batch\)"):
        bench_entry.refuse_a_forward_this_harness_cannot_call(
            loader.HARNESS_STEP, SentenceTransformerLike(), "made_up_framework"
        )


def test_a_registered_nonstandard_forward_is_not_refused():
    loader = importlib.import_module("trainbench.loader")
    bench_entry.refuse_a_forward_this_harness_cannot_call(
        loader.HARNESS_STEP, SentenceTransformerLike(), "sentence_transformers"
    )  # does not raise


# --- incompatible combinations are refused, not silently mismeasured ---------


def test_gradcache_with_a_framework_owned_step_is_refused(config_mapping):  # noqa: F811
    loader = importlib.import_module("trainbench.loader")
    config = bench(
        config_mapping,
        **{"run.purpose": "probe", "loss.name": "cached_mnrl", "loss.mini_batch": 2},
    )

    def loss_fn(*_a: Any, **_k: Any) -> torch.Tensor:
        raise AssertionError("not called")

    loss_fn.gradcache_backward = lambda *a, **k: None  # noqa: ARG005
    built = tevatron_built(loss_fn)
    step = loader.Step(
        owner=loader.FRAMEWORK, callable="x.Model.forward", batch_keys=("query", "passage")
    )

    with pytest.raises(RuntimeError, match="loss=cached_mnrl with framework=tevatron"):
        bench_entry.train(built, batches(2), config, CPU, adapter_step=step)


def test_packing_with_a_framework_owned_step_is_refused(config_mapping):  # noqa: F811
    loader = importlib.import_module("trainbench.loader")
    config = bench(config_mapping, **{"run.purpose": "probe", "dataloader.packing": True})
    step = loader.Step(
        owner=loader.FRAMEWORK, callable="x.Model.forward", batch_keys=("query", "passage")
    )

    with pytest.raises(RuntimeError, match="dataloader.packing=true with framework=tevatron"):
        bench_entry.train(tevatron_built(), batches(2), config, CPU, adapter_step=step)
