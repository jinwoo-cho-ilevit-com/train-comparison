"""Applying an axis, and proving it took.

Every test here comes in a pair. One shows the axis being applied and read back;
the other breaks the thing being read — an enable hook that does nothing, a model
compiled at the wrong mode, a marker that matches no parameter — and shows the
same capture refusing it. A probe that only ever sees the working case certifies
the request rather than the result, which is the failure `trainbench/applied.py`
exists to prevent.

Two axes stay unapplied and are asserted to stay that way:

* `precision.name` — bf16 is not chosen here. The load dtype is decided by
  `trainbench/probe/steps.py::dtype_for` from the device, so `axes.py` has no
  site at which it turns bf16 on, and a capture without one would certify a
  choice this module never made. mxfp8/nvfp4 additionally need
  transformer-engine, absent from every environment.
* `train.offload` — inseparable from ZeRO: `deepspeed.initialize` returns the
  engine, optimizer and dataloader from one call, and deepspeed is absent from
  every environment.

Nothing here imports transformers, peft, liger or deepspeed. None of them is
installed in the environment the test suite runs in, and an axis whose only proof
is a package this suite cannot import is an axis with no proof.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import get_args

import pytest
import torch

from trainbench import axes
from trainbench.applied import (
    _CAPTURES,
    AppliedMismatch,
    Built,
    assert_matches,
    capture,
)
from trainbench.config_schema import ATTN_IMPL, AttnConfig, ModelConfig, axis_knobs

# The fixture and the config builders are shared with tests/test_applied.py rather
# than re-declared: two definitions of "a valid composed config" would drift, and
# the drift would be invisible because each file would still pass on its own.
from .test_applied import bench, config_mapping, gemma  # noqa: F401

CPU = torch.device("cpu")


@pytest.fixture
def composed(config_mapping):  # noqa: F811 - the imported fixture, requested once
    """The composed config mapping, under a name that does not shadow the import.

    Taking `config_mapping` directly in every test would have ruff read each
    parameter as a redefinition of the imported fixture. Re-requesting it once is
    the whole of the indirection.
    """
    return config_mapping


def axis(state, name):
    return next(a for a in state.axes if a.axis == name)


def plain_model() -> torch.nn.Module:
    """A real nn.Module: assemble builds an optimizer out of its parameters."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    return model


def towered_model(prefix: str) -> torch.nn.Module:
    """A model with a named tower, so parameter names look like a checkpoint's."""
    model = torch.nn.Module()
    model.add_module(prefix, torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)))
    model.add_module("language_model", torch.nn.Linear(2, 2))
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    return model


# --- attn.name (load_kwargs) -------------------------------------------------
#
# `load_kwargs` is one of the four call sites docs/CONTRACTS.md §2 fixes, and the
# only place `attn.name` is applied — yet nothing exercised it: `return {}` kept
# all 325 tests green. The capture side was covered, but a capture only ever sees
# a model somebody else built with these kwargs.


def test_load_kwargs_asks_for_the_requested_attention(composed):
    for name, impl in ATTN_IMPL.items():
        kwargs = axes.load_kwargs(bench(composed, **{"attn.name": name}))

        # The key is transformers' own parameter name. A different spelling is
        # accepted silently by `from_pretrained(**kwargs)` as an unused argument,
        # and the model is then built on the default implementation.
        assert kwargs == {"attn_implementation": impl}, name


def test_every_attention_value_has_a_kwarg_to_ask_for(composed):
    """A value in the schema with no entry in `ATTN_IMPL` would raise here rather
    than fall back, but only if something walks the whole set."""
    declared = set(get_args(AttnConfig.model_fields["name"].annotation))

    assert set(ATTN_IMPL) == declared


# --- compile.mode ------------------------------------------------------------


def test_a_compiled_model_reports_the_mode_it_was_compiled_at(composed):
    config = bench(composed, **{"compile.mode": "default"})

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")

    assert "compile.mode" in names
    assert axis(capture(built, config), "compile.mode").applied == "default"


def test_max_autotune_is_not_the_same_axis_value_as_default(composed):
    """The two differ only in inductor's configuration, so a capture that read the
    wrapper's presence alone would call an untuned run autotuned."""
    config = bench(composed, **{"compile.mode": "max-autotune", "train.warmup_discard_steps": 20})

    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")

    autotuned = axis(capture(built, config), "compile.mode")
    assert autotuned.applied == "max-autotune"
    assert autotuned.matches
    assert autotuned.detail["max_autotune"] is True


def test_a_model_compiled_at_the_wrong_mode_is_a_mismatch(composed):
    """The break: everything else about the run is identical, and the number it
    would produce is a `default` number wearing the `max-autotune` label."""
    config = bench(composed, **{"compile.mode": "max-autotune", "train.warmup_discard_steps": 20})

    state = capture(Built(model=torch.compile(plain_model(), mode="default")), config)

    assert axis(state, "compile.mode").applied == "default"
    with pytest.raises(AppliedMismatch, match="compile.mode"):
        assert_matches(state, config)


def test_an_uncompiled_model_does_not_pass_as_a_compiled_run(composed):
    config = bench(composed, **{"compile.mode": "default"})

    state = capture(Built(model=plain_model()), config)

    assert axis(state, "compile.mode").applied == "none"
    assert not axis(state, "compile.mode").matches


def test_compiling_while_dynamo_is_disabled_is_not_a_compiled_run(composed, monkeypatch):
    """`TORCHDYNAMO_DISABLE=1` still hands back an OptimizedModule. A capture that
    stopped at the wrapper would report a compiled run for an eager one, and the
    variable is the sort of thing that ends up set in a container image."""
    config = bench(composed, **{"compile.mode": "default"})
    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")
    monkeypatch.setattr(torch._dynamo.config, "disable", True)

    compiled = axis(capture(built, config), "compile.mode")

    assert compiled.applied == "none"
    assert compiled.detail["dynamo_disabled"] is True


def test_compile_none_leaves_the_model_alone(composed):
    model = plain_model()

    built, names = axes.assemble(model, bench(composed), CPU, framework="native")

    assert built.model is model
    assert "compile.mode" not in names


def test_regional_compilation_is_read_back_as_regional(composed):
    """Regional wraps the repeated blocks and leaves the root alone, so the root
    being unwrapped is what tells it apart from whole-model compilation."""

    class Blocky(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

        def compile_repeated_blocks(self):
            self.blocks[0] = torch.compile(self.blocks[0])

    config = bench(composed, **{"compile.mode": "regional"})

    built, _ = axes.assemble(Blocky(), config, CPU, framework="native")

    assert axis(capture(built, config), "compile.mode").applied == "regional"


def test_regional_is_refused_on_a_model_that_cannot_do_it(composed):
    config = bench(composed, **{"compile.mode": "regional"})

    with pytest.raises(axes.UnappliedAxis, match="compile_repeated_blocks"):
        axes.assemble(plain_model(), config, CPU, framework="native")


# --- kernel.name -------------------------------------------------------------


class LigerRMSNorm(torch.nn.Module):
    """Stands in for a class a kernel library substituted into the model."""


LigerRMSNorm.__module__ = "liger_kernel.transformers.rms_norm"


def test_a_plain_model_carries_no_kernel(composed):
    config = bench(composed)

    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")

    assert axis(capture(built, config), "kernel.name").applied == "none"


def test_a_patched_model_is_not_kernel_none(composed):
    """The break, and the case that matters most: nothing in `axes.patch` applied
    a kernel, so a capture keyed on what this module did would say `none`. The
    framework adapters patch transformers themselves — that is what Unsloth is —
    and the model is where that becomes visible."""
    config = bench(composed)
    model = plain_model()
    model.add_module("norm", LigerRMSNorm())

    state = capture(Built(model=model), config)

    assert axis(state, "kernel.name").applied == "liger"
    with pytest.raises(AppliedMismatch, match="kernel.name"):
        assert_matches(state, config)


def test_kernel_values_with_no_patch_are_refused(composed):
    for name in ("liger", "fla", "kernels_hub"):
        with pytest.raises(axes.UnappliedAxis, match="kernel="):
            axes.patch(bench(composed, **{"kernel.name": name}))


# --- freeze.ple --------------------------------------------------------------
#
# The capture side is covered in tests/test_applied.py, but every one of those
# tests feeds an already-frozen fixture straight to `capture` and so never runs
# `assemble`. That left the apply site bare: replacing the `requires_grad_(False)`
# loop in `_freeze` with a bare call kept all 325 tests green. These two go
# through `assemble`, which is where a run reaches it. The axis is 46.8% of
# gemma-4-E2B's parameters and `_ple_report` has already shipped one silent
# failure on this exact marker.


def ple_model() -> torch.nn.Module:
    """A model whose parameter names carry gemma-4's per-layer-embedding marker."""
    model = torch.nn.Module()
    model.add_module(
        "embed_tokens_per_layer", torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    )
    model.add_module("language_model", torch.nn.Linear(2, 2))
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    return model


def test_freezing_the_ple_tables_goes_through_assemble(composed):
    config = gemma(composed, **{"freeze.ple": True})

    built, names = axes.assemble(ple_model(), config, CPU, framework="native")

    assert "freeze.ple" in names
    frozen = axis(capture(built, config), "freeze.ple")
    assert frozen.applied == "True"
    assert frozen.detail == {"matched": 4, "frozen": 4}


def test_a_ple_freeze_that_did_not_take_is_a_mismatch(composed):
    """The break, and the one that costs most: these tables are roughly half the
    model, so a freeze that quietly did nothing moves the optimizer-memory number
    this model is in the study to demonstrate."""
    config = gemma(composed, **{"freeze.ple": True})
    model = ple_model()
    built, _ = axes.assemble(model, config, CPU, framework="native")
    model.embed_tokens_per_layer[0].weight.requires_grad_(True)

    state = capture(built, config)

    assert axis(state, "freeze.ple").applied == "partial"
    with pytest.raises(AppliedMismatch, match="freeze.ple"):
        assert_matches(state, config)


# --- freeze.vision_tower -----------------------------------------------------


def test_every_architecture_has_a_recorded_vision_marker():
    """A new architecture without one would raise at apply time and read as
    undetermined forever, which is safe but silent. This is where it gets said."""
    declared = set(get_args(ModelConfig.model_fields["arch"].annotation))

    assert set(axes.VISION_PARAM_MARKERS) == declared


def test_freezing_the_vision_tower_is_read_off_the_parameters(composed):
    config = bench(composed, **{"freeze.vision_tower": True})

    built, names = axes.assemble(towered_model("visual"), config, CPU, framework="native")

    assert "freeze.vision_tower" in names
    frozen = axis(capture(built, config), "freeze.vision_tower")
    assert frozen.applied == "True"
    assert frozen.detail == {"matched": 4, "frozen": 4, "arch": "qwen3_vl"}


def test_a_half_frozen_tower_is_not_a_frozen_tower(composed):
    config = bench(composed, **{"freeze.vision_tower": True})
    model = towered_model("visual")
    built, _ = axes.assemble(model, config, CPU, framework="native")
    model.visual[0].weight.requires_grad_(True)

    state = capture(built, config)

    assert axis(state, "freeze.vision_tower").applied == "partial"
    with pytest.raises(AppliedMismatch, match="freeze.vision_tower"):
        assert_matches(state, config)


def test_a_marker_that_matches_nothing_is_undetermined_not_frozen(composed):
    """The break, and the failure `_ple_report` already shipped once: a marker that
    fits no parameter freezes nothing, and "nothing is frozen" is the same answer a
    correct marker gives for a tower left training."""
    config = bench(composed, **{"freeze.vision_tower": True})

    state = capture(Built(model=towered_model("encoder")), config)

    assert axis(state, "freeze.vision_tower").applied is None
    assert axis(state, "freeze.vision_tower").detail["matched"] == 0


def test_the_markers_are_architecture_specific(composed):
    """gemma-4 calls it `vision_tower` and both Qwen models call it `visual`. One
    marker for all three would match nothing on two of them."""
    gemma_model = towered_model("vision_tower")

    under_gemma = capture(Built(model=gemma_model), gemma(composed))
    under_qwen = capture(Built(model=gemma_model), bench(composed))

    assert under_gemma.axes and axis(under_gemma, "freeze.vision_tower").applied == "False"
    assert axis(under_qwen, "freeze.vision_tower").applied is None


def test_an_architecture_with_no_recorded_marker_is_refused():
    with pytest.raises(axes.UnappliedAxis, match="no vision-tower parameter marker"):
        axes.vision_parameters(towered_model("visual"), "llama9")


# --- train.gradient_checkpointing --------------------------------------------


class Checkpointable(torch.nn.Module):
    """A model that exposes transformers' checkpointing hook.

    `honours` is what makes the pair of tests below possible: a model that accepts
    the call and changes nothing is exactly what a silent no-op looks like.
    """

    def __init__(self, honours: bool = True):
        super().__init__()
        self.block = torch.nn.Linear(2, 2)
        self.block.gradient_checkpointing = False
        self.honours = honours
        self.enable_calls: list = []
        self.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.enable_calls.append(gradient_checkpointing_kwargs)
        if self.honours:
            self.block.gradient_checkpointing = True


def test_checkpointing_goes_through_the_models_own_hook(composed):
    config = bench(composed, **{"train.gradient_checkpointing": "full"})
    model = Checkpointable()

    built, names = axes.assemble(model, config, CPU, framework="native")

    assert "train.gradient_checkpointing" in names
    # Non-reentrant is not a preference: the reentrant variant skips recomputation
    # entirely when nothing entering a block requires grad, which is what the
    # freeze axes produce, and freeze x checkpointing is a cell of the ablation.
    assert model.enable_calls == [{"use_reentrant": False}]
    assert axis(capture(built, config), "train.gradient_checkpointing").applied == "full"


def test_a_model_that_swallows_the_request_is_caught(composed):
    """The break. The call returns cleanly, the run trains, and nothing but the
    flags on the modules says the activation memory was never traded away."""
    config = bench(composed, **{"train.gradient_checkpointing": "full"})
    model = Checkpointable(honours=False)

    built, _ = axes.assemble(model, config, CPU, framework="native")
    state = capture(built, config)

    assert model.enable_calls, "the hook was called"
    assert axis(state, "train.gradient_checkpointing").applied == "none"
    with pytest.raises(AppliedMismatch, match="train.gradient_checkpointing"):
        assert_matches(state, config)


def test_checkpointing_and_compile_compose(composed):
    """Both axes end up applied when both are asked for, and the capture finds the
    flags through the compiled wrapper.

    Not an ordering test. `OptimizedModule.__getattr__` delegates to `_orig_mod`,
    so the hook reaches through either way and reversing the two in
    `_apply_to_model` leaves this green — which is why the ordering rationale in
    that docstring now says which half of it is a choice.
    """
    config = bench(composed, **{"train.gradient_checkpointing": "full", "compile.mode": "default"})
    model = Checkpointable()

    built, _ = axes.assemble(model, config, CPU, framework="native")

    assert built.model is not model, "the compiled wrapper replaced the model"
    assert model.enable_calls == [{"use_reentrant": False}]
    assert axis(capture(built, config), "train.gradient_checkpointing").applied == "full"


def test_a_model_with_no_checkpointing_hook_is_refused(composed):
    config = bench(composed, **{"train.gradient_checkpointing": "full"})

    with pytest.raises(axes.UnappliedAxis, match="gradient_checkpointing_enable"):
        axes.assemble(plain_model(), config, CPU, framework="native")


def test_a_model_that_exposes_no_flag_is_undetermined(composed):
    state = capture(Built(model=plain_model()), bench(composed))

    assert axis(state, "train.gradient_checkpointing").applied is None


def test_selective_checkpointing_is_refused(composed):
    config = bench(composed, **{"train.gradient_checkpointing": "selective"})

    with pytest.raises(axes.UnappliedAxis, match="selective"):
        axes.assemble(Checkpointable(), config, CPU, framework="native")


# --- peft.mode ---------------------------------------------------------------


def test_a_model_with_no_adapter_is_a_full_finetune(composed):
    config = bench(composed)

    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")

    assert axis(capture(built, config), "peft.mode").applied == "full"


def test_an_adapter_nothing_here_attached_is_still_seen(composed):
    """The break. `axes.py` refuses to attach an adapter, so a capture keyed on
    what this module did would report `full` for every run — including one whose
    framework adapter wrapped the model on its way in."""
    config = bench(composed)
    model = plain_model()
    model.peft_config = {"default": SimpleNamespace(peft_type="PeftType.LORA")}

    state = capture(Built(model=model), config)

    assert axis(state, "peft.mode").applied == "lora"
    with pytest.raises(AppliedMismatch, match="peft.mode"):
        assert_matches(state, config)


def test_qlora_is_told_apart_by_the_quantised_base(composed):
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32})
    model = plain_model()
    model.peft_config = {"default": SimpleNamespace(peft_type="PeftType.LORA")}
    model.is_loaded_in_4bit = True

    assert axis(capture(Built(model=model), config), "peft.mode").applied == "qlora"


def test_lora_is_refused_rather_than_run_as_a_full_finetune(composed):
    """Silently training every parameter under a LoRA label is the headline
    comparison of this study reported backwards."""
    for mode in ("lora", "qlora"):
        config = bench(composed, **{"peft.mode": mode, "peft.r": 32})
        with pytest.raises(axes.UnappliedAxis, match="peft.mode"):
            axes.assemble(plain_model(), config, CPU, framework="native")


# --- parallel.strategy -------------------------------------------------------


class DistributedDataParallel(torch.nn.Module):
    """Named for what the capture matches on: the wrapper's class name."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def test_a_single_process_run_reports_single(composed):
    config = bench(composed)

    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")

    strategy = axis(capture(built, config), "parallel.strategy")
    assert strategy.applied == "single"
    assert strategy.detail == {"world_size": 1, "process_group": False}


def test_single_is_checked_against_the_process_group_not_only_the_wrapper(composed, monkeypatch):
    """The break. Launched under torchrun with `parallel=single`, every rank runs
    its own unsynchronised replica and the wall clock looks like a speedup."""
    import torch.distributed as dist

    config = bench(composed)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)

    state = capture(Built(model=plain_model()), config)

    assert axis(state, "parallel.strategy").applied == "unwrapped(world_size=4)"
    with pytest.raises(AppliedMismatch, match="parallel.strategy"):
        assert_matches(state, config)


def test_a_wrapped_model_reports_its_wrapper(composed):
    config = bench(composed)

    state = capture(Built(model=DistributedDataParallel(plain_model())), config)

    assert axis(state, "parallel.strategy").applied == "ddp"


def test_strategies_that_wrap_or_shard_are_refused(composed):
    for strategy, match in (
        ("ddp", "wraps the model"),
        ("fsdp2", "wraps the model"),
        ("zero2", "deepspeed.initialize"),
        ("zero3", "deepspeed.initialize"),
    ):
        config = bench(composed, **{"parallel.strategy": strategy})
        with pytest.raises(axes.UnappliedAxis, match=match):
            axes.assemble(plain_model(), config, CPU, framework="native")


# --- parallel.cross_device_negatives -----------------------------------------


def test_the_local_loss_declares_that_it_does_not_gather(composed):
    config = bench(composed)

    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")

    gathered = axis(capture(built, config), "parallel.cross_device_negatives")
    assert gathered.applied == "False"
    assert gathered.matches


def test_a_loss_from_somewhere_else_cannot_certify_this_axis(composed):
    """The break: the declaration is what the local closure carries, so any other
    loss is undetermined rather than assumed not to gather."""
    state = capture(Built(loss_fn=lambda q, d: q), bench(composed))

    assert axis(state, "parallel.cross_device_negatives").applied is None


# --- dataloader.* ------------------------------------------------------------


def dataset(columns: tuple[str, ...] | None = None):
    """A torch dataset that optionally declares its columns the way HF's does."""
    data = torch.utils.data.TensorDataset(torch.zeros(8, 2))
    if columns is not None:
        data.column_names = list(columns)
    return data


def assembled_loader(composed, dataset_obj, **overrides):
    config = bench(composed, **overrides)
    built, _ = axes.assemble(plain_model(), config, CPU, framework="native", dataset=dataset_obj)
    return built, config


def test_the_backend_is_the_iterator_that_was_built(composed):
    built, config = assembled_loader(composed, dataset(), **{"train.batch_size": 4})

    assert axis(capture(built, config), "dataloader.backend").applied == "torch"


def test_a_foreign_iterator_is_not_the_torch_backend(composed):
    """The break. DALI replaces the DataLoader rather than configuring one, so the
    object's type is the whole of the evidence."""

    class DALIGenericIterator:
        dataset = None
        collate_fn = None

    DALIGenericIterator.__module__ = "nvidia.dali.plugin.pytorch"
    config = bench(composed)

    state = capture(Built(dataloader=DALIGenericIterator()), config)

    assert axis(state, "dataloader.backend").applied == "dali"
    with pytest.raises(AppliedMismatch, match="dataloader.backend"):
        assert_matches(state, config)


def test_a_run_with_no_loader_leaves_the_dataloader_axes_undetermined(composed):
    config = bench(composed)

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")
    state = capture(built, config)

    assert "dataloader.backend" not in names
    assert {a.axis for a in state.undetermined()} >= {
        "dataloader.backend",
        "dataloader.packing",
        "dataloader.pretokenize",
    }


def test_torchs_own_collate_is_evidence_that_nothing_packs(composed):
    built, config = assembled_loader(composed, dataset(), **{"train.batch_size": 4})

    assert axis(capture(built, config), "dataloader.packing").applied == "False"


def test_an_unrecognised_collate_is_undetermined_rather_than_unpacked(composed):
    """The break: a packing implementation would live in the collate, so calling
    an unknown one `False` is how a packed run gets an unpacked label — and the
    other way round."""
    built, config = assembled_loader(composed, dataset(), **{"train.batch_size": 4})
    built.dataloader.collate_fn = lambda rows: rows

    packing = axis(capture(built, config), "dataloader.packing")

    assert packing.applied is None
    assert "axis_packing" in packing.detail["reason"]


def test_pretokenised_rows_are_visible_on_the_dataset(composed):
    """The break. `pretokenize` moves work out of the timed window, so the loader
    is identical either way and only its dataset can tell."""
    built, config = assembled_loader(
        composed, dataset(("input_ids", "attention_mask")), **{"train.batch_size": 4}
    )

    state = capture(built, config)

    assert axis(state, "dataloader.pretokenize").applied == "True"
    with pytest.raises(AppliedMismatch, match="dataloader.pretokenize"):
        assert_matches(state, config)


def test_untokenised_rows_read_as_untokenised(composed):
    built, config = assembled_loader(
        composed, dataset(("qry", "pos_text")), **{"train.batch_size": 4}
    )

    assert axis(capture(built, config), "dataloader.pretokenize").applied == "False"


def test_a_dataset_that_declares_no_columns_is_undetermined(composed):
    built, config = assembled_loader(composed, dataset(), **{"train.batch_size": 4})

    assert axis(capture(built, config), "dataloader.pretokenize").applied is None


def test_packing_and_pretokenize_are_refused(composed):
    for override in ({"dataloader.packing": True}, {"dataloader.pretokenize": True}):
        with pytest.raises(axes.UnappliedAxis):
            axes.assemble(
                plain_model(),
                bench(composed, **override),
                CPU,
                framework="native",
                dataset=dataset(),
            )


# --- the two that stay unapplied ---------------------------------------------


def test_precision_and_offload_are_still_unapplied():
    """Named rather than left implicit. `axis-wired` reports them as the remaining
    gap, and this is the assertion that has to be deleted to close it."""
    unwired = set(axes.IMPLEMENTED) ^ {
        "attn.name",
        "compile.mode",
        "dataloader.backend",
        "dataloader.packing",
        "dataloader.pretokenize",
        "framework.name",
        "freeze.ple",
        "freeze.vision_tower",
        "kernel.name",
        "loss.name",
        "optim.name",
        "parallel.cross_device_negatives",
        "parallel.strategy",
        "peft.mode",
        "train.gradient_checkpointing",
    }

    assert unwired == set()
    assert "precision.name" not in axes.IMPLEMENTED
    assert "train.offload" not in axes.IMPLEMENTED


def test_an_inert_configuration_claims_to_have_applied_nothing(composed):
    """`compile=none`, `peft=full` and `freeze.*=false` change nothing, so naming
    them would make the applied list a copy of the config."""
    built, names = axes.assemble(plain_model(), bench(composed), CPU, framework="native")

    assert set(names) == {"optim.name", "loss.name", "framework.name"}
    assert built.model is not None


def test_capture_reports_an_axis_with_no_probe_as_undetermined(composed):
    """The half of the no-probe guarantee that produces `assert_matches`' input.

    `tests/test_applied.py` covers the consuming half with a synthetic axis. That
    left this one uncovered: mutating `capture`'s no-probe branch to
    `AxisState(axis, requested, requested, ...)` — one word — certified
    `precision.name` and `train.offload` as applied and left all 325 tests green.

    Derived from `_CAPTURES` rather than naming an axis, so wiring one does not
    break it. The empty-set guard is the point of the derivation: when the last
    axis is wired this fails loudly instead of passing on nothing to examine, and
    it gets deleted then, together with the `axis-wired` line in
    docs/audit-baseline.json.
    """
    unwired = set(axis_knobs()) - set(_CAPTURES)
    assert unwired, (
        "every axis now has a capture probe; delete this test and the axis-wired "
        "baseline entry together rather than leaving a check with nothing to examine"
    )

    state = capture(Built(model=plain_model()), bench(composed))

    for name in unwired:
        recorded = axis(state, name)
        assert recorded.applied is None, name
        assert recorded.detail["reason"] == "no capture probe implemented", name


def test_a_quality_run_is_enforced_exactly_like_a_timing_run(composed):
    """`quality` is in `ENFORCED_PURPOSES` and nothing held it there: dropping it
    left the suite green, and a long quality run would then have produced a loss
    curve for settings no one checked."""
    config = bench(composed, **{"run.purpose": "quality"})

    state = capture(Built(model=plain_model()), config)

    with pytest.raises(AppliedMismatch, match="purpose=quality"):
        assert_matches(state, config)
