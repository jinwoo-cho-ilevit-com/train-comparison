"""Applying an axis, and proving it took.

Every test here comes in a pair. One shows the axis being applied and read back;
the other breaks the thing being read — an enable hook that does nothing, a model
compiled at the wrong mode, a marker that matches no parameter — and shows the
same capture refusing it. A probe that only ever sees the working case certifies
the request rather than the result, which is the failure `trainbench/applied.py`
exists to prevent.

Two axes reach their inert value here and nowhere else, and both are now read
back by a probe in `trainbench/applied.py` rather than assumed:

* `precision.name` — bf16 is not chosen here. The load dtype is decided by
  `trainbench/probe/steps.py::dtype_for` from the device, so `axes.py` only
  refuses the values it cannot put into effect: mxfp8/nvfp4 need
  transformer-engine, absent from every environment. What makes bf16 a claim
  rather than a hope is `applied._capture_precision`, which reads the dtype off
  the weights — a run loaded in fp32 is refused however the config reads.
* `train.offload` — inseparable from ZeRO: `deepspeed.initialize` returns the
  engine, optimizer and dataloader from one call, and deepspeed is absent from
  every environment. `assemble` therefore refuses everything but `none`, and
  `applied._capture_offload` proves `none` by finding every parameter and
  optimizer state tensor on the device the model computes on.

Nothing here imports transformers, liger or deepspeed. None of them is installed
in the environment the test suite runs in, and an axis whose only proof is a
package this suite cannot import is an axis with no proof.

peft is the exception, and it is imported on purpose: it *is* installed here
(0.20.0), and `peft.mode` was refused on the stated grounds that it was not. The
adapter tests below build a real `get_peft_model` model, because the question the
axis turns on — what `get_peft_model` does to base parameters — cannot be answered
by a stand-in that does whatever the test sets it to do.

transformers is a second exception, in one test and for the same reason. It is
installed (5.14.1) — the paragraph above is about liger and deepspeed, which are
not. `train.gradient_checkpointing` is read back off an attribute transformers
sets (`_gradient_checkpointing_func`), and a stand-in that sets that attribute
proves only that the test and the probe agree with each other. The test builds a
real `PreTrainedModel` from a config: no weights, no network, no GPU.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import get_args

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from trainbench import axes
from trainbench.applied import (
    AppliedMismatch,
    Built,
    assert_matches,
    capture,
)
from trainbench.config_schema import (
    ATTN_IMPL,
    AttnConfig,
    KernelConfig,
    ModelConfig,
    PrecisionConfig,
    axis_knobs,
)
from trainbench.embedding import last_token_pool, packed_last_token_pool

# The fixture and the config builders are shared with tests/test_applied.py rather
# than re-declared: two definitions of "a valid composed config" would drift, and
# the drift would be invisible because each file would still pass on its own.
from .test_applied import bench, config_mapping, gemma  # noqa: F401

CPU = torch.device("cpu")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


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


# --- peft.mode=qlora, the half that is a dict (load_kwargs) ------------------
#
# QLoRA is LoRA over a 4-bit base. The adapter half is the same `get_peft_model`
# call as plain LoRA; the base half is a `BitsAndBytesConfig` that has to reach
# `from_pretrained`, because weights already materialised in bf16 cannot be
# quantised afterwards. What that call is asked for is a dict, and a dict is
# assertable on a laptop — `BitsAndBytesConfig` is a transformers dataclass and
# imports with bitsandbytes absent.
#
# What is not asserted anywhere is that 4-bit weights come back. bitsandbytes
# quantises on CUDA, there is no GPU here, and the package is not installed: that
# half is 측정 안 함, and the tests below are about the request rather than the
# result. `_peft` still refuses the axis outright, so nothing in this repository
# runs a qlora step yet on any device.


def on_cuda(config):
    """The same config as if `device=cuda` had been composed.

    `device.get_device` takes the override at its word instead of probing for
    hardware — that is what lets a CUDA-only branch be read on a machine without
    one. Nothing is allocated, loaded or quantised by any test that uses this.
    """
    return config.model_copy(update={"device": "cuda"})


def test_qlora_asks_from_pretrained_for_a_four_bit_base(composed):
    config = on_cuda(bench(composed, **{"peft.mode": "qlora", "peft.r": 32}))

    kwargs = axes.load_kwargs(config)

    # The key is transformers' own parameter name, for the reason the attention
    # test above gives: a misspelled one is accepted and ignored, and the run then
    # trains a full-precision base under the qlora label.
    assert set(kwargs) == {"attn_implementation", "quantization_config"}
    quantisation = kwargs["quantization_config"]
    # Named rather than imported: this file imports no transformers (see the module
    # docstring), and what matters is the object `from_pretrained` receives.
    assert type(quantisation).__name__ == "BitsAndBytesConfig"
    assert quantisation.load_in_4bit is True
    assert quantisation.bnb_4bit_quant_type == "nf4"
    assert quantisation.bnb_4bit_use_double_quant is True
    # bf16 is the only precision `step_context` lets a run reach; a compute dtype
    # that disagreed with it would be a precision no axis asked for.
    assert quantisation.bnb_4bit_compute_dtype is torch.bfloat16


def test_full_and_lora_are_loaded_without_a_quantisation_config(composed):
    """One half of the break: a `quantization_config` returned for every run would
    satisfy the test above word for word, and would quantise the full finetune that
    is this study's baseline."""
    for mode, rank in (("full", 0), ("lora", 32)):
        config = on_cuda(bench(composed, **{"peft.mode": mode, "peft.r": rank}))

        assert set(axes.load_kwargs(config)) == {"attn_implementation"}, mode


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_a_qlora_run_cannot_start_off_cuda(composed, device):
    """The other half, and the one that matters on this machine.

    bitsandbytes quantises on CUDA. Handing the config back anyway would leave the
    quantisation to be ignored somewhere downstream, and an ignored quantisation is
    a full-precision base whose speed gets reported as 4-bit. The device is
    overridden explicitly rather than left to resolve, so the refusal is asserted on
    a GPU host too.

    More than one non-CUDA device, because "off cuda" is the claim. Asserting it
    for `cpu` alone leaves `device.type == "cpu"` — one character from what is
    written — passing the whole suite while quantising nothing on this laptop's
    mps and on an Intel host's xpu. `get_device` takes the string at its word, so
    none of these has to exist here.
    """
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32}).model_copy(
        update={"device": device}
    )

    with pytest.raises(axes.UnappliedAxis, match=f"device={device} would load"):
        axes.load_kwargs(config)


def test_qlora_gates_on_the_device_the_run_resolves_to(composed, monkeypatch):
    """`device: null` is the config default and what every pod composes, so the
    resolution path is the one a real run takes — and no test walked it: every
    other qlora test names a device outright.

    The accelerator is stubbed rather than read, because what is under test is the
    gate rather than this host: a machine with no CUDA would pass a gate that
    quantised on any accelerator it found, which is how mps would get 4-bit weights
    it cannot make.
    """
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32}).model_copy(
        update={"device": None}
    )
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)

    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("mps"))
    with pytest.raises(axes.UnappliedAxis, match="device=mps"):
        axes.load_kwargs(config)

    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
    assert "quantization_config" in axes.load_kwargs(config)


def test_the_qlora_compute_dtype_stays_a_precision_a_run_can_reach(composed):
    """`QLORA_4BIT` fixes bf16 compute on the grounds that `step_context` refuses
    every other precision. That is one constant resting on another function's
    refusal with nothing between them: the day `mxfp8` or `nvfp4` becomes
    reachable, a qlora run under it computes in bf16 and the study prints the fp8
    recipe's name over a bf16 number. Nothing else pins the pair together, so this
    does — it is the notice, not the fix."""
    reachable = []
    for name in get_args(PrecisionConfig.model_fields["name"].annotation):
        try:
            axes.step_context(bench(composed, **{"precision.name": name}))
        except axes.UnappliedAxis:
            continue
        reachable.append(name)

    assert reachable == ["bf16"], (
        f"precision {reachable} can be reached now, and QLORA_4BIT still computes in "
        f"{axes.QLORA_4BIT['bnb_4bit_compute_dtype']}; decide what the combination "
        "means before a run measures one under the other's name"
    )
    assert axes.QLORA_4BIT["bnb_4bit_compute_dtype"] is torch.bfloat16


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


class FusedRMSNormGated(torch.nn.Module):
    """Stands in for the fla class transformers builds each Qwen3.5 gated norm out
    of when the package is installed (`modeling_qwen3_5.py:409`)."""


FusedRMSNormGated.__module__ = "fla.modules.fused_norm_gate"


class UnslothRMSNorm(torch.nn.Module):
    """A replacement from something that is not a kernel axis value."""


UnslothRMSNorm.__module__ = "unsloth.models.llama"


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


def qwen35(mapping, **overrides):
    """A config on Qwen3.5, the one architecture of the three with a kernel path
    for both liger and fla. Defined here rather than in tests/test_applied.py,
    which docs/CONTRACTS.md §1 shares across lanes; `instruction_prompt` has to go
    with the arch because the schema refuses an invented prompt on a generative
    model."""
    return bench(
        mapping,
        **{"model.arch": "qwen3_5", "model.instruction_prompt": None},
        **overrides,
    )


def test_a_request_for_liger_on_a_stock_model_is_a_mismatch(composed):
    """The other break, and the one this axis exists to catch: the config asks for
    liger and the built model carries none of it. `patch` reporting success is not
    evidence — only the model is — so the run stops before it produces a number."""
    config = qwen35(composed, **{"kernel.name": "liger"})

    state = capture(Built(model=plain_model()), config)

    assert axis(state, "kernel.name").applied == "none"
    with pytest.raises(AppliedMismatch, match="kernel.name"):
        assert_matches(state, config)


def patched_modelling_module(
    monkeypatch,
    *,
    replacement,
    module_name: str = "transformers.models.fake.modeling_fake",
    class_name: str = "StockRMSNorm",
) -> type[torch.nn.Module]:
    """A modelling module after a kernel library rebound one of its names, and the
    stock class an instance built before that still carries.

    This is the whole mechanism of a kernel patch — `modeling_x.Name = LigerName`
    and then construction — so the state it produces is what tells a fully covered
    model from a half covered one. Built here rather than with the real thing
    because liger-kernel does not install on this machine.

    `module_name` is a parameter because the module a kernel library rebinds is
    not always a transformers one: on Qwen3.5 the first library in is fla, and a
    second one patches over classes that belong to `fla.modules`.
    """
    module = ModuleType(module_name)

    class StockModule(torch.nn.Module):
        pass

    StockModule.__module__ = module_name
    StockModule.__name__ = class_name
    StockModule.__qualname__ = class_name
    setattr(module, class_name, replacement)
    monkeypatch.setitem(sys.modules, module_name, module)
    return StockModule


def test_a_model_the_kernel_only_half_reached_is_not_a_full_kernel_run(composed, monkeypatch):
    """The fail-open case this axis makes reachable: package presence is a scan
    over the whole model, so one patched module answers it exactly like every
    module patched, and the throughput ships under the kernel's name either way.
    A module still built from a class the patch superseded is the evidence that
    the model was assembled around the patch rather than after it."""
    stock = patched_modelling_module(monkeypatch, replacement=LigerRMSNorm)
    config = qwen35(composed, **{"kernel.name": "liger"})
    model = plain_model()
    model.add_module("patched", LigerRMSNorm())
    model.add_module("built_before_the_patch", stock())

    state = capture(Built(model=model), config)

    assert axis(state, "kernel.name").applied == "partial(liger)"
    with pytest.raises(AppliedMismatch, match="kernel.name"):
        assert_matches(state, config)


def test_a_model_built_entirely_after_the_patch_is_a_kernel_run(composed, monkeypatch):
    """The other half of the pair, and what keeps the check above from being a
    refusal of the axis: liger patching the decoder and leaving the rest of the
    model alone is the library's documented behaviour, so coverage is recorded for
    the reader rather than judged against a threshold."""
    patched_modelling_module(monkeypatch, replacement=LigerRMSNorm)
    config = qwen35(composed, **{"kernel.name": "liger"})
    model = plain_model()
    model.add_module("patched", LigerRMSNorm())

    state = capture(Built(model=model), config)

    assert axis(state, "kernel.name").applied == "liger"
    assert axis(state, "kernel.name").detail["kernel_modules"] == {"liger": 1}
    assert "kernel.name" not in [state_.axis for state_ in state.mismatched()]


def test_a_class_replaced_by_something_this_cannot_name_is_undetermined(composed, monkeypatch):
    """A framework adapter that rewrites transformers under the run — what Unsloth
    is — leaves modules whose class was superseded by no kernel this names. What
    ran is then unknown, and unknown blocks a timing run like a mismatch does."""
    stock = patched_modelling_module(monkeypatch, replacement=UnslothRMSNorm)
    config = qwen35(composed)
    model = plain_model()
    model.add_module("built_before_the_patch", stock())

    state = capture(Built(model=model), config)

    assert axis(state, "kernel.name").applied is None
    assert "kernel.name" in [state_.axis for state_ in state.undetermined()]


def test_a_qwen35_on_an_image_with_fla_reads_back_as_fla(composed):
    """`kernel=none` is the default of every run in configs/config.yaml, and on
    this architecture no image can satisfy it: transformers builds the gated norms
    out of fla whenever the package is there. Fail-closed and correct, and it is
    why this model produces no number on any axis until the run asks for fla."""
    config = qwen35(composed)
    model = plain_model()
    model.add_module("norm", FusedRMSNormGated())

    state = capture(Built(model=model), config)

    assert axis(state, "kernel.name").applied == "fla"
    with pytest.raises(AppliedMismatch, match="kernel.name"):
        assert_matches(state, config)


# The patch calls themselves cannot run here: liger-kernel and flash-linear-attention
# depend on triton, which publishes no macOS wheel, and `kernels` fetches
# device-specific prebuilt kernels from the Hub. What is testable is the wiring —
# which value routes where, what is refused before anything is imported, and what
# happens when the package is absent or exports a different name — and that is what
# these do, against stub modules. Whether a real patch takes inside a framework
# image is a first-pod check; `_capture_kernel` above is what will answer it.


def liger_stub(*, entrypoint: str | None) -> tuple[object, list[str]]:
    """A stand-in for `liger_kernel.transformers`, plus the log of what it applied."""
    calls: list[str] = []
    module = SimpleNamespace(apply_liger_kernel_to_llama=lambda: calls.append("llama"))
    if entrypoint is not None:
        setattr(module, entrypoint, lambda: calls.append(entrypoint))
    return module, calls


def install_liger(monkeypatch, module) -> None:
    """Put a stub where `importlib.import_module` will find it. `None` makes the
    import raise, which is how an environment without the package behaves."""
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", module)


def test_kernel_none_patches_nothing(composed):
    assert axes.patch(bench(composed)) == []


def test_every_kernel_the_schema_offers_routes_to_a_patcher():
    """A value added to the schema with no patcher would otherwise reach
    `KERNEL_PATCHERS[name]` and die on a KeyError, which reads as a broken axis
    rather than an unimplemented one."""
    declared = set(get_args(KernelConfig.model_fields["name"].annotation))

    assert set(axes.KERNEL_PATCHERS) | {"none"} == declared


def test_liger_calls_the_entrypoint_for_this_architecture(composed, monkeypatch):
    # An image without fla, stated rather than inherited from the host: Qwen3.5 on
    # an image that has fla is refused before any entrypoint is reached, and the
    # suite runs inside those images too.
    fla_environment(monkeypatch, fla=None)
    module, calls = liger_stub(entrypoint="apply_liger_kernel_to_qwen3_5")
    install_liger(monkeypatch, module)
    config = qwen35(composed, **{"kernel.name": "liger"})

    applied = axes.patch(config)

    assert applied == ["kernel.name"]
    assert calls == ["apply_liger_kernel_to_qwen3_5"]


def test_liger_reaches_gemma4_on_a_pin_that_has_the_entrypoint(composed, monkeypatch):
    """`LIGER_UNSUPPORTED` used to refuse this outright, citing Liger-Kernel#1186,
    and the pinned wheel says otherwise: liger-kernel 0.8.1 defines
    `apply_liger_kernel_to_gemma4` and maps `gemma4` onto it. A refusal written
    from an issue tracker would have kept one of the study's three models out of
    this axis for the whole campaign."""
    module, calls = liger_stub(entrypoint="apply_liger_kernel_to_gemma4")
    install_liger(monkeypatch, module)
    config = gemma(composed, **{"kernel.name": "liger"})

    assert axes.patch(config) == ["kernel.name"]
    assert calls == ["apply_liger_kernel_to_gemma4"]


def test_liger_on_a_pin_without_the_entrypoint_names_what_that_pin_has(composed, monkeypatch):
    """The other side of the same fact: 0.8.0 has gemma-4's *text* entrypoint and
    not its multimodal one, so which image is running decides this. The version
    boundary is answered by the installed package rather than by a table, and the
    refusal has to carry the correction with it."""
    module, _ = liger_stub(entrypoint="apply_liger_kernel_to_gemma4_text")
    install_liger(monkeypatch, module)
    config = gemma(composed, **{"kernel.name": "liger"})

    with pytest.raises(axes.UnappliedAxis, match=r"apply_liger_kernel_to_gemma4_text") as refusal:
        axes.patch(config)

    assert "has no apply_liger_kernel_to_gemma4()" in str(refusal.value)


def test_the_refusal_reads_the_export_list_a_lazy_module_actually_answers(composed, monkeypatch):
    """`liger_kernel.transformers` is a lazy module: every `apply_liger_kernel_to_*`
    lives in `if TYPE_CHECKING:` and `__getattr__`, it defines no `__dir__`, and
    CPython's `dir(module)` returns `__dict__`'s keys. So the refusal built from
    `dir()` printed `it exports []` — a correction with nothing in it, on the one
    path that exists to carry one."""
    lazy = ModuleType("liger_kernel.transformers")
    lazy.__all__ = [
        "apply_liger_kernel_to_qwen3_5",
        "apply_liger_kernel_to_llama",
        "AutoLigerKernel",
    ]
    install_liger(monkeypatch, lazy)
    config = gemma(composed, **{"kernel.name": "liger"})

    with pytest.raises(axes.UnappliedAxis) as refusal:
        axes.patch(config)

    assert "['apply_liger_kernel_to_llama', 'apply_liger_kernel_to_qwen3_5']" in str(refusal.value)


def test_liger_records_an_entrypoint_for_every_architecture_under_test():
    """The suffix rule is what stands between this table and a guess, and the
    coverage assertion is what stops an architecture from quietly dropping out:
    a missing key reads as `liger` being inapplicable to that model, which is the
    state the deleted `LIGER_UNSUPPORTED` produced for gemma-4."""
    assert axes.LIGER_ENTRYPOINTS == {
        arch: f"{axes.LIGER_ENTRYPOINT_PREFIX}{arch}" for arch in axes.LIGER_ENTRYPOINTS
    }
    assert set(axes.LIGER_ENTRYPOINTS) == set(axes.VISION_PARAM_MARKERS)


def test_liger_on_an_architecture_with_no_recorded_entrypoint_is_refused(composed, monkeypatch):
    """Nothing recorded is not the same as unsupported, and the refusal has to say
    which one it is. The table is emptied of this arch rather than a fourth one
    invented: `model.arch` is a Literal of exactly the three under test."""
    monkeypatch.delitem(axes.LIGER_ENTRYPOINTS, "qwen3_vl")
    install_liger(monkeypatch, liger_stub(entrypoint=None)[0])
    config = bench(composed, **{"kernel.name": "liger"})

    with pytest.raises(axes.UnappliedAxis, match="no entrypoint for arch='qwen3_vl'"):
        axes.patch(config)


def test_liger_without_the_package_is_refused_as_an_axis_not_an_import_error(composed, monkeypatch):
    """`UnappliedAxis` rather than the raw `ImportError`: the audit counts a refusal
    as the axis declining a value it cannot put into effect, and anything else as a
    value that broke for an unrelated reason."""
    fla_environment(monkeypatch, fla=None)
    install_liger(monkeypatch, None)
    config = qwen35(composed, **{"kernel.name": "liger"})

    with pytest.raises(axes.UnappliedAxis, match="liger-kernel installed"):
        axes.patch(config)


def test_a_wrong_entrypoint_name_reports_what_liger_exports(composed, monkeypatch):
    """`LIGER_ENTRYPOINTS` holds a name that could not be checked against an
    installed package. When it is wrong the refusal has to carry the correction,
    or the first pod reports `AttributeError: apply_liger_kernel_to_qwen3_5`."""
    fla_environment(monkeypatch, fla=None)
    install_liger(monkeypatch, liger_stub(entrypoint=None)[0])
    config = qwen35(composed, **{"kernel.name": "liger"})

    with pytest.raises(axes.UnappliedAxis, match=r"apply_liger_kernel_to_llama"):
        axes.patch(config)


def test_fla_is_refused_where_transformers_takes_no_fla_path(composed, monkeypatch):
    """Only Qwen3.5 imports fla in transformers. Asking for it on the other two
    would name a library the model never enters, which `_capture_kernel` would then
    read back as `none`."""
    monkeypatch.setattr(axes, "_fla_fast_path", lambda: (True, ""))

    for config in (
        bench(composed, **{"kernel.name": "fla"}),
        gemma(composed, **{"kernel.name": "fla"}),
    ):
        with pytest.raises(axes.UnappliedAxis, match="no fla kernel path"):
            axes.patch(config)


def test_fla_without_its_fast_path_is_refused_rather_than_measured(composed, monkeypatch):
    """The silent-fallback case: transformers logs one line and runs the torch
    implementation for the layers that are 75% of this model, so a run that went
    ahead would publish the fallback's speed under fla's name."""
    monkeypatch.setattr(axes, "_fla_fast_path", lambda: (False, "fla not installed"))
    config = qwen35(composed, **{"kernel.name": "fla"})

    with pytest.raises(axes.UnappliedAxis, match="fla not installed"):
        axes.patch(config)


def fla_environment(monkeypatch, *, fla: str | None = "0.5.0", causal_conv1d=True, cuda=True):
    """The three things transformers looks at before it binds fla, and no more.

    The environment is stubbed rather than the predicate: `_fla_fast_path` is the
    entirety of what this axis decides, and a test that replaces it with a lambda
    observes nothing about fla at all — it watches a function with no side effects
    return a constant nobody reads.
    """
    installed = {"fla": fla is not None, "causal_conv1d": causal_conv1d}
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in installed:
            return object() if installed[name] else None
        return real_find_spec(name, *args, **kwargs)

    def version(distribution):
        if fla is not None and distribution in axes.FLA_DISTRIBUTIONS:
            return fla
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)


def test_fla_is_applied_when_the_environment_provides_it(composed, monkeypatch):
    fla_environment(monkeypatch)
    config = qwen35(composed, **{"kernel.name": "fla"})

    assert axes.patch(config) == ["kernel.name"]


def test_fla_below_the_version_transformers_requires_is_refused(composed, monkeypatch):
    """`is_flash_linear_attention_available()` carries a `>= 0.2.2` floor, and
    under it transformers binds nothing — the model comes out stock and the run
    dies at the capture, a long way from the version that caused it."""
    fla_environment(monkeypatch, fla="0.2.1")
    config = qwen35(composed, **{"kernel.name": "fla"})

    with pytest.raises(axes.UnappliedAxis, match="0.2.1 is below the 0.2.2"):
        axes.patch(config)


def test_fla_without_causal_conv1d_is_refused_though_its_classes_would_bind(composed, monkeypatch):
    """The two predicates are not one predicate. `is_flash_linear_attention_available`
    does not mention causal-conv1d; `is_causal_conv1d_available` is a separate gate
    behind a separate import, and only their conjunction is the fused Gated
    DeltaNet path. This environment is the gap between them: fla's classes are in
    the model, the fast path is not, and a run that went ahead would publish the
    torch gated delta rule under fla's name."""
    fla_environment(monkeypatch, causal_conv1d=False)
    config = qwen35(composed, **{"kernel.name": "fla"})

    assert axes._fla_binding() == (True, "")
    with pytest.raises(axes.UnappliedAxis, match="causal_conv1d not installed"):
        axes.patch(config)


def test_fla_without_fla_core_is_refused_though_the_version_check_passes(composed, monkeypatch):
    """The distribution that answers `importlib.metadata.version` is not the one
    that ships the code. `flash-linear-attention`'s wheel carries `fla/layers` and
    `fla/models` and no `fla/__init__.py`; `fla.ops.gated_delta_rule` and
    `fla.modules.FusedRMSNormGated` — the two symbols transformers imports — are in
    `fla-core`, which it declares as a dependency and which every env lock pins
    separately. An image holding only the first answers the version floor and then
    dies inside `modeling_qwen3_5`'s import, which is a broken image rather than a
    slow one, and the axis is what has to notice."""
    fla_environment(monkeypatch)
    real_version = importlib.metadata.version

    def without_fla_core(distribution):
        if distribution == axes.FLA_OPS_DISTRIBUTION:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return real_version(distribution)

    monkeypatch.setattr(importlib.metadata, "version", without_fla_core)
    config = qwen35(composed, **{"kernel.name": "fla"})

    with pytest.raises(axes.UnappliedAxis, match="fla-core is not installed"):
        axes.patch(config)


def test_the_fla_binding_needs_the_package_a_readable_version_and_cuda(monkeypatch):
    """The predicate itself, value by value, since everything above only sees its
    answer. Every arm is transformers': package, floor, and
    `is_torch_cuda_available()` — a CPU box with fla installed still runs torch."""
    fla_environment(monkeypatch)
    assert axes._fla_binding() == (True, "")

    fla_environment(monkeypatch, cuda=False)
    assert axes._fla_binding()[0] is False

    fla_environment(monkeypatch, fla=None)
    assert axes._fla_binding() == (False, "fla not installed")

    # Installed, but no distribution answers for it and importing it fails: the
    # floor cannot be checked, and an unchecked floor is not a satisfied one.
    fla_environment(monkeypatch, fla=None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    available, reason = axes._fla_binding()
    assert available is False
    assert "version cannot be read" in reason


def test_kernel_none_is_refused_where_transformers_binds_fla_anyway(composed, monkeypatch):
    """`none` is a claim about the model like every other value. On Qwen3.5 with
    fla installed the claim is false before the run starts — transformers binds the
    library while it imports the modelling module — so the refusal belongs here,
    where it can name the cause, rather than at `assert_matches` after a model has
    been built and the message is `requested 'none', applied 'fla'`."""
    fla_environment(monkeypatch)
    config = qwen35(composed)

    assert config.kernel.name == "none"
    with pytest.raises(axes.UnappliedAxis, match="kernel=none on arch=qwen3_5"):
        axes.patch(config)


def test_a_probe_of_the_same_run_is_not_refused(composed, monkeypatch):
    """The refusal above is a guard on reported numbers, and a probe reports none.
    Phase 0 is where "this image cannot give you kernel=none" is supposed to be
    discovered, so a probe has to reach the model load and let the capture record
    `requested none, applied fla` — refusing it here would remove the only run that
    can say so."""
    fla_environment(monkeypatch)
    config = qwen35(composed, **{"run.purpose": "probe"})

    assert axes.patch(config) == []


def test_kernel_none_is_refused_on_the_binding_and_not_on_the_fast_path(composed, monkeypatch):
    """Which of the two predicates decides this. The classes land in the model as
    soon as fla binds, whether or not causal-conv1d is there to complete the fused
    path, so a `none` guarded by `_fla_fast_path` would wave through exactly the
    environment where the model is made of fla and the label says otherwise."""
    fla_environment(monkeypatch, causal_conv1d=False)
    config = qwen35(composed)

    assert axes._fla_fast_path()[0] is False
    with pytest.raises(axes.UnappliedAxis, match="kernel=none on arch=qwen3_5"):
        axes.patch(config)


def test_kernel_none_stands_where_no_kernel_binds_itself(composed, monkeypatch):
    """The other side, or the refusal above would be a blanket one: without the
    package there is nothing to bind, and on an architecture transformers takes no
    fla path for, an image full of fla changes nothing."""
    fla_environment(monkeypatch, fla=None)
    assert axes.patch(qwen35(composed)) == []

    fla_environment(monkeypatch)
    assert axes.patch(bench(composed)) == []
    assert axes.patch(gemma(composed)) == []


def fla_that_cannot_be_imported(monkeypatch, exc: BaseException) -> None:
    """An image where fla is installed, publishes no distribution metadata, and
    raises on import.

    Not a hypothetical: fla imports triton at module scope, and triton raises on a
    box with no usable device — a `RuntimeError`, not an `ImportError`. The
    metadata half is what makes the import happen at all; with a readable version
    `_fla_version` answers before it gets there.
    """
    fla_environment(monkeypatch, fla=None)
    installed_elsewhere = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "fla" else installed_elsewhere(name, *a, **k),
    )
    real_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "fla":
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)


def test_a_kernel_none_run_whose_fla_import_fails_is_refused_as_an_axis(composed, monkeypatch):
    """The package is present and unimportable, which is neither of the two
    answers `_fla_binding` has. `_fla_version` caught `ImportError` alone, so the
    triton error left `patch` as itself — the run stopped either way, but the
    audit sorts a refused axis from an unrelated breakage by exception type, and
    this one filed as breakage. `UnappliedAxis` is not merely a rename: it is the
    only class here that says the axis declined, and it carries what raised."""
    fla_that_cannot_be_imported(monkeypatch, RuntimeError("triton: no CUDA device found"))
    config = qwen35(composed)

    assert config.kernel.name == "none"
    with pytest.raises(axes.UnappliedAxis, match="triton: no CUDA device found"):
        axes.patch(config)

    # The same environment reached from the other side. `kernel=fla` asks the same
    # predicate through `_fla_fast_path`, and an escape there would be the same
    # escape.
    with pytest.raises(axes.UnappliedAxis, match="triton: no CUDA device found"):
        axes.patch(qwen35(composed, **{"kernel.name": "fla"}))


def test_a_kernel_the_image_binds_over_is_refused_at_patch(composed, monkeypatch):
    """The image binds fla whatever the run asked for, so on this architecture
    `liger` is not liger: the model comes out of both libraries, `_capture_kernel`
    reads `mixed(fla,liger)`, and no setting names that. Consulting the binding
    only under `kernel=none` left this value to die at `assert_matches`, a whole
    model build away from the image that caused it."""
    fla_environment(monkeypatch)
    module, calls = liger_stub(entrypoint="apply_liger_kernel_to_qwen3_5")
    install_liger(monkeypatch, module)
    config = qwen35(composed, **{"kernel.name": "liger"})

    with pytest.raises(axes.UnappliedAxis, match=r"mixed\(fla,liger\)") as refusal:
        axes.patch(config)

    assert "kernel=liger on arch=qwen3_5" in str(refusal.value)
    # Refused before the library was touched. A patcher that ran and then raised
    # would leave transformers rebound for every later run in the process.
    assert calls == []


def test_a_quality_run_refuses_an_environment_bound_kernel_too(composed, monkeypatch):
    """The gate is `ENFORCED_PURPOSES`, not `timing`. A quality run publishes a
    number as much as a timing one does — the retrieval metric it exists to
    produce — and narrowing this to timing would let that metric be published
    against a model made of a library the config says is absent. Only a probe is
    exempt, because discovering exactly this is what a probe is for."""
    fla_environment(monkeypatch)

    for purpose in ("timing", "quality"):
        with pytest.raises(axes.UnappliedAxis, match="kernel=none on arch=qwen3_5"):
            axes.patch(qwen35(composed, **{"run.purpose": purpose}))

    assert axes.patch(qwen35(composed, **{"run.purpose": "probe"})) == []


def test_the_supersession_scan_covers_the_kernel_packages(composed, monkeypatch):
    """A second library patching over the first leaves modules built from the
    first one's classes, and those classes belong to a kernel package rather than
    to transformers. Scanning transformers alone would read this model as fully
    fla — every kernel module in it answers `fla` — while half of it was rebound
    to liger after those modules were built."""
    stock = patched_modelling_module(
        monkeypatch,
        replacement=LigerRMSNorm,
        module_name="fla.modules.fused_norm_gate",
        class_name="FusedRMSNormGated",
    )
    config = qwen35(composed, **{"kernel.name": "fla"})
    model = plain_model()
    model.add_module("built_before_the_second_patch", stock())

    state = capture(Built(model=model), config)

    assert axis(state, "kernel.name").applied == "partial(liger)"
    with pytest.raises(AppliedMismatch, match="kernel.name"):
        assert_matches(state, config)


def test_a_nested_class_is_not_read_as_a_superseded_module(composed, monkeypatch):
    """`Outer.Inner` and a module-level `Inner` are two names that share a
    `__name__`. The scan resolves a class's own name in its own module, so without
    the qualname guard a nested class is compared against whatever the module
    binds at top level — and one kernel library rebinding that top-level name
    would make every nested module in the model read as half-patched, turning a
    correct run into `partial(...)` and refusing it."""
    nested = patched_modelling_module(monkeypatch, replacement=LigerRMSNorm, class_name="RMSNorm")
    nested.__qualname__ = "Outer.RMSNorm"
    config = bench(composed)
    model = plain_model()
    model.add_module("nested", nested())

    state = capture(Built(model=model), config)

    assert axis(state, "kernel.name").applied == "none"
    assert axis(state, "kernel.name").detail["superseded"] == 0
    assert "kernel.name" not in [state_.axis for state_ in state.mismatched()]


def test_an_fla_below_the_floor_does_not_bind_kernel_none(composed, monkeypatch):
    """The version floor is half of what makes `kernel=none` refusable. Below
    0.2.2 transformers binds nothing, so the model really is stock and `none` is
    the honest label. A binding that asked only whether the package is present
    would refuse the one image on which this run is correct, and the refusal would
    read as the image shipping a kernel it does not bind.

    The floor version itself is the other half. transformers' predicate is `>=`,
    so the version equal to the floor binds — and a comparison one character off
    would refuse the exact release transformers accepts, on an image that really
    does build the model out of fla."""
    floor = ".".join(map(str, axes.FLA_MIN_VERSION))
    fla_environment(monkeypatch, fla="0.2.1")
    config = qwen35(composed)

    assert axes._environment_bound_kernel(config) == ""
    assert axes.patch(config) == []

    fla_environment(monkeypatch, fla=floor)

    assert axes._environment_bound_kernel(config) == "fla"
    assert axes.patch(qwen35(composed, **{"kernel.name": "fla"})) == ["kernel.name"]


def test_kernels_hub_is_dropped_from_every_place_that_could_still_offer_it():
    """PLAN.md decision 6, closed on both sides.

    Deleting `configs/kernel/kernels_hub.yaml` alone put the value out of reach of a
    *composed* run and nothing else: `scripts/bench.py::preflight` is handed configs
    built straight from the schema, and the Literal kept offering it. So the config
    file, the schema and the patcher table are asserted together — the two reasons
    the value is gone (the call site needs a model, and `envs/native`'s
    `kernels==0.16.0` sits on transformers' exclusive upper bound) hold everywhere,
    not only where Hydra composes.
    """
    assert not (CONFIGS / "kernel" / "kernels_hub.yaml").exists()
    assert sorted(p.stem for p in (CONFIGS / "kernel").glob("*.yaml")) == ["fla", "liger", "none"]
    assert "kernels_hub" not in get_args(KernelConfig.model_fields["name"].annotation)
    assert "kernels_hub" not in axes.KERNEL_PATCHERS


def test_a_kernel_the_schema_no_longer_offers_is_refused_rather_than_a_key_error(composed):
    """The value is gone from the schema, so nothing composes it — but `patch` is
    also reachable from a hand-built config (a stale plan JSON on a pod is exactly
    that). It has to refuse in this module's own vocabulary rather than die on a
    `KeyError` that reads as a broken axis."""
    config = bench(composed)
    object.__setattr__(config.kernel, "name", "kernels_hub")

    with pytest.raises(axes.UnappliedAxis) as refusal:
        axes.patch(config)

    assert "KERNEL_PATCHERS" in str(refusal.value)


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


class CheckpointedBlock(torch.nn.Module):
    """One matmul and one pointwise op — the two sides of the selective policy."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(4, 4))
        self.gradient_checkpointing = False

    def forward(self, x):
        return torch.sigmoid(x @ self.weight)


class Checkpointing(torch.nn.Module):
    """transformers' checkpointing mechanism, cut down to what the axis touches.

    `PreTrainedModel._set_gradient_checkpointing` (transformers 5.14.1) turns the
    kwargs into `functools.partial(torch.utils.checkpoint.checkpoint, **kwargs)`,
    stores that partial on every module carrying the flag, and those modules call
    it with their own `__call__`. Reproduced rather than mocked because the
    question `selective` turns on — which operators get recomputed in the backward
    pass — has no answer that does not run one.

    Three ways the request gets swallowed are settable here, because all three are
    silent and none raises. `honours=False` accepts the call and leaves the flag
    off — the run trains and never trades the activation memory away.
    `keeps_context=False` switches checkpointing on and drops the policy, which is
    a real `full` run reported under the `selective` label. `installs=` puts the
    model's own checkpointing on the block instead of the one asked for, which is
    what a framework image that already checkpoints does; the kwargs are accepted
    and never reach the block.
    """

    def __init__(self, honours: bool = True, keeps_context: bool = True, installs: object = None):
        super().__init__()
        self.block = CheckpointedBlock()
        self.honours = honours
        self.keeps_context = keeps_context
        self.installs = installs
        self.enable_calls: list = []
        self.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.enable_calls.append(gradient_checkpointing_kwargs)
        if not self.honours:
            return
        kwargs = dict(gradient_checkpointing_kwargs or {})
        if not self.keeps_context:
            kwargs.pop("context_fn", None)
        self.block.gradient_checkpointing = True
        self.block._gradient_checkpointing_func = self.installs or functools.partial(
            torch.utils.checkpoint.checkpoint, **kwargs
        )

    def forward(self, x):
        if self.block.gradient_checkpointing:
            return self.block._gradient_checkpointing_func(self.block.__call__, x)
        return self.block(x)


def test_checkpointing_goes_through_the_models_own_hook(composed):
    config = bench(composed, **{"train.gradient_checkpointing": "full"})
    model = Checkpointing()

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
    model = Checkpointing(honours=False)

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
    model = Checkpointing()

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


class ExecutedOps(TorchDispatchMode):
    """Counts the operators that actually execute.

    Selective checkpointing serves a saved operator's output from its cache rather
    than running it again, so an operator that stops appearing here during the
    backward pass is one that stopped being recomputed. This mode sits outside the
    checkpoint's own dispatch modes, which is why a cache hit is invisible to it.
    """

    def __init__(self):
        self.counts: collections.Counter = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(getattr(func, "overloadpacket", func))] += 1
        return func(*args, **(kwargs or {}))


def recomputed_ops(composed, mode: str, model=None) -> collections.Counter:
    """Operators executed during the backward pass of a run at this setting.

    `model` takes a stand-in that swallows the request some other way, so what a
    swallowed request actually recomputes is measured against the three settings
    rather than argued about.
    """
    config = bench(composed, **{"train.gradient_checkpointing": mode})
    built, _ = axes.assemble(model or Checkpointing(), config, CPU, framework="native")
    x = torch.rand(4, 4, requires_grad=True)
    with ExecutedOps() as counter:
        out = built.model(x)
        forward = counter.counts.copy()
        out.sum().backward()
    return counter.counts - forward


def test_the_selective_policy_saves_the_matmul_and_recomputes_the_rest(composed):
    """The three settings are three different backward passes, not three labels.

    Runs on CPU on purpose: the policy is a dispatch-level decision, so which
    operators re-execute is observable without a GPU. What is *not* observable
    here is the point of the axis — the activation memory it trades for that
    recompute, and what the trade costs in step time. Both are 측정 안 함.
    """
    none = recomputed_ops(composed, "none")
    full = recomputed_ops(composed, "full")
    selective = recomputed_ops(composed, "selective")

    # `none` is the floor, and it is not zero: the two gradient matmuls run in
    # every backward pass whatever this axis is set to. Nothing is *re*computed —
    # the forward's activations are still alive — so the pointwise op does not
    # reappear, and the matmul count here is what the other two are measured
    # against.
    assert none["aten.sigmoid"] == 0
    # Full checkpointing re-runs the whole block: one extra matmul, one sigmoid.
    assert full["aten.mm"] == none["aten.mm"] + 1
    assert full["aten.sigmoid"] == 1
    # Selective still recomputes the pointwise op — it is checkpointing, not an
    # elaborate `none` — but the matmul comes back from the cache. That single
    # operator is the entire difference between the two values.
    assert selective["aten.mm"] == none["aten.mm"]
    assert selective["aten.sigmoid"] == 1


def test_selective_asks_for_the_policy_and_reads_back_as_selective(composed):
    config = bench(composed, **{"train.gradient_checkpointing": "selective"})
    model = Checkpointing()

    built, names = axes.assemble(model, config, CPU, framework="native")

    assert "train.gradient_checkpointing" in names
    (kwargs,) = model.enable_calls
    # Not a preference: `context_fn` is only honoured by the non-reentrant
    # implementation.
    assert kwargs["use_reentrant"] is False
    assert kwargs["context_fn"].func is torch.utils.checkpoint.create_selective_checkpoint_contexts
    assert axis(capture(built, config), "train.gradient_checkpointing").applied == "selective"


def test_a_model_that_drops_the_policy_is_caught(composed):
    """The break. Checkpointing is on, the run trains, and the only thing that
    says the selective policy never reached the block is the partial next to the
    flag — where it reads back as plain `full`."""
    config = bench(composed, **{"train.gradient_checkpointing": "selective"})
    model = Checkpointing(keeps_context=False)

    built, _ = axes.assemble(model, config, CPU, framework="native")
    state = capture(built, config)

    assert model.enable_calls[0]["context_fn"], "the policy was asked for"
    assert axis(state, "train.gradient_checkpointing").applied == "full"
    with pytest.raises(AppliedMismatch, match="train.gradient_checkpointing"):
        assert_matches(state, config)


def test_a_selectively_checkpointed_model_does_not_pass_as_a_full_run(composed):
    """The other direction: `full` and `selective` recompute different operators,
    so a run of one reported under the other is a different measurement."""
    selective = bench(composed, **{"train.gradient_checkpointing": "selective"})
    built, _ = axes.assemble(Checkpointing(), selective, CPU, framework="native")

    state = capture(built, bench(composed, **{"train.gradient_checkpointing": "full"}))

    assert axis(state, "train.gradient_checkpointing").applied == "selective"
    with pytest.raises(AppliedMismatch, match="train.gradient_checkpointing"):
        assert_matches(state, bench(composed, **{"train.gradient_checkpointing": "full"}))


def test_a_context_this_cannot_name_is_undetermined_rather_than_full(composed):
    """A block checkpointed through somebody else's context is not `full`: which
    operators it recomputes is unknown, and unknown blocks a reportable run."""
    model = Checkpointing()
    model.block.gradient_checkpointing = True
    model.block._gradient_checkpointing_func = functools.partial(
        torch.utils.checkpoint.checkpoint, use_reentrant=False, context_fn=lambda: None
    )

    state = capture(Built(model=model), bench(composed))

    assert axis(state, "train.gradient_checkpointing").applied is None
    assert "context_fn" in axis(state, "train.gradient_checkpointing").detail["reason"]


def save_everything(ctx, op, *args, **kwargs):
    """Somebody else's selective policy, one end of the range: nothing recomputes."""
    return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE


def save_nothing(ctx, op, *args, **kwargs):
    """The other end: everything recomputes."""
    return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE


def selective_through(policy):
    """The checkpoint function a model carrying its own selective policy has.

    Identical in shape to the one `axes` asks transformers to build — same
    `checkpoint`, same `use_reentrant`, same `create_selective_checkpoint_contexts`
    — and different only in the policy inside it, which is the whole of what the
    two settings measure.
    """
    return functools.partial(
        torch.utils.checkpoint.checkpoint,
        use_reentrant=False,
        context_fn=functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy
        ),
    )


@pytest.mark.parametrize(
    ("policy", "equivalent"), [(save_everything, "none"), (save_nothing, "full")]
)
def test_a_foreign_selective_policy_does_not_pass_as_this_axis(composed, policy, equivalent):
    """The break. A framework image that turns on its own selective checkpointing
    leaves the block wrapped in a context built by torch's selective factory — so
    a probe that recognises the factory calls it `selective` — while its backward
    pass is one of the other two settings' backward passes. Measured here rather
    than argued: `save_everything` recomputes what `none` recomputes and
    `save_nothing` recomputes what `full` does, and neither recomputes what this
    axis does.
    """
    foreign = recomputed_ops(
        composed, "selective", Checkpointing(installs=selective_through(policy))
    )
    ours = recomputed_ops(composed, "selective")
    reference = recomputed_ops(composed, equivalent)
    # The two operators the policy decides between; `aten.detach` differs with the
    # wrapping rather than with what gets recomputed.
    interesting = lambda ops: (ops["aten.mm"], ops["aten.sigmoid"])  # noqa: E731

    assert interesting(foreign) == interesting(reference), "a run of the other setting"
    assert interesting(foreign) != interesting(ours), "and not a run of this one"

    config = bench(composed, **{"train.gradient_checkpointing": "selective"})
    built, _ = axes.assemble(
        Checkpointing(installs=selective_through(policy)), config, CPU, framework="native"
    )
    state = capture(built, config)

    assert axis(state, "train.gradient_checkpointing").applied is None
    assert policy.__qualname__ in axis(state, "train.gradient_checkpointing").detail["reason"]
    with pytest.raises(AppliedMismatch, match="train.gradient_checkpointing"):
        assert_matches(state, config)


def test_this_axiss_policy_is_found_when_torchs_parameter_is_bound_by_keyword(composed):
    """The other direction of the same check: `policy_fn_or_list` is torch 2.13.0's
    parameter name, and a caller is free to bind it by keyword. Reading only
    positional arguments would refuse a run that carries exactly this policy."""
    config = bench(composed, **{"train.gradient_checkpointing": "selective"})
    by_keyword = functools.partial(
        torch.utils.checkpoint.checkpoint,
        use_reentrant=False,
        context_fn=functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts,
            policy_fn_or_list=axes.selective_checkpoint_policy,
        ),
    )

    built, _ = axes.assemble(Checkpointing(installs=by_keyword), config, CPU, framework="native")

    assert axis(capture(built, config), "train.gradient_checkpointing").applied == "selective"


def test_a_reentrant_checkpoint_does_not_pass_as_a_full_run(composed):
    """`full` is not the flag plus any checkpoint. The reentrant implementation
    skips the recompute when nothing entering the block requires grad — which is
    what a frozen tower emits, and `freeze.*` is crossed with this axis — so it
    would report `none`'s backward pass under `full`'s label. Here the cost is
    larger than the label: the block's output leaves the autograd graph, so the
    run trains without the checkpointed parameters ever receiving a gradient.
    """
    config = bench(composed, **{"train.gradient_checkpointing": "full"})
    reentrant = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=True)
    built, _ = axes.assemble(Checkpointing(installs=reentrant), config, CPU, framework="native")
    asked_for, _ = axes.assemble(Checkpointing(), config, CPU, framework="native")

    frozen_output = torch.rand(4, 4)  # no requires_grad, as a frozen tower produces
    with pytest.warns(UserWarning, match="None of the inputs have requires_grad"):
        assert built.model(frozen_output).requires_grad is False
    assert asked_for.model(frozen_output).requires_grad is True

    state = capture(built, config)

    assert axis(state, "train.gradient_checkpointing").applied is None
    assert "use_reentrant" in axis(state, "train.gradient_checkpointing").detail["reason"]
    with pytest.raises(AppliedMismatch, match="train.gradient_checkpointing"):
        assert_matches(state, config)


def test_a_checkpoint_function_whose_keywords_cannot_be_read_is_not_full(composed):
    """The flag says checkpointing; the callable next to it is not torch's
    `checkpoint` under a partial, so nothing about what it recomputes is readable.
    This one recomputes nothing at all — it runs the block straight through — and
    `full` is what it read back as while the flag alone was the evidence."""
    config = bench(composed, **{"train.gradient_checkpointing": "full"})
    straight_through = lambda fn, x: fn(x)  # noqa: E731

    ops = recomputed_ops(composed, "full", Checkpointing(installs=straight_through))
    none = recomputed_ops(composed, "none")
    assert (ops["aten.mm"], ops["aten.sigmoid"]) == (none["aten.mm"], none["aten.sigmoid"])

    built, _ = axes.assemble(
        Checkpointing(installs=straight_through), config, CPU, framework="native"
    )
    state = capture(built, config)

    assert axis(state, "train.gradient_checkpointing").applied is None
    assert "functools.partial" in axis(state, "train.gradient_checkpointing").detail["reason"]
    with pytest.raises(AppliedMismatch, match="train.gradient_checkpointing"):
        assert_matches(state, config)


def test_a_lookalike_checkpoint_function_is_not_read_as_full(composed):
    """An object carrying `.func` and `.keywords` is not a checkpoint.

    The refusal above names `functools.partial`, and the check behind it was
    duck-typing on those two attributes — so anything shaped like a partial had its
    `keywords` read as the run's evidence of what the backward pass re-runs, while
    the callable installed on the block is whatever else the object is. The reason
    said more than the check did, and this stand-in read back as a clean `full`.
    """
    config = bench(composed, **{"train.gradient_checkpointing": "full"})
    model = Checkpointing()
    model.block.gradient_checkpointing = True
    model.block._gradient_checkpointing_func = SimpleNamespace(
        func=torch.utils.checkpoint.checkpoint, keywords={"use_reentrant": False}
    )

    state = capture(Built(model=model), config)

    assert axis(state, "train.gradient_checkpointing").applied is None
    assert "functools.partial" in axis(state, "train.gradient_checkpointing").detail["reason"]
    with pytest.raises(AppliedMismatch, match="train.gradient_checkpointing"):
        assert_matches(state, config)


def test_a_lookalike_selective_context_is_not_read_as_selective(composed):
    """The same hole one line further in: `context_fn` was duck-typed too, so a
    stand-in carrying `.func` and `.args` had this axis's policy read off it while
    the context the block enters is built by something else entirely."""
    config = bench(composed, **{"train.gradient_checkpointing": "selective"})
    model = Checkpointing()
    model.block.gradient_checkpointing = True
    model.block._gradient_checkpointing_func = functools.partial(
        torch.utils.checkpoint.checkpoint,
        use_reentrant=False,
        context_fn=SimpleNamespace(
            func=torch.utils.checkpoint.create_selective_checkpoint_contexts,
            args=(axes.selective_checkpoint_policy,),
        ),
    )

    state = capture(Built(model=model), config)

    assert axis(state, "train.gradient_checkpointing").applied is None
    assert (
        "create_selective_checkpoint_contexts"
        in axis(state, "train.gradient_checkpointing").detail["reason"]
    )
    with pytest.raises(AppliedMismatch, match="train.gradient_checkpointing"):
        assert_matches(state, config)


def test_the_policy_honours_every_operator_on_its_own_save_list():
    """The policy against its own list, operator by operator.

    `SELECTIVE_CHECKPOINT_SAVED_OPS` is transcribed from torch's
    `compute_intensive_ops`, and everything else in this file reaches the policy
    through a stand-in block whose forward pass contains one `aten.mm` and one
    pointwise op. So a policy that recomputed `bmm`, `addmm`, `_scaled_mm` or
    either SDPA packet — attention and biased linear layers, which is most of what
    selective is for on a GPU — left every other test in this file green.

    Overloads rather than the packets themselves, because an overload is what the
    dispatcher hands the policy at run time and `overloadpacket` is the reduction
    the policy performs on it. Passing the packet would test the list against
    itself.

    What this cannot reach is the GPU half of the list. On CPU, scaled dot product
    attention dispatches as `_scaled_dot_product_flash_attention_for_cpu`, which is
    not on the list at all — see `docs/methodology.md` §gradient_checkpointing.
    """
    saved = torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
    recomputed = torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

    assert axes.SELECTIVE_CHECKPOINT_SAVED_OPS, "an empty save list saves nothing"
    for packet in axes.SELECTIVE_CHECKPOINT_SAVED_OPS:
        overloads = [getattr(packet, name) for name in packet.overloads()]
        assert overloads, f"{packet} offers no overload to dispatch"
        for overload in overloads:
            assert axes.selective_checkpoint_policy(None, overload) is saved, (
                f"{overload} is on the save list and would be recomputed"
            )

    # The other direction, so a policy that returns MUST_SAVE for everything —
    # which recomputes nothing and is the backward pass of `none` — cannot satisfy
    # the loop above. `sigmoid` is the pointwise operator the stand-in block runs.
    for absent in (torch.ops.aten.sigmoid.default, torch.ops.aten.add.Tensor):
        assert axes.selective_checkpoint_policy(None, absent) is recomputed


def test_a_real_transformers_model_reads_back_at_all_three_settings(composed):
    """The probe against the thing it reads, rather than against a stand-in.

    Everything above builds `Checkpointing`, which sets `_gradient_checkpointing_func`
    because this file says transformers does. That is the test and the probe
    agreeing with each other. Here a real `PreTrainedModel` sets it: architecture
    only, weights initialised in-process, nothing fetched.

    Llama rather than one of the three models under study — those need their
    checkpoints, and this asks about the transformers hook, which is on
    `PreTrainedModel` rather than on any architecture. That the three do carry the
    hook is 측정 안 함 here.
    """
    from transformers import AutoConfig, AutoModel

    architecture = AutoConfig.for_model(
        "llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
    )
    for mode in ("none", "full", "selective"):
        config = bench(composed, **{"train.gradient_checkpointing": mode})
        model = AutoModel.from_config(architecture)

        built, _ = axes.assemble(model, config, CPU, framework="native")

        state = capture(built, config)
        # Not `assert_matches`: on CPU `precision.name` and `optim.name` mismatch
        # whatever this axis does (docs/CONTRACTS.md §6), so a whole-state assert
        # here would pass for the wrong reason and fail for one too.
        assert axis(state, "train.gradient_checkpointing").applied == mode, mode


def test_a_real_transformers_model_recomputes_differently_at_all_three_settings(composed):
    """The three settings are three backward passes on a real `PreTrainedModel`,
    not only three attributes on one.

    The test above reads the attribute the probe reads, so on its own it certifies
    that the request reached the model and nothing about what the model then does
    with it. Here the model runs, and the operators the backward pass re-executes
    are counted through the same dispatch mode the stand-in is measured with. The
    axis's claim — matmuls come back from the cache, everything else is recomputed
    — is what separates the counts.

    `model.train()` is not incidental: transformers checkpoints under
    `if self.gradient_checkpointing and self.training`, so an eval-mode model
    recomputes nothing at any setting and all three counts would agree.

    What is still 측정 안 함 here is the trade itself: the activation memory saved
    and the step time paid. Both need the GPU pods.
    """
    from transformers import AutoConfig, AutoModel

    architecture = AutoConfig.for_model(
        "llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
    )
    backward: dict[str, collections.Counter] = {}
    for mode in ("none", "full", "selective"):
        config = bench(composed, **{"train.gradient_checkpointing": mode})
        torch.manual_seed(0)
        model = AutoModel.from_config(architecture)
        built, _ = axes.assemble(model, config, CPU, framework="native")
        built.model.train()
        tokens = torch.randint(0, 64, (2, 8))
        with ExecutedOps() as counter:
            out = built.model(input_ids=tokens).last_hidden_state
            forward = counter.counts.copy()
            out.sum().backward()
        backward[mode] = counter.counts - forward

    none, full, selective = backward["none"], backward["full"], backward["selective"]
    # Llama's MLP is gated: `silu` is the pointwise op that reappears only when a
    # block is re-run, and `mm` is what the policy holds back.
    assert none["aten.silu"] == 0, "nothing is recomputed without checkpointing"
    assert full["aten.silu"] > 0, "full checkpointing re-runs the block"
    assert full["aten.mm"] > none["aten.mm"], "including its matmuls"
    assert selective["aten.silu"] == full["aten.silu"], "selective still recomputes pointwise"
    assert selective["aten.mm"] == none["aten.mm"], "and serves every matmul from the cache"


def test_the_saved_operators_are_torchs_own_compute_intensive_list():
    """The policy is the axis, so the list is pinned to where it came from.

    `torch._functorch.partitioners.get_default_op_list().compute_intensive_ops` is
    the classification the min-cut partitioner uses; taking it whole is what makes
    this policy a transcription rather than a choice. Private, so it is asserted
    here rather than imported into `axes.py`: a framework image whose torch moved
    it would break a measured run instead of this test.
    """
    from torch._functorch.partitioners import get_default_op_list

    assert set(axes.SELECTIVE_CHECKPOINT_SAVED_OPS) == set(
        get_default_op_list().compute_intensive_ops
    )


def test_the_policy_saves_a_matmul_and_recomputes_a_pointwise_op():
    """The policy function itself, at the granularity torch dispatches at."""
    policy = axes.selective_checkpoint_policy
    save = torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE

    assert policy(None, torch.ops.aten.mm.default) == save
    # An overload other than `default` is the same operator, and the packet-level
    # comparison is what keeps it covered.
    assert policy(None, torch.ops.aten.mm.out) == save
    assert policy(None, torch.ops.aten.sigmoid.default) != save


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


def four_bit_lora_model(as_dict: bool = False, **recipe):
    """A LoRA model over a 4-bit base, quantised by the recipe given.

    Both markers are set where a real load leaves them: `is_loaded_in_4bit` on the
    model (`quantizers/quantizer_bnb_4bit.py`) and the recipe on `model.config`
    (`quantizers/auto.py`, which assigns it before the quantizer is even built —
    so a model carrying the flag carries the recipe too). `as_dict` is the other
    spelling `model.config` can hold, the one a pre-quantised checkpoint declares
    in its own config.json.
    """
    model = plain_model()
    model.peft_config = {"default": SimpleNamespace(peft_type="PeftType.LORA")}
    model.is_loaded_in_4bit = True
    values = {**axes.QLORA_4BIT, **recipe}
    model.config.quantization_config = values if as_dict else SimpleNamespace(**values)
    return model


@pytest.mark.parametrize("as_dict", [False, True], ids=["config_object", "config_dict"])
def test_qlora_is_told_apart_by_the_quantised_base(composed, as_dict):
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32})

    state = capture(Built(model=four_bit_lora_model(as_dict)), config)

    assert axis(state, "peft.mode").applied == "qlora"
    # And the recipe reaches the result file. `build_record` publishes the config
    # dump and `applied.to_dict()`, and `QLORA_4BIT` is in neither — it is a module
    # constant, not a schema field (docs/CONTRACTS.md §5). Without this, a result
    # says "4-bit" and no audit afterwards can say which 4-bit it was.
    assert axis(state, "peft.mode").detail["base_quantisation"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        # A string, because the dtype has to survive `json.dumps` into the record.
        "bnb_4bit_compute_dtype": "bfloat16",
    }


@pytest.mark.parametrize(
    "wrong",
    [
        {"bnb_4bit_quant_type": "fp4"},
        {"bnb_4bit_use_double_quant": False},
        {"bnb_4bit_compute_dtype": torch.float16},
    ],
    ids=["fp4", "no_double_quant", "fp16_compute"],
)
def test_a_four_bit_base_on_another_recipe_is_not_qlora(composed, wrong):
    """The subtler break, and the one a "is it 4-bit at all" check waves through.

    Each of these loads 4-bit weights, so `is_loaded_in_4bit` is True and the
    adapter is the same LoRA — every marker the axis used to be read from agrees.
    They are different techniques: fp4 and nf4 are different quantisers, double
    quantisation is a second pass over the constants, and an fp16 compute dtype is
    a precision no axis asked for. Measuring one and printing QLoRA's name over it
    is this module's whole failure mode, so the applied value is one nothing can
    match and the run stops.
    """
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32, "run.purpose": "timing"})

    state = capture(Built(model=four_bit_lora_model(**wrong)), config)

    assert axis(state, "peft.mode").applied != "qlora"
    # Named in the value, so the failure says which knob rather than "mismatch".
    assert next(iter(wrong)) in axis(state, "peft.mode").applied
    with pytest.raises(AppliedMismatch, match="peft.mode"):
        assert_matches(state, config)


def test_a_base_that_says_it_is_four_bit_without_saying_how_is_undetermined(composed):
    """`is_loaded_in_4bit` with no recipe on the config. transformers never
    produces this, but a framework adapter setting the flag its own way would, and
    the fail-safe direction is the one this module is built on: not readable is not
    "the usual recipe"."""
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32, "run.purpose": "timing"})
    model = four_bit_lora_model()
    del model.config.quantization_config

    state = capture(Built(model=model), config)

    assert axis(state, "peft.mode").applied == "qlora(recipe=unreadable)"
    with pytest.raises(AppliedMismatch, match="peft.mode"):
        assert_matches(state, config)


def test_a_qlora_request_over_an_unquantised_base_blocks_the_run(composed):
    """The break for the line above, and the outcome `load_kwargs` exists to make
    impossible: the adapter is the same object either way, so a run that asked for
    qlora and got a bf16 base is indistinguishable by adapter type — and its step
    time is what the study would print for 4-bit. `purpose` is spelled out because
    what is being asserted is that the reportable run is the one that dies."""
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32, "run.purpose": "timing"})
    model = plain_model()
    model.peft_config = {"default": SimpleNamespace(peft_type="PeftType.LORA")}

    state = capture(Built(model=model), config)

    assert axis(state, "peft.mode").applied == "lora"
    with pytest.raises(AppliedMismatch, match="peft.mode"):
        assert_matches(state, config)


def test_lora_attaches_a_real_adapter_and_reads_back_as_lora(composed):
    """The pair for this axis, against real peft rather than a stand-in.

    Silently training every parameter under a LoRA label is the headline comparison
    of this study reported backwards, so what is asserted is not that `assemble`
    returned something, but that the thing it returned has adapter parameters and
    a frozen base."""
    config = bench(composed, **{"peft.mode": "lora", "peft.r": 8, "peft.alpha": 16})

    built, applied = axes.assemble(plain_model(), config, CPU, framework="native")

    assert "peft.mode" in applied
    assert axis(capture(built, config), "peft.mode").applied == "lora"
    trainable = [n for n, p in built.model.named_parameters() if p.requires_grad]
    assert trainable, "an adapter with no trainable parameter trains nothing"
    assert all("lora_" in name for name in trainable), trainable
    # The wrapper, not the base that `get_peft_model` mutated on its way past.
    # It injects lora layers into the base *and* sets `peft_config` on it, so every
    # assertion above holds for the base too — discarding the returned wrapper left
    # the whole suite green until this line existed. `assemble` returns a model for
    # exactly this reason (docs/CONTRACTS.md §2).
    assert type(built.model).__module__.startswith("peft"), type(built.model)


def quantised_model() -> torch.nn.Module:
    """A model as `from_pretrained` returns one under a `BitsAndBytesConfig`.

    The flag rather than real 4-bit weights: bitsandbytes quantises on CUDA and
    there is none here, so what is under test is the gate rather than the weights.
    `is_loaded_in_4bit` is one of the two readings `applied._capture_peft` uses to
    tell `qlora` from `lora`, which is why `_base_is_quantised` uses both.
    """
    model = plain_model()
    model.is_loaded_in_4bit = True
    return model


def test_qlora_attaches_the_adapter_to_a_base_that_arrived_quantised(composed):
    """The two halves are `load_kwargs`' `BitsAndBytesConfig` and this call, and
    what this site adds is the check that the first one happened."""
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32})

    built, names = axes.assemble(quantised_model(), config, CPU, framework="native")

    assert "peft.mode" in names
    assert type(built.model).__module__.startswith("peft")
    trainable = [n for n, p in built.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable), trainable


def test_qlora_over_an_unquantised_base_is_refused_at_the_adapter(composed):
    """The realistic failure: the quantisation is requested in `load_kwargs`, a
    caller that skipped it gets no error from `from_pretrained`, and plain LoRA
    would then be measured under the QLoRA label. Weights already materialised in
    bf16 cannot be quantised here, so refusing is the whole of what this site can
    do about it."""
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32})

    with pytest.raises(axes.UnappliedAxis, match="carries no quantisation"):
        axes.assemble(plain_model(), config, CPU, framework="native")


def test_qlora_does_not_run_the_preamble_that_would_turn_on_another_axis(composed, monkeypatch):
    """`prepare_model_for_kbit_training` is the documented QLoRA preamble and it is
    deliberately not called: it enables gradient checkpointing by default, which is
    a separate axis here, and it upcasts the norms to fp32, which
    `applied._capture_precision` reads as `mixed(bf16,fp32)` against a bf16
    request. Either one turns a qlora cell into a measurement of something else, so
    the call is refused rather than left to be noticed."""
    import peft

    called = []
    monkeypatch.setattr(
        peft, "prepare_model_for_kbit_training", lambda *a, **k: called.append(a) or a[0]
    )
    config = bench(composed, **{"peft.mode": "qlora", "peft.r": 32})

    built, names = axes.assemble(quantised_model(), config, CPU, framework="native")

    assert called == []
    assert "train.gradient_checkpointing" not in names
    assert not any(
        getattr(module, "gradient_checkpointing", False)
        for _, module in built.model.named_modules()
    )


def test_an_adapter_freezes_the_base_whatever_the_freeze_axes_did(composed):
    """The measurement the schema validator rests on.

    `get_peft_model` sets `requires_grad=False` on every base parameter, and the
    result is the same whether or not a freeze axis ran first — so `freeze.ple=true`
    and `freeze.ple=false` build the same model under an adapter. That is why the
    combination is refused at config time rather than defined: there are not two
    states to tell apart. If a future peft stops doing this, this test fails and
    the validator is what has to be revisited."""
    config = bench(composed, **{"peft.mode": "lora", "peft.r": 8})
    pre_frozen = plain_model()
    for param in pre_frozen.parameters():
        param.requires_grad_(False)

    a, _ = axes.assemble(plain_model(), config, CPU, framework="native")
    b, _ = axes.assemble(pre_frozen, config, CPU, framework="native")

    def grads(built):
        return {n: p.requires_grad for n, p in built.model.named_parameters()}

    assert grads(a) == grads(b)
    assert not any(g for n, g in grads(a).items() if "lora_" not in n)


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


def world_of(monkeypatch, size: int | None):
    """A process group of `size` ranks, or none at all when `size` is None.

    Stubbed rather than started: a real rendezvous would need a second process,
    and what is under test is which world each strategy accepts.
    """
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: size is not None)
    monkeypatch.setattr(dist, "get_world_size", lambda: size)


@pytest.mark.parametrize("strategy", ["ddp", "fsdp2", "zero2", "zero3"])
def test_a_sharding_strategy_without_a_process_group_is_refused(composed, strategy):
    """Nothing here starts one. A rendezvous invented by the harness would decide
    the world size the run is measured at, and the world size is the setting."""
    config = bench(composed, **{"parallel.strategy": strategy})

    with pytest.raises(axes.UnappliedAxis, match="needs an initialised process group"):
        axes.assemble(plain_model(), config, CPU, framework="native")


@pytest.mark.parametrize("strategy", ["ddp", "fsdp2", "zero2", "zero3"])
def test_a_sharding_strategy_over_one_rank_is_the_single_gpu_run(composed, monkeypatch, strategy):
    """The break for the test above: a gate that only asked whether a group exists
    passes under `torchrun --nproc_per_node=1`, and every one of these values then
    measures one GPU and publishes it under a distributed label. `_gather_with_grad`
    already refuses `world_size=1` on these grounds; this is that rule for the
    wrappers."""
    world_of(monkeypatch, 1)
    config = bench(composed, **{"parallel.strategy": strategy})

    with pytest.raises(axes.UnappliedAxis, match="world_size=1"):
        axes.assemble(plain_model(), config, CPU, framework="native")


def ddp_recorder(monkeypatch) -> dict:
    """The DDP stand-in, keeping the argument instead of discarding it.

    The wrapper is not what these tests are about — what is, is the one argument
    `_parallel` derives — and the lambda that used to stand in for it took
    `device_ids` only to drop it, so every derivation read the same.
    """
    seen: dict = {}

    def stub(module, device_ids=None):
        seen["device_ids"] = device_ids
        return DistributedDataParallel(module)

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", stub)
    return seen


def test_ddp_wraps_the_model_and_reads_back_as_ddp(composed, monkeypatch):
    """The whole pair in one run: `_parallel` builds the wrapper and
    `applied._capture_parallel_strategy` finds it by class name. The wrapper is
    torch's own — a stand-in named `DistributedDataParallel` would satisfy the
    capture while proving nothing about what this module builds."""
    world_of(monkeypatch, 2)
    config = bench(composed, **{"parallel.strategy": "ddp"})
    ddp_recorder(monkeypatch)

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")

    assert "parallel.strategy" in names
    assert axis(capture(built, config), "parallel.strategy").applied == "ddp"


class IndexedParameter(torch.nn.Parameter):
    """A parameter that reports a device with an index, on a host that has none.

    Only the report is faked: `_parallel` reads `p.device` and nothing else, so
    this is the input to the derivation rather than a stand-in for the derivation.
    CPU tensors carry `index=None` however they are allocated
    (`torch.zeros(2, device=torch.device("cpu", 0)).device.index is None`), which
    is why the indexed half cannot be reached with a real device here.
    """

    INDEX = 3

    @property
    def device(self):
        return torch.device("cuda", self.INDEX)


def test_the_ddp_replica_is_pinned_to_the_device_its_parameters_are_on(composed, monkeypatch):
    """A guessed index would put every rank's replica on device 0 — the harm the
    comment at the derivation claims to prevent, and until now the only thing
    saying so. `ids = [0]` left the whole suite green."""
    world_of(monkeypatch, 2)
    seen = ddp_recorder(monkeypatch)
    model = plain_model()
    model[0].weight = IndexedParameter(model[0].weight.data)
    config = bench(composed, **{"parallel.strategy": "ddp"})

    axes.assemble(model, config, CPU, framework="native")

    assert seen["device_ids"] == [IndexedParameter.INDEX]


def test_a_cpu_replica_is_wrapped_without_device_ids(composed, monkeypatch):
    """The other half, and the one this host can answer for real. torch refuses
    `device_ids` for a module on CPU — `self.device_type == "cpu"` with a truthy
    `device_ids` is a `ValueError`
    (`torch/nn/parallel/distributed.py:932-946`, torch 2.13.0, read in this
    worktree's install) — so `[None]`, which is what dropping the index check
    produces, would not wrap a CPU run at all."""
    world_of(monkeypatch, 2)
    seen = ddp_recorder(monkeypatch)
    config = bench(composed, **{"parallel.strategy": "ddp"})

    assert next(plain_model().parameters()).device.index is None

    axes.assemble(plain_model(), config, CPU, framework="native")

    assert seen["device_ids"] is None


def test_fsdp2_shards_in_place_and_the_capture_reads_it_off_the_mro(composed, monkeypatch):
    """`fully_shard` mutates: the module keeps its identity and gains `FSDPModule`
    in its MRO under a class renamed `FSDP<original>`
    (`torch/distributed/fsdp/_fully_shard/_fsdp_init.py:421-430`, torch 2.13.0,
    read in this worktree's install). The stub reproduces both bases in that order.

    Nothing named `FullyShardedDataParallel` exists anywhere in the result — that
    is FSDP1's wrapper class, and matching it was how the capture missed FSDP2
    entirely and would have labelled an FSDP1 run `fsdp2`. What the capture reads
    now is the MRO, which is what the API guarantees; the class *name* is whatever
    the checkpoint's class was with a prefix.
    """
    from torch.distributed.fsdp import FSDPModule

    world_of(monkeypatch, 2)
    sharded = []

    def fully_shard(module, **kwargs):
        module.__class__ = type(f"FSDP{type(module).__name__}", (FSDPModule, type(module)), {})
        sharded.append(module)
        return module

    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", fully_shard)
    config = bench(composed, **{"parallel.strategy": "fsdp2"})

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")

    assert "parallel.strategy" in names
    assert sharded and type(built.model).__name__.startswith("FSDP")
    assert "FullyShardedDataParallel" not in {
        type(m).__name__ for _, m in built.model.named_modules()
    }
    assert axis(capture(built, config), "parallel.strategy").applied == "fsdp2"


def test_fsdp1s_wrapper_is_named_rather_than_reported_as_the_axis_value(composed):
    """FSDP1 and FSDP2 shard differently and this study measures one of them. A
    model that arrived through the old API is given a name no setting carries, so
    `assert_matches` stops it — the table used to map this class straight onto
    `fsdp2`, which would have published FSDP1's step time under FSDP2's label."""

    class FullyShardedDataParallel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

    built = Built(model=FullyShardedDataParallel(plain_model()))
    config = bench(composed, **{"parallel.strategy": "fsdp2"})

    assert axis(capture(built, config), "parallel.strategy").applied == "fsdp1"


# --- parallel.strategy=zero2/zero3 and train.offload -------------------------
#
# deepspeed is not installed here and is sdist-only in every env lock, so it is
# built on the pod. What is testable is the config this module hands `initialize`
# and what it keeps on `Built`; whether a real engine answers the readers
# `applied.py` calls is 확인 안 함 and is written up in `.plans/notes/axes.md`.


class DeepSpeedEngine(torch.nn.Module):
    """Named for what `applied.PARALLEL_WRAPPERS` matches on, and answering the
    three readers `applied.zero_stage`/`offload_targets` call.

    `forward`, `backward` and `step` are the engine's own step surface. The
    forward delegates so a run can be driven through this object; the other two
    only record, because what they log is the question — nothing here knows what
    deepspeed's do, and this file imports no deepspeed to ask.
    """

    def __init__(self, module, config):
        super().__init__()
        self.module = module
        self.ds_config = config
        self.driven: list[str] = []
        self.forwards = 0

    def forward(self, *args, **kwargs):
        self.forwards += 1
        return self.module(*args, **kwargs)

    def backward(self, loss):
        self.driven.append("backward")
        loss.backward()

    def step(self):
        self.driven.append("step")

    def zero_optimization_stage(self):
        return self.ds_config["zero_optimization"]["stage"]

    def zero_offload_optimizer(self):
        return self.ds_config["zero_optimization"].get("offload_optimizer")

    def zero_offload_param(self):
        return self.ds_config["zero_optimization"].get("offload_param")


def install_deepspeed(monkeypatch):
    """A stand-in for `deepspeed`, plus the log of what `initialize` was handed."""
    calls = []

    def initialize(model=None, optimizer=None, config=None, **kwargs):
        calls.append({"model": model, "optimizer": optimizer, "config": config})
        return DeepSpeedEngine(model, config), object(), None, None

    module = ModuleType("deepspeed")
    module.initialize = initialize
    monkeypatch.setitem(sys.modules, "deepspeed", module)
    return calls


@pytest.mark.parametrize(
    "strategy,offload,stage,sections",
    [
        ("zero2", "none", 2, {}),
        ("zero3", "optimizer", 3, {"offload_optimizer": {"device": "cpu"}}),
        ("zero3", "param", 3, {"offload_param": {"device": "cpu"}}),
        (
            "zero3",
            "both",
            3,
            {"offload_optimizer": {"device": "cpu"}, "offload_param": {"device": "cpu"}},
        ),
    ],
)
def test_zero_hands_deepspeed_the_stage_and_the_offload_sections(
    composed, monkeypatch, strategy, offload, stage, sections
):
    """Both axes are one call. The sections are the two the engine answers about —
    `zero_offload_optimizer()` and `zero_offload_param()` read
    `zero_config.offload_optimizer` and `.offload_param` — so the request and the
    read-back are the same two names."""
    world_of(monkeypatch, 2)
    calls = install_deepspeed(monkeypatch)
    config = bench(composed, **{"parallel.strategy": strategy, "train.offload": offload})

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")

    handed = calls[0]["config"]
    assert handed["zero_optimization"] == {"stage": stage, **sections}
    assert handed["train_micro_batch_size_per_gpu"] == config.train.batch_size
    assert "parallel.strategy" in names
    assert ("train.offload" in names) == (offload != "none")

    state = capture(built, config)
    assert axis(state, "parallel.strategy").applied == strategy
    assert axis(state, "train.offload").applied == offload


def test_the_batch_deepspeed_is_told_about_is_the_one_the_loop_feeds(composed, monkeypatch):
    """`grad_accum > 1`, which no composed config in this repository sets: with it
    at 1 the micro batch and the total batch are the same number, so the assertion
    above holds equally against `batch_size * grad_accum` — and deepspeed would
    derive a `train_batch_size` `grad_accum * world_size` too large, partitioning
    and accumulating for a workload no config asked for.

    The other two keys were asserted by nothing at all. `gradient_accumulation_steps`
    is what tells deepspeed where a step boundary is, and the loop feeds
    `config.train.grad_accum` micro-batches between `optimizer.step()` calls
    (`scripts/bench.py`); hardcoding 1 makes every micro-batch a step to the engine
    and none to the harness. `bf16` is the numeric regime, and dropping it leaves a
    run labelled `precision=bf16` on deepspeed's default — which
    `applied._capture_precision` cannot see, because it reads the weights' dtype
    and the weights are bf16 either way.
    """
    world_of(monkeypatch, 2)
    calls = install_deepspeed(monkeypatch)
    config = bench(composed, **{"parallel.strategy": "zero2", "train.grad_accum": 4})

    axes.assemble(plain_model(), config, CPU, framework="native")

    handed = calls[0]["config"]
    assert config.train.grad_accum == 4
    assert handed["train_micro_batch_size_per_gpu"] == config.train.batch_size
    assert handed["train_micro_batch_size_per_gpu"] != (
        config.train.batch_size * config.train.grad_accum
    )
    assert handed["gradient_accumulation_steps"] == config.train.grad_accum
    assert handed["bf16"] == {"enabled": True}
    # The whole dict, so a key added here has to be said out loud: `optimizer` and
    # `training_data` are absent on purpose and the two tests below say why.
    assert set(handed) == {
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "zero_optimization",
        "bf16",
    }


def test_the_optimizer_on_built_is_the_one_deepspeed_was_given(composed, monkeypatch):
    """deepspeed returns its own wrapper around it, and recording that would report
    `optim.name` as the wrapper's class — blocking every ZeRO run on an axis that
    has nothing to do with ZeRO. What the wrapper does with this instance is
    확인 안 함 and is the test below's subject; what is settled here is only which
    of the two `optim.name` is read off."""
    world_of(monkeypatch, 2)
    calls = install_deepspeed(monkeypatch)
    config = bench(composed, **{"parallel.strategy": "zero2"})

    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")

    assert built.optimizer is calls[0]["optimizer"]
    assert axis(capture(built, config), "optim.name").applied == "adamw_unfused"


def test_the_measured_step_never_drives_the_engine(composed, monkeypatch):
    """The gap a ZeRO row rests on, frozen so it cannot be claimed shut by prose.

    The real measured loop is run here — `scripts/bench.py::train`, the same entry
    `tests/test_smoke_cpu.py` drives — over a `Built` that `assemble` produced with
    both ZeRO axes on. The engine's forward is used, and its `backward`/`step` are
    not: the loop issues `loss.backward()` and `built.optimizer.step()` on the
    instance handed to `initialize`. So `parallel.strategy=zero2` and
    `train.offload=optimizer` read back as applied off the engine's config while
    the step that was timed is a plain single-process one.

    That is why `axes._deepspeed` states no delegation. Wiring the loop to the
    engine belongs to `scripts/bench.py`, and when it lands this test inverts —
    `driven` becomes `["backward", "step"] * steps` — and the caveat in the
    docstring comes off with it. Until then the number a pod would publish under
    a ZeRO label is 확인 안 함, and nothing but this says so.
    """
    from .test_smoke_cpu import TinyEmbedder, bench_entry, micro_batch

    world_of(monkeypatch, 2)
    install_deepspeed(monkeypatch)
    config = bench(
        composed,
        **{
            "run.purpose": "probe",
            "train.steps": 3,
            "train.warmup_discard_steps": 1,
            "train.batch_size": 2,
            "data.limit": 8,
            "parallel.strategy": "zero2",
            "train.offload": "optimizer",
        },
    )

    built, _ = axes.assemble(TinyEmbedder(), config, CPU, framework="native")
    state = capture(built, config)
    summary = bench_entry.train(built, [micro_batch(2) for _ in range(6)], config, CPU)

    # Both axes certified, from the engine's config alone.
    assert axis(state, "parallel.strategy").applied == "zero2"
    assert axis(state, "train.offload").applied == "optimizer"
    # And the steps that produced the number went through neither engine method.
    # `forwards` is what keeps that from being the absence of a run: the engine was
    # in the path for every micro-batch and was driven for none of them.
    assert summary["steps_measured"] == 2
    assert built.model.forwards == config.train.steps * config.train.grad_accum
    assert built.model.driven == []


def test_the_dataloader_is_ours_and_not_the_one_deepspeed_would_return(composed, monkeypatch):
    """`initialize` will build a `DeepSpeedDataLoader` if handed `training_data`,
    and `applied._capture_dataloader_backend` decides that axis from the loader's
    class — the run would read back as neither `torch` nor `dali`."""
    world_of(monkeypatch, 2)
    calls = install_deepspeed(monkeypatch)
    config = bench(composed, **{"parallel.strategy": "zero3"})
    rows = [{"input_ids": torch.arange(4), "attention_mask": torch.ones(4, dtype=torch.long)}]

    built, _ = axes.assemble(plain_model(), config, CPU, framework="native", dataset=rows)

    assert "training_data" not in calls[0]
    assert isinstance(built.dataloader, torch.utils.data.DataLoader)
    assert axis(capture(built, config), "dataloader.backend").applied == "torch"


def test_offload_without_a_zero_stage_is_refused_rather_than_given_one(composed, monkeypatch):
    """`offload_optimizer` and `offload_param` are sections of `zero_optimization`,
    so there is no offload without a stage and no schema field that names one.
    Picking a stage here would put a setting in the measured path that no config
    records — and the audit, which composes one group at a time, would then read
    `train.offload` as fully applicable while every configured cross of it was
    running at a stage nobody chose."""
    world_of(monkeypatch, 2)
    install_deepspeed(monkeypatch)
    config = bench(composed, **{"train.offload": "both"})

    assert config.parallel.strategy == "single"
    with pytest.raises(axes.UnappliedAxis, match="needs a ZeRO stage"):
        axes.assemble(plain_model(), config, CPU, framework="native")


def test_zero_without_deepspeed_is_refused_as_an_axis_not_an_import_error(composed, monkeypatch):
    """`UnappliedAxis` rather than the raw `ImportError`, for the reason
    `_patch_liger` gives: the audit counts a refusal as the axis declining a value
    and anything else as a value that broke for an unrelated reason."""
    world_of(monkeypatch, 2)
    monkeypatch.setitem(sys.modules, "deepspeed", None)
    config = bench(composed, **{"parallel.strategy": "zero2"})

    with pytest.raises(axes.UnappliedAxis, match="deepspeed is not importable"):
        axes.assemble(plain_model(), config, CPU, framework="native")


# --- loss.name ---------------------------------------------------------------
#
# The equivalence test below is the one `docs/review-findings.md` asks for by
# name, and it is the point of this section: a GradCache bug and a GradCache
# speedup look the same from the outside. The loop still runs, the step is still
# timed, and only the gradient is wrong.
#
# `scripts/bench.py` computes `info_nce` inline and never calls `built.loss_fn`,
# so nothing in the measurement path exercises any of this yet. That is why the
# refusal test is here too: until the harness calls it, the only protection
# against a cached_mnrl run measuring plain in-batch negatives is that the plain
# signature cannot be called at all.


class TinyEncoder(torch.nn.Module):
    """A model shaped the way `trainbench/probe/steps.py::encode` expects one.

    Small enough to compare gradients in float64, and with a dropout layer,
    because the two forward passes GradCache makes have to draw the same masks.

    Counts its calls and the widest batch it was handed: encoding the whole batch
    in one call would give identical gradients while saving none of the memory
    this axis exists to save, and the equivalence test alone cannot tell those
    apart.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(11, 6)
        self.proj = torch.nn.Linear(6, 6)
        self.drop = torch.nn.Dropout(dropout)
        self.calls = 0
        self.widest = 0

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        self.calls += 1
        self.widest = max(self.widest, int(input_ids.shape[0]))
        return SimpleNamespace(last_hidden_state=self.drop(self.proj(self.embed(input_ids))))


def tiny_encoder(dropout: float = 0.0) -> TinyEncoder:
    """Two calls give two models with identical weights, so their gradients are
    comparable. float64 keeps the tolerance about the algorithm rather than about
    accumulation order."""
    torch.manual_seed(7)
    return TinyEncoder(dropout).double().train()


def pair_batch(rows: int = 8, width: int = 5) -> dict[str, torch.Tensor]:
    """Right-padded rows, queries first — the shape `bench.py`'s collate produces."""
    generator = torch.Generator().manual_seed(11)
    lengths = torch.tensor([2 + (index % (width - 1)) for index in range(rows)])
    return {
        "input_ids": torch.randint(1, 11, (rows, width), generator=generator),
        "attention_mask": (torch.arange(width) < lengths[:, None]).long(),
    }


# The id the multimodal fixtures below use as the image placeholder. A real
# processor expands one image into many of them; one is enough here, and it is what
# lets `TinyImageEncoder` tell a piece holding the wrong pixels from one holding
# the right ones.
IMAGE_TOKEN = 10


class TinyImageEncoder(TinyEncoder):
    """`TinyEncoder` plus the one thing that makes splitting a batch hard: pixels
    that have to land on the row they came from.

    Consumes them the way transformers' VL models do — one feature per image,
    scattered into the placeholder positions in the order those placeholders appear
    — and, like them, refuses a piece whose image count does not match its
    placeholder count.

    That refusal is not decoration, and it is the reason a wrong split is caught
    rather than measured. A piece is a contiguous run of rows, so the only thing a
    piece boundary decides is *how many* images fall on each side of it; get that
    number wrong and some piece is handed a count its rows did not ask for. There
    is no arrangement of contiguous pieces that misattributes pixels while every
    count still matches.

    Two pixel layouts, because the three checkpoints have two (measured 2026-08-02
    from the real processors): with `image_grid_thw=None` `pixel_values` carries one
    leading entry per image, which is gemma-4's; with a grid it carries one per
    patch and the grid is what says where each image's patches end, which is both
    Qwen processors'.

    The pooled representation mixes across the row — `TinyEncoder` is pointwise, so
    the last token would not depend on any pixel and an image test built on it
    would grade nothing. The mixing is strictly within a row, so a piece is still
    the same function of its rows as the whole batch is.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__(dropout)
        self.pixels = torch.nn.Linear(4, 6)

    @staticmethod
    def _per_image(pixel_values, grid):
        if grid is None:
            return pixel_values
        sizes = grid.reshape(int(grid.shape[0]), -1).prod(dim=1).tolist()
        if sum(sizes) != int(pixel_values.shape[0]):
            raise ValueError(
                f"{int(pixel_values.shape[0])} patch(es) against a grid describing "
                f"{sum(sizes)}; this piece's pixels and its grid disagree"
            )
        return torch.stack([image.mean(dim=0) for image in torch.split(pixel_values, sizes)])

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        output_hidden_states=False,
    ):
        self.calls += 1
        self.widest = max(self.widest, int(input_ids.shape[0]))
        hidden = self.embed(input_ids)
        if pixel_values is not None:
            features = self.pixels(self._per_image(pixel_values, image_grid_thw))
            spots = input_ids == IMAGE_TOKEN
            if int(spots.sum()) != int(features.shape[0]):
                raise ValueError(
                    f"{int(spots.sum())} image placeholder(s) against {int(features.shape[0])} "
                    "image feature(s); this piece was handed pixels its rows did not ask for"
                )
            hidden = hidden.masked_scatter(spots.unsqueeze(-1), features)
        attended = attention_mask.unsqueeze(-1).to(hidden.dtype)
        context = (hidden * attended).sum(dim=1, keepdim=True) / attended.sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        return SimpleNamespace(last_hidden_state=self.drop(self.proj(hidden + context)))


def tiny_image_encoder(dropout: float = 0.0) -> TinyImageEncoder:
    torch.manual_seed(7)
    return TinyImageEncoder(dropout).double().train()


def image_pair_batch(images_per_row, width: int = 6, grid=None) -> dict[str, torch.Tensor]:
    """A right-padded batch whose row `i` carries `images_per_row[i]` images.

    `grid` chooses the layout, and the two are the two the real processors return:
    `None` gives gemma-4's `pixel_values` of one row per image, while a sequence of
    `(t, h, w)` per image gives the Qwen processors', where `pixel_values` has one
    row per patch and `image_grid_thw` travels alongside it.
    """
    generator = torch.Generator().manual_seed(11)
    rows = len(images_per_row)
    ids = torch.randint(1, IMAGE_TOKEN, (rows, width), generator=generator)
    mask = torch.zeros(rows, width, dtype=torch.long)
    for index, count in enumerate(images_per_row):
        ids[index, :count] = IMAGE_TOKEN
        # Every placeholder attended, and the rows still ragged so that a pooling
        # side passed as a literal would show up.
        mask[index, : max(count + 1, 2 + index % (width - 1))] = 1
    batch = {"input_ids": ids, "attention_mask": mask}
    if not sum(images_per_row):
        # No pixels at all rather than an empty `pixel_values`: a batch of rows that
        # carry no image is what the processor returns for a text-only draw, and it
        # is the comparison the memory test needs.
        return batch
    if grid is None:
        patches = sum(images_per_row)
    else:
        thw = torch.tensor(list(grid), dtype=torch.long)
        batch["image_grid_thw"] = thw
        patches = int(thw.prod(dim=1).sum())
    batch["pixel_values"] = torch.randn((patches, 4), generator=generator, dtype=torch.float64)
    return batch


def cached(composed, **overrides):
    return bench(composed, **{"loss.name": "cached_mnrl", "loss.mini_batch": 4, **overrides})


class SubsetRows(torch.utils.data.Dataset):
    """Dict rows in the pinned subset's schema, declaring their columns the way
    `scripts/bench.py::PairDataset` does — including the image columns, whose value
    is `None` for a row that carries no image."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.column_names = sorted({key for row in rows for key in row})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def subset_rows(count: int = 4, *, qry_image: bool = False, pos_image: bool = False):
    return SubsetRows(
        [
            {
                "mmeb_config": "A-OKVQA",
                "qry": f"<|image_1|>\nq{index}",
                "pos_text": f"p{index}",
                "qry_image": object() if qry_image else None,
                "pos_image": object() if pos_image else None,
            }
            for index in range(count)
        ]
    )


def whole_batch_backward(model, batch, temperature):
    """Reference for a model without dropout: one forward, one backward."""
    from trainbench.embedding import info_nce
    from trainbench.probe import steps

    pooled = steps.encode(model, batch, "right")
    half = pooled.shape[0] // 2
    loss = info_nce(pooled[:half], pooled[half:], temperature)
    loss.backward()
    return loss.detach()


def chunked_backward(model, batch, temperature, size):
    """Reference for a model with dropout: the same chunked forward, every graph
    kept at once.

    That is the computation GradCache reproduces — it saves the memory, not the
    arithmetic. A single unchunked forward draws its dropout masks in one call and
    is a different function of the RNG, so it is not the right reference here.
    """
    from trainbench.embedding import info_nce
    from trainbench.probe import steps

    pieces = axes._split_rows(batch, size)
    pooled = torch.cat([steps.encode(model, piece, "right") for piece in pieces])
    half = pooled.shape[0] // 2
    loss = info_nce(pooled[:half], pooled[half:], temperature)
    loss.backward()
    return loss.detach()


def grads_differ(left, right, rtol=1e-6, atol=1e-8):
    return [
        name
        for (name, want), (_, got) in zip(
            left.named_parameters(), right.named_parameters(), strict=True
        )
        if not torch.allclose(got.grad, want.grad, rtol=rtol, atol=atol)
    ]


@pytest.mark.parametrize("mini_batch", [4, 3])
def test_gradcache_computes_the_gradient_a_plain_backward_would(composed, mini_batch):
    """The equivalence test `docs/review-findings.md` requires before this axis can
    be measured. Without it, a wrong gradient is reported as a speedup.

    `mini_batch=3` over eight rows is the uneven split: the schema only requires it
    to be no larger than the batch, so the last piece is short and the cache has to
    be sliced by what each piece actually returned rather than by a fixed stride.
    """
    config = cached(composed, **{"loss.mini_batch": mini_batch})
    batch = pair_batch()
    reference, gradcached = tiny_encoder(), tiny_encoder()
    expected = whole_batch_backward(reference, batch, config.loss.temperature)

    loss_fn, _ = axes._loss(config)
    loss = loss_fn.gradcache_backward(gradcached, batch, padding_side="right")

    assert float(loss) == pytest.approx(float(expected), rel=1e-12)
    for (name, want), (_, got) in zip(
        reference.named_parameters(), gradcached.named_parameters(), strict=True
    ):
        assert want.grad is not None, name
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-10, atol=1e-12, msg=name)


def test_gradcache_encodes_the_batch_in_mini_batch_sized_calls(composed):
    """The break for the test above: a `gradcache_backward` that encoded the whole
    batch in one call would pass it exactly, and would be plain MNRL with extra
    steps — same memory ceiling, same batch limit, reported as GradCache.

    This reads the *shape of the forward calls* and nothing else. An implementation
    that encodes piece by piece while keeping every piece's graph alive passes it
    exactly and saves no memory at all, which is what the test below measures
    instead. The two are not interchangeable and this one was named as if it were.
    """
    config = cached(composed)
    model = tiny_encoder()

    loss_fn, _ = axes._loss(config)
    loss_fn.gradcache_backward(model, pair_batch(rows=8), padding_side="right")

    assert model.widest == 4
    # Twice per piece: once with no graph to fill the cache, once with one to
    # consume it. Two pieces of four rows.
    assert model.calls == 4


class LiveActivations:
    """Peak activation memory, measured as autograd's own saved tensors.

    `saved_tensors_hooks` is called for every tensor a graph keeps for backward and
    the packed object is dropped when that graph is freed, so holding a weak
    reference to it and counting the ones still alive is a direct reading of how
    much activation memory is live at once — which is the quantity this axis exists
    to reduce, and the one `model.widest`/`model.calls` cannot see.

    Elements rather than bytes: the tensors here are float64 test tensors, and the
    comparison is between two runs of the same arithmetic on the same dtype.
    """

    class _Held:
        __slots__ = ("tensor", "__weakref__")

        def __init__(self, tensor):
            self.tensor = tensor

    def __init__(self) -> None:
        self.refs: list[weakref.ref] = []
        self.peak = 0

    def _pack(self, tensor):
        held = self._Held(tensor)
        self.refs.append(weakref.ref(held))
        alive = [ref() for ref in self.refs]
        alive = [held for held in alive if held is not None]
        self.refs = [weakref.ref(held) for held in alive]
        self.peak = max(self.peak, sum(held.tensor.numel() for held in alive))
        return held

    def _unpack(self, held):
        return held.tensor

    @contextlib.contextmanager
    def watching(self):
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            yield


def test_gradcache_holds_less_activation_memory_than_a_plain_backward(composed):
    """The reason this axis exists, measured — and the break for every test above it.

    Removing the `no_grad` from the first pass leaves the equivalence, the RNG
    replay, the scale, the call count and the widest-batch assertions all green: the
    arithmetic is unchanged and the forward calls are still piece-sized. What
    changes is that every piece's graph stays alive in `representations`, so the run
    holds *more* activation memory than a plain backward while reporting a GradCache
    number into the "20% to 2.4x" argument. Nothing but a memory reading can tell
    those apart.
    """
    batch = pair_batch(rows=8)
    plain = LiveActivations()
    with plain.watching():
        whole_batch_backward(tiny_encoder(), batch, 0.05)

    gradcached = LiveActivations()
    loss_fn, _ = axes._loss(cached(composed))
    with gradcached.watching():
        loss_fn.gradcache_backward(tiny_encoder(), batch, padding_side="right")

    assert plain.peak > 0, "the hooks saw no saved tensor at all; this measured nothing"
    assert gradcached.peak < plain.peak, (
        f"GradCache held {gradcached.peak} elements of live activation against the plain "
        f"backward's {plain.peak}: it is paying the second forward pass and saving nothing"
    )


def test_the_recompute_replays_the_masks_the_cache_was_built_from(composed):
    """Dropout is live under `model.train()`, and `peft.dropout` is an ablation
    setting in this study."""
    config = cached(composed)
    batch = pair_batch()
    reference, gradcached = tiny_encoder(0.5), tiny_encoder(0.5)
    loss_fn, _ = axes._loss(config)

    torch.manual_seed(3)
    expected = chunked_backward(reference, batch, config.loss.temperature, 4)
    torch.manual_seed(3)
    loss = loss_fn.gradcache_backward(gradcached, batch, padding_side="right")

    assert float(loss) == pytest.approx(float(expected), rel=1e-12)
    assert not grads_differ(reference, gradcached, rtol=1e-10, atol=1e-12)


def test_a_recompute_that_does_not_replay_the_rng_is_a_wrong_gradient(composed, monkeypatch):
    """The break. The second pass draws fresh dropout masks, so the cached gradient
    belongs to representations that no longer exist — and nothing says so: the loss
    is unchanged, the step is timed, the number looks like a GradCache number."""
    config = cached(composed)
    batch = pair_batch()
    reference, gradcached = tiny_encoder(0.5), tiny_encoder(0.5)
    loss_fn, _ = axes._loss(config)
    monkeypatch.setattr(axes, "_restore_rng", lambda state, device: None)

    torch.manual_seed(3)
    chunked_backward(reference, batch, config.loss.temperature, 4)
    torch.manual_seed(3)
    loss_fn.gradcache_backward(gradcached, batch, padding_side="right")

    assert grads_differ(reference, gradcached), (
        "the masks were replayed anyway, so the RNG handling this is meant to pin "
        "is not what makes the equivalence test pass"
    )


def test_scale_multiplies_the_gradient_and_leaves_the_reported_loss_alone(composed):
    """`grad_accum` passes `1/N`. Scaling the returned loss instead would make every
    accumulated run's recorded loss N times too small, and that value is what a
    quality run's curve is read from."""
    config = cached(composed)
    batch = pair_batch()
    whole, halved = tiny_encoder(), tiny_encoder()
    loss_fn, _ = axes._loss(config)

    full = loss_fn.gradcache_backward(whole, batch, padding_side="right")
    half = loss_fn.gradcache_backward(halved, batch, padding_side="right", scale=0.5)

    assert float(half) == pytest.approx(float(full), rel=1e-12)
    for (name, want), (_, got) in zip(
        whole.named_parameters(), halved.named_parameters(), strict=True
    ):
        torch.testing.assert_close(got.grad, want.grad * 0.5, rtol=1e-10, atol=1e-12, msg=name)


def test_a_batch_whose_pixels_cannot_be_attributed_to_rows_is_refused(composed):
    """Pixels with no map from rows to them are still refused, which is what keeps
    the map from being optional.

    `pixel_values` counts patches (Qwen-VL) or images (gemma-4), never rows.
    `images_per_row` is what says where a row's share of it begins, and a caller
    that does not pass one has given this function no way to cut it — so it stops,
    exactly as it did before any of this was splittable. Splitting by position
    would hand one row's pixels to another and nothing downstream could tell the
    resulting embedding from a real one.
    """
    batch = pair_batch()
    batch["pixel_values"] = torch.zeros(3, 4)
    loss_fn, _ = axes._loss(cached(composed))

    with pytest.raises(RuntimeError, match="pixel_values"):
        loss_fn.gradcache_backward(tiny_encoder(), batch, padding_side="right")


def test_pixels_are_never_cut_by_position_even_when_the_counts_would_line_up(composed):
    """`pixel_values` whose leading dimension happens to equal the row count is
    still not row-sliced.

    This is the one shape where the old rule — "leading entry per row, or refuse" —
    would have accepted a multimodal batch, and it would have been right by
    accident: eight images across eight rows says nothing about which row each one
    belongs to, and here the first row carries two of them while two rows carry
    none. Drop `IMAGE_PAYLOAD_KEYS` from the check and this batch splits silently
    down the middle.
    """
    counts = (2, 0, 1, 1, 2, 1, 1, 0)
    batch = image_pair_batch(counts)
    assert batch["pixel_values"].shape[0] == batch["attention_mask"].shape[0]
    loss_fn, _ = axes._loss(cached(composed))

    with pytest.raises(RuntimeError, match="pixel_values"):
        loss_fn.gradcache_backward(tiny_image_encoder(), batch, padding_side="right")

    # And with the map it is not merely accepted, it is cut where the rows are:
    # the first piece holds the four images its four rows asked for, not the first
    # four of the batch.
    pieces = axes._split_rows(batch, 4, counts)
    assert [int(piece["pixel_values"].shape[0]) for piece in pieces] == [4, 4]
    assert [int((piece["input_ids"] == IMAGE_TOKEN).sum()) for piece in pieces] == [4, 4]


# gemma-4 puts one leading entry per image in `pixel_values`; the Qwen processors
# put one per patch and send `image_grid_thw` alongside. Both were read off the real
# processors on 2026-08-02 (trainbench/axes.py::IMAGE_PAYLOAD_KEYS records the
# shapes), and both are parametrised below rather than one standing in for the
# other: the patch layout needs a second derivation — the grid split by images,
# multiplied out — that the image layout does not exercise at all.
PIXEL_LAYOUTS = {
    "gemma4-per-image": None,
    "qwen-per-patch": ((1, 2, 2), (1, 4, 2), (1, 2, 3), (1, 3, 4), (1, 2, 2), (1, 6, 2)),
}


@pytest.mark.parametrize("layout", sorted(PIXEL_LAYOUTS))
@pytest.mark.parametrize("mini_batch", [4, 3])
def test_gradcache_on_an_image_batch_computes_the_gradient_a_plain_backward_would(
    composed, layout, mini_batch
):
    """The equivalence test, on the data this axis exists for.

    GradCache is a memory technique and images are most of the memory, so a
    version of it that only works on text measures the cheap half of its own
    subject. What has to hold is what held for text: the gradient every parameter
    ends up with is the one a single unsplit backward would have produced —
    including `pixels`, the only parameter a wrong row->pixel map can reach.

    `mini_batch=3` over six rows is the uneven split, and it is the case that
    matters most here: the pieces then cut the batch at a row whose image count is
    not a multiple of anything, so the boundary has to come from the counts rather
    than from a stride.
    """
    counts = (2, 0, 1, 1, 1, 1)
    config = cached(composed, **{"loss.mini_batch": mini_batch})
    batch = image_pair_batch(counts, grid=PIXEL_LAYOUTS[layout])
    reference, gradcached = tiny_image_encoder(), tiny_image_encoder()
    expected = whole_batch_backward(reference, batch, config.loss.temperature)

    loss_fn, _ = axes._loss(config)
    loss = loss_fn.gradcache_backward(
        gradcached, batch, padding_side="right", images_per_row=counts
    )

    assert float(loss) == pytest.approx(float(expected), rel=1e-12)
    assert reference.pixels.weight.grad.abs().sum() > 0, (
        "the reference never used its pixels, so this compares two text-only runs"
    )
    for (name, want), (_, got) in zip(
        reference.named_parameters(), gradcached.named_parameters(), strict=True
    ):
        assert want.grad is not None, name
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-10, atol=1e-12, msg=name)


@pytest.mark.parametrize("layout", sorted(PIXEL_LAYOUTS))
def test_an_image_moved_across_a_piece_boundary_does_not_produce_a_number(composed, layout):
    """The break for the test above, and a statement of exactly how much of the map
    is load-bearing.

    A piece is a contiguous run of rows, so the only thing its boundary decides is
    how many images fall on each side of it. Move one across that boundary — here
    the map gives row 4 an image that row 3 owns, with `mini_batch=4` cutting
    between them — and the first piece is handed three images against four
    placeholders, which is the mismatch a real VL model raises on too. The failure
    is loud: the run stops instead of returning a loss built from another row's
    pixels. The two layouts stop at different sentences and both are the same
    finding — under the per-image layout the piece's image count is wrong, under
    the per-patch one its grid and its pixels stop agreeing first.

    The other half of the same fact, and the reason this test moves an image
    *across* a boundary rather than anywhere: a map that misplaces an image between
    two rows of the *same* piece changes nothing at all. The piece holds the same
    pixels in the same order either way, and so does the unsplit batch. Only the
    boundaries are load-bearing, which is why `scripts/bench.py`'s counts are
    checked against the placeholders they stand for in `tests/test_smoke_cpu.py`
    rather than only here.
    """
    counts = (2, 0, 1, 1, 1, 1)
    batch = image_pair_batch(counts, grid=PIXEL_LAYOUTS[layout])
    loss_fn, _ = axes._loss(cached(composed))

    moved = list(counts)
    moved[3] -= 1
    moved[4] += 1
    with pytest.raises(ValueError, match="did not ask for|pixels and its grid disagree"):
        loss_fn.gradcache_backward(
            tiny_image_encoder(), batch, padding_side="right", images_per_row=moved
        )


def test_a_map_that_does_not_describe_this_batch_is_refused(composed):
    """A count vector of the wrong length is a map for a different batch, and every
    boundary it produces is off. Refused where it arrives rather than sliced with,
    because a short vector would otherwise just make the last pieces empty."""
    counts = (1, 1, 1, 1)
    batch = image_pair_batch(counts)
    loss_fn, _ = axes._loss(cached(composed))

    with pytest.raises(RuntimeError, match="images_per_row has 3 entries"):
        loss_fn.gradcache_backward(
            tiny_image_encoder(), batch, padding_side="right", images_per_row=counts[:3]
        )


def test_a_grid_that_disagrees_with_the_counts_is_refused(composed):
    """`image_grid_thw` has one row per image and so does `images_per_row`'s sum.
    Two answers to one question, and the patch boundaries are built out of both —
    so a disagreement is refused rather than resolved in favour of either."""
    counts = (1, 1, 1, 1)
    batch = image_pair_batch(counts, grid=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)))
    batch["image_grid_thw"] = batch["image_grid_thw"][:3]
    loss_fn, _ = axes._loss(cached(composed))

    with pytest.raises(RuntimeError, match="image_grid_thw has 3 rows"):
        loss_fn.gradcache_backward(
            tiny_image_encoder(), batch, padding_side="right", images_per_row=counts
        )


def test_gradcache_holds_less_activation_than_a_plain_backward_on_an_image_batch(composed):
    """The reason this axis exists, measured with the pixels in.

    Two readings, because the first alone would pass a split that copied every
    pixel into every piece: GradCache holds less live activation than the plain
    backward, *and* halving `mini_batch` halves what it holds. A `pixel_values`
    replicated per piece would keep the second number flat while the first stayed
    green.

    The text-only figures are taken from the same model on the same rows with no
    images, so the gap between them is the pixels and nothing else.
    """
    counts = (2, 0, 1, 1, 1, 1)
    batch = image_pair_batch(counts)

    def held(run) -> int:
        watcher = LiveActivations()
        with watcher.watching():
            run()
        return watcher.peak

    plain = held(lambda: whole_batch_backward(tiny_image_encoder(), batch, 0.05))

    def gradcached(size: int) -> int:
        loss_fn, _ = axes._loss(cached(composed, **{"loss.mini_batch": size}))
        return held(
            lambda: loss_fn.gradcache_backward(
                tiny_image_encoder(), batch, padding_side="right", images_per_row=counts
            )
        )

    wide, narrow = gradcached(6), gradcached(2)
    text_only = held(
        lambda: whole_batch_backward(tiny_image_encoder(), image_pair_batch((0,) * 6), 0.05)
    )

    assert plain > text_only, "the pixels reached no graph, so this measured a text-only run"
    assert wide < plain, (
        f"GradCache held {wide} elements against the plain backward's {plain}: it is paying "
        "the second forward pass and saving nothing"
    )
    assert narrow < wide, (
        f"mini_batch=2 held {narrow} elements and mini_batch=6 held {wide}: the pieces are "
        "not carrying their own share of the pixels, which is what a replicated "
        "pixel_values looks like"
    )


def test_a_packed_batch_cannot_be_split_by_rows(composed):
    """`dataloader.packing` x `loss=cached_mnrl`, which compose together.

    A packed batch is one row carrying its boundaries in `cu_seqlens`, so there is no
    per-row leading dimension to slice and no `attention_mask` to pool with. Pooling
    it as if there were would read some other sequence's last token while both axes
    reported applied. `scripts/bench.py` refuses the pair before the loop opens; this
    is the backstop under that, so the combination cannot reach a number by another
    route.
    """
    packed = axes.PackedCollate()(
        [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])},
            {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1])},
        ]
    )
    loss_fn, _ = axes._loss(cached(composed))

    with pytest.raises(RuntimeError, match="attention_mask"):
        loss_fn.gradcache_backward(tiny_encoder(), packed, padding_side="right")


def test_the_cuda_rng_is_saved_and_put_back_on_the_device_it_came_from(monkeypatch):
    """Every measured run is CUDA and no test here is: this suite runs on CPU, so the
    branch that replays a CUDA dropout mask had no execution evidence at all — and
    replaying that mask is the whole of GradCache's correctness.

    What this pins is the branch's shape: the state is read from the device the batch
    is on and written back to that same device, and a CPU run touches neither call.
    What it cannot pin is that a real device's generator replays; that needs a GPU
    pod and is 측정 안 함.
    """
    device = torch.device("cuda", 1)
    read, written = [], []
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda where: read.append(where) or torch.tensor([7], dtype=torch.uint8),
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, where: written.append((state, where))
    )

    axes._restore_rng(axes._rng_state(device), device)

    assert read == [device]
    assert [where for _, where in written] == [device]
    assert [int(state[0]) for state, _ in written] == [7]

    axes._restore_rng(axes._rng_state(CPU), CPU)

    assert read == [device], "a CPU run read the CUDA generator"
    assert len(written) == 1, "a CPU run wrote the CUDA generator"


def test_a_batch_carrying_a_value_that_is_not_a_tensor_is_refused(composed):
    """The same refusal as above for the other half of what a real batch carries.
    `mmeb_config` is a string per row and the subset ships it; copying it into every
    piece would attach one row's label to rows it does not describe, and there is
    nothing downstream that could notice."""
    batch = pair_batch()
    batch["mmeb_config"] = ["A-OKVQA"] * 8
    loss_fn, _ = axes._loss(cached(composed))

    with pytest.raises(RuntimeError, match="mmeb_config"):
        loss_fn.gradcache_backward(tiny_encoder(), batch, padding_side="right")


def test_the_cached_loss_refuses_to_be_called_like_the_plain_one(composed):
    """GradCache is a backward strategy, not a loss function: it consumes the model
    and the batch and *produces* the pooled embeddings. A harness reaching for
    `built.loss_fn(queries, documents)` would compute ordinary in-batch negatives
    and label the number cached_mnrl, which is what this raise turns into a crash."""
    loss_fn, _ = axes._loss(cached(composed))

    with pytest.raises(RuntimeError, match="cannot be computed from pooled embeddings"):
        loss_fn(torch.zeros(2, 3), torch.zeros(2, 3))


def test_the_two_shapes_a_harness_has_to_choose_between(composed):
    """The interface `scripts/bench.py` selects on, pinned from this side.

    A plain loss is called `(queries, documents)` and has no `gradcache_backward`;
    the cached one has `gradcache_backward` and raises on that signature. So
    `getattr(built.loss_fn, "gradcache_backward", None)` decides which path a step
    takes, and neither answer can be reached by accident: reaching for the plain
    signature under a cached config raises instead of measuring in-batch negatives,
    which is the test below this one.
    """
    plain, _ = axes._loss(bench(composed))
    cached_fn, _ = axes._loss(cached(composed))

    assert getattr(plain, "gradcache_backward", None) is None
    assert callable(getattr(cached_fn, "gradcache_backward", None))
    assert cached_fn.mini_batch == 4


def test_the_cached_loss_is_read_back_as_cached_mnrl(composed):
    config = cached(composed)

    built, names = axes.assemble(
        plain_model(), config, CPU, framework="native", dataset=dataset(("qry", "pos_text"))
    )

    state = capture(built, config)
    assert axis(state, "loss.name").applied == "cached_mnrl"
    assert axis(state, "loss.name").matches
    assert "loss.name" in names


def test_gradcache_is_not_refused_on_the_subsets_this_study_measures(composed):
    """The axis reaches the data the runs actually read.

    `configs/data/*.yaml` are MMEB draws — speed.yaml records "0 rows without a
    query image or positive" — so a refusal on rows carrying images was a refusal
    of every configured run, and it left GradCache measurable only on the half of
    its own subject that costs the least memory. `_split_rows` now attributes those
    pixels from the per-row counts the collate records, so the dataset is no longer
    what decides.
    """
    for columns in (
        {"qry_image": True},
        {"pos_image": True},
        {"qry_image": True, "pos_image": True},
    ):
        built, names = axes.assemble(
            plain_model(),
            cached(composed),
            CPU,
            framework="native",
            dataset=subset_rows(**columns),
        )

        assert "loss.name" in names
        assert axis(capture(built, cached(composed)), "loss.name").applied == "cached_mnrl"


def test_the_same_subset_schema_without_images_in_it_is_not_refused(composed):
    """Text-only draws stay unrefused too. Four of the twenty MMEB configs carry no
    `qry_image` and thirteen no `pos_image`, and `bench.py::Collate` skips a `None`
    there — so a draw can hold the column and no image. Both shapes reach the axis
    now, and this is the one that always did."""
    built, names = axes.assemble(
        plain_model(), cached(composed), CPU, framework="native", dataset=subset_rows()
    )

    assert "loss.name" in names
    assert axis(capture(built, cached(composed)), "loss.name").applied == "cached_mnrl"


def test_the_feature_type_that_reading_depends_on_is_still_called_image():
    """`axes.image_columns` decides by comparing a type name to the literal
    "Image", and `scripts/audit_plan.py`'s two `axis-values` fixtures are told
    apart by what it answers.

    It no longer decides whether `loss=cached_mnrl` runs — only `None` versus an
    answer does that now — so a rename in `datasets` would cost the audit its
    resolution rather than costing a run its correctness. `datasets` is imported
    here rather than stood in for because the documented setup installs it
    (`uv sync --extra compose --extra native`).
    """
    from datasets import Image, Value

    declared = SubsetRows([{"qry": "a", "qry_image": None}])
    declared.features = {"qry": Value("string"), "qry_image": Image()}

    assert axes.image_columns(declared) == ["qry_image"]


def test_gradcache_is_refused_when_nothing_says_the_rows_can_be_split(composed):
    """Not knowing is not evidence that the batches split, and this is the refusal
    that stayed after rows carrying images stopped being one.

    A dataset that declares no columns, or whose rows are not mappings, is one the
    collate cannot ask "how many images does this row have" — and without that
    answer `_split_rows` refuses every batch that carries pixels. `assemble` called
    with no dataset at all is the same case, and it is how `audit_plan.py` probed
    every axis value until 2026-08-02.
    """
    unreadable = (
        None,
        # Declares nothing about itself.
        dataset(),
        # Declares the subset's columns, but its rows are not mappings, so what is
        # in the image column cannot be read off them either.
        dataset(("qry", "qry_image", "pos_text")),
    )
    for candidate in unreadable:
        with pytest.raises(axes.UnappliedAxis, match="does not say whether its rows carry"):
            axes.assemble(
                plain_model(), cached(composed), CPU, framework="native", dataset=candidate
            )


def test_the_plain_loss_is_not_refused_by_the_data_the_cached_one_is(composed):
    """The break for the three above: a refusal that fired on `loss=mnrl` too would
    turn the image subsets into "no loss runs here at all", which is a different and
    false statement — plain MNRL pools the batch as it comes and never splits it."""
    built, names = axes.assemble(
        plain_model(),
        bench(composed),
        CPU,
        framework="native",
        dataset=dataset(("mmeb_config", "qry", "qry_image", "pos_text")),
    )

    assert "loss.name" in names
    assert axis(capture(built, bench(composed)), "loss.name").applied == "mnrl"


def test_a_plain_loss_under_a_cached_config_stops_the_run(composed):
    """The break for the capture probe. The config asking for GradCache is not
    evidence that GradCache ran, and `purpose=timing` has to die here rather than
    after the first number."""
    config = cached(composed)
    plain, _ = axes._loss(bench(composed))

    state = capture(Built(loss_fn=plain), config)

    assert axis(state, "loss.name").applied == "mnrl"
    with pytest.raises(AppliedMismatch, match="loss.name"):
        assert_matches(state, config)


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


def test_a_gathering_loss_declares_that_it_gathers(composed):
    config = bench(composed, **{"parallel.cross_device_negatives": True})

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")

    gathered = axis(capture(built, config), "parallel.cross_device_negatives")
    assert gathered.applied == "True"
    assert gathered.matches
    # Unlike the false case, this one is an action and is named.
    assert "parallel.cross_device_negatives" in names


def test_the_declaration_describes_the_closure_that_was_built():
    """The declaration `applied._capture_cross_device_negatives` reads is evidence
    only if it comes from the branch that built the closure. Copying it off the
    config would make the probe a mirror of the request — and because the branch is
    chosen by that same field, no config-level test can tell the two apart. What can
    be told apart, and is what actually goes wrong, is a declaration that disagrees
    with what the closure does: this pins both closures' behaviour against what they
    declare.
    """
    torch.manual_seed(0)
    queries, documents = torch.randn(2, 3), torch.randn(2, 3)

    gathering, declared = axes._in_batch_scoring(0.05, gather=True)
    assert declared is True
    with pytest.raises(RuntimeError, match="needs an initialised process group"):
        gathering(queries, documents)

    local, declared = axes._in_batch_scoring(0.05, gather=False)
    assert declared is False
    assert torch.isfinite(local(queries, documents))


def test_the_gathering_loss_a_harness_calls_dies_without_a_world(composed):
    """The fail-closed guarantee is only a guarantee where it is reached. This is the
    loss object a harness is handed, called the way a harness calls it: on a single
    process it raises rather than quietly scoring local in-batch negatives under the
    cross-device label."""
    config = bench(composed, **{"parallel.cross_device_negatives": True})
    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")

    torch.manual_seed(0)
    with pytest.raises(RuntimeError, match="needs an initialised process group"):
        built.loss_fn(torch.randn(2, 3), torch.randn(2, 3))


def test_the_cached_loss_gathers_when_both_axes_are_asked_for(composed):
    """`loss=cached_mnrl parallel=single_cross_device` composes, and nothing
    exercised it: the cached branch declares its own gathering and scores through
    its own closure, so a cached loss that declared `False` — or scored without the
    gather — would have been read back as a run with world-wide negatives while
    computing local ones."""
    config = cached(composed, **{"parallel.cross_device_negatives": True})

    built, names = axes.assemble(
        plain_model(), config, CPU, framework="native", dataset=dataset(("qry", "pos_text"))
    )

    state = capture(built, config)
    assert axis(state, "loss.name").applied == "cached_mnrl"
    assert axis(state, "parallel.cross_device_negatives").applied == "True"
    assert {"loss.name", "parallel.cross_device_negatives"} <= set(names)
    # And the gather is inside the path GradCache scores through, not only in the
    # attribute: on one process the step dies before it produces a number.
    with pytest.raises(RuntimeError, match="needs an initialised process group"):
        built.loss_fn.gradcache_backward(tiny_encoder(), pair_batch(), padding_side="right")


def test_the_axis_has_a_config_that_can_ask_for_it():
    """An axis nothing can request is an axis that cannot be run — the shape this
    repository keeps producing. Every other `parallel` variant sets this false, so
    composing the group is the only place the request can be checked."""
    from hydra import compose, initialize_config_dir

    from trainbench.compose import resolve

    from .conftest import CONFIG_DIR

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = resolve(
            compose(
                config_name="config",
                overrides=["device=cpu", "parallel=single_cross_device"],
            )
        )[0]

    assert config.parallel.cross_device_negatives
    # Paired with `single` rather than `ddp` because `assemble` still refuses every
    # wrapper: under `ddp` the value would be refused for the strategy and the two
    # settings of this axis would never be compared.
    assert config.parallel.strategy == "single"


def test_gathering_without_a_process_group_stops_the_run_before_a_number():
    with pytest.raises(RuntimeError, match="needs an initialised process group"):
        axes._gather_with_grad(torch.zeros(2, 3))


def test_gathering_over_one_rank_is_plain_mnrl_and_is_refused(monkeypatch):
    """The break. With `world_size=1` the gather is a no-op and the loss is exactly
    the local one — while the capture probe still reports the axis as applied,
    because the closure really does gather. Nothing downstream could tell that
    number from a world-sized one."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    with pytest.raises(RuntimeError, match="world_size=1"):
        axes._gather_with_grad(torch.zeros(2, 3))


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _world_negatives_worker(rank: int, world: int, port: int) -> None:
    """One rank of the two-process check. Its assertions are its exit code.

    Module level because `torch.multiprocessing.spawn` pickles the target by
    reference; a closure could not be sent to the child.
    """
    import torch.distributed as dist

    from trainbench.embedding import info_nce

    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    try:
        rows, dim, temperature = 3, 4, 0.05
        # Every rank builds every rank's tensors, so each one already knows what
        # the world matrix has to contain; the comparison is not a copy of the
        # gather it is checking.
        everyone = [
            torch.randn(rows, dim, generator=torch.Generator().manual_seed(100 + other))
            for other in range(world)
        ]
        queries = everyone[rank].clone().requires_grad_(True)
        documents = (everyone[rank] + 0.5).clone().requires_grad_(True)

        gathered = axes._gather_with_grad(queries)
        assert gathered.shape == (rows * world, dim)
        # Rank order is the whole of the label correction. `info_nce` pairs row i
        # with column i, and that is the positive only if rank r's local row i sits
        # at world index r * rows + i on both sides.
        for other in range(world):
            torch.testing.assert_close(gathered[other * rows : (other + 1) * rows], everyone[other])

        scores, gathers = axes._in_batch_scoring(temperature, True)
        assert gathers is True
        loss = scores(queries, documents)

        expected_queries = torch.cat(everyone).requires_grad_(True)
        expected_documents = (torch.cat(everyone) + 0.5).requires_grad_(True)
        expected = info_nce(expected_queries, expected_documents, temperature)
        torch.testing.assert_close(loss, expected)

        loss.backward()
        expected.backward()
        # What the grad-passing assignment buys. A plain `dist.all_gather` returns
        # buffers with no history, so this is None: the step runs, the timer records
        # it, and the model has learned nothing.
        assert queries.grad is not None
        torch.testing.assert_close(
            queries.grad, expected_queries.grad[rank * rows : (rank + 1) * rows]
        )
    finally:
        dist.destroy_process_group()


def _world_gradcache_worker(rank: int, world: int, port: int) -> None:
    """One rank of the cached x cross-device check. Its assertions are its exit code.

    The two axes compose (`loss=cached_mnrl parallel=single_cross_device`) and each
    was only ever exercised alone. What has to hold is that GradCache's cache is
    `d(world loss)/d(local representations)` — the gather has to happen in the pass
    that scores the cache, not in one that never reaches the model. A cached branch
    that scored locally would produce a smaller, wrong gradient here and nothing
    outside a real world could tell.
    """
    import torch.distributed as dist
    from hydra import compose, initialize_config_dir

    from trainbench.compose import resolve
    from trainbench.embedding import info_nce
    from trainbench.probe import steps

    from .conftest import CONFIG_DIR

    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    try:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            config = resolve(
                compose(
                    config_name="config",
                    overrides=[
                        "device=cpu",
                        "loss=cached_mnrl",
                        "loss.mini_batch=2",
                        "parallel=single_cross_device",
                    ],
                )
            )[0]
        assert config.loss.name == "cached_mnrl"
        assert config.parallel.cross_device_negatives

        def rank_batch(of_rank: int) -> dict:
            """Rank-dependent rows, known to every rank. With identical batches the
            slices of the world matrix would be interchangeable and a wrong rank
            order could not show up."""
            batch = pair_batch(rows=4)
            batch["input_ids"] = (batch["input_ids"] + of_rank) % 10 + 1
            return batch

        reference, gradcached = tiny_encoder(), tiny_encoder()

        # The reference builds the world matrix itself, from every rank's rows,
        # without gathering anything — so it shares no code with what it checks.
        # The other ranks' embeddings are detached because that is what
        # `_gather_with_grad` makes them: this rank ends up holding
        # d(world loss)/d(its own embeddings) and nothing else.
        world_queries, world_documents = [], []
        for other in range(world):
            pooled = steps.encode(reference, rank_batch(other), "right")
            if other != rank:
                pooled = pooled.detach()
            half = pooled.shape[0] // 2
            world_queries.append(pooled[:half])
            world_documents.append(pooled[half:])
        expected = info_nce(
            torch.cat(world_queries), torch.cat(world_documents), config.loss.temperature
        )
        expected.backward()

        batch = rank_batch(rank)

        loss_fn, applied = axes._loss(config)
        assert "parallel.cross_device_negatives" in applied
        assert loss_fn.axis_cross_device_negatives is True
        loss = loss_fn.gradcache_backward(gradcached, batch, padding_side="right")

        torch.testing.assert_close(loss, expected.detach())
        for (name, want), (_, got) in zip(
            reference.named_parameters(), gradcached.named_parameters(), strict=True
        ):
            assert want.grad is not None, name
            torch.testing.assert_close(got.grad, want.grad, rtol=1e-10, atol=1e-12, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="gloo is the only CPU backend that all_gathers",
)
def test_gradcache_caches_the_world_gradient_and_not_the_local_one():
    """Two real processes, for the combination neither axis's own tests reach."""
    import torch.multiprocessing as mp

    mp.spawn(_world_gradcache_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="gloo is the only CPU backend that all_gathers",
)
def test_negatives_are_drawn_from_every_rank():
    """Two real processes, because `dist.all_gather` cannot be exercised in one.

    gloo is what makes this runnable without a GPU, and it is also the limit: the
    NCCL path and every cost of the gather are 측정 안 함 here. Nothing in this test
    is a measurement.
    """
    import torch.multiprocessing as mp

    mp.spawn(_world_negatives_worker, args=(2, _free_port()), nprocs=2, join=True)


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


def token_rows(lengths=(3, 5, 2)):
    """Rows that already carry token ids, as `pretokenize` leaves them.

    With the mask a tokenizer returns beside them: `PackedCollate` refuses a row
    that records its padding nowhere, and a row tokenised on its own carries an
    all-ones mask rather than no mask at all.
    """
    return [
        {"input_ids": torch.arange(n) + 1, "attention_mask": torch.ones(n, dtype=torch.long)}
        for n in lengths
    ]


class RowDataset(torch.utils.data.Dataset):
    """A dataset of dict rows that declares nothing about itself."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


# --- dataloader.pretokenize --------------------------------------------------


def test_pretokenize_moves_the_tokenisation_off_the_timed_step(composed):
    """Applied and read back: the loader is identical either way, so the dataset
    handed to it is the only thing that can say the work moved."""
    raw = RowDataset([{"qry": "a"}, {"qry": "b"}, {"qry": "c"}, {"qry": "d"}])
    tokenised = axes.pretokenize(raw, lambda row: {"input_ids": torch.arange(3), **row})

    built, config = assembled_loader(
        composed, tokenised, **{"dataloader.pretokenize": True, "train.batch_size": 4}
    )

    assert "input_ids" in tokenised.column_names
    assert axis(capture(built, config), "dataloader.pretokenize").applied == "True"


def test_pretokenize_refuses_an_encode_that_tokenises_nothing():
    """The break. A passthrough `encode` leaves the tokenisation inside the step
    while the axis reports as applied — the substitution UnappliedAxis exists for."""
    raw = RowDataset([{"qry": "a"}, {"qry": "b"}])

    with pytest.raises(axes.UnappliedAxis, match="not tokenised"):
        axes.pretokenize(raw, lambda row: row)


def test_a_pretokenize_request_over_untokenised_rows_is_refused(composed):
    """The break. Nothing here can tokenise — there is no processor in this module —
    so building the loader anyway would time the tokenisation under a pretokenized
    label."""
    with pytest.raises(axes.UnappliedAxis, match="pretokenize"):
        assembled_loader(
            composed,
            dataset(("qry", "pos_text")),
            **{"dataloader.pretokenize": True, "train.batch_size": 4},
        )


class CountingEncode:
    """An `encode` that records every row it saw, and can be armed to fail if it
    is ever called again."""

    def __init__(self):
        self.calls = []
        self.armed = False

    def __call__(self, row):
        if self.armed:
            raise AssertionError(f"encode ran after the timed window opened, on {row}")
        self.calls.append(row["qry"])
        return {"input_ids": torch.arange(3) + 1, **row}


def test_pretokenize_runs_every_row_now_and_nothing_inside_the_loop(composed):
    """The break, and what the title of the test above only asserts about column
    names. `pretokenize` is a move, not a label: an implementation that encodes
    inside `__getitem__` and advertises `input_ids` off one probed row is
    indistinguishable to the loader, to the capture probe and to `assert_matches` —
    the columns are identical — while 100% of the tokenisation stays inside the
    measured step and the run publishes it as pretokenized. Counting the calls on
    both sides of the window is the only thing that separates the two.
    """
    encode = CountingEncode()
    raw = RowDataset([{"qry": "a"}, {"qry": "b"}, {"qry": "c"}, {"qry": "d"}])

    tokenised = axes.pretokenize(raw, encode)

    assert encode.calls == ["a", "b", "c", "d"]

    built, config = assembled_loader(
        composed,
        tokenised,
        **{"dataloader.pretokenize": True, "train.batch_size": 2, "data.num_workers": 0},
    )
    encode.armed = True

    batches = list(built.dataloader)

    assert len(batches) == 2
    assert all("input_ids" in batch for batch in batches)
    assert axis(capture(built, config), "dataloader.pretokenize").applied == "True"


def test_a_dataset_that_advertises_ids_it_does_not_hand_over_is_refused(composed):
    """The break. `column_names` is what a dataset says about itself and the rows
    are what the step is handed; a dataset answering the first question one way and
    the second another gets the axis certified off the answer nobody trains on."""

    class Advertises(torch.utils.data.Dataset):
        column_names = ["input_ids", "qry"]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"qry": "a"}

    with pytest.raises(axes.UnappliedAxis, match="first row carries none"):
        assembled_loader(
            composed,
            Advertises(),
            **{"dataloader.pretokenize": True, "train.batch_size": 2},
        )


# --- dataloader.packing ------------------------------------------------------


def test_a_packed_batch_is_one_row_carrying_its_own_boundaries():
    rows = token_rows((3, 5, 2))

    batch = axes.PackedCollate()(rows)

    assert batch["input_ids"].shape == (1, 10)
    assert batch["cu_seqlens"].tolist() == [0, 3, 8, 10]
    assert batch["seq_lengths"].tolist() == [3, 5, 2]
    # Restarted per sequence: to a positional encoding a packed batch is otherwise
    # one long sequence, and the rows would be positioned as each other's context.
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]]
    # No padding is the point of packing, so every token in the batch is a real one.
    # Counted off the rows that went in, not off `seq_lengths`: that vector is derived
    # from the sequences this collate just concatenated, so comparing the two is the
    # batch agreeing with itself and holds for a batch made entirely of PAD.
    assert int(batch["input_ids"].numel()) == sum(row["input_ids"].numel() for row in rows) == 10
    # What the harness has to lift out before `model(**tensors)`, and what is left
    # once it has. A boundary key reaching the forward pass is a TypeError; a model
    # input mistaken for a boundary is a tensor that never arrives.
    assert set(batch) - set(axes.PACKED_BOUNDARY_KEYS) == {"input_ids", "position_ids"}
    assert set(axes.PACKED_BOUNDARY_KEYS) <= set(batch)


def test_packing_is_read_back_off_the_collate_the_loader_got(composed):
    built, config = assembled_loader(
        composed,
        RowDataset(token_rows((3, 5, 2, 4))),
        **{"dataloader.packing": True, "train.batch_size": 4},
    )

    assert isinstance(built.dataloader.collate_fn, axes.PackedCollate)
    assert axis(capture(built, config), "dataloader.packing").applied == "True"


def test_a_packed_batch_under_an_unpacked_request_is_refused(composed):
    """The break. The collate is the evidence, so a run that packs while its config
    says otherwise is a mismatch rather than a detail."""
    built, _ = assembled_loader(
        composed,
        RowDataset(token_rows((3, 5, 2, 4))),
        **{"dataloader.packing": True, "train.batch_size": 4},
    )
    asked_for_unpacked = bench(composed, **{"dataloader.packing": False})

    state = capture(built, asked_for_unpacked)

    assert axis(state, "dataloader.packing").applied == "True"
    with pytest.raises(AppliedMismatch, match="dataloader.packing"):
        assert_matches(state, asked_for_unpacked)


def test_packing_untokenised_rows_stops_rather_than_packing_nothing(composed):
    """The break. Without a tokenizer this collate has nothing to concatenate, and
    a batch quietly built out of the raw dicts would be measured as packed."""
    built, _ = assembled_loader(
        composed,
        dataset(("qry", "pos_text")),
        **{"dataloader.packing": True, "train.batch_size": 4},
    )

    with pytest.raises(RuntimeError, match="input_ids"):
        built.dataloader.collate_fn([{"qry": "a"}, {"qry": "b"}])


def test_an_empty_sequence_is_refused_rather_than_packed():
    """The break. A zero-length sequence takes no room in the pack, so its pooled
    embedding would be the previous sequence's last token under its name."""
    with pytest.raises(ValueError, match="empty"):
        axes.PackedCollate()(
            [
                {"input_ids": torch.arange(3), "attention_mask": torch.ones(3, dtype=torch.long)},
                {"input_ids": torch.zeros(0), "attention_mask": torch.zeros(0, dtype=torch.long)},
            ]
        )


# --- dataloader.packing over a tokenize callable -----------------------------
#
# The `tokenize` path is the one a harness with a processor has to use — it is the
# only way raw text reaches a pack — and it had no test at all. What it does with a
# padding tokenizer is the failure this axis cannot survive: the PADs are
# concatenated as real tokens, so tokens/s counts work the model never did and
# `packed_last_token_pool` reads a PAD as some sequence's embedding, while the run
# still certifies `dataloader.packing=True` off the class attribute.


def test_a_tokenize_callable_without_a_pad_id_is_refused():
    """The break. Optional `pad_id` means the default construction of this path is
    the one that cannot recognise the padding it is about to pack."""
    with pytest.raises(ValueError, match="needs pad_id"):
        axes.PackedCollate(tokenize=lambda rows: [torch.arange(3)])


def test_a_padding_tokenizer_is_refused_rather_than_packed():
    """The break. `pad_sequence` is the natural way to turn a batch of texts into
    tensors, and it pads; so does every HF tokenizer at its default. Without this
    the batch below packs 3 PADs out of 6 tokens and reports them as measured."""
    from torch.nn.utils.rnn import pad_sequence

    def tokenize(rows):
        return list(pad_sequence([torch.arange(n) + 1 for n in (3, 1, 2)], batch_first=True))

    with pytest.raises(ValueError, match="contain pad id 0"):
        axes.PackedCollate(tokenize=tokenize, pad_id=0)([{}, {}, {}])


def test_a_left_padded_sequence_is_refused_by_the_packing_collate():
    """The same failure padded the other way. Every padded fixture beside this one
    pads right, so a pad scan reduced to "is the last position PAD" reads as correct
    over the whole suite — and misses gemma-4, whose `model.padding_side` is `left`
    and whose padding lands in front of the ids rather than behind them.

    Left padding is what `embedding.last_token_pool` exists to distinguish, so a
    pack built out of these rows pools PAD positions while reporting the axis
    applied.
    """

    def tokenize(rows):
        return [torch.tensor([1, 2, 3]), torch.tensor([0, 0, 4]), torch.tensor([0, 5, 6])]

    with pytest.raises(ValueError, match="contain pad id 0"):
        axes.PackedCollate(tokenize=tokenize, pad_id=0)([{}, {}, {}])


def test_a_tokenize_callable_returning_a_rectangle_is_refused():
    """The break. The same padded batch, handed over as the 2-D tensor it is:
    flattening it was how PAD entered the pack without anything raising."""
    with pytest.raises(ValueError, match="rectangle is a padded batch"):
        axes.PackedCollate(tokenize=lambda rows: torch.zeros(3, 4, dtype=torch.long), pad_id=0)(
            [{}, {}, {}]
        )


def test_a_two_dimensional_sequence_is_refused_rather_than_flattened():
    """The break. Per-sequence tensors straight out of `tokenizer(..., return_tensors)`
    are `(1, n)`, and a list of them from a padded batch is a rectangle in pieces."""
    with pytest.raises(ValueError, match="1-D sequences"):
        axes.PackedCollate(tokenize=lambda rows: [torch.zeros(2, 3, dtype=torch.long)], pad_id=7)(
            [{}]
        )


def test_the_tokenize_path_packs_the_sequences_the_callable_returned():
    """Applied, not merely permitted: the ids in the pack are the callable's own,
    in its own order, and the boundaries describe them."""
    seen = []

    def tokenize(rows):
        seen.append(len(rows))
        return [torch.tensor([5, 6, 7]), torch.tensor([8]), torch.tensor([9, 10])]

    batch = axes.PackedCollate(tokenize=tokenize, pad_id=0)([{"qry": "a"}, {"qry": "b"}])

    assert seen == [2]
    assert batch["input_ids"].tolist() == [[5, 6, 7, 8, 9, 10]]
    assert batch["cu_seqlens"].tolist() == [0, 3, 4, 6]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 0, 1]]


def test_the_loader_axis_never_packs_rows_it_cannot_check_for_padding(composed):
    """`_dataloader` builds `PackedCollate()` — no tokenizer, so no `pad_id`, and
    `_refuse_pad_id` returns on the first line. A row that also brings no
    `attention_mask` was packed with nothing at all having looked at it: the ids
    below are padded and the batch this used to yield was `(1, 12)` of which 6
    tokens were PAD, counted by tokens/s as work and pooled as two sequences'
    embeddings, while `capture` still certified `dataloader.packing=True`.

    Drawn through the loader `assemble` built rather than by calling the collate
    directly, because it is that construction — the one with no processor behind
    it — that the harness inherits when it does not replace the collate.
    """
    padded = axes.PretokenizedDataset(
        [{"input_ids": torch.tensor([1, 2, 3, 0, 0, 0])} for _ in range(2)]
    )
    built, _ = assembled_loader(
        composed,
        padded,
        **{
            "dataloader.packing": True,
            "dataloader.pretokenize": True,
            "train.batch_size": 2,
            "data.num_workers": 0,
        },
    )

    with pytest.raises(ValueError, match="carry no 'attention_mask'"):
        next(iter(built.dataloader))


def test_pretokenised_rows_that_arrived_padded_are_refused_by_their_own_mask():
    """The break on the other path. `pretokenize` hands the collate whatever the
    caller's `encode` produced, and an `encode` that tokenised the rows as a batch
    padded them — the mask is where the row admits it."""
    rows = [
        {"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.tensor([1, 1, 1])},
        {"input_ids": torch.tensor([8, 0, 0]), "attention_mask": torch.tensor([1, 0, 0])},
    ]

    with pytest.raises(ValueError, match="arrive padded"):
        axes.PackedCollate()(rows)


def test_a_mask_that_does_not_describe_its_row_is_refused():
    """The break. A mask read against the wrong ids would report any padding it
    liked, including none."""
    rows = [{"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.tensor([1, 1])}]

    with pytest.raises(ValueError, match="does not describe the row"):
        axes.PackedCollate()(rows)


def test_unpadded_rows_keep_their_mask_and_pack():
    """The check above must not refuse the rows it exists to protect."""
    rows = [
        {"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.ones(3, dtype=torch.long)},
        {"input_ids": torch.tensor([8]), "attention_mask": torch.ones(1, dtype=torch.long)},
    ]

    batch = axes.PackedCollate()(rows)

    assert batch["input_ids"].tolist() == [[5, 6, 7, 8]]


def test_packing_over_pretokenised_rows_runs_end_to_end(composed):
    """Both halves of this axis on at once, through the loader `assemble` built and
    the batch it actually yields — which is what neither the class attribute the
    capture probe reads nor `audit_plan.py`'s dataset-free `assemble` can witness.

    The rows are deliberately ragged: torch's own collate cannot stack them, so a
    run that reached this line without packing would have raised instead.
    """
    tokenised = axes.pretokenize(
        RowDataset([{"qry": "a"}, {"qry": "bb"}, {"qry": "ccc"}]),
        # The mask a tokenizer returns beside the ids travels with the row: it is
        # the record of padding `PackedCollate` reads, and a row carrying none is
        # refused rather than packed unchecked.
        lambda row: {
            "input_ids": torch.arange(len(row["qry"]) + 1) + 1,
            "attention_mask": torch.ones(len(row["qry"]) + 1, dtype=torch.long),
        },
    )
    built, config = assembled_loader(
        composed,
        tokenised,
        **{
            "dataloader.packing": True,
            "dataloader.pretokenize": True,
            "train.batch_size": 3,
            "data.num_workers": 0,
        },
    )

    batch = next(iter(built.dataloader))
    state = capture(built, config)

    assert batch["input_ids"].shape == (1, 9)
    assert batch["seq_lengths"].tolist() == [2, 3, 4]
    assert axis(state, "dataloader.packing").applied == "True"
    assert axis(state, "dataloader.pretokenize").applied == "True"


def test_a_config_offers_packing_over_pretokenised_rows():
    """AGENTS.md: a new experiment variant comes from config composition, never a
    code change. Packing needs unpadded per-sequence ids and `pretokenize` is what
    produces them, yet every config offering packing left `pretokenize: false` — the
    one combination this module supports without a tokenizer was inexpressible."""
    from hydra import compose, initialize_config_dir

    from trainbench.compose import resolve

    from .conftest import CONFIG_DIR

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        mapping = resolve(
            compose(
                config_name="config",
                overrides=["device=cpu", "dataloader=torch_packed_pretokenized"],
            )
        )[1]

    assert mapping["dataloader"]["backend"] == "torch"
    assert mapping["dataloader"]["packing"] is True
    assert mapping["dataloader"]["pretokenize"] is True


# --- packed pooling ----------------------------------------------------------


def test_pooling_a_packed_batch_matches_pooling_the_same_rows_padded():
    """Packing changes the layout, not the embedding. Padded and packed are pooled
    by two different functions, so this is the only place they are compared."""
    lengths = [3, 5, 2]
    total = sum(lengths)
    flat = torch.arange(total * 4, dtype=torch.float32).reshape(1, total, 4)
    cu_seqlens = torch.tensor([0, 3, 8, 10], dtype=torch.int32)

    padded = torch.zeros(len(lengths), max(lengths), 4)
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    for row, (start, length) in enumerate(zip([0, 3, 8], lengths, strict=True)):
        padded[row, :length] = flat[0, start : start + length]
        mask[row, :length] = 1

    packed_pooled = packed_last_token_pool(flat, cu_seqlens)
    padded_pooled = last_token_pool(padded, mask, padding_side="right")

    assert torch.equal(packed_pooled, padded_pooled)


def test_boundaries_that_do_not_describe_the_batch_are_refused():
    """The break. cu_seqlens is the only description of where a sequence ends, so
    one that disagrees with the batch pools every sequence at the wrong token."""
    flat = torch.zeros(1, 10, 4)

    with pytest.raises(ValueError, match="cu_seqlens ends at"):
        packed_last_token_pool(flat, torch.tensor([0, 3, 8], dtype=torch.int32))
    with pytest.raises(ValueError, match="empty or out of order"):
        packed_last_token_pool(flat, torch.tensor([0, 3, 3, 10], dtype=torch.int32))
    with pytest.raises(ValueError, match="start at 0"):
        packed_last_token_pool(flat, torch.tensor([1, 3, 10], dtype=torch.int32))


def test_a_padded_batch_is_not_pooled_by_the_packed_path():
    """The break. `last_token_pool` checks the padding side; the packed path cannot,
    because a packed batch has no padding — so it refuses the shape outright rather
    than pooling a rectangle whose PADs it would read as content."""
    with pytest.raises(ValueError, match="a packed batch is one row"):
        packed_last_token_pool(torch.zeros(3, 5, 4), torch.tensor([0, 5], dtype=torch.int32))


# --- optim.name = muon -------------------------------------------------------
#
# Muon is `pytorch-optimizer`'s, a py3-none-any wheel whose only dependencies are
# numpy and torch, so the Newton-Schulz iteration runs on the CPU this suite runs
# on and these tests take a real step rather than inspect a class name. What no
# test here can show is throughput: Muon's claim is about convergence and about
# optimizer-state memory, both of which need a GPU and a real checkpoint. 측정 안 함.
#
# The parameter split is the decision that qualifies every Muon row in the report,
# and `docs/methodology.md` §5 carries it. It is pinned below as well, because a
# note nobody can fail is a note that drifts.


def muon(composed, **overrides):
    return bench(composed, **{"optim.name": "muon", **overrides})


def test_the_muon_step_moves_the_weights_because_it_read_the_gradient(composed):
    """A real step with a control, because "the weights moved" is not evidence that
    an optimizer optimised anything.

    The earlier form of this test asserted only that every tensor differed after
    `step()`. Decoupled weight decay satisfies that on its own — it multiplies
    every parameter by `1 - lr*wd` before any gradient is looked at — so a Muon
    whose `step` never reads `p.grad` passed it, which is the failure this test
    was written to make impossible. The same build is therefore stepped twice from
    the same initial weights, once with the gradients backward produced and once
    with those gradients zeroed, and the two must land in different places: the
    only difference between the runs is the gradient.
    """

    def step(*, gradients: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        torch.manual_seed(0)
        model = plain_model()
        batch = torch.randn(4, 2)
        built, names = axes.assemble(model, muon(composed), CPU, framework="native")
        assert "optim.name" in names
        assert type(built.optimizer).__name__ == "Muon"
        started = [p.detach().clone() for p in model.parameters()]
        model(batch).sum().backward()
        if not gradients:
            for p in model.parameters():
                p.grad.zero_()
        built.optimizer.step()
        return started, [p.detach().clone() for p in model.parameters()]

    started, stepped = step(gradients=True)
    _, decayed = step(gradients=False)

    assert all(not torch.equal(was, now) for was, now in zip(started, stepped, strict=True))
    assert all(
        not torch.equal(gradient, decay) for gradient, decay in zip(stepped, decayed, strict=True)
    )


def test_the_muon_update_is_the_orthogonalised_gradient_not_an_elementwise_one(composed):
    """What separates Muon from the AdamW it contains, asserted on the update that
    reached the weight rather than on the flag that was supposed to route it.

    Newton-Schulz drives the singular values of the update toward one another; an
    elementwise optimizer rescales entries and leaves the gradient's spectrum as
    spread as it found it. So the gradient here is built with a known spectrum —
    smallest over largest exactly 0.1 — and the update that lands on the weight is
    measured two ways: its own spectrum must come out far flatter than the
    gradient's, and its direction must be the gradient's orthogonalisation `U @ Vt`
    rather than the gradient itself.

    Both numbers separate the two paths by a wide margin (measured on this build:
    Muon 0.60 flatness / 0.98 cosine, the same optimizer with `use_muon=False`
    0.003-0.10 / 0.66-0.73), so a Muon built with every group on the AdamW side
    fails here — the case the run record could not previously distinguish.
    """
    width = 16
    torch.manual_seed(0)
    layer = torch.nn.Linear(width, width, bias=False)
    model = torch.nn.Sequential(layer)
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    built, _ = axes.assemble(model, muon(composed), CPU, framework="native")

    left, _ = torch.linalg.qr(torch.randn(width, width))
    right, _ = torch.linalg.qr(torch.randn(width, width))
    gradient = (left @ torch.diag(torch.logspace(0, -1, width)) @ right.T).double()
    before = layer.weight.detach().clone()
    layer.weight.grad = gradient.float().clone()
    built.optimizer.step()
    update = (layer.weight.detach() - before).double()

    grad_spectrum = torch.linalg.svdvals(gradient)
    assert float(grad_spectrum.min() / grad_spectrum.max()) == pytest.approx(0.1, abs=1e-6)

    update_spectrum = torch.linalg.svdvals(update)
    assert float(update_spectrum.min() / update_spectrum.max()) > 0.4

    u, _, vt = torch.linalg.svd(gradient)
    orthogonalised = -(u @ vt)
    cosine = float(
        (update.flatten() @ orthogonalised.flatten()) / (update.norm() * orthogonalised.norm())
    )
    assert cosine > 0.9


def spread_gradient(shape: tuple[int, int]) -> torch.Tensor:
    """A gradient of known spectrum: smallest singular value exactly a tenth of the
    largest, at any shape. Newton-Schulz drives that ratio toward one and an
    elementwise optimizer leaves it as it found it, so the ratio is what separates
    the two paths."""
    rows, cols = shape
    rank = min(shape)
    left, _ = torch.linalg.qr(torch.randn(rows, rank))
    right, _ = torch.linalg.qr(torch.randn(cols, rank))
    return (left @ torch.diag(torch.logspace(0, -1, rank)) @ right.T).double()


def muon_step(built, weights, gradients) -> list[torch.Tensor]:
    """One step with the given gradients on every weight at once, returning what
    landed on each. Stepping the whole model rather than one tensor is the point:
    a build that orthogonalises only the tensor a test happens to watch is the
    substitution these assertions have to see."""
    before = [w.detach().clone() for w in weights]
    for weight, gradient in zip(weights, gradients, strict=True):
        weight.grad = gradient.float().clone()
    built.optimizer.step()
    return [(w.detach() - was).double() for w, was in zip(weights, before, strict=True)]


def assert_orthogonalised(update, gradient, *, lr, where):
    """The update that reached the weight is the gradient's orthogonalisation.

    Three independent readings, because each one alone has a way through. The
    spectrum admits any elementwise rescaling that happens to come out flat; the
    direction admits an update of the right shape and the wrong size — `bigstep`,
    lr x100, keeps both. The norm closes that: an orthogonalised update has every
    singular value near one, so its Frobenius norm is fixed by the rank and the
    step size at `lr * sqrt(k)` times a constant Muon sets internally (measured
    0.72-0.89 across the shapes asserted here). The window admits that and refuses
    both a step two orders too large and an update that never happened.
    """
    spectrum = torch.linalg.svdvals(gradient)
    assert float(spectrum.min() / spectrum.max()) == pytest.approx(0.1, abs=1e-6), where

    update_spectrum = torch.linalg.svdvals(update)
    assert float(update_spectrum.min() / update_spectrum.max()) > 0.4, where

    u, _, vt = torch.linalg.svd(gradient, full_matrices=False)
    orthogonalised = -(u @ vt)
    cosine = float(
        (update.flatten() @ orthogonalised.flatten()) / (update.norm() * orthogonalised.norm())
    )
    assert cosine > 0.9, where

    scale = float(update.norm()) / (lr * min(update.shape) ** 0.5)
    assert 0.2 < scale < 3.0, (where, scale)


@pytest.mark.parametrize("shape", [(24, 8), (8, 24)])
def test_the_muon_update_is_orthogonalised_on_a_non_square_matrix_too(composed, shape):
    """The same assertion off the square, and past the first step.

    The square 16x16 above is the shape a real checkpoint almost never has, and it
    is the one shape where Newton-Schulz's transpose branch — the iteration runs on
    the smaller side and transposes back — is never taken. Three Muons that
    orthogonalise only the square case, only the small case, or only step 1, and
    hand everything else to sign-SGD, kept the run record honest and the square
    test green. Both non-square orientations are asserted because tall and wide
    take opposite sides of that branch, and step 2 because a first-step-only
    orthogonalisation is otherwise indistinguishable from the real thing.
    """
    config = muon(composed)
    rows, cols = shape
    torch.manual_seed(0)
    layer = torch.nn.Linear(cols, rows, bias=False)
    assert tuple(layer.weight.shape) == shape
    model = torch.nn.Sequential(layer)
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    built, _ = axes.assemble(model, config, CPU, framework="native")
    gradient = spread_gradient(shape)

    for step in (1, 2):
        (update,) = muon_step(built, [layer.weight], [gradient])
        assert_orthogonalised(update, gradient, lr=config.optim.lr, where=f"{shape} step {step}")


def test_muon_orthogonalises_every_trainable_matrix_not_only_the_one_the_test_watches(composed):
    """The tensor coverage the assertion above cannot give.

    Every behavioural test here used to build a model with one trainable matrix,
    so a Muon that routed the first matrix to Newton-Schulz and every other one to
    the internal AdamW passed all of them — and `_capture_optim` cannot see it
    either, because `use_muon` is set per group and that build sets it honestly.
    Three matrices of three different shapes are stepped together and all three are
    asserted, twice.
    """
    config = muon(composed)
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 24, bias=False),
        torch.nn.LayerNorm(24),
        torch.nn.Linear(24, 8, bias=False),
        torch.nn.Linear(8, 8, bias=False),
    )
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    weights = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    # Named, so shrinking the model back to one matrix fails here rather than
    # quietly narrowing what the loop below covers.
    assert [tuple(w.shape) for w in weights] == [(24, 8), (8, 24), (8, 8)]

    built, _ = axes.assemble(model, config, CPU, framework="native")
    gradients = [spread_gradient(tuple(w.shape)) for w in weights]

    for step in (1, 2):
        updates = muon_step(built, weights, gradients)
        for weight, update, gradient in zip(weights, updates, gradients, strict=True):
            assert_orthogonalised(
                update, gradient, lr=config.optim.lr, where=f"{tuple(weight.shape)} step {step}"
            )


def test_a_muon_that_orthogonalises_nothing_does_not_read_back_as_muon(composed):
    """The record side of the same substitution.

    `use_muon` is a param-group flag and the class name is `Muon` either way, so a
    build that put every group on the internal AdamW side produced a run record
    byte-identical to an honest one: `{'class': 'Muon', 'fused': False,
    'param_groups': 2}`. A published number could not then be attributed to either
    optimizer after the fact. `_capture_optim` now counts the trainable tensors on
    the orthogonalised side, and zero of them is undetermined rather than `muon` —
    which stops a timing run instead of labelling one.
    """
    config = muon(composed, **{"run.purpose": "timing"})
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    )
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    matrices = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    built, _ = axes.assemble(model, config, CPU, framework="native")

    honest = axis(capture(built, config), "optim.name")
    for group in built.optimizer.param_groups:
        group["use_muon"] = False
    disguised_state = capture(built, config)
    disguised = axis(disguised_state, "optim.name")

    assert honest.applied == "muon"
    # Every trainable matrix, not merely a nonzero count: a build that routed one
    # of the three to Newton-Schulz and the rest to the internal AdamW would still
    # report `muon`, and the number is what makes that visible in the run record.
    assert len(matrices) == 3
    assert honest.detail["newton_schulz_tensors"] == len(matrices)
    assert honest.detail["use_muon"] == [True, False]
    assert disguised.applied is None
    assert disguised.detail["newton_schulz_tensors"] == 0
    assert honest.detail != disguised.detail
    with pytest.raises(AppliedMismatch, match=r"optim\.name: .*undetermined"):
        assert_matches(disguised_state, config)


def test_a_frozen_matrix_is_not_counted_as_a_tensor_muon_will_orthogonalise(composed):
    """Muon skips a parameter with no gradient, so a frozen matrix sits in the
    `use_muon` group without ever entering Newton-Schulz. The guard used to count
    it: a model whose every matrix is frozen passed a refusal whose sentence is
    "every tensor that steps would take the AdamW path", which is exactly what
    would then happen — only the 1D tensors would be moving, through the internal
    AdamW, under Muon's name.
    """
    model = torch.nn.Sequential(torch.nn.Linear(3, 3))
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
    model[0].weight.requires_grad_(False)

    with pytest.raises(axes.UnappliedAxis, match="no trainable >=2D parameter"):
        axes.assemble(model, muon(composed), CPU, framework="native")


def test_muon_is_refused_rather_than_crashing_where_pytorch_optimizer_is_absent(composed):
    """`from pytorch_optimizer import Muon` was a bare import, and the documented
    setup command did not install that distribution — so on a clean clone and in
    five of the six framework images this axis did not refuse, it took `assemble`
    down partway through with ModuleNotFoundError. The documented command carries
    it now (`doc-commands` demands this very import of the lock it produces), but
    the five images still do not, and `_patch_liger` wraps its import for the same
    reason: "this environment cannot provide the axis" is an unapplied axis, not a
    crash.
    """
    with pytest.MonkeyPatch.context() as patched:
        patched.setitem(sys.modules, "pytorch_optimizer", None)

        with pytest.raises(axes.UnappliedAxis, match="pytorch-optimizer"):
            axes.assemble(plain_model(), muon(composed), CPU, framework="native")


def test_the_muon_run_reads_back_as_muon(composed):
    config = muon(composed)

    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")

    assert axis(capture(built, config), "optim.name").applied == "muon"


def test_muon_orthogonalises_the_matrices_and_hands_the_rest_to_adamw(composed):
    """Two groups, split on `p.ndim`. Newton-Schulz needs a matrix, so a 1D tensor
    in the Muon group is not a preference but a shape error waiting for the first
    step; and a parameter in neither group is one the run never trains."""
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3))
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    built, _ = axes.assemble(model, muon(composed), CPU, framework="native")

    groups = built.optimizer.param_groups
    assert [group["use_muon"] for group in groups] == [True, False]
    assert all(p.ndim >= 2 for p in groups[0]["params"])
    assert all(p.ndim < 2 for p in groups[1]["params"])
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


def test_an_embedding_table_is_orthogonalised_here_rather_than_handed_to_adamw(composed):
    """The documented deviation, made falsifiable.

    Muon's own documentation says embeddings and the LM head belong to AdamW, and
    this build cannot do that: `_optimizer` is handed `model.parameters()`, which
    carries no names, and an embedding matrix is indistinguishable from a hidden
    weight matrix without them. So the embedding goes through Newton-Schulz, which
    is the condition `docs/methodology.md` §5 puts on reading a Muon row — most of
    all gemma-4's, where PLAN.md's hypothesis is about PLE tables being handed
    *away* from Muon.

    Asserted rather than described so that the day someone gives `_optimizer` the
    names, this test fails and the methodology note is rewritten with the code
    instead of outliving it.
    """
    embedding = torch.nn.Embedding(5, 4)
    model = torch.nn.Sequential(embedding, torch.nn.Linear(4, 4))
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    built, _ = axes.assemble(model, muon(composed), CPU, framework="native")

    muon_group, _ = built.optimizer.param_groups
    assert any(p is embedding.weight for p in muon_group["params"])


def test_under_lora_every_trained_tensor_is_on_the_muon_side(composed):
    """The headline comparison is full finetuning against LoRA, and the two arms do
    not get the same optimizer. Every trainable tensor an adapter adds is 2D, so the
    internal AdamW group holds nothing but frozen 1D tensors — LoRA x muon is pure
    Muon, while full x muon sends norms and biases to AdamW. Recorded in
    `docs/methodology.md` §5 as a condition on comparing the two rows.
    """
    config = bench(
        composed,
        **{"optim.name": "muon", "peft.mode": "lora", "peft.r": 4, "peft.alpha": 8},
    )
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    built, _ = axes.assemble(model, config, CPU, framework="native")

    trained = [p for p in built.model.parameters() if p.requires_grad]
    assert trained and all(p.ndim >= 2 for p in trained)
    muon_group, adamw_group = built.optimizer.param_groups
    assert all(any(p is held for held in muon_group["params"]) for p in trained)
    assert not any(p.requires_grad for p in adamw_group["params"])


def test_muon_is_refused_for_a_model_with_no_matrix_to_orthogonalise(composed):
    """Every tensor would fall through to the internal AdamW, and the run would be
    AdamW measured under Muon's name — the substitution this module exists to
    refuse. `applied._capture_optim` could not catch it: the class is still Muon."""
    model = torch.nn.Sequential(torch.nn.LayerNorm(3))
    model.config = SimpleNamespace(_attn_implementation="sdpa", sub_configs=())

    with pytest.raises(axes.UnappliedAxis, match="no trainable >=2D parameter"):
        axes.assemble(model, muon(composed), CPU, framework="native")


def test_adamw_8bit_off_cuda_is_refused_rather_than_stepped_in_32_bit(composed):
    """bitsandbytes keeps its state in a CUDA kernel. A 32-bit AdamW built here
    would step correctly and be published as 8-bit — which is also what
    `applied._capture_optim` refuses, since it reads this axis off the class."""
    with pytest.raises(axes.UnappliedAxis, match="device=cpu would step a 32-bit AdamW"):
        axes.assemble(
            plain_model(),
            bench(composed, **{"optim.name": "adamw_8bit"}),
            CPU,
            framework="native",
        )


def test_adamw_8bit_builds_the_bitsandbytes_optimizer_without_optim_bits(composed, monkeypatch):
    """`optim_bits` is the trap. Its default is 32 and `AdamW8bit` still quantises,
    because the constructor hardcodes 8 and raises `ValueError` for any other value
    of that argument — so passing 8 "to be explicit" is the one call that fails
    (`bitsandbytes/optim/adamw.py:105-123`). The stub raises exactly as the library
    does, so a version of `_adamw_8bit` that passed it would fail here."""
    built_with = {}

    class AdamW8bit(torch.optim.AdamW):
        def __init__(self, params, lr=1e-3, weight_decay=1e-2, optim_bits=32, **kwargs):
            if optim_bits != 32:
                raise ValueError("AdamW8bit only supports optim_bits=32")
            built_with.update({"lr": lr, "weight_decay": weight_decay, **kwargs})
            super().__init__(params, lr=lr, weight_decay=weight_decay)

    module = ModuleType("bitsandbytes.optim")
    module.AdamW8bit = AdamW8bit
    monkeypatch.setitem(sys.modules, "bitsandbytes.optim", module)
    config = on_cuda(bench(composed, **{"optim.name": "adamw_8bit"}))

    built, names = axes.assemble(plain_model(), config, torch.device("cuda"), framework="native")

    assert "optim.name" in names
    assert type(built.optimizer).__name__ == "AdamW8bit"
    assert built_with == {"lr": config.optim.lr, "weight_decay": config.optim.weight_decay}
    assert axis(capture(built, config), "optim.name").applied == "adamw_8bit"


def test_a_muon_optimizer_under_an_adamw_request_stops_the_run(composed):
    """The capture side of the pair. `scripts/bench.py` calls `assert_matches`
    before `train(...)`, so this is what stops a mislabelled optimizer from
    reaching a number."""
    config = muon(composed, **{"run.purpose": "timing"})
    built, _ = axes.assemble(plain_model(), config, CPU, framework="native")
    requested_adamw = bench(composed, **{"optim.name": "adamw_fused", "run.purpose": "timing"})

    state = capture(built, requested_adamw)

    assert axis(state, "optim.name").applied == "muon"
    with pytest.raises(AppliedMismatch, match=r"optim\.name: requested 'adamw_fused'"):
        assert_matches(state, requested_adamw)


# --- precision.name ----------------------------------------------------------
#
# transformer-engine does not import here: the shim wheel carries the Python
# surface but the compiled half (`transformer-engine-torch`) is sdist-only and
# built on the pod. So these run against a stand-in shaped like the pinned
# release's surface — the two recipe classes, the two availability checks, and
# `autocast` — and what they pin is which of those this module calls and in which
# order. Whether an A100 refuses is a hardware fact, not a code one, and it is the
# gate below that turns it into a refusal instead of a wrong number.


def install_transformer_engine(monkeypatch, *, available=True, tuples=True):
    """A stand-in for the three Transformer Engine modules this axis reads.

    `available` decides what the support checks answer, and `tuples` decides in
    which of the two shapes — the private `_compute_*_support` pair returns
    `(bool, str)` and whether the public wrappers do is 확인 안 함, so both are
    exercised. A `(False, reason)` tuple is truthy, which is exactly how an
    unsupported device would be read as a supported one.
    """
    entered = []
    recipes = ModuleType("transformer_engine.common.recipe")
    for name in ("MXFP8BlockScaling", "NVFP4BlockScaling"):
        setattr(recipes, name, type(name, (), {}))
    quantization = ModuleType("transformer_engine.pytorch.quantization")
    answer = (available, "" if available else "compute capability 10.0 or higher required")
    for name in ("is_mxfp8_available", "is_nvfp4_available"):
        setattr(quantization, name, lambda answer=answer: answer if tuples else answer[0])

    @contextlib.contextmanager
    def autocast(enabled=True, recipe=None):
        entered.append({"enabled": enabled, "recipe": recipe})
        yield

    pytorch = ModuleType("transformer_engine.pytorch")
    pytorch.autocast = autocast
    for name, module in {
        "transformer_engine.common.recipe": recipes,
        "transformer_engine.pytorch.quantization": quantization,
        "transformer_engine.pytorch": pytorch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return entered


@pytest.mark.parametrize(
    "precision,recipe_class", [("mxfp8", "MXFP8BlockScaling"), ("nvfp4", "NVFP4BlockScaling")]
)
def test_an_fp8_recipe_wraps_the_step_and_is_what_the_run_is_read_by(
    composed, monkeypatch, precision, recipe_class
):
    """These recipes keep bf16 parameters and cast inside the step, so the weights
    cannot say which one ran. `Built.precision_recipe` is the only evidence, which
    is why `assemble` builds one as well as `step_context`."""
    entered = install_transformer_engine(monkeypatch)
    config = bench(composed, **{"precision.name": precision})

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")
    with axes.step_context(config):
        pass

    assert "precision.name" in names
    assert type(built.precision_recipe).__name__ == recipe_class
    assert type(entered[0]["recipe"]).__name__ == recipe_class
    assert entered[0]["enabled"] is True
    assert axis(capture(built, config), "precision.name").applied == precision


@pytest.mark.parametrize("tuples", [True, False], ids=["tuple", "bool"])
@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4"])
def test_a_device_that_cannot_execute_the_recipe_is_refused(
    composed, monkeypatch, precision, tuples
):
    """The failure this gate exists for. `check_recipe_support` covers MXFP8 and its
    `elif` chain never mentions NVFP4, so an ungated nvfp4 run on an A100 enters the
    region without complaint and prints a bf16 number under the fp4 label. Both are
    gated here instead, and in both answer shapes — `(False, reason)` is truthy, so
    a gate that read the result as a bool would let every unsupported device
    through.

    `assemble` does not raise: docs/CONTRACTS.md §2 gives the fp8 decision to
    `step_context`, and `scripts/bench.py` enters it before the timer. What
    `assemble` must not do is hand back a recipe, because that is the one thing
    `applied._capture_precision` would certify the run on.
    """
    install_transformer_engine(monkeypatch, available=False, tuples=tuples)
    config = bench(composed, **{"precision.name": precision})

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")

    assert built.precision_recipe is None
    assert "precision.name" not in names
    assert axis(capture(built, config), "precision.name").applied != precision
    with pytest.raises(axes.UnappliedAxis, match="not executable on this device"):
        axes.step_context(config)


def test_both_fp8_precisions_are_gated_and_neither_is_left_to_transformer_engine():
    """The table read as a table: a value here with no checker would be entered
    ungated, and that is the state NVFP4 is in inside Transformer Engine itself."""
    assert axes.TE_PRECISIONS == {
        "mxfp8": ("MXFP8BlockScaling", "is_mxfp8_available"),
        "nvfp4": ("NVFP4BlockScaling", "is_nvfp4_available"),
    }
    assert set(axes.TE_PRECISIONS) | {"bf16"} == set(
        get_args(PrecisionConfig.model_fields["name"].annotation)
    )


def test_a_support_check_that_will_not_answer_is_not_a_device_that_supports_it(
    composed, monkeypatch
):
    """A raising check and a missing one are both "unknown", and unknown has to be
    a refusal: the alternative is a region entered on a device nobody asked."""
    install_transformer_engine(monkeypatch)
    quantization = sys.modules["transformer_engine.pytorch.quantization"]
    config = bench(composed, **{"precision.name": "nvfp4"})

    def raises():
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(quantization, "is_nvfp4_available", raises)
    with pytest.raises(axes.UnappliedAxis, match="raised RuntimeError"):
        axes.step_context(config)

    monkeypatch.delattr(quantization, "is_nvfp4_available")
    with pytest.raises(axes.UnappliedAxis, match="has no is_nvfp4_available"):
        axes.step_context(config)


def test_without_transformer_engine_the_fp8_precisions_are_refused_as_axes(composed):
    """What this host is, stated rather than stubbed. The refusal is `UnappliedAxis`
    rather than an ImportError so the audit reads it as the axis declining a value
    it cannot put into effect, and it names the recipe so a pod record says which
    of the four call sites gave up and why."""
    assert importlib.util.find_spec("transformer_engine") is None

    for precision in ("mxfp8", "nvfp4"):
        config = bench(composed, **{"precision.name": precision})
        with pytest.raises(axes.UnappliedAxis, match="Transformer Engine recipe") as refusal:
            axes.step_context(config)
        assert "not importable here" in str(refusal.value)


def test_bf16_still_enters_no_region_and_carries_no_recipe(composed, monkeypatch):
    """The other half of the pair: a recipe built for every run would put an fp8
    region around the study's baseline, and `_capture_precision` would then read
    every bf16 run as the recipe's name."""
    entered = install_transformer_engine(monkeypatch)
    config = bench(composed)

    built, names = axes.assemble(plain_model(), config, CPU, framework="native")
    with axes.step_context(config):
        pass

    assert built.precision_recipe is None
    assert "precision.name" not in names
    assert entered == []


# --- the last two to be wired ------------------------------------------------


def test_every_axis_the_schema_declares_is_wired():
    """Spelled out rather than compared against `axis_knobs()`, which would make
    it a tautology: adding a knob would extend both sides at once. This is where
    an axis leaving the wired set has to be argued for.

    It replaces `test_precision_and_offload_are_still_unapplied`, which pinned the
    complement — the fifteen that were wired, plus an assertion that these two
    were not. Its own docstring named itself as the thing to delete when they were.
    """
    assert set(axes.IMPLEMENTED) == {
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
        "precision.name",
        "train.gradient_checkpointing",
        "train.offload",
    }
    assert set(axes.IMPLEMENTED) == set(axis_knobs())


def test_an_inert_configuration_claims_to_have_applied_nothing(composed):
    """`compile=none`, `peft=full` and `freeze.*=false` change nothing, so naming
    them would make the applied list a copy of the config."""
    built, names = axes.assemble(plain_model(), bench(composed), CPU, framework="native")

    assert set(names) == {"optim.name", "loss.name", "framework.name"}
    assert built.model is not None


def test_a_quality_run_is_enforced_exactly_like_a_timing_run(composed):
    """`quality` is in `ENFORCED_PURPOSES` and nothing held it there: dropping it
    left the suite green, and a long quality run would then have produced a loss
    curve for settings no one checked."""
    config = bench(composed, **{"run.purpose": "quality"})

    state = capture(Built(model=plain_model()), config)

    with pytest.raises(AppliedMismatch, match="purpose=quality"):
        assert_matches(state, config)


# --- the adapter's two demands on this module --------------------------------
#
# `trainbench/loader.py` states two things it needs and enters neither itself:
# the axes its framework computes inside its own training step, and the numeric
# regime that framework trains in. Both arrive through `assemble`/`step_context`
# because docs/CONTRACTS.md §2 fixes those as the sites, and until this wave
# nothing carried either one across.


def test_an_adapter_can_declare_the_axes_it_owns_and_capture_marks_them(composed):
    """tevatron's `DenseModel.forward` computes the loss and the cross-device
    gather itself (decision 5), so those two axes are not ours to read back. The
    declaration comes off the object the adapter returned — `assemble` puts it on
    `Built` — and `capture` turns it into the third axis state rather than into a
    mismatch that would refuse every tevatron cell."""
    config = bench(composed)

    built, _ = axes.assemble(
        plain_model(),
        config,
        CPU,
        framework="tevatron",
        owned_axes={"loss.name": "DenseModel.forward computes it"},
    )

    assert "loss.name" in built.owned_axes
    state = capture(built, config)
    owned = axis(state, "loss.name")
    assert owned.state == "framework_owned"
    assert owned.owner == "tevatron"
    # Never a certification: an owned axis carries no value, or an adapter would be
    # certifying itself for having declined to look.
    assert owned.applied is None
    # And an enforced run is not refused *over this axis*. Other axes on this CPU
    # host still disagree, which is what the message is read for rather than the
    # exception type — the whole failure being fixed here is `loss.name` alone
    # stopping every tevatron timing run.
    with pytest.raises(AppliedMismatch) as refusal:
        assert_matches(state, config)
    assert "loss.name" not in str(refusal.value)


def test_an_adapter_cannot_own_an_axis_the_boundary_does_not_allow(composed):
    """`FRAMEWORK_OWNABLE` is what bounds the disclaimer. Passing the declaration
    through `assemble` must not become a way around it — an adapter free to own
    `optim.name` could own everything, and capture would certify a run by not
    looking at it."""
    config = bench(composed)

    built, _ = axes.assemble(
        plain_model(),
        config,
        CPU,
        framework="tevatron",
        owned_axes={"optim.name": "not the adapter's to claim"},
    )

    claimed = axis(capture(built, config), "optim.name")
    assert claimed.state != "framework_owned"
    assert claimed.owner is None


def test_the_context_a_framework_requires_is_established_and_really_on(composed):
    """axolotl loads `embed_tokens`/`lm_head` in fp32 beside a bf16 body and needs
    `torch.autocast` for the matmul between them (decision 1). The adapter states
    the requirement and never opens its own `with`; this is the site that enters
    it. Asserted through `torch.is_autocast_enabled`, not through the returned
    object's type — `torch.autocast` constructs happily and then disables itself."""
    required = SimpleNamespace(
        kind="autocast", device_type="cpu", dtype="bfloat16", reason="axolotl trains in autocast"
    )

    assert not torch.is_autocast_enabled("cpu")
    with axes.step_context(bench(composed), required):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
    assert not torch.is_autocast_enabled("cpu")


def test_a_required_context_for_a_device_this_run_is_not_on_is_refused(composed):
    """`torch.autocast(device_type="cuda")` on a host with no CUDA warns and sets
    `enabled=False` (`torch/amp/autocast_mode.py`), so the step would be measured
    outside the framework's own regime under the framework's own label. Refusing is
    the only outcome that does not publish that."""
    required = SimpleNamespace(
        kind="autocast", device_type="cuda", dtype="bfloat16", reason="axolotl trains in autocast"
    )

    with pytest.raises(axes.UnappliedAxis, match="cuda autocast"):
        axes.step_context(bench(composed), required)


def test_a_kind_this_site_cannot_establish_is_refused_by_name(composed):
    """The contract fixes the set of kinds an adapter may ask for; this table is
    the other half. A kind admitted there and missing here would be a contract
    promising a region nothing enters."""
    required = SimpleNamespace(
        kind="graph_capture", device_type="cpu", dtype="bfloat16", reason="invented"
    )

    with pytest.raises(axes.UnappliedAxis, match="STEP_CONTEXT_ESTABLISHERS"):
        axes.step_context(bench(composed), required)


def test_an_fp8_recipe_and_a_required_context_cannot_both_be_in_effect(composed):
    """Two answers to what precision the step runs in. Nesting them would report an
    fp8 number for a step that also ran under bf16 autocast, and the axis would
    still read back as applied because the recipe object exists either way.

    Decided on the requested precision rather than on whether the recipe can be
    built, which is what makes it reachable on a host with no Transformer Engine —
    asking the recipe first put this branch behind an import that fails here.
    """
    required = SimpleNamespace(
        kind="autocast", device_type="cpu", dtype="bfloat16", reason="axolotl trains in autocast"
    )
    config = bench(composed, **{"precision.name": "mxfp8"})

    with pytest.raises(axes.UnappliedAxis, match="two regimes"):
        axes.step_context(config, required)
