"""Where an axis is turned on.

`trainbench/applied.py` reads back what a run ended up with; this module is the
other half — what asks for it. They are separated because they are checked
against each other: an axis is certified only when the code that applies it and
the code that verifies it are both present, and `IMPLEMENTED` here is required to
equal the set of capture probes over there.

The four functions below are the complete set of places a measured run turns an
axis on. Their number and shape are fixed here, in Wave 0, because the harness
that calls them (`scripts/bench.py`) is built later by a different lane: without
an agreed set of call sites, the lane that adds axes and the lane that adds the
harness each invent one.

    patch(config)                           -> applied      BEFORE construction
    load_kwargs(config)                     -> from_pretrained kwargs
    assemble(model, config, device, ...)    -> (Built, applied)
    step_context(config)                    -> context manager around the step

`patch` comes first because some kernels replace classes rather than instances.
Liger's documented sequence is `apply_liger_kernel_to_llama()` and then
"# 2. Instantiate patched model" (Liger README), so a model built before the
patch is a model the axis never reached.

`assemble` returns everything at once rather than exposing a builder per piece,
because some axes do not let the pieces be built separately:
`deepspeed.initialize(model=..., model_parameters=..., training_data=...)`
returns the engine, the optimizer and the dataloader from a single call
(DeepSpeed docs, cifar-10 and bert-pretraining tutorials), so `zero2`/`zero3` and
the offload axis cannot be split across three independent hooks. It returns the
model for the same reason `torch.compile`, `get_peft_model` and FSDP do: they
replace the model rather than mutate it.

`step_context` exists because precision is not only a construction-time choice —
fp8 recipes wrap the forward pass — and an axis with nowhere to live is an axis
that gets applied somewhere unverified.

An axis value with no implementation raises `UnappliedAxis` rather than falling
back to the default. A silent substitution is the failure this whole mechanism
exists to prevent, and it is not less of one for happening in our own code.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.metadata
import importlib.util
from typing import Any

import torch

from trainbench.applied import ENFORCED_PURPOSES, Built
from trainbench.config_schema import BenchConfig
from trainbench.device import get_device
from trainbench.embedding import info_nce

# gemma-4's per-layer embeddings. Every one of the 108 PLE tensors carries this
# in its name — `language_model.embed_tokens_per_layer.weight`,
# `layers.N.per_layer_input_gate`, `per_layer_model_projection`, and so on
# (docs/model-spec.md, read off model.safetensors.index.json). Substring rather
# than an enumerated list because the layer index is part of the name.
PLE_PARAM_MARKER = "per_layer"

# Vision-tower parameters, per architecture. Read off each checkpoint rather than
# guessed: docs/model-spec.md says in as many words that a guessed marker freezes
# zero tensors and records that as success, which is the failure `_ple_report`
# had already shipped once with `altup`.
#
# Measured 2026-08-01 from each repo's safetensors header on `main`:
#
#   Qwen/Qwen3-VL-Embedding-2B   model.visual.*         315 of  625 tensors
#   Qwen/Qwen3.5-0.8B            model.visual.*         153 of  488 tensors
#   google/gemma-4-E2B           model.vision_tower.*   658 of 2011 tensors
#                                model.embed_vision.*     1
#
# Both Qwen models keep the projector inside the tower (`visual.merger`), while
# gemma-4 keeps it outside as `embed_vision.embedding_projection`; it is included
# so the axis means the same thing on all three. gemma-4's `audio_tower` is
# deliberately not here — it is a third tower, not the vision one.
VISION_PARAM_MARKERS = {
    "qwen3_vl": ("visual.",),
    "qwen3_5": ("visual.",),
    "gemma4": ("vision_tower.", "embed_vision."),
}

# Class definitions from these packages inside a built model are the evidence
# that a kernel library patched it. Liger replaces the transformers classes
# (`apply_liger_kernel_to_llama()` then "# 2. Instantiate patched model"), so a
# model built afterwards carries their modules; a model built before it does not.
KERNEL_MODULE_ROOTS = {
    "liger_kernel": "liger",
    "fla": "fla",
    "kernels": "kernels_hub",
}

# Liger's per-architecture entrypoint, by this repo's `model.arch`. The naming
# form is the one the Liger README documents (`apply_liger_kernel_to_llama()`),
# and the suffix is transformers' own `model_type`: `configuration_qwen3_5.py`,
# `configuration_qwen3_vl.py` and `configuration_gemma4.py` in transformers 5.14.1
# declare exactly `qwen3_5`, `qwen3_vl` and `gemma4`, which is why `arch` can be
# used as the suffix rather than mapped through a second table.
#
# UNVERIFIED, and deliberately so rather than silently: liger-kernel cannot be
# installed on the machine this was written on — it depends on triton under a
# `sys_platform == 'linux'` marker and triton publishes no macOS wheel — so the
# spelling below is a hypothesis about a package that could not be imported to
# check it. `_patch_liger` therefore refuses with the `apply_liger_kernel_to_*`
# names the installed package really exports instead of failing with an
# AttributeError, so the first pod that runs this either applies the kernel or
# prints the name to write here.
LIGER_ENTRYPOINTS = {
    # Liger-Kernel#1119 (PLAN.md).
    "qwen3_5": "apply_liger_kernel_to_qwen3_5",
}

# Architectures Liger is known *not* to reach, with the reason. Separate from the
# absence of an entrypoint above because "we know it does not work" and "nothing
# is recorded either way" are different states, and only the first has a citation.
LIGER_UNSUPPORTED = {
    "gemma4": "Liger-Kernel#1186 is open (PLAN.md), so gemma-4 has no Liger path",
}

# Architectures whose transformers implementation takes its kernels from
# flash-linear-attention. Of the three models under test only Qwen3.5 does:
# `models/qwen3_5/modeling_qwen3_5.py` imports `FusedRMSNormGated` from
# `fla.modules` and the Gated DeltaNet ops from `fla.ops.gated_delta_rule`
# (transformers 5.14.1), while `models/qwen3_vl/` and `models/gemma4/` name fla
# nowhere. Requesting fla on the other two would be a run labelled after a library
# whose code the model never enters.
FLA_ARCHS = frozenset({"qwen3_5"})

# What transformers requires before it binds fla at all, mirrored here rather than
# imported because this module never imports transformers (it is absent from the
# orchestrator-side environment). `is_flash_linear_attention_available()` is
# `is_torch_cuda_available() and _is_package_available("fla") and
# version >= 0.2.2` (transformers 5.14.1, `utils/import_utils.py:869`). The floor
# is checked here rather than left to that predicate: below it transformers binds
# nothing, `_capture_kernel` reads `none` back off the model, and the run dies far
# from the version that caused it.
#
# `fla` publishes no entry in transformers' `PACKAGE_DISTRIBUTION_MAPPING`, so
# that predicate resolves the version by distribution name and falls back to
# importing the package — the same two steps, in the same order, as below.
FLA_MIN_VERSION = (0, 2, 2)
FLA_DISTRIBUTIONS = ("flash-linear-attention", "fla")

# Columns whose presence means the rows arrived already tokenised. Read off the
# dataset the loader was built around, because that is where `pretokenize` moves
# the work to — the loader itself looks the same either way.
TOKENIZED_COLUMNS = ("input_ids",)

# Columns that put an image in a row. These are the two `scripts/prepare_data.py`
# writes into both subsets; they are repeated rather than imported because
# `trainbench/` does not import from `scripts/`, and a subset that spells them
# differently is still caught by its declared feature type (`image_columns`).
# What reads this is `loss=cached_mnrl`, which needs to know that a dataset can be
# read as rows at all before it will claim the axis
# (`_gradcache_needs_splittable_data`).
IMAGE_COLUMNS = ("qry_image", "pos_image")

# Batch entries whose leading dimension counts images or image patches rather than
# rows, with what each one counts. Read off the real processors rather than guessed:
# every name and every meaning below was produced by running the processor on
# batches whose rows carried different numbers of images (1/1/1/1, 1/0/2/1, 2/1),
# transformers 5.14.1, 2026-08-02.
#
#   Qwen/Qwen3-VL-Embedding-2B   Qwen3VLProcessor   pixel_values        patches
#   Qwen/Qwen3.5-0.8B            Qwen3VLProcessor   image_grid_thw      images
#   google/gemma-4-E2B           Gemma4Processor    pixel_values        images
#                                                   image_position_ids  images
#
# Both Qwen checkpoints load the same processor class and produced the same two
# keys. Their `pixel_values` is `(sum of t*h*w over the batch's images, 1536)` —
# four images gridded [1,4,4], [1,6,8], [1,8,10] and [1,12,12] came back as
# (288, 1536), and 16+48+80+144 is 288 — while `image_grid_thw` is `(images, 3)`.
# gemma-4 counts images in the leading dimension directly: three images came back
# as pixel_values (3, 2520, 768) and image_position_ids (3, 2520, 2).
#
# `mm_token_type_ids` is deliberately absent. All three processors return it as
# `(rows, sequence)`, so it is row-aligned and the row rule below already covers it.
IMAGE_PAYLOAD_KEYS = ("pixel_values", "image_grid_thw", "image_position_ids")

# The per-image tensor whose rows are that image's `(t, h, w)` patch grid. It is
# what turns a per-row image count into a per-row *patch* count for the Qwen
# processors, whose `pixel_values` counts patches: split this by images, multiply
# each row out, and the cumulative sum is where each batch row's pixels begin.
IMAGE_GRID_KEY = "image_grid_thw"

# What a packed batch carries. `cu_seqlens` and `seq_lengths` are boundaries, not
# model inputs: `model(**tensors)` would reject them, so the harness has to lift
# them out of the batch before the forward pass (they belong to pooling and to
# token accounting). Named here so the collate and its consumers agree on one
# spelling.
PACKED_BOUNDARY_KEYS = ("cu_seqlens", "seq_lengths")

# Axis knobs this module can actually put into effect. Required to equal
# `applied._CAPTURES`: `audit_plan.py`'s `axis-wired` check enforces it, and
# `tests/test_applied.py::test_applied_and_verified_sets_agree` pins it.
#
# `precision.name` and `train.offload` sit here on the same terms as
# `dataloader.packing`: this module decides them by refusing every value it cannot
# put into effect, and the inert value it does accept needs nothing done to it.
# `step_context` refuses every precision but bf16, and bf16 needs no autocast
# region only because the weights are already in it; `assemble` refuses every
# offload but none, and none is an optimizer built where the model is. Both of
# those are premises rather than actions, which is why they were left out until
# `applied._capture_precision` and `_capture_offload` began reading them back off
# the model and the optimizer. An axis is wired when something applies it and
# something checks it, and for these two the checking half was the missing one.
IMPLEMENTED = frozenset(
    {
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
)


class UnappliedAxis(RuntimeError):
    """An axis value that nothing here can put into effect.

    Raised instead of returning the default, because the default would then be
    measured under the requested value's name.
    """


def patch(config: BenchConfig) -> list[str]:
    """Axes that must be applied before the model exists.

    Kernel libraries monkey-patch the transformers classes, so patching after
    construction leaves the already-built modules untouched: the run would report
    a kernel that never ran.

    Each value routes to one patcher, and a value with no patcher is refused here
    rather than reaching a `KeyError` further in: a new kernel added to the schema
    has to be turned on by something, and a missing entry has to say so in the
    vocabulary the rest of this module refuses in.

    `none` is refusable too, and for the same reason every other value is: it is a
    claim about the model, not about whether this function did anything. Where the
    architecture takes a kernel library from the environment
    (`_environment_bound_kernel`), a run that would report a number is stopped here
    rather than left to die at `assert_matches` after the model is built — same
    refusal, at the site that can name the cause.

    That one is limited to the purposes whose numbers are reported, unlike every
    other refusal here. The others decline a value the run asked for; this one
    declines because of what the image contains, and a probe is precisely the run
    that exists to find that out (docs/CONTRACTS.md §2 does not block probe or
    profile). Refusing them would delete the channel that reports it.

    It is also asked of every value and not only of `none`. The environment binds
    its library whatever the run requested, so on such an architecture `liger` is
    a model made of both — `mixed(fla,liger)`, which is no setting — and reading
    the binding only under `none` left that case to die at `assert_matches`, a
    model-build away from the image that caused it.

    What is proven and what is not, kept apart because only the first is evidence:
    the routing, the refusals and the per-architecture support table are exercised
    by `tests/test_axes.py` against stub modules. The patch calls themselves are
    not — none of `liger_kernel`, `fla` or `kernels` installs on the machine this
    was written on (liger and fla need triton, which has no macOS wheel; `kernels`
    fetches device-specific prebuilt kernels from the Hub). Whether a real patch
    takes inside a framework image is a first-pod check, and `applied._capture_kernel`
    is what will answer it — it reads the packages that defined the built model's
    module classes, so a patch that did not take reports `none` against a request
    for `liger` and blocks the run.
    """
    name = config.kernel.name
    reported = config.run.purpose in ENFORCED_PURPOSES
    if reported and (bound := _environment_bound_kernel(config)) and bound != name:
        raise UnappliedAxis(_environment_bound_refusal(config, name, bound))
    if name == "none":
        return []
    patcher = KERNEL_PATCHERS.get(name)
    if patcher is None:
        raise UnappliedAxis(
            f"kernel={name} has no patcher in KERNEL_PATCHERS; nothing here can turn it on."
        )
    return patcher(config)


def _patch_liger(config: BenchConfig) -> list[str]:
    """Liger-Kernel, applied by replacing the transformers classes for one
    architecture before anything is instantiated.

    The architecture is decided before the import, so an unsupported model is
    refused for being unsupported rather than for whatever the environment happens
    to be missing. gemma-4 is the case that matters: Liger-Kernel#1186 is open, and
    a patcher that silently no-ops there would put a `liger` label on a stock run.

    transformers ships its own Liger integration, and it is not this one:
    `integrations/liger.py::apply_liger_kernel` takes a built model and calls
    `_apply_liger_kernel_to_instance(model=...)`. That is the post-construction
    half of the same library; the call site docs/CONTRACTS.md §2 fixes for
    `kernel.name` is this one, which is also the sequence the Liger README
    documents ("# 2. Instantiate patched model").
    """
    arch = config.model.arch
    if reason := LIGER_UNSUPPORTED.get(arch):
        raise UnappliedAxis(f"kernel=liger on arch={arch}: {reason}.")
    entrypoint = LIGER_ENTRYPOINTS.get(arch)
    if entrypoint is None:
        raise UnappliedAxis(
            f"kernel=liger records no entrypoint for arch={arch!r}; known: "
            f"{sorted(LIGER_ENTRYPOINTS)}. Nothing recorded is not the same as supported — "
            "add it once a run has shown Liger reaching this architecture."
        )
    try:
        module = importlib.import_module("liger_kernel.transformers")
    except ImportError as exc:
        raise UnappliedAxis(
            f"kernel=liger needs liger-kernel installed before the model is built ({exc}). "
            "It is in envs/native; a run without it would be measuring stock kernels."
        ) from exc
    apply = getattr(module, entrypoint, None)
    if not callable(apply):
        exported = sorted(n for n in dir(module) if n.startswith("apply_liger_kernel_to_"))
        raise UnappliedAxis(
            f"liger_kernel.transformers has no {entrypoint}(); it exports {exported}. "
            "LIGER_ENTRYPOINTS holds a name that could not be checked against an installed "
            "package — correct it from this list rather than patching a different model."
        )
    apply()
    return ["kernel.name"]


def _patch_fla(config: BenchConfig) -> list[str]:
    """flash-linear-attention, whose application is a precondition rather than a call.

    There is no `apply_fla_to_*`. transformers binds fla while it imports the
    modelling module — `from fla.modules import FusedRMSNormGated` under
    `is_flash_linear_attention_available()` — and picks the fused Gated DeltaNet
    path per layer at construction. So the only thing a pre-construction site can
    do is refuse a run whose fast path will not be there, and that refusal is the
    axis: without fla the layers that are 75% of Qwen3.5 fall back to the torch
    implementation behind a single log line, no exception and no warning
    (docs/support-matrix.md), and the number would be published as fla's.

    Nothing is asserted about the run beyond this. `applied._capture_kernel` is
    what decides whether fla is in the built model, and it can see it: the fast
    path installs `fla.modules.FusedRMSNormGated`, a class from the fla package,
    as a submodule.
    """
    arch = config.model.arch
    if arch not in FLA_ARCHS:
        raise UnappliedAxis(
            f"kernel=fla on arch={arch}: transformers takes no fla kernel path for this "
            f"architecture; only {sorted(FLA_ARCHS)} import fla. The run would carry the label "
            "and none of the library."
        )
    available, reason = _fla_fast_path()
    if not available:
        raise UnappliedAxis(
            f"kernel=fla cannot take the Gated DeltaNet fast path here: {reason}. transformers "
            "falls back to the torch implementation with one log line, so the run would measure "
            "the fallback under fla's name."
        )
    return ["kernel.name"]


def _fla_binding() -> tuple[bool, str]:
    """Whether transformers will bind fla's classes while it imports the model.

    Mirrors `is_flash_linear_attention_available()` alone — package, version floor,
    CUDA — and nothing else. This is the predicate that decides whether
    `modeling_qwen3_5.py:409` builds each gated norm as `fla.modules
    .FusedRMSNormGated` or as the stock `Qwen3_5RMSNormGated`, and it is what
    `_capture_kernel` ends up reading off the built model.

    `causal_conv1d` is deliberately not here. It is a second, independent predicate
    (`is_causal_conv1d_available()`, `import_utils.py:875`) behind a second import
    guard (`modeling_qwen3_5.py:68` and `:73`), so a run can have fla's classes in
    the model and still not take the fused Gated DeltaNet path.
    """
    if importlib.util.find_spec("fla") is None:
        return False, "fla not installed"
    installed = _fla_version()
    if installed is None:
        return False, (
            f"fla is installed but its version cannot be read from {list(FLA_DISTRIBUTIONS)} "
            "or fla.__version__, so the floor transformers applies cannot be checked"
        )
    if installed < FLA_MIN_VERSION:
        return False, (
            f"fla {'.'.join(map(str, installed))} is below the "
            f"{'.'.join(map(str, FLA_MIN_VERSION))} transformers requires, so transformers "
            "binds nothing from it"
        )
    if not torch.cuda.is_available():
        return False, "no CUDA device, and transformers gates the fla import on one"
    return True, ""


def _fla_version() -> tuple[int, ...] | None:
    """The installed fla release, as leading integers, or None if unreadable.

    Only the release part is compared: the floor is a release and a suffix
    (`0.5.0.dev0`, `0.4.1+cu126`) never changes which side of it a version is on.

    The fallback import is the one step here that runs somebody else's code, and
    fla runs a lot of it — it imports triton at module scope, which raises
    whatever triton raises on a box without a working device. `ImportError` means
    the package is not importable and the caller reads that as absent; anything
    else means the package is there and broken, which is neither "fla binds" nor
    "fla is absent" and cannot be answered by returning either. It becomes the
    axis's own refusal so that `patch` stops the run at the site that can name the
    cause, instead of letting a triton error out of a function whose contract is
    a version number.
    """
    for distribution in FLA_DISTRIBUTIONS:
        try:
            return _release(importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        declared = getattr(importlib.import_module("fla"), "__version__", None)
    except ImportError:
        return None
    except Exception as exc:
        raise UnappliedAxis(
            f"fla is installed but importing it raises {type(exc).__name__}: {exc}. "
            "transformers imports it at module scope while it builds Qwen3.5, so nothing here "
            "can tell whether the model would carry fla's classes — and an image where the "
            "import itself fails measures no kernel value at all."
        ) from exc
    return _release(declared) if declared else None


def _release(version: str) -> tuple[int, ...] | None:
    parts: list[int] = []
    for part in version.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def _fla_fast_path() -> tuple[bool, str]:
    """Whether transformers will take the fused Gated DeltaNet path here.

    `modeling_qwen3_5.py:219` builds `is_fast_path_available` out of both imports —
    the causal-conv1d functions and fla's chunked/recurrent gated delta rule — and
    `:426` logs one line and runs the torch implementation when either is missing.
    So this is the conjunction of two predicates, which is also what
    `scripts/audit_plan.py` records as this axis's packages
    (`flash-linear-attention`, `causal-conv1d`).
    """
    bound, reason = _fla_binding()
    if not bound:
        return False, reason
    if importlib.util.find_spec("causal_conv1d") is None:
        return False, "causal_conv1d not installed"
    return True, ""


def _environment_bound_kernel(config: BenchConfig) -> str:
    """The kernel library this run gets whether or not it asked for one.

    fla is not applied by a call: transformers imports it at module scope and
    builds Qwen3.5's gated norms out of it, so on an image that has it there is no
    such thing as a Qwen3.5 run without it. That makes `kernel=none` — the default
    of every run in `configs/config.yaml` — a request nothing can satisfy on that
    architecture, and every image in `envs/` ships fla (docs/support-matrix.md).
    """
    if config.model.arch not in FLA_ARCHS:
        return ""
    bound, _ = _fla_binding()
    return "fla" if bound else ""


def _environment_bound_refusal(config: BenchConfig, name: str, bound: str) -> str:
    """Why a requested kernel cannot be what this image measures.

    Two consequences, because the requested value decides which one the run would
    have hit. `none` would publish a model made of the bound library under the
    label `none`; any other value would publish a model made of both, which
    `_capture_kernel` reports as `mixed(...)` and no setting names.
    """
    head = (
        f"kernel={name} on arch={config.model.arch}: transformers binds {bound} while it "
        "imports the modelling module, so this run would build a model made of that "
    )
    if name == "none":
        tail = (
            "library's classes and report it as `none`. Nothing here can unbind it — "
            f"kernel={bound} is what this environment measures, and a run without it needs "
            "an image that does not ship the package."
        )
    else:
        mixed = ",".join(sorted({bound, name}))
        tail = (
            f"library's classes as well as {name}'s. Nothing here can unbind it, so the model "
            f"comes out mixed({mixed}) and would be refused after it was built — "
            f"kernel={bound} is what this environment measures, and a kernel={name} run needs "
            "an image that does not ship the package."
        )
    return head + tail


def _patch_kernels_hub(config: BenchConfig) -> list[str]:
    """kernels-hub dispatch, refused because its entrypoints are not this call site.

    transformers 5.14.1 turns hub kernels on in two places, both of which need the
    model: `from_pretrained(use_kernels=True)`, which ends in
    `model.set_use_kernels(...)` (`modeling_utils.py`), and
    `integrations/hub_kernels.py::kernelize(model)`, which reads `model.device` to
    pick the device-specific kernel to fetch. The one pre-construction knob,
    `USE_HUB_KERNELS`, is read when that module is first imported and only ever
    turns dispatch *off*.

    So this value cannot be applied from `patch` without pretending, and moving it
    to `load_kwargs` or `assemble` is a contract change (docs/CONTRACTS.md §2
    assigns `kernel.name` to this site). Refused until that is decided, rather than
    applied at a site the contract does not name.
    """
    raise UnappliedAxis(
        "kernel=kernels_hub is turned on by from_pretrained(use_kernels=True) and "
        "integrations.hub_kernels.kernelize(model), both of which need a model; the patch site "
        "this axis is assigned to (docs/CONTRACTS.md §2) runs before one exists."
    )


# The dispatch itself. `none` is not here: it is the absence of a patch, and an
# entry for it would be a function that exists to do nothing.
KERNEL_PATCHERS = {
    "liger": _patch_liger,
    "fla": _patch_fla,
    "kernels_hub": _patch_kernels_hub,
}


# QLoRA's base quantisation, in the spelling `from_pretrained` takes. Fixed here
# rather than offered as config: the axis this study measures is `peft.mode`, and
# a knob for the quantiser's internals would be a new schema field, which
# docs/CONTRACTS.md §5 makes a contract change. The values are QLoRA's own recipe —
# 4-bit NF4 weights with the quantisation constants themselves quantised. The
# compute dtype is bf16 because that is the only precision this module lets a run
# reach (`step_context` refuses the rest), so a different one here would compute in
# a precision no axis asked for.
QLORA_4BIT = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
}


def load_kwargs(config: BenchConfig) -> dict[str, Any]:
    """Keyword arguments for `from_pretrained`.

    Attention is set here rather than afterwards because transformers validates
    and may downgrade the request during construction; setting it later would mean
    the model was built once with the wrong one.

    `peft.mode=qlora` asks for its base quantisation here for the same reason and a
    stronger one: 4-bit weights are produced while the checkpoint is being read, and
    a model already materialised in bf16 cannot be turned into one afterwards.

    Only the request is built here, and only where it can be honoured. bitsandbytes
    quantises on CUDA, so on any other device this refuses instead of returning a
    config whose effect is unknown — an ignored quantisation is the outcome that
    matters, because the run would then train a full-precision base and report its
    speed under the qlora label. The adapter half is refused separately and
    unconditionally in `_peft`, so no qlora run starts from this checkout on any
    device; this is the earlier of the two gates and the one that survives if that
    refusal is ever lifted.
    """
    kwargs: dict[str, Any] = {"attn_implementation": config.attn.impl}
    if config.peft.mode == "qlora":
        device = get_device(config.device)
        if device.type != "cuda":
            raise UnappliedAxis(
                f"peft.mode=qlora needs a 4-bit base and bitsandbytes quantises on CUDA; "
                f"device={device.type} would load the base in full precision and measure "
                "that under the qlora label."
            )
        # Imported here rather than at module scope: this is the only path that
        # needs transformers, and the probe adapters that call `load_kwargs` are
        # meant to be importable without it.
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(**QLORA_4BIT)
    return kwargs


def ple_parameters(model: Any) -> list[tuple[str, Any]]:
    """The per-layer embedding tensors, by name."""
    return [(n, p) for n, p in model.named_parameters() if PLE_PARAM_MARKER in n]


def vision_parameters(model: Any, arch: str) -> list[tuple[str, Any]]:
    """The vision tower's tensors, by name, for this architecture.

    Raises for an architecture with no measured marker rather than returning
    nothing: an empty list would freeze nothing and read as a tower that happens
    to be small, and `freeze.vision_tower` would then be reported as applied.
    """
    markers = VISION_PARAM_MARKERS.get(arch)
    if markers is None:
        raise UnappliedAxis(
            f"no vision-tower parameter marker is recorded for arch={arch!r}; "
            f"known: {sorted(VISION_PARAM_MARKERS)}. Read it off the checkpoint's "
            "safetensors header before adding one (docs/model-spec.md)."
        )
    return [
        (name, param)
        for name, param in model.named_parameters()
        if any(marker in name for marker in markers)
    ]


def assemble(
    model: Any,
    config: BenchConfig,
    device: torch.device,
    framework: str,
    dataset: Any = None,
) -> tuple[Built, list[str]]:
    """Build everything a run needs, and report which axes that put into effect.

    `framework` is passed in by the adapter that is running rather than read from
    the config: the config says which framework was requested, and the whole point
    of the capture side is that the request is not evidence of what ran.

    Does not report success beyond naming the axes it applied — `applied.capture`
    decides that by inspecting the result. A function that both acts and certifies
    its own action cannot catch the case where the action did not take.
    """
    applied: list[str] = []
    if config.parallel.strategy in ("zero2", "zero3") or config.train.offload != "none":
        raise UnappliedAxis(
            f"parallel={config.parallel.strategy} / offload={config.train.offload} needs "
            "deepspeed.initialize, which returns the model, optimizer and dataloader "
            "together; it has to be built here rather than by the pieces below."
        )
    if config.parallel.strategy != "single":
        raise UnappliedAxis(
            f"parallel.strategy={config.parallel.strategy} wraps the model (DDP, FSDP2) "
            "and needs an initialised process group; not implemented."
        )

    model, names = _apply_to_model(model, config)
    applied += names
    optimizer, names = _optimizer(model.parameters(), config, device)
    applied += names
    # Before the loss is built, not after: an axis that no batch of this run's data
    # can turn on is refused here rather than named applied and crashed at step 1.
    if config.loss.name == "cached_mnrl":
        _gradcache_needs_splittable_data(dataset)
    loss, names = _loss(config)
    applied += names
    loader, names = _dataloader(dataset, config)
    applied += names

    built = Built(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        loss_fn=loss,
        framework=framework,
    )
    return built, [*applied, "framework.name"]


def step_context(config: BenchConfig) -> contextlib.AbstractContextManager:
    """Context wrapping one training step.

    bf16 needs none: the model is already loaded in that dtype, so an autocast
    region would be a second, different answer to the same question. The fp8
    recipes do need one, and refusing here is what keeps a bf16 step from being
    measured under their name.
    """
    if config.precision.name != "bf16":
        raise UnappliedAxis(
            f"precision={config.precision.name} needs a Transformer Engine recipe "
            "around the forward pass, which is not implemented."
        )
    return contextlib.nullcontext()


def _apply_to_model(model: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """Axes that change the model itself. May return a different object.

    One ordering constraint is real and one is a structuring choice, and they are
    labelled here because an invented reason is worse than none.

    Real: freezing runs before peft, because peft freezes every base parameter and
    the freeze axes would have nothing left to decide afterwards; and all of it
    runs before the optimizer is built (docs/CONTRACTS.md §2 fixes this — FSDP2
    needs the optimizer built over sharded parameters), so the optimizer holds the
    parameters the run actually trains.

    A choice: `_compile` is last because it is the only site that replaces the
    object rather than mutating it, so keeping it last means every other site
    receives the model it was handed. This is not a correctness requirement. An
    earlier version of this docstring claimed checkpointing had to precede compile
    because "the compiled wrapper is not that model"; that is false —
    `OptimizedModule.__getattr__` delegates to `_orig_mod`, so the hook reaches
    through, the flags get set on the inner modules, and `named_modules()` still
    finds them. Reversing the two leaves the suite green. Whether the reverse
    order costs anything at run time is unmeasured.

    An axis whose configured value is the inert one — `compile=none`,
    `peft=full`, `freeze.*=false` — applies nothing and so is not named here.
    What it means is read back by `applied.capture`, which looks at the object
    rather than at this list.
    """
    applied: list[str] = []
    applied += _freeze(model, config)
    model, names = _peft(model, config)
    applied += names
    applied += _gradient_checkpointing(model, config)
    model, names = _compile(model, config)
    applied += names
    return model, applied


def _freeze(model: Any, config: BenchConfig) -> list[str]:
    """Turn off gradients for the tensors each freeze axis names.

    A marker that matches nothing is left to the capture probe rather than raised
    on here: `applied._capture_freeze_ple` reports zero matches as undetermined,
    which blocks a reportable run without stopping a probe whose job is to find
    out what the checkpoint actually contains.
    """
    applied: list[str] = []
    if config.freeze.ple:
        for _, param in ple_parameters(model):
            param.requires_grad_(False)
        applied.append("freeze.ple")
    if config.freeze.vision_tower:
        for _, param in vision_parameters(model, config.model.arch):
            param.requires_grad_(False)
        applied.append("freeze.vision_tower")
    return applied


def _peft(model: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """Adapter attachment. `full` attaches nothing, which is the whole of it.

    The freeze collision `docs/CONTRACTS.md` §2 left open is settled, by measurement
    rather than by choosing: `get_peft_model` sets `requires_grad=False` on every
    base parameter, and the result is byte-identical whether or not a freeze axis
    ran first. So there is no combined meaning to define — under an adapter the
    freeze axes have no state to be in, and `config_schema.py` refuses the
    combination outright instead of letting two identical models occupy two rows of
    the ablation table.

    `qlora` stays refused. It is LoRA over a 4-bit base, so the adapter half is this
    same call and the quantisation half is a `BitsAndBytesConfig` that belongs in
    `load_kwargs` — and bitsandbytes only quantises on CUDA. Implementing it here
    from a machine that cannot build one would mean shipping the path unrun.
    """
    if config.peft.mode == "full":
        return model, []
    if config.peft.mode == "qlora":
        raise UnappliedAxis(
            "peft.mode=qlora needs a 4-bit base, which is a BitsAndBytesConfig passed to "
            "from_pretrained and a CUDA device; neither is exercised here, so it is "
            "refused rather than run as plain LoRA under a QLoRA label."
        )

    from peft import LoraConfig, get_peft_model

    # all-linear rather than a per-architecture list. Naming target modules per
    # model would make the axis mean something different for each of the three,
    # and this benchmark compares the same request across them.
    adapted = get_peft_model(
        model,
        LoraConfig(
            r=config.peft.r,
            lora_alpha=config.peft.alpha,
            lora_dropout=config.peft.dropout,
            target_modules="all-linear",
        ),
    )
    return adapted, ["peft.mode"]


# Which operators `gradient_checkpointing=selective` keeps rather than recomputes.
# The policy *is* the axis: two different save lists measure two different
# techniques, so this one is transcribed rather than chosen. It is torch's own
# `compute_intensive_ops` (torch 2.13.0,
# `torch/_functorch/partitioners.py::get_default_op_list`) — the classification the
# min-cut partitioner already uses to decide what is too expensive to recompute.
# Taking it whole leaves no per-operator judgement of ours to defend, and it is the
# list the published "save the matmuls" policy is made of.
#
# Kept verbatim including `convolution_backward`, which cannot appear inside the
# forward region a checkpoint context wraps: dropping entries would turn a
# transcription back into a selection.
#
# Packets rather than overloads because that is the granularity torch classifies
# at (`get_aten_target` reduces a node's target to its `overloadpacket`), and
# because a list of overloads would silently stop covering `aten.mm.out` and
# friends. `create_selective_checkpoint_contexts` rejects packets in list form, so
# the comparison goes through a policy function instead.
SELECTIVE_CHECKPOINT_SAVED_OPS = (
    torch.ops.aten.mm,
    torch.ops.aten.convolution,
    torch.ops.aten.convolution_backward,
    torch.ops.aten.bmm,
    torch.ops.aten.addmm,
    torch.ops.aten._scaled_dot_product_flash_attention,
    torch.ops.aten._scaled_dot_product_efficient_attention,
    torch.ops.aten._flash_attention_forward,
    torch.ops.aten._efficient_attention_forward,
    torch.ops.aten.upsample_bilinear2d,
    torch.ops.aten._scaled_mm,
)


def selective_checkpoint_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
    """Save the compute-intensive operators, recompute everything else.

    `MUST_SAVE` rather than `PREFER_SAVE`: torch documents `MUST_*` as the form
    "the policy should not be overridden by other subsystems like torch.compile"
    (`torch/utils/checkpoint.py`, `CheckpointPolicy`). `compile.mode` is crossed
    with this axis in the ablation, and under `PREFER_SAVE` the compiled cells
    would be measuring inductor's partitioner decision under this axis's label.
    """
    packet = getattr(op, "overloadpacket", op)
    if packet in SELECTIVE_CHECKPOINT_SAVED_OPS:
        return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
    return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE


def _gradient_checkpointing(model: Any, config: BenchConfig) -> list[str]:
    """Trade compute for activation memory.

    `use_reentrant=False` is not a tuning choice here: the reentrant variant skips
    the recomputation entirely when no input to a checkpointed block requires
    grad, which is exactly what a frozen vision tower produces — and `freeze.*`
    and this axis are crossed in the ablation. `selective` additionally has no
    choice: `context_fn` is only honoured by the non-reentrant implementation.

    `selective` goes through the same hook as `full` rather than walking the module
    tree, because transformers turns `gradient_checkpointing_kwargs` into
    `functools.partial(torch.utils.checkpoint.checkpoint, **kwargs)` and hands that
    partial to every block that checkpoints (`PreTrainedModel._set_gradient_checkpointing`,
    transformers 5.14.1). So the difference between the two values is entirely the
    `context_fn` in those kwargs, and that partial is also what
    `applied._capture_gradient_checkpointing` reads back off the modules.
    """
    mode = config.train.gradient_checkpointing
    if mode == "none":
        return []
    if mode not in ("full", "selective"):
        raise UnappliedAxis(f"train.gradient_checkpointing={mode} has no implementation here.")
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise UnappliedAxis(
            f"{type(model).__name__} has no gradient_checkpointing_enable, so nothing here "
            f"can turn train.gradient_checkpointing={mode} on for it."
        )
    kwargs: dict[str, Any] = {"use_reentrant": False}
    if mode == "selective":
        kwargs["context_fn"] = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts,
            selective_checkpoint_policy,
        )
    enable(gradient_checkpointing_kwargs=kwargs)
    return ["train.gradient_checkpointing"]


def _compile(model: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """`torch.compile`, whole-model or per repeated block.

    The schema's values other than `none` are torch's own mode spellings, so they
    are passed through rather than translated; a translation table would be a
    second place for `max-autotune` to be spelled. `none` is spelled that way
    because YAML reads a bare `off` as boolean False.

    Regional compilation goes through the model's own `compile_repeated_blocks`
    rather than a walk of the module tree: which blocks repeat is a property of
    the architecture, and guessing it would compile the wrong thing under the
    right name.
    """
    mode = config.compile.mode
    if mode == "none":
        return model, []
    if mode == "regional":
        compile_blocks = getattr(model, "compile_repeated_blocks", None)
        if not callable(compile_blocks):
            raise UnappliedAxis(
                f"compile=regional needs {type(model).__name__}.compile_repeated_blocks, "
                "which this model does not have."
            )
        compile_blocks()
        return model, ["compile.mode"]
    return torch.compile(model, mode=mode), ["compile.mode"]


def _optimizer(params: Any, config: BenchConfig, device: torch.device) -> tuple[Any, list[str]]:
    """`fused` follows the device: the fused AdamW kernel is CUDA-only and asking
    for it on CPU raises. The capture probe reports the unfused case as a
    different applied value, so a CPU run cannot report a fused number.

    Muon comes from `pytorch-optimizer` rather than from a copy written here.
    `envs/native/pyproject.toml` pins that distribution for this axis and
    `audit_plan.py`'s `AXIS_PACKAGES` maps `optim/muon` onto it, so a local
    reimplementation would make both of those records false while changing what
    gets measured: the number this benchmark publishes should be the one a
    practitioner gets from the library everyone installs, not from our own
    Newton-Schulz loop.

    **How the parameters are split, and what that qualifies.** Muon orthogonalises
    a >=2D update and has nothing to do to a 1D one, so the split here is
    `p.ndim >= 2` into the Muon group and everything below it into the internal
    AdamW group — two param groups, which is what `applied._capture_optim` records
    as `param_groups`. The usual further exclusion — embedding tables and the LM
    head to AdamW as well — is **not** applied, because it cannot be: this function
    receives `model.parameters()`, an iterable with no names attached, and every
    way of telling an embedding matrix from a hidden weight matrix needs the names
    or the modules. `docs/methodology.md` states the consequence as a condition on
    reading any Muon row, because it is not a code detail: under this split
    gemma-4's PLE tables go *through* Newton-Schulz rather than around it, which is
    the opposite of the arrangement PLAN.md's gemma-4 hypothesis is about.

    `adamw_8bit` stays refused on the same grounds as `peft.mode=qlora`:
    bitsandbytes' 8-bit optimizer state is a CUDA kernel, so implementing it from a
    machine that cannot run one would mean shipping the path unrun.
    """
    if config.optim.name == "adamw_fused":
        built = torch.optim.AdamW(
            params,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            fused=device.type == "cuda",
        )
        return built, ["optim.name"]
    if config.optim.name != "muon":
        raise UnappliedAxis(
            f"optim={config.optim.name} has no implementation here; adamw_8bit is "
            "bitsandbytes' 8-bit optimizer state, which is a CUDA kernel — the package is "
            "in envs/native but nothing here can run one, so it stays refused rather than "
            "measured as plain AdamW under an 8-bit label."
        )

    # `pytorch-optimizer` is pinned by `envs/native/pyproject.toml` and by the root
    # `native` extra, which the documented setup command installs — `doc-commands`
    # collects this lazy import and demands it of that lock. Five of the six
    # framework images still lack it. Refused rather than raised for the same
    # reason `_patch_liger` wraps its import: an axis the environment cannot
    # provide is an unapplied axis, and a bare ModuleNotFoundError takes down
    # `assemble` mid-way instead of naming the one axis unavailable here.
    try:
        from pytorch_optimizer import Muon
    except ImportError as exc:
        raise UnappliedAxis(
            f"optim=muon needs pytorch-optimizer, which is not importable here ({exc}). "
            "It is pinned by envs/native and by the root 'native' extra, which the "
            "documented setup command installs; five of the six framework images do not."
        ) from exc

    held = list(params)
    orthogonalised = [p for p in held if p.ndim >= 2]
    elementwise = [p for p in held if p.ndim < 2]
    # Counted over the *trainable* tensors: a frozen parameter has no gradient, so
    # Muon skips it at the step and it never enters Newton-Schulz. Counting it here
    # would let a model whose every matrix is frozen past a guard whose whole
    # sentence is that the run would then be AdamW under Muon's name — which is
    # exactly what it would be, since only the 1D tensors would still be stepping.
    if not any(p.requires_grad for p in orthogonalised):
        raise UnappliedAxis(
            "optim=muon orthogonalises the >=2D tensors and hands the rest to an internal "
            "AdamW; this model has no trainable >=2D parameter, so every tensor that steps "
            "would take the AdamW path and the run would be AdamW measured under Muon's name."
        )
    # `use_muon` is required on every group — `Muon.__init__` raises without it —
    # and it is what routes a group to Newton-Schulz or to the internal AdamW.
    # Both groups take the configured lr: `OptimConfig` has one, and giving the
    # AdamW half a second, invented one would put a number in the measured path
    # that no config records. Muon rescales it per tensor internally
    # (`get_adjusted_lr`), which is the library's behaviour and not ours to
    # restate here.
    groups: list[dict[str, Any]] = [
        {
            "params": orthogonalised,
            "use_muon": True,
            "lr": config.optim.lr,
            "weight_decay": config.optim.weight_decay,
        }
    ]
    if elementwise:
        groups.append(
            {
                "params": elementwise,
                "use_muon": False,
                "lr": config.optim.lr,
                "weight_decay": config.optim.weight_decay,
            }
        )
    return Muon(groups), ["optim.name"]


def _dataloader(dataset: Any, config: BenchConfig) -> tuple[Any, list[str]]:
    """DALI replaces the DataLoader with its own iterator rather than configuring
    one (DALI docs: "replacing the standard DataLoader with DALIClassificationIterator"),
    which is why this builds the loader instead of returning kwargs for it."""
    if config.dataloader.backend != "torch":
        raise UnappliedAxis(
            f"dataloader.backend={config.dataloader.backend} builds its own iterator; "
            "not implemented."
        )
    if dataset is None:
        return None, []
    applied = ["dataloader.backend"]

    # packing lives in the collate, which is the only place a batch is assembled —
    # and the only place `applied._capture_dataloader_packing` looks. Left at None
    # the loader takes torch's own collate, which pads and cannot pack; that is
    # positive evidence of `False` rather than the absence of evidence.
    collate = None
    if config.dataloader.packing:
        collate = PackedCollate()
        applied.append("dataloader.packing")

    if config.dataloader.pretokenize:
        if not tokenized_columns(dataset):
            raise UnappliedAxis(
                f"dataloader.pretokenize=true, but the dataset declares "
                f"{declared_columns(dataset)} and none of {list(TOKENIZED_COLUMNS)}; "
                "tokenising needs the model's processor, which this module never sees, so "
                "the caller pretokenises with axes.pretokenize(dataset, encode) before "
                "assemble. Building the loader anyway would leave the tokenisation inside "
                "the timed step under a pretokenized label."
            )
        if not tokenized_row(dataset):
            raise UnappliedAxis(
                f"dataloader.pretokenize=true and the dataset declares "
                f"{declared_columns(dataset)}, but its first row carries none of "
                f"{list(TOKENIZED_COLUMNS)}. Column names are what a dataset says about "
                "itself; the row is what the timed step is handed, and only the second one "
                "is evidence. This catches a dataset that advertises token ids it does not "
                "hand over. It does not catch one that tokenises inside __getitem__ — that "
                "row carries ids too — which is why axes.pretokenize is what owns the move "
                "out of the timed window and what the tests attack for it."
            )
        applied.append("dataloader.pretokenize")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=collate,
    )
    # The backend axis is only applied once a loader exists; without a dataset
    # nothing was decided, and the capture probe reports it undetermined. An axis
    # left at `false` is not named: false is the absence of an application rather
    # than one, and `applied.capture` reads what the loader actually got either way.
    #
    # A `pretokenize=false` request over rows that arrive tokenised anyway is not
    # refused here. Nothing was applied, so there is nothing for this function to
    # say — and the capture probe reads `True` off the dataset and blocks the run,
    # which is the check being in the half that inspects rather than the half that
    # asks.
    return loader, applied


def declared_columns(dataset: Any) -> list[str] | None:
    """The column names a dataset declares, or None if it declares none.

    None and `[]` are different answers and the difference is the whole point:
    a dataset that says nothing leaves `dataloader.pretokenize` undetermined,
    while one that lists its columns has answered. Defined here rather than inside
    the capture probe because `_dataloader` has to ask the same question before it
    will claim the axis, and two readings of "what columns does this have" would
    drift the way the two column lists in D1 did.
    """
    columns = getattr(dataset, "column_names", None)
    if columns is None:
        features = getattr(dataset, "features", None)
        columns = list(features) if features is not None else None
    return None if columns is None else sorted(columns)


def tokenized_row(dataset: Any) -> bool:
    """Whether the dataset's first row really hands over token ids.

    One row, because that is the cheapest question that touches the data instead of
    its description — and the difference between the two is the whole of what this
    answers. A dataset with no first row answers False: an empty dataset makes every
    downstream check pass by having nothing to examine. So does one that cannot be
    indexed at all (an iterable dataset) — nothing here can read what it will yield,
    and an unreadable dataset that blocks the run is the safe half of the two ways
    to be wrong about it.
    """
    try:
        row = dataset[0]
    except (IndexError, KeyError, TypeError, StopIteration):
        return False
    keys = row.keys() if hasattr(row, "keys") else ()
    return any(column in keys for column in TOKENIZED_COLUMNS)


def tokenized_columns(dataset: Any) -> list[str] | None:
    """Which of `TOKENIZED_COLUMNS` a dataset declares; None if it declares none."""
    columns = declared_columns(dataset)
    return None if columns is None else sorted(set(columns) & set(TOKENIZED_COLUMNS))


def image_columns(dataset: Any) -> list[str] | None:
    """Which of a dataset's columns put an image in a row; None if it does not say.

    Two readings, and the cheaper one first:

    * a declared `datasets.Image` feature is the Hub's own statement that the
      column holds images, and reading a row to confirm it would decode the very
      image the declaration names;
    * otherwise the `IMAGE_COLUMNS` names are checked against the rows themselves.
      The name alone is not the answer: a subset can carry the column and no image
      in it — four of the twenty MMEB configs have no `qry_image` and thirteen no
      `pos_image`, and `scripts/bench.py::Collate` skips a `None` there — so
      refusing on the name would refuse a text-only draw stored in the subset's
      schema, which is a different and false claim. The rows are read only when
      nothing declares its types, which is the in-memory
      `scripts/bench.py::PairDataset` case; nothing is decoded that the dataset had
      not already materialised.

    `None` and `[]` are different answers, and the difference is the point: a
    dataset that declares no columns, or whose rows are not mappings, has not said
    there are no images. `_gradcache_needs_splittable_data` refuses on `None` alone
    — silence is what it cannot work with, while a row that does carry an image is
    splittable from the per-row counts the collate records.
    """
    columns = declared_columns(dataset)
    if columns is None:
        return None
    features = getattr(dataset, "features", None)
    if features:
        return sorted(
            column for column in columns if type(features.get(column)).__name__ == "Image"
        )
    named = [column for column in columns if column in IMAGE_COLUMNS]
    if not named:
        return []
    try:
        rows = [dataset[index] for index in range(len(dataset))]
    except TypeError:
        return None
    if not all(hasattr(row, "get") for row in rows):
        return None
    return sorted(column for column in named if any(row.get(column) is not None for row in rows))


class PretokenizedDataset(torch.utils.data.Dataset):
    """Rows tokenised before the timed window opened.

    Declares `column_names` because that is the only place the difference shows:
    `pretokenize` does not change what the loader is, it changes what the loader is
    handed, so both `_dataloader` and `applied._capture_dataloader_pretokenize`
    read the answer off the dataset.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError(
                "pretokenizing produced no rows; an empty dataset would make every "
                "downstream check pass by having nothing to examine"
            )
        self.rows = rows
        self.column_names = sorted({key for row in rows for key in row})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def pretokenize(dataset: Any, encode: Any) -> PretokenizedDataset:
    """Run `encode` over every row now, so the measured step does not.

    This is the whole of `dataloader.pretokenize=true`: the tokenisation does not
    change, it moves out of the timed window. `encode` is passed in because
    tokenising needs the model's processor and this module never sees one — what
    this module owns is that the move happened and that the result says so.

    An `encode` that returns rows without token ids is refused rather than
    accepted: the work would still be inside the step while the run reported the
    axis as applied, which is the substitution `UnappliedAxis` exists for.
    """
    rows = [encode(dataset[index]) for index in range(len(dataset))]
    tokenized = PretokenizedDataset(rows)
    if not tokenized_columns(tokenized):
        raise UnappliedAxis(
            f"encode produced columns {tokenized.column_names}, none of them "
            f"{list(TOKENIZED_COLUMNS)}; the rows are not tokenised, so the tokenisation is "
            "still inside the timed step and pretokenize would be a label on unchanged work."
        )
    return tokenized


class PackedCollate:
    """A batch as one concatenated sequence instead of a padded rectangle.

    Padding is what packing removes: `batch_size` rows of differing length become
    a single `(1, total)` row, and where each sequence ends travels with the batch
    as `cu_seqlens` — the cumulative-length vector varlen attention kernels take,
    and the same vector `embedding.packed_last_token_pool` pools on. The speedup is
    theirs, not this collate's: without a varlen kernel this produces one long
    sequence and the attention is quadratic over it. That half is unmeasured here —
    there is no GPU in this checkout.

    `position_ids` restarts at 0 per sequence. Without it a packed batch is one
    long sequence to any positional encoding, and the rows would attend across
    their own boundaries.

    A class rather than a closure for the reasons `scripts/bench.py::Collate`
    gives: `applied._capture_dataloader_packing` reads `axis_packing` off the
    collate, and `configs/data/*.yaml` set `num_workers: 8`, which a local closure
    cannot be pickled into.

    `tokenize` is optional because this module has no processor. With it, it takes
    the raw rows of one batch and returns one 1-D tensor of token ids per sequence,
    in the order the loss expects to find them; without it the rows must already
    carry `input_ids`, which is what `dataloader.pretokenize=true` produces.

    Both paths are checked for padding rather than asked about it, because a padded
    batch packed here is the one failure this class cannot survive: PAD concatenated
    into the pack is counted by tokens/s as work the model did, and
    `packed_last_token_pool` reads some sequences' embedding off a PAD position —
    while the run still certifies `dataloader.packing=True`, since that answer comes
    off `axis_packing` below. A padding batch tokenizer (`pad_sequence`, or a
    tokenizer left at its default `padding=True`) is the natural thing to hand
    `tokenize`, so `pad_id` is required with it and every sequence is searched for
    that id; rows that arrive with an `attention_mask` have already recorded where
    their padding is, and that record is read.

    A checkpoint whose pad id *is* its eos id cannot be checked this way: real
    sequences would end in it and this refuses them. That refusal is the honest
    outcome — pretokenize instead, where each row is tokenised alone and no padding
    is written at all.
    """

    # Read back by applied._capture_dataloader_packing. Declared rather than
    # inferred: an unrecognised collate is undetermined there, which is what keeps
    # a hand-rolled one from being read as either answer.
    axis_packing = True

    def __init__(self, tokenize: Any = None, pad_id: int | None = None) -> None:
        if tokenize is not None and pad_id is None:
            raise ValueError(
                "PackedCollate(tokenize=...) needs pad_id as well: this path calls a "
                "tokenizer on the raw batch, every batch tokenizer pads by default, and "
                "PAD packed as a real token inflates tokens/s and becomes some sequence's "
                "pooled embedding while the run still reports packing as applied. Tokenise "
                "with padding=False and pass the id the tokenizer would have padded with."
            )
        self.tokenize = tokenize
        self.pad_id = pad_id

    def _sequences(self, rows: list[Any]) -> list[torch.Tensor]:
        if self.tokenize is not None:
            produced = self.tokenize(rows)
            if isinstance(produced, torch.Tensor):
                raise ValueError(
                    f"tokenize returned a single {tuple(produced.shape)} tensor. A rectangle "
                    "is a padded batch; iterating its rows would pack every PAD in it as a "
                    "real token. Return one 1-D tensor of ids per sequence instead."
                )
            sequences = [torch.as_tensor(sequence) for sequence in produced]
        else:
            missing = [i for i, row in enumerate(rows) if "input_ids" not in row]
            if missing:
                raise RuntimeError(
                    f"rows {missing[:8]} carry no 'input_ids' and this collate was built "
                    "without a tokenize callable, so there is nothing to pack. Either set "
                    "dataloader.pretokenize=true or hand PackedCollate a tokenizer."
                )
            self._refuse_masked_padding(rows)
            sequences = [torch.as_tensor(row["input_ids"]) for row in rows]
        flat = []
        for index, sequence in enumerate(sequences):
            if sequence.dim() != 1:
                raise ValueError(
                    f"sequence {index} has shape {tuple(sequence.shape)}; a pack is built out "
                    "of 1-D sequences and flattening a rectangle here would pack its padding "
                    "as content. Give one 1-D tensor per sequence (squeeze a (1, n) row)."
                )
            flat.append(sequence)
        self._refuse_pad_id(flat)
        empty = [i for i, sequence in enumerate(flat) if sequence.numel() == 0]
        if empty:
            raise ValueError(
                f"sequences {empty[:8]} are empty; a zero-length sequence has no last token "
                "to pool, and packing it would silently give it the previous sequence's."
            )
        return flat

    @staticmethod
    def _refuse_masked_padding(rows: list[Any]) -> None:
        """A row that carries an `attention_mask` has already said where its PAD is.

        Tokenising a batch of rows ahead of time with a padding tokenizer produces
        exactly that: ids padded to the longest row of the batch, and a mask that
        records it. The mask is the one place the row admits the padding, so it is
        read rather than trusted — without it the ids alone are indistinguishable
        from a genuinely uniform batch.
        """
        padded = []
        for index, row in enumerate(rows):
            mask = row.get("attention_mask") if hasattr(row, "get") else None
            if mask is None:
                continue
            mask = torch.as_tensor(mask).reshape(-1)
            ids = torch.as_tensor(row["input_ids"]).reshape(-1)
            if mask.numel() != ids.numel():
                raise ValueError(
                    f"row {index} carries {ids.numel()} ids and a {mask.numel()}-long "
                    "attention_mask; the mask is this row's only record of where its padding "
                    "is, and one that does not describe the row cannot be read as either answer."
                )
            real = int(mask.sum())
            if real != mask.numel():
                padded.append(f"{index} ({mask.numel() - real} of {mask.numel()} PAD)")
        if padded:
            raise ValueError(
                f"rows {padded[:8]} arrive padded, by their own attention_mask. Packing them "
                "would concatenate PAD as real tokens: tokens/s would count work the model "
                "never did and packed_last_token_pool would read a PAD as some sequence's "
                "embedding. Tokenise each row on its own, with padding=False."
            )

    def _refuse_pad_id(self, sequences: list[torch.Tensor]) -> None:
        """No sequence may contain the id the tokenizer pads with.

        The `attention_mask` check above covers rows that brought their own record;
        this covers the `tokenize` path, where the callable returns bare id tensors
        and a padding tokenizer is the default one. `pad_id` is None only on the
        pretokenized path, where the caller tokenised row by row and had nothing to
        pad against — stated in the class docstring, and the reason it stays
        optional there rather than becoming a second declaration to trust.
        """
        if self.pad_id is None:
            return
        contaminated = [
            f"{index} ({int((sequence == self.pad_id).sum())} of {sequence.numel()})"
            for index, sequence in enumerate(sequences)
            if bool((sequence == self.pad_id).any())
        ]
        if contaminated:
            raise ValueError(
                f"sequences {contaminated[:8]} contain pad id {self.pad_id}. Every token in a "
                "packed batch is measured as a real one and can be pooled as a sequence's "
                "embedding, so a pack holding PAD reports a throughput and an embedding that "
                "belong to neither the model nor the data. Tokenise with padding=False."
            )

    def __call__(self, rows: list[Any]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("packing an empty batch; there is nothing to measure in it")
        sequences = self._sequences(rows)
        lengths = torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.int32)
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
        return {
            "input_ids": torch.cat(sequences).unsqueeze(0),
            "position_ids": torch.cat(
                [torch.arange(sequence.numel()) for sequence in sequences]
            ).unsqueeze(0),
            "cu_seqlens": cu_seqlens,
            "seq_lengths": lengths,
        }


def _gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Every rank's rows, concatenated in rank order, with this rank's slice still
    attached to its own graph.

    `dist.all_gather` writes into pre-allocated buffers, and those buffers carry
    no autograd history. A loss built on its raw output therefore produces *no*
    gradient for the local embeddings at all: the step runs, the timer records it,
    and the model has learned nothing. Assigning the local tensor back over its own
    slice before the concatenation is what restores that path, and it is the whole
    reason this is not two lines of `dist.all_gather`.

    The other ranks' rows stay constants here, so this rank ends up holding
    d(world loss)/d(its own embeddings) and nothing else. Summing those partial
    terms across ranks is a gradient all-reduce — which is DDP's job, and
    `parallel.strategy=ddp` is still refused by `assemble`. Until it is not, a
    multi-rank run of this axis measures the loss's cost correctly and trains
    incorrectly; that is stated in the closure below and in the report, rather
    than hidden by a scale factor chosen to make one line look right.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "parallel.cross_device_negatives=true needs an initialised process group; "
            "this process has none, so there are no other ranks to draw negatives from. "
            "The axis is applied — the loss gathers — and this is where a run without a "
            "world dies, before it has produced a number."
        )
    world = dist.get_world_size()
    if world < 2:
        raise RuntimeError(
            "parallel.cross_device_negatives=true with world_size=1 gathers nothing: the "
            "loss is exactly plain in-batch MNRL, and it would be reported under the "
            "cross-device label. Run it on more than one rank or turn the axis off."
        )
    gathered = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor.contiguous())
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def _in_batch_scoring(temperature: float, gather: bool) -> tuple[Any, bool]:
    """The `(queries, documents) -> loss` half of both losses, and a literal saying
    whether the closure that was built gathers.

    The bool is returned by the branch that built the closure rather than copied
    from the config, because `applied._capture_cross_device_negatives` reads it off
    the loss as the evidence of what the loss does. A value copied from the request
    would make that probe a mirror of the request.

    Both sides are gathered, not the documents alone. `info_nce` labels row i with
    column i, and `all_gather` returns buffers in rank order, so rank r's local row
    i lands at global index `r * local_rows + i` on *both* sides: the positive is
    still the diagonal and `info_nce` is reused exactly as it is. Gathering only
    the documents would leave a (local, world) matrix needing an explicit
    `rank * local_rows` label offset — a second, hand-written spelling of which
    column is the positive, which is the kind of arithmetic that is wrong silently.
    `tests/test_axes.py` pins the ordering invariant that replaces it.

    The cost is real and is part of what the axis measures: every rank scores the
    full world-by-world matrix, so the similarity compute is duplicated world-fold
    while the negatives per row grow by the same factor.
    """
    if gather:

        def mnrl_across_ranks(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
            return info_nce(_gather_with_grad(queries), _gather_with_grad(documents), temperature)

        return mnrl_across_ranks, True

    def mnrl(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        return info_nce(queries, documents, temperature)

    return mnrl, False


def _rng_state(device: torch.device) -> dict[str, Any]:
    """Enough of the RNG state to replay one forward pass.

    GradCache runs every piece of the batch twice — once under `no_grad` to build
    the cache and once with a graph to consume it — and the two passes have to be
    the same function of the same weights. Under `model.train()` they are not:
    dropout draws a fresh mask each call (`peft.dropout` is an ablation setting in
    this study, and attention dropout is live in train mode), so the recomputed
    representations would not be the ones the cached gradient was computed for.
    The result is a wrong gradient that still looks like a gradient — the exact
    shape of failure `docs/review-findings.md` asks the equivalence test to rule
    out.

    The `cuda` branch has no execution evidence: this suite runs on CPU and every
    measured run is CUDA, so what a real device's generator does across the two
    passes is 측정 안 함 and needs a GPU pod. `tests/test_axes.py` pins the branch's
    shape — which device is read and which is written back — against stand-ins for
    `torch.cuda`'s two calls, and that is all it pins.
    """
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    """Put the RNG back where `_rng_state` found it, so the recompute replays."""
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def _cumulative(counts: Any) -> list[int]:
    """`[0, c0, c0+c1, ...]` — where each row's share of a tensor begins."""
    bounds = [0]
    for count in counts:
        bounds.append(bounds[-1] + int(count))
    return bounds


def _image_bounds(batch: dict[str, Any], images_per_row: Any, rows: int) -> dict[str, list[int]]:
    """Where each row's images, and each row's image patches, begin.

    Returns at most two boundary vectors of length `rows + 1`, keyed by what they
    count. `images` comes straight from the per-row counts the collate recorded.
    `patches` exists only where the batch carries `IMAGE_GRID_KEY`: that tensor has
    one row per image, so slicing *it* by the image boundaries and multiplying each
    slice out gives the patch count of every batch row — which is what the Qwen
    processors' `pixel_values` is indexed by.

    Both are derived from the same per-row counts, so a batch cannot end up with an
    image boundary and a patch boundary that disagree about which row an image
    belongs to.
    """
    if images_per_row is None:
        return {}
    counts = [int(count) for count in images_per_row]
    if len(counts) != rows:
        raise RuntimeError(
            f"images_per_row has {len(counts)} entries for a {rows}-row batch. It is the "
            "map from rows to pixels, so one that does not describe this batch would "
            "attribute pixels by an offset nothing downstream could detect."
        )
    if any(count < 0 for count in counts):
        raise RuntimeError(f"images_per_row carries a negative count: {counts}")
    bounds = {"images": _cumulative(counts)}

    grid = batch.get(IMAGE_GRID_KEY)
    if torch.is_tensor(grid):
        total = bounds["images"][-1]
        if int(grid.shape[0]) != total:
            raise RuntimeError(
                f"{IMAGE_GRID_KEY} has {int(grid.shape[0])} rows but images_per_row accounts "
                f"for {total} image(s). That tensor is one row per image, so the two are "
                "answers to the same question and only one of them can be right; splitting "
                "under either would put some row's patches in another row's piece."
            )
        per_image = grid.reshape(total, -1).prod(dim=1).tolist() if total else []
        bounds["patches"] = _cumulative(per_image)
    return bounds


def _split_rows(
    batch: dict[str, Any], size: int, images_per_row: Any = None
) -> list[dict[str, Any]]:
    """The batch cut into `size`-row pieces along the batch dimension.

    Sliced rather than re-collated: the padded width stays whatever the collate
    produced, so a piece is the same tensor content as the corresponding rows of
    the whole batch and the recomputed representations are the same numbers.

    Three ways an entry can be attributed to rows, and everything else is refused:

    * one leading entry per row — `input_ids`, `attention_mask`, `mm_token_type_ids`;
    * one leading entry per image, cut at the image boundaries `images_per_row`
      implies — gemma-4's `pixel_values` and `image_position_ids`, and the Qwen
      processors' `image_grid_thw`;
    * one leading entry per image *patch*, cut where the grid says each row's
      images end — the Qwen processors' `pixel_values`.

    `images_per_row` is what makes the last two possible, and it is passed in rather
    than inferred: the collate that built the batch is the only place that still
    knows which row each image came from (`scripts/bench.py::Collate`). Without it
    an image payload is refused exactly as before, which is why a caller that
    forgets to thread it through gets a stopped run rather than a shifted one.

    An entry in `IMAGE_PAYLOAD_KEYS` is never fitted to the row rule even if its
    leading dimension happens to equal the row count. Those keys count images or
    patches by measurement, and a batch where that number coincides with the row
    count would otherwise be split by a rule that is right by accident.

    Non-tensor entries stay refused: they cannot be shown to be row-aligned, and a
    value silently copied into every piece is a value shared by rows it does not
    describe.
    """
    mask = batch.get("attention_mask")
    if not torch.is_tensor(mask):
        raise RuntimeError(
            "loss=cached_mnrl splits the batch by rows and pools with attention_mask; "
            "this batch has none, so neither the split nor the pooling is defined."
        )
    rows = int(mask.shape[0])
    if rows % 2:
        raise RuntimeError(
            f"{rows} rows is odd, but queries and documents are the two halves of this "
            "batch; the pairing that InfoNCE scores would be off by one."
        )

    bounds = _image_bounds(batch, images_per_row, rows)
    row_bounds = list(range(rows + 1))
    attribution: dict[str, list[int]] = {}
    unattributable: list[str] = []
    for key, value in batch.items():
        if not torch.is_tensor(value):
            unattributable.append(key)
            continue
        leading = int(value.shape[0])
        if key in IMAGE_PAYLOAD_KEYS:
            counted = next((name for name, vector in bounds.items() if vector[-1] == leading), None)
            if counted is None:
                unattributable.append(key)
            else:
                attribution[key] = bounds[counted]
        elif leading == rows:
            attribution[key] = row_bounds
        else:
            unattributable.append(key)

    if unattributable:
        counted = {name: vector[-1] for name, vector in bounds.items()}
        raise RuntimeError(
            f"loss=cached_mnrl cannot attribute {sorted(unattributable)} to rows: they carry "
            f"neither one leading entry per row ({rows}) nor a count this batch's "
            f"images_per_row explains ({counted or 'none supplied'}). pixel_values is the "
            "case this stops — its leading dimension counts patches (Qwen-VL) or images "
            "(gemma-4), so it is splittable only alongside the per-row image counts the "
            "collate recorded. Splitting it by position would hand one row's pixels to "
            "another row, and nothing downstream can tell a wrong embedding from a right one."
        )

    pieces = []
    for start in range(0, rows, size):
        stop = min(start + size, rows)
        piece = {}
        for key, value in batch.items():
            lower, upper = attribution[key][start], attribution[key][stop]
            # A piece none of whose rows carried an image gets no image payload at
            # all, rather than an empty one. That is the shape the processor
            # returns for a text-only batch — `scripts/bench.py::Collate` passes
            # `images=` only when there are some — so it is the shape the model is
            # known to accept. What a real checkpoint does with a zero-length
            # `pixel_values` is 측정 안 함: there is no GPU here and no model large
            # enough to ask.
            if key in IMAGE_PAYLOAD_KEYS and lower == upper:
                continue
            piece[key] = value[lower:upper]
        pieces.append(piece)
    return pieces


def _gradcache_needs_splittable_data(dataset: Any) -> None:
    """Refuse `loss=cached_mnrl` unless the data this run reads can be split by rows.

    `_split_rows` is the per-batch half of this and it is fail-closed, but an axis
    whose every batch it refuses is not an applicable axis: `assemble` would name
    `loss.name` applied, `applied.capture` would agree with the request, and the run
    would die at step 1 — after the axis had been counted. `axis-values` in
    `scripts/audit_plan.py` counts exactly that report and never reaches a batch, so
    without this refusal the axis reads as applicable while no measured run can turn
    it on. That is the "the check passed and there was nothing to check" shape this
    repository keeps producing, and the honest answer is to refuse here.

    **Rows that carry images are no longer refused.** They were, and the refusal
    covered every configured run — both subsets are MMEB draws and all three models
    turn those images into `pixel_values` — which left the axis with the largest
    part of the memory it exists to save outside anything this study could measure.
    `_split_rows` now attributes an image payload to rows from the per-row image
    counts `scripts/bench.py::Collate` records, so an image batch splits.

    What is still refused is a dataset nothing can read as rows: `image_columns`
    answers `None` when the dataset declares no columns, or when its rows are not
    mappings, or when it cannot be indexed at all. That is also the reading the
    collate needs — it counts a row's images by asking the row for them — so a
    dataset that cannot answer it produces batches whose pixels have no row counts,
    and `_split_rows` refuses every one of them. Not knowing is not evidence that
    the batches split, and this function exists precisely so that a run which
    cannot be split is never first counted as one that can.
    """
    if image_columns(dataset) is None:
        raise UnappliedAxis(
            "loss=cached_mnrl splits the batch by rows, and this run's dataset "
            f"({type(dataset).__name__}) does not say whether its rows carry images: it "
            "declares no columns, or no rows this can read as mappings. Rows that carry "
            "images are fine — _split_rows attributes their pixels from the per-row image "
            "counts the collate records — but a dataset whose rows cannot be read produces "
            "no such counts, so the axis is refused here rather than reported applied and "
            "then crashed at the first batch."
        )


def _loss(config: BenchConfig) -> tuple[Any, list[str]]:
    """The loss a run computes, and where its negatives come from.

    Whether `cached_mnrl` can run against a given run's *data* is decided in
    `assemble` by `_gradcache_needs_splittable_data`, which is where the dataset
    is. This function builds the arithmetic; that one decides applicability.

    Two axes are decided here — `loss.name` and
    `parallel.cross_device_negatives` — because both are properties of the same
    closure and neither is readable anywhere else. `applied._capture_loss` and
    `applied._capture_cross_device_negatives` read the attributes this attaches; a
    loss built anywhere else declares nothing and comes back undetermined, which
    blocks a reportable run.

    **`cached_mnrl` is not callable as `(queries, documents)`.** GradCache is a
    backward strategy, not a loss function: it needs the model and the batch so it
    can encode the batch twice, and pooled embeddings are what it produces rather
    than what it consumes. The returned callable therefore raises if called that
    way, which is deliberate — `scripts/bench.py` currently computes `info_nce`
    inline and never touches `built.loss_fn`, so a harness that reached for the
    plain signature would measure ordinary in-batch negatives and label the number
    `cached_mnrl`. Refusing to be called is what turns that into a crash on the
    first step instead. The real entry point is `gradcache_backward`.
    """
    if config.loss.name not in ("mnrl", "cached_mnrl"):
        raise UnappliedAxis(f"loss={config.loss.name} has no implementation here.")

    scores, gathers = _in_batch_scoring(
        config.loss.temperature, config.parallel.cross_device_negatives
    )
    # `cross_device_negatives=false` is the absence of an application rather than
    # one, so it is not named — the convention `_apply_to_model` states. `loss.name`
    # is named either way because a loss is always built.
    applied = ["loss.name"] + (["parallel.cross_device_negatives"] if gathers else [])

    if config.loss.name == "mnrl":
        loss_fn = scores
        loss_fn.axis_value = "mnrl"
        loss_fn.axis_cross_device_negatives = gathers
        return loss_fn, applied

    # The schema guarantees both of these for cached_mnrl (`_gradcache_mini_batch_fits`):
    # non-null, and no larger than the batch it splits.
    mini_batch = config.loss.mini_batch

    def gradcache_backward(
        model: Any,
        batch: dict[str, Any],
        *,
        padding_side: str,
        scale: float = 1.0,
        images_per_row: Any = None,
    ) -> torch.Tensor:
        """One batch's gradient, accumulated `mini_batch` rows at a time.

        The point of GradCache is that no activation graph for more than
        `mini_batch` rows exists at once, while the loss is still scored over every
        row of the batch — which is what makes a contrastive batch larger than
        memory possible, and what this axis measures.

        Three passes, in order:

        1. encode every piece under `no_grad`, keeping only the representations;
        2. score all of them at once, with the representations as leaves, and read
           `d(loss)/d(representations)` off those leaves — the cache;
        3. encode every piece again *with* a graph and seed its backward with that
           piece's slice of the cache.

        Step 3 is where the RNG state saved in step 1 is replayed: without it
        dropout draws different masks in the two passes and the cached gradient
        belongs to representations that no longer exist.

        `scale` multiplies the gradient, not the returned loss — the caller
        accumulates micro-batches with `scale=1/grad_accum` and still records the
        unscaled loss, which is the convention `scripts/bench.py` already uses.

        `images_per_row` is how many images each row of `batch` put into the batch,
        in batch-row order. It is the whole of what makes a multimodal batch
        splittable: the processor flattens every row's images into one
        `pixel_values`, and the collate that fed it is the last place that still
        knows which row each one came from. A caller that omits it on a batch
        carrying pixels gets the refusal `_split_rows` has always raised.

        Returns the detached loss. Nothing here reads a device tensor into Python:
        the conversion is a synchronisation and belongs outside the timed window.
        """
        from trainbench.probe.steps import encode

        pieces = _split_rows(batch, mini_batch, images_per_row)
        device = pieces[0]["attention_mask"].device
        states = []
        representations = []
        with torch.no_grad():
            for piece in pieces:
                states.append(_rng_state(device))
                representations.append(encode(model, piece, padding_side))
        cached = torch.cat(representations).detach().requires_grad_(True)

        half = cached.shape[0] // 2
        loss = scores(cached[:half], cached[half:])
        (loss * scale).backward()
        cache = cached.grad

        start = 0
        for piece, state in zip(pieces, states, strict=True):
            _restore_rng(state, device)
            with_graph = encode(model, piece, padding_side)
            rows = with_graph.shape[0]
            torch.autograd.backward(with_graph, grad_tensors=cache[start : start + rows])
            start += rows
        return loss.detach()

    def cached_mnrl(queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "loss=cached_mnrl cannot be computed from pooled embeddings: GradCache is a "
            "backward strategy that encodes the batch twice, so it needs the model and the "
            "batch, not the representations. Call `built.loss_fn.gradcache_backward(model, "
            "batch, padding_side=..., scale=...)`. Reaching for this signature would have "
            "measured plain in-batch negatives and labelled the number cached_mnrl."
        )

    cached_mnrl.axis_value = "cached_mnrl"
    cached_mnrl.axis_cross_device_negatives = gathers
    cached_mnrl.gradcache_backward = gradcache_backward
    # The split size the harness is measuring, exposed so a result can record it
    # without re-deriving it from the config.
    cached_mnrl.mini_batch = mini_batch
    return cached_mnrl, applied
