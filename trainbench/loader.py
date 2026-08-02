"""One entry point that loads any of the six frameworks, and what each build says
about itself.

Every probe module hid its load inside a `run()` closure, so five of the six
frameworks were reachable by config and unable to produce a number. The framework
call itself lives here now; the probes wrap it in their own reporting.

`AdapterOut` is the `loader-bench` boundary in memory and
`tests/fixtures/adapter_out.sample.json` is the same shape serialised. The rules
below are checked twice on purpose: the contract file validates the sample without
importing anything, and these validate the live object. `trainbench/kernels.py`
carries the same split for the `attention` block.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, get_args

import torch

from trainbench import axes, kernels
from trainbench.applied import FRAMEWORK_OWNABLE, AppliedMismatch
from trainbench.config_schema import BenchConfig, FrameworkConfig, axis_knobs
from trainbench.embedding import align_padding_side

FRAMEWORKS = frozenset(get_args(FrameworkConfig.model_fields["name"].annotation))
# Each framework's own `load(config, device, load_kwargs)` lives here, one module
# per framework, because only these files are exempt from the root lock.
PROBE_PACKAGE = "trainbench.probe"

HARNESS = "harness"
FRAMEWORK = "framework"
STEP_OWNERS = (HARNESS, FRAMEWORK)

AUTOCAST = "autocast"
# Widening this is a contract change: `axes.step_context` has to grow a branch
# that can establish the new kind.
CONTEXT_KINDS = (AUTOCAST,)
ESTABLISHED_BY = "axes.step_context"

UNVERIFIED = "확인 안 함"

# The `attention` block of the build fingerprint, whose shape belongs to the
# `kernel-provenance` boundary. `kernel.name` is a different axis.
BUILD_FINGERPRINT_KEY = kernels.BUILD_FINGERPRINT_KEY
# The key a run record carries the whole fingerprint under. Re-exported so the
# producer here and the record writer name it in one place.
RUN_RECORD_KEY = kernels.RUN_RECORD_KEY
ROOT_MODULE = "model"
# Root plus children plus grandchildren. Deeper is per-layer and grows with the
# checkpoint; shallower cannot tell a wrapper (peft, DenseModel) from the backbone
# it wraps, which is the difference this map exists to show.
MODULE_CLASS_DEPTH = 2

ATTENTION_AXIS = "attn.name"

# peft modes that make parameters trainable *after* this module is done: the
# adapter is attached by `axes._peft`, inside `axes.assemble`.
ADAPTER_ATTACHING_PEFT_MODES = ("lora", "qlora")


class AdapterRefusal(AppliedMismatch):
    """A framework built something this harness must not put a number on.

    An `AppliedMismatch` because that is the type `scripts/bench.py`'s `refusing()`
    catches. A refusal outside its catch list leaves no result file at all, so a
    build this harness declined would be published as a pod that crashed rather
    than as a setting that was refused.
    """


# --- the boundary types -------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """Who runs the training step, and what a batch has to contain for it."""

    owner: str
    callable: str | None
    batch_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.owner not in STEP_OWNERS:
            raise AdapterRefusal(f"step.owner {self.owner!r} not in {list(STEP_OWNERS)}")
        if self.owner == FRAMEWORK and not self.callable:
            raise AdapterRefusal("a framework-owned step has to name the callable it runs")
        if self.owner == HARNESS and self.callable is not None:
            raise AdapterRefusal(
                f"step.owner is 'harness' but step.callable is {self.callable!r}; "
                "a harness-driven step names no framework callable"
            )
        if not self.batch_keys:
            raise AdapterRefusal("step.batch_keys must name at least one key")

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "callable": self.callable,
            "batch_keys": list(self.batch_keys),
        }


@dataclass(frozen=True)
class StepContext:
    """An execution context the framework trains inside, stated but not entered.

    `docs/CONTRACTS.md` §2 fixes `axes.step_context` as the only site that
    establishes a precision context, so this is a requirement and never a
    `with` block of the adapter's own.
    """

    kind: str
    device_type: str
    dtype: str
    reason: str
    established_by: str = ESTABLISHED_BY

    def __post_init__(self) -> None:
        if self.kind not in CONTEXT_KINDS:
            raise AdapterRefusal(
                f"required_step_context.kind {self.kind!r} not in {list(CONTEXT_KINDS)}; "
                "a kind axes.step_context cannot establish"
            )
        for name in ("device_type", "dtype", "reason"):
            if not str(getattr(self, name)).strip():
                raise AdapterRefusal(f"required_step_context.{name} must be a non-empty string")
        if self.established_by != ESTABLISHED_BY:
            raise AdapterRefusal(
                f"required_step_context.established_by is {self.established_by!r}, not "
                f"{ESTABLISHED_BY!r}; that is the only place a precision context is established"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "device_type": self.device_type,
            "dtype": self.dtype,
            "reason": self.reason,
            "established_by": self.established_by,
        }


@dataclass(frozen=True)
class EntryPoint:
    """The framework's own documented training entry point, and how ours differs.

    `differs` and the citation move together in both directions: an unknown is
    written as one, and a source that says it is unverified cannot carry a verdict
    beside it.
    """

    framework: str
    harness_uses: str
    differs: bool | None
    source: str

    def __post_init__(self) -> None:
        for name in ("framework", "harness_uses", "source"):
            if not str(getattr(self, name)).strip():
                raise AdapterRefusal(f"documented_entry_point.{name} must be a non-empty string")
        unverified = UNVERIFIED in self.source
        if self.differs is None:
            if not unverified:
                raise AdapterRefusal(
                    f"documented_entry_point.differs is None but source does not say "
                    f"{UNVERIFIED!r}; an unknown has to be written as one"
                )
            return
        if unverified:
            raise AdapterRefusal(
                f"documented_entry_point.differs is {self.differs!r} while source says "
                f"{UNVERIFIED!r}; an unverified entry point cannot also carry a verdict"
            )
        if self.differs != (self.framework != self.harness_uses):
            raise AdapterRefusal(
                f"documented_entry_point.differs is {self.differs} but the two entry points "
                "say otherwise"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "harness_uses": self.harness_uses,
            "differs": self.differs,
            "source": self.source,
        }


@dataclass(frozen=True)
class AdapterOut:
    """What one framework built, and everything the harness needs to measure it."""

    framework: str
    model: Any
    processor: Any
    step: Step
    owned_axes: Mapping[str, str]
    required_step_context: StepContext | None
    fingerprint: Mapping[str, Any]
    documented_entry_point: EntryPoint

    def to_payload(self) -> dict[str, Any]:
        """The `loader-bench` JSON projection. `model`/`processor` become classes."""
        context = self.required_step_context
        return {
            "framework": self.framework,
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "step": self.step.to_dict(),
            "owned_axes": dict(self.owned_axes),
            "required_step_context": None if context is None else context.to_dict(),
            "fingerprint": dict(self.fingerprint),
            "documented_entry_point": self.documented_entry_point.to_dict(),
        }


ADAPTER_OUT_FIELDS = frozenset(f.name for f in fields(AdapterOut))


# --- the build fingerprint ----------------------------------------------------


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def module_classes(model: Any, depth: int = MODULE_CLASS_DEPTH) -> dict[str, str]:
    """Class names of the root and every module within `depth` of it."""
    classes = {ROOT_MODULE: type(model).__name__}
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return classes
    for name, module in named_modules():
        if name and name.count(".") < depth:
            classes[name] = type(module).__name__
    return classes


def tensor_count(fingerprint: Mapping[str, Any]) -> int:
    """Parameters and buffers together.

    A framework that adds tensors has no obligation to add parameters, and
    unsloth's gemma-4 build is the case that made the distinction matter.
    """
    return len(fingerprint["parameter_dtypes"]) + len(fingerprint["buffer_dtypes"])


def build_fingerprint(
    model: Any,
    config: BenchConfig,
    *,
    requested_ref: str | None = None,
    revision_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """What the framework changed that we did not ask for.

    `applied.py` reads back the axes the run requested; this is the complement, so
    a difference between two frameworks shows up as a confound rather than as a
    speed number. The `attention` block comes from `kernels.read_fingerprint`,
    which validates itself before returning.
    """
    return {
        "module_classes": module_classes(model),
        "parameter_dtypes": {n: _dtype_name(p.dtype) for n, p in model.named_parameters()},
        "buffer_dtypes": {n: _dtype_name(b.dtype) for n, b in model.named_buffers()},
        "trainable_parameter_names": [n for n, p in model.named_parameters() if p.requires_grad],
        BUILD_FINGERPRINT_KEY: kernels.read_fingerprint(
            model,
            axis=ATTENTION_AXIS,
            value=config.attn.name,
            requested=config.attn.impl,
            requested_ref=requested_ref,
            revision_resolver=revision_resolver,
        ),
    }


def fingerprint_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Every way two builds differ, as the confounds they are.

    Reported rather than judged: which of them invalidates a comparison is a
    reading of the results, not something this file can decide.
    """
    diff: dict[str, Any] = {}
    if left["module_classes"] != right["module_classes"]:
        diff["module_classes"] = (left["module_classes"], right["module_classes"])
    if (counts := (tensor_count(left), tensor_count(right)))[0] != counts[1]:
        diff["tensor_count"] = counts
    for key in ("parameter_dtypes", "buffer_dtypes"):
        regimes = (sorted(set(left[key].values())), sorted(set(right[key].values())))
        if regimes[0] != regimes[1]:
            diff[key] = regimes
    trainable = (len(left["trainable_parameter_names"]), len(right["trainable_parameter_names"]))
    if trainable[0] != trainable[1]:
        diff["trainable_parameter_count"] = trainable
    resolved = tuple(
        f[BUILD_FINGERPRINT_KEY]["resolved"]["attn_implementation"] for f in (left, right)
    )
    if resolved[0] != resolved[1]:
        diff[BUILD_FINGERPRINT_KEY] = resolved
    return diff


def attaches_an_adapter_later(config: BenchConfig) -> bool:
    """Whether something after the load still has to make parameters trainable.

    unsloth freezes every parameter without a LoRA marker inside
    `from_pretrained`, and the LoRA that unfreezes it is attached by `axes._peft`
    inside `axes.assemble` — later than any load here. So a frozen build under
    `peft.mode=lora`/`qlora` is the expected intermediate state, and the
    frozen-graph refusal belongs to whoever holds the assembled model.
    """
    return config.peft.mode in ADAPTER_ATTACHING_PEFT_MODES


def refuse_a_build_the_fingerprint_condemns(
    framework: str,
    fingerprint: Mapping[str, Any],
    context: StepContext | None,
    *,
    adapter_attaches_later: bool = False,
) -> None:
    """The two states the fingerprint exists to stop, at the site that reads it.

    Both already happened. unsloth returned a model with every parameter frozen
    and the backward pass survived it; axolotl leaves two modules in fp32 and
    upstream absorbs the difference in an autocast this harness has to be told
    about.

    Public and re-callable so the deferred half has no second definition: the
    caller that holds the model after `axes.assemble` runs this again with
    `adapter_attaches_later=False`, and that call is what actually catches a LoRA
    run whose adapter never attached.
    """
    dtypes = fingerprint["parameter_dtypes"]
    trainable = fingerprint["trainable_parameter_names"]
    if not trainable and not adapter_attaches_later:
        raise AdapterRefusal(
            f"{framework} built a model with no trainable parameter among "
            f"{len(dtypes)}; a step over a frozen graph still produces a number, and "
            "that number is the speed of a model that learns nothing"
        )
    regimes = sorted({dtypes[name] for name in trainable})
    if len(regimes) > 1 and context is None:
        raise AdapterRefusal(
            f"{framework} trains parameters in more than one dtype ({regimes}) but declares "
            "no required_step_context, so the harness would measure it in a regime the "
            "framework does not use"
        )


# --- the adapters -------------------------------------------------------------


@dataclass(frozen=True)
class Adapter:
    """One framework's load call and everything it declares about the result."""

    name: str
    step: Step
    documented_entry_point: EntryPoint
    owned_axes: Mapping[str, str] = field(default_factory=dict)
    required_step_context: StepContext | None = None
    # Whether `from_pretrained` is reachable on this path. Where the framework owns
    # that call the axis is left unapplied rather than guessed at a keyword, and
    # the capture side reads back the mismatch.
    honours_load_kwargs: bool = False
    # sentence-transformers pools inside its own module, so forcing the configured
    # side onto its tokeniser would change an input it is meant to observe
    # untouched (trainbench/probe/sentence_transformers.py).
    aligns_padding_side: bool = True

    def __post_init__(self) -> None:
        if self.name not in FRAMEWORKS:
            raise AdapterRefusal(f"{self.name!r} is not one of {sorted(FRAMEWORKS)}")
        unknown = sorted(set(self.owned_axes) - set(axis_knobs()))
        if unknown:
            raise AdapterRefusal(f"{self.name}.owned_axes names non-axes {unknown}")
        outside = sorted(set(self.owned_axes) - set(FRAMEWORK_OWNABLE))
        if outside:
            raise AdapterRefusal(
                f"{self.name}.owned_axes claims {outside}, which applied.FRAMEWORK_OWNABLE "
                "does not let an adapter disclaim; capture would file them as applied"
            )
        for axis, reason in self.owned_axes.items():
            if not str(reason).strip():
                raise AdapterRefusal(f"{self.name}.owned_axes[{axis!r}] must give a reason")
        if self.step.owner == FRAMEWORK and not self.owned_axes:
            raise AdapterRefusal(
                f"{self.name}.step.owner is 'framework' but owned_axes is empty; a framework "
                "that runs its own step subsumes at least one axis, and declaring none files "
                "it as applied by this harness"
            )
        if self.step.owner == HARNESS and self.owned_axes:
            raise AdapterRefusal(
                f"{self.name}.step.owner is 'harness' but owned_axes claims "
                f"{sorted(self.owned_axes)}; the harness cannot hand an axis to a step it "
                "runs itself"
            )

    @property
    def module(self) -> str:
        return f"{PROBE_PACKAGE}.{self.name}"

    def load(
        self, config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]
    ) -> tuple[Any, Any]:
        """The framework call, which lives in the per-image adapter module.

        Not here: `scripts/audit_plan.py`'s `doc-commands` demands every
        third-party import under `trainbench/` of the root lock and exempts
        exactly those six files, because each framework installs only inside its
        own image. Importing them here would ask the root lock for all six at
        once, which is the resolution `envs/` exists because nobody can satisfy.
        """
        return importlib.import_module(self.module).load(config, device, load_kwargs)


HARNESS_STEP = Step(owner=HARNESS, callable=None, batch_keys=("input_ids", "attention_mask"))


ADAPTERS: dict[str, Adapter] = {
    adapter.name: adapter
    for adapter in (
        Adapter(
            name="native",
            step=HARNESS_STEP,
            honours_load_kwargs=True,
            documented_entry_point=EntryPoint(
                framework="transformers AutoModel.from_pretrained + a hand-written loop",
                harness_uses="transformers AutoModel.from_pretrained + a hand-written loop",
                differs=False,
                source="trainbench/probe/native.py — the reference path, which is this "
                "harness itself; transformers documents no training entry point of its own "
                "for an embedding model with no LM head",
            ),
        ),
        Adapter(
            name="unsloth",
            step=HARNESS_STEP,
            documented_entry_point=EntryPoint(
                framework="FastVisionModel.from_pretrained -> for_training(model) -> TRL "
                "SFTTrainer (unsloth_train delegates to trainer.train)",
                harness_uses="FastVisionModel.from_pretrained(full_finetuning=...) + the "
                "harness loop; no for_training(), no SFTTrainer",
                differs=True,
                source="unsloth 2026.7.6 models/vision.py:2279-2316 (for_training), "
                "trainer.py:326-330 and :434 (unsloth_train -> trainer.train, "
                "UnslothTrainer(SFTTrainer)); .plans/research/unsloth.md §5",
            ),
        ),
        Adapter(
            name="ms_swift",
            step=HARNESS_STEP,
            documented_entry_point=EntryPoint(
                framework="swift.cli.main:cli_main -> TrainerFactory 'embedding' -> "
                "swift.trainers.EmbeddingTrainer with loss_map['infonce']",
                harness_uses="swift.get_model_processor + the harness loop over "
                "trainbench.embedding.info_nce; no swift trainer",
                differs=True,
                source="ms-swift swift/trainers/trainer_factory.py:13-19, "
                "swift/loss/mapping.py:6-16, swift/trainers/mixin.py:1046-1054, "
                "entry_points.txt; .plans/research/ms-swift.md §5",
            ),
        ),
        Adapter(
            name="sentence_transformers",
            step=HARNESS_STEP,
            honours_load_kwargs=True,
            aligns_padding_side=False,
            documented_entry_point=EntryPoint(
                framework="SentenceTransformerTrainer(BaseTrainer).compute_loss, which hands "
                "the features to a loss object that calls the model forward itself",
                harness_uses="SentenceTransformer(...) + the harness loop over "
                "trainbench.embedding.info_nce; no trainer and no ST loss class",
                differs=True,
                source="sentence-transformers 5.6.1 base/trainer.py:76, :459-509 "
                "(compute_loss; no training_step override in the wheel), "
                "sentence_transformer/trainer.py:36; .plans/research/sentence-transformers.md §4",
            ),
        ),
        Adapter(
            name="tevatron",
            step=Step(
                owner=FRAMEWORK,
                callable="tevatron.retriever.modeling.DenseModel.forward",
                batch_keys=("query", "passage"),
            ),
            owned_axes={
                "loss.name": "EncoderModel.forward scores, divides by its own temperature "
                "and computes cross-entropy itself, so the harness loss never runs in this "
                "cell (pinned dd06310 retriever/modeling/encoder.py:69-77)",
                "parallel.cross_device_negatives": "the same forward performs the gather "
                "when is_ddp, so whether negatives cross devices is tevatron's decision "
                "(encoder.py:65-67, :104-115)",
            },
            documented_entry_point=EntryPoint(
                framework="EncoderModel.forward(query=, passage=) — encode, pool, normalise, "
                "score, InfoNCE and the distributed gather in one call",
                harness_uses="DenseModel.load(...) + that same forward driven by the harness "
                "timer, with loss and cross-device negatives recorded as framework-owned",
                differs=True,
                source="pinned dd06310 src/tevatron/retriever/modeling/encoder.py:52-87 and "
                "dense.py:18-46; trainbench/probe/steps.py:253 expects "
                "model(**batch) -> last_hidden_state, a signature this forward has no "
                "argument for; .plans/research/tevatron.md §3",
            ),
        ),
        Adapter(
            name="axolotl",
            step=HARNESS_STEP,
            required_step_context=StepContext(
                kind=AUTOCAST,
                device_type="cuda",
                dtype="bfloat16",
                reason="axolotl leaves embed_tokens and lm_head in fp32 and trains inside HF "
                "Trainer's autocast, which accelerate's default launcher always turns on. "
                "The restoring branch needs adapter/FSDP/cut_cross_entropy and this path has "
                "none of them, so measured without the context native (pure bf16) and "
                "axolotl are two numeric regimes reported under one label.",
            ),
            documented_entry_point=EntryPoint(
                framework="axolotl.cli.main:main -> cli/train.py -> axolotl.train:train -> "
                "setup_trainer(...) -> an HF Trainer subclass, launched under accelerate",
                harness_uses="ModelLoader(cfg, tokenizer).load() + the harness loop inside "
                "axes.step_context; no trainer and no Accelerator",
                differs=True,
                source="axolotl 0.18.0 entry_points.txt, cli/main.py:99-106 (launcher "
                "defaults to accelerate), train.py:69-84 and :584-596; "
                ".plans/research/axolotl.md §5-6",
            ),
        ),
    )
}

_missing = sorted(FRAMEWORKS - set(ADAPTERS))
if _missing:
    # A framework added to the schema without an adapter is the state bench.py was
    # in for five of them: reachable by config, unable to produce a number.
    raise AdapterRefusal(f"the schema names {_missing} and no adapter loads them")


# --- the entry point ----------------------------------------------------------


def describe(
    adapter: Adapter,
    model: Any,
    processor: Any,
    config: BenchConfig,
    **fingerprint_kwargs: Any,
) -> AdapterOut:
    """Fingerprint a built model and bind it to what its adapter declares."""
    fingerprint = build_fingerprint(model, config, **fingerprint_kwargs)
    refuse_a_build_the_fingerprint_condemns(
        adapter.name,
        fingerprint,
        adapter.required_step_context,
        adapter_attaches_later=attaches_an_adapter_later(config),
    )
    return AdapterOut(
        framework=adapter.name,
        model=model,
        processor=processor,
        step=adapter.step,
        owned_axes=dict(adapter.owned_axes),
        required_step_context=adapter.required_step_context,
        fingerprint=fingerprint,
        documented_entry_point=adapter.documented_entry_point,
    )


def load(config: BenchConfig, device: torch.device, **fingerprint_kwargs: Any) -> AdapterOut:
    """Build the model `config.framework.name` asks for.

    `scripts/bench.py` calls this with those two arguments and reads the eight
    fields off the result, so the framework name that reaches `axes.assemble` is
    the adapter's own literal rather than the config's request.
    """
    adapter = ADAPTERS[config.framework.name]
    load_kwargs = axes.load_kwargs(config) if adapter.honours_load_kwargs else {}
    model, processor = adapter.load(config, device, load_kwargs)
    if adapter.aligns_padding_side:
        align_padding_side(processor, config.model.padding_side)
    return describe(adapter, model, processor, config, **fingerprint_kwargs)
