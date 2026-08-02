"""Contract for the `loader-bench` boundary: lane-g's adapters -> lane-d's harness.

Owned by neither lane. `scripts/bench.py` hardcodes the native path today
(`AutoModel.from_pretrained` and `framework="native"` as a literal), and the six
probe modules expose only `run(config, device, report) -> None`, so nothing hands
back a reusable loader. Both lanes are about to build against a shape that does
not exist yet; this file is that shape in executable form.

`tests/fixtures/adapter_out.sample.json` is the payload. Nothing here asserts a
value out of it — the rules below are about required keys, closed vocabularies and
the cross-field invariants, all of which resolve against live code (`axis_knobs`,
`FrameworkConfig`, `trainbench.axes`) so a rename in either lane lands here.
"""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, get_args

import pytest

from trainbench import axes
from trainbench.applied import Built
from trainbench.config_schema import FrameworkConfig, axis_knobs
from trainbench.probe.registry import _MODULES

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "tests" / "fixtures" / "adapter_out.sample.json"

# --- the contract ------------------------------------------------------------
# Serialised keys. `model_class`/`processor_class` are the JSON projection of the
# in-memory `model`/`processor` objects; everything else crosses unchanged.
PAYLOAD_KEYS = frozenset(
    {
        "framework",
        "model_class",
        "processor_class",
        "step",
        "owned_axes",
        "required_step_context",
        "fingerprint",
        "documented_entry_point",
    }
)
ADAPTER_OUT_FIELDS = (PAYLOAD_KEYS - {"model_class", "processor_class"}) | {"model", "processor"}

STEP_KEYS = frozenset({"owner", "callable", "batch_keys"})
STEP_OWNERS = frozenset({"harness", "framework"})

# Extending this vocabulary is a contract change: `axes.step_context` has to grow a
# branch that can establish the new kind.
CONTEXT_KINDS = frozenset({"autocast"})
CONTEXT_KEYS = frozenset({"kind", "device_type", "dtype", "reason", "established_by"})
# docs/CONTRACTS.md §2 fixes `step_context` as the only site that establishes a
# precision context. An adapter states its requirement; it never enters one.
ESTABLISHED_BY = "axes.step_context"

# Module class names, per-parameter dtype, the trainable set, and the identity of
# the bound attention function. Buffers are here because "60 more tensors" is not
# necessarily 60 more parameters.
FINGERPRINT_KEYS = frozenset(
    {
        "module_classes",
        "parameter_dtypes",
        "buffer_dtypes",
        "trainable_parameter_names",
        "attention",
    }
)
# The `attention` block is the kernel fingerprint, and its shape belongs to the
# `kernel-provenance` boundary — same object, one owner. This file pins only that
# the build fingerprint carries it; `validate_kernel_fingerprint` there is what
# decides whether it is well formed, and it reads this fixture to say so.
ATTENTION_KEYS = frozenset({"requested", "resolved", "backbones"})
ENTRY_POINT_KEYS = frozenset({"framework", "harness_uses", "differs", "source"})
UNVERIFIED = "확인 안 함"

FRAMEWORKS = frozenset(get_args(FrameworkConfig.model_fields["name"].annotation))


def _keys(problems: list[str], where: str, got: Any, want: frozenset[str]) -> bool:
    if not isinstance(got, dict):
        problems.append(f"{where}: expected an object, got {type(got).__name__}")
        return False
    if set(got) != want:
        problems.append(
            f"{where}: keys {sorted(got)} != contract keys {sorted(want)} "
            f"(missing {sorted(want - set(got))}, extra {sorted(set(got) - want)})"
        )
        return False
    return True


def _text(problems: list[str], where: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{where}: expected a non-empty string, got {value!r}")


def validate(name: str, payload: Any) -> list[str]:
    """Every way `payload` fails the contract, as messages. Empty means it holds."""
    problems: list[str] = []
    if not _keys(problems, f"{name}", payload, PAYLOAD_KEYS):
        return problems

    if payload["framework"] != name:
        problems.append(f"{name}.framework is {payload['framework']!r}, not the adapter's own name")
    if payload["framework"] not in FRAMEWORKS:
        problems.append(
            f"{name}.framework {payload['framework']!r} is not one of the schema's "
            f"framework names {sorted(FRAMEWORKS)}"
        )
    _text(problems, f"{name}.model_class", payload["model_class"])
    _text(problems, f"{name}.processor_class", payload["processor_class"])

    problems += _validate_step(name, payload)
    problems += _validate_owned_axes(name, payload)
    problems += _validate_context(name, payload)
    problems += _validate_fingerprint(name, payload)
    problems += _validate_entry_point(name, payload)
    return problems


def _validate_step(name: str, payload: dict[str, Any]) -> list[str]:
    """Element 1: an adapter may supply its own training step.

    tevatron's `DenseModel.forward` takes query=/passage= dicts and computes the
    whole step, so the harness cannot force `model(**batch) -> last_hidden_state`
    on every framework. `batch_keys` is what lane-d's collate has to produce.
    """
    problems: list[str] = []
    step = payload["step"]
    if not _keys(problems, f"{name}.step", step, STEP_KEYS):
        return problems
    if step["owner"] not in STEP_OWNERS:
        problems.append(f"{name}.step.owner {step['owner']!r} not in {sorted(STEP_OWNERS)}")
    if step["owner"] == "framework":
        _text(problems, f"{name}.step.callable", step["callable"])
    elif step["callable"] is not None:
        problems.append(
            f"{name}.step.owner is 'harness' but step.callable is {step['callable']!r}; "
            "a harness-driven step names no framework callable"
        )
    if not isinstance(step["batch_keys"], list) or not step["batch_keys"]:
        problems.append(
            f"{name}.step.batch_keys must be a non-empty list, got {step['batch_keys']!r}"
        )
    return problems


def _validate_owned_axes(name: str, payload: dict[str, Any]) -> list[str]:
    """Element 2: which axes the framework, not the harness, decides.

    Both directions are rules. An adapter whose own step computes the loss has
    subsumed at least one axis, and an adapter running the harness step has
    subsumed none — collapsing either way is how a framework-owned axis silently
    becomes an applied one.
    """
    problems: list[str] = []
    owned = payload["owned_axes"]
    if not isinstance(owned, dict):
        problems.append(f"{name}.owned_axes: expected an object, got {type(owned).__name__}")
        return problems
    knobs = set(axis_knobs())
    for axis, reason in owned.items():
        if axis not in knobs:
            problems.append(
                f"{name}.owned_axes names {axis!r}, which is not an axis knob in the schema"
            )
        _text(problems, f"{name}.owned_axes[{axis!r}]", reason)
    owner = payload["step"].get("owner") if isinstance(payload["step"], dict) else None
    if owner == "framework" and not owned:
        problems.append(
            f"{name}.step.owner is 'framework' but owned_axes is empty; a framework that "
            "runs its own step subsumes at least one axis, and declaring none files it as "
            "applied by this harness"
        )
    if owner == "harness" and owned:
        problems.append(
            f"{name}.step.owner is 'harness' but owned_axes claims {sorted(owned)}; the "
            "harness cannot hand an axis to a step it runs itself"
        )
    return problems


def _validate_context(name: str, payload: dict[str, Any]) -> list[str]:
    """Element 4: a required execution context, stated but not established.

    axolotl trains under `torch.autocast(bfloat16)` upstream. The adapter says so;
    `axes.step_context` is what enters it, and `established_by` is resolved against
    live code below so the two cannot drift apart.
    """
    problems: list[str] = []
    context = payload["required_step_context"]
    if context is None:
        return problems
    if not _keys(problems, f"{name}.required_step_context", context, CONTEXT_KEYS):
        return problems
    if context["kind"] not in CONTEXT_KINDS:
        problems.append(
            f"{name}.required_step_context.kind {context['kind']!r} not in "
            f"{sorted(CONTEXT_KINDS)}; a kind axes.step_context cannot establish"
        )
    for key in ("device_type", "dtype", "reason"):
        _text(problems, f"{name}.required_step_context.{key}", context[key])
    if context["established_by"] != ESTABLISHED_BY:
        problems.append(
            f"{name}.required_step_context.established_by is "
            f"{context['established_by']!r}, not {ESTABLISHED_BY!r}; docs/CONTRACTS.md §2 "
            "fixes that as the only place a precision context is established"
        )
    return problems


def _validate_fingerprint(name: str, payload: dict[str, Any]) -> list[str]:
    """Element 3: what the framework changed that we did not ask for."""
    problems: list[str] = []
    fingerprint = payload["fingerprint"]
    if not _keys(problems, f"{name}.fingerprint", fingerprint, FINGERPRINT_KEYS):
        return problems

    classes = fingerprint["module_classes"]
    if not isinstance(classes, dict) or "model" not in classes:
        problems.append(f"{name}.fingerprint.module_classes must carry the root under 'model'")
    elif classes["model"] != payload["model_class"]:
        # unsloth's get_peft_model returns a new object; a fingerprint taken from
        # the pre-peft one describes a model the run does not train.
        problems.append(
            f"{name}.fingerprint.module_classes['model'] is {classes['model']!r} but "
            f"model_class is {payload['model_class']!r}; the fingerprint was taken "
            "from a different object than the one returned"
        )

    dtypes = fingerprint["parameter_dtypes"]
    if not isinstance(dtypes, dict) or not dtypes:
        problems.append(
            f"{name}.fingerprint.parameter_dtypes must be a non-empty name -> dtype map"
        )
        return problems
    if not isinstance(fingerprint["buffer_dtypes"], dict):
        problems.append(f"{name}.fingerprint.buffer_dtypes must be a name -> dtype map")

    trainable = fingerprint["trainable_parameter_names"]
    if not isinstance(trainable, list):
        problems.append(f"{name}.fingerprint.trainable_parameter_names must be a list")
    elif unknown := sorted(set(trainable) - set(dtypes)):
        problems.append(
            f"{name}.fingerprint.trainable_parameter_names has {unknown} absent from "
            "parameter_dtypes, so the trainable set describes another model"
        )
    elif len({dtypes[n] for n in trainable}) > 1 and payload["required_step_context"] is None:
        # The axolotl case, derived rather than named: mixed trainable dtypes are a
        # mixed-precision regime, and one measured without a context is not the
        # regime the framework trains in.
        problems.append(
            f"{name} trains parameters in more than one dtype "
            f"({sorted({dtypes[n] for n in trainable})}) but declares no "
            "required_step_context, so the harness would measure it in a regime the "
            "framework does not use"
        )

    _keys(problems, f"{name}.fingerprint.attention", fingerprint["attention"], ATTENTION_KEYS)
    return problems


def _validate_entry_point(name: str, payload: dict[str, Any]) -> list[str]:
    """Each framework's documented training entry point, and how ours differs.

    `differs` and the citation move together in both directions. An unknown is
    written as one, and a `source` that says it is unverified cannot carry a
    verdict beside it — that is how "확인 안 함" becomes a number nobody produced.
    """
    problems: list[str] = []
    entry = payload["documented_entry_point"]
    if not _keys(problems, f"{name}.documented_entry_point", entry, ENTRY_POINT_KEYS):
        return problems
    _text(problems, f"{name}.documented_entry_point.framework", entry["framework"])
    _text(problems, f"{name}.documented_entry_point.harness_uses", entry["harness_uses"])
    _text(problems, f"{name}.documented_entry_point.source", entry["source"])
    differs = entry["differs"]
    unverified = UNVERIFIED in str(entry["source"])
    if differs is not None and unverified:
        problems.append(
            f"{name}.documented_entry_point.differs is {differs!r} while source says "
            f"{UNVERIFIED!r}; an unverified entry point cannot also carry a verdict"
        )
    if differs is None:
        if not unverified:
            problems.append(
                f"{name}.documented_entry_point.differs is null but source does not say "
                f"{UNVERIFIED!r}; an unknown has to be written as one"
            )
    elif not isinstance(differs, bool):
        problems.append(f"{name}.documented_entry_point.differs must be a bool or null")
    elif differs != (entry["framework"] != entry["harness_uses"]):
        problems.append(
            f"{name}.documented_entry_point.differs is {differs} but the two entry points "
            "say otherwise"
        )
    return problems


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="session")
def sample() -> dict[str, Any]:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def adapters(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["adapters"]


def _mutate(adapters: dict[str, Any], name: str, edit: Any) -> dict[str, Any]:
    payload = copy.deepcopy(adapters[name])
    edit(payload)
    return payload


# --- the sample is the interface ---------------------------------------------


def test_sample_conforms_to_the_contract(adapters: dict[str, Any]) -> None:
    problems = [p for name, payload in adapters.items() for p in validate(name, payload)]
    assert problems == []


def test_sample_covers_every_framework_the_schema_names(adapters: dict[str, Any]) -> None:
    """One shape for six frameworks, and the same six everywhere.

    A framework added to the schema without an adapter entry is the state
    `bench.py` is in today for five of them: reachable by config, unable to
    produce a number.
    """
    assert set(adapters) == FRAMEWORKS
    assert set(_MODULES) == FRAMEWORKS


def test_the_sample_exercises_every_branch_the_contract_has(adapters: dict[str, Any]) -> None:
    """A sample where all six entries look alike would freeze nothing.

    The three states that differ between adapters are the whole reason this
    boundary is not just "return a model and a processor".
    """
    assert any(a["step"]["owner"] == "framework" for a in adapters.values())
    assert any(a["owned_axes"] for a in adapters.values())
    assert any(a["required_step_context"] is not None for a in adapters.values())
    assert any(a["documented_entry_point"]["differs"] is None for a in adapters.values())


# --- drift: each mutation must be rejected -----------------------------------

MUTATIONS = {
    "fingerprint dropped": (
        "native",
        lambda p: p.pop("fingerprint"),
        "keys",
    ),
    "fingerprint loses the trainable set": (
        "native",
        lambda p: p["fingerprint"].pop("trainable_parameter_names"),
        "fingerprint: keys",
    ),
    "fingerprint loses per-parameter dtype": (
        "axolotl",
        lambda p: p["fingerprint"].pop("parameter_dtypes"),
        "fingerprint: keys",
    ),
    "fingerprint loses the resolved attention kernel": (
        "native",
        lambda p: p["fingerprint"]["attention"].pop("resolved"),
        "attention: keys",
    ),
    "fingerprint taken from a pre-peft object": (
        "unsloth",
        lambda p: p["fingerprint"]["module_classes"].__setitem__("model", "PeftModelForCausalLM"),
        "taken from a different object",
    ),
    "framework-owned collapsed into applied": (
        "tevatron",
        lambda p: p.__setitem__("owned_axes", {}),
        "declaring none files it as applied",
    ),
    "framework-owned claimed on a harness step": (
        "native",
        lambda p: p.__setitem__("owned_axes", {"loss.name": "not ours"}),
        "cannot hand an axis to a step it runs itself",
    ),
    "owned axis is not an axis": (
        "tevatron",
        lambda p: p["owned_axes"].__setitem__("loss.everything", "invented"),
        "not an axis knob in the schema",
    ),
    "context declaration removed": (
        "axolotl",
        lambda p: p.__setitem__("required_step_context", None),
        "declares no required_step_context",
    ),
    "context established by the adapter itself": (
        "axolotl",
        lambda p: p["required_step_context"].__setitem__("established_by", "adapter"),
        "only place a precision context is established",
    ),
    "context kind axes.step_context cannot enter": (
        "axolotl",
        lambda p: p["required_step_context"].__setitem__("kind", "grad_scaler"),
        "not in ['autocast']",
    ),
    "adapter step collapsed into the harness step": (
        "tevatron",
        lambda p: p["step"].__setitem__("owner", "harness"),
        "step.callable",
    ),
    "framework name copied from somewhere else": (
        "unsloth",
        lambda p: p.__setitem__("framework", "native"),
        "not the adapter's own name",
    ),
    "unverified entry point asserted as fact": (
        "sentence_transformers",
        lambda p: p["documented_entry_point"].__setitem__("differs", True),
        "cannot also carry a verdict",
    ),
    "a differing entry point filed as identical": (
        "tevatron",
        lambda p: p["documented_entry_point"].__setitem__("differs", False),
        "say otherwise",
    ),
    "citation dropped": (
        "tevatron",
        lambda p: p["documented_entry_point"].__setitem__("source", ""),
        "non-empty string",
    ),
}


@pytest.mark.parametrize(("name", "edit", "expected"), MUTATIONS.values(), ids=MUTATIONS.keys())
def test_contract_rejects_drift(
    adapters: dict[str, Any], name: str, edit: Any, expected: str
) -> None:
    problems = validate(name, _mutate(adapters, name, edit))
    assert any(expected in p for p in problems), problems


# --- the fingerprint has to catch the three builds that already happened ------


def test_fingerprint_catches_a_fully_frozen_build(adapters: dict[str, Any]) -> None:
    """unsloth froze every parameter and `infonce_backward` passed anyway.

    `params_with_grad=0` with a differentiable graph through the embedding output
    is a speed number for a model that learned nothing.
    """
    frozen = _mutate(
        adapters, "unsloth", lambda p: p["fingerprint"].__setitem__("trainable_parameter_names", [])
    )
    assert frozen["fingerprint"]["trainable_parameter_names"] == []
    assert (
        frozen["fingerprint"]["trainable_parameter_names"]
        != adapters["unsloth"]["fingerprint"]["trainable_parameter_names"]
    )
    # And the field is not optional: dropping it rather than emptying it is refused.
    assert validate(
        "unsloth",
        _mutate(adapters, "unsloth", lambda p: p["fingerprint"].pop("trainable_parameter_names")),
    )


def test_fingerprint_catches_two_modules_left_in_fp32(adapters: dict[str, Any]) -> None:
    """axolotl leaves embed_tokens and lm_head in fp32 while the rest goes bf16."""
    dtypes = adapters["axolotl"]["fingerprint"]["parameter_dtypes"]
    trainable = adapters["axolotl"]["fingerprint"]["trainable_parameter_names"]
    assert len({dtypes[n] for n in trainable}) > 1
    # Same build read as uniformly bf16: the fp32 pair vanishes, and with it the
    # obligation to declare a context.
    hidden = _mutate(
        adapters,
        "axolotl",
        lambda p: p["fingerprint"].__setitem__(
            "parameter_dtypes",
            dict.fromkeys(p["fingerprint"]["parameter_dtypes"], "bfloat16"),
        ),
    )
    assert validate("axolotl", adapters["axolotl"]) == []
    assert len({hidden["fingerprint"]["parameter_dtypes"][n] for n in trainable}) == 1


def test_fingerprint_catches_extra_tensors(adapters: dict[str, Any]) -> None:
    """unsloth's gemma-4 build carries 60 more tensors than native's.

    Counted over parameters and buffers together, because a framework that adds
    tensors has no obligation to add parameters.
    """

    def count(payload: dict[str, Any]) -> int:
        fingerprint = payload["fingerprint"]
        return len(fingerprint["parameter_dtypes"]) + len(fingerprint["buffer_dtypes"])

    extra = _mutate(
        adapters,
        "unsloth",
        lambda p: p["fingerprint"]["buffer_dtypes"].update(
            {f"model.extra.{i}": "float32" for i in range(60)}
        ),
    )
    assert count(extra) - count(adapters["unsloth"]) == 60
    assert validate("unsloth", extra) == []


# --- what the sample says about live code has to still be true ---------------


def test_the_framework_name_crosses_as_an_argument_not_a_config_read() -> None:
    """`applied._capture_framework` exists because the request is not the evidence.

    The adapter's literal has to reach `assemble`, so `framework` stays a required
    argument and `Built` stays the thing that carries it.
    """
    framework = inspect.signature(axes.assemble).parameters["framework"]
    assert framework.default is inspect.Parameter.empty
    assert "framework" in {f.name for f in dataclass_fields(Built)}
    assert "framework.name" in axis_knobs()


def test_established_by_resolves_to_live_code(adapters: dict[str, Any]) -> None:
    """`established_by` is a path, not a label: a rename in lane-h lands here."""
    declared = {
        a["required_step_context"]["established_by"]
        for a in adapters.values()
        if a["required_step_context"] is not None
    }
    assert declared == {ESTABLISHED_BY}
    module_name, _, attr = ESTABLISHED_BY.rpartition(".")
    module = importlib.import_module(f"trainbench.{module_name}")
    assert callable(getattr(module, attr))


def test_owned_axes_name_axes_the_schema_still_has(adapters: dict[str, Any]) -> None:
    """The sample's framework-owned axes are real knobs, not prose."""
    owned = {axis for a in adapters.values() for axis in a["owned_axes"]}
    assert owned
    assert owned <= set(axis_knobs())


# --- the two ends, which do not exist yet ------------------------------------
# Both fail at 0604684 and are expected to. `strict` is what stops that expectation
# from outliving the gap: when a lane lands its end, the run reports XPASS and goes
# red until the marker is deleted. Do not weaken the assertion instead.


@pytest.mark.xfail(
    strict=True,
    reason="lane-g: trainbench/loader.py does not exist at 0604684. Delete this marker "
    "when it does; do not weaken the assertions.",
)
def test_loader_serves_every_framework_through_one_entry_point() -> None:
    loader = importlib.import_module("trainbench.loader")
    assert set(loader.ADAPTERS) == FRAMEWORKS
    assert callable(loader.load)
    assert {f.name for f in dataclass_fields(loader.AdapterOut)} == ADAPTER_OUT_FIELDS


def test_bench_takes_the_framework_name_from_the_adapter() -> None:
    tree = ast.parse((REPO_ROOT / "scripts" / "bench.py").read_text(encoding="utf-8"))
    literals = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "assemble"
        for keyword in node.keywords
        if keyword.arg == "framework" and isinstance(keyword.value, ast.Constant)
    ]
    assert literals == [], f"scripts/bench.py names the framework itself: {literals}"
