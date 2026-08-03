"""Mechanical consistency check between the plan, the docs, and the code.

Run at the end of every wave. Exits non-zero on any failed invariant.

This exists because the human-readable checklist already failed once: the fixed
subset was pushed with 54.6% duplicate positives and 31.4% missing query images,
and the success criteria in use at the time (row count, config coverage) reported
a clean pass. Checks here must be things a wrong answer cannot satisfy — which is
also why several of them were rewritten after the first review: a check that
matched bare identifiers anywhere in the tree, or a filename in any directory,
reported a pass for exactly the state it was written to catch.

    python scripts/audit_plan.py            # all checks; this is the wave gate
    python scripts/audit_plan.py --only config-consumed   # not a gate, see --help

`doc-commands` executes the documented entry point commands, so a run leaves an
`outputs/` directory behind.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
# Where config values are allowed to be consumed from.
CODE_ROOTS = (REPO / "trainbench", REPO / "scripts")
# Reporting a value is not applying it. trainbench/applied.py reads every axis in
# order to say what was requested, so counting it as a consumer would certify an
# axis as wired on the strength of the code that only labels it.
#
# This file is excluded because it lives under `scripts/` and so scans itself. It
# once passed `attn.name` on the strength of its own prose about how it certifies
# axes; the read detection is AST-based now and prose can no longer satisfy it,
# but a checker is still not a consumer of the values it checks.
NOT_A_CONSUMER = (REPO / "trainbench" / "applied.py", Path(__file__).resolve())

# The measurement entry point. Named here because a check that only asks for
# "some caller" is satisfied by callers that cannot block anything.
BENCH_ENTRY_POINT = REPO / "scripts" / "bench.py"


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    # How many problems this check found, when it counts anything. Recorded in the
    # baseline so that progress and regression *inside* an already-failing check
    # are visible: without it, reverting an entire wave's work left the summary
    # line byte-identical, because the check was failing before and after.
    # None where a count is meaningless — `assert-called` either finds the entry
    # point calling the axis machinery or it does not.
    count: int | None = None


CHECKS: dict[str, Callable[[], Result]] = {}


def check(name: str) -> Callable[[Callable[[], Result]], Callable[[], Result]]:
    def register(fn: Callable[[], Result]) -> Callable[[], Result]:
        CHECKS[name] = fn
        return fn

    return register


def _code_files(exclude: tuple[Path, ...] = ()) -> list[Path]:
    return [
        path
        for root in CODE_ROOTS
        for path in sorted(root.rglob("*.py"))
        if path not in exclude and path.name != "config_schema.py"
    ]


# What a config object is called where it is read. The value is only reachable
# through it, so the access has to start there: an unanchored match counted
# `anything.data.subset_rows` inside an unrelated function as a read of
# `config.data.subset_rows`, and `getattr(model, "name")` as a read of
# `model.name`. `data`, `run`, `train` and `model` are common name fragments.
#
# The anchor is the segment immediately before the group, so the config may be
# held on something: `self.config.model.add_generation_prompt` in `bench.py` is a
# read, and requiring a bare name would have reported that knob unread while the
# harness passes it to the tokenizer.
#
# The cost of anchoring is that an alias is invisible: `prepare_data.py` binds
# `data = config.data` and then reads `data.subset_rows`, which this reports as
# unread. That is a known false alarm, tracked in the baseline, and the safe
# direction — the other one certifies knobs nothing reads.
CONFIG_OBJECT_NAMES = frozenset({"config", "cfg"})


class _DropUnreachable(ast.NodeTransformer):
    """Branches a literal condition can never enter.

    `if False:\n    _ = config.data.subset_rows` is a read the interpreter never
    performs, and it is one of three ways a reviewer defeated the previous
    text-stripping version of this check.

    Only literal conditions fold. `if 1 == 2:` is not evaluated here, and nothing
    in this module proves that a read reaches anything — `config-consumed` asks
    whether the knob is read at all, and `axis-values` is the check that asks
    whether reading it does something.
    """

    def visit_If(self, node: ast.If) -> Any:
        self.generic_visit(node)
        try:
            taken = bool(ast.literal_eval(node.test))
        except (ValueError, TypeError, SyntaxError):
            return node
        return node.body if taken else node.orelse


def _string(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _config_reads(source: str) -> set[str]:
    """Every `group.key` this source reads off the config object.

    Parsed rather than searched. A regex over raw source is satisfied by a file
    that merely *talks about* the knob, and stripping the prose first only moved
    the hole: a docstring assigned to a name (`_NOTE = "config.data.subset_rows"`)
    is an `ast.Assign`, not an `ast.Expr`, so it survived the strip and counted as
    a read. `assert-called` has been AST-parsed since it was written, for exactly
    this reason; this check now is too, and a string can no longer be a read
    however it is spelled.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    tree = _DropUnreachable().visit(tree)
    reads: set[str] = set()
    for node in ast.walk(tree):
        # config.group.key
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            if _is_config_object(node.value.value):
                reads.add(f"{node.value.attr}.{node.attr}")
        # config["group"]["key"]
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript):
            group, key = _string(node.value.slice), _string(node.slice)
            if group and key and _is_config_object(node.value.value):
                reads.add(f"{group}.{key}")
        # getattr(config.group, "key")
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr":
            target = node.args[0] if node.args else None
            key = _string(node.args[1]) if len(node.args) > 1 else None
            if key and isinstance(target, ast.Attribute) and _is_config_object(target.value):
                reads.add(f"{target.attr}.{key}")
    return reads


def _is_config_object(node: ast.expr) -> bool:
    """`config`, `cfg`, or either of them held on something (`self.config`).

    What is anchored is the name directly in front of the group, which is what
    tells `config.data.subset_rows` apart from `anything.data.subset_rows`. How
    the config itself was reached does not change that.
    """
    if isinstance(node, ast.Name):
        return node.id in CONFIG_OBJECT_NAMES
    return isinstance(node, ast.Attribute) and node.attr in CONFIG_OBJECT_NAMES


def _code_reads(exclude: tuple[Path, ...] = ()) -> set[str]:
    return set().union(*(_config_reads(p.read_text()) for p in _code_files(exclude)), set())


# Directories under configs/ that are not Hydra config groups. Each entry is a
# claim that nothing composes the directory, so its keys are not run settings and
# holding code to "reads every one of them" would be meaningless.
NOT_A_CONFIG_GROUP = {
    "experiment": (
        "orchestrator manifests read by scripts/orchestrate.py; absent from "
        "configs/config.yaml defaults, so Hydra never composes them into a run"
    ),
}


def _composed_groups() -> set[str]:
    """Config groups the root config actually composes.

    A Hydra config group is defined by being in `defaults`; a directory that is
    not there never reaches a run. Deriving the set rather than listing exceptions
    is what keeps `configs/experiment/` out without a special case.
    """
    root_config = CONFIGS / "config.yaml"
    if not root_config.exists():
        # Empty rather than an exception, and safe in both directions: the callers
        # then find no leaves and no composed groups, which their own empty-input
        # guards turn into failures instead of a silent pass.
        return set()
    root = yaml.safe_load(root_config.read_text()) or {}
    return {key for entry in root.get("defaults", []) if isinstance(entry, dict) for key in entry}


def _config_leaf_keys() -> dict[str, set[str]]:
    """Leaf keys per composed config group, e.g. {"attn": {"name"}}."""
    composed = _composed_groups()
    groups: dict[str, set[str]] = {}
    for path in CONFIGS.rglob("*.yaml"):
        if path.parent == CONFIGS or path.parent.name not in composed:
            continue
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            groups.setdefault(path.parent.name, set()).update(k for k in data if isinstance(k, str))
    return groups


def _reads_dotted(code: str, group: str, key: str) -> bool:
    """Whether the code reads this specific config value.

    Matching the bare identifier finds `name` in eighteen unrelated files, so the
    access has to look like one through the config object: `config.attn.name`,
    `cfg["attn"]["name"]`, or `getattr(config.attn, "name")`.
    """
    return f"{group}.{key}" in _config_reads(code)


def _nothing_to_check(items, what: str) -> str | None:
    """Why an empty input set is a failure rather than a pass.

    `data-pinned` reported "every data config pins a commit sha" when
    `configs/data/` was absent, because zero configs all pin one. The item was in
    the baseline, so passing marked it FIXED and blocked the gate — which is the
    only reason anyone noticed. Every check that iterates a set has this shape,
    so each one says what it expected to find.
    """
    if items:
        return None
    return f"found no {what}; a check with nothing to examine passes for the wrong reason"


@check("config-groups")
def every_config_directory_is_composed_or_declared() -> Result:
    """Every directory under configs/ is either composed or declared not to be.

    `config-consumed` only looks at composed groups, which is what keeps
    orchestrator manifests out of it. That narrowing would otherwise fail open: a
    real axis group added to configs/ but forgotten in `defaults` would vanish
    from every check at once while also never reaching a run.
    """
    composed = _composed_groups()
    directories = {p.name for p in CONFIGS.iterdir() if p.is_dir()}
    if empty := _nothing_to_check(directories, "directories under configs/"):
        return Result("config-groups", False, empty)
    problems = [
        f"{name} is neither in configs/config.yaml defaults nor declared as not a group"
        for name in sorted(directories - composed - set(NOT_A_CONFIG_GROUP))
    ]
    problems += [
        f"{name} is declared as not a config group but the root config composes it"
        for name in sorted(composed & set(NOT_A_CONFIG_GROUP))
    ]
    # A declaration for a directory that does not exist yet excludes nothing, so
    # it is reported rather than failed: the risk this check guards is an
    # unclassified directory, and a lane may land its declaration before its files.
    pending = sorted(set(NOT_A_CONFIG_GROUP) - directories)
    note = (
        f", {len(pending)} declared but not yet present ({', '.join(pending)})" if pending else ""
    )
    return Result(
        "config-groups",
        not problems,
        f"{len(composed)} composed group(s), "
        f"{len(NOT_A_CONFIG_GROUP) - len(pending)} declared non-group(s){note}"
        if not problems
        else "; ".join(problems),
    )


# Leaves the schema turns into something else before any code sees them. The
# scan skips config_schema.py, so the derivation is invisible to it and the
# source leaf would read as unconsumed. Kept explicit and small: each entry is a
# claim that reading the derived name is reading the original.
DERIVED_LEAVES = {
    # AttnConfig.impl maps the axis value to transformers' name for it. Deriving
    # rather than configuring both is what stops `name: fa3, impl: sdpa`.
    "attn.name": "attn.impl",
}


@check("config-consumed")
def config_fields_are_read_by_code() -> Result:
    """Every config knob must be read somewhere, through the config object.

    An unread knob is worse than a missing one: `attn=fa4` composes, validates,
    prints, and is recorded in the result JSON while changing nothing. Eight of
    twelve ablation axes were in exactly that state.
    """
    leaves = _config_leaf_keys()
    if empty := _nothing_to_check(leaves, "config groups under configs/"):
        return Result("config-consumed", False, empty)
    reads = _code_reads(exclude=NOT_A_CONSUMER)

    def consumed(group: str, key: str) -> bool:
        leaf = f"{group}.{key}"
        return leaf in reads or DERIVED_LEAVES.get(leaf, "") in reads

    orphans = [
        f"{group}.{key}"
        for group, keys in sorted(leaves.items())
        for key in sorted(keys)
        if not consumed(group, key)
    ]
    return Result(
        "config-consumed",
        not orphans,
        "all config leaves are read by code"
        if not orphans
        else f"{len(orphans)} unread knob(s): {', '.join(orphans)}",
        count=len(orphans),
    )


@check("axis-wired")
def axes_are_applied_and_verified() -> Result:
    """Every axis must have both a place that applies it and a probe that reads
    it back off the model.

    Separate from config-consumed because the two failures look identical from
    outside and are not: an axis nothing applies measures the default, and an axis
    nothing verifies measures whatever the fallback was. Only the pair is evidence.
    """
    sys.path.insert(0, str(REPO))
    from trainbench import axes
    from trainbench.applied import _CAPTURES
    from trainbench.config_schema import axis_knobs

    declared = set(axis_knobs())
    captured = set(_CAPTURES)
    if empty := _nothing_to_check(declared, "axes declared by the schema"):
        return Result("axis-wired", False, empty)
    problems = []
    unknown = sorted(captured - declared)
    lopsided = sorted(captured ^ axes.IMPLEMENTED)
    unwired = sorted(declared - captured)
    if unknown:
        problems.append(f"capture probe for undeclared axis: {', '.join(unknown)}")
    if lopsided:
        problems.append(f"applied and verified sets disagree on: {', '.join(lopsided)}")
    if unwired:
        problems.append(f"{len(unwired)} axis/axes with no capture probe: {', '.join(unwired)}")
    return Result(
        "axis-wired",
        not problems,
        f"all {len(declared)} axes are applied and verified"
        if not problems
        else "; ".join(problems),
        # The unwired axes, not the problem strings: a set disagreement and a
        # missing probe are one problem line each however many axes they name.
        count=len(unwired) + len(unknown) + len(lopsided),
    )


def _calls(path: Path, function: str) -> bool:
    """Whether the file really calls `function`, rather than mentioning it.

    Parsed rather than searched: a docstring or a comment satisfies a substring
    test, and this check exists precisely to catch a guarantee that is talked
    about but not invoked.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == function
        for node in ast.walk(tree)
    )


# Config groups whose values select an optimisation. Every leaf key in these is
# either an axis or listed below as deliberately not one.
AXIS_GROUPS = frozenset(
    {
        "attn",
        "kernel",
        "precision",
        "compile",
        "optim",
        "loss",
        "peft",
        "freeze",
        "dataloader",
        "parallel",
        "framework",
    }
)
# Leaves inside an axis group that parameterise the choice rather than being one.
# Changing `optim.lr` changes the run, but it is not a thing that can silently
# fall back to something else, which is what the applied check is for.
NOT_AN_AXIS = frozenset(
    {
        "optim.lr",
        "optim.weight_decay",
        "loss.temperature",
        "loss.mini_batch",
        "peft.r",
        "peft.alpha",
        "peft.dropout",
    }
)


@check("axis-fields")
def axis_group_leaves_are_classified() -> Result:
    """Every leaf in an axis group must be marked `Axis()` or listed as not one.

    The marker is opt-in, and nothing otherwise obliges a new field to carry it.
    A `precision.recipe` added without one would be read by axes.py (so
    `config-consumed` passes), absent from `axis_knobs()` (so `axis-wired` never
    sees it), and outside `assert_matches` — the same shape of hole as the eight
    unwired axes, one field narrower.
    """
    sys.path.insert(0, str(REPO))
    from trainbench.config_schema import axis_knobs

    marked = set(axis_knobs())
    leaves = {g: k for g, k in _config_leaf_keys().items() if g in AXIS_GROUPS}
    if empty := _nothing_to_check(leaves, f"config files in any of {sorted(AXIS_GROUPS)}"):
        return Result("axis-fields", False, empty)
    problems = []
    for group, keys in sorted(leaves.items()):
        for key in sorted(keys):
            leaf = f"{group}.{key}"
            if leaf not in marked and leaf not in NOT_AN_AXIS:
                problems.append(leaf)
    stray = sorted(n for n in NOT_AN_AXIS if n in marked)
    if stray:
        problems.append(f"marked as an axis but listed as not one: {', '.join(stray)}")
    return Result(
        "axis-fields",
        not problems,
        f"every leaf in {len(leaves)} axis group(s) present is classified"
        if not problems
        else f"unclassified: {', '.join(problems)}",
    )


# What the measurement entry point has to invoke: every call site in axes.py plus
# the verification. Each is silently skippable, and skipping one is invisible in
# the result — a harness that never calls `step_context` runs an fp8 recipe that
# never wrapped the forward pass while the capture probe still finds the swapped
# modules and reports a match. The list is kept equal to the hooks documented in
# docs/CONTRACTS.md; a hook missing from it is a hook nothing requires.
ENTRY_POINT_CALLS = ("patch", "load_kwargs", "assemble", "step_context", "assert_matches")

# The name the training loop binds its loss to. Every binding of it has to come
# from the callable `assemble` built, and that is a question about this file
# rather than about who calls whom: the hooks above are satisfied by an entry
# point that calls `assemble`, keeps `built.model` and `built.optimizer`, and then
# computes the loss some other way.
LOSS_BINDING = "loss"
BUILT_LOSS = "loss_fn"

# The one legitimate way a `loss` binding in the entry point does not read
# `built.loss_fn`: a framework whose own forward computes it. Two facts have to
# agree before that is exempt below — the entry point's registry names the
# framework, and the adapter declares it owns the axis. The registry is an edit;
# the declaration is what is true about the framework, and it is what
# `applied._owned` acts on when it exempts the axis from the capture.
STEP_RUNNER_REGISTRY = "FRAMEWORK_OWNED_STEP_RUNNERS"
OWNED_LOSS_AXIS = "loss.name"
ADAPTER_DECLARATION = "Adapter"
OWNED_AXES_FIELD = "owned_axes"
# Read as source next to the entry point, not imported: this check is an AST walk
# over a file it is handed, and the declaration has to be pointable somewhere else
# the same way `BENCH_ENTRY_POINT` is.
ADAPTER_SOURCE = REPO / "trainbench" / "loader.py"


def _frameworks_declaring_loss_ownership(path: Path) -> set[str]:
    """Every adapter in `path` whose `owned_axes` claims it computes the loss itself.

    Read off the `Adapter(...)` declarations rather than a list here, for
    `applied.py`'s reason: a hand-written list fails open, and an adapter missing
    from it would read as exempt instead of as undeclared.
    """
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()
    owners: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != ADAPTER_DECLARATION:
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        name, owned = keywords.get("name"), keywords.get(OWNED_AXES_FIELD)
        if not isinstance(name, ast.Constant) or not isinstance(owned, ast.Dict):
            continue
        declared = {key.value for key in owned.keys if isinstance(key, ast.Constant)}
        if OWNED_LOSS_AXIS in declared:
            owners.add(name.value)
    return owners


def _registered_step_runner_frameworks(tree: ast.AST) -> set[str]:
    """The framework names `STEP_RUNNER_REGISTRY` maps to a runner, off its literal."""
    registered: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign) or not isinstance(node.value, ast.Dict):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == STEP_RUNNER_REGISTRY for t in targets):
            continue
        registered |= {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    return registered


def _loss_bindings(path: Path) -> tuple[list[int], bool]:
    """`(lines binding `loss` from something other than `.loss_fn`, was it used)`.

    Two passes, because the object is consumed under two names. `built.loss_fn(...)`
    reads directly; the GradCache path binds `getattr(built.loss_fn, ...)` to a name
    outside the timed window and calls that, so a name whose value was derived from
    `.loss_fn` counts as the same object. Everything else binding `loss` is a loss
    the run computed for itself under a label `capture` certified off `built.loss_fn`.

    One exemption, and it is derived rather than declared here: a loss returned by
    something the entry point read out of `STEP_RUNNER_REGISTRY` is a framework's
    own, and only if every framework in that registry has declared
    `OWNED_LOSS_AXIS` in `ADAPTER_SOURCE`. A framework added to the registry that
    declares nothing leaves the binding flagged, which is the point — the check
    would otherwise wave through exactly the uncertified loss it exists to catch.
    Kept out of `derived`: an exempt binding suppresses `stray`, and it is not
    evidence that `built.loss_fn` was consumed, so `used` still has to be earned.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return [], False

    def reads_built_loss(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Attribute) and child.attr == BUILT_LOSS
            for child in ast.walk(node)
        )

    derived = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and node.value is not None and reads_built_loss(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def from_built_loss(value: ast.expr) -> bool:
        return reads_built_loss(value) or any(
            isinstance(child, ast.Name) and child.id in derived for child in ast.walk(value)
        )

    registered = _registered_step_runner_frameworks(tree)
    exempt: set[str] = set()
    if registered and registered <= _frameworks_declaring_loss_ownership(ADAPTER_SOURCE):
        exempt = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign | ast.AnnAssign)
            and node.value is not None
            and any(
                isinstance(child, ast.Name) and child.id == STEP_RUNNER_REGISTRY
                for child in ast.walk(node.value)
            )
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }

    def from_a_framework_owned_runner(value: ast.expr) -> bool:
        return any(isinstance(child, ast.Name) and child.id in exempt for child in ast.walk(value))

    stray: list[int] = []
    used = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == LOSS_BINDING for t in targets):
            continue
        if from_built_loss(node.value):
            used = True
        elif not from_a_framework_owned_runner(node.value):
            stray.append(node.lineno)
    return sorted(stray), used


@check("assert-called")
def the_measurement_entry_point_calls_the_axis_machinery() -> Result:
    """The harness that reports numbers must go through axes.py and applied.py,
    and the loss it reports must be the one `assemble` built.

    "Some caller exists somewhere" is not the property that matters. The probe
    calls `assert_matches` too, and `purpose=probe` returns immediately from every
    check — so an entry point written without the call would leave this green
    while measuring unverified settings. The named entry point is what has to call
    it, and the same is true of every hook an axis is applied through.

    Calling the hooks is not enough on its own. A loop that calls `assemble` and
    then computes `info_nce` inline satisfies every name above while `loss.name`
    is certified off a `built.loss_fn` nothing consumed — the live shape of that
    defect, since `axes._loss` builds `cached_mnrl` as a different callable and the
    result would carry its label over ordinary in-batch negatives. So the loss
    bindings in the file are read too, not just its call list.
    """
    if not BENCH_ENTRY_POINT.exists():
        return Result(
            "assert-called",
            False,
            f"{BENCH_ENTRY_POINT.relative_to(REPO)} does not exist yet, so nothing enforces "
            "the axis machinery for a reportable run",
        )
    entry = BENCH_ENTRY_POINT.relative_to(REPO).as_posix()
    missing = [name for name in ENTRY_POINT_CALLS if not _calls(BENCH_ENTRY_POINT, name)]
    stray, used = _loss_bindings(BENCH_ENTRY_POINT)
    problems = []
    if missing:
        problems.append(
            f"never calls {', '.join(missing)}; those axes would be applied or verified "
            "nowhere while the run still reports numbers"
        )
    if not used:
        problems.append(
            f"binds no {LOSS_BINDING} from built.{BUILT_LOSS}; the callable that "
            "certifies loss.name would be built, read by capture, and never used"
        )
    if stray:
        problems.append(
            f"computes {LOSS_BINDING} outside built.{BUILT_LOSS} at line(s) "
            f"{', '.join(str(line) for line in stray)}; the run would report a loss the "
            "loss axis did not produce"
        )
    return Result(
        "assert-called",
        not problems,
        f"{entry} calls {', '.join(ENTRY_POINT_CALLS)} and takes every {LOSS_BINDING} "
        f"from built.{BUILT_LOSS}"
        if not problems
        else f"{entry} " + "; ".join(problems),
    )


TREE_PREFIX = " │├└─"
TREE_INDENT = 4
# What the layout tree is held to describe. It names Python modules, documents,
# and the manifests that define the project — not every artifact on disk.
DOCUMENTABLE = re.compile(r"[A-Za-z0-9_.\-]+\.(?:py|md|toml|yaml|json|lock)")
# Package markers. `trainbench/metrics/__init__.py` is the module's whole body,
# but what a reader needs documented is `trainbench/metrics/`, and that is
# reported as an undocumented directory. Listing both would be noise.
NOT_DOCUMENTABLE = frozenset({"__init__.py"})


def _layout_entries(block: str) -> list[tuple[str, bool]]:
    """`(declared path, is a directory)` for every entry in the layout tree.

    The tree is read as a tree: an entry's path is its name under the directories
    that contain it. Comparing basenames instead — as this did first — matches
    `config.py` against 73 files here, most of them inside `.venv`, so the check
    stays green after the declared file is deleted and red for files that exist
    under a directory it never looked at.
    """
    parents: dict[int, str] = {}
    entries = []
    for line in block.splitlines():
        entry = line.lstrip(TREE_PREFIX)
        if not entry.strip():
            continue
        depth = (len(line) - len(entry)) // TREE_INDENT
        name = entry.split("#")[0].split()[0]
        # Everything below this depth belonged to whatever came before.
        parents = {d: p for d, p in parents.items() if d < depth}
        if depth == 0:
            # The root line names the repository itself, contributing no segment.
            parents[0] = name.rstrip("/")
            continue
        path = "/".join([parents[d] for d in sorted(parents) if d > 0] + [name.rstrip("/")])
        if name.endswith("/"):
            parents[depth] = name.rstrip("/")
        entries.append((path, name.endswith("/")))
    return entries


def missing_plan_files(block: str, root: Path) -> list[str]:
    """Paths the layout tree declares that do not exist where it puts them.

    Directories are checked too. The file regex used to be the only gate on what
    was checkable, so an entry without an extension was used as a parent path and
    never verified: `configs/nonexistent/` sat in the block and passed.
    """
    missing = []
    for path, is_directory in _layout_entries(block):
        if is_directory:
            if not (root / path).is_dir():
                missing.append(path + "/")
        elif DOCUMENTABLE.fullmatch(path.rsplit("/", 1)[-1]) and not (root / path).exists():
            missing.append(path)
    return missing


def undocumented_files(block: str, tracked: list[str]) -> list[str]:
    """Repository files inside a directory the tree enumerates that it does not name.

    The other direction. `missing_plan_files` only asks whether what is written
    down exists, which is why `trainbench/metrics/`, `scripts/bench.py` and four
    test modules were all absent from the block while the check passed.

    Scope is taken from the tree rather than declared separately, so it cannot
    drift from it: a directory the tree lists children for claims to be a complete
    listing and is enumerated; one listed without children (`docker/`, `envs/`,
    `trainbench/probe/`, each config group) is documented as a unit and its
    contents are not claimed. Descent stops at the first undocumented segment, so
    an undocumented directory is reported once rather than once per file under it.
    """
    entries = _layout_entries(block)
    documented = {path for path, _ in entries}
    # "" is the repository root, whose children are the top-level entries.
    enumerated = {path.rsplit("/", 1)[0] if "/" in path else "" for path in documented}
    undocumented = set()
    for path in tracked:
        segments = path.split("/")
        for depth, segment in enumerate(segments):
            # Dot-entries (`.github/`, `.claude/`, `.pre-commit-config.yaml`) are
            # tooling rather than the source layout, and the tree does not
            # describe them.
            if segment.startswith("."):
                break
            if "/".join(segments[:depth]) not in enumerated:
                break  # inside a directory documented as a unit
            current = "/".join(segments[: depth + 1])
            if current in documented:
                continue
            if depth < len(segments) - 1:
                undocumented.add(current + "/")
            elif segment not in NOT_DOCUMENTABLE and DOCUMENTABLE.fullmatch(segment):
                undocumented.add(current)
            break
    return sorted(undocumented)


def _tracked_files() -> list[str]:
    return subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True
    ).stdout.split()


@check("plan-files")
def the_plan_layout_and_the_repository_agree() -> Result:
    """PLAN.md's repository layout must describe reality, and all of it.

    Both directions. A block that only has to be true about what it mentions is
    kept true by mentioning less, and that is what happened: four Wave 3 files
    were added and none of them reached the block while this reported a pass.
    """
    plan = (REPO / "PLAN.md").read_text()
    # The language tag is optional in the fence: requiring a bare ``` meant that
    # adding `text` after it — an ordinary markdown edit — silently disabled the
    # check, which then reported the absence as a pass.
    block = re.search(r"```[a-z]*\n(train-comparison/.*?)```", plan, re.S)
    if not block:
        return Result(
            "plan-files",
            False,
            "PLAN.md has no repository layout block; the check that the declared "
            "layout matches reality then has nothing to compare and passes for it",
        )
    tracked = _tracked_files()
    missing = sorted(set(missing_plan_files(block.group(1), REPO)))
    undocumented = undocumented_files(block.group(1), tracked)
    problems = []
    if missing:
        problems.append(f"{len(missing)} declared but absent: {', '.join(missing)}")
    if undocumented:
        problems.append(f"{len(undocumented)} present but undeclared: {', '.join(undocumented)}")
    # Reported alongside the two directions rather than instead of them. With no
    # file list the reverse direction is vacuous and has to say so, but the
    # forward one still works and its answer is still worth having.
    if empty := _nothing_to_check(tracked, "git-tracked files to compare the layout against"):
        problems.append(empty)
    # The scope is stated in the passing line too: a reader who sees this green
    # needs to know that `docker/`, `envs/` and the config groups are documented
    # as units, so nothing inside them was examined.
    scope = (
        f"scope: {len(tracked)} tracked path(s) under the directories the tree "
        "enumerates; directories it lists without children are opaque, and "
        "dot-entries and __init__.py are excluded"
    )
    return Result(
        "plan-files",
        not problems,
        f"PLAN.md's layout and the repository agree both ways ({scope})"
        if not problems
        else f"{'; '.join(problems)} ({scope})",
        count=len(missing) + len(undocumented),
    )


@check("evidence-committed")
def measured_claims_have_committed_artifacts() -> Result:
    """Numbers in the support matrix must trace to a committed result file.

    Convention 16 asks for evidence logs; `.gitignore` excludes `outputs/`, so
    without a committed evidence directory every measured claim lives only in
    hand-written prose. A `.gitkeep` satisfies "a file exists" and nothing else,
    so the artifact has to parse and carry the fields that make it traceable.
    """
    matrix = REPO / "docs" / "support-matrix.md"
    if not matrix.exists():
        return Result(
            "evidence-committed",
            False,
            "docs/support-matrix.md is absent; deleting the document whose numbers "
            "this check traces turns the check green, which is not the same as the "
            "numbers being traceable",
        )
    tracked = subprocess.run(
        ["git", "ls-files", "docs/evidence"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.split()
    usable, rejected = [], []
    for name in tracked:
        if not name.endswith(".json"):
            continue
        try:
            payload = json.loads((REPO / name).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            rejected.append(f"{name} ({type(exc).__name__})")
            continue
        missing = [k for k in ("git_commit", "config") if k not in payload]
        if missing:
            rejected.append(f"{name} (no {'/'.join(missing)})")
        else:
            usable.append(name)
    if usable:
        note = f"; ignored {len(rejected)}" if rejected else ""
        return Result("evidence-committed", True, f"{len(usable)} traceable artifact(s){note}")
    return Result(
        "evidence-committed",
        False,
        "docs/support-matrix.md states measured results but docs/evidence/ holds no "
        "committed run record carrying git_commit and config"
        + (f"; rejected: {', '.join(rejected)}" if rejected else ""),
    )


# What an axis variant needs installed to be more than a label. Derived per
# variant file so that adding configs/optim/muon.yaml without an implementation
# fails here instead of passing by not being listed.
AXIS_PACKAGES = {
    "attn/fa2": ("flash-attn",),
    "attn/fa3": ("flash-attn",),
    "attn/fa4": ("flash-attn",),
    "kernel/liger": ("liger-kernel",),
    "kernel/fla": ("flash-linear-attention", "causal-conv1d"),
    "optim/adamw_8bit": ("bitsandbytes",),
    "optim/muon": ("pytorch-optimizer",),
    "parallel/zero2": ("deepspeed",),
    "parallel/zero3": ("deepspeed",),
    # `nvidia-dali` on PyPI is NVIDIA's own placeholder — its summary reads "A fake
    # package to warn the user they are not installing the correct package". The
    # CUDA-13 build is a separate distribution on pypi.nvidia.com, so the name here
    # could never be satisfied by a correct install.
    "dataloader/dali": ("nvidia-dali-cuda130",),
    "dataloader/dali_packed": ("nvidia-dali-cuda130",),
    # peft/loss/framework were exempt via NON_AXIS_GROUPS, which hid two real
    # gaps: bitsandbytes is in 2 of 6 envs while qlora is offered to all six, and
    # no environment has a GradCache implementation at all — while three shipped
    # manifests already assign GPUs to the loss axis.
    "peft/qlora": ("bitsandbytes",),
    # `loss/cached_mnrl` was here, needing `gradcache` (`grad-cache` is a 404;
    # upstream declares `name="GradCache"`, which PEP 503 normalises to
    # `gradcache`). It needs no package now: `axes._loss` implements the three
    # passes directly against `probe/steps.py::encode` and `embedding.py::info_nce`
    # rather than importing the library, which is only in `envs/native`'s lock —
    # importing it would make the axis raise ImportError everywhere else while the
    # arithmetic it needs is `torch.autograd.backward(reps, grad_tensors=cache)`.
    # It has moved to AXIS_NEEDS_NOTHING; leaving it here would have this check
    # fail the moment anyone drops a dependency nothing imports.
    "framework/unsloth": ("unsloth",),
    "framework/ms_swift": ("ms-swift",),
    "framework/sentence_transformers": ("sentence-transformers",),
    "framework/tevatron": ("tevatron",),
    "framework/axolotl": ("axolotl",),
}
# Variants that need nothing beyond the core stack.
AXIS_NEEDS_NOTHING = frozenset(
    {
        "attn/sdpa",
        "attn/flex",
        "kernel/none",
        "precision/bf16",
        "optim/adamw_fused",
        "parallel/single",
        # The all-gather is torch.distributed's own, in the loss closure
        # (`axes._gather_with_grad`); nothing is installed for it.
        "parallel/single_cross_device",
        "parallel/ddp",
        "parallel/fsdp2",
        "dataloader/torch",
        "dataloader/torch_packed",
        # Tokenising ahead of the timed window is the caller's own processor doing
        # the same work earlier (`axes.pretokenize`); nothing is installed for it.
        "dataloader/torch_pretokenized",
        # Packing without a tokenizer needs rows that already carry unpadded ids,
        # which is what pretokenize produces — the combination `axes.PackedCollate`
        # supports with nothing handed in, and which no config could express.
        "dataloader/torch_packed_pretokenized",
        "peft/full",
        "peft/lora",
        "loss/mnrl",
        "loss/cached_mnrl",
        "freeze/none",
        "freeze/vision_tower",
        "freeze/ple",
        "freeze/vision_and_ple",
        "compile/none",
        "compile/default",
        "compile/max_autotune",
        "compile/regional",
        "framework/native",
    }
)
# Config groups that select something other than an optimisation, so no package
# backs them.
# Groups that select something other than an optimisation, so no package backs
# them. `framework`, `peft`, `loss`, `freeze` and `compile` used to be here and
# are not: exempting a whole group from the package check is how `peft/qlora`
# (bitsandbytes in 2 of 6 envs) and `loss/cached_mnrl` (no GradCache anywhere)
# stayed invisible while `axis-fields` counted both as axes.
NON_AXIS_GROUPS = frozenset({"data", "model", "run", "train", "experiment"})


@check("axis-packages")
def axes_have_their_packages() -> Result:
    """A config group offering an axis whose package is absent everywhere is a
    knob that cannot do anything.

    Every variant file must be classified. An unclassified one is a failure, not
    a pass: silence is how `optim/muon` sat here needing an implementation that
    no environment had.
    """
    found = [p for p in [REPO / "uv.lock", *(REPO / "envs").glob("*/uv.lock")] if p.exists()]
    if empty := _nothing_to_check(found, "uv.lock files"):
        return Result("axis-packages", False, empty)
    locks = {p: p.read_text() for p in found}
    problems = []
    for path in sorted(CONFIGS.rglob("*.yaml")):
        group = path.parent.name
        if path.parent == CONFIGS or group in NON_AXIS_GROUPS:
            continue
        variant = f"{group}/{path.stem}"
        if variant in AXIS_NEEDS_NOTHING:
            continue
        packages = AXIS_PACKAGES.get(variant)
        if packages is None:
            problems.append(f"{variant} is unclassified: state its packages or that it needs none")
            continue
        for package in packages:
            # "Backed somewhere", which is weaker than the property that finally
            # matters: a package in one image cannot satisfy an axis for the image
            # that runs it. Pairing each framework image against the axes its
            # manifests request needs configs/experiment/, and belongs with those
            # manifests rather than here. Until then this reports which envs have
            # it so the gap is visible rather than implied.
            envs = [p.parent.name for p, text in locks.items() if f'name = "{package}"' in text]
            if not envs:
                problems.append(f"{variant} needs {package}, absent from every env")
    return Result(
        "axis-packages",
        not problems,
        # Not "backed": this is a string search for the distribution name in some
        # env's lock. It does not prove the package installs, imports, builds its
        # CUDA kernels, or is present in the image that runs that axis.
        "every offered axis names a package in some env lock"
        if not problems
        else f"{len(problems)} problem(s): {'; '.join(problems)}",
        count=len(problems),
    )


def _normalise(distribution: str) -> str:
    """PEP 503 normalisation. `uv export` prints `pytorch-optimizer`, the metadata
    says `pytorch_optimizer`, and comparing the two unnormalised finds nothing."""
    return re.sub(r"[-_.]+", "-", distribution).lower()


def _repo_module_names() -> set[str]:
    """Modules that resolve inside this repository rather than from an install.

    `tests/` adds `scripts/` to `sys.path` and imports `audit_plan`, `orchestrate`,
    `publish_result` and `report` from it, none of which are distributions.
    """
    return (
        {path.name for path in REPO.iterdir() if path.is_dir()}
        | {path.stem for path in (REPO / "scripts").glob("*.py")}
        | {"__future__"}
    )


def _per_image_adapters() -> set[Path]:
    """The probe adapters that live in one framework image each.

    `trainbench/probe/registry.py` imports them lazily and says why: unsloth is
    absent from the axolotl image and importing it there would fail for reasons
    unrelated to the question. `envs/` has one directory per such image, and each
    directory's own lock pins that framework — so the root lock is the wrong place
    to demand them, and `_demanded_imports` skips these files.
    """
    envs = REPO / "envs"
    if not envs.is_dir():
        return set()
    return {
        REPO / "trainbench" / "probe" / f"{path.name.replace('-', '_')}.py"
        for path in envs.iterdir()
        if path.is_dir()
    }


def _demanded_imports() -> dict[str, set[str]]:
    """Third-party top-level modules the documented setup has to provide, and who
    imports them.

    Collected from `tests/`, `trainbench/` and `scripts/` — the suite, the package
    and the entry points are all things the documented command claims to make
    runnable. Function-level imports count. They do not break collection the way a
    module-level one does, but the run still fails when the module is absent, and
    the import this widening was written for is one of them: `optim=muon`'s
    `from pytorch_optimizer import Muon`, inside `trainbench/axes.py`, reached a
    clean clone as an axis that refused itself and an ablation missing a row.

    The framework adapters are skipped (`_per_image_adapters`), so a lazy import
    added inside one of those files is not covered here.
    """
    modules: dict[str, set[str]] = {}
    local = _repo_module_names()
    skipped = _per_image_adapters()
    roots = [REPO / name for name in ("tests", "trainbench", "scripts")]
    for path in sorted(p for root in roots for p in root.rglob("*.py")):
        if path in skipped:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root not in sys.stdlib_module_names and root not in local:
                    modules.setdefault(root, set()).add(str(path.relative_to(REPO)))
    return modules


def _distributions_for(modules: dict[str, set[str]]) -> dict[str, str]:
    """Module name -> distribution name, read off what is installed here.

    `import yaml` comes from `pyyaml` and `import hydra` from `hydra-core`; nothing
    in the source says so, and the installed metadata is the only place that does.
    A module this environment does not have falls back to its own name, which is
    reported as missing either way — that is the safe direction.
    """
    from importlib.metadata import packages_distributions

    installed = packages_distributions()
    return {module: (installed.get(module) or [module])[0] for module in modules}


def _locked_distributions(flags: str) -> tuple[set[str], str | None]:
    """What `uv sync <flags>` would install, as normalised distribution names."""
    run = subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", *flags.split()],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if run.returncode != 0:
        tail = (run.stderr.strip().splitlines() or ["no output"])[-1]
        return set(), f"`uv export{flags}` exits {run.returncode}: {tail}"
    found = re.findall(r"^([A-Za-z0-9._-]+)==", run.stdout, re.M)
    return {_normalise(name) for name in found}, None


@check("doc-commands")
def documented_commands_are_runnable() -> Result:
    """README/AGENTS commands must work as written, and install what this repo imports.

    The documented bootstrap once installed no extras, so `uv run pytest` failed
    on a missing hydra while local development used a different flag set. And the
    documented smoke command was rejected by this project's own batch validator
    for as long as that validator existed, because nothing ever ran it — so the
    entry point commands are executed here rather than pattern-matched.

    The install half used to be a regex asking whether the `uv sync` line carried
    `--extra compose`, justified as "but tests import hydra". Hydra stopped being
    the only thing the tests import long before anyone noticed: `peft`, `datasets`
    and `transformers` are all in the `native` extra, no documented command asks
    for it, and this reported `5 documented command(s) install what the tests need`
    throughout. A rule naming one package cannot answer a question about all of
    them, so the imports are collected from the source and each one is looked for
    in the lock the documented command actually resolves to.

    Collected from the package and the entry points as well as from `tests/`. The
    suite-only scan asked a narrower question than the one the answer was used
    for: `optim=muon`'s only import of `pytorch-optimizer` is lazy and inside
    `trainbench/axes.py`, so it appeared in no test, and a clean clone got an
    ablation with the optim axis quietly refusing itself while this check reported
    that the documented command installs everything.
    """
    problems = []
    found = 0
    modules = _demanded_imports()
    if empty := _nothing_to_check(modules, "third-party imports under tests/ or the package"):
        return Result("doc-commands", False, empty)
    distributions = _distributions_for(modules)
    checked_locks: dict[str, set[str]] = {}
    for name in ("README.md", "AGENTS.md"):
        text = (REPO / name).read_text()
        found += len(re.findall(r"uv sync[^\n`]*", text)) + len(
            re.findall(r"python scripts/env_report\.py[^\n`]*", text)
        )
        for match in re.finditer(r"uv sync([^\n`]*)", text):
            flags = match.group(1).rstrip()
            if flags not in checked_locks:
                locked, failure = _locked_distributions(flags)
                checked_locks[flags] = locked
                if failure:
                    problems.append(f"{name}: {failure}")
                    continue
            locked = checked_locks[flags]
            missing = sorted(
                f"{module} ({distributions[module]}, imported by "
                f"{', '.join(sorted(modules[module]))})"
                for module in modules
                if _normalise(distributions[module]) not in locked
            )
            if missing:
                problems.append(
                    f"{name}: `uv sync{flags}` installs {len(locked)} distribution(s) but the "
                    f"repo imports {len(missing)} it does not provide: {'; '.join(missing)}"
                )
        for match in re.finditer(r"python (scripts/env_report\.py[^\n`]*)", text):
            command = match.group(1).split()
            run = subprocess.run(
                [sys.executable, *command],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if run.returncode != 0:
                tail = (run.stderr.strip().splitlines() or ["no output"])[-1]
                problems.append(f"{name}: `{' '.join(command)}` exits {run.returncode}: {tail}")
    if empty := _nothing_to_check(found, "documented commands in README.md or AGENTS.md"):
        return Result("doc-commands", False, empty)
    scope = (
        f"{found} documented command(s); {len(modules)} third-party module(s) imported under "
        f"tests/, trainbench/ or scripts/ ({len(_per_image_adapters())} per-image probe "
        "adapter(s) skipped), resolved to distributions by this environment's metadata and "
        "looked for in the lock each documented `uv sync` produces"
    )
    return Result(
        "doc-commands",
        not problems,
        f"every documented command runs as written and installs what this repo imports ({scope})"
        if not problems
        else f"{'; '.join(problems)} ({scope})",
        count=len(problems),
    )


FRAMEWORK_DOCKERFILE = REPO / "docker" / "Dockerfile.framework"
# uv reports staleness on stderr; everything else it can fail with is a different
# problem and is reported as itself.
LOCK_IS_STALE = "needs to be updated"


def _lock_is_current(directory: Path) -> str | None:
    """None if the `uv.lock` there still agrees with its `pyproject.toml`.

    `uv sync --frozen` does not ask this and neither did anything else, which is
    how five of six env locks drifted a whole dependency behind the root
    `pyproject.toml`. `--check` is the flag that asks.
    """
    run = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "NO_COLOR": "1"},
    )
    if run.returncode == 0:
        return None
    tail = (run.stderr.strip().splitlines() or ["no output"])[-1]
    return "stale: regenerate with `uv lock`" if LOCK_IS_STALE in tail else f"unverified: {tail}"


@check("env-locks")
def image_locks_agree_with_their_projects() -> Result:
    """Every lock an image installs from is current, and the build asserts it.

    Both halves are one property. A lock that has drifted from its
    `pyproject.toml` puts something in the image other than what the study
    declared, and version is the confound this benchmark must keep visible — so
    the drift has to be impossible rather than merely unlikely.

    Nothing asked either half for as long as both were wrong. The Dockerfile
    synced with `--frozen`, which installs from the lock without checking that the
    lock is current, under a comment claiming a stale lock would fail the build.
    Five of the six env locks were stale under that comment.

    A failure uv reports for some other reason is `unverified`, not a pass: an
    audit that cannot reach the answer has not got the answer.
    """
    directories = sorted(path.parent for path in (REPO / "envs").glob("*/uv.lock"))
    if empty := _nothing_to_check(directories, "envs/*/uv.lock files"):
        return Result("env-locks", False, empty)
    # The root lock last, because `doc-commands` exports from it and a drifted one
    # would make that check answer about a resolution no image ever installs.
    directories.append(REPO)
    problems = [
        f"{directory.relative_to(REPO) if directory != REPO else '.'}/uv.lock is {failure}"
        for directory in directories
        if (failure := _lock_is_current(directory))
    ]
    if not FRAMEWORK_DOCKERFILE.exists():
        return Result(
            "env-locks",
            False,
            f"{FRAMEWORK_DOCKERFILE.relative_to(REPO)} is absent; nothing then says how the "
            "images install, and a fresh lock alone does not make a build honour it",
        )
    # Instruction lines only. A comment mentioning `uv sync --locked` is prose, and
    # counting it would let a Dockerfile that runs nothing at all satisfy the
    # emptiness guard below — the shape `_nothing_to_check` exists for.
    syncs = [
        match.group(1)
        for line in FRAMEWORK_DOCKERFILE.read_text().splitlines()
        if not line.lstrip().startswith("#")
        for match in re.finditer(r"uv sync([^\n&|]*)", line)
    ]
    if empty := _nothing_to_check(syncs, "`uv sync` invocations in docker/Dockerfile.framework"):
        return Result("env-locks", False, empty)
    problems += [
        f"docker/Dockerfile.framework: `uv sync{flags.rstrip()}` does not pass --locked, so a "
        "stale lock builds silently"
        for flags in syncs
        if "--locked" not in flags.split()
    ]
    scope = (
        f"{len(directories)} lock(s) re-resolved with `uv lock --check`; {len(syncs)} "
        "`uv sync` invocation(s) in docker/Dockerfile.framework"
    )
    return Result(
        "env-locks",
        not problems,
        f"every lock agrees with its pyproject.toml and every image sync asserts it ({scope})"
        if not problems
        else f"{'; '.join(problems)} ({scope})",
        count=len(problems),
    )


# Where the ABI claim for a URL-pinned binary lives. Resolved from REPO at call
# time, not bound here, so a check run against another tree reads that tree's.
PREBUILT_WHEELS = Path("docs/prebuilt-wheels.yaml")
# PEP 427, without the optional build tag none of ours carry:
# flash_attn-2.8.3.post1-cp313-cp313-linux_x86_64.whl
WHEEL_FILENAME = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>.+)\.whl$"
)
# The release tag carries the ABI the binary was built for, because a wheel
# filename cannot: `flash-attn-2.8.3.post1-cu130-torch2.13.0-cp313`.
TORCH_CLAIM = re.compile(r"(?:^|-)torch(?P<version>\d+(?:\.\d+)*)(?:-|$)")
CUDA_CLAIM = re.compile(r"(?:^|-)(?P<cuda>cu\d+)(?:-|$)")
DOCKERFILE_ARCHS = re.compile(r'TRAINBENCH_CUDA_ARCHS="(?P<archs>[^"]*)"')


def _lock_contents(directory: Path) -> tuple[str, dict[str, dict]]:
    """`requires-python` and the packages, keyed by name, of one uv.lock."""
    lock = tomllib.loads((directory / "uv.lock").read_text())
    return lock.get("requires-python", ""), {p["name"]: p for p in lock.get("package", [])}


def _url_sourced(directory: Path) -> dict[str, str]:
    """Package name -> URL, for everything a lock installs from a bare URL.

    These are the ones the resolver checks nothing about. A registry or git
    source is resolved against metadata; a URL wheel is taken at its word.
    """
    _, packages = _lock_contents(directory)
    return {
        name: url
        for name, package in packages.items()
        if (url := (package.get("source") or {}).get("url"))
    }


def _admits_only(requires_python: str, minor: int) -> bool:
    """Whether a lock's requires-python admits 3.<minor> and no neighbouring minor.

    Containment alone is not the property: `>=3.13` contains 3.13 and also every
    later interpreter, and a cp313 binary loaded by 3.14 is the failure this is
    here to stop.
    """
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    try:
        spec = SpecifierSet(requires_python)
    except InvalidSpecifier:
        return False
    return spec.contains(f"3.{minor}.0") and not any(
        spec.contains(f"3.{neighbour}.0") for neighbour in (minor - 1, minor + 1)
    )


def prebuilt_wheel_problems(
    record: dict, requires_python: str, packages: dict[str, dict], archs_declared: str | None
) -> list[str]:
    """Everything that disagrees about one prebuilt wheel.

    The wheel is a binary we compiled, pinned by URL. Nothing between here and a
    pod re-checks what it was built against: uv installs a URL wheel without
    asking, and the mismatch surfaces as a CUDA error mid-run, on a pod, after
    the pod-hour has been spent. So the claim is compared here — against the
    artifact's own name, against what the lock resolves, and against the archs
    the image declares.
    """
    env, package = record.get("env"), record.get("package")
    where = f"docs/prebuilt-wheels.yaml {env}/{package}"
    entry = packages.get(package)
    if entry is None:
        return [
            f"{where}: envs/{env}/uv.lock has no {package}; the record outlived what it records"
        ]

    problems = []
    url = (entry.get("source") or {}).get("url")
    if url is None:
        kind = ", ".join(entry.get("source") or {}) or "nothing"
        problems.append(
            f"{where}: the lock installs {package} from {kind}, not from the recorded wheel URL — "
            "the prebuilt binary is not what the image gets"
        )
    elif url != record.get("url"):
        problems.append(f"{where}: the lock installs {url}, not the recorded URL")
    if entry.get("version") != record.get("version"):
        problems.append(
            f"{where}: the lock resolves {package} {entry.get('version')}, recorded "
            f"{record.get('version')}"
        )
    digests = [wheel.get("hash") for wheel in entry.get("wheels", [])]
    if digests != [f"sha256:{record.get('sha256')}"]:
        problems.append(f"{where}: the lock pins {digests}, recorded sha256:{record.get('sha256')}")

    abi = record.get("abi") or {}
    # The artifact's own name. A record may not claim an ABI its filename denies.
    parts = urllib.parse.urlparse(record.get("url") or "").path.split("/")
    filename = WHEEL_FILENAME.match(parts[-1]) if parts else None
    if filename is None:
        problems.append(f"{where}: {parts[-1] if parts else 'the url'} is not a PEP 427 wheel name")
    else:
        for claim, actual, what in (
            (record.get("version"), filename["version"], "version"),
            (abi.get("python"), filename["python"], "python tag"),
        ):
            if claim != actual:
                problems.append(
                    f"{where}: the wheel filename says {what} {actual}, recorded {claim}"
                )
    # The release tag, which is where the torch/CUDA ABI is stated at all.
    tag = parts[-2] if len(parts) > 1 else ""
    torch_claimed = TORCH_CLAIM.search(tag)
    cuda_claimed = CUDA_CLAIM.search(tag)
    if torch_claimed is None or cuda_claimed is None:
        problems.append(
            f"{where}: the release tag `{tag}` states no torch/CUDA ABI, so nothing says what this "
            "binary was built against"
        )
    else:
        if torch_claimed["version"] != abi.get("torch"):
            problems.append(
                f"{where}: the release tag was built for torch {torch_claimed['version']}, "
                f"recorded {abi.get('torch')}"
            )
        if cuda_claimed["cuda"] != abi.get("cuda"):
            problems.append(
                f"{where}: the release tag was built for {cuda_claimed['cuda']}, "
                f"recorded {abi.get('cuda')}"
            )

    # What the lock resolves. This is the mutation that has no other alarm: a
    # torch bump re-locks cleanly and breaks the binary at load time.
    torch = packages.get("torch")
    if torch is None:
        problems.append(f"{where}: envs/{env}/uv.lock resolves no torch to compare the ABI against")
    else:
        resolved, _, local = str(torch.get("version")).partition("+")
        if resolved != abi.get("torch"):
            problems.append(
                f"{where}: the lock resolves torch {resolved}, but the wheel was built against "
                f"{abi.get('torch')} — the C++ ABI it links is gone"
            )
        if local != abi.get("cuda"):
            problems.append(
                f"{where}: the lock resolves torch on {local or 'no CUDA build'}, but the "
                f"wheel was built for {abi.get('cuda')}"
            )

    python_tag = str(abi.get("python") or "")
    if not (minor := re.fullmatch(r"cp3(\d+)", python_tag)):
        problems.append(f"{where}: `{python_tag}` is not a CPython 3.x tag this can compare")
    elif not _admits_only(requires_python, int(minor[1])):
        problems.append(
            f"{where}: envs/{env}/uv.lock admits requires-python {requires_python!r}, which is not "
            f"3.{minor[1]} alone — the wheel is {python_tag} only"
        )

    if archs_declared is None:
        problems.append(
            f"{where}: docker/Dockerfile.framework declares no TRAINBENCH_CUDA_ARCHS, so nothing "
            "says which GPUs the image expects the wheel to have kernels for"
        )
    elif (declared := [a for a in archs_declared.split(";") if a]) != [
        str(a) for a in abi.get("cuda_archs") or []
    ]:
        problems.append(
            f"{where}: the image declares CUDA archs {declared}, the wheel was built for "
            f"{abi.get('cuda_archs')} — flash-attn emits no PTX, so an arch outside the wheel is "
            "a dead pod, not a slow one"
        )
    return problems


@check("prebuilt-wheels")
def prebuilt_wheels_match_what_the_locks_resolve() -> Result:
    """Every URL-pinned binary wheel is the one docs/prebuilt-wheels.yaml records.

    flash-attn is compiled here rather than downloaded: 13,663 s on a GitHub
    runner, 88% of the native image build. The wheel that replaced that build is
    ours, and it is valid for exactly one torch, one CUDA, one interpreter and
    one set of GPU archs — none of which uv re-checks when it installs a URL.

    Both directions are the same property. A record whose lock no longer installs
    it is a build that silently went back to compiling; a URL wheel with no record
    is a binary in the images whose provenance nobody wrote down.
    """
    record_file = REPO / PREBUILT_WHEELS
    if not record_file.exists():
        return Result("prebuilt-wheels", False, f"{PREBUILT_WHEELS} is missing")
    records = (yaml.safe_load(record_file.read_text()) or {}).get("wheels") or []
    if empty := _nothing_to_check(records, f"wheels in {PREBUILT_WHEELS}"):
        return Result("prebuilt-wheels", False, empty)
    archs = (
        found["archs"]
        if FRAMEWORK_DOCKERFILE.exists()
        and (found := DOCKERFILE_ARCHS.search(FRAMEWORK_DOCKERFILE.read_text()))
        else None
    )
    problems = []
    for record in records:
        directory = REPO / "envs" / str(record.get("env"))
        if not (directory / "uv.lock").exists():
            problems.append(
                f"docs/prebuilt-wheels.yaml {record.get('env')}/{record.get('package')} names "
                f"envs/{record.get('env')}, which has no uv.lock"
            )
            continue
        problems += prebuilt_wheel_problems(record, *_lock_contents(directory), archs)
    recorded = {(r.get("env"), r.get("package")) for r in records}
    problems += [
        f"envs/{directory.name}/uv.lock installs {name} from a URL with no record in "
        f"{PREBUILT_WHEELS}: {url}"
        for directory in sorted(path.parent for path in (REPO / "envs").glob("*/uv.lock"))
        for name, url in sorted(_url_sourced(directory).items())
        if (directory.name, name) not in recorded
    ]
    scope = (
        f"{len(records)} recorded wheel(s); {len(list((REPO / 'envs').glob('*/uv.lock')))} env "
        "lock(s) scanned for URL sources"
    )
    return Result(
        "prebuilt-wheels",
        not problems,
        f"every prebuilt wheel matches the ABI its lock resolves ({scope})"
        if not problems
        else f"{'; '.join(problems)} ({scope})",
        count=len(problems),
    )


@check("data-pinned")
def measured_data_is_pinned() -> Result:
    """A run is not reproducible if its corpus is a moving branch.

    Rejecting only null would let `revision: main` through, which reads as pinned
    and is not — the Hub re-resolves it on every pull.
    """
    from trainbench.config_schema import COMMIT_SHA

    configs = sorted((CONFIGS / "data").glob("*.yaml"))
    if empty := _nothing_to_check(configs, "config files in configs/data/"):
        return Result("data-pinned", False, empty)
    problems = []
    for path in configs:
        revision = (yaml.safe_load(path.read_text()) or {}).get("revision")
        if revision in (None, "null", ""):
            problems.append(f"{path.name} has no pinned revision")
        elif not COMMIT_SHA.fullmatch(str(revision)):
            problems.append(f"{path.name} pins {revision!r}, which is not a commit sha")
    return Result(
        "data-pinned",
        not problems,
        "every data config pins a commit sha" if not problems else "; ".join(problems),
    )


def model_spec_problems(spec: dict, configs: dict) -> list[str]:
    """Every specified value compared against the config value it governs."""
    problems = []
    for name in sorted(set(spec) ^ set(configs)):
        side = "docs/model-spec.yaml" if name in spec else "configs/model"
        problems.append(f"{name} is only in {side}")
    for name in sorted(set(spec) & set(configs)):
        actual = configs[name]
        expected_id, actual_id = spec[name].get("hf_id"), actual.get("hf_id")
        if expected_id != actual_id:
            problems.append(f"{name}.hf_id: spec {expected_id!r} != config {actual_id!r}")
        for field, expected in sorted((spec[name].get("config") or {}).items()):
            if field not in actual:
                problems.append(f"{name}.{field} is specified but absent from the config")
            elif actual[field] != expected:
                problems.append(f"{name}.{field}: spec {expected!r} != config {actual[field]!r}")
    return problems


@check("model-spec")
def model_spec_matches_config() -> Result:
    """The per-model usage decisions in docs/model-spec.yaml must equal the config.

    Compared value by value. Checking that the words appear somewhere passes
    whether the value is true or false, which is precisely the drift it was meant
    to stop: with last-token pooling, add_generation_prompt silently changes which
    token becomes the embedding.
    """
    spec_path = REPO / "docs" / "model-spec.yaml"
    if not spec_path.exists():
        return Result("model-spec", False, "docs/model-spec.yaml is missing")
    spec = (yaml.safe_load(spec_path.read_text()) or {}).get("models") or {}
    configs = {
        p.stem: yaml.safe_load(p.read_text()) or {} for p in (CONFIGS / "model").glob("*.yaml")
    }
    if empty := _nothing_to_check(spec and configs, "models in both the spec and configs/model/"):
        return Result("model-spec", False, empty)
    problems = model_spec_problems(spec, configs)
    return Result(
        "model-spec",
        not problems,
        f"{len(spec)} model(s) match docs/model-spec.yaml value for value"
        if not problems
        else "; ".join(problems),
    )


# Rows in the synthetic dataset this check hands the dataloader axis. Equal to
# `configs/train/default.yaml`'s `batch_size`, so one batch is one full batch and
# no variant is judged on a short final one.
_BATCH_ROWS = 16
# Applied to every variant, unlike the per-variant companions below. Worker
# processes are a property of a run, not of whether an axis value can be applied,
# and `configs/data/*.yaml` ask for 8 of them — which this check would pay for
# once per variant, to build one batch it pulls in-process.
AXIS_VALUE_BASE_OVERRIDES = ("data.num_workers=0",)
# Length of each synthetic sequence. Uniform, see `_AxisValueRows`.
_ROW_TOKENS = 4


class _AxisValueRows:
    """The smallest dataset the dataloader axis will accept as real.

    `axis-values` called `assemble` with no dataset until 2026-08-02, which made
    it structurally vacuous for that axis in both directions. `axes._dataloader`
    returns at `if dataset is None` before it reaches packing or pretokenize, so
    `PackedCollate.__call__` could be replaced with `raise NotImplementedError`
    and this check's output stayed identical to the byte — while `loss/cached_mnrl`
    was reported inert for the opposite reason, GradCache refusing a `None`
    dataset it would have accepted.

    `input_ids` is declared *and* handed over, because `_dataloader` asks the two
    questions separately: the column list is what the dataset says about itself,
    the first row is what the timed step is actually given.

    Rows are uniform length so the non-packed variants, which get torch's default
    collate here, still stack. Ragged rows would make `dataloader/torch` fail on a
    rectangle the audit itself built, and counting that as the axis being
    inapplicable would be this check inventing a defect.

    An axis can be applicable to one shape of data and not another, and a single
    fixture answers for whichever shape it happens to have. So there are two, and
    they differ in exactly one thing: whether a row carries an image. Measured
    empirically, that difference moves `loss/cached_mnrl` and nothing else.
    """

    carries_image = False

    @property
    def column_names(self) -> list[str]:
        return [
            "input_ids",
            "attention_mask",
            *(["qry_image"] if self.carries_image else []),
        ]

    def __len__(self) -> int:
        return _BATCH_ROWS

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        if index >= _BATCH_ROWS:
            raise IndexError(index)
        # The mask travels with the ids because that is what a tokenizer hands back
        # and what `axes.PackedCollate` reads: a row that records its padding
        # nowhere is refused, and `dataloader/torch_packed*` would then be counted
        # inapplicable over a fixture no processor would have produced.
        row = {
            "input_ids": torch.arange(_ROW_TOKENS),
            "attention_mask": torch.ones(_ROW_TOKENS, dtype=torch.long),
        }
        if self.carries_image:
            # A tensor rather than None: `axes.image_columns` reads a `None` here as
            # the column being empty, and torch's default collate cannot stack it,
            # which would make the two fixtures differ by a second thing.
            row["qry_image"] = torch.zeros(3, 2, 2)
        return row


class _AxisValueRowsWithImages(_AxisValueRows):
    """The shape of data this study actually measures.

    Both configured subsets are MMEB draws and `configs/data/speed.yaml` records
    "0 rows without a query image or positive", so every measured run reads rows
    that carry images. A value that applies only to the text-only fixture cannot
    be turned on by any run this study is configured to make — which is why the
    two are reported apart rather than averaged into one number.
    """

    carries_image = True


# Values a group needs alongside it to compose at all. Not exemptions: without
# these the schema rejects the override and the value would be miscounted as
# inapplicable for a reason that has nothing to do with whether axes.py can apply
# it. `max_autotune` spends its first steps benchmarking kernels, which the schema
# requires be discarded (MAX_AUTOTUNE_MIN_WARMUP_STEPS).
AXIS_VALUE_COMPANIONS: dict[str, tuple[str, ...]] = {
    "compile/max_autotune": ("train.warmup_discard_steps=20",),
    # The PLE tables exist only in gemma-4; the schema refuses the axis on the
    # other two models rather than letting it freeze nothing.
    "freeze/ple": ("model=gemma4_e2b",),
    "freeze/vision_and_ple": ("model=gemma4_e2b",),
    # Gated DeltaNet is a Qwen3.5 layer, so `axes.FLA_ARCHS` is `{qwen3_5}` and
    # the default model refuses `kernel=fla` on architecture before the packages
    # are ever asked about. Without this the value is uncountable in every
    # environment, including an image that ships fla and CUDA.
    "kernel/fla": ("model=qwen3_5_0_8b",),
}


def flag_knob_values() -> tuple[dict[str, list[str]], list[str]]:
    """Axis knobs varied by dotted override, and their offered values.

    `axis-values` enumerates variant *files* per config group, so a group that
    varies by `train.gradient_checkpointing=selective` instead of by a file per
    value was invisible to it — `train` is in `NON_AXIS_GROUPS` and every value of
    both its knobs went uncounted. That is not an exemption anyone chose: the
    group's `default.yaml` explains why a file per value would mean a file per
    cell of the cross with `train.offload`, and then records that the audit cannot
    see the two knobs as the cost. This is the cost being paid instead.

    Read off the schema rather than listed here, so a new `Axis()` in one of these
    sections is counted without being remembered into a list. A knob whose values
    cannot be enumerated is returned as a problem rather than skipped — silence is
    how the group got here.
    """
    from trainbench.config_schema import BenchConfig, Strict

    knobs: dict[str, list[str]] = {}
    problems: list[str] = []
    for section, info in BenchConfig.model_fields.items():
        sub = info.annotation
        if section not in NON_AXIS_GROUPS:
            continue
        if not (isinstance(sub, type) and issubclass(sub, Strict)):
            continue
        for name, spec in sub.model_fields.items():
            if not (spec.json_schema_extra or {}).get("axis"):
                continue
            values = [str(v) for v in get_args(spec.annotation)]
            if not values:
                problems.append(
                    f"{section}.{name} is an axis whose values are not enumerable from "
                    f"{spec.annotation!r}, so this check cannot try them"
                )
                continue
            knobs[f"{section}.{name}"] = values
    return knobs, problems


@check("axis-values")
def every_offered_axis_value_can_be_applied() -> Result:
    """How many of the values each axis group offers can actually be applied.

    `axis-wired` asks whether an axis has an apply site and a capture probe. It is
    a membership test on knob names, and an axis passes it while accepting exactly
    one value — the inert one. Both readings are needed and neither substitutes for
    the other: after Wave 2, `axis-wired` fell from 12 unwired to 2 while 7 of the
    12 ablation groups still took only their default. That is the state
    `docs/review-findings.md` D4 describes as "the same experiment under different
    names", and the check meant to catch it was reporting progress.

    Each variant is composed and pushed through all four call sites, with a
    synthetic dataset, and one batch is pulled through whatever loader comes back.
    `UnappliedAxis` is the axis refusing the value, which is the honest outcome and
    what gets counted. Any other exception is reported as its own problem rather
    than folded into the count — a value that fails for an unrelated reason has not
    been measured either way.

    The dataset and the batch are both load-bearing, and neither is obvious.
    Passing no dataset made this vacuous for the whole dataloader axis, because
    `axes._dataloader` returns at `if dataset is None` before it reaches packing or
    pretokenize. Passing one but not iterating leaves it half vacuous, because a
    collate is not called until a batch is drawn — so packing, which lives entirely
    in the collate, would still be certified by code that never ran.

    Every variant is tried against two fixtures differing in one thing — whether a
    row carries an image — because applicability is not always a property of the
    axis alone. A value accepted on one shape and refused on the other is reported
    by name and *not counted*, since counting it answers a question nobody asked.
    `loss/cached_mnrl` is the live case: GradCache is implemented, and
    `_split_rows` refuses every batch carrying `pixel_values`, so it applies to a
    text-only subset and to no run this study is configured to make. Counted as
    applicable it read `loss 2/2`, which a reader takes as "GradCache is ready to
    measure".

    What it still does not prove: that packing is *correct*. A collate that
    concatenated wrongly applies the axis as far as this check can see. Equivalence
    is `tests/test_axes.py`'s question and the capture probe's, not this one's.
    """
    sys.path.insert(0, str(REPO))
    import functools

    import torch
    from hydra import compose, initialize_config_dir

    from trainbench import applied, axes
    from trainbench.compose import resolve

    class _Tiny(torch.nn.Module):
        """The stub every value is pushed through.

        It carries transformers' checkpointing hook because `train.gradient_checkpointing`
        is applied through it (`PreTrainedModel._set_gradient_checkpointing`, 5.14.1),
        and a stub without it would report `full` and `selective` inapplicable — the
        check inventing a defect rather than finding one, the same way ragged rows
        would have made `dataloader/torch` fail on a rectangle the audit built.
        """

        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(4, 4)
            self.block.gradient_checkpointing = False

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
            self.block.gradient_checkpointing = True
            self.block._gradient_checkpointing_func = functools.partial(
                torch.utils.checkpoint.checkpoint, **dict(gradient_checkpointing_kwargs or {})
            )

    variants: dict[str, list[str]] = {}
    for path in sorted(CONFIGS.rglob("*.yaml")):
        group = path.parent.name
        if path.parent == CONFIGS or group in NON_AXIS_GROUPS:
            continue
        variants.setdefault(group, []).append(path.stem)
    # Knobs that vary by dotted override instead of by a file per value. Keyed by
    # the knob's dotted name, which is also the override spelling, so they go
    # through the same loop rather than a second one — and are reported apart from
    # each other, since `train`'s two knobs are crossed rather than alternatives.
    knobs, unenumerable = flag_knob_values()
    variants.update(knobs)
    if empty := _nothing_to_check(variants, "axis config groups"):
        return Result("axis-values", False, empty)

    def attempt(config: Any, dataset: Any, verify: str | None = None) -> Exception | None:
        """`None` if every call site accepted the value, else what refused it.

        `verify` names a knob to read back off what was built. Only the flag knobs
        pass one: applicability alone is a weak reading — an apply site gutted to
        `return []` raises nothing and would be counted applied — and for these the
        stub is a model the probe can actually answer about. For a variant *file*
        the same read-back would report `attn`, `precision` and the rest
        undetermined against a four-parameter `nn.Linear`, which says nothing about
        the axis.
        """
        try:
            axes.patch(config)
            axes.load_kwargs(config)
            # What `scripts/bench.py` does before `assemble`, and the only place
            # `dataloader.pretokenize=true` is actually carried out: `_dataloader`
            # inspects the dataset for token ids but never produces them. Without
            # this the axis was certified by a fixture that arrived tokenised.
            if config.dataloader.pretokenize:
                dataset = axes.pretokenize(dataset, lambda row: dict(row))
            built, _ = axes.assemble(
                _Tiny(), config, torch.device("cpu"), "native", dataset=dataset
            )
            # One batch, because building a loader does not run its collate and
            # packing lives entirely in the collate. Constructing the loader and
            # calling the axis applied is how a gutted `PackedCollate.__call__`
            # left this check's output unchanged.
            if built.dataloader is not None:
                next(iter(built.dataloader))
            with axes.step_context(config):
                pass
            if verify is not None:
                state = next(a for a in applied.capture(built, config).axes if a.axis == verify)
                if state.applied != state.requested:
                    raise applied.AppliedMismatch(
                        f"requested {state.requested!r}, read back {state.applied!r} "
                        f"({state.detail.get('reason', state.detail)})"
                    )
        except Exception as exc:  # noqa: BLE001 - reported, never counted as applied
            return exc
        return None

    shapes = {"text-only": _AxisValueRows, "image-carrying": _AxisValueRowsWithImages}
    applicable: dict[str, int] = {}
    inert: list[str] = []
    broken: list[str] = []
    data_dependent: list[str] = []
    for group, names in sorted(variants.items()):
        applicable[group] = 0
        for name in sorted(names):
            variant = f"{group}/{name}"
            overrides = [
                f"{group}={name}",
                *AXIS_VALUE_BASE_OVERRIDES,
                *AXIS_VALUE_COMPANIONS.get(variant, ()),
            ]
            try:
                with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
                    config = resolve(compose(config_name="config", overrides=overrides))[0]
            except Exception as exc:  # noqa: BLE001 - a value that will not compose
                broken.append(f"{variant} (compose {type(exc).__name__}: {str(exc)[:80]})")
                continue
            outcomes = {
                shape: attempt(config, rows(), verify=group if group in knobs else None)
                for shape, rows in shapes.items()
            }
            unexpected = [
                f"{variant} on {shape} data ({type(exc).__name__}: {str(exc)[:80]})"
                for shape, exc in outcomes.items()
                if exc is not None and not isinstance(exc, axes.UnappliedAxis)
            ]
            if unexpected:
                broken.extend(unexpected)
                continue
            accepted = sorted(shape for shape, exc in outcomes.items() if exc is None)
            if len(accepted) == len(outcomes):
                applicable[group] += 1
            elif accepted:
                # Applicable to one shape of data and refused on the other. Counting
                # it would answer a question nobody asked — the study measures the
                # image-carrying shape, so a text-only-only value is a value no
                # configured run can turn on. Named rather than counted, because
                # `loss 2/2` reads as "GradCache is ready to measure" and it is not.
                refusal = next(exc for exc in outcomes.values() if exc is not None)
                data_dependent.append(
                    f"{variant} (applies to {'/'.join(accepted)} data only: {str(refusal)[:90]})"
                )

    total = sum(len(v) for v in variants.values())
    usable = sum(applicable.values())
    for group, count in sorted(applicable.items()):
        if count <= 1 and len(variants[group]) > 1:
            inert.append(f"{group} {count}/{len(variants[group])}")
    problems = []
    if unenumerable:
        problems.append(f"{len(unenumerable)} axis knob(s) not tried: {'; '.join(unenumerable)}")
    if inert:
        problems.append(f"{len(inert)} group(s) offering one usable value: {', '.join(inert)}")
    if data_dependent:
        problems.append(
            f"{len(data_dependent)} value(s) applicable only to data this study does not "
            f"measure: {'; '.join(data_dependent)}"
        )
    if broken:
        problems.append(f"{len(broken)} value(s) failed for another reason: {'; '.join(broken)}")
    return Result(
        "axis-values",
        not problems,
        f"{usable}/{total} offered axis values apply on both {'/'.join(shapes)} data"
        if not problems
        else f"{usable}/{total} applicable on both {'/'.join(shapes)} data; " + "; ".join(problems),
        count=len(unenumerable) + len(inert) + len(data_dependent) + len(broken),
    )


OPEN_VERDICTS = Path("docs/open-verdicts.json")

# What every ledger item carries. Enforced rather than assumed: an item missing
# its pass criterion is a TODO, and a TODO is what this check exists to replace.
VERDICT_FIELDS = (
    "id",
    "axis",
    "finding",
    "owner",
    "verdict",
    "summary",
    "closes_when",
    "anchor",
    "closed",
)
# The criterion a human re-runs. `observed` is what it does today, so a reader can
# tell a criterion that was never run from one that was run and failed.
CLOSES_WHEN_FIELDS = ("criterion", "command", "expected", "observed")
# What the gate itself evaluates. `test` is the strong one: a named test function
# has to exist. `text`/`absent` are for items whose deliverable *is* text — a
# methodology section, a manifest override, a retracted sentence.
ANCHOR_FIELDS = {
    "test": ("file", "names"),
    "text": ("file", "pattern"),
    "absent": ("file", "pattern"),
}


def _defined_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def anchor_holds(anchor: dict[str, Any], root: Path) -> bool:
    """Whether the repository now shows what this item was waiting for."""
    path = root / anchor["file"]
    if not path.exists():
        return False
    text = path.read_text()
    if anchor["kind"] == "test":
        return set(anchor["names"]) <= _defined_names(text)
    found = re.search(anchor["pattern"], text) is not None
    return found if anchor["kind"] == "text" else not found


def _committed_verdict_ids(root: Path) -> set[str]:
    """Every id this ledger has ever carried, read out of git history.

    The obvious way to make this check green is to delete the item. It has
    happened twice here already in other checks — `plan-files` stayed true by
    mentioning less, `doc-commands` was satisfied by removing the import — so the
    ledger is not allowed to be its own only witness. History is the witness: an
    id that was ever committed has to still be there. Its absence is reported by
    name, and getting rid of the evidence means rewriting published history.

    An id that was never committed is not pinned, so a typo caught before the
    commit costs nothing and one caught after it is permanent.

    Added lines are gathered first and searched as one text, rather than matching
    an id per line: a ledger written without indentation puts every id on one
    line, and a per-line pattern would find the first and pin nothing else.
    """
    log = subprocess.run(
        ["git", "log", "-p", "--format=", "--", str(OPEN_VERDICTS)],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout
    added = "\n".join(line for line in log.splitlines() if line.startswith("+"))
    return set(re.findall(r'"id":\s*"([^"]+)"', added))


def verdict_ledger_problems(
    payload: Any, root: Path, committed: set[str]
) -> tuple[list[str], list[str], list[str]]:
    """Problems, open ids, closed ids. Pure, so the tests can drive every state."""
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return ([f"{OPEN_VERDICTS} carries no `items` list"], [], [])
    problems: list[str] = []
    open_ids: list[str] = []
    closed_ids: list[str] = []
    for position, item in enumerate(payload["items"]):
        label = f"item {position}"
        if not isinstance(item, dict):
            problems.append(f"{label} is not an object")
            continue
        label = str(item.get("id", label))
        if missing := [field for field in VERDICT_FIELDS if field not in item]:
            problems.append(f"{label}: missing {', '.join(missing)}")
            continue
        if label in open_ids or label in closed_ids:
            problems.append(f"{label}: duplicate id")
            continue
        closes = item["closes_when"]
        if not isinstance(closes, dict) or any(f not in closes for f in CLOSES_WHEN_FIELDS):
            problems.append(f"{label}: closes_when needs {', '.join(CLOSES_WHEN_FIELDS)}")
            continue
        anchor = item["anchor"]
        kind = anchor.get("kind") if isinstance(anchor, dict) else None
        if kind not in ANCHOR_FIELDS or any(f not in anchor for f in ANCHOR_FIELDS[kind]):
            problems.append(
                f"{label}: anchor must name a kind in {'/'.join(ANCHOR_FIELDS)} "
                f"and carry its fields"
            )
            continue
        # An empty name list is a subset of every file's functions, so the item
        # would close on nothing at all — the vacuous-truth shape this file's
        # `_nothing_to_check` exists for, one level down.
        if kind == "test" and not anchor["names"]:
            problems.append(f"{label}: a test anchor naming no test closes on nothing")
            continue
        if kind != "test":
            try:
                re.compile(anchor["pattern"])
            except re.error as exc:
                # Reported rather than raised: a gate that crashes on a typo in
                # one item stops answering for the other eighteen.
                problems.append(f"{label}: anchor pattern does not compile ({exc})")
                continue
        holds = anchor_holds(anchor, root)
        if holds and item["closed"] is None:
            # Landing the fix is not the same as recording what it was measured
            # against, and the recording is the part this repository keeps
            # skipping. Same shape as a baseline entry that starts passing.
            problems.append(f"{label}: anchor now holds; record the run in `closed` and close it")
        elif not holds and item["closed"] is not None:
            problems.append(f"{label}: recorded closed but its anchor is gone ({anchor['file']})")
        elif holds:
            closed_ids.append(label)
        else:
            open_ids.append(label)
    # Every id the ledger still carries, whatever state it is in. `open_ids` and
    # `closed_ids` are not enough: an item pulled into `problems` — a malformed
    # entry, or one whose anchor holds with no `closed` record — is still present,
    # and counting it absent reported the same item twice, once falsely. The
    # history check asks "did this id disappear", not "is it in good order".
    known = {
        entry["id"]
        for entry in payload["items"]
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    problems += [
        f"{lost}: was committed to this ledger and is now absent"
        for lost in sorted(committed - known)
    ]
    return problems, open_ids, closed_ids


@check("verdicts-closed")
def reverification_verdicts_have_been_acted_on() -> Result:
    """The re-verification verdicts' "minimum action before merge" lists, tracked.

    Six axes went through implement -> adversarial verification -> fix -> re-verify.
    Round one: six implementers all reported break-evidence, six verifiers all
    overturned them. Round two returned six conditional verdicts, each with a list
    of things to do before merge — and the lists were not read, because the gate
    was green. Green was never the evidence; that is this file's opening paragraph.
    Whether anyone acted on a verdict was held in one person's memory, and this
    check moves it to where `axis-wired` and `config-consumed` already are.

    An item's state is *derived*, never declared. Each one names an `anchor` the
    repository either shows or does not, so prose cannot close anything: writing
    "fixed" in the summary changes nothing. Four states, three of them red:

      anchor false, `closed` empty  -> open, which is the ordinary red
      anchor false, `closed` filled -> the fix was landed and then lost
      anchor true,  `closed` empty  -> landed but the run behind it is unrecorded
      anchor true,  `closed` filled -> closed

    What it deliberately does not do is run the mutations. Most items close when a
    named mutation goes red — `MUT2=pad-last-only` passes 191 tests today and has
    to fail — and running those here would cost tens of seconds per gate and
    contend with the lanes for the same tree, which `doc-commands` already did
    once by shelling out to `uv sync`. So the split is: the gate asks whether the
    artifact that kills the mutation exists, and the item carries the command and
    today's observed outcome so a person, or a re-verification lane, runs it. A
    named test that exists but is vacuous satisfies the gate and not the item —
    which is why `closed` has to quote the run, and why closing is a reviewer's
    act rather than an author's.

    Two ways this could go hollow, and what stops each:

      Delete the item. Stopped by `_committed_verdict_ids`: git history names
      every id the ledger ever carried, and a missing one is reported.

      Repoint an anchor at a test that already passes. Not stopped. It is one
      line in a reviewed diff next to a `closed` record that has to quote a
      mutation run, and it is the same class of act as deleting the item — but
      the check cannot tell it from an honest rename.
    """
    path = REPO / OPEN_VERDICTS
    if not path.exists():
        return Result(
            "verdicts-closed",
            False,
            f"{OPEN_VERDICTS} is absent; the re-verification verdicts then have no "
            "record in this repository and every open item is invisible",
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return Result("verdicts-closed", False, f"{OPEN_VERDICTS} does not parse: {exc}")
    committed = _committed_verdict_ids(REPO)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        if empty := _nothing_to_check(payload["items"], "verdict items"):
            return Result("verdicts-closed", False, empty)
    problems, open_ids, closed_ids = verdict_ledger_problems(payload, REPO, committed)
    history = f"history pins {len(committed)} id(s)"
    if not problems and not open_ids:
        return Result(
            "verdicts-closed",
            True,
            f"all {len(closed_ids)} verdict item(s) closed with their anchors standing ({history})",
        )
    detail = []
    if open_ids:
        detail.append(f"{len(open_ids)} open: {', '.join(open_ids)}")
    detail += problems
    return Result(
        "verdicts-closed",
        False,
        "; ".join(detail) + f" ({len(closed_ids)} closed, {history})",
        count=len(open_ids) + len(problems),
    )


BASELINE = REPO / "docs" / "audit-baseline.json"


def load_baseline() -> dict[str, dict[str, Any]]:
    """Failures that are known, accepted, and scheduled.

    Without this the audit is unsatisfiable: its first run failed six of seven
    checks, and those six *are* the remaining work. Treating every failure as a
    blocker would mean no wave could ever close. So the audit tracks regressions
    instead — a new failure blocks, and a baseline entry that starts passing also
    blocks, because a stale baseline quietly grants amnesty to future breakage.

    Each entry carries `count` as well as `note`. Membership alone made the gate
    blind to everything a wave did inside a check that was already failing:
    deleting the whole of Wave 2's capture layer left `7/11 passing, 0 new
    failure(s), 0 newly fixed` unchanged to the byte. A string entry from an older
    baseline is read as a note with no recorded count, which disables only the
    size comparison for that entry.
    """
    if not BASELINE.exists():
        return {}
    raw = json.loads(BASELINE.read_text())
    return {
        name: {"note": entry, "count": None} if isinstance(entry, str) else entry
        for name, entry in raw.items()
    }


def merge_baseline(
    baseline: dict[str, dict[str, Any]], results: list[Result]
) -> dict[str, dict[str, Any]]:
    """The new baseline after a full run.

    Existing annotations survive. They are the schedule — each entry names the
    wave that resolves it — and overwriting them with a placeholder turns the
    baseline back into the excuse it was written not to be. Counts do not survive:
    they describe the run being recorded.
    """
    failing = {r.name: r for r in results if not r.ok}
    return {
        name: {
            "note": baseline.get(name, {}).get("note", "unscheduled"),
            "count": failing[name].count,
        }
        for name in sorted(failing)
    }


def classify(
    results: list[Result], baseline: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """New failures, newly passing, failures that changed size in either direction,
    and baseline entries whose recorded size cannot be compared to one.

    A shrunk count blocks for the same reason a newly passing check blocks: the
    baseline now grants amnesty for problems that no longer exist, and carrying
    that forward is how a stale entry stops meaning anything.

    The fourth answer is the one that used to be an exception. `count` is compared
    with `>`, `docs/audit-baseline.json` is hand-edited between waves, and a
    quoted number there — `"count": "3"` — raised TypeError out of this function
    and took the whole run with it: no check reported its result, including the
    twelve that had nothing to do with the malformed entry. Naming the entry is
    strictly more than the gate said before, and it still blocks.
    """
    regressions = [r.name for r in results if not r.ok and r.name not in baseline]
    fixed = [r.name for r in results if r.ok and r.name in baseline]
    grew, shrank, unreadable = [], [], []
    for r in results:
        if r.ok or r.count is None or r.name not in baseline:
            continue
        was = baseline[r.name].get("count")
        # `None` is an entry with no recorded count — the older string-only shape,
        # which `load_baseline` normalises to that and which disables only this
        # comparison. Anything else that is not a whole number is a malformed entry.
        if was is None:
            continue
        if not isinstance(was, int) or isinstance(was, bool):
            unreadable.append(f"{r.name} count={was!r}")
            continue
        if was == r.count:
            continue
        (grew if r.count > was else shrank).append(f"{r.name} {was}->{r.count}")
    return regressions, fixed, grew, shrank, unreadable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(CHECKS), default=None)
    parser.add_argument("--skip", nargs="*", choices=sorted(CHECKS), default=[])
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="record current failures as accepted; use only with a stated schedule",
    )
    args = parser.parse_args(argv)

    selected = [n for n in (args.only or sorted(CHECKS)) if n not in args.skip]
    if not selected:
        print("no checks selected")
        return 1
    partial = set(selected) != set(CHECKS)
    results = [CHECKS[name]() for name in selected]
    baseline = load_baseline()

    if args.update_baseline:
        if partial:
            # A partial run cannot know that the unselected checks still fail, and
            # writing the baseline from it deletes their entries — after which the
            # next full run has no record that they were ever accepted.
            print("refusing --update-baseline on a partial run: drop --only/--skip")
            return 1
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        merged = merge_baseline(baseline, results)
        BASELINE.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"baseline written with {len(merged)} accepted failure(s): {BASELINE}")
        return 0

    width = max(len(r.name) for r in results)
    regressions, fixed, grew, shrank, unreadable = classify(results, baseline)
    for r in results:
        if r.ok and r.name in baseline:
            status = "FIXED"
        elif r.ok:
            status = "PASS"
        elif r.name in baseline:
            status = "KNOWN"
        else:
            status = "NEW"
        suffix = f"  [{baseline[r.name]['note']}]" if not r.ok and r.name in baseline else ""
        print(f"{status:<5} {r.name:<{width}}  {r.detail}{suffix}")

    print(
        f"\n{sum(1 for r in results if r.ok)}/{len(results)} passing, "
        f"{len(regressions)} new failure(s), {len(fixed)} newly fixed, "
        f"{len(grew)} grew, {len(shrank)} shrank, {len(unreadable)} unreadable"
    )
    if partial:
        print(f"PARTIAL RUN: {len(CHECKS) - len(selected)} check(s) not run; not a wave gate")
    if regressions:
        print(f"BLOCKED: new failures not in baseline: {', '.join(regressions)}")
    if grew:
        print(f"BLOCKED: accepted failures got worse: {', '.join(grew)}")
    if fixed:
        print(f"BLOCKED: baseline is stale, these now pass: {', '.join(fixed)}")
        print("  run --update-baseline after confirming, so amnesty is not carried forward")
    if shrank:
        print(f"BLOCKED: baseline is stale, these shrank: {', '.join(shrank)}")
        print("  run --update-baseline after confirming, so amnesty is not carried forward")
    if unreadable:
        print(f"BLOCKED: baseline entries with an uncomparable count: {', '.join(unreadable)}")
        print("  a count is the whole number of problems the entry was accepted at")
    return 1 if (regressions or fixed or grew or shrank or unreadable) else 0


if __name__ == "__main__":
    sys.exit(main())
