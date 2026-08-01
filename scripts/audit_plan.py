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
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
# Where config values are allowed to be consumed from.
CODE_ROOTS = (REPO / "trainbench", REPO / "scripts")
# Reporting a value is not applying it. trainbench/applied.py reads every axis in
# order to say what was requested, so counting it as a consumer would certify an
# axis as wired on the strength of the code that only labels it.
#
# This file is excluded for a blunter reason: it lives under `scripts/`, so it
# scans itself, and its own docstrings spell out the patterns it searches for.
# `attn.name` passed for exactly that reason and nothing else — the only wired
# axis was certified by the checker's prose about how it certifies axes.
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


def _code_text(exclude: tuple[Path, ...] = ()) -> str:
    return "\n".join(p.read_text() for p in _code_files(exclude))


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

    Matching the bare identifier finds `name` in eighteen unrelated files. The
    value is only reached through the config object, so the access has to look
    like one: `config.attn.name`, `cfg.attn.name`, or `["attn"]["name"]`.
    """
    attribute = rf"\.{re.escape(group)}\s*\.\s*{re.escape(key)}\b"
    subscript = rf"\[\s*[\"']{re.escape(group)}[\"']\s*\]\s*\[\s*[\"']{re.escape(key)}[\"']\s*\]"
    # Anchored on the config object. An unanchored first argument matched any
    # variable whose name merely contained the group: `getattr(model, "name")`
    # counted as reading `model.name`, and `getattr(runner, "purpose")` as
    # `run.purpose`. `data`, `run`, `train` and `model` are common name fragments.
    getattr_call = (
        rf"getattr\(\s*(?:config|cfg)\.{re.escape(group)}\s*,\s*[\"']{re.escape(key)}[\"']"
    )
    return any(re.search(p, code) for p in (attribute, subscript, getattr_call))


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
    code = _code_text(exclude=NOT_A_CONSUMER)

    def consumed(group: str, key: str) -> bool:
        if _reads_dotted(code, group, key):
            return True
        derived = DERIVED_LEAVES.get(f"{group}.{key}")
        return bool(derived) and _reads_dotted(code, *derived.split("."))

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


@check("assert-called")
def the_measurement_entry_point_calls_the_axis_machinery() -> Result:
    """The harness that reports numbers must go through axes.py and applied.py.

    "Some caller exists somewhere" is not the property that matters. The probe
    calls `assert_matches` too, and `purpose=probe` returns immediately from every
    check — so an entry point written without the call would leave this green
    while measuring unverified settings. The named entry point is what has to call
    it, and the same is true of every hook an axis is applied through.
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
    return Result(
        "assert-called",
        not missing,
        f"{entry} calls {', '.join(ENTRY_POINT_CALLS)}"
        if not missing
        else f"{entry} never calls {', '.join(missing)}; those axes would be applied or "
        "verified nowhere while the run still reports numbers",
    )


TREE_PREFIX = " │├└─"
TREE_INDENT = 4


def missing_plan_files(block: str, root: Path) -> list[str]:
    """Files declared in a layout tree that do not exist where it puts them.

    The tree is read as a tree: an entry's path is its name under the directories
    that contain it. Comparing basenames instead — as this did first — matches
    `config.py` against 73 files here, most of them inside `.venv`, so the check
    stays green after the declared file is deleted and red for files that exist
    under a directory it never looked at.
    """
    parents: dict[int, str] = {}
    missing = []
    for line in block.splitlines():
        entry = line.lstrip(TREE_PREFIX)
        if not entry.strip():
            continue
        depth = (len(line) - len(entry)) // TREE_INDENT
        name = entry.split("#")[0].split()[0]
        if name.endswith("/"):
            # A directory: everything below this depth belongs to something else.
            parents = {d: p for d, p in parents.items() if d < depth}
            parents[depth] = name.rstrip("/")
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.(?:py|md|toml|yaml|lock)", name):
            continue
        # parents[0] is the repository root itself, so it contributes no segment.
        declared = "/".join([parents[d] for d in sorted(parents) if d > 0] + [name])
        if not (root / declared).exists():
            missing.append(declared)
    return missing


@check("plan-files")
def files_named_in_plan_exist() -> Result:
    """PLAN.md's repository layout must describe reality, not intent."""
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
    missing = missing_plan_files(block.group(1), REPO)
    return Result(
        "plan-files",
        not missing,
        "every file named in PLAN.md exists"
        if not missing
        else f"{len(missing)} named but absent: {', '.join(sorted(set(missing)))}",
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
    "kernel/kernels_hub": ("kernels",),
    "precision/mxfp8": ("transformer-engine",),
    "precision/nvfp4": ("transformer-engine",),
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
    # `grad-cache` is a 404. Upstream declares `name="GradCache"`, which PEP 503
    # normalises to `gradcache`.
    "loss/cached_mnrl": ("gradcache",),
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
        "parallel/ddp",
        "parallel/fsdp2",
        "dataloader/torch",
        "dataloader/torch_packed",
        "peft/full",
        "peft/lora",
        "loss/mnrl",
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


@check("doc-commands")
def documented_commands_are_runnable() -> Result:
    """README/AGENTS commands must work as written.

    The documented bootstrap once installed no extras, so `uv run pytest` failed
    on a missing hydra while local development used a different flag set. And the
    documented smoke command was rejected by this project's own batch validator
    for as long as that validator existed, because nothing ever ran it — so the
    entry point commands are executed here rather than pattern-matched.
    """
    problems = []
    found = 0
    for name in ("README.md", "AGENTS.md"):
        text = (REPO / name).read_text()
        found += len(re.findall(r"uv sync[^\n`]*", text)) + len(
            re.findall(r"python scripts/env_report\.py[^\n`]*", text)
        )
        for match in re.finditer(r"uv sync([^\n`]*)", text):
            flags = match.group(1)
            if "--extra compose" not in flags and "--all-extras" not in flags:
                problems.append(
                    f"{name}: `uv sync{flags}` omits the compose extra, but tests import hydra"
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
    return Result(
        "doc-commands",
        not problems,
        f"{found} documented command(s) install what the tests need and run as written"
        if not problems
        else "; ".join(problems),
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
}


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

    Each variant is composed and pushed through all four call sites. `UnappliedAxis`
    is the axis refusing the value, which is the honest outcome and what gets
    counted. Any other exception is reported as its own problem rather than folded
    into the count — a value that fails for an unrelated reason has not been
    measured either way.
    """
    sys.path.insert(0, str(REPO))
    import torch
    from hydra import compose, initialize_config_dir

    from trainbench import axes
    from trainbench.compose import resolve

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(4, 4)

    variants: dict[str, list[str]] = {}
    for path in sorted(CONFIGS.rglob("*.yaml")):
        group = path.parent.name
        if path.parent == CONFIGS or group in NON_AXIS_GROUPS:
            continue
        variants.setdefault(group, []).append(path.stem)
    if empty := _nothing_to_check(variants, "axis config groups"):
        return Result("axis-values", False, empty)

    applicable: dict[str, int] = {}
    inert: list[str] = []
    broken: list[str] = []
    for group, names in sorted(variants.items()):
        applicable[group] = 0
        for name in sorted(names):
            variant = f"{group}/{name}"
            overrides = [f"{group}={name}", *AXIS_VALUE_COMPANIONS.get(variant, ())]
            try:
                with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
                    config = resolve(compose(config_name="config", overrides=overrides))[0]
                axes.patch(config)
                axes.load_kwargs(config)
                axes.assemble(_Tiny(), config, torch.device("cpu"), "native")
                with axes.step_context(config):
                    pass
            except axes.UnappliedAxis:
                continue
            except Exception as exc:  # noqa: BLE001 - reported, never counted as applied
                broken.append(f"{variant} ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            applicable[group] += 1

    total = sum(len(v) for v in variants.values())
    usable = sum(applicable.values())
    for group, count in sorted(applicable.items()):
        if count <= 1 and len(variants[group]) > 1:
            inert.append(f"{group} {count}/{len(variants[group])}")
    problems = []
    if inert:
        problems.append(f"{len(inert)} group(s) offering one usable value: {', '.join(inert)}")
    if broken:
        problems.append(f"{len(broken)} value(s) failed for another reason: {'; '.join(broken)}")
    return Result(
        "axis-values",
        not problems,
        f"{usable}/{total} offered axis values can be applied"
        if not problems
        else f"{usable}/{total} applicable; " + "; ".join(problems),
        count=len(inert) + len(broken),
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
) -> tuple[list[str], list[str], list[str], list[str]]:
    """New failures, newly passing, and failures that changed size in either direction.

    A shrunk count blocks for the same reason a newly passing check blocks: the
    baseline now grants amnesty for problems that no longer exist, and carrying
    that forward is how a stale entry stops meaning anything.
    """
    regressions = [r.name for r in results if not r.ok and r.name not in baseline]
    fixed = [r.name for r in results if r.ok and r.name in baseline]
    grew, shrank = [], []
    for r in results:
        if r.ok or r.count is None or r.name not in baseline:
            continue
        was = baseline[r.name].get("count")
        if was is None or was == r.count:
            continue
        (grew if r.count > was else shrank).append(f"{r.name} {was}->{r.count}")
    return regressions, fixed, grew, shrank


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
    regressions, fixed, grew, shrank = classify(results, baseline)
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
        f"{len(grew)} grew, {len(shrank)} shrank"
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
    return 1 if (regressions or fixed or grew or shrank) else 0


if __name__ == "__main__":
    sys.exit(main())
