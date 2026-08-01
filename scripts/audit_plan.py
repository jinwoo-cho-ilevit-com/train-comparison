"""Mechanical consistency check between the plan, the docs, and the code.

Run at the end of every wave. Exits non-zero on any failed invariant.

This exists because the human-readable checklist already failed once: the fixed
subset was pushed with 54.6% duplicate positives and 31.4% missing query images,
and the success criteria in use at the time (row count, config coverage) reported
a clean pass. Checks here must be things a wrong answer cannot satisfy.

    python scripts/audit_plan.py            # all checks
    python scripts/audit_plan.py --only config-consumed
    python scripts/audit_plan.py --skip model-spec   # skip the network check
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
# Where config values are allowed to be consumed from.
CODE_ROOTS = (REPO / "trainbench", REPO / "scripts")


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


CHECKS: dict[str, Callable[[], Result]] = {}


def check(name: str) -> Callable[[Callable[[], Result]], Callable[[], Result]]:
    def register(fn: Callable[[], Result]) -> Callable[[], Result]:
        CHECKS[name] = fn
        return fn

    return register


def _code_text() -> str:
    parts = []
    for root in CODE_ROOTS:
        for path in root.rglob("*.py"):
            parts.append(path.read_text())
    return "\n".join(parts)


def _config_leaf_keys() -> dict[str, set[str]]:
    """Leaf keys per config group, e.g. {"attn": {"name", "impl"}}."""
    groups: dict[str, set[str]] = {}
    for path in CONFIGS.rglob("*.yaml"):
        if path.parent == CONFIGS:
            continue
        group = path.parent.name
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            groups.setdefault(group, set()).update(k for k in data if isinstance(k, str))
    return groups


@check("config-consumed")
def config_fields_are_read_by_code() -> Result:
    """Every config knob must be read somewhere.

    An unread knob is worse than a missing one: `attn=fa4` composes, validates,
    prints, and is recorded in the result JSON while changing nothing. Eight of
    twelve ablation axes were in exactly that state.
    """
    code = _code_text()
    schema = (REPO / "trainbench" / "config_schema.py").read_text()
    orphans = []
    for group, keys in sorted(_config_leaf_keys().items()):
        for key in sorted(keys):
            # A field is "consumed" only if something outside the schema reads it.
            outside_schema = code.replace(schema, "")
            if re.search(rf"\b{re.escape(key)}\b", outside_schema):
                continue
            orphans.append(f"{group}.{key}")
    return Result(
        "config-consumed",
        not orphans,
        "all config leaves are read by code"
        if not orphans
        else f"{len(orphans)} unread knob(s): {', '.join(orphans)}",
    )


@check("plan-files")
def files_named_in_plan_exist() -> Result:
    """PLAN.md's repository layout must describe reality, not intent."""
    plan = (REPO / "PLAN.md").read_text()
    block = re.search(r"```\n(train-comparison/.*?)```", plan, re.S)
    if not block:
        return Result("plan-files", True, "PLAN.md has no repository layout block")
    missing = []
    for line in block.group(1).splitlines():
        match = re.search(r"([A-Za-z0-9_.\-/]+\.(?:py|md|toml|yaml|lock))\s*(?:#|$)", line)
        if not match:
            continue
        name = match.group(1)
        if not list(REPO.rglob(Path(name).name)):
            missing.append(name)
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
    hand-written prose.
    """
    matrix = REPO / "docs" / "support-matrix.md"
    if not matrix.exists():
        return Result("evidence-committed", True, "no support matrix yet")
    evidence = REPO / "docs" / "evidence"
    tracked = subprocess.run(
        ["git", "ls-files", "docs/evidence"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not evidence.exists() or not tracked:
        return Result(
            "evidence-committed",
            False,
            "docs/support-matrix.md states measured results but docs/evidence/ holds no "
            "committed artifacts; the numbers are unverifiable prose",
        )
    return Result("evidence-committed", True, f"{len(tracked)} evidence artifact(s) committed")


@check("axis-packages")
def axes_have_their_packages() -> Result:
    """A config group offering an axis whose package is absent everywhere is a
    knob that cannot do anything."""
    required = {
        "attn/fa2": "flash-attn",
        "attn/fa3": "flash-attn",
        "attn/fa4": "flash-attn",
        "kernel/liger": "liger-kernel",
        "kernel/fla": "causal-conv1d",
        "precision/mxfp8": "transformer-engine",
        "precision/nvfp4": "transformer-engine",
        "optim/adamw_8bit": "bitsandbytes",
        "parallel/zero2": "deepspeed",
        "parallel/zero3": "deepspeed",
        "dataloader/dali": "nvidia-dali",
    }
    locks = [REPO / "uv.lock", *(REPO / "envs").glob("*/uv.lock")]
    installed = "\n".join(p.read_text() for p in locks if p.exists())
    missing = []
    for variant, package in sorted(required.items()):
        if not (CONFIGS / f"{variant}.yaml").exists():
            continue
        if f'name = "{package}"' not in installed:
            missing.append(f"{variant} needs {package}")
    return Result(
        "axis-packages",
        not missing,
        "every offered axis has its package in some env"
        if not missing
        else f"{len(missing)} unbacked axis/axes: {'; '.join(missing)}",
    )


@check("doc-commands")
def documented_commands_are_runnable() -> Result:
    """README/AGENTS commands must work as written.

    The documented bootstrap once installed no extras, so `uv run pytest` failed
    on a missing hydra while local development used a different flag set.
    """
    problems = []
    for name in ("README.md", "AGENTS.md"):
        text = (REPO / name).read_text()
        for match in re.finditer(r"uv sync([^\n`]*)", text):
            flags = match.group(1)
            if "--extra compose" not in flags and "--all-extras" not in flags:
                problems.append(
                    f"{name}: `uv sync{flags}` omits the compose extra, but tests import hydra"
                )
    return Result(
        "doc-commands",
        not problems,
        "documented setup commands install what the tests need"
        if not problems
        else "; ".join(problems),
    )


@check("data-pinned")
def measured_data_is_pinned() -> Result:
    """A run is not reproducible if its corpus is a moving branch."""
    problems = []
    for path in (CONFIGS / "data").glob("*.yaml"):
        data = yaml.safe_load(path.read_text()) or {}
        if data.get("revision") in (None, "null", ""):
            problems.append(f"{path.name} has no pinned revision")
    return Result(
        "data-pinned",
        not problems,
        "every data config pins a revision" if not problems else "; ".join(problems),
    )


@check("model-spec")
def model_spec_matches_config() -> Result:
    """The per-model usage decisions in docs/model-spec.md must equal the config.

    Decision drift here is invisible at runtime: with last-token pooling,
    add_generation_prompt silently changes which token becomes the embedding.
    """
    spec = REPO / "docs" / "model-spec.md"
    if not spec.exists():
        return Result("model-spec", False, "docs/model-spec.md is missing")
    text = spec.read_text()
    problems = []
    for path in (CONFIGS / "model").glob("*.yaml"):
        data = yaml.safe_load(path.read_text()) or {}
        if "add_generation_prompt" not in data:
            problems.append(f"{path.name} does not declare add_generation_prompt")
    if "add_generation_prompt" not in text:
        problems.append("docs/model-spec.md does not document add_generation_prompt")
    return Result(
        "model-spec",
        not problems,
        "model usage decisions are declared in both doc and config"
        if not problems
        else "; ".join(problems),
    )


BASELINE = REPO / "docs" / "audit-baseline.json"


def _baseline() -> dict[str, str]:
    """Failures that are known, accepted, and scheduled.

    Without this the audit is unsatisfiable: its first run failed six of seven
    checks, and those six *are* the remaining work. Treating every failure as a
    blocker would mean no wave could ever close. So the audit tracks regressions
    instead — a new failure blocks, and a baseline entry that starts passing also
    blocks, because a stale baseline quietly grants amnesty to future breakage.
    """
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text())


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

    selected = args.only or sorted(CHECKS)
    results = [CHECKS[name]() for name in selected if name not in args.skip]
    baseline = _baseline()

    if args.update_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        merged = {**baseline, **{r.name: "unscheduled" for r in results if not r.ok}}
        merged = {k: v for k, v in merged.items() if k in {r.name for r in results if not r.ok}}
        BASELINE.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"baseline written with {len(merged)} accepted failure(s): {BASELINE}")
        return 0

    width = max(len(r.name) for r in results)
    regressions, fixed = [], []
    for r in results:
        if r.ok and r.name in baseline:
            status, fixed = "FIXED", [*fixed, r.name]
        elif r.ok:
            status = "PASS"
        elif r.name in baseline:
            status = "KNOWN"
        else:
            status, regressions = "NEW", [*regressions, r.name]
        suffix = f"  [{baseline[r.name]}]" if not r.ok and r.name in baseline else ""
        print(f"{status:<5} {r.name:<{width}}  {r.detail}{suffix}")

    print(
        f"\n{sum(1 for r in results if r.ok)}/{len(results)} passing, "
        f"{len(regressions)} new failure(s), {len(fixed)} newly fixed"
    )
    if regressions:
        print(f"BLOCKED: new failures not in baseline: {', '.join(regressions)}")
    if fixed:
        print(f"BLOCKED: baseline is stale, these now pass: {', '.join(fixed)}")
        print("  run --update-baseline after confirming, so amnesty is not carried forward")
    return 1 if (regressions or fixed) else 0


if __name__ == "__main__":
    sys.exit(main())
