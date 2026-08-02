"""Launch one pod per experiment manifest and keep a ledger of what happened.

Composition happens here, not on the pod: each run's resolved config is passed in
as an environment variable, so the config that ran is exactly the config recorded,
and no image needs Hydra.

    python scripts/orchestrate.py --experiment 'phase0-*' --dry-run

What each pod does is declared in `configs/experiment/`, one file per pod. The
sweep is not a product of two command-line lists any more: a combination that ran
has a file naming it, which is what makes the run repeatable (convention 02 §3).

Start with one manifest. Verifying the image, secret injection and result upload
on a single pod costs one pod-hour; discovering a broken entrypoint across
eighteen costs eighteen.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from rich.console import Console

from trainbench import pods
from trainbench.compose import resolve
from trainbench.config import git_state
from trainbench.config_schema import axis_knobs
from trainbench.record import write_json

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
MANIFEST_DIR = CONFIG_DIR / "experiment"
BASELINES_PATH = MANIFEST_DIR / "_baselines.yaml"

# Framework images are named after their env directory, which uses hyphens.
IMAGE_SUFFIX = {
    "native": "native",
    "unsloth": "unsloth",
    "ms_swift": "ms-swift",
    "sentence_transformers": "sentence-transformers",
    "tevatron": "tevatron",
    "axolotl": "axolotl",
}

# Manifest keys. Unknown keys are refused rather than ignored: a typo in an
# experiment definition must not silently drop the thing it was meant to set.
REQUIRED_KEYS = frozenset({"phase", "model", "framework", "run", "baseline"})
OPTIONAL_KEYS = frozenset({"description", "gpu_type_id", "axis", "settings", "overrides"})

# A pod with no canonical baseline is allowed only where there is no number to
# compare across hosts. Any measuring purpose must name one (PLAN.md, 다중 pod 분할
# 규칙 3): without a shared reference run, results from different hosts cannot go
# in the same table.
NO_BASELINE = "none"
PURPOSES_WITHOUT_BASELINE = frozenset({"probe"})

# What the pod images can execute today. `scripts/bench.py` (Wave 3) is what makes
# a timing purpose runnable; until then, launching one buys an hour of GPU to
# learn that the entry point is missing.
# Widened once `scripts/bench.py` existed and `docker/entrypoint.sh` grew an arm
# that calls it. Ownership of both moved from lane C to lane G at Wave 3 start
# (docs/CONTRACTS.md §1).
RUNNABLE_PURPOSES = frozenset({"probe", "timing", "profile", "quality"})

# Every secret an experiment pod has any use for. `.env.example` names exactly one
# — "Experiment pods: model/dataset pull, result push, Trackio Space sync" — and
# everything else the pod needs (TRAINBENCH_*, INFISICAL_*) is handed over in the
# pod's env dict rather than read out of Infisical.
#
# An allowlist rather than a longer FORBIDDEN_ON_POD, and the reason is a
# measurement, not a preference. The project's `dev` environment injects 27 names
# (measured 2026-08-02 by diffing `os.environ` under `infisical run --env=dev`
# against a plain shell). Four of them are the ones below. One is HF_TOKEN. The
# other **22 pass a deny check and reach the pod with no use for them** — cloud,
# database and model-provider credentials that nobody thought to enumerate here.
# A deny list can only ever hold the names someone remembered, and the thing it
# guards grows without asking it. This one is what the pod is for.
ALLOWED_ON_POD = frozenset({"HF_TOKEN"})

# The Infisical environment an experiment pod reads. `--infisical-env` names what
# the *pod* gets, never what the orchestrator itself reads — the orchestrator's
# secrets come from the `infisical run --env=dev` that wraps this script. So the
# pod's own environment is the only correct default, and `dev` was a leftover from
# before that environment existed.
#
# Either default fails closed, because the scope check refuses whatever it cannot
# vouch for. The difference is which mistake is quiet: with `dev` as the default,
# forgetting the flag stops the campaign; with this one, forgetting it is right.
POD_INFISICAL_ENV = "pod"

# Account-wide credentials, called out by name when the scope check refuses. These
# are no longer what the check *tests* — `ALLOWED_ON_POD` is — but naming them
# separately tells an operator which of the extras are the urgent ones.
# RUNPOD_API_KEY would let a probe delete the sweep that created it; GITHUB_TOKEN
# carries write:packages and belongs to the image build alone; the universal-auth
# pair makes a short-lived token pointless, since an identity that can mint its
# own tokens is not bounded by the one it was handed.
FORBIDDEN_ON_POD = (
    "RUNPOD_API_KEY",
    "GITHUB_TOKEN",
    "INFISICAL_UNIVERSAL_AUTH_CLIENT_ID",
    "INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET",
)

# The pod kills itself this long before the orchestrator's own deadline, so a hung
# run stops billing on its own even if the orchestrator dies first. It is a ceiling,
# not a fixed subtraction: see pod_timeout_seconds.
SELF_KILL_MARGIN_SECONDS = 120

# Why a run is in a pod's plan. The baseline is a fixed reference workload that
# every measuring pod repeats so host-to-host variance is measured rather than
# assumed; it is not what the pod was created to answer.
ROLE_BASELINE = "baseline"
ROLE_EXPERIMENT = "experiment"

# The one config group whose values cannot share a pod: each framework is its own
# image, and an image is what a pod runs. `check_axis_not_split` therefore allows
# it across pods — and says so, because an axis compared across hosts in silence is
# the thing that guard exists to prevent. Phase 0 split it eighteen ways per model
# and nothing said a word.
CROSS_POD_GROUP = "framework"


class ManifestError(ValueError):
    """An experiment definition that cannot be trusted to describe a pod."""


@dataclass(frozen=True)
class Run:
    """One resolved setting a pod executes."""

    name: str
    role: str
    overrides: tuple[str, ...]
    config: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Everything the pod needs to execute this run, resolved.

        The config travels, not just the Hydra override strings. No pod image has
        Hydra — the antlr4 pin is incompatible with axolotl, which is why
        composition happens here — so a pod handed only overrides cannot resolve
        the settings it owns, and a sweep pod owns more than one.
        """
        return {
            "name": self.name,
            "role": self.role,
            "overrides": list(self.overrides),
            "config": self.config,
        }


@dataclass(frozen=True)
class Experiment:
    """One manifest: everything one pod is asked to do."""

    name: str
    phase: str
    model: str
    framework: str
    purpose: str
    baseline: str
    axis: str | None = None
    description: str = ""
    gpu_type_id: str | None = None
    settings: dict[str, list[str]] = field(default_factory=dict)
    overrides: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "experiment": self.name,
            "phase": self.phase,
            "model": self.model,
            "framework": self.framework,
            "purpose": self.purpose,
            "axis": self.axis,
            "baseline": self.baseline,
        }


def load_manifest(path: Path) -> Experiment:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ManifestError(f"{path.name}: expected a mapping")
    keys = set(raw)
    if missing := REQUIRED_KEYS - keys:
        raise ManifestError(f"{path.name}: missing {', '.join(sorted(missing))}")
    if unknown := keys - REQUIRED_KEYS - OPTIONAL_KEYS:
        raise ManifestError(f"{path.name}: unknown key(s) {', '.join(sorted(unknown))}")
    settings = raw.get("settings") or {}
    if settings and not raw.get("axis"):
        raise ManifestError(
            f"{path.name}: settings without an axis. The axis name is what proves no "
            "other pod is running part of the same comparison"
        )
    purpose = str(raw["run"])
    baseline = str(raw["baseline"])
    if baseline == NO_BASELINE and purpose not in PURPOSES_WITHOUT_BASELINE:
        raise ManifestError(
            f"{path.name}: purpose '{purpose}' produces numbers, so it must name a "
            "canonical baseline; results from different hosts are otherwise not comparable"
        )
    # The converse, and it is not symmetry for its own sake. The baseline names its
    # own model and framework, so adding one to a probe pod puts a run of a
    # different combination on a pod built to answer a question about this one —
    # in this pod's image, filed under this pod's directory.
    if baseline != NO_BASELINE and purpose in PURPOSES_WITHOUT_BASELINE:
        raise ManifestError(
            f"{path.name}: purpose '{purpose}' produces no number to compare across "
            f"hosts, so it must declare 'baseline: {NO_BASELINE}'; naming "
            f"'{baseline}' adds a run of another model and framework to this pod"
        )
    exp = Experiment(
        name=path.stem,
        phase=str(raw["phase"]),
        model=str(raw["model"]),
        framework=str(raw["framework"]),
        purpose=purpose,
        baseline=baseline,
        axis=raw.get("axis"),
        description=raw.get("description", ""),
        gpu_type_id=raw.get("gpu_type_id"),
        settings={str(k): list(v) for k, v in settings.items()},
        overrides=list(raw.get("overrides") or []),
    )
    # The label has to match what runs. It is what the split guard reports, and a
    # pod that sweeps `attn` under the name `loss` files its own violation under a
    # heading nobody is looking at.
    moved = axes_touched(exp)
    if moved and exp.axis and exp.axis not in moved:
        raise ManifestError(
            f"{path.name}: declares axis '{exp.axis}' but its overrides move "
            f"{', '.join(sorted(moved))}"
        )
    return exp


def axis_moved_by(override: str) -> str | None:
    """The ablation axis a Hydra override moves, if it moves one.

    `loss=cached_mnrl` selects a config group and every axis knob in that group
    moves with it; `train.gradient_checkpointing=selective` names a knob outright.
    Both are how a manifest changes what is measured, so both have to be visible
    to the guard that keeps one axis on one host.

    Derived from the schema's `Axis()` markers rather than a list kept here: a
    hand-written list stops covering an axis the moment someone adds one, and this
    feeds a check whose whole job is to not have gaps.

    It reads override *keys*, which makes it deliberately conservative in one
    direction — selecting a group counts as moving every axis in it, even for a
    file that happens to leave them alone — and blind in another: editing what a
    config group's file contains moves an axis without any manifest changing.
    """
    key, sep, _ = override.partition("=")
    if not sep:
        return None
    key = key.lstrip("+~")
    knobs = axis_knobs()
    if "." in key:
        return key if key in knobs else None
    return key if any(knob.startswith(f"{key}.") for knob in knobs) else None


def pod_overrides(exp: Experiment) -> list[str]:
    """The overrides `plan_runs` puts in front of every run this pod executes.

    Shared with `axes_touched` rather than spelled out twice: `framework` is an
    ablation axis and it arrives here, not through `overrides`, so a guard reading
    only the manifest's own lists could not see the axis the pod is defined by.
    """
    return [f"framework={exp.framework}", f"model={exp.model}", f"run={exp.purpose}"]


def axes_touched(exp: Experiment) -> dict[str, list[str]]:
    """Every axis this manifest moves, and the override that shows it.

    Three sources, because a pod's settings arrive by three routes: the identity
    overrides `plan_runs` prepends, the overrides fixed for the whole pod, and the
    ones swept per setting. A value pinned on one pod and a different value pinned
    on another is the same cross-host comparison as a sweep torn in half, and
    reads as neither.
    """
    found: dict[str, list[str]] = {}
    swept = [override for extra in exp.settings.values() for override in extra]
    for override in [*pod_overrides(exp), *exp.overrides, *swept]:
        if axis := axis_moved_by(override):
            found.setdefault(axis, []).append(override)
    return found


def is_cross_pod_group(axis: str) -> bool:
    """Whether this axis is compared across pods by construction rather than by mistake."""
    return axis == CROSS_POD_GROUP or axis.startswith(f"{CROSS_POD_GROUP}.")


def load_baselines(path: Path) -> dict[str, list[str]]:
    """The canonical baseline runs, defined once for the whole campaign.

    Defined in one file on purpose. A baseline copied into every manifest drifts,
    and a baseline that differs per pod measures nothing about the pods.
    """
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {str(name): list(body["overrides"]) for name, body in raw.items()}


def load_experiments(directory: Path = MANIFEST_DIR) -> list[Experiment]:
    paths = sorted(p for p in directory.glob("*.yaml") if not p.name.startswith("_"))
    experiments = [load_manifest(p) for p in paths]
    check_axis_not_split(experiments)
    check_one_baseline_one_gpu(experiments)
    return experiments


def split_axes(experiments: list[Experiment]) -> dict[tuple[str, str], dict[str, list[str]]]:
    """Every (model, axis) that more than one pod moves, with the evidence per pod."""
    seen: dict[tuple[str, str], dict[str, list[str]]] = {}
    for exp in experiments:
        touched = axes_touched(exp)
        if exp.axis:
            touched.setdefault(exp.axis, ["declared"])
        for axis, evidence in touched.items():
            seen.setdefault((exp.model, axis), {})[exp.name] = evidence
    return {key: owners for key, owners in seen.items() if len(owners) > 1}


def _split_detail(split: dict[tuple[str, str], dict[str, list[str]]]) -> str:
    return "; ".join(
        f"{model} x {axis} split across "
        + ", ".join(f"{name} ({', '.join(why)})" for name, why in sorted(owners.items()))
        for (model, axis), owners in sorted(split.items())
    )


def cross_pod_notes(experiments: list[Experiment]) -> list[str]:
    """The axes `check_axis_not_split` allows across pods, named one by one.

    Only `framework` is here, and only because each of its values is a different
    image. Saying it is the whole point: Phase 0 compared six frameworks per model
    across eighteen hosts and the guard that exists to catch a cross-host
    comparison was structurally unable to see the one the campaign was made of.
    """
    return [
        f"{model} x {axis} is compared across {len(owners)} pod(s) by construction "
        f"({', '.join(sorted(owners))}); each value is its own image, so the "
        "canonical baseline is the only thing making these hosts comparable"
        for (model, axis), owners in sorted(split_axes(experiments).items())
        if is_cross_pod_group(axis)
    ]


def check_axis_not_split(experiments: list[Experiment]) -> None:
    """No axis may be compared across two pods for the same model.

    Pods are different physical hosts. Comparing FA2 on one host against FA3 on
    another measures the hosts as much as the kernels, so PLAN.md forbids it. The
    rule is enforceable here because a manifest is exactly one pod.

    What counts as touching an axis is read off the overrides, not off the
    manifest's `axis` field. Keying on the declaration alone left a way through
    that needed no ill intent: `axis` is required only when a manifest has
    `settings`, so two pods each pinning one value of the same axis in
    `overrides` — one `loss=mnrl`, one `loss=cached_mnrl` — declared nothing,
    collided with nothing, and ran half a comparison each on a different host.

    `framework` is the declared exception rather than a hole: its values are six
    images and an image cannot be shared. It is allowed through here and reported
    by `cross_pod_notes`, which is the half that was missing — the guard used to
    be blind to it because `framework` reaches a run through `pod_overrides`.
    """
    refused = {
        key: owners
        for key, owners in split_axes(experiments).items()
        if not is_cross_pod_group(key[1])
    }
    if refused:
        raise ManifestError(
            f"an axis is split across pods, which invalidates it: {_split_detail(refused)}"
        )


def check_one_baseline_one_gpu(experiments: list[Experiment]) -> None:
    """Pods whose numbers get compared must ask for the same GPU.

    The canonical baseline is the instrument that makes two pods' numbers
    comparable: every measuring pod runs it and a deviation over 3% throws the pod
    out. Run that cohort on two different accelerators and the deviation is a fact
    about the accelerator, after which the gate discards whichever pod was right.
    PLAN.md bans mixing GPU types outright ("GPU 혼용은 어떤 경우에도 금지") and
    that sentence was the entire enforcement — `gpu_type_id` is a free string that
    nothing read back, so eighteen A100 manifests and three B200 manifests sat in
    one directory with nothing able to tell a campaign from a mistake.

    A measuring pod must name its own GPU rather than inherit `--gpu-type-id`:
    that default is one value for the whole invocation, so an omission puts the
    pod on whatever the last caller typed while its cohort runs on what they
    declared, and the ledger records the difference as if it had been intended.
    """
    measuring = [e for e in experiments if e.baseline != NO_BASELINE]
    if undeclared := sorted(e.name for e in measuring if not e.gpu_type_id):
        raise ManifestError(
            "a measuring pod must declare gpu_type_id, or its comparison spans "
            f"whatever GPU each invocation defaulted to: {', '.join(undeclared)}"
        )
    cohorts: dict[str, dict[str, list[str]]] = {}
    for exp in measuring:
        cohorts.setdefault(exp.baseline, {}).setdefault(str(exp.gpu_type_id), []).append(exp.name)
    mixed = {name: gpus for name, gpus in cohorts.items() if len(gpus) > 1}
    if mixed:
        detail = "; ".join(
            f"baseline '{name}' spans "
            + ", ".join(
                f"{gpu} ({', '.join(sorted(names))})" for gpu, names in sorted(gpus.items())
            )
            for name, gpus in sorted(mixed.items())
        )
        raise ManifestError(f"one baseline compared across two GPU types: {detail}")


def select(experiments: list[Experiment], patterns: list[str]) -> list[Experiment]:
    if not patterns:
        return experiments
    chosen = [e for e in experiments if any(fnmatch.fnmatch(e.name, p) for p in patterns)]
    if not chosen:
        raise ManifestError(f"no experiment matches {', '.join(patterns)}")
    return chosen


def resolved_config(overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
        # resolve() validates: a bad variant fails on the laptop, not after the
        # image has been pulled onto a billing GPU.
        return resolve(cfg)[1]


def plan_runs(exp: Experiment, baselines: dict[str, list[str]]) -> list[Run]:
    """Every setting this pod executes, baseline first.

    The baseline is composed from its own overrides alone, not from the pod's, so
    that it is the same workload on every pod. That is what makes a >3% deviation
    attributable to the host rather than to the experiment.
    """
    runs = []
    if exp.baseline != NO_BASELINE:
        if exp.baseline not in baselines:
            raise ManifestError(f"{exp.name}: unknown baseline '{exp.baseline}'")
        overrides = baselines[exp.baseline]
        runs.append(
            Run(
                name=f"baseline:{exp.baseline}",
                role=ROLE_BASELINE,
                overrides=tuple(overrides),
                config=resolved_config(list(overrides)),
            )
        )
    base = [*pod_overrides(exp), *exp.overrides]
    if not exp.settings:
        own = [
            Run(
                name=exp.name,
                role=ROLE_EXPERIMENT,
                overrides=tuple(base),
                config=resolved_config(base),
            )
        ]
        return [*runs, *own]
    for setting, extra in exp.settings.items():
        overrides = [*base, *extra]
        runs.append(
            Run(
                name=setting,
                role=ROLE_EXPERIMENT,
                overrides=tuple(overrides),
                config=resolved_config(overrides),
            )
        )
    return runs


def own_runs(runs: list[Run]) -> list[Run]:
    """The runs a pod was created to perform, in order, baseline excluded."""
    return [r for r in runs if r.role == ROLE_EXPERIMENT]


def image_for(exp: Experiment, registry: str, tag: str) -> str:
    if exp.framework not in IMAGE_SUFFIX:
        raise ManifestError(f"{exp.name}: no image is built for framework '{exp.framework}'")
    return f"{registry}-{IMAGE_SUFFIX[exp.framework]}:{tag}"


def image_digest(image: str) -> str | None:
    """The registry digest behind a mutable tag, or None if it cannot be read.

    `:latest` moves. Without the digest a result names an image that may already
    be a different image, which is the same failure as recording a commit from a
    dirty tree. Never guessed: an unknown digest is None and the caller decides.
    """
    buildx = ["docker", "buildx", "imagetools", "inspect", "--format"]
    attempts = (
        [*buildx, "{{json .Manifest.Digest}}", image],
        ["skopeo", "inspect", "--format", "{{.Digest}}", f"docker://{image}"],
    )
    for command in attempts:
        try:
            out = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        digest = out.stdout.strip().strip('"')
        if digest.startswith("sha256:"):
            return digest
    return None


def infisical_token() -> str:
    """A short-lived token for the pod, never the machine identity itself.

    Its absence is why a gated model answered 401 and the combination was filed as
    unsupported: with no token the pod runs without HF_TOKEN and every gated
    checkpoint looks like a checkpoint that does not exist.

    **The orchestrator's own `INFISICAL_TOKEN` is not reused**, and that is the
    whole point of this function. The documented way to run this script is
    `infisical run --env=dev -- python scripts/orchestrate.py`, which puts a
    dev-stored `INFISICAL_TOKEN` in the environment. That token is bound to `dev`
    and ignores `--env` entirely — measured 2026-08-02: it returns the same 26
    names for `--env=dev`, `--env=pod`, and for an environment that does not
    exist. Picking it up meant the documented invocation handed every pod a
    dev-wide token no matter which environment the operator selected, and
    separating the environments could not fix it.

    A token can still be supplied deliberately, under a name that says whose it
    is. `INFISICAL_TOKEN` names the caller's; `TRAINBENCH_POD_INFISICAL_TOKEN`
    names the pod's, and nothing puts it there by accident.

    The client secret is not passed on the command line — the Infisical CLI reads
    both halves of the universal-auth identity from the environment it inherits.
    """
    if token := os.environ.get("TRAINBENCH_POD_INFISICAL_TOKEN"):
        return token
    identity = ("INFISICAL_UNIVERSAL_AUTH_CLIENT_ID", "INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET")
    if not all(os.environ.get(name) for name in identity):
        raise RuntimeError(
            "no TRAINBENCH_POD_INFISICAL_TOKEN and no universal-auth identity in the "
            "environment; run under `infisical run --env=dev --`"
        )
    out = subprocess.run(
        ["infisical", "login", "--method=universal-auth", "--plain", "--silent"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    token = out.stdout.strip()
    if not token:
        raise RuntimeError("infisical login returned an empty token")
    return token


def pod_env(
    exp: Experiment,
    runs: list[Run],
    image: str,
    digest: str | None,
    token: str,
    commit: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    """Environment for one pod.

    Everything a result needs to be interpretable later travels with the pod:
    which commit, which image digest, and a token that lets it read the secrets
    it needs. What does not travel is anything that can act on the account.
    """
    own = own_runs(runs)
    if not own:
        raise ManifestError(f"{exp.name}: a pod plan with no run of the pod's own")
    env = {
        "TRAINBENCH_EXPERIMENT": exp.name,
        # The pod's OWN first run, never the baseline. The baseline names its own
        # model and framework, so handing it to the single-config entry point makes
        # the pod measure a different combination than the one it is filed under.
        # The full ordered plan travels beside it, each entry carrying a resolved
        # config, so a sweep pod can execute every setting it owns without Hydra.
        "TRAINBENCH_CONFIG_JSON": json.dumps(own[0].config),
        "TRAINBENCH_PLAN_JSON": json.dumps([r.summary() for r in runs]),
        "TRAINBENCH_PURPOSE": exp.purpose,
        "TRAINBENCH_RESULT_REPO": args.result_repo,
        "TRAINBENCH_GIT_COMMIT": commit,
        "TRAINBENCH_IMAGE": image,
        "TRAINBENCH_TIMEOUT_SECONDS": str(pod_timeout_seconds(args)),
        "INFISICAL_TOKEN": token,
        "INFISICAL_ENV": args.infisical_env,
    }
    if digest:
        env["TRAINBENCH_IMAGE_DIGEST"] = digest
    if project_id := args.infisical_project_id:
        # A machine-identity token has no project of its own; the CLI refuses to
        # fetch secrets without one.
        env["INFISICAL_PROJECT_ID"] = project_id
    # Checking `env` alone was security theatre: it is a literal built fifteen lines
    # above from a fixed set of keys, so the comprehension could never match. What
    # reaches the pod is not this dict — entrypoint.sh runs the workload under
    # `infisical run`, which injects everything the token can read. See
    # pod_reachable_secret_names.
    leaked = [name for name in FORBIDDEN_ON_POD if name in env]
    if leaked:
        raise RuntimeError(f"refusing to launch: {', '.join(leaked)} would reach an experiment pod")
    return env


def pod_reachable_secret_names(
    token: str, project_id: str, env: str = POD_INFISICAL_ENV
) -> set[str]:
    """Secret NAMES the pod's own token can read. Never values.

    The pod does not receive secrets through its env dict; it receives a token,
    and `entrypoint.sh` runs the workload under `infisical run`, which injects the
    whole environment that token can see. So the only meaningful question is what
    the token's scope is, and the only way to answer it is to ask with the token.

    `env` is the Infisical environment the pod will be handed (`INFISICAL_ENV`),
    passed in rather than read from `os.environ` here. Reading it from the
    orchestrator's own environment made the probe answer for whichever environment
    the operator happened to be running under while the pod ran in the one they
    passed on the command line — and separating the pod's secrets into their own
    environment is precisely the fix this guard recommends, so the guard would
    have gone blind at the moment it was acted on.

    The subprocess gets a sanitised environment so that what the orchestrator
    already holds cannot be mistaken for what the pod can reach.

    **What Infisical injected, not what the child process saw.** The same command
    is run twice from the same sanitised environment — once under `infisical run`
    and once without — and the answer is the difference. Subtracting only the
    names we set was wrong by exactly the variables an operating system adds to
    any child it spawns: on macOS `LC_CTYPE` and `__CF_USER_TEXT_ENCODING` came
    back as two secrets the pod supposedly reached, and a correctly scoped `pod`
    environment holding one secret was refused for holding three. A deny list
    never noticed, because locale variables were not on it.
    """
    clean = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", ""),
        "INFISICAL_TOKEN": token,
    }
    # chr(10) rather than an escape: this source is passed through argv, where a
    # literal newline inside the quotes is a syntax error.
    names = [sys.executable, "-c", "import os;print(chr(10).join(sorted(os.environ)))"]
    injected = _env_names(
        ["infisical", "run", f"--env={env}", f"--projectId={project_id}", "--", *names], clean
    )
    if injected is None:
        raise RuntimeError(f"could not read the pod token's scope in env '{env}'")
    ambient = _env_names(names, clean)
    if ambient is None:
        raise RuntimeError("could not read the environment a plain subprocess starts with")
    return injected - ambient


def _env_names(command: list[str], env: dict[str, str]) -> set[str] | None:
    """Environment variable names `command` prints, or None if it failed.

    None rather than an exception because one caller expects failure: asking for
    an environment that does not exist is how `token_is_bound_to_one_environment`
    tells a scoped token from one that ignores `--env`.
    """
    out = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    return set(out.stdout.split()) if out.returncode == 0 else None


def token_is_bound_to_one_environment(token: str, project_id: str) -> bool:
    """Whether this token returns the same secrets whatever environment is asked for.

    Measured, not assumed, and the measurement is a name no environment has. A
    token that honours `--env` makes the CLI fail on it; a token bound to one
    environment answers with that environment's secrets regardless, so a non-empty
    answer for an environment that cannot exist is the binding itself.

    Comparing two real environments would not do: they are allowed to hold the
    same names, and then a scoped token would look bound.

    Why this is checked at all — measured 2026-08-02, names and counts only:

        machine identity   --env=dev  26   --env=pod   1   --env=<nonexistent>  error
        dev-stored token   --env=dev  26   --env=pod  26   --env=<nonexistent>  26

    The second row is a service token bound to `dev`, and it was the token the
    documented invocation handed to every pod (see `infisical_token`). Today the
    scope check would refuse it anyway for the 25 extra secrets — but that is a
    fact about `dev`'s contents, not about the token, and it stops being true the
    day `dev` is tidied up.
    """
    nowhere = f"trainbench-no-such-env-{uuid.uuid4().hex}"
    try:
        return bool(pod_reachable_secret_names(token, project_id, nowhere))
    except RuntimeError:
        return False


def assert_pod_scope_is_safe(token: str, project_id: str, env: str = POD_INFISICAL_ENV) -> set[str]:
    """Refuse to launch unless the pod's token reaches exactly `ALLOWED_ON_POD`.

    This is the check `pod_env` was supposed to be. It measures the property that
    matters — what the pod can obtain — rather than what we chose to hand it.

    `env` must be the same Infisical environment the pod is handed, or this
    reports on a scope no pod will ever have.

    Both directions are refusals, and the second is not padding. A token that
    cannot read HF_TOKEN produces the failure already on this repository's record:
    every gated checkpoint answers 401, and the combination gets filed as
    unsupported by a pod that was never equipped to answer. An empty scope is the
    same refusal — a check whose subject is an empty set passes by having nothing
    to examine, which is how this repository has been wrong eight times.
    """
    if token_is_bound_to_one_environment(token, project_id):
        raise RuntimeError(
            f"refusing to launch: the pod's token ignores --env, so asking for '{env}' "
            "changes nothing about what it can read. It is bound to one environment "
            "(a service token), and separating the pod's secrets cannot reach it. "
            "Let the orchestrator mint a token from the universal-auth identity, or "
            "set TRAINBENCH_POD_INFISICAL_TOKEN to one that honours --env."
        )
    reachable = pod_reachable_secret_names(token, project_id, env)
    if extra := sorted(reachable - ALLOWED_ON_POD):
        urgent = [name for name in FORBIDDEN_ON_POD if name in extra]
        raise RuntimeError(
            f"refusing to launch: the pod's Infisical token can read {len(extra)} "
            f"secret(s) the pod has no use for: {', '.join(extra)}. "
            + (f"Account-wide among them: {', '.join(urgent)}. " if urgent else "")
            + "entrypoint.sh runs the workload under `infisical run`, which injects "
            "everything that token can see, so an experiment pod would hold all of "
            f"them. Scope the machine identity to {', '.join(sorted(ALLOWED_ON_POD))} "
            f"or point --infisical-env at an environment holding only that; env "
            f"'{env}' does not."
        )
    if missing := sorted(ALLOWED_ON_POD - reachable):
        raise RuntimeError(
            f"refusing to launch: the pod's Infisical token cannot read "
            f"{', '.join(missing)} in env '{env}'. Without it every gated checkpoint "
            "answers 401 and the combination is filed as unsupported by a pod that "
            "was never equipped to load it — a spent pod-hour producing a wrong "
            "result rather than no result."
        )
    return reachable


def pod_timeout_seconds(args: argparse.Namespace) -> int:
    """The pod's own deadline, always strictly inside the orchestrator's.

    The margin is a ceiling, not a fixed subtraction. Subtracting a constant and
    then flooring it inverted the relationship on short deadlines —
    `--timeout-minutes 1` gave the pod 120s against the orchestrator's 60s, so the
    watcher gave up first and the pod it was meant to outlive kept billing. On a
    short deadline the margin shrinks with it instead.
    """
    total = args.timeout_minutes * 60
    margin = min(SELF_KILL_MARGIN_SECONDS, total // 2)
    return max(1, total - margin)


def positive_minutes(raw: str) -> int:
    """A deadline of zero or less is a pod with no deadline, which bills until noticed."""
    minutes = int(raw)
    if minutes < 1:
        raise argparse.ArgumentTypeError("must be at least 1 minute")
    return minutes


def default_project_id() -> str:
    """The project the machine identity should read, from the repo's own config."""
    path = REPO_ROOT / ".infisical.json"
    if not path.exists():
        return ""
    return json.loads(path.read_text()).get("workspaceId", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        nargs="+",
        default=[],
        metavar="GLOB",
        help="manifest name(s) in configs/experiment, glob allowed; default all",
    )
    # GHCR namespace must match the GitHub account (jinwoo-cho-ilevit-com), which
    # differs from the Hugging Face account (jinwoo-cho) used for data/results.
    parser.add_argument("--registry", default="ghcr.io/jinwoo-cho-ilevit-com/trainbench")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--gpu-type-id", default="NVIDIA A100-SXM4-80GB")
    parser.add_argument("--result-repo", default="jinwoo-cho/trainbench-results")
    parser.add_argument(
        "--infisical-env",
        default=POD_INFISICAL_ENV,
        help="Infisical environment the pod reads through its token, and what the "
        "pre-launch scope check asks. Defaults to the pod's own environment; the "
        "orchestrator's own secrets come from the `infisical run` wrapping it, not "
        "from here",
    )
    parser.add_argument("--infisical-project-id", default=default_project_id())
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument("--timeout-minutes", type=positive_minutes, default=60)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="launch from a modified tree; the recorded commit then omits the code that ran",
    )
    parser.add_argument(
        "--allow-unpinned-image",
        action="store_true",
        help="launch from a mutable tag when the registry digest cannot be read",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/orchestrate.json"))
    args = parser.parse_args(argv)

    try:
        experiments = select(load_experiments(), args.experiment)
    except ManifestError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    baselines = load_baselines(BASELINES_PATH)

    git = git_state()
    if git["dirty"] and not args.allow_dirty and not args.dry_run:
        console.print(
            f"[red]working tree is dirty[/red]; commit {git['commit'][:12]} does not contain "
            "the code that would run. Commit, or pass --allow-dirty"
        )
        return 2

    # Written into the ledger as well as printed: the console scrolls past and the
    # ledger is what a merge reads months later, which is when "was this axis
    # compared across hosts" stops being obvious.
    cross_pod = cross_pod_notes(experiments)
    for note in cross_pod:
        console.print(f"[yellow]cross-pod axis[/yellow] {note}")

    ledger: dict[str, Any] = {
        "started_at": time.time(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "registry": args.registry,
        "tag": args.tag,
        "cross_pod_axes": cross_pod,
        "experiments": [],
    }
    digests: dict[str, str | None] = {}
    plans: dict[str, list[Run]] = {}
    for exp in experiments:
        try:
            image = image_for(exp, args.registry, args.tag)
            plans[exp.name] = plan_runs(exp, baselines)
        except Exception as exc:  # noqa: BLE001 - a variant that will not compose stops the sweep
            console.print(f"[red]{exp.name}: {exc}[/red]")
            return 2
        if image not in digests:
            digests[image] = image_digest(image)
        ledger["experiments"].append(
            {
                **exp.summary(),
                "image": image,
                "image_digest": digests[image],
                "gpu_type_id": exp.gpu_type_id or args.gpu_type_id,
                "runs": [r.summary() for r in plans[exp.name]],
                "pod_id": None,
                "launch_error": None,
                "outcome": None,
                # What the watch currently cannot see. Written while it is true and
                # cleared by the outcome, which carries the finished story. An
                # orchestrator killed mid-sweep — which is how the last three pods
                # ended — otherwise leaves a ledger that says nothing about ten
                # minutes of failed reads.
                "unreadable": None,
            }
        )

    entries = {e["experiment"]: e for e in ledger["experiments"]}
    console.print(f"{len(experiments)} experiment(s), up to {args.max_concurrent} pods at a time")

    if args.dry_run:
        for exp in experiments:
            entry = entries[exp.name]
            digest = entry["image_digest"]
            pinned = digest[7:19] if digest else "[yellow]unpinned[/yellow]"
            runs = len(entry["runs"])
            console.print(f"  {exp.name:44s} {runs} run(s)  {pinned}", highlight=False)
        ledger["dry_run"] = True
        console.print(f"wrote {write_json(args.out, ledger)}")
        return 0

    try:
        token = infisical_token()
        # What the pod can obtain, not what we chose to hand it. Names only.
        reachable = assert_pod_scope_is_safe(token, args.infisical_project_id, args.infisical_env)
    except (RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(
        f"infisical token acquired; pod scope is {len(reachable)} secret(s), none forbidden"
    )

    pending = list(experiments)
    running: dict[str, Experiment] = {}

    def blind(pod_id: str, message: str) -> None:
        """Say — and record — that a pod cannot be seen, while it is happening.

        `wait_for_any` blocks inside itself, so this is the only point at which a
        blind watch is audible before its ceiling is reached.
        """
        console.print(f"[yellow]{message}[/yellow]")
        if (exp := running.get(pod_id)) is not None:
            entries[exp.name]["unreadable"] = message
            write_json(args.out, ledger)

    watch = pods.PodWatch(timeout_seconds=args.timeout_minutes * 60, on_blind=blind)
    # The ledger is written at every state change, not at the end: an orchestrator
    # killed mid-sweep must still leave behind the ids of the pods it started, or
    # they bill until someone finds them by hand.
    write_json(args.out, ledger)

    while pending or watch.watching:
        while pending and len(running) < args.max_concurrent:
            exp = pending.pop(0)
            entry = entries[exp.name]
            if exp.purpose not in RUNNABLE_PURPOSES:
                entry["launch_error"] = (
                    f"purpose '{exp.purpose}' has no entry point in the pod image yet"
                )
                console.print(f"[yellow]skipped[/yellow] {exp.name}: {entry['launch_error']}")
                write_json(args.out, ledger)
                continue
            digest = entry["image_digest"]
            if not digest and not args.allow_unpinned_image:
                entry["launch_error"] = "registry digest unresolved; the tag may move under the run"
                console.print(f"[yellow]skipped[/yellow] {exp.name}: {entry['launch_error']}")
                write_json(args.out, ledger)
                continue
            # Launch the digest, not the tag: between planning and launch, `latest`
            # can become a different image.
            reference = f"{entry['image']}@{digest}" if digest else entry["image"]
            spec = pods.PodSpec(
                name=f"trainbench-{exp.name}"[:63],
                image=reference,
                gpu_type_id=entry["gpu_type_id"],
                env=pod_env(exp, plans[exp.name], reference, digest, token, git["commit"], args),
            )
            try:
                pod = pods.create(spec)
            except Exception as exc:  # noqa: BLE001 - a failed launch is a recorded outcome
                entry["launch_error"] = str(exc)
                console.print(f"[red]launch failed[/red] {spec.name}: {exc}")
                write_json(args.out, ledger)
                continue
            pod_id = pod["id"]
            entry["pod_id"] = pod_id
            entry["launched_at"] = time.time()
            running[pod_id] = exp
            watch.track(pod_id)
            write_json(args.out, ledger)
            console.print(f"[green]launched[/green] {spec.name} -> {pod_id}")

        if not watch.watching:
            break

        for outcome in watch.wait_for_any():
            exp = running.pop(outcome.pod_id, None)
            if exp is not None:
                entries[exp.name]["outcome"] = outcome.to_dict()
                # The live note is superseded by the outcome, which carries the
                # whole spell rather than its latest moment.
                entries[exp.name]["unreadable"] = None
                colour = "red" if outcome.reason == pods.REASON_UNREADABLE else "default"
                console.print(
                    f"[{colour}]{outcome.reason}[/{colour}] {exp.name} ({outcome.pod_id})"
                )
                if outcome.unreadable:
                    console.print(f"  [yellow]{outcome.unreadable}[/yellow]")
            # Terminate unconditionally: a pod left running keeps billing, and the
            # result has already been uploaded by the entrypoint.
            try:
                pods.terminate(outcome.pod_id)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]terminate failed[/red] {outcome.pod_id}: {exc}")
            write_json(args.out, ledger)

    ledger["finished_at"] = time.time()
    console.print(f"wrote {write_json(args.out, ledger)}")
    console.print("[bold]verify no pods are still billing: runpod list-pods[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
