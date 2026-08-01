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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from rich.console import Console

from trainbench import pods
from trainbench.compose import resolve
from trainbench.config import git_state
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
RUNNABLE_PURPOSES = frozenset({"probe"})

# Never reachable from an experiment pod. RUNPOD_API_KEY is account-wide and would
# let a probe delete the sweep that created it; GITHUB_TOKEN carries write:packages
# and belongs to the image build alone. The universal-auth pair is on the list
# because an identity that can mint its own tokens makes a short-lived token
# pointless.
FORBIDDEN_ON_POD = (
    "RUNPOD_API_KEY",
    "GITHUB_TOKEN",
    "INFISICAL_UNIVERSAL_AUTH_CLIENT_ID",
    "INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET",
)

# The pod kills itself this long before the orchestrator's own deadline, so a hung
# run stops billing on its own even if the orchestrator dies first.
SELF_KILL_MARGIN_SECONDS = 120


class ManifestError(ValueError):
    """An experiment definition that cannot be trusted to describe a pod."""


@dataclass(frozen=True)
class Run:
    """One resolved setting a pod executes."""

    name: str
    overrides: tuple[str, ...]
    config: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "overrides": list(self.overrides)}


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
    return Experiment(
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
    return experiments


def check_axis_not_split(experiments: list[Experiment]) -> None:
    """No axis may be compared across two pods for the same model.

    Pods are different physical hosts. Comparing FA2 on one host against FA3 on
    another measures the hosts as much as the kernels, so PLAN.md forbids it. The
    rule is enforceable here because a manifest is exactly one pod.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for exp in experiments:
        if exp.axis:
            seen.setdefault((exp.model, exp.axis), []).append(exp.name)
    split = {key: names for key, names in seen.items() if len(names) > 1}
    if split:
        detail = "; ".join(
            f"{model} x {axis} split across {', '.join(sorted(names))}"
            for (model, axis), names in sorted(split.items())
        )
        raise ManifestError(f"an axis is split across pods, which invalidates it: {detail}")


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
                overrides=tuple(overrides),
                config=resolved_config(list(overrides)),
            )
        )
    base = [f"framework={exp.framework}", f"model={exp.model}", f"run={exp.purpose}"]
    base += exp.overrides
    if not exp.settings:
        return [*runs, Run(name=exp.name, overrides=tuple(base), config=resolved_config(base))]
    for setting, extra in exp.settings.items():
        overrides = [*base, *extra]
        runs.append(
            Run(name=setting, overrides=tuple(overrides), config=resolved_config(overrides))
        )
    return runs


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

    The client secret is not passed on the command line — the Infisical CLI reads
    both halves of the universal-auth identity from the environment it inherits.
    """
    if token := os.environ.get("INFISICAL_TOKEN"):
        return token
    identity = ("INFISICAL_UNIVERSAL_AUTH_CLIENT_ID", "INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET")
    if not all(os.environ.get(name) for name in identity):
        raise RuntimeError(
            "no INFISICAL_TOKEN and no universal-auth identity in the environment; "
            "run under `infisical run --env=dev --`"
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
    env = {
        "TRAINBENCH_EXPERIMENT": exp.name,
        # The first run, for the single-config entry point. The full ordered plan
        # travels beside it so a sweep pod knows every setting it owns.
        "TRAINBENCH_CONFIG_JSON": json.dumps(runs[0].config),
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


def pod_reachable_secret_names(token: str, project_id: str) -> set[str]:
    """Secret NAMES the pod's own token can read. Never values.

    The pod does not receive secrets through its env dict; it receives a token,
    and `entrypoint.sh` runs the workload under `infisical run`, which injects the
    whole environment that token can see. So the only meaningful question is what
    the token's scope is, and the only way to answer it is to ask with the token.

    The subprocess gets a sanitised environment so that what the orchestrator
    already holds cannot be mistaken for what the pod can reach.
    """
    clean = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", ""),
        "INFISICAL_TOKEN": token,
    }
    out = subprocess.run(
        [
            "infisical",
            "run",
            f"--env={os.environ.get('INFISICAL_ENV', 'dev')}",
            f"--projectId={project_id}",
            "--",
            sys.executable,
            "-c",
            # chr(10) rather than an escape: this source is passed through argv,
            # where a literal newline inside the quotes is a syntax error.
            "import os;print(chr(10).join(sorted(os.environ)))",
        ],
        env=clean,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"could not read the pod token's scope: {out.stderr.strip()[:300]}")
    return set(out.stdout.split()) - set(clean)


def assert_pod_scope_is_safe(token: str, project_id: str) -> set[str]:
    """Refuse to launch while the pod's token can read secrets it has no business with.

    This is the check `pod_env` was supposed to be. It measures the property that
    matters — what the pod can obtain — rather than what we chose to hand it.
    """
    reachable = pod_reachable_secret_names(token, project_id)
    leaked = sorted(set(FORBIDDEN_ON_POD) & reachable)
    if leaked:
        raise RuntimeError(
            "refusing to launch: the pod's Infisical token can read "
            f"{', '.join(leaked)}. entrypoint.sh injects everything that token can "
            "see, so an experiment pod would hold them. Scope the machine identity "
            "to a pod-only secret set (or a separate Infisical environment) — "
            "lengthening FORBIDDEN_ON_POD cannot reach this."
        )
    return reachable


def pod_timeout_seconds(args: argparse.Namespace) -> int:
    return max(SELF_KILL_MARGIN_SECONDS, args.timeout_minutes * 60 - SELF_KILL_MARGIN_SECONDS)


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
    parser.add_argument("--infisical-env", default="dev")
    parser.add_argument("--infisical-project-id", default=default_project_id())
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument("--timeout-minutes", type=int, default=60)
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

    ledger: dict[str, Any] = {
        "started_at": time.time(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "registry": args.registry,
        "tag": args.tag,
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
        reachable = assert_pod_scope_is_safe(token, args.infisical_project_id)
    except (RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(
        f"infisical token acquired; pod scope is {len(reachable)} secret(s), none forbidden"
    )

    pending = list(experiments)
    watch = pods.PodWatch(timeout_seconds=args.timeout_minutes * 60)
    running: dict[str, Experiment] = {}
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
                console.print(f"{outcome.reason} {exp.name} ({outcome.pod_id})")
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
