"""Run the framework x model sweep across many pods.

Composition happens here, not on the pod: each combination's resolved config is
passed in as an environment variable, so the config that ran is exactly the config
recorded, and no image needs Hydra.

    python scripts/orchestrate.py --frameworks native --models qwen3_5_0_8b --max-concurrent 1

Start with one combination. Verifying the image, secret injection and result
upload on a single pod costs one pod-hour; discovering a broken entrypoint across
eighteen costs eighteen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from rich.console import Console

from trainbench import pods
from trainbench.compose import resolve
from trainbench.record import write_json

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

FRAMEWORKS = ["native", "unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"]
MODELS = ["qwen3_vl_emb_2b", "qwen3_5_0_8b", "gemma4_e2b"]

# Framework images are named after their env directory, which uses hyphens.
IMAGE_SUFFIX = {
    "native": "native",
    "unsloth": "unsloth",
    "ms_swift": "ms-swift",
    "sentence_transformers": "sentence-transformers",
    "tevatron": "tevatron",
    "axolotl": "axolotl",
}


def resolved_config(framework: str, model: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"framework={framework}", f"model={model}", "run=probe", *overrides],
        )
        return resolve(cfg)[1]


def pod_env(config: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    """Environment for one pod.

    RUNPOD_API_KEY is deliberately absent: a probe pod has no reason to hold the
    key that created it. Secrets arrive through the Infisical machine identity.
    """
    env = {
        "TRAINBENCH_CONFIG_JSON": json.dumps(config),
        "TRAINBENCH_RESULT_REPO": args.result_repo,
        "INFISICAL_ENV": args.infisical_env,
    }
    if args.infisical_project_id:
        env["INFISICAL_PROJECT_ID"] = args.infisical_project_id
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameworks", nargs="+", default=FRAMEWORKS, choices=FRAMEWORKS)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    # GHCR namespace must match the GitHub account (jinwoo-cho-ilevit-com), which
    # differs from the Hugging Face account (jinwoo-cho) used for data/results.
    parser.add_argument("--registry", default="ghcr.io/jinwoo-cho-ilevit-com/trainbench")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--gpu-type-id", default="NVIDIA A100-SXM4-80GB")
    parser.add_argument("--result-repo", default="jinwoo-cho/trainbench-results")
    parser.add_argument("--infisical-env", default="dev")
    parser.add_argument("--infisical-project-id", default="")
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument("--timeout-minutes", type=int, default=60)
    parser.add_argument("--override", nargs="*", default=[], help="extra Hydra overrides")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    parser.add_argument("--out", type=Path, default=Path("outputs/orchestrate.json"))
    args = parser.parse_args(argv)

    combos = [(f, m) for f in args.frameworks for m in args.models]
    console.print(f"{len(combos)} combinations, up to {args.max_concurrent} pods at a time")

    plan = []
    for framework, model in combos:
        config = resolved_config(framework, model, args.override)
        plan.append(
            {
                "framework": framework,
                "model": model,
                "image": f"{args.registry}-{IMAGE_SUFFIX[framework]}:{args.tag}",
                "config": config,
            }
        )

    if args.dry_run:
        for item in plan:
            console.print(f"  {item['framework']:22s} {item['model']:18s} {item['image']}")
        write_json(args.out, {"dry_run": True, "plan": [_summary(p) for p in plan]})
        return 0

    pending = list(plan)
    running: dict[str, dict[str, Any]] = {}
    launched: list[dict[str, Any]] = []

    while pending or running:
        while pending and len(running) < args.max_concurrent:
            item = pending.pop(0)
            spec = pods.PodSpec(
                name=f"probe-{item['framework']}-{item['model']}",
                image=item["image"],
                gpu_type_id=args.gpu_type_id,
                env=pod_env(item["config"], args),
            )
            try:
                pod = pods.create(spec)
            except Exception as exc:  # noqa: BLE001 - a failed launch is a recorded outcome
                console.print(f"[red]launch failed[/red] {spec.name}: {exc}")
                launched.append({**_summary(item), "pod_id": None, "launch_error": str(exc)})
                continue
            pod_id = pod["id"]
            running[pod_id] = item
            launched.append({**_summary(item), "pod_id": pod_id})
            console.print(f"[green]launched[/green] {spec.name} -> {pod_id}")

        if not running:
            break

        finished = pods.wait_for_any(list(running), args.timeout_minutes * 60)
        for pod_id in finished:
            item = running.pop(pod_id, None)
            if item:
                console.print(f"finished {item['framework']} x {item['model']} ({pod_id})")
            # Terminate unconditionally: a pod left running keeps billing, and the
            # result has already been uploaded by the entrypoint.
            try:
                pods.terminate(pod_id)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]terminate failed[/red] {pod_id}: {exc}")

    path = write_json(args.out, {"launched": launched, "finished_at": time.time()})
    console.print(f"wrote {path}")
    console.print("[bold]verify no pods are still billing: runpod list-pods[/bold]")
    return 0


def _summary(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k != "config"}


if __name__ == "__main__":
    sys.exit(main())
