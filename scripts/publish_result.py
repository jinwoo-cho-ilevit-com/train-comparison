"""Upload one pod's artifacts to the shared results repo.

Each pod writes to its own directory, and a sweep pod to one subdirectory per
setting. Concurrent writes to the same file corrupt data, and eighteen pods finish
at unpredictable times, so there is no shared file to write; within one pod, the
settings of an axis would overwrite each other just as surely.

Three things get published, and the difference between them is the point:

    start      before the model is touched, so a pod that dies during the pull is
               still on record as having started
    result     what the run produced
    fallback   a real record saying the run produced nothing

Without the first and the last, "no result" and "never launched" arrive at the
report as the same absence, and a support matrix cannot tell them apart.

Runs inside every framework image, so it stays import-light: no torch, no Hydra.
A run that failed because the stack is broken must still be able to say so.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# The Hub rate-limits and occasionally 500s. Eighteen pods finishing together is
# exactly when that happens, and a dropped upload is a lost pod-hour.
RETRY_DELAYS = (2, 8, 30)

STARTED_NAME = "started.json"
RESULT_NAME = "result.json"

# Anything else becomes a separator. Run names carry `:` (`baseline:canonical`),
# and a path segment is not the place to find out how a Hub repo handles it.
UNSAFE_IN_PATH = re.compile(r"[^A-Za-z0-9._-]+")


def setting_dir(label: str) -> str:
    """The directory one setting of a sweep publishes into.

    A sweep pod runs every value of one axis, so it produces one result per
    setting and needs a destination derivable from the setting. The plan item's
    `name` is that key; its index is not, because index 0 of one manifest and
    index 0 of another are different settings that would land on the same path.

    A directory rather than a `result-<setting>.json` filename because
    `report.load_artifacts` reads files named `result.json` and skips everything
    else — a renamed file would upload cleanly and never reach the matrix.
    """
    segment = UNSAFE_IN_PATH.sub("-", label).strip("-.")
    if not segment:
        raise ValueError(f"setting label {label!r} leaves nothing usable as a path segment")
    return segment


def result_path_in_repo(directory: str, label: str | None) -> str:
    """Where one result lands. Unlabelled — a single-run pod — keeps its old path."""
    if label is None:
        return f"{directory}/{RESULT_NAME}"
    return f"{directory}/{setting_dir(label)}/{RESULT_NAME}"


def result_dir_in_repo(config: dict[str, Any]) -> str:
    framework = config.get("framework", {}).get("name", "unknown")
    model = config.get("model", {}).get("name", "unknown")
    # Pod id disambiguates repeats of the same combination (a re-run after a
    # baseline-deviation rejection, for instance).
    pod = os.environ.get("RUNPOD_POD_ID") or "local"
    return f"results/{framework}/{model}/{pod}"


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    """The fields that make an artifact traceable, read from the pod's environment.

    Deliberately mirrors `trainbench.record.build_record` for the keys a reader
    needs — `git_commit`, `image_digest`, `config` — without importing it, so a
    pod whose framework stack is unimportable can still publish provenance.
    """
    commit = os.environ.get("TRAINBENCH_GIT_COMMIT")
    return {
        "git_commit": commit or "unknown",
        # Unknowable here: the image has no .git. The orchestrator refuses to
        # launch from a dirty tree, which is what makes the commit meaningful.
        "git_dirty": None,
        "git_source": "env" if commit else "unavailable",
        "image": os.environ.get("TRAINBENCH_IMAGE"),
        "image_digest": os.environ.get("TRAINBENCH_IMAGE_DIGEST"),
        "experiment": os.environ.get("TRAINBENCH_EXPERIMENT"),
        "config": config,
        "host": {
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
            "runpod_datacenter": os.environ.get("RUNPOD_DC_ID"),
        },
        "recorded_at": time.time(),
    }


def started_record(config: dict[str, Any]) -> dict[str, Any]:
    return {**provenance(config), "status": "started"}


def fallback_record(config: dict[str, Any], reason: str) -> dict[str, Any]:
    """A record for a run that produced no result file.

    Shaped like a probe result on purpose: the report reads one structure, and a
    pod that produced nothing shows up as a failed check rather than as a gap that
    looks like a combination nobody tried.
    """
    return {
        **provenance(config),
        "status": "no_result",
        "probe": {
            "framework": config.get("framework", {}).get("name", "unknown"),
            "model": config.get("model", {}).get("name", "unknown"),
            "all_ok": False,
            "unexpected_passes": [],
            "applied": None,
            "checks": [
                {
                    "name": "result_file",
                    "ok": False,
                    "expected_failure": False,
                    "detail": {},
                    "error": reason,
                    "error_type": "NoResult",
                    "traceback": None,
                }
            ],
        },
    }


def with_retry(action, describe: str, sleep=time.sleep) -> Any:
    """Retry with backoff, then give up loudly.

    Silence here is the expensive failure: the pod terminates either way, so an
    upload that quietly failed costs the whole run.
    """
    last: Exception | None = None
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - every Hub failure is retried the same way
            last = exc
            if delay is None:
                break
            print(f"{describe} failed ({type(exc).__name__}), retry {attempt + 1} in {delay}s")
            sleep(delay)
    raise RuntimeError(f"{describe} failed after {len(RETRY_DELAYS) + 1} attempts") from last


def publish(path: Path, repo: str, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    # The results repo is private and may not exist on the first pod of a
    # campaign. Every pod asserting it exists is cheaper than a race over who
    # creates it.
    with_retry(
        lambda: api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True),
        f"create_repo {repo}",
    )
    with_retry(
        lambda: api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo,
            repo_type="dataset",
        ),
        f"upload {path_in_repo}",
    )
    print(f"uploaded {path_in_repo} to {repo}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hub dataset repo id")
    parser.add_argument("--config", required=True, type=Path, help="resolved config JSON")
    parser.add_argument("--mode", choices=("start", "result", "fallback"), default="result")
    parser.add_argument("--result", type=Path, help="result JSON; required for result/fallback")
    parser.add_argument("--reason", default="", help="why there is no result (fallback mode)")
    parser.add_argument("--out-dir", type=Path, help="where to write a generated record")
    parser.add_argument(
        "--label",
        help="name of the setting this result belongs to; a sweep pod publishes one per setting",
    )
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    directory = result_dir_in_repo(config)
    try:
        destination = result_path_in_repo(directory, args.label)
    except ValueError as exc:
        # Falling back to the shared name would let two settings overwrite each
        # other, which is the failure this destination exists to prevent.
        print(exc, file=sys.stderr)
        return 2

    if args.mode == "start":
        out_dir = args.out_dir or args.config.parent
        path = out_dir / STARTED_NAME
        path.write_text(json.dumps(started_record(config), indent=2, ensure_ascii=False))
        publish(path, args.repo, f"{directory}/{STARTED_NAME}")
        return 0

    if args.result is None:
        print("--result is required for this mode", file=sys.stderr)
        return 2

    if args.mode == "fallback":
        reason = args.reason or "the run produced no result file"
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(fallback_record(config, reason), indent=2, ensure_ascii=False)
        )
        print(f"no result at {args.result}; publishing a fallback record: {reason}")
    elif not args.result.exists():
        print(f"no result at {args.result}", file=sys.stderr)
        return 1

    publish(args.result, args.repo, destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
