"""Upload one pod's result to the shared results repo.

Each pod writes to its own path. Concurrent writes to the same file corrupt data,
and 18 pods finish at unpredictable times, so there is no shared file to write.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def result_path_in_repo(result: dict) -> str:
    probe = result.get("probe", {})
    framework = probe.get("framework", "unknown")
    model = probe.get("model", "unknown")
    # Pod id disambiguates repeats of the same combination (a re-run after a
    # baseline-deviation rejection, for instance).
    pod = result.get("host", {}).get("runpod_pod_id") or "local"
    return f"results/{framework}/{model}/{pod}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--repo", required=True, help="Hub dataset repo id")
    args = parser.parse_args(argv)

    if not args.result.exists():
        print(f"no result at {args.result}", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    result = json.loads(args.result.read_text())
    path_in_repo = result_path_in_repo(result)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=str(args.result),
        path_in_repo=path_in_repo,
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"uploaded {path_in_repo} to {args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
