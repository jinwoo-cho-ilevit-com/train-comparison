"""Merge probe results into the support matrix.

Only rewrites the generated section of docs/support-matrix.md. Everything above
the marker is hand-written analysis and stays untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MARKER = "<!-- generated: probe results -->"

FRAMEWORKS = ["native", "unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"]
MODELS = ["qwen3_vl_emb_2b", "qwen3_5_0_8b", "gemma4_e2b"]


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.rglob("*.json"))]


def cell(result: dict[str, Any] | None) -> str:
    """One matrix cell. Absent stays '미확인' — never inferred from a neighbour."""
    if result is None:
        return "미확인"
    probe = result.get("probe", {})
    checks = probe.get("checks", [])
    if not checks:
        return "미확인"
    failed = [c for c in checks if not c["ok"]]
    if not failed:
        return f"OK ({len(checks)}/{len(checks)})"
    first = failed[0]
    return f"FAIL {first['name']} ({first.get('error_type') or 'error'})"


def render(results: list[dict[str, Any]]) -> str:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        probe = result.get("probe", {})
        index[(probe.get("framework"), probe.get("model"))] = result

    lines = [
        MARKER,
        "",
        "## 모델 x 프레임워크 적재 검증 (자동 생성)",
        "",
        f"결과 {len(results)}건. 셀이 없는 조합은 미확인으로 남긴다.",
        "",
        "| | " + " | ".join(MODELS) + " |",
        "|---|" + "|".join(["---"] * len(MODELS)) + "|",
    ]
    for framework in FRAMEWORKS:
        row = [cell(index.get((framework, model))) for model in MODELS]
        lines.append(f"| {framework} | " + " | ".join(row) + " |")

    lines += [
        "",
        "### 실행 환경별 해석 버전",
        "",
        "| 조합 | torch | transformers | 프레임워크 |",
        "|---|---|---|---|",
    ]
    for (framework, model), result in sorted(index.items(), key=lambda kv: str(kv[0])):
        packages = result.get("packages", {})
        version = next(
            (
                c["detail"].get("version")
                for c in result.get("probe", {}).get("checks", [])
                if c["name"] == "framework_version"
            ),
            "-",
        )
        lines.append(
            f"| {framework} x {model} | {packages.get('torch', '-')} | "
            f"{packages.get('transformers', '-')} | {version} |"
        )

    lines += ["", "### 실패 상세", ""]
    any_failure = False
    for (framework, model), result in sorted(index.items(), key=lambda kv: str(kv[0])):
        for check in result.get("probe", {}).get("checks", []):
            if check["ok"]:
                continue
            any_failure = True
            lines.append(
                f"- **{framework} x {model} / {check['name']}** — {check.get('error_type')}"
            )
            if check.get("error"):
                lines.append(f"  - `{check['error'].strip().splitlines()[0][:180]}`")
    if not any_failure:
        lines.append("실패 없음.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="directory of result JSON")
    parser.add_argument("--matrix", type=Path, default=Path("docs/support-matrix.md"))
    args = parser.parse_args(argv)

    results = load_results(args.results)
    if not results:
        print(f"no results under {args.results}", file=sys.stderr)
        return 1

    generated = render(results)
    existing = args.matrix.read_text() if args.matrix.exists() else ""
    head = existing.split(MARKER)[0].rstrip() if MARKER in existing else existing.rstrip()
    args.matrix.write_text(f"{head}\n\n{generated}")
    print(f"merged {len(results)} results into {args.matrix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
