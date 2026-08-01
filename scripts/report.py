"""Merge probe results into the support matrix.

Only rewrites the generated section of docs/support-matrix.md. Everything above
the marker is hand-written analysis and stays untouched.

    python scripts/report.py --results downloaded/ --ledger outputs/orchestrate.json

Three distinctions this file exists to preserve:

* An **expected failure** is an answer, not a broken cell. Unsloth refusing a VLM
  checkpoint is the documented limitation the probe went to confirm.
* An expected failure that **passed** means the support matrix is wrong. The run
  is the only place that knows, and `all_ok` cannot say it.
* "Launched and produced nothing" is not "never attempted". Collapsing them turns
  a lost pod-hour into a combination nobody notices was never measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARKER = "<!-- generated: probe results -->"

FRAMEWORKS = ["native", "unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"]
MODELS = ["qwen3_vl_emb_2b", "qwen3_5_0_8b", "gemma4_e2b"]

RESULT_NAME = "result.json"
STARTED_NAME = "started.json"

NOT_ATTEMPTED = "미시도"
NO_RESULT = "결과 없음(기동됨)"
LAUNCH_FAILED = "기동 실패"
UNSUPPORTED = "미지원(문서화됨)"


@dataclass
class Artifact:
    """One JSON file a pod uploaded."""

    path: Path
    payload: dict[str, Any]
    kind: str  # "result" or "started"
    framework: str
    model: str
    timestamp: float

    @property
    def produced_result(self) -> bool:
        return self.kind == "result" and self.payload.get("status") != "no_result"


def _combination(payload: dict[str, Any]) -> tuple[str, str]:
    probe = payload.get("probe") or {}
    config = payload.get("config") or {}
    framework = probe.get("framework") or config.get("framework", {}).get("name") or "unknown"
    model = probe.get("model") or config.get("model", {}).get("name") or "unknown"
    return framework, model


def load_artifacts(results_dir: Path) -> tuple[list[Artifact], list[str]]:
    """Every readable artifact, plus the files that could not be read.

    An unparseable file is reported and stepped over. A pod that vanished
    mid-upload must not take the other seventeen results down with it.
    """
    artifacts, skipped = [], []
    for path in sorted(results_dir.rglob("*.json")):
        if path.name not in (RESULT_NAME, STARTED_NAME):
            continue
        # Relative, because these names end up in a committed document and the
        # absolute path of whoever ran the merge is not part of the evidence.
        shown = path.relative_to(results_dir)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(f"{shown}: {type(exc).__name__}")
            continue
        if not isinstance(payload, dict):
            skipped.append(f"{shown}: not a JSON object")
            continue
        framework, model = _combination(payload)
        artifacts.append(
            Artifact(
                path=shown,
                payload=payload,
                kind="result" if path.name == RESULT_NAME else "started",
                framework=framework,
                model=model,
                # `recorded_at` exists on records this repo generates; run records
                # written by the probe carry no clock, so the file's own mtime is
                # the fallback. Both only ever order artifacts against each other.
                timestamp=float(payload.get("recorded_at") or path.stat().st_mtime),
            )
        )
    return artifacts, skipped


def newest_per_combination(
    artifacts: list[Artifact],
) -> tuple[dict[tuple[str, str], Artifact], list[str]]:
    """Newest artifact per combination, and a warning for every one it displaced.

    A combination measured twice is usually a deliberate re-run, but it can also
    be two pods that both thought they owned it. Silently keeping one hides the
    second case, so every superseded artifact is named.
    """
    ranked: dict[tuple[str, str], list[Artifact]] = {}
    for artifact in artifacts:
        ranked.setdefault((artifact.framework, artifact.model), []).append(artifact)
    chosen, duplicates = {}, []
    for key, group in ranked.items():
        # A real result outranks a bare `started`, then newest wins.
        group.sort(key=lambda a: (a.produced_result, a.kind == "result", a.timestamp), reverse=True)
        chosen[key] = group[0]
        for superseded in group[1:]:
            duplicates.append(f"{key[0]} x {key[1]}: ignored {superseded.path}")
    return chosen, duplicates


def load_ledger(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """What the orchestrator says it launched, keyed by combination."""
    if path is None or not path.exists():
        return {}
    ledger = json.loads(path.read_text())
    return {(entry["framework"], entry["model"]): entry for entry in ledger.get("experiments", [])}


def checks_of(artifact: Artifact | None) -> list[dict[str, Any]]:
    if artifact is None:
        return []
    return (artifact.payload.get("probe") or {}).get("checks") or []


def cell(artifact: Artifact | None, launched: dict[str, Any] | None) -> str:
    """One matrix cell. Absent stays '미시도' — never inferred from a neighbour."""
    if artifact is None:
        if launched is None:
            return NOT_ATTEMPTED
        if launched.get("launch_error"):
            return LAUNCH_FAILED
        return NO_RESULT if launched.get("pod_id") else NOT_ATTEMPTED
    if not artifact.produced_result:
        return NO_RESULT
    checks = checks_of(artifact)
    if not checks:
        return NO_RESULT
    # An expected failure is the answer the probe went looking for, so it does not
    # make the cell read as broken.
    graded = [c for c in checks if not c.get("expected_failure")]
    failed = [c for c in graded if not c["ok"]]
    if failed:
        first = failed[0]
        return f"FAIL {first['name']} ({first.get('error_type') or 'error'})"
    documented = len(checks) - len(graded)
    if not graded:
        # Every check was a documented limitation, so nothing was left to grade.
        # That is the "unsupported" verdict Phase 3 reports, not a pass.
        return f"{UNSUPPORTED} ({documented}건)"
    suffix = f", 문서화된 한계 {documented}건" if documented else ""
    return f"OK ({len(graded)} checks{suffix})"


def unexpected_passes(artifact: Artifact | None) -> list[str]:
    """Checks the support matrix says cannot work, which did."""
    if artifact is None:
        return []
    probe = artifact.payload.get("probe") or {}
    recorded = probe.get("unexpected_passes")
    if recorded is not None:
        return list(recorded)
    return [c["name"] for c in checks_of(artifact) if c.get("expected_failure") and c["ok"]]


def render(
    chosen: dict[tuple[str, str], Artifact],
    ledger: dict[tuple[str, str], dict[str, Any]],
    duplicates: list[str],
    skipped: list[str],
) -> str:
    results = [a for a in chosen.values() if a.produced_result]
    lines = [
        MARKER,
        "",
        "## 모델 x 프레임워크 적재 검증 (자동 생성)",
        "",
        f"결과 {len(results)}건, 아티팩트 {len(chosen)}건. "
        f"`{NOT_ATTEMPTED}`는 pod을 띄운 적이 없는 조합, "
        f"`{NO_RESULT}`는 띄웠으나 결과 파일이 올라오지 않은 조합, "
        f"`{UNSUPPORTED}`는 모든 체크가 문서화된 한계였던 조합이다.",
        "",
        "| | " + " | ".join(MODELS) + " |",
        "|---|" + "|".join(["---"] * len(MODELS)) + "|",
    ]
    for framework in FRAMEWORKS:
        row = [
            cell(chosen.get((framework, model)), ledger.get((framework, model))) for model in MODELS
        ]
        lines.append(f"| {framework} | " + " | ".join(row) + " |")

    surprises = {}
    for key, artifact in sorted(chosen.items()):
        if names := unexpected_passes(artifact):
            surprises[key] = names
    if surprises:
        lines += [
            "",
            "### 지원 매트릭스가 틀렸다 — 실패할 것으로 표시한 체크가 통과했다",
            "",
            "문서화된 한계가 사라졌다는 뜻이므로, 해당 셀의 근거를 다시 확인해야 한다.",
            "",
        ]
        for (framework, model), names in surprises.items():
            lines.append(f"- **{framework} x {model}** — {', '.join(names)}")

    lines += [
        "",
        "### 실행 환경별 해석 버전",
        "",
        "| 조합 | torch | transformers | 프레임워크 |",
        "|---|---|---|---|",
    ]
    for (framework, model), artifact in sorted(chosen.items()):
        packages = artifact.payload.get("packages") or {}
        version = next(
            (
                c["detail"].get("version")
                for c in checks_of(artifact)
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
    for (framework, model), artifact in sorted(chosen.items()):
        for check in checks_of(artifact):
            if check["ok"] or check.get("expected_failure"):
                continue
            any_failure = True
            lines.append(
                f"- **{framework} x {model} / {check['name']}** — {check.get('error_type')}"
            )
            if check.get("error"):
                lines.append(f"  - `{check['error'].strip().splitlines()[0][:180]}`")
    if not any_failure:
        lines.append("실패 없음.")

    if duplicates or skipped:
        lines += ["", "### 병합에서 제외한 파일", ""]
        lines += [f"- 중복: {item}" for item in duplicates]
        lines += [f"- 판독 불가: {item}" for item in skipped]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="directory of result JSON")
    parser.add_argument(
        "--ledger",
        type=Path,
        help="orchestrator ledger, so a launched pod that produced nothing is not read as untried",
    )
    parser.add_argument("--matrix", type=Path, default=Path("docs/support-matrix.md"))
    args = parser.parse_args(argv)

    artifacts, skipped = load_artifacts(args.results)
    for item in skipped:
        print(f"skipped {item}", file=sys.stderr)
    ledger = load_ledger(args.ledger)
    if not artifacts and not ledger:
        print(f"no artifacts under {args.results} and no ledger", file=sys.stderr)
        return 1

    chosen, duplicates = newest_per_combination(artifacts)
    for item in duplicates:
        print(f"duplicate {item}", file=sys.stderr)

    generated = render(chosen, ledger, duplicates, skipped)
    existing = args.matrix.read_text() if args.matrix.exists() else ""
    head = existing.split(MARKER)[0].rstrip() if MARKER in existing else existing.rstrip()
    args.matrix.write_text(f"{head}\n\n{generated}")
    print(f"merged {len(chosen)} artifact(s) into {args.matrix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
