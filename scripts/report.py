"""Merge what the pods uploaded into the support matrix and the results tables.

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

An artifact goes down exactly one of three lanes, because the three answer
different questions and one table cannot hold all three:

* **matrix** — a probe (or the fallback record standing in for one). What loads.
* **measurement** — a run with a measuring purpose. What it cost. These carry
  `metrics` and no probe checks; reading them through `checks_of` reported a run
  that produced every figure as "기동됨, 결과 없음".
* **baseline** — the canonical reference workload every measuring pod repeats.
  It is filed by *pod*, not by cell: its config names one fixed combination, so
  every pod's copy would otherwise land on the single `(native, qwen3_5_0_8b)`
  cell and be discarded as a duplicate of whatever probed that cell.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import publish_result

from trainbench.metrics import validity

MARKER = "<!-- generated: probe results -->"

# The generated section's heading, and the text a rival hand-written section would
# share with it. Two tables under the same heading, one above the marker and one
# below, is a document that contradicts itself in the order a reader reads it.
MATRIX_HEADING = "모델 x 프레임워크 적재 검증"
GENERATED_HEADING = f"## {MATRIX_HEADING} (자동 생성)"

FRAMEWORKS = ["native", "unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"]
MODELS = ["qwen3_vl_emb_2b", "qwen3_5_0_8b", "gemma4_e2b"]

RESULT_NAME = publish_result.RESULT_NAME
STARTED_NAME = publish_result.STARTED_NAME

NOT_ATTEMPTED = "미시도"
NO_RESULT = "결과 없음(기동됨)"
LAUNCH_FAILED = "기동 실패"
UNSUPPORTED = "미지원(문서화됨)"
NO_METRICS = "지표 없음"
OOM = "OOM(메모리 한계)"
NOT_TRAINED = "학습하지 않은 런"

# The record field naming the campaign an artifact belongs to. `record.build_record`
# and `publish_result.provenance` both stamp it, so an artifact without one is not
# something this repository produced and its position in time is unknown.
IDENTITY_FIELD = "recorded_at"

# A run that hit the memory ceiling. It is a result — the setting does not fit on
# this GPU — and it is neither "slow", "unsupported" nor "never attempted".
STATUS_OOM = "oom"

# `scripts/bench.py`'s `REFUSED_STATUS` / `FAILED_STATUS` prefixes, mirrored
# rather than imported: that module pulls torch and every framework adapter to
# run a training step, and this file runs in the merge environment, which has
# none of that installed. Both are distinct from `STATUS_OOM` above and from
# `publish_result.fallback_record`'s `"no_result"` — four different reasons a
# setting can carry no figures, and `cell()` used to read only the first as
# distinct and let the other three collapse into "launched, produced nothing".
AXIS_REFUSED_PREFIX = "axis-refused"
RUN_FAILED_PREFIX = "run-failed"
AXIS_REFUSED = "축 거부됨"
RUN_FAILED = "런 실패"

# The parenthesised detail in a `bench.py` status string — `(stage,
# ErrorType)` for a refusal, `(ErrorType)` for a failure — without the reason
# sentence after the em dash, which can run long enough to break a table row.
_STATUS_DETAIL_RE = re.compile(r"\(([^)]*)\)")

# The record's own summary of which axes its framework took over. The vocabulary
# belongs to the `applied-axes` boundary; this file only reads it back.
FRAMEWORK_OWNED_KEY = "framework_owned"

STACK_UNKNOWN = "스택 미상"

# Purposes that exist to produce numbers. `probe` is absent: it answers whether a
# combination loads, which is the matrix's question and not this one.
MEASURING_PURPOSES = frozenset({"timing", "profile", "quality"})

# `orchestrate.plan_runs` names every canonical baseline run `baseline:<name>`,
# and the pod publishes it under `publish_result.setting_dir(<that name>)`. The
# prefix is sanitised the same way rather than spelled out, so the two cannot
# drift apart into a baseline that uploads fine and is never recognised.
BASELINE_RUN_PREFIX = "baseline:"
BASELINE_DIR_PREFIX = publish_result.UNSAFE_IN_PATH.sub("-", BASELINE_RUN_PREFIX)

# What a pod's baseline is compared on. The median step time is the measured
# quantity itself; samples/s and tokens/s are derived from it through per-step
# counts that differ with padding, so a deviation in one of those does not say
# whether the *host* was slower.
BASELINE_METRIC = "step_seconds_p50"

# What a record that declares no threshold of its own was measured under. Not a
# second source of truth: it is the default of `measurement.baseline_tolerance`
# (trainbench/config_schema.py), and the value a record carries wins over it.
BASELINE_DEVIATION_LIMIT = 0.03

# Where a record states the threshold its campaign ran under. `metrics.summarise`
# writes both from `config.measurement`, so a tolerance derived on the first pod
# arrives here instead of being declared and then ignored.
TOLERANCE_FIELD = "baseline_tolerance"
TOLERANCE_STATUS_FIELD = "baseline_tolerance_status"
TOLERANCE_CALIBRATED = "calibrated"
TOLERANCE_UNDECLARED = "미선언"

# Printed next to every deviation figure. docs/methodology.md §4 states this
# outright, and a threshold that looks derived is worse than no threshold: the
# first GPU pod repeats the baseline five times and the measured spread is what
# fixes the number. Until then this is a convention, not evidence.
BASELINE_DEVIATION_SOURCE = (
    "**미교정 임계값이다.** docs/methodology.md §4가 근거 없는 값이라고 명시한다 — "
    "동일 pod에서 baseline을 5회 반복해 편차를 실측한 뒤 확정한다. 실측 편차가 이 값을 "
    "넘으면 임계값이 아니라 측정 절차를 고쳐야 한다는 신호다. 아래 판정은 그 교정 전의 "
    "잠정 판정이다."
)

# The counterpart, for when the records say the number was derived rather than
# assumed. It names the field so a reader can check the claim in the artifact.
BASELINE_DEVIATION_CALIBRATED_SOURCE = (
    "**교정된 임계값이다.** baseline 레코드의 `metrics.measurement.baseline_tolerance`가 "
    "이 값을 싣고 `baseline_tolerance_status`가 `calibrated`라고 적는다. 판정은 그 값으로 "
    "냈다."
)

POD_OK = "OK"
POD_INVALID = "무효"
POD_NO_BASELINE = "기준선 없음"
POD_UNJUDGED = "판정 불가"


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

    @property
    def oom(self) -> bool:
        """Whether the run ended at the memory ceiling rather than producing figures."""
        return str(self.payload.get("status") or "").lower() == STATUS_OOM

    @property
    def refused(self) -> bool:
        """Whether an axis this run asked for could not be put into effect.

        `scripts/bench.py::refusal_record` stamps this prefix and carries no
        `metrics` (by that function's own contract), so this is never true at
        the same time as `.metrics` being set.
        """
        return str(self.payload.get("status") or "").startswith(AXIS_REFUSED_PREFIX)

    @property
    def failed(self) -> bool:
        """Whether a diagnosable failure stopped this run before it measured a window.

        `scripts/bench.py::failure_status` stamps this prefix, distinct from an
        OOM (`metrics.oom_status` files that under `STATUS_OOM` instead) and from
        a refusal (a setting declined before it tried, not one that crashed).
        """
        return str(self.payload.get("status") or "").startswith(RUN_FAILED_PREFIX)

    @property
    def graded_here(self) -> bool:
        """Whether this artifact holds the probe checks this matrix is made of.

        A cell is a (framework, model) pair, but an artifact is a run, and a later
        phase produces runs of the same pair that carry no probe. Ranking on
        recency alone would let a timing result take over the cell that a probe
        answered, and the cell would read as if the probe had produced nothing.
        """
        return self.produced_result and bool(checks_of(self))

    @property
    def purpose(self) -> str:
        """What the run was for, from its own resolved config."""
        run = (self.payload.get("config") or {}).get("run") or {}
        return str(run.get("purpose") or "unknown")

    @property
    def measuring(self) -> bool:
        return self.purpose in MEASURING_PURPOSES

    @property
    def metrics(self) -> dict[str, Any] | None:
        """`metrics.summarise`'s output, as `build_record(**extra)` carried it.

        None means the run reported no figures. For a measuring purpose that is a
        finding and not a blank: the pod-hour was spent and nothing came back.
        """
        value = self.payload.get("metrics")
        return value if isinstance(value, dict) else None

    @property
    def _tail(self) -> tuple[str, ...]:
        """Path segments below `results/{framework}/{model}/`.

        Anchored on the model directory named by the artifact's own config, so a
        `--results` directory that is the repo root and one that is already the
        `results/` subtree both resolve to the same pod. The baseline anchors
        correctly too — it is filed under *its own* combination on every pod,
        which is the whole reason it needs filing by pod instead.
        """
        parts = self.path.parts
        for index in range(len(parts) - 1, 0, -1):
            if parts[index - 1] == self.model:
                return parts[index:]
        return ()

    @property
    def pod(self) -> str:
        """Which pod produced this — the unit a baseline deviation is charged to.

        `host.runpod_pod_id` first: `record.host_spec` and `publish_result` both
        write it, and it is the pod's own answer rather than an inference from
        where the file was filed.
        """
        recorded = (self.payload.get("host") or {}).get("runpod_pod_id")
        if recorded:
            return str(recorded)
        tail = self._tail
        return tail[0] if tail else "unknown"

    @property
    def label(self) -> str | None:
        """The setting this artifact is one of, or None for a single-run pod.

        A sweep publishes one directory per setting under the pod's own, so the
        segment between the pod and the file names the run.
        """
        tail = self._tail
        return tail[1] if len(tail) > 2 else None

    @property
    def is_baseline(self) -> bool:
        return (self.label or "").startswith(BASELINE_DIR_PREFIX)

    @property
    def run_name(self) -> str:
        return self.label or str(self.payload.get("experiment") or "(단일 런)")


def _combination(payload: dict[str, Any]) -> tuple[str, str]:
    probe = payload.get("probe") or {}
    config = payload.get("config") or {}
    framework = probe.get("framework") or config.get("framework", {}).get("name") or "unknown"
    model = probe.get("model") or config.get("model", {}).get("name") or "unknown"
    return framework, model


def recorded_identity(payload: dict[str, Any]) -> float | None:
    """The clock the run itself read, or None if the record does not carry one.

    Never `path.stat().st_mtime`. A clean clone downloads every artifact in one
    go, so the file clock is the download and not campaign order: with the forty
    artifacts in the results repo carrying no `recorded_at`, eight of eighteen
    cells selected the previous campaign's file.
    """
    value = payload.get(IDENTITY_FIELD)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def load_artifacts(results_dir: Path) -> tuple[list[Artifact], list[str]]:
    """Every readable artifact, plus the files that could not be read.

    An unparseable file is reported and stepped over. A pod that vanished
    mid-upload must not take the other seventeen results down with it. A file
    that cannot say when it was recorded is refused the same way — every producer
    here stamps it (`record.build_record`, `publish_result.provenance`), so one
    without it is not this campaign's and dating it by its own mtime is a guess.
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
        timestamp = recorded_identity(payload)
        if timestamp is None:
            skipped.append(
                f"{shown}: no `{IDENTITY_FIELD}`; the file's own clock is the download "
                "time in a clean clone, not the campaign this artifact belongs to"
            )
            continue
        framework, model = _combination(payload)
        artifacts.append(
            Artifact(
                path=shown,
                payload=payload,
                kind="result" if path.name == RESULT_NAME else "started",
                framework=framework,
                model=model,
                timestamp=timestamp,
            )
        )
    return artifacts, skipped


@dataclass(frozen=True)
class Lanes:
    """The three kinds of artifact, separated before anything is ranked."""

    matrix: list[Artifact]
    measured: list[Artifact]
    baselines: list[Artifact]


def split_lanes(artifacts: list[Artifact]) -> Lanes:
    """Sort artifacts by the question they can answer.

    Baselines leave the matrix first. Every pod runs the same canonical workload,
    whose config names `native x qwen3_5_0_8b`, so left in they all pile onto that
    one cell, lose to the probe that graded it, and are reported as duplicates —
    which is how a per-pod control run disappears from a per-cell table.

    Measuring runs leave next. They carry no probe checks, so the matrix could
    only ever render them as "launched, produced nothing".
    """
    matrix, measured, baselines = [], [], []
    for artifact in artifacts:
        if artifact.is_baseline:
            baselines.append(artifact)
        elif artifact.measuring:
            measured.append(artifact)
        else:
            matrix.append(artifact)
    return Lanes(matrix=matrix, measured=measured, baselines=baselines)


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
        # An artifact carrying probe checks outranks one that does not, a real
        # result outranks a bare `started`, then newest wins.
        group.sort(
            key=lambda a: (a.graded_here, a.produced_result, a.kind == "result", a.timestamp),
            reverse=True,
        )
        chosen[key] = group[0]
        for superseded in group[1:]:
            duplicates.append(f"{key[0]} x {key[1]}: ignored {superseded.path}")
    return chosen, duplicates


def load_ledger(path: Path | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """What the orchestrator says it launched, grouped by combination.

    Every entry is kept, not one per combination. Manifests are per pod and a
    combination has more than one: a phase0 probe and a phase2 sweep both name
    native x gemma4_e2b. Keeping whichever the dict happened to write last threw
    away three of twenty-one entries and handed three cells to a phase2 entry that
    was skipped for having no entry point — so a probe pod that launched and
    uploaded nothing read as a pod that never launched, which is the exact
    distinction the module docstring exists to preserve, inverted.
    """
    if path is None or not path.exists():
        return {}
    ledger = json.loads(path.read_text())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in ledger.get("experiments", []):
        grouped.setdefault((entry["framework"], entry["model"]), []).append(entry)
    return grouped


def launch_state(entries: list[dict[str, Any]] | None) -> str:
    """What the ledger says about a cell, across every experiment naming it.

    Ordered by how much a reader can conclude: a pod that started is a spent
    pod-hour whatever else happened on that combination, and only a cell no
    manifest ever reached is untried.
    """
    if not entries:
        return NOT_ATTEMPTED
    if any(entry.get("pod_id") for entry in entries):
        return NO_RESULT
    if any(entry.get("launch_error") for entry in entries):
        return LAUNCH_FAILED
    return NOT_ATTEMPTED


def checks_of(artifact: Artifact | None) -> list[dict[str, Any]]:
    if artifact is None:
        return []
    return (artifact.payload.get("probe") or {}).get("checks") or []


def training_verdict(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Whether this record is a measurement of training, and why not when it is not.

    The rule is `trainbench.metrics.validity.training_verdict` and nothing here.
    This file used to carry a second implementation of it, so tightening the
    library left the speed table ranking runs the library refuses — and loosening
    it left the table dropping runs the library passes — with no test failing
    either way.

    What is left is the extraction: the rule needs `peft.mode` and the device, and
    neither is in `metrics`. A record that does not carry `peft.mode` is refused
    rather than checked without it, for the reason the library gives for an absent
    `total_params` — skipping the comparison is how a full finetune that froze
    most of its tensors gets published as a full finetune.
    """
    mode = ((payload.get("config") or {}).get("peft") or {}).get("mode")
    if not isinstance(mode, str) or not mode:
        return False, [
            f"`config.peft.mode`={mode!r}: nothing says how many parameter tensors "
            "this run was supposed to train, so the counts cannot be judged"
        ]
    return validity.training_verdict(
        payload.get("metrics"),
        peft_mode=mode,
        device=str(payload.get("device", "")),
    )


def stack_of(artifact: Artifact) -> tuple[str, str] | None:
    """The two packages that differ across the six framework images, or None.

    Measured: native / sentence_transformers / tevatron / axolotl are on
    transformers 5.14.1 + torch 2.13.0, ms_swift on 5.12.1, unsloth on 5.5.0 +
    torch 2.11.0, and the uv conflict that produced that split is documented as
    unresolvable. A record that cannot name its stack is not ranked against one
    that can.
    """
    packages = artifact.payload.get("packages") or {}
    torch_version = packages.get("torch")
    transformers_version = packages.get("transformers")
    if not torch_version or not transformers_version:
        return None
    return str(torch_version), str(transformers_version)


def stack_label(key: tuple[str, str] | None) -> str:
    if key is None:
        return STACK_UNKNOWN
    return f"torch {key[0]} + transformers {key[1]}"


def owned_axes(artifact: Artifact) -> list[dict[str, Any]]:
    """The axes this run's framework took over, as the record itself named them.

    Read through `applied.framework_owned` rather than by matching a state
    string: the vocabulary belongs to the `applied-axes` boundary and this file
    renders whatever it finds there.
    """
    applied = artifact.payload.get("applied")
    if not isinstance(applied, dict):
        return []
    names = applied.get(FRAMEWORK_OWNED_KEY)
    if not isinstance(names, list):
        return []
    by_axis = {a.get("axis"): a for a in applied.get("axes") or [] if isinstance(a, dict)}
    return [by_axis.get(name) or {"axis": name} for name in names]


@dataclass(frozen=True)
class PodVerdict:
    """Whether one pod's numbers may be compared with the other pods'."""

    pod: str
    value: float | None
    deviation: float | None
    status: str
    note: str

    @property
    def usable(self) -> bool:
        return self.status == POD_OK


# A pod nothing is known about is not a pod that passed. Used where a verdict is
# looked up for an artifact whose pod never reached the gate.
_UNKNOWN_POD = PodVerdict("unknown", None, None, POD_NO_BASELINE, "판정 기록이 없는 파드")


def _baseline_value(artifact: Artifact) -> tuple[float | None, str]:
    metrics = artifact.metrics
    if metrics is None:
        return None, f"baseline 레코드에 `metrics`가 없다 ({artifact.path})"
    raw = metrics.get(BASELINE_METRIC)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
        return None, f"baseline의 `{BASELINE_METRIC}`가 양수가 아니다: {raw!r}"
    if metrics.get("profiled"):
        # AGENTS.md: a profiled step is inflated by an unmeasured amount, so it
        # cannot stand in for the host's speed.
        return None, "baseline이 프로파일러를 켠 채 측정됐다 — 호스트 속도의 근거가 못 된다"
    return float(raw), ""


@dataclass(frozen=True)
class Tolerance:
    """The deviation threshold pod verdicts were decided with, and where it came from."""

    value: float
    status: str
    notes: list[str]


def declared_tolerance(baselines: list[Artifact]) -> Tolerance:
    """The threshold the baseline records themselves declare.

    The first GPU pod is supposed to derive this number from the noise floor and
    record it. Until this read it, deriving it changed nothing: the verdicts came
    from a constant in this file, so a campaign run at a measured 8.1% would have
    been judged at the 3% it was run to replace, with `calibrated` printed in the
    record beside it.

    Disagreement is a finding rather than an average. Records naming different
    thresholds are not one campaign, so the strictest decides and the spread is
    printed — admitting a pod under a threshold the other records never declared
    is the direction that publishes an incomparable number.
    """
    declared: dict[float, set[str]] = {}
    for artifact in baselines:
        measurement = (artifact.metrics or {}).get("measurement")
        if not isinstance(measurement, dict):
            continue
        value = measurement.get(TOLERANCE_FIELD)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            continue
        status = measurement.get(TOLERANCE_STATUS_FIELD)
        declared.setdefault(float(value), set()).add(
            status if isinstance(status, str) and status else TOLERANCE_UNDECLARED
        )
    if not declared:
        return Tolerance(BASELINE_DEVIATION_LIMIT, TOLERANCE_UNDECLARED, [])
    value = min(declared)
    notes = []
    if len(declared) > 1:
        spread = ", ".join(f"{v:g}" for v in sorted(declared))
        notes.append(
            f"baseline 레코드들이 서로 다른 `measurement.{TOLERANCE_FIELD}`를 싣고 있다 "
            f"({spread}) — 가장 엄격한 {value:g}로 판정했다. 한 캠페인이 하나의 임계값을 "
            "쓰지 않았다는 뜻이고, 그 자체가 결과다"
        )
    return Tolerance(value, "/".join(sorted(declared[value])), notes)


def baseline_gate(
    baselines: list[Artifact], measured: list[Artifact], tolerance: Tolerance | None = None
) -> tuple[dict[str, PodVerdict], float | None, list[str]]:
    """Per-pod verdicts, the reference they were compared against, and warnings.

    The reference is the **lower median** of the pods' baseline figures. Two
    alternatives were rejected: the first pod makes the whole campaign depend on
    launch order and lets a single slow host invalidate every other pod, and a
    mean is dragged by the outlier it is supposed to expose. `median_low` also
    keeps the reference a number some pod actually measured, which is the same
    reason `metrics.percentile` refuses to interpolate.

    A pod that produced measurements but no baseline is not silently fine. It is
    the case this gate exists for — nothing says whether its host was comparable —
    so it gets its own verdict rather than an absence.
    """
    tolerance = tolerance or declared_tolerance(baselines)
    newest: dict[str, Artifact] = {}
    notes: list[str] = list(tolerance.notes)
    for artifact in sorted(baselines, key=lambda a: (a.produced_result, a.timestamp)):
        if (previous := newest.get(artifact.pod)) is not None:
            notes.append(f"{artifact.pod}: baseline이 둘 이상 — {previous.path}를 무시했다")
        newest[artifact.pod] = artifact

    values: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for pod, artifact in newest.items():
        value, reason = _baseline_value(artifact)
        if value is None:
            reasons[pod] = reason
        else:
            values[pod] = value

    reference = statistics.median_low(values.values()) if values else None
    verdicts: dict[str, PodVerdict] = {}
    for pod in sorted(newest):
        if pod not in values:
            verdicts[pod] = PodVerdict(pod, None, None, POD_UNJUDGED, reasons[pod])
            continue
        value = values[pod]
        deviation = abs(value - reference) / reference
        over = deviation > tolerance.value
        verdicts[pod] = PodVerdict(
            pod,
            value,
            deviation,
            POD_INVALID if over else POD_OK,
            "임계값 초과 — 이 파드의 결과는 다른 파드와 같은 표에 들어갈 수 없다" if over else "",
        )

    for artifact in measured:
        if artifact.pod in verdicts:
            continue
        verdicts[artifact.pod] = PodVerdict(
            artifact.pod,
            None,
            None,
            POD_NO_BASELINE,
            "측정 결과는 있는데 canonical baseline 레코드가 없다 — 호스트 비교 근거가 없다",
        )

    if len(values) == 1:
        pod = next(iter(values))
        notes.append(
            f"baseline을 낸 파드가 {pod} 하나뿐이라 기준값이 곧 자기 자신이다. "
            "편차 0%는 측정된 일치가 아니라 비교 대상의 부재다"
        )
    if not values and (baselines or measured):
        notes.append("어느 파드도 쓸 수 있는 baseline 수치를 내지 않아 편차를 계산하지 못했다")
    return verdicts, reference, notes


def _measured_cell(
    reported: list[Artifact],
    oom: list[Artifact],
    refused: list[Artifact],
    failed: list[Artifact],
    verdicts: dict[str, PodVerdict],
) -> str:
    """What a cell says when measuring ran on it but no probe ever did.

    Loading is not in question once a run trained on the combination, so the cell
    does not read as untried. The pod's standing and the training-validity gate
    are counted rather than folded in: a cell reading `측정 3건` while all three
    were frozen graphs is the same collapse as reading them as never attempted.

    `refused` and `failed` are counted the same way `oom` is: each is a spent
    pod-hour that answered something about the setting, and folding either into
    `reported`'s count would claim a figure that was never produced.
    """
    parts = []
    if reported:
        invalid = sum(1 for a in reported if not verdicts.get(a.pod, _UNKNOWN_POD).usable)
        untrained = sum(1 for a in reported if not training_verdict(a.payload)[0])
        suffix = f", 파드 판정 미통과 {invalid}건" if invalid else ""
        suffix += f", 학습 미확인 {untrained}건" if untrained else ""
        parts.append(f"측정 {len(reported)}건(probe 없음{suffix})")
    if oom:
        parts.append(f"{OOM} {len(oom)}건")
    if refused:
        parts.append(f"{AXIS_REFUSED} {len(refused)}건")
    if failed:
        parts.append(f"{RUN_FAILED} {len(failed)}건")
    return ", ".join(parts)


def _status_detail(artifact: Artifact) -> str:
    """The parenthesised `(stage, ErrorType)` / `(ErrorType)` a status string
    carries, or the whole status if it carries no such detail."""
    status = str(artifact.payload.get("status") or "")
    match = _STATUS_DETAIL_RE.search(status)
    return match.group(1) if match else status


def _status_cell(artifact: Artifact) -> str | None:
    """A distinct label for a refused or failed record with no probe checks.

    `scripts/bench.py::refusal_record`'s own docstring says this file prints
    `status` verbatim for such a record; until this existed nothing did — the
    record has no `metrics` and no `checks`, so it fell through to the generic
    `NO_RESULT`, indistinguishable from a pod that vanished with nothing at all.
    Returns None for anything else, so the caller's existing `NO_RESULT` stands.
    """
    if artifact.refused:
        return f"{AXIS_REFUSED} ({_status_detail(artifact)})"
    if artifact.failed:
        return f"{RUN_FAILED} ({_status_detail(artifact)})"
    return None


def cell(
    artifact: Artifact | None,
    launched: list[dict[str, Any]] | None,
    measured: list[Artifact] | None = None,
    verdicts: dict[str, PodVerdict] | None = None,
) -> str:
    """One matrix cell. Absent stays '미시도' — never inferred from a neighbour."""
    verdicts = verdicts or {}
    # Only a run that reported figures says the combination trained. A measuring
    # run that produced none is a spent pod-hour, and the ledger's "launched,
    # produced nothing" is the honest answer for the cell. OOM, an axis refusal
    # and a diagnosable failure are the three exceptions: each reports no
    # figures and is still its own answer about the setting, not one collapsed
    # answer.
    oom = [a for a in measured or [] if a.oom]
    axis_refused = [a for a in measured or [] if a.refused]
    run_failed = [a for a in measured or [] if a.failed]
    reported = [a for a in measured or [] if a.metrics and not a.oom]
    if (reported or oom or axis_refused or run_failed) and (
        artifact is None or not artifact.produced_result or not checks_of(artifact)
    ):
        return _measured_cell(reported, oom, axis_refused, run_failed, verdicts)
    if artifact is None:
        return launch_state(launched)
    if not artifact.produced_result:
        return _status_cell(artifact) or NO_RESULT
    checks = checks_of(artifact)
    if not checks:
        return _status_cell(artifact) or NO_RESULT
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


def _number(value: Any, spec: str) -> str:
    """A figure, or '-' where there is none. Never a zero standing in for absence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return format(value, spec)


def _gib(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return format(value / 1024**3, ".2f")


def _verdict_cell(verdict: PodVerdict) -> str:
    if verdict.deviation is None:
        return verdict.status
    return f"{verdict.status} (편차 {verdict.deviation * 100:.2f}%)"


# Which config field carries each axis's requested value, and which packages in
# `record.package_versions` can silently change what that request actually ran
# as. Keyed by the axis name so the caption and the table share one vocabulary.
#
#   attn    — `kernels` present without `flash-attn` makes transformers rewrite
#             an `attn=fa2/3/4` request onto a Hub kernel instead of refusing it
#             (`FLASH_ATTN_KERNEL_FALLBACK`, transformers/modeling_flash_attention_
#             utils.py). Two runs both labelled `fa2` can be measuring different
#             kernels depending only on whether `kernels` is installed.
#   compile — `triton` is what `compile.mode` other than `none` actually compiles
#             through (torch inductor's CUDA backend).
#   kernel  — Gated DeltaNet's fused path needs both `fla-core` (the distribution
#             that ships `fla.ops`/`fla.modules`) and `causal-conv1d`
#             (`trainbench/axes.py::_fla_fast_path`); `flash-linear-attention` is
#             the wrapper that pins the first, carried alongside it rather than
#             instead of it (`trainbench/record.py::_TRACKED_PACKAGES`).
AXIS_CONFOUND = {
    "attn": (("attn", "name"), ("flash-attn", "kernels")),
    "compile": (("compile", "mode"), ("triton",)),
    "kernel": (("kernel", "name"), ("flash-linear-attention", "fla-core", "causal-conv1d")),
}


def _config_axis(config: dict[str, Any], group: str, field: str) -> str:
    value = ((config.get(group) or {}).get(field)) if isinstance(config, dict) else None
    return str(value) if value else "-"


def _axis_confound_header() -> list[str]:
    """One column per axis value, followed by that axis's own confound packages.

    Built from `AXIS_CONFOUND` rather than spelled out a second time: a package
    added there without a matching column here is a package this table cannot
    have been checked against.
    """
    columns = []
    for axis, (_, packages) in AXIS_CONFOUND.items():
        columns.append(axis)
        columns.extend(packages)
    return columns


def _axis_confound_row(artifact: Artifact) -> list[str]:
    config = artifact.payload.get("config") or {}
    packages = artifact.payload.get("packages") or {}
    cells = []
    for _axis, (field, package_names) in AXIS_CONFOUND.items():
        cells.append(_config_axis(config, *field))
        cells.extend(packages.get(name, "-") for name in package_names)
    return cells


def _axis_confound_table(rows: list[Artifact]) -> list[str]:
    """The axis value beside the package(s) that can substitute what it asked for.

    Rendered per stack, immediately under that stack's figures — not as a version
    dump at the end of the document — so a reader checking one run's number reads
    the confound in the same glance rather than cross-referencing a separate table.
    """
    header = ["런", "파드", *_axis_confound_header()]
    out = [
        "",
        "축을 흔드는 패키지: 같은 스택 안에서도 이 값들에 따라 요청한 축이 실제로 무엇으로 "
        "실행됐는지가 갈린다.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for artifact in rows:
        cells = [artifact.run_name, artifact.pod, *_axis_confound_row(artifact)]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _figure_table(rows: list[Artifact], verdicts: dict[str, PodVerdict]) -> list[str]:
    out = [
        "",
        "| 런 | 파드 | 프레임워크 x 모델 | 목적 | step p50 (s) | p95 (s) | mean (s) "
        "| samples/s | tokens/s | peak mem (GiB) | steps 계측/폐기/측정 | 파드 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for artifact in rows:
        metrics = artifact.metrics or {}
        verdict = verdicts.get(artifact.pod, _UNKNOWN_POD)
        out.append(
            f"| {artifact.run_name} | {artifact.pod} | "
            f"{artifact.framework} x {artifact.model} | {artifact.purpose} | "
            f"{_number(metrics.get('step_seconds_p50'), '.4f')} | "
            f"{_number(metrics.get('step_seconds_p95'), '.4f')} | "
            f"{_number(metrics.get('step_seconds_mean'), '.4f')} | "
            f"{_number(metrics.get('samples_per_second'), '.2f')} | "
            f"{_number(metrics.get('tokens_per_second'), '.1f')} | "
            f"{_gib(metrics.get('peak_memory_bytes'))} | "
            f"{metrics.get('steps_timed', '-')}/{metrics.get('steps_discarded', '-')}"
            f"/{metrics.get('steps_measured', '-')} | "
            f"{_verdict_cell(verdict)} |"
        )
    return out


def _counts_table(rows: list[Artifact]) -> list[str]:
    out = [
        "",
        "| 런 | 파드 | rows/step | padded tokens/step | tokens/step | images/step "
        "| images dropped/step | images dropped 합계 | MFU |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for artifact in rows:
        metrics = artifact.metrics or {}
        out.append(
            f"| {artifact.run_name} | {artifact.pod} | "
            f"{_number(metrics.get('rows_per_step'), '.1f')} | "
            f"{_number(metrics.get('padded_tokens_per_step'), '.1f')} | "
            f"{_number(metrics.get('tokens_per_step'), '.1f')} | "
            f"{_number(metrics.get('images_per_step'), '.2f')} | "
            f"{_number(metrics.get('images_dropped_per_step'), '.2f')} | "
            f"{metrics.get('images_dropped_total', '-')} | "
            f"{metrics.get('mfu') if metrics.get('mfu') is not None else '없음'} |"
        )
    reasons = sorted(
        {str(a.metrics.get("mfu_reason")) for a in rows if (a.metrics or {}).get("mfu") is None}
        - {"None"}
    )
    return out + [f"\nMFU가 비어 있는 이유: {reason}" for reason in reasons]


def by_stack(rows: list[Artifact]) -> list[tuple[tuple[str, str] | None, list[Artifact]]]:
    """The runs grouped by the stack they were measured on, unknown stacks last."""
    grouped: dict[tuple[str, str] | None, list[Artifact]] = {}
    for artifact in rows:
        grouped.setdefault(stack_of(artifact), []).append(artifact)
    return sorted(grouped.items(), key=lambda item: (item[0] is None, item[0] or ()))


def _ranked_by_stack(
    rows: list[Artifact], verdicts: dict[str, PodVerdict], tables: bool = True
) -> list[str]:
    """One ranking per stack, never one ranking across stacks.

    The six images cannot be unified (transformers 5.14.1 / 5.12.1 / 5.5.0, torch
    2.13.0 / 2.12.1 / 2.11.0) and the uv conflict behind that is documented as
    unresolvable. A single table puts the image in the ranking and presents it as
    a result; the heading between the tables is what stops a reader comparing
    across them without meeting the stack first.
    """
    lines: list[str] = []
    for key, group in by_stack(rows):
        lines += ["", f"##### 해석 스택: {stack_label(key)}"]
        if key is None:
            lines += [
                "",
                "이 런들은 `packages.torch`/`packages.transformers`를 싣지 않아 어느 스택에서 "
                "잰 것인지 레코드가 답하지 못한다. 다른 스택과 같은 순위표에 넣지 않는다.",
            ]
        lines += _figure_table(group, verdicts)
        lines += _axis_confound_table(group)
        if tables:
            lines += _counts_table(group)
    return lines


def render_measurements(measured: list[Artifact], verdicts: dict[str, PodVerdict]) -> list[str]:
    """The figures a measuring run produced, and what is missing from the ones that did not.

    Four things leave the comparable ranking, each because putting it in makes a
    number mean something it does not: a profiled step (inflated by an unmeasured
    amount), a run that did not train, an OOM, and a run on another stack.
    """
    if not measured:
        return []
    ordered = sorted(measured, key=lambda a: (a.pod, a.run_name, a.path))
    oom = [a for a in ordered if a.oom]
    rest = [a for a in ordered if not a.oom]
    figured = [a for a in rest if a.metrics]
    barren = [a for a in rest if not a.metrics]
    untrained = [(a, training_verdict(a.payload)[1]) for a in figured]
    untrained = [(a, why) for a, why in untrained if why]
    trained = [a for a in figured if not training_verdict(a.payload)[1]]
    timed = [a for a in trained if not a.metrics.get("profiled")]
    profiled = [a for a in trained if a.metrics.get("profiled")]

    lines = [
        "",
        "### 측정 결과",
        "",
        f"측정 목적(`{'`/`'.join(sorted(MEASURING_PURPOSES))}`) 런 {len(ordered)}건 중 "
        f"수치를 낸 것 {len(figured)}건, `{NO_METRICS}` {len(barren)}건, "
        f"`{OOM}` {len(oom)}건, `{NOT_TRAINED}` {len(untrained)}건. "
        "각 수치가 무엇을 센 것인지는 아래 '지표 정의'에 있다.",
    ]

    if timed:
        lines += _ranked_by_stack(timed, verdicts)

    dropped = [a for a in trained if (a.metrics or {}).get("images_dropped_total")]
    if dropped:
        lines += [
            "",
            "#### 이미지를 텍스트로만 읽은 런",
            "",
            "`images_dropped`가 0이 아니다. 프로세서가 받지 못한 이미지가 있었다는 뜻이므로, "
            "이 런들은 이미지 코퍼스를 **텍스트 전용 뷰로** 읽었다. 같은 표의 다른 런과 "
            "같은 작업량을 잰 것이 아니다.",
            "",
        ]
        for artifact in dropped:
            metrics = artifact.metrics or {}
            lines.append(
                f"- **{artifact.run_name}** ({artifact.pod}, {artifact.model}) — "
                f"버려진 이미지 {metrics.get('images_dropped_total')}장, "
                f"처리된 이미지 {metrics.get('images_read_total', '-')}장"
            )

    if profiled:
        lines += [
            "",
            "#### 프로파일러가 켜진 런 — 비교표에 인용하지 않는다",
            "",
            "AGENTS.md: 프로파일러를 켜고 잰 수치는 보고하지 않는다. 프로파일러가 이 저장소에서 "
            "iteration time을 얼마나 부풀리는지는 **측정 안 함**(docs/methodology.md)이므로 "
            "보정도 불가능하다. 아래는 커널 분해용 기록이며 위 표의 수치와 같은 축에 놓을 수 없다.",
        ]
        lines += _ranked_by_stack(profiled, verdicts, tables=False)

    if untrained:
        lines += [
            "",
            f"#### {NOT_TRAINED} — 속도 결과로 인용하지 않는다",
            "",
            "`grad_norm`·`trainable_params`·loss가 이 런이 학습했다고 말하지 않는다. 전 "
            "파라미터가 얼어 있어도 임베딩 출력을 통해 backward는 정상 종료하므로, 이런 런의 "
            "step 시간은 학습이 아니라 forward 한 번의 비용이다. **수치를 렌더하지 않는다.**",
            "",
        ]
        for artifact, why in untrained:
            lines.append(
                f"- **{artifact.run_name}** ({artifact.pod}, {artifact.framework} x "
                f"{artifact.model}) — {'; '.join(why)}"
            )

    if oom:
        lines += [
            "",
            f"#### {OOM} — 결과의 한 범주이지 결과의 부재가 아니다",
            "",
            "메모리 상한에 닿아 끝난 런이다. 이 설정이 이 GPU에 들어가지 않는다는 답이므로, "
            "`미시도`로도 `미지원`으로도 렌더하지 않는다.",
            "",
        ]
        for artifact in oom:
            lines.append(
                f"- **{artifact.run_name}** ({artifact.pod}, {artifact.framework} x "
                f"{artifact.model}, {artifact.purpose}) — `{artifact.path}`"
            )

    if barren:
        lines += [
            "",
            f"#### {NO_METRICS} — 측정 목적으로 돌았으나 수치가 없다",
            "",
            "파드 시간은 썼는데 `metrics`가 레코드에 없다. 실패한 런이거나 결과가 잘린 런이며, "
            "어느 쪽이든 이 조합의 수치는 아직 존재하지 않는다.",
            "",
        ]
        for artifact in barren:
            status = artifact.payload.get("status") or (
                "기동 기록만 있다"
                if artifact.kind == "started"
                else "결과 파일은 올라왔는데 `metrics` 키가 없다"
            )
            lines.append(
                f"- **{artifact.run_name}** ({artifact.pod}, {artifact.framework} x "
                f"{artifact.model}, {artifact.purpose}) — {status}, `{artifact.path}`"
            )
    return lines


def _tolerance_source(tolerance: Tolerance) -> str:
    """Which provenance sentence the threshold gets.

    Only a record that says `calibrated` earns the calibrated wording; an
    undeclared threshold and one declared `uncalibrated` are the same claim.
    """
    if tolerance.status == TOLERANCE_CALIBRATED:
        return BASELINE_DEVIATION_CALIBRATED_SOURCE
    return BASELINE_DEVIATION_SOURCE


def render_baseline_gate(
    verdicts: dict[str, PodVerdict],
    reference: float | None,
    notes: list[str],
    tolerance: Tolerance,
) -> list[str]:
    """The cross-pod deviation gate, with the threshold's provenance attached."""
    if not verdicts:
        return []
    shown = (
        f"{reference:.4f}s"
        if reference is not None
        else "없음 — 계산할 수 있는 baseline 수치가 하나도 없다"
    )
    lines = [
        "",
        "### 파드 baseline 편차 게이트",
        "",
        f"모든 측정 파드가 canonical baseline 1개를 돌리고, `{BASELINE_METRIC}`를 파드끼리 "
        f"비교한다. 기준값은 파드별 값의 **하위 중앙값**이다 — 첫 파드를 기준으로 삼으면 기동 "
        "순서가 판정을 정하고, 평균을 쓰면 드러내야 할 이상치가 기준값을 끌고 간다. 중앙값은 "
        "어느 파드가 실제로 낸 값이기도 하다(보간하지 않는다).",
        "",
        f"임계값 {tolerance.value * 100:.2f}% — {_tolerance_source(tolerance)}",
        "",
        f"기준값: {shown}",
        "",
        "| 파드 | baseline step p50 (s) | 편차 | 판정 | 비고 |",
        "|---|---|---|---|---|",
    ]
    for pod in sorted(verdicts):
        verdict = verdicts[pod]
        deviation = "-" if verdict.deviation is None else f"{verdict.deviation * 100:.2f}%"
        lines.append(
            f"| {pod} | {_number(verdict.value, '.4f')} | {deviation} | "
            f"{verdict.status} | {verdict.note or '-'} |"
        )
    unusable = [v for v in verdicts.values() if not v.usable]
    if unusable:
        lines += [
            "",
            "**위 판정을 통과하지 못한 파드의 수치는 다른 파드의 수치와 같은 비교에 넣을 수 "
            "없다.** 재실행하거나 폐기한다. 측정 결과 표의 '파드 판정' 열이 어느 행이 여기 "
            "해당하는지를 표시한다.",
            "",
        ]
        lines += [f"- {v.pod}: {v.status} — {v.note or '-'}" for v in unusable]
    if notes:
        lines += ["", *[f"- {note}" for note in notes]]
    return lines


def render_metric_definitions(measured: list[Artifact]) -> list[str]:
    """The definitions the records carried, verbatim.

    Read out of the results rather than imported from `trainbench.metrics`: the
    definition that belongs next to a number is the one that travelled with it.
    A record written before a definition changed would otherwise be re-labelled
    by whatever the current code says, which is how a number gets misread later.
    """
    carried: dict[str, dict[str, list[str]]] = {}
    missing = []
    for artifact in sorted(measured, key=lambda a: (a.pod, a.run_name)):
        definitions = (artifact.metrics or {}).get("metric_definitions")
        if not isinstance(definitions, dict) or not definitions:
            if artifact.metrics is not None:
                missing.append(artifact)
            continue
        for name, text in definitions.items():
            carried.setdefault(str(name), {}).setdefault(str(text), []).append(artifact.run_name)
    if not carried and not missing:
        return []
    lines = [
        "",
        "### 지표 정의",
        "",
        "레코드가 싣고 온 정의 그대로다. 정의 없는 숫자는 나중에 오독된다 — 특히 `tokens`는 "
        "패딩을 세지 않고 모델별 상수(`instruction_prompt`)를 포함하며, `rows`는 `samples`의 "
        "2배다.",
        "",
    ]
    for name in sorted(carried):
        texts = carried[name]
        if len(texts) == 1:
            lines.append(f"- **{name}** — {next(iter(texts))}")
            continue
        lines.append(f"- **{name}** — 런마다 정의가 다르다. 이 이름의 수치는 서로 비교할 수 없다:")
        for text, runs in sorted(texts.items()):
            lines.append(f"  - `{', '.join(sorted(set(runs)))}`: {text}")
    if missing:
        lines += [
            "",
            "정의를 싣지 않은 런(수치의 의미를 이 문서만으로는 확정할 수 없다): "
            + ", ".join(f"`{a.run_name}` ({a.pod})" for a in missing),
        ]
    return lines


def render_owned_axes(artifacts: list[Artifact]) -> list[str]:
    """The axes a run handed to its framework, so a cell that lost one says so.

    Measuring the framework's own training step is what makes the comparison a
    comparison of frameworks (PLAN.md 결정 5), and its price is that the ablation
    grid is ragged: on a tevatron cell, `loss.name` and
    `parallel.cross_device_negatives` are not this harness's to apply. A report
    that does not show it renders such a cell exactly like one that ran the axis.
    """
    rows = [(a, owned_axes(a)) for a in sorted(artifacts, key=lambda a: (a.pod, a.run_name))]
    rows = [(a, axes) for a, axes in rows if axes]
    if not rows:
        return []
    lines = [
        "",
        "### 프레임워크 소유 축 — 이 런이 돌린 축이 아니다",
        "",
        "프레임워크가 학습 스텝 안에서 직접 처리해 이 하네스가 적용하지도 되읽지도 못한 축이다. "
        "같은 표의 다른 런과 **같은 ablation 그리드를 돈 것이 아니다.** 순위표가 아니므로 "
        "표로 렌더하지 않는다 — 여기 적힌 것은 수치가 아니라 그 수치가 답하지 않는 축이다.",
        "",
    ]
    for artifact, axes in rows:
        lines.append(
            f"- **{artifact.run_name}** ({artifact.pod}, {artifact.framework} x {artifact.model})"
        )
        for axis in axes:
            reason = (axis.get("detail") or {}).get("reason") or "이유 미기록"
            lines.append(
                f"  - `{axis.get('axis')}` — 상태 `{axis.get('state') or '-'}`, "
                f"소유 {axis.get('owner') or '-'}: {reason}"
            )
    return lines


def render(
    chosen: dict[tuple[str, str], Artifact],
    ledger: dict[tuple[str, str], list[dict[str, Any]]],
    duplicates: list[str],
    skipped: list[str],
    measured: list[Artifact] | None = None,
    baselines: list[Artifact] | None = None,
) -> str:
    measured = list(measured or [])
    baselines = list(baselines or [])
    tolerance = declared_tolerance(baselines)
    verdicts, reference, gate_notes = baseline_gate(baselines, measured, tolerance)
    per_combination: dict[tuple[str, str], list[Artifact]] = {}
    for artifact in measured:
        per_combination.setdefault((artifact.framework, artifact.model), []).append(artifact)

    results = [a for a in chosen.values() if a.produced_result]
    lines = [
        MARKER,
        "",
        GENERATED_HEADING,
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
            cell(
                chosen.get((framework, model)),
                ledger.get((framework, model)),
                per_combination.get((framework, model)),
                verdicts,
            )
            for model in MODELS
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

    lines += render_measurements(measured, verdicts)
    lines += render_baseline_gate(verdicts, reference, gate_notes, tolerance)
    lines += render_owned_axes(measured + baselines + list(chosen.values()))
    lines += render_metric_definitions(measured + baselines)

    if duplicates or skipped:
        lines += ["", "### 병합에서 제외한 파일", ""]
        lines += [f"- 중복: {item}" for item in duplicates]
        lines += [f"- 판독 불가: {item}" for item in skipped]
    return "\n".join(lines) + "\n"


def document_head(existing: str) -> str:
    """The hand-written part of the matrix document, or a refusal to guess.

    Only the generated section is rewritten, so whatever is above the marker
    survives a merge. Two things above it are not survivable, and both end the
    merge rather than producing a document nobody can act on:

    * **A second marker.** Everything from the first one down is replaced, so a
      document with two generated sections loses whatever a human wrote between
      them. Splitting on the first marker and saying nothing did exactly that.
    * **A rival matrix.** A hand-written table under this section's own heading is
      read before the generated one and outranks it by position. The repository
      shipped `OK (7/7)` above and `미시도` below, for the same cell, in the same
      document. Neither table is wrong to exist; a reader cannot tell which is
      current, and this script cannot delete prose it does not own.
    """
    if existing.count(MARKER) > 1:
        raise ValueError(
            f"{existing.count(MARKER)} '{MARKER}' markers; everything from the first "
            "one down is replaced, so a merge would drop what is between them. Leave "
            "one marker"
        )
    head = existing.split(MARKER)[0] if MARKER in existing else existing
    rival = [
        f"line {number}: {line.strip()}"
        for number, line in enumerate(head.splitlines(), start=1)
        if line.startswith("#") and MATRIX_HEADING in line
    ]
    if rival:
        raise ValueError(
            f"a hand-written '{MATRIX_HEADING}' section sits above the marker "
            f"({'; '.join(rival)}). A reader meets it first and it will not agree "
            "with the generated table for long. Fold what is still true into the "
            "surrounding prose, or retitle it to name its own scope and evidence"
        )
    return head.rstrip()


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

    lanes = split_lanes(artifacts)
    chosen, duplicates = newest_per_combination(lanes.matrix)
    for item in duplicates:
        print(f"duplicate {item}", file=sys.stderr)

    # On stderr as well as in the document: whoever runs the merge is the person
    # who can still re-run the pod, and they read a terminal, not a diff.
    verdicts, _, _ = baseline_gate(lanes.baselines, lanes.measured)
    for pod, verdict in sorted(verdicts.items()):
        if not verdict.usable:
            print(f"pod {pod}: {verdict.status} — {verdict.note}", file=sys.stderr)

    generated = render(
        chosen,
        ledger,
        duplicates,
        skipped,
        measured=lanes.measured,
        baselines=lanes.baselines,
    )
    existing = args.matrix.read_text() if args.matrix.exists() else ""
    try:
        head = document_head(existing)
    except ValueError as exc:
        print(f"{args.matrix}: {exc}", file=sys.stderr)
        return 2
    args.matrix.write_text(f"{head}\n\n{generated}")
    print(
        f"merged {len(chosen)} probe, {len(lanes.measured)} measurement and "
        f"{len(lanes.baselines)} baseline artifact(s) into {args.matrix}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
