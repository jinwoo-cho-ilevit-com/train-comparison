# lane-i — 통합

## Scope

루트 문서와 판정 원장은 여러 레인이 건드릴 대상이라 **마지막에 한 번 모은다.** 이 레인은
다른 레인이 전부 병합된 뒤에 돈다.

## Owns

- `AGENTS.md`
- `PLAN.md`
- `docs/CONTRACTS.md`
- `docs/support-matrix.md`
- `README.md`
- `docs/open-verdicts.json`
- `docs/audit-baseline.json`

## 할 일

### 1. `AGENTS.md ## Commands`에 세 줄 추가

지금 Setup / Test / lint / Config-path check만 있다. 저장소를 읽어서는 알 수 없는 것 셋이
빠져 있다:

```markdown
- Run: `infisical run --env=dev -- uv run python scripts/orchestrate.py --experiment 'phase0-*'`
  (`--dry-run` prints the plan and launches nothing)
- Report: `uv run python scripts/report.py --results <dir> --ledger outputs/orchestrate-<phase>.json`
  Results live in the HF results repo, not on disk — download them first. **Pass one campaign's
  artifacts only**: selection falls back to file mtime when `recorded_at` is absent, and a mixed
  directory silently renders cells from an older campaign.
- One setting: `uv run python scripts/bench.py --config <resolved>.json --out result.json`
  It takes a *resolved* config JSON (Hydra's antlr4 pin is incompatible with axolotl), composed by
  `scripts/compose_config.py`. One setting per process — a sweep is the pod re-running this file,
  not a loop inside it.
```

**mtime 경고는 실측이다** — 결과 40건 중 `recorded_at` 0건이고, timestamp를 같게 두면 18칸 중
8칸이 지난 캠페인 것을 고른다. lane-b가 이것을 고치면 경고 문구를 그에 맞게 갱신한다.

Smoke 명령은 넣지 않는다 — `env_report.py`는 training smoke가 아니고(파일이 이미 그렇게 적고
있다) 모델을 적재하는 진짜 스모크는 GPU에서만 된다.

### 2. `PLAN.md`에 신설 파일 등재

`trainbench/collate.py`, `trainbench/metrics/`, `trainbench/loader.py`, `trainbench/kernels.py`,
그리고 신설 테스트. `plan-files`는 양방향이라 트리에 없는 파일도, 파일에 없는 트리도 빨개진다.

→ `infisical run --env=dev -- uv run python scripts/audit_plan.py` — `plan-files` PASS

### 3. `docs/CONTRACTS.md` §2 개정 (결정 1)

`axes.step_context`가 정밀도 컨텍스트의 유일한 자리라는 계약이 axolotl autocast 결정으로
움직인다. 프레임워크가 요구하는 컨텍스트를 그 자리로 끌어오는 형태가 되므로, 계약이 그것을
표현해야 한다. lane-g가 만든 실제 배선을 읽고 쓴다.

### 4. `docs/support-matrix.md` 갱신

- 결정 4를 반영한다 — 스택이 다른 셀을 나란히 놓지 않는다
- 결정 6을 기록한다 — `kernels_hub`를 버린 이유
- 결정 5의 대가를 기록한다 — ablation 그리드가 프레임워크마다 들쭉날쭉하다는 것, 그리고
  tevatron 셀에 `loss`/`cross_device_negatives` 축이 없다는 것
- lane-g의 권장 경로 대조 결과를 기록한다 — 여섯 중 몇이 그 프레임워크의 문서화된 학습
  진입점을 쓰는지

### 5. `docs/open-verdicts.json` 종결 취합

여러 레인이 판정을 닫으므로 한 번에 모은다. 최소한:

- `qwen3-vl-query-prompt-may-go-in-twice` — lane-f가 토큰화로 확정
- `images-carry-a-code-snapshot-nothing-checks-is-current` — lane-b가 probe 갈래 preflight를
  넣으면 앵커가 생긴다

**닫는 것은 리뷰어 행위이고 런을 인용해야 한다.** CPU 증거로 파드가 답할 항목을 닫지 않는다 —
오늘 그렇게 닫으려다 게이트에 잡힌 선례가 있다.

## Completion criteria

- `AGENTS.md`에 Run / Report / one-setting 세 줄이 있고, 각각 저장소를 읽어서는 알 수 없는
  사실을 담는다
- `plan-files` PASS
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- `CONTRACTS.md` §2가 lane-g의 실제 배선과 일치한다
- `verdicts-closed`가 2 open이다 (남는 둘은 파드가 답할 `loss-empty-pixel-slice`와
  `loss-gradcache-memory-and-overhead`)
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- 4 → 2로 줄면 감사가 `shrank`로 BLOCK한다. 그것은 결함이 아니라 래칫이다 —
  `docs/audit-baseline.json`의 `verdicts-closed` count와 note를 **줄어든 상태에 맞게 고쳐야**
  통과한다. note는 남은 둘이 왜 파드 없이 닫힐 수 없는지를 계속 말해야 한다
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py` (exit 0)
- 베이스라인의 `verdicts-closed` note에 적힌 lane-f/lane-b 몫이 실제로 닫혔는지 대조한다.
  두 레인이 앵커를 만들지 못했으면 **닫지 말고 note를 갱신한다** — 원장의 `closed`는
  "합의된 조치가 착지했다"이지 "결함이 사라졌다"가 아니다
- 게이트 넷이 전부 통과한다
  → `uv run ruff check && uv run ruff format --check`
  → `infisical run --env=dev -- uv run pytest`
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
  → `infisical run --env=dev -- uv run python scripts/env_report.py device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4`
- [human] `docs/support-matrix.md`가 스택이 다른 셀을 나란히 놓지 않는다
  → verdict: ____  by: ____  at: ____
- [human] 결정 5의 대가(들쭉날쭉한 그리드)가 표에서 읽힌다
  → verdict: ____  by: ____  at: ____

## Out of scope

- 코드 — 이 레인은 문서와 원장만 만진다
- 파드가 답할 판정 둘 — 닫지 않는다
- `docs/methodology.md` — **lane-e** 소유
