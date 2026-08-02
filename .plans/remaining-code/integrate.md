# integrate — 통합 (wave 3, 단독)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`, 그리고 **모든 `.plans/notes/*.md`**.
> 다른 여덟 레인이 전부 머지된 뒤에 돈다. **한 번만 돈다** — `shrank` 래칫 때문에
> 반복 실행할 수 없다.

## 목표

루트 문서와 판정 원장을 한 번에 모은다. 코드는 쓰지 않는다.

## Owns

```
AGENTS.md
PLAN.md                      산문. 레이아웃 블록은 머지 단계가 이미 등재했다
README.md
docs/CONTRACTS.md
docs/support-matrix.md
docs/open-verdicts.json      구조 전체
tests/test_audit.py
.plans/
```

`docs/audit-baseline.json` 과 `scripts/audit_plan.py` 는 **머지 단계가 이미 처리했다.**
여기서 다시 손대면 래칫을 두 번 움직인다. 상태만 확인한다.

## 입력 — 여덟 레인이 남긴 것

`.plans/notes/` 의 파일들을 전부 읽는다:

| 레인 | 넘긴 것 |
|---|---|
| split | seam 의 이름과 필드, 신설 파일 |
| capture | 파드가 답할 축별 질문 |
| measure | `grad_norm`/`trainable_params` 정의, trackio 스키마 제거분 |
| report | `check_axis_not_split` 설계 판단, `gradcache` 핀 판단, trackio config 제거분 |
| probe | `grad_norm`/`trainable_params` 정의(측정 쪽과 대조됐어야 한다) |
| kernels | methodology 정정 사실, `kernels` 핀 요청 |
| packing | 프롬프트 결착 결과와 시퀀스 길이 변화 |
| adapters | **`CONTRACTS.md` §2 를 쓰기 위한 실제 배선** |
| axes | `kernels_hub` 제거의 남은 지점 목록 |

## 작업 1 — `AGENTS.md ## Commands` 에 세 줄

지금 Setup / Test / lint / Config-path check 넷뿐이다. 빠진 세 사실은 **저장소를 읽어서는
알 수 없는 것**이다:

```markdown
- Run: `infisical run --env=dev -- uv run python scripts/orchestrate.py --experiment 'phase0-*'`
  (`--dry-run` prints the plan and launches nothing)
- Report: `uv run python scripts/report.py --results <dir> --ledger outputs/orchestrate-<phase>.json`
  Results live in the HF results repo, not on disk — download them first. **Pass one campaign's
  artifacts only.**
- One setting: `uv run python scripts/bench.py --config <resolved>.json --out result.json`
  It takes a *resolved* config JSON (Hydra's antlr4 pin is incompatible with axolotl), composed by
  `scripts/compose_config.py`. One setting per process — a sweep is the pod re-running this file,
  not a loop inside it.
```

**Report 줄의 경고 문구는 report 레인이 무엇을 고쳤는지에 따라 바뀐다.**
mtime fallback 이 사라졌으면 경고를 그에 맞게 고쳐 쓴다. `.plans/notes/report.md` 를 읽고 쓴다.

**smoke 명령을 추가하지 않는다** — `env_report.py` 는 학습 smoke 가 아니고
(그 파일이 스스로 그렇게 적고 있다) 진짜 모델 적재 smoke 는 GPU 에서만 된다.

## 작업 2 — `docs/CONTRACTS.md` §2 개정 (결정 1)

"`axes.step_context` 가 precision 컨텍스트의 유일한 집이다"라는 계약이 axolotl autocast 결정
아래로 옮겨간다. "프레임워크가 요구하는 컨텍스트를 그 자리로 끌어온다"가 된다.

**adapters 의 실제 배선을 읽고 쓴다.** `.plans/notes/adapters.md` 와
`trainbench/loader.py` 를 열어 `required_step_context` 가 실제로 어떻게 생겼는지 보고 쓴다.
계획이 그럴 것이라고 적어둔 대로 쓰지 않는다.

## 작업 3 — `docs/support-matrix.md`

- 결정 4: 스택이 다른 칸을 나란히 놓지 않는다
- 결정 6: `kernels_hub` 를 왜 버렸는지
- 결정 5 의 대가: **ablation 그리드가 프레임워크마다 들쭉날쭉해진다.** tevatron 칸에는
  `loss` / `cross_device_negatives` 축이 없다. 표에서 그것이 읽혀야 한다
- adapters 의 진입점 대조표: 여섯 중 몇이 프레임워크의 문서화된 학습 진입점을 쓰는가

## 작업 4 — `docs/open-verdicts.json`

닫힌 것을 확인한다:
- `images-carry-a-code-snapshot-nothing-checks-is-current` — report 레인이 앵커를 만들었다
- `qwen3-vl-query-prompt-may-go-in-twice` — packing 레인이 토큰화로 결착했다

**남는 둘은 남는 것이 옳다**: `loss-empty-pixel-slice-on-a-real-checkpoint-is-unverified`,
`loss-gradcache-memory-and-overhead-are-unmeasured`. 실제 체크포인트와 GPU 없이 답이 안 나온다.

**닫는 것은 저자가 아니라 리뷰어의 행위이고 런을 인용해야 한다.**
CPU 증거로 파드가 답할 항목을 닫지 않는다 — 그렇게 닫으려다 게이트에 잡힌 선례가 있다(`1a7b7c7`).

**교차 확인**: baseline 의 `verdicts-closed` note 에 적힌 packing/report 몫이 **실제로 닫힌 것과
맞는지** 본다. 그 레인들이 앵커를 만들지 못했으면 **닫지 말고 note 를 고친다** —
원장의 `closed` 는 "합의된 조치가 착지했다"이지 "결함이 사라졌다"가 아니다.

앵커가 저장소에 실재하는지 직접 확인한다:
```
git grep -n "test_a_pod_whose_image_predates_the_config_it_is_handed_does_not_measure"
git grep -n "test_the_query_instruction_prompt_appears_once_in_a_templated_row"
```

## 작업 5 — `.plans/` 정리

`.plans/notes/`, `.plans/deps/` 의 내용 중 처리된 것을 정리하고, 처리되지 않은 것은
**왜 남았는지와 함께 남긴다.** `.plans/research/` 는 그대로 둔다 — 다음 캠페인의 입력이다.

## 완료 조건

1. `AGENTS.md` 에 Run/Report/one-setting 세 줄이 있고, 각각이 저장소를 읽어서는 알 수 없는
   사실을 하나씩 싣는다
2. `plan-files` PASS →
   `infisical run --env=dev -- uv run python scripts/audit_plan.py`
3. `CONTRACTS.md` §2 가 adapters 의 **실제 배선**과 맞는다 — 배선을 열어 확인했다
4. `verdicts-closed` 가 **2 open** →
   `infisical run --env=dev -- uv run python scripts/audit_plan.py`
5. **네 게이트 전부 exit 0** — 필터 없이. `--only`/`--skip` 은 wave 게이트가 아니다
   (감사 스크립트가 스스로 `PARTIAL RUN: ... not a wave gate` 라고 찍는다)
6. `pytest tests/contract -q` → **122 passed, 0 xfailed**
7. **[human]** `docs/support-matrix.md` 가 스택이 다른 칸을 나란히 놓지 않는다 →
   verdict: ____ by: ____ at: ____
8. **[human]** 결정 5 의 대가(들쭉날쭉한 그리드)가 표에서 읽힌다 →
   verdict: ____ by: ____ at: ____

## 하지 않는 것

- 어떤 종류의 코드도
- 파드가 답할 판정 둘
- `docs/methodology.md` — **kernels** 레인이 wave 1 에서 했다
- `docs/audit-baseline.json`, `scripts/audit_plan.py`, `envs/**`, `pyproject.toml`, `uv.lock` —
  **머지 단계가 이미 처리했다.** 상태만 확인한다
- `--update-baseline` 실행. 머지 단계의 것이다
