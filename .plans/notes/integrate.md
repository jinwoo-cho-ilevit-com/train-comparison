# integrate — 아홉 레인의 요청을 처리한 결과와 남은 것

base `d0d0aa0395ce209acfc818a408198f1135e535c3`, 브랜치 `wave3b-integrate`.
여기 적힌 상태는 전부 이 워크트리에서 이번 세션에 직접 확인했다.

**`.plans/notes/<lane>.md` 아홉 개는 지우지 않는다.** 각 레인이 무엇을 실행해서
그렇게 판단했는지가 그 파일에만 있고, 이 문서는 그 요청들의 **처분 대장**이다.
요청 원문과 근거는 원래 자리에서 읽는다.

---

## 1. 이 레인이 처리한 것

| 요청한 레인 | 요청 | 어디에 |
|---|---|---|
| — (브리프 작업 1) | Run / Report / one-setting 세 명령 | `AGENTS.md ## Commands` |
| capture | `CONTRACTS.md` 의 `applied.py` 블록이 옛 모양 | `docs/CONTRACTS.md §2` — `Built.precision_recipe`/`owned_axes`, `AxisState.owner`/`.state`, `AppliedState.framework_owned()`, 최상위 `framework_owned` 키 |
| kernels | `kernel-provenance` 구현이 둘인 이유를 문서에 | `docs/CONTRACTS.md §2` 새 소절 "경계 규칙의 구현이 둘인 이유" |
| adapters (§3.4) | `loader-bench` 구현이 둘인 이유 + axolotl autocast 배선 | 같은 소절 + `§2` 의 `step_context` 문단(배선 자체는 wire 레인이 이미 적었다) |
| kernels | 커널 신원은 버전이 아니라 repo+revision | 같은 소절 마지막 문단 |
| wire (§3) | `docs/support-matrix.md:128` trackio 행 삭제 | 삭제 + 결정 3 한 문단 |
| wire (§3), axes (§1) | `docs/support-matrix.md:504` `kernel/kernels_hub` 행 삭제 | 삭제 |
| axes (§1) | `PLAN.md` 의 `kernels_hub` 언급 | `PLAN.md:576` 행에서 제거 (`:414` 는 이미 처리돼 있었다) |
| probe (§2.1) | `axes_verified` 가 `all_matched:false` 로 통과한다는 문장이 거짓 | `docs/support-matrix.md` "초록이지만 그대로 믿으면 안 되는 것" 첫 항목을 다시 씀 + CPU 호스트 항목 추가 |
| — (브리프 작업 3) | 결정 4 / 5 의 대가 / 6, 어댑터 진입점 대조표 | `docs/support-matrix.md` 새 절 "이 표를 나란히 읽으면 안 되는 자리" |
| — | `report.py` 가 mtime 으로 캠페인을 고른다는 "재현" 절의 문장이 거짓 | 같은 파일, 그 절을 다시 씀 |

`docs/open-verdicts.json` 은 **바꾸지 않았다.** 아래 3번을 읽는다.

## 2. 새로 생긴 검사 하나

`tests/test_audit.py::test_the_documented_commands_doc_commands_never_runs_are_named_here`.

`doc-commands` 는 "every documented command runs as written" 이라고 보고하면서
실제로는 `uv sync` 줄과 `scripts/env_report.py` 줄 **두 모양만** 실행한다. 이번에
`AGENTS.md` 에 실행되지 않는 명령 셋이 늘었으므로, 그 차이를 문장이 아니라 검사로
못박았다. `scripts/audit_plan.py` 는 통합자 전용이라 문구 자체는 고치지 않았다.

부수기(둘 다 원복 확인):
- `audit_plan` 의 실행 정규식을 `env_report|orchestrate` 로 넓힘 →
  `AssertionError: ['.../python', 'scripts/orchestrate.py', '--experiment', "'phase0-*'"]`
- `AGENTS.md` 의 Report 줄 제거 →
  `AssertionError: AGENTS.md no longer documents ['scripts/report.py']`

## 3. `docs/open-verdicts.json` — 3 open 이고, 그것이 지금 옳다

브리프의 완료 조건 4 는 **2 open** 을 요구하지만 이 세션은 그 상태를 만들지 않았다.
만들려면 `images-carry-a-code-snapshot-nothing-checks-is-current` 를 닫아야 하고,
그것은 CPU 증거로 파드 판정을 닫는 행위다. 같은 브리프가 그것을 금지하고(선례 `1a7b7c7`)
2026-08-03 머지 단계가 report 레인의 닫기를 이미 한 번 되돌렸다.

이 세션에서 확인한 것:

- 앵커 둘 다 실재한다 —
  `tests/test_pods.py:2848::test_a_pod_whose_image_predates_the_config_it_is_handed_does_not_measure`,
  `tests/test_prompt.py:218::test_the_query_instruction_prompt_appears_once_in_a_templated_row`
- `qwen3-vl-query-prompt-may-go-in-twice` 는 packing 레인이 실 프로세서 측정과 함께 닫았다.
  원장의 `closed.evidence` 가 그 측정을 인용한다
- `images-carry-...` 는 `closes_when.command` 가 A100 파드 1대를 지정하고 **이 세션은
  파드를 띄우지 않았다.** 감사도 같은 말을 한다: `anchor now holds; record the run in
  closed and close it`

따라서 `verdicts-closed` 는 3 open 으로 남고 baseline 과 일치해 `0 grew, 0 shrank` 다.
**2 open 으로 내리는 것은 다음 파드 발사의 일이다.** 그때 baseline 도 함께 움직인다.

## 4. 머지 단계에 남는 것 — 전부 통합자 전용 파일이다

이 레인은 `docs/audit-baseline.json`, `scripts/audit_plan.py`, `envs/**`, 루트
`pyproject.toml`, `uv.lock` 을 건드리지 않았다. 아래는 이번 세션에 상태를 확인한 것만 적는다.

### 4.1 trackio 의 마지막 몫 (wire §2)

`pyproject.toml:65` 의 `tracking = ["trackio>=0.34"]` 가 남아 있다. 스키마와
`configs/run/*.yaml` 쪽은 이미 빠졌으므로 config 합성은 어느 순서로도 깨지지 않는다.
extra 삭제 + 루트 `uv lock` + `envs/*/uv.lock` 여섯 재생성을 **한 커밋**으로.
(여섯 env 가 루트 `trainbench` 를 경로 의존하므로 하나만 빼면 여섯이 전부 stale 이다 —
wire 레인이 실행해서 확인하고 되돌렸다.)

`docs/evidence/env-report-cpu-qwen3_5_0_8b-native.json:74-75` 가 아직 `trackio_*` 두
키를 담고 있다. 스키마가 더 이상 그 필드를 갖지 않으므로 재생성하거나, 그 아티팩트가
어느 커밋의 것인지 파일 안에 명시한다.

### 4.2 감사 스크립트 (axes §3, §3b / measure §2.1)

- `AXIS_VALUE_COMPANIONS` 에 `train.offload/{optimizer,param,both}` 의 동반
  `parallel=zero2|zero3`. **이것 없이는 `train.offload` 가 코드가 다 맞아도 `1/4` 로
  남는다** — 감사가 그룹을 하나씩 합성해 stage 없이 offload 를 시도하기 때문이다.
  확인함: 지금 이 dict 에는 `compile/max_autotune` 과 `freeze/ple`, `freeze/vision_and_ple`
  셋뿐이다
- 파일 그룹에도 `attempt(..., verify=...)` 를 건다. axes 레인 실측으로 `parallel` 의
  적용 지점을 비워도 수가 움직이지 않았다 — 지금 이 체크는 파일 그룹의 빈 적용 지점을
  **못 본다**. `precision`/`kernel` 은 `_Tiny` 스텁으로 원리적으로 못 보므로 제외
- `configs/measurement/` 그룹을 올릴 때 `NON_AXIS_GROUPS` 에 `"measurement"` 도 함께.
  확인함: 지금 그 frozenset 은 `{data, model, run, train, experiment}` 이고
  `configs/measurement/` 는 없다. 셋이 한 커밋에 가지 않으면 `axis-packages` 가 새로 빨개진다

### 4.3 `envs/native` (deps/kernels.txt)

`kernels==0.16.0` 이 transformers 5.14.1 의 창(`0.15.2 <= v < 0.16.0`, 상한 배타적)
밖이다. 결정 6 으로 `kernel=kernels_hub` 축은 사라졌지만 **두 번째 영향이 남는다** —
`is_kernels_available()` 이 False 면 `flash_attention_2` 요청의 Hub fallback 도 켜지지
않는다. 실행으로는 **확인 안 함**(이 호스트에 `kernels` 가 설치되지 않는다).

### 4.4 소유 경계 때문에 남은 한 줄짜리

- `trainbench/probe/steps.py:169` 가 `axes._environment_bound_kernel` 을 밑줄 이름으로
  가로질러 부른다(probe §2.2). 공개 이름으로 바꾸면 그 한 줄이 같이 움직인다
- `docs/model-spec.md` 의 "instruction prompt 는 쿼리 쪽만 싣는다" 가 이 체크포인트의
  템플릿 아래에서 거짓이다(packing §3). packing 레인이 positive 행도 1회 싣는 것을
  실측했다. 없애려면 빈 system 턴을 명시적으로 넘겨야 하고 그것은 설계 결정이다
- tevatron 의 `framework_version` 이 `"unknown"` 으로 통과한다(probe §2.4). 실릴 값은
  버전 문자열이 아니라 `envs/tevatron/uv.lock` 의 커밋 sha 이고, 배선이
  `trainbench/probe/types.py` 와 `scripts/report.py` 에 걸쳐 있다

## 5. 닫히지 않은 판단 — 다음 캠페인의 입력

### 5.1 `gradcache` 죽은 핀 (report §4) — 이 레인이 정하지 않았다

`envs/native/pyproject.toml` 이 상류 GradCache 를 핀하는데 `grad_cache` 를 import 하는
코드가 저장소에 0건이고, 실제 구현은 `trainbench/axes.py` 의 손으로 짠
`gradcache_backward` 다. 두 선택지의 비용이 갈린다:

- (a) 상류 `grad_cache.GradCache` 로 교체하고 핀을 살린다 — `optim=muon` 이 세운
  "라이브러리를 쓰고 손으로 짜지 않는다" 와 일치한다. **비용은 측정 안 함**:
  이 하네스의 4분할 스텝에 얹히는지 아무도 확인하지 않았다
- (b) 손으로 짠 구현을 유지하고 핀을 뺀다 — 그러면 `docs/methodology.md` 에
  "`cached_mnrl` 은 상류 GradCache 가 아니라 이 저장소 구현" 이라고 적어야 한다.
  그 파일은 kernels 레인 소유다

(a) 의 비용을 재지 않은 채 고르는 것이 이 항목이 세 wave 를 건너온 이유다.
**파드가 `loss-gradcache-memory-and-overhead-are-unmeasured` 를 닫을 때 같이 정한다.**

### 5.2 `seq_idx` 계약 개정 (wire §4)

`.plans/remaining-code/seqidx.md` 로 **권한을 가진 단독 레인에 배정됐다.** 이 레인은
계약 파일을 고칠 권한이 없고, 그 레인이 integrate 와 동시에 돈다. 여기서는 아무것도
하지 않는다.

### 5.3 `adapter_out.sample.json` 의 `differs` 브랜치 (adapters §4)

fixture 의 `sentence_transformers.documented_entry_point.differs` 가 `null` 인데
live 어댑터는 `true` 로 선언한다 — `.plans/research/sentence-transformers.md` 가 핀된
휠에서 둘 다 인용해 이 호스트에서 답이 나왔기 때문이다. 계약 테스트는 fixture 만
검증하므로 지금은 초록이지만 **두 문서가 같은 칸에 대해 다른 말을 한다.**

고치려면 `test_the_sample_exercises_every_branch_the_contract_has` 가 요구하는
`differs is None` 브랜치를 **어느 칸으로 옮길지** 함께 정해야 한다. 계약 파일은 아무도
소유하지 않으므로 개정본은 권한을 받은 단독 레인이 낸다. 이 레인은 fixture 를
건드리지 않았다.

## 6. 파드가 답해야 할 것

레인별 목록이 이미 각 노트에 있다 — `capture` (축 4), `axes §5` (축 13),
`wire §8` (배선 10), `kernels` (6), `packing` (2), `probe §3` (4), `adapters §5` (5).
여기서 합치지 않는다: 합친 목록은 원문의 근거와 끊어지고, 그것이
`HAZARDS.md §2` 가 금지하는 옮겨 적기의 시작이다.
