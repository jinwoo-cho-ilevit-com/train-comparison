# 남은 코드 작업 — worktree 캠페인

## 왜

Phase 0 가 두 번 돌아 **5/18 → 12/18**이고, 네 프레임워크가 세 모델을 모두 적재하며 같은
파라미터 수를 학습한다(625 / 473 / 988). 프레임워크 간 속도 비교의 전제가 처음 섰다.

남은 것은 대부분 코드다. **범위: 파드 없이 쓰고 검증할 수 있는 것 전부.**

실행 엔진은 **Workflow 스크립트 + git worktree 격리**다. 레인마다 워크트리 하나, 브랜치 하나,
독립 검증자 하나. 머지는 워크플로가 아니라 메인 세션이 브랜치 하나씩 한다.

시작 전에 반드시: **`HAZARDS.md`** — 이 저장소가 이미 겪은 것 전부.

## 캠페인 base

```
e5926bc9d3c8148b2f2525ed4c4bd4e8e4e5986f
```

모든 워크트리는 **ref 이름이 아니라 이 40자 SHA** 로 고정한다. 직전 팬아웃은 `origin/main`
에서 잘려 6커밋 뒤였고 그래서 무너졌다(`HAZARDS.md §4.1`).

## 레인 이름 — 옛 문자와의 번역표

**동결된 계약 파일의 xfail `reason` 문자열은 옛 레인 문자를 이름으로 부른다**
(`lane-c has not landed…`, `lane-b: report.load_artifacts…`, `lane-d: scripts/bench.py:936…`).
문자를 그대로 재사용하면 에이전트가 남의 파일을 연다. 그래서 레인은 역할 이름으로 부른다.

| 계약이 부르는 이름 | 이 캠페인의 레인 | 소유 |
|---|---|---|
| lane-a | **probe** 로 흡수 | tevatron 적재 shim |
| lane-b | **report** | `scripts/report.py` 외 |
| lane-c | **capture** | `trainbench/applied.py` |
| lane-d | **split**(wave 0) + **measure**(wave 1) 로 분할 | `bench.py` 분해 / 측정 계약 |
| lane-e | **kernels** | `trainbench/kernels.py` |
| lane-f | **packing** | `trainbench/collate.py` 의미 |
| lane-g | **adapters** | `trainbench/loader.py` |
| lane-h | **axes** | `trainbench/axes.py` |
| lane-i | **integrate** | 루트 문서·원장 |

`docs/open-verdicts.json` 의 `owner` 필드와 `docs/audit-baseline.json` 의 note 는 **또 다른**
문자 체계를 쓴다(D=axes, C=오케스트레이션, B=probe, F=이미지). 그 파일들의 문자는 이 표와
무관하다. 문자를 보면 역할로 옮겨 읽는다.

## 결정 (유지)

| # | 결정 | 선택 | 기각한 대안과 이유 |
|---|---|---|---|
| 1 | axolotl dtype | autocast 로 감싸 axolotl 을 그대로 잰다 | 임베딩을 bf16 으로 되돌리기 — native 와 같은 체제가 되지만 axolotl 이 실제로 학습할 모델이 아니게 된다 |
| 2 | packing 격리 | `cu_seq_lens_*` 를 varlen 커널로 전달 | 블록 대각 마스크 직접 생성 — transformers 가 이미 만들고, 4D 마스크를 주면 fa2 varlen 경로가 꺼진다 |
| 3 | trackio | 스키마에서 제거 | 구현 — 6개 env lock 전부에 넣고 이미지 재빌드, 측정 중 네트워크 I/O 가 교란 |
| 4 | 버전 교란 | `report.py` 가 같은 스택끼리만 줄 세운다 | 한 표에 다 넣고 버전을 열로 — 독자가 순위를 먼저 읽고 각주를 나중에 읽는다 |
| 5 | 어댑터 경계 | 프레임워크의 학습 스텝을 그대로 잰다 | 베이스 인코더만 꺼내 공통 루프 — 프레임워크가 아니라 우리 루프를 재게 된다 |
| 6 | `kernel=kernels_hub` | 축 값을 버린다 | 적용 지점을 모델 생성 후로 이동 — "kernel/attn 은 모델 생성 전에만 바꿀 수 있다"는 설계 전제가 깨진다 |
| 7 | 방법론 보강 범위 | 리서치가 찾은 새 축까지 전부 | 나중 배치 — 어댑터 인터페이스를 두 번 설계하게 된다 |

## 완료 정의 — xfail 38개

동결된 계약 5개가 들고 있는 `xfail(strict=True)` 38개가 이 캠페인의 완료 정의다.
계약 파일 자체는 **아무도 소유하지 않고 아무도 고치지 않는다.** 마커를 지우는 것만이 허용되며,
그것도 자기 레인에 배정된 마커만이다.

| 계약 | 마커 | 지우는 레인 |
|---|---|---|
| `tests/contract/test_applied_axes.py` | 30 | capture |
| `tests/contract/test_record_report.py` | 1 (`test_the_producer_stamps_the_identity`) | split |
| `tests/contract/test_record_report.py` | 5 | report |
| `tests/contract/test_loader_bench.py` | 1 (`test_bench_takes_the_framework_name_from_the_adapter`) | split |
| `tests/contract/test_loader_bench.py` | 1 (`test_loader_serves_every_framework_through_one_entry_point`) | adapters |
| `tests/contract/test_collate_metrics.py` | 0 | — |
| `tests/contract/test_kernel_provenance.py` | 0 | — |

계약 0개짜리 둘은 **게이트가 없다는 뜻이다.** kernels 레인은 `tests/test_kernels.py` 를 자기가
만들어 게이트로 삼는다. `test_applied_axes.py::test_the_contract_defers_nothing` 은 남은 마커를
AST 로 세는 xfail-strict 이므로 **마지막에 지운다** — 그것이 "다 끝났다"의 신호다.

종착점:
```
infisical run --env=dev -- uv run pytest tests/contract -q
  → 122 passed, 0 xfailed        (캠페인 base 에서는 84 passed, 38 xfailed)
```

## 경계표 — 계약 5개

| name | 두 끝 | test | sample |
|---|---|---|---|
| collate-metrics | split/measure ↔ packing | `tests/contract/test_collate_metrics.py` | `tests/fixtures/microbatch.sample.json` |
| loader-bench | split ↔ adapters | `tests/contract/test_loader_bench.py` | `tests/fixtures/adapter_out.sample.json` |
| applied-axes | capture ↔ axes | `tests/contract/test_applied_axes.py` | `tests/fixtures/axis_state.sample.json` |
| record-report | split/measure ↔ report | `tests/contract/test_record_report.py` | `tests/fixtures/run_record.sample.json` |
| kernel-provenance | kernels ↔ adapters | `tests/contract/test_kernel_provenance.py` | `tests/fixtures/kernel_fingerprint.sample.json` |

계약이 틀렸다고 판단하면 **고치지 말고 `boundaryRequests` 로 요청한다.** 통합 wave 가 하나의
개정본을 낸다. 양쪽이 각자 자기 편을 패치한 것이 `f102cd2`/`5971874`/`e5926bc` 세 커밋이
수습한 사고다.

## wave 구조

```
wave 0   split                                   단독·직렬. 머지 전엔 아무것도 안 돈다
wave 1   capture · measure · report · probe · kernels     5 병렬
wave 2   packing · adapters · axes                        3 병렬
wave 3   integrate                                        단독
```

### wave 0 — split

| owns | |
|---|---|
| `scripts/bench.py` | `88–564` 를 **순수 이동**으로 `trainbench/collate.py` 에 옮긴다 |
| ⊕ `trainbench/collate.py` | 신설 |
| `tests/test_smoke_cpu.py` | `bench_entry.*` 참조 10곳 재배선 |
| `trainbench/record.py` | `build_record` 에 `recorded_at` |

`assemble(framework=…)` 을 **비상수**로 만드는 seam 을 `bench.py` 안에 남긴다. 바인딩의
필드 이름은 `AdapterOut` 과 **정확히 같게** 한다:
`framework, model, processor, step, owned_axes, required_step_context, fingerprint,
documented_entry_point`. split 은 앞의 셋만 채우고 나머지는 `None`/빈 값으로 둔다.

**금지 · `trainbench/loader.py` 를 만들지 않는다.** `test_loader_serves_every_framework_through_one_entry_point`
는 `set(loader.ADAPTERS) == FRAMEWORKS`, `callable(loader.load)`, `AdapterOut` 필드명만 본다 —
6키 스텁이면 **XPASS 하고 strict 라서 빨개진다.** 그 마커는 adapters 의 것이다.

**금지 · 동작 변경.** 분해는 순수 이동이고 기존 스위트가 그것을 증명한다.
`bench.py` 는 `patch`/`load_kwargs`/`assemble`/`step_context`/`assert_matches` 다섯 호출과
`built.loss_fn` 바인딩을 **텍스트로** 유지해야 한다(`assert-called` 게이트).
`axes.assemble` 의 `framework` 파라미터는 기본값 없이 남고 `Built` 는 `framework` 필드를 유지한다.

### wave 1

| 레인 | owns |
|---|---|
| **capture** | `trainbench/applied.py`, `tests/test_applied.py`, `tests/contract/test_applied_axes.py`(마커 제거만) |
| **measure** | `trainbench/metrics/`, `trainbench/config_schema.py`, `tests/test_metrics.py`, `tests/test_config.py`, `tests/test_data.py` |
| **report** | `scripts/report.py`, `scripts/orchestrate.py`, `scripts/prepare_data.py`, `docker/entrypoint.sh`, `configs/run/`, `tests/test_report.py`, `tests/test_pods.py`, `tests/contract/test_record_report.py`(마커 제거만), `docs/open-verdicts.json`(항목 1개만). **루트 `pyproject.toml` 은 아니다** — trackio extra 제거는 노트로 넘긴다 |
| **probe** | `trainbench/probe/steps.py`, `trainbench/probe/sentence_transformers.py`, `trainbench/probe/tevatron.py`, `tests/test_probe.py` |
| **kernels** | ⊕`trainbench/kernels.py`, ⊕`tests/test_kernels.py`, `docs/methodology.md` |

**measure ↔ probe 경계 (테스트가 없는 경계다)**: 둘 다 `grad_norm`/`trainable_params` 를
정의한다 — probe 는 프로브 시점 거부 가드로, measure 는 측정 시점 유효성 게이트로.
두 정의가 어긋나면 리포트의 게이트와 프로브의 거부가 다른 말을 하게 되고 **어떤 테스트도
그것을 비교하지 않는다.** 양쪽 다 `.plans/notes/<lane>.md` 에 자기 정의를 정확히 적는다.
머지 단계가 대조한다.

### wave 2

| 레인 | owns | 선행 |
|---|---|---|
| **packing** | `trainbench/collate.py`(의미), `trainbench/prompt.py`, `configs/model/`, `tests/test_prompt.py`, `docs/open-verdicts.json`(항목 1개만) | split, measure |
| **adapters** | ⊕`trainbench/loader.py`, `trainbench/probe/{native,unsloth,ms_swift,axolotl,registry}.py`, `tests/contract/test_loader_bench.py`(마커 1) | split, capture, kernels, probe |
| **axes** | `trainbench/axes.py`, `configs/{optim,precision,train,parallel,dataloader,peft,kernel}/`, `tests/test_axes.py` | capture |

adapters 는 `scripts/bench.py` 를 **건드리지 않는다** — wave 0 의 seam 으로 충분하다.
axes 가 필요한 `config_schema.py` 의 `kernels_hub` 리터럴 제거와 `audit_plan.py` 의
`axis-packages` 표 항목 제거는 **노트로 넘기고 머지 단계가 적용한다.**

### wave 3 — integrate

`AGENTS.md`, `PLAN.md`(산문), `docs/CONTRACTS.md`, `docs/support-matrix.md`, `README.md`,
`docs/open-verdicts.json`(구조), `.plans/`, `tests/test_audit.py`.

`verdicts-closed` 는 **2 open 이 예상 종착점**이다 — `loss-empty-pixel-slice…` 와
`loss-gradcache-memory…` 는 실제 체크포인트와 GPU 없이 닫을 수 없고, 남는 것이 옳다.
4 → 2 는 `shrank` 로 BLOCK 되며 그것은 결함이 아니라 래칫이다. baseline 갱신은 머지 단계가 한다.

## 통합자 전용 — 어떤 레인도 건드리지 않는다

`docs/audit-baseline.json`, 루트 `PLAN.md` 의 레이아웃 블록, `envs/**/pyproject.toml`,
`envs/**/uv.lock`, 루트 `pyproject.toml`, `uv.lock`, `docs/prebuilt-wheels.yaml`,
`scripts/audit_plan.py`.

레인은 요청만 한다:
- `.plans/notes/<lane>.md` — baseline note 산문, 루트 문서에 올라가야 할 사실, 다른 레인 파일에
  필요한 한 줄짜리 변경
- `.plans/deps/<lane>.txt` — 패키지 요구

이렇게 하는 이유는 `HAZARDS.md §5`(baseline 은 양방향 래칫이고 네 레인이 그 수를 움직인다)와
`§4.2`(신설 파일이 `plan-files` 를 빨갛게 하는데 등재 권한이 없다)다.

## 레인 규율

`HAZARDS.md §10` 이 목록이다. 산출물 스키마가 그것을 필드로 강제한다:

- `baseVerified` — `git reset --hard <40자 SHA>` 후 되읽은 SHA. **동일성**이지 ancestry 가 아니다.
  이어서 `pytest tests/contract -q` 와 `audit_plan.py` 가 그 wave 에 고정된 수치를 그대로
  찍는지 확인한다 (SHA 비교만으로는 `aborted-wave1-lane-e` 형태를 못 잡는다)
- `gates[]` — 네 게이트의 **원문 마지막 40줄** + 직전 `git rev-parse HEAD` + `emptyInputProof`
- `mutations[]` — 추가한 검사마다 하나. `liveDefinitionProof` 필수. 수가 안 움직이면 `inert: true`
- `xfailsCleared[]` — nodeid 를 **혼자 돌려서** 확인. 요약줄 읽기 금지
- `boundaryRequests[]` — 계약 개정 요청
- `outOfBounds[]` — 소유 밖 파일을 건드렸으면 숨기지 않는다
- `notMeasured[]` — 못 낸 숫자는 여기 적고 "확인 안 함"

**자기 판정 필드는 없다.** `passed`/`readyToMerge` 를 두지 않는다. 판정은 그 일을 하지 않은
검증 에이전트가 레인 워크트리에서 게이트를 재실행해서 내린다.

## 네 게이트

```
uv run ruff check && uv run ruff format --check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/audit_plan.py
infisical run --env=dev -- uv run python scripts/env_report.py \
  device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4
```

캠페인 base 에서의 값 (이 세션에서 직접 실행함):
`ruff` 통과 74 files / `pytest` 877 passed, 38 xfailed /
`audit` 12/15 passing, 0 new failure(s), 0 newly fixed, 0 grew, 0 shrank /
`env_report` JSON 기록.

`plan-files` 가 **자기가 만든 신설 파일만으로** 빨간 것은 레인의 실패가 아니다.
`newFiles` 를 채워 보고하고 머지 단계가 등재한다. 그 외의 이유로 빨갛거나 다른 체크가 새로
빨개지면 그것은 레인의 실패다.

## 전체 완료 조건

- 모든 레인의 완료 조건 충족 → 각 레인 브리프
- `pytest tests/contract -q` → **122 passed, 0 xfailed**
- 네 게이트 통과
- `config-consumed` 0
- `axis-values` 단일값 그룹 0 — **단 `precision` 은 예외일 수 있다.**
  리서치가 확정했다: `mxfp8` 은 compute capability 10.x 전용, `nvfp4` 는 CC ≥ 10.0 전용이고
  이 스터디의 파드는 A100 이다 (`.plans/research/axis-libraries.md §6.4`).
  RunPod 에서 CC 10.x 를 확보할 수 있는지는 **확인 안 함**. 남는다면 그것은 **코드 결함이
  아니라 하드웨어 사실**이고, 그 구별이 baseline note 에 그대로 적혀야 한다
- `verdicts-closed` **2 open** (파드 판정 둘)
- 리뷰(micro 9 + macro 4) → 적대적 검증 → 수정 → 재검증 완료
- `[pod]` Phase 0 18/18 → verdict: ____ by: ____ at: ____

## 제외

- 파드가 숫자를 내야 답이 나오는 것 — 노이즈 바닥과 3% 임계값 유도, 프로파일러 오버헤드,
  deterministic on/off, 데이터로딩 병목 판정, loss parity, GPU clock lock 가능 여부,
  호스트 지문, liger 커버리지 분포, `loss-empty-pixel-slice`, `loss-gradcache-memory-and-overhead`
- 캠페인 실행 자체 (Phase 0 18/18, Phase 2 ablation, Phase 3 프레임워크 측정, 품질 런)
- `docs/report.md` — 산출물이고 숫자가 나와야 쓴다
