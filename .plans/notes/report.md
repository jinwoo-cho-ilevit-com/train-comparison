# report 레인 노트 (wave 1)

머지 단계가 읽는다. 여기 적힌 숫자는 전부 이 워크트리에서 직접 실행해 얻은 것이다.

## 머지 단계가 해야 할 것

### 1. `docs/audit-baseline.json` — 두 체크가 줄었다

```
infisical run --env=dev -- uv run python scripts/audit_plan.py
  12/15 passing, 0 new failure(s), 0 newly fixed, 0 grew, 2 shrank, 0 unreadable
  BLOCKED: baseline is stale, these shrank: config-consumed 4->2, verdicts-closed 4->3
```

- `config-consumed` 4 -> 2. `scripts/prepare_data.py`의 `data = config.data` 별칭을 없애
  `config.data.*`로 직접 읽게 했다. `data.subset_rows`/`data.push_subset` 오탐 2건이 사라졌고
  남은 2건은 `run.trackio_project`/`run.trackio_space_id`로, **아래 3번이 처리되기 전에는
  0이 될 수 없다.** baseline note의 (1)번 문단은 이제 사실이 아니다.
- `verdicts-closed` 4 -> 3. `images-carry-a-code-snapshot-nothing-checks-is-current`를 닫았다.

### 2. `PLAN.md` 레이아웃 — 신설 파일 없음

이 레인은 파일을 만들지 않았다. `plan-files`는 초록이다.

### 3. trackio 제거는 한쪽만으로 착지할 수 없다 (결정 3)

**내 몫(`configs/run/*.yaml` 4개)을 제거하지 않았다.** 제거하면 이 워크트리의 네 번째 게이트가
즉시 빨개진다. 실측:

```
$ printf 'purpose: timing\nprofiler: false\n' > configs/run/timing.yaml
$ infisical run --env=dev -- uv run python scripts/env_report.py device=cpu \
    model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4
pydantic_core._pydantic_core.ValidationError: 1 validation error for BenchConfig
run.trackio_project
  Field required [type=missing, input_value={'purpose': 'timing', 'profiler': False}, input_type=dict]
```

`config_schema.RunConfig.trackio_project`는 기본값 없는 필수 필드이고 `Strict`가
`extra="forbid"`이므로 **yaml만 빼도, 스키마만 빼도 config 합성이 죽는다.** measure 레인도
같은 이유로 자기 쪽만 뺄 수 없다. 두 변경은 **한 커밋에 같이** 들어가야 한다:

- `configs/run/{timing,profile,quality,probe}.yaml`에서 `trackio_project`/`trackio_space_id` 삭제
- `trainbench/config_schema.py`의 `RunConfig`에서 같은 두 필드 삭제 (measure 레인 몫)
- 루트 `pyproject.toml`의 `tracking = ["trackio>=0.34"]` extra 삭제 + `uv.lock` 재해석
  (**통합자 전용**)

그 뒤 `config-consumed`가 0이 된다.

### 4. `gradcache` 죽은 핀 — 판단 [human 확인 항목]

**핀을 유지하고, 손으로 짠 구현을 라이브러리로 바꾸는 결정은 wave 3로 넘긴다.** 근거:

- `envs/native/pyproject.toml:45`가 `gradcache @ git+https://github.com/luyug/GradCache.git`를
  핀한다. 실측: `grad_cache`를 import하는 코드는 `trainbench/`, `scripts/`, `tests/` 통틀어
  **0건**이다. 실제 구현은 `trainbench/axes.py:1763`의 손으로 짠 `gradcache_backward`다.
- 그럼에도 지금 핀을 빼자고 하지 않는 이유: `envs/**`는 통합자 전용이고, 이 호스트에서
  재해석할 수 있는 lock이 여섯 중 하나뿐이라(`env-locks`/`doc-commands`가 검사한다)
  핀 하나를 빼는 것이 이미지의 다른 부분까지 움직인다. 죽은 핀의 비용은 이미지 크기이고,
  잘못 뺐을 때의 비용은 측정 자체다.
- **wave 3가 둘 중 하나를 정해야 한다**: (a) `axes._loss`의 손으로 짠 구현을 상류
  `grad_cache.GradCache`로 교체하고 핀을 살린다 — `optim=muon`이 세운 "라이브러리를 쓰고
  손으로 짜지 않는다"와 일치한다. (b) 손으로 짠 구현을 유지하고 핀을 뺀다 — 이 경우
  `docs/methodology.md`에 "cached_mnrl은 상류 GradCache가 아니라 이 저장소 구현"이라고
  적어야 한다.
- (a)의 비용은 **측정 안 함** — `grad_cache.GradCache`가 이 하네스의 4분할 스텝에 얹히는지
  확인하지 않았다.

## 이 레인이 내린 설계 결정

### `check_axis_not_split`이 `framework.name`에 침묵하던 문제 (작업 7)

**규칙을 코드에 넣되 `framework`는 명시적 예외로 두고, 예외를 원장에 기록한다.**

- 원인은 판정 규칙이 아니라 입력이었다. `axes_touched`가 `exp.overrides`와 `exp.settings`만
  읽는데 `framework`/`model`/`run`은 매니페스트 최상위 필드이고 `plan_runs`가 오버라이드로
  **주입**한다. 그래서 `framework.name`은 축 목록에 아예 들어오지 않았다.
- 고친 방식: `pod_overrides(exp)`를 만들어 `plan_runs`와 `axes_touched`가 **같은 문자열**을
  쓴다. 둘이 갈라질 수 없다.
- 그러면 `framework`가 모델마다 6개 파드로 갈라진 것이 보이고, 그대로 두면 캠페인 전체가
  거부된다. 프레임워크 값 하나가 이미지 하나이므로 **한 파드가 공유할 수 없다** — 이것은
  결함이 아니라 스터디의 사실이다. `CROSS_POD_GROUP = "framework"`가 그 예외이고,
  `cross_pod_notes()`가 그것을 문장으로 만들어 콘솔과 원장(`cross_pod_axes` 키) 양쪽에 남긴다.
- 예외는 **그룹 하나 너비**다. framework가 다른 두 파드가 `loss`를 나눠 가지면 여전히 거부된다
  (`test_the_axis_split_guard_still_refuses_a_real_split_beside_the_framework_one`).
- 대안(순수 예외 문서화)을 기각한 이유: 문서는 `plan_runs`가 오버라이드를 바꾸면 따라오지
  않는다. 이 결함 자체가 "규칙은 산문에만 있었다"의 결과다.

### `report.py`의 학습 유효성 게이트

`training_verdict`는 `tests/contract/test_record_report.py`가 들고 있는 정의와 같은 네 가지를
본다. 한 가지만 다르다: **계약본은 `config.peft.mode`를 직접 인덱싱하고 이쪽은 `.get`으로
읽는다.** 계약의 레코드는 항상 그 필드를 싣지만 리포트는 임의의 아티팩트를 받는다.
단언을 약하게 만든 것이 아니라 KeyError로 병합 전체가 죽지 않게 한 것이다.

부작용 하나를 기록한다: `metrics`는 있는데 게이트 필드(`grad_norm` 등)가 **없는** 레코드도
속도 표에서 빠진다. "학습했는지 말할 수 없는 레코드"와 "학습하지 않은 레코드"를 같이 다룬다.
`tests/test_smoke_cpu.py`의 대조군 레코드가 여기 걸려서 게이트 필드를 넣었다 (소유 밖 수정,
산출물 `outOfBounds`에 적었다).

## 열린 질문 (이 레인이 닫지 않았다)

- **baseline 게이트는 학습 유효성을 보지 않는다.** `_baseline_value`는 `step_seconds_p50`가
  양수이고 프로파일러가 꺼져 있으면 통과시킨다. 얼어붙은 그래프로 잰 baseline이 모든 파드의
  기준값이 될 수 있다. 같은 결함 계열이지만 이 레인의 9개 작업에 없어서 손대지 않았다.
- `report.py`는 원장(`outputs/orchestrate-*.json`)을 여전히 **결과 선별에 쓰지 않는다.**
  작업 1은 (a) 기록된 신원으로 거부하는 쪽을 골랐고, (b) 원장으로 거르는 쪽은 열려 있다.
  (a)로 충분한 이유: 이 저장소의 모든 생산자가 `recorded_at`을 찍는다
  (`record.build_record:177`, `publish_result.provenance:101`).
- `images-carry-a-code-snapshot...` 판정을 닫은 근거는 **파드가 아니라 CPU에서의 entrypoint
  실행**이다. `closes_when.command`가 지정한 A100 파드 1대는 **실행 안 함**. 항목의
  `closed.evidence`에 그대로 적었고, 리뷰어가 그 기준을 다르게 읽으면 되돌릴 수 있다.
