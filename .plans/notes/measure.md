# measure — 머지 단계로 넘기는 것

base `03396a64a967bda1c72814876359de9a04a78162`, 브랜치 `wave1-measure`.

## 1. `grad_norm` / `trainable_params` — 측정 시점 정의 (probe 레인과 대조용)

`.plans/remaining-code/measure.md` 가 요구한 대조 항목이다. probe 레인의
`.plans/notes/probe.md` 와 나란히 놓고 읽는다.

구현 위치: `trainbench/metrics/validity.py`.

| 이름 | 무엇을 세는가 | 언제 재는가 | 0의 의미 |
|---|---|---|---|
| `trainable_params` | `requires_grad=True` 인 파라미터 **텐서 개수**. 원소 수가 아니다 | 측정 창이 끝난 뒤 모델에서 읽는다 | 이 모델은 학습할 수 없다. 속도 결과가 아니다 |
| `total_params` | 파라미터 텐서 총 개수 | 같음 | — |
| `params_with_grad` | `requires_grad` 이면서 `.grad is not None` 인 텐서 개수 | 같음 | 손실에서 모든 파라미터가 끊겼다 |
| `grad_norm` | `.grad` 를 가진 모든 파라미터에 대한 전역 L2 norm. float64 누적 | **마지막 측정 스텝의 backward 직후, optimizer 가 0으로 지우기 전** | backward 가 어떤 파라미터에도 닿지 않았다 |

probe 쪽(`trainbench/probe/steps.py::infonce_backward`, 415-417줄)과 맞춘 것:

- 세 카운트 모두 **텐서 개수**로 동일하다. 단위가 갈리면 같은 이름이 두 값을 갖는다
- probe 는 `grad_norm` 을 정의하지 않는다. 측정 시점에만 존재하는 항목이고,
  그래서 `METRIC_DEFINITIONS` 에 정의 문장을 넣어 결과 JSON 과 함께 이동시켰다
- probe 는 `trainable==0` / `with_grad==0` 에서 **예외를 던져 프로브를 거부**하고,
  measure 는 **레코드를 남기되 속도 결과로 읽히지 않게** 판정한다. 프로브 시점에는
  버릴 셀이고 측정 시점에는 이미 지불한 파드 시간이라 범주가 다르다

## 2. 통합자가 적용할 것

### 2.1 `configs/measurement/` 그룹 신설

`MeasurementConfig` 는 지금 `BenchConfig.measurement` 의 **스키마 기본값**으로만 존재한다
(`+measurement.repeats=10` 식 오버라이드는 된다). YAML 그룹으로 올리려면 레인이 손댈 수
없는 파일 셋이 함께 움직여야 한다:

- `configs/config.yaml` 의 `defaults` 에 `- measurement: default` 추가
- `configs/measurement/default.yaml` 신설
- `scripts/audit_plan.py` 의 `NON_AXIS_GROUPS` 에 `"measurement"` 추가

세 번째가 빠지면 `axis-packages` 가 `measurement/default` 를 미분류로 잡아 **새 실패**가
난다(레인에서 확인함: 그 체크는 `NON_AXIS_GROUPS` 밖의 모든 variant 파일을
`AXIS_PACKAGES`/`AXIS_NEEDS_NOTHING` 에서 찾는다). 그래서 그룹 신설을 하지 않았다.

올린 뒤 `config-consumed` 는 늘지 않는다 — 여덟 leaf 전부 `trainbench/metrics/` 가
`config.measurement.<field>` 로 읽는다.

### 2.2 trackio 스키마 제거 — 한 커밋 안에서만 가능

**하지 않았다.** `RunConfig.trackio_project`/`trackio_space_id` 만 지우면
`Strict(extra="forbid")` 가 `configs/run/*.yaml` 의 남은 두 키를 거부한다.
이 워크트리에서 직접 재현했다:

```
run.trackio_project
  Extra inputs are not permitted [type=extra_forbidden, input_value='train-comparison', ...]
run.trackio_space_id
  Extra inputs are not permitted [type=extra_forbidden, input_value=None, ...]
```

config-path 게이트가 그 자리에서 죽는다. 스키마 두 줄과 `configs/run/*.yaml` 네 파일
(+ `pyproject.toml` 의 `tracking` extra, report 레인)은 **같은 커밋**에 빠져야 한다.
`config-consumed` 가 4 → 2 로 줄어드는 것은 YAML 쪽이 빠질 때이지 스키마 쪽이 빠질 때가
아니다 — leaf 목록은 `configs/` 에서 읽는다. `shrank` BLOCK 은 그때 발생한다.

### 2.3 `data.max_seq_len` 을 축으로 — 레인 셋이 함께 움직여야 한다

**하지 않았다.** `Axis()` 를 붙이는 순간 필요한 것:

1. `trainbench/applied.py` 의 `_CAPTURES` 에 `data.max_seq_len` 캡처 프로브 (**capture 레인**)
2. `trainbench/axes.py` 의 `IMPLEMENTED` 에 같은 이름 (**axes 레인**, wave 2)
3. `int` 는 `get_args` 로 값을 못 뽑으므로 `flag_knob_values()` 가 "not enumerable" 로
   보고한다 → `axis-values` count 증가 → `grew` BLOCK. `Literal[...]` 로 값을 열거하거나
   `scripts/audit_plan.py` 가 숫자 축을 다루도록 바꿔야 한다 (**통합자**)

1이 없으면 `capture` 가 `no capture probe implemented` 로 undetermined 를 내고
`assert_matches` 가 **모든 timing 런을 거부**한다. `HAZARDS.md §3` 의 `axis-wired`
baseline note 사고와 같은 모양이다. `boundaryRequests` 에 blocking 으로 올렸다.

## 3. bench.py 배선 — split 이 소유, 이번 wave 범위 밖

`scripts/bench.py` 는 이 레인 소유가 아니다(`.plans/remaining-code/measure.md` 하지 않는 것).
`metrics` 쪽은 준비됐고 호출부가 바뀌어야 레코드에 실린다:

- `metrics.summarise(..., config=config, padded_tokens_per_step=counted["padded_tokens"]/kept)`
  — `config` 하나로 `measurement` 블록과 `profiled` 가 채워지고, 선언된 분모가 실제로
  세어졌는지 검사된다. `padded_tokens` 를 `extra_counts` 에 그대로 두어도 동작한다
- `built.model` 에서 `metrics.gradient_norm` / `metrics.parameter_counts` 를
  **`optimizer.zero_grad` 직전**에 읽어 summary 에 넣는다. 그 뒤에 읽으면 항상 0이다
- `metrics.build_timer(device, config.measurement.instrument)` 로 타이머를 만든다
- 스텝 루프를 `except BaseException as exc: if metrics.is_oom(exc): record.update(metrics.oom_status(exc, peak_bytes=...))` 로 감싼다
- `config.measurement.repeats` 만큼 반복할 때 `metrics.repeat_seeds(...)` 의 값을
  반복마다 기록한다 (MLPerf CLOSED: 같은 줄에 같은 seed 가 두 번 오면 안 된다)

## 4. 재지 않은 것

- CUDA event 계측기(`CudaEventTimer`)는 **CPU 에서 도는 이 호스트에서 실행되지 않았다.**
  CPU 에서 검증한 것은 "선택되고, CUDA 가 없으면 거부한다"까지다
- warmup·repeats·집계 방식의 **값**은 전부 미교정이다. `baseline_tolerance` 0.03 은
  `AGENTS.md` 가 근거 없이 들고 있던 상수이며 `baseline_tolerance_calibrated=false` 로
  결과에 실린다. 파드가 노이즈 바닥을 잰 뒤에 정해진다
