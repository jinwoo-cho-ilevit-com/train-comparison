# measure — 측정 계약 (wave 1)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`.
> split 이 이미 머지됐다. `trainbench/collate.py` 는 존재하지만 **네 것이 아니다**(packing 레인).

## 목표

속도 벤치마크의 분모·통계·유효성을 config 로 끌어낸다. 지금 하네스는 이 넷 중 어느 것도
설정으로 갖고 있지 않고, 그래서 어댑터마다 다른 답을 고를 수 있으며 그 편차가 측정하려는
축 효과보다 클 수 있다.

## Owns

```
trainbench/metrics/
trainbench/config_schema.py
tests/test_metrics.py
tests/test_config.py
tests/test_data.py
```

## 작업 1 — 토큰 회계

**처리량의 분모가 정의돼 있지 않다.** 패딩을 세느냐 실토큰을 세느냐가 dataloader 축
(packing on/off)의 순위를 뒤집는다. NLP 데이터셋에서 패딩이 토큰의 절반까지 가고,
GLUE-CoLA seq128 에서는 89%다.

임베딩 특유의 함정: **in-batch negative 를 쓰면 배치 크기가 목적함수의 일부다**(negative 수).
packing 이 유효 배치를 바꾸는 순간 packing 은 속도 축이기를 그만두고 목적함수 축이 된다.

계약:
- 실토큰 수와 패딩 토큰 수는 **별개 필드**
- 하네스는 프레임워크 자신의 `tokens/sec` 을 **절대 쓰지 않는다.** 원시 카운터에서 다시 계산한다
- 분모가 config 에 선언되고 결과에 실린다

지금 카운트가 나오는 세 자리는 collate 안에 있다(split 이후 `trainbench/collate.py`).
정의는 `trainbench/metrics/__init__.py:39-64` 의 `METRIC_DEFINITIONS` 에 있다.
**정의는 네 것이고 카운트 지점은 packing 레인의 것이다** — 경계 `collate-metrics`
(`tests/contract/test_collate_metrics.py`, `tests/fixtures/microbatch.sample.json`)가
그 사이를 못박고 있다. 필드를 늘려야 하면 `boundaryRequests` 로 요청한다.

## 작업 2 — 측정 통계

config 에 warmup 스텝·반복 횟수·계측기·집계 통계에 대한 knob 이 **하나도 없다.**

참고 관행 (기존 리서치가 수집한 것 — 재확인은 선택):
- MLPerf 는 Small LLM finetuning 에서 **10회 이상 반복**하고 Olympic scoring(최대·최소를
  버리고 평균)을 쓴다
- **wall clock 이 아니라 CUDA event** 로 잰다
- `torch.compile` JIT 때문에 warmup 10 스텝
- 꼬리가 두꺼운 지연 분포에서는 산술평균 대신 trimmed mean

**넷 다 config knob 이어야 한다.** 값은 여기서 정하지 않는다 — 파드가 노이즈 바닥을 재기 전엔
어떤 값도 근거가 없다.

지금 `trainbench/metrics/__init__.py` 는 `percentile`(nearest-rank, 보간 없음),
`StepTimer`, `summarise` 를 갖고 있다. 집계를 config 에서 고르게 만드는 것이 이 작업이다.

## 작업 3 — 학습 유효성 게이트

unsloth 3칸이 **모든 파라미터가 얼어붙은 채** `infonce_backward` 를 통과했다
(`params_with_grad=0, trainable_params=0`). 필드 사례: unsloth 46,000 tok/s 에서 grad norm 0.

런 레코드가 실어야 할 것: `grad_norm`, `trainable_params`, `loss[0]`/`loss[-1]`,
`peak_memory_bytes`. 그리고 **`grad_norm` 이 0 이거나 loss 가 감소하지 않는 런은 속도 결과가
아니다.**

`tests/contract/test_record_report.py::test_a_run_that_trained_nothing_is_not_published_as_a_speed_result`
가 이 경계를 못박는다. 그 마커는 **report 레인**이 지우지만(소비 쪽), 레코드에 그 필드를
넣는 것은 이 레인이다. `tests/fixtures/run_record.sample.json` 이 모양을 정한다.

### probe 레인과의 경계 — 테스트가 없는 경계다

**probe 레인도 `grad_norm`/`trainable_params` 를 정의한다** — 프로브 시점 거부 가드로
(`trainbench/probe/steps.py`). 이 레인은 측정 시점 게이트로 정의한다.
두 정의가 어긋나면 리포트의 게이트와 프로브의 거부가 다른 말을 하게 되고,
**어떤 테스트도 그것을 비교하지 않는다.**

→ `.plans/notes/measure.md` 에 네 정의를 정확히 적는다:
무엇을 세는가(텐서 개수인가 원소 수인가), 언제 재는가, 0의 의미는 무엇인가.
`trainable_params` 는 지금 **텐서 개수**이지 원소 수가 아니다(`docs/support-matrix.md`).
머지 단계가 두 노트를 대조한다.

## 작업 4 — 피크 메모리와 OOM

메모리가 축 목록에 아예 없었다. 프레임워크마다 VRAM 이 조용히 다르면(활성화 체크포인팅
기본값이 다르고 옵티마이저 상태 배치가 다르다) 속도만 비교하는 것은 비교가 아니다.

**OOM 은 그 자체로 하나의 결과 범주다. "느림"이 아니다.**
지금 설계는 그것을 "미지원"으로 파일링할 위험이 있다 — OOM 레코드는 metrics 도 probe 도 없어서
`report.cell` 이 `launch_state` 로 떨어지고 **시도조차 안 한 조합처럼 렌더된다.**
표시는 report 레인이지만 **레코드가 OOM 을 구별할 수 있어야** 그것이 가능하다.

## 작업 5 — 시퀀스 길이 축

`data.max_seq_len` 을 축으로 만들고 스코프 라벨(single-GPU, seqlen=N)을 결과 산문에 강제한다.
프레임워크 속도비는 (모델, 시퀀스 길이, GPU 수)의 함수이고 **그 공간 안에서 순위가 뒤집힌다** —
한 측정에서 시퀀스가 길어지자 speedup 이 1.7배에서 1.1배로 떨어졌다.

`scripts/audit_plan.py` 의 `axis-values`/`axis-fields` 가 이것을 축으로 인식해야 한다.

## 작업 6 — seed 정책과 3% 임계값 (스키마만)

**스키마는 만들되 값은 파드가 정한다.**

- **seed**: MLPerf CLOSED 는 `/dev/urandom` 에서 뽑고 run 마다 기록하며
  "no other run can log the same seed on the same line"을 요구한다. 고정 seed 로 반복하면
  분포가 아니라 한 점을 재측정하는 것이다. **반복 런마다 다른 seed 가 기록되는 스키마**를 만든다.
  정책 변경은 노이즈 바닥 측정 후다
- **3% 임계값**: `AGENTS.md` 의 3% 는 근거 없는 상수다. GPU 경합만으로 표준편차 30배·평균 +21%가
  관측된 사례가 있다. **config 에서 읽고 현재 값을 "미교정"으로 표시**한다

## 작업 7 — trackio 스키마 제거 (결정 3)

`run.trackio_project`, `run.trackio_space_id` 는 `config-consumed` 가 잡는 **진짜 미소비**다.
스키마 쪽을 이 레인이 제거한다. `configs/run/*.yaml` 4개와 `pyproject.toml` 의 `tracking` extra 는
**report 레인**이 제거한다. **경계에서 맞춘다** — 한쪽만 빠지면 config 합성이 깨진다.
`.plans/notes/measure.md` 에 무엇을 언제 뺐는지 적는다.

## 완료 조건

1. 실토큰·패딩 토큰이 따로 기록되고 프레임워크 자체 tokens/s 를 쓰지 않는다 →
   `uv run pytest tests/test_metrics.py -k token_accounting`
2. warmup·반복 횟수·계측기·집계 통계가 전부 config knob →
   `uv run pytest tests/test_metrics.py -k statistics`
3. `grad_norm` 0 또는 loss 비감소 런이 속도 결과로 기록되지 않는다.
   **테스트는 실제 모양으로 돈다** — 전 파라미터를 얼리고 임베딩 출력에 훅을 걸어 backward 가
   유한한 loss 를 내게 한다(unsloth 가 낸 바로 그 모양) →
   `uv run pytest tests/test_metrics.py -k validity_gate`
4. 피크 메모리가 속도와 함께 기록되고 OOM 이 별도 범주 →
   `uv run pytest tests/test_metrics.py -k peak_memory`
5. `data.max_seq_len` 이 축으로 인식된다 →
   `infisical run --env=dev -- uv run python scripts/audit_plan.py`
6. seed·임계값 스키마가 있고 값이 "파드가 답할 질문"으로 등록된다 →
   `uv run pytest tests/test_config.py -k seed_policy`
7. 각 검사를 되돌리면 죽는다. mutation 출력 인용. **사보타주 전에 `co_filename`/`co_firstlineno`
   확인**(`HAZARDS.md §3`)
8. 네 게이트. `config-consumed` 는 trackio 2개가 빠져 줄어든다 — **`docs/audit-baseline.json` 을
   고치지 않는다.** 줄어든 것 자체가 `shrank` 로 BLOCK 되며 그것은 머지 단계가 처리한다.
   너는 `.plans/notes/measure.md` 에 "config-consumed 에서 trackio 2건 제거" 라고만 적는다

## 하지 않는 것

- `trainbench/collate.py` 의 카운트 지점 — **packing** 레인. 필드가 필요하면 `boundaryRequests`
- `trainbench/probe/steps.py` — **probe** 레인
- `scripts/bench.py` — split 이 이미 정리했다. 여기서 더 손대지 않는다
- `scripts/report.py` 의 표시 — **report** 레인
- `configs/run/*.yaml`, `pyproject.toml` — **report** 레인
- `docs/audit-baseline.json`, `scripts/audit_plan.py` — **통합자 전용**
