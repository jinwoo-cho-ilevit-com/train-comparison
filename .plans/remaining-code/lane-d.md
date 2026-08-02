# lane-d — 측정 계약 + bench 분해

## Scope

**이 레인이 먼저 돈다.** `scripts/bench.py`를 다른 레인들이 나눠 가질 수 있게 쪼개고, 그 위에
측정 계약을 세운다. 리서치가 찾은 축 넷(A 학습 유효성, B 토큰 회계, D 측정 통계, H 피크 메모리)과
축 I(시퀀스 길이)가 전부 여기다.

## Owns

- `scripts/bench.py`
- `trainbench/metrics/`
- `trainbench/probe/steps.py`
- `trainbench/config_schema.py`

## 할 일

### 0. `bench.py` 분해 — 다른 레인의 선행

프롬프트·packing·토큰 회계·측정 통계·유효성 게이트·피크 메모리·어댑터가 전부 이 한 파일을
건드린다. 한 파일은 한 레인만 소유하므로 그대로 두면 일곱 레인이 직렬이 된다.

```
scripts/bench.py        얇은 진입점                     이 레인 소유
trainbench/collate.py   Collate, PackedBatches, MicroBatch   신설 → lane-f 소유
trainbench/metrics/     토큰 회계, 통계, 피크 메모리          이 레인 소유
trainbench/loader.py    프레임워크 어댑터 레지스트리          신설, 자리만 → lane-g 소유
```

**동작이 바뀌면 안 된다.** 분해는 순수 이동이고, 그 사실이 기존 스위트로 증명되어야 한다.

### 1. 토큰 회계 계약 (축 B)

처리량의 **분모가 정의돼 있지 않다.** 패딩 토큰을 세느냐 실토큰을 세느냐에 따라 dataloader
축(packing on/off)의 순위가 뒤집힌다. NLP 데이터셋에서 패딩이 전체 토큰의 최대 50%,
GLUE-CoLA seq128에서는 89%라는 측정이 있다.

그리고 임베딩 특유의 함정: in-batch negatives에서 **배치 크기는 목적함수의 일부**(negative
개수)다. packing이 effective batch를 바꾸는 순간 packing은 속도 축이 아니라 목적함수 축이 된다.

**계약**:
- 실토큰 수와 패딩 토큰 수가 **별도 필드**다
- 하네스가 프레임워크의 자체 `tokens/sec`를 **절대 쓰지 않고** 원시 카운터에서 재계산한다
- 분모가 무엇인지가 config에 명시되고 결과에 실린다

### 2. 측정 통계 (축 D)

지금 config에 warmup step 수, 반복 run 수, 타이밍 계측기, 집계 통계가 **하나도 없다.**
어댑터마다 제각각 정하게 되고 그 차이가 축 효과보다 클 수 있다.

참고 관행(전부 이번 리서치에서 fetch): MLPerf는 Small LLM finetuning에 최소 10회 반복 +
최고·최저 제거 후 평균(Olympic scoring). CUDA event 타이밍(wall clock 아님). torch.compile JIT
때문에 warmup 10 step. heavy-tailed 지연에서는 arithmetic mean 대신 trimmed mean.

전부 **config knob**이어야 한다.

### 3. 학습 유효성 게이트 (축 A)

오늘 unsloth 세 셀이 파라미터가 전부 얼어붙은 채 `infonce_backward`를 통과했다
(`params_with_grad=0`, `trainable_params=0`). `steps.infonce_backward`에 가드를 넣어 막았지만,
그건 `trainable_params`만 본다.

필드에 같은 사례가 있다 — 어떤 재현 연구가 unsloth의 46,000 tok/s에서 **grad norm 0**을
관측했다. 권장 검증은 넷: grad norm이 0이 아닐 것, 학습 가능 파라미터가 기대치와 같을 것,
loss가 감소할 것, 메모리 사용이 기대와 맞을 것.

run 레코드가 `grad_norm`, `trainable_params`, `loss[0]`/`loss[-1]`, `peak_memory_bytes`를 싣고,
**grad_norm이 0이거나 loss가 감소하지 않으면 그 run은 속도 결과가 아니다.**

### 4. 피크 메모리 (축 H)

축 목록에 메모리가 아예 없었다. 프레임워크마다 조용히 다른 VRAM을 쓰면(다른 activation
checkpointing 기본값, 다른 optimizer state 배치) 속도만 비교하는 것은 비교가 아니다.
그리고 **OOM은 "느림"이 아니라 별도 결과 범주**여야 한다 — 지금 설계에서는 미지원으로 파일링될
위험이 있다.

### 5. 시퀀스 길이 축 (축 I)

프레임워크 속도비는 (모델, 시퀀스 길이, GPU 수)의 함수이고 그 안에서 **순위가 뒤집힌다** —
어떤 모델에서 시퀀스가 길어지자 speedup이 1.7x에서 1.1x로 떨어진 측정이 있다. 축 목록에
시퀀스 길이가 없었다.

`data.max_seq_len`을 축으로 만들고, 결과 서술에 스코프 라벨(single-GPU, seqlen=N)을 강제한다.

### 6. seed 정책과 3% 임계값 — 스키마만

둘 다 리서치가 반박한 가정이다. **스키마는 만들되 값은 파드가 정한다.**

- seed: MLPerf CLOSED는 `/dev/urandom`에서 뽑고 run마다 기록하며 "no other run can log the
  same seed on the same line"을 요구한다. 고정 seed로 반복하면 분포가 아니라 한 점을 재측정하는
  것이다. **반복 run마다 seed가 달라지고 기록되는 스키마**를 만든다
- 3% 임계값: GPU 경합만으로 표준편차 30배가 관측된 사례가 있다. 임계값을 config에서 읽게 하고,
  현재 값에 **"미교정"** 표시를 유지한다

## Completion criteria

- 분해 후 동작이 바뀌지 않는다
  → `infisical run --env=dev -- uv run pytest`
- 실토큰과 패딩 토큰이 별도로 기록되고, 프레임워크의 자체 tokens/sec를 쓰지 않는다
  → `uv run pytest tests/test_metrics.py -k token_accounting`
- warmup / 반복 수 / 타이밍 계측기 / 집계 통계가 전부 config knob이다
  → `uv run pytest tests/test_metrics.py -k statistics`
- grad_norm이 0이거나 loss가 감소하지 않은 run이 속도 결과로 기록되지 않는다. 테스트는
  **실제 모양**으로 돈다 — 전 파라미터를 얼리고 임베딩 출력에 훅을 걸어 backward가 유한한 손실을
  내는 그래프(unsloth가 만든 것과 같은 형태)
  → `uv run pytest tests/test_metrics.py -k validity_gate`
- 피크 메모리가 속도와 함께 기록되고, OOM이 별도 범주다
  → `uv run pytest tests/test_metrics.py -k peak_memory`
- `data.max_seq_len`이 축으로 인식된다
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- seed와 임계값 스키마가 있고 값은 "파드가 답할 질문"으로 등재됐다
  → `uv run pytest tests/test_config.py -k seed_policy`
- 위 각 검사를 되돌리면 죽는다
  → 변이 출력 그대로 인용

## Out of scope

- 노이즈 바닥 실측, 3% 임계값 유도, loss parity 실측, 프로파일러 오버헤드 — 전부 파드
- `trainbench/collate.py`의 내용 — 만들기만 하고 소유는 **lane-f**
- `trainbench/loader.py`의 내용 — 자리만 만들고 소유는 **lane-g**
- `configs/run/*.yaml`의 trackio 제거 — **lane-b** 소유. 스키마 쪽만 이 레인이 맞춘다
