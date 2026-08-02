# 남은 코드 작업 전부 (파드 수치가 필요 없는 것)

## Context

Phase 0가 두 번 돌았다. **5/18 → 12/18**이고, 네 프레임워크가 세 모델을 모두 적재하며
**같은 파라미터 수를 학습한다**(625 / 473 / 988). 프레임워크 간 속도 비교의 전제가 처음 섰다.

조사 셋이 이 계획의 근거다 — 저장소 전수 조사(파일:줄), varlen 도달 가능성 조사, 그리고
크로스 프레임워크 벤치마크 방법론의 현재 표준 조사. 마지막 것이 **축 목록의 큰 구멍 여섯을
찾았고 제 가정 둘을 반박했다.**

**범위**: 파드 없이 쓰고 검증할 수 있는 것 전부. 제외 = 숫자를 만들어야 답이 나오는 것.

---

## 결정 (2026-08-02)

| # | 결정 | 선택 | 따라오는 것 |
|---|---|---|---|
| 1 | axolotl dtype | **autocast로 감싸 axolotl을 그대로 잰다** | `CONTRACTS §2` 개정 |
| 2 | packing 격리 | **`cu_seqlens`를 varlen 커널로 전달** | 조사가 전제를 뒤집어 범위 축소 |
| 3 | trackio | **스키마에서 제거** | `configs/run/*.yaml` 4개 + `pyproject.toml` extra |
| 4 | 버전 교란 | **`report.py`가 같은 스택끼리만 줄 세운다** | 스택이 다른 셀의 동일 순위표 편입 거부 |
| 5 | 어댑터 경계 | **프레임워크의 학습 스텝을 그대로 잰다** | **축 소유권 상태 신설** |
| 6 | `kernel=kernels_hub` | **축 값을 버린다** | `kernel`은 none/liger/fla 셋 |
| 7 | 방법론 보강 범위 | **리서치가 찾은 새 축까지 전부** | 아래 A/B/D/G/H/I |

---

## 축 표 (커버리지 기록)

| 축 | 상태 | 근거 |
|---|---|---|
| 프레임워크 간 측정 의미론 | **decided** | 결정 1·4·5 |
| 축 적용 검증 계약 (capture) | **decided** | 레인 C |
| 어댑터 경계 | **decided** | 결정 5 + G(지문) |
| 공허한 검사 | **decided** | 레인 B |
| 아티팩트 신원 | **decided** | 레인 B + J(digest, 이미 있음) |
| 레인 경계·파일 소유권 | **decided** | 아래 레인표. *외부 표준 없음 — null result* |
| 범위 경계 (코드 vs 파드) | **decided** | 이 문서의 "제외" |
| 증거 기준 | **decided** | 각 레인 완료 조건 |
| **A 학습 유효성 게이트** | **decided** | 레인 D |
| **B 토큰 회계 계약** | **decided** | 레인 D |
| **D 측정 통계** | **decided** | 레인 D |
| **G 빌드 지문 + 커널 provenance** | **decided** | 레인 E + F |
| **H 피크 메모리 / OOM 범주** | **decided** | 레인 D |
| **I 시퀀스 길이 축 + 스코프 라벨** | **decided** | 레인 D |
| C 수렴 등가성 (loss parity) | **open — 파드** | 설계는 레인 D, 측정은 파드 |
| E GPU clock/power lock | **open — 파드** | RunPod 컨테이너에서 `nvidia-smi -lgc` 가능한지 미확인 |
| F 호스트 지문 | **open — 파드** | 같은 A100 이름이 같은 기계가 아님 |
| K 에너지 | **not applicable** | MLPerf Power는 optional, 2026 필수 근거 미확인 |

---

## 리서치가 반박한 제 가정 둘 — 문서에 올린다, 지금 고치지 않는다

**seed 고정.** MLPerf CLOSED는 seed를 `/dev/urandom`에서 뽑고 run마다 기록하며
*"no other run can log the same seed on the same line"*을 요구한다. 우리는 고정 seed를 쓴다 —
같은 seed로 반복하면 분포가 아니라 **한 점을 재측정**하는 것이다. 반복 run 정책(레인 D의 D축)과
함께 결정되어야 하므로 레인 D가 스키마를 만들되, 정책 변경은 노이즈 바닥 측정 후다.

**3% 임계값.** GPU 경합만으로 표준편차 30배·평균 +21%가 관측된 사례가 있다. 3%가 실측 노이즈
바닥보다 작으면 유효한 run을 계속 기각한다. `AGENTS.md`의 3%는 근거 없는 상수이고, 첫 파드에서
canonical baseline 10회 반복으로 유도해야 한다.

**null result 둘.** 프로파일러 오버헤드에 대한 외부 표준이 **존재하지 않는다**(MLPerf 규칙
전문에 계측 오버헤드 조항 없음) — `methodology.md` §1은 직접 재는 것 외에 길이 없다.
`optimum-benchmark`가 우리 6종 중 3종을 커버하므로 config 스키마와 report 필드는 그것을 참조한다.

---

## 확정된 사실

### packing 격리는 대부분 이미 되어 있다

`transformers` 5.14.1이 `position_ids` 재시작만으로 블록 대각 격리를 만든다
(`masking_utils.py:735-764`, `:972-975`, `:718-728`). `PackedCollate`가 세 전제를 만족한다
(`masking_utils.py:858-867`).

| arch | 격리 | 근거 |
|---|---|---|
| `qwen3_vl` / `gemma4` | **됨** | `modeling_qwen3_vl.py:800-814`, `modeling_gemma4.py:1696-1708` |
| `qwen3_5` | **절반** | linear_attention이 `position_ids`를 안 보고 `kwargs.get("cu_seq_lens_q")`를 읽는다 (`modeling_qwen3_5.py:549`). torch fallback(`:248-258`)은 그것도 삼킨다 |

**마스크를 직접 만들면 해롭다** — 4D `attention_mask`가 non-None이면 `masking_utils.py:855-856`
에서 조기 반환해 fa2 varlen 경로가 꺼진다. `methodology.md:495-533`의 §10.1/§10.2는 틀렸다.

**그리고 커스텀/Hub 커널이면 다시 열린다** — `attn_implementation` 이름이
`AttentionMaskInterface`에 등록돼 있지 않으면 transformers가 **마스크 생성을 건너뛰고
`attention_mask=None`을 넘긴다**(공식 문서). 격리 상실이 검사 가능한 조건이 됐다.

### 이미지 digest가 커널을 고정하지 못한다

`kernels` 라이브러리가 attention 커널을 **런타임에 Hub에서** 해석하고 `AttentionInterface`에
자동 등록한다. `kernels.lock` + `get_locked_kernel()`이 있으나 우리는 안 쓴다. 멀티모달은
`attn_implementation`을 sub-config별 dict로 받으므로, 문자열 하나가 어느 타워에 걸렸는지가
모델마다 다르다 — `methodology.md` §9의 `kernel_modules` 미측정과 같은 자리다.

### 두 프레임워크가 막힌 지점

**tevatron** — `transformers 5.14.1`이 composite config에서 `pad_token_id`를 최상위에서
옮겼다(실측: `Gemma4Config`/`Qwen3VLConfig`/`Qwen3_5Config` 없음, `get_text_config()`에 있음;
비-composite는 둘 다 있음). 핀 고정 `encoder.py:166-168`이 최상위를 읽는다. `config=`를
`hf_kwargs`로 넘긴다 — `:166`이 `**hf_kwargs`를 전달하고 composite에 `setattr`가 통한다(실측).

**axolotl** — `loaders/model.py:433-436`이 FSDP가 아니면 무조건 `embed_tokens`/`lm_head`를
fp32로 두고, 복귀 분기(`:456-475`)는 `adapter`/FSDP/`cut_cross_entropy` 셋 다 없으면 안 돈다.

### 결정 5가 만드는 새 요구사항 — 축 소유권

tevatron `DenseModel.forward`(`encoder.py:52-87`)가 인코딩·풀링·정규화·스코어링·InfoNCE·
분산 게더를 **전부 자기가 한다.** 프레임워크의 스텝을 그대로 재기로 했으므로 tevatron 셀에서
`loss`와 `parallel.cross_device_negatives`는 우리 것이 아니다. 지금 축 상태는 "미구현"과
"적용됨" 둘뿐이라 `assert_matches`가 그것을 불일치로 읽는다.

**확인 안 함**: sentence_transformers도 자기 손실을 갖는지.

### 구현해도 매칭되지 않는 capture 넷

`applied.py`가 config와 절대 같아질 수 없는 값을 돌려준다: `adamw_8bit`→`adamw8bit`(`:265-266`),
`zero2/zero3`→`deepspeed`(`:724-728`), `mxfp8/nvfp4`→**영원히 `None`**(`:773`, `:819-825`),
`offload` 4값→`none`/`offloaded(dev)`(`:914-915`). 뒤 둘은 구조 변경이다.

### 축 값의 패키지는 전부 있다

`bitsandbytes`(`envs/native/pyproject.toml:33`), `transformer-engine`(:31), `deepspeed`(:39),
`nvidia-dali`(:41), `liger-kernel`(:22). **의존성 공백이 아니라 코드 공백이다.**

### 가드 셋 중 둘은 이미 있다 (앞선 진술 정정)

GPU 혼용 금지 `orchestrate.py:338-373`, 네트워크 볼륨 금지 `pods.py:283-290`.
`check_axis_not_split`의 사각은 `overrides`가 아니라 **`framework`/`model`이 매니페스트 최상위
필드**라 `axes_touched`(`:272-284`)가 원리적으로 못 보는 것이다.

---

## 새 병목 — `scripts/bench.py`

프롬프트·packing·토큰 회계·측정 통계·유효성 게이트·피크 메모리·어댑터가 **전부 이 한 파일**을
건드린다. 한 파일은 한 레인만 소유한다는 규칙상, 그대로 두면 일곱 레인이 직렬이 된다.

**그래서 레인 D가 먼저 `bench.py`를 모듈로 쪼갠다.** 이후 레인들은 파일을 나눠 소유한다.

```
scripts/bench.py            얇은 진입점 (레인 D 소유)
trainbench/collate.py       Collate, PackedBatches, MicroBatch   (신설, 레인 D가 만들고 F/G가 소유)
trainbench/metrics/         토큰 회계, 통계, 피크 메모리          (레인 D 소유)
trainbench/loader.py        프레임워크 어댑터 레지스트리          (신설, 레인 D가 자리만, 레인 G 소유)
```

---

## 레인표

| lane | 소유 | security |
|---|---|---|
| A — tevatron 적재 | `trainbench/probe/tevatron.py` | |
| B — 가드·위생 | `scripts/report.py`, `scripts/orchestrate.py`, `scripts/prepare_data.py`, `docker/entrypoint.sh`, `trainbench/probe/sentence_transformers.py`, `configs/run/`, `pyproject.toml` | true |
| C — capture 구조 | `trainbench/applied.py` | |
| D — 측정 계약 + bench 분해 | `scripts/bench.py`, `trainbench/metrics/`, `trainbench/probe/steps.py`, `trainbench/config_schema.py` | |
| E — 커널 provenance | `trainbench/kernels.py`(신설), `docs/methodology.md` | true |
| F — packing + 프롬프트 | `trainbench/collate.py`, `trainbench/prompt.py`, `configs/model/` | |
| G — 어댑터 + 지문 | `trainbench/loader.py`, `trainbench/probe/`(나머지 5종) | |
| H — 축 구현 | `trainbench/axes.py`, `configs/` | |
| I — 통합 | `AGENTS.md`, `PLAN.md`, `docs/CONTRACTS.md`, `docs/support-matrix.md`, `README.md` | |

`docs/open-verdicts.json`은 레인 I 소유 — 여러 레인이 판정을 닫으므로 마지막에 한 번 모은다.

---

## 경계표

| name | lanes | test | sample |
|---|---|---|---|
| collate-metrics | lane-d, lane-f | tests/contract/test_collate_metrics.py | tests/fixtures/microbatch.sample.json |
| loader-bench | lane-d, lane-g | tests/contract/test_loader_bench.py | tests/fixtures/adapter_out.sample.json |
| applied-axes | lane-c, lane-h | tests/contract/test_applied_axes.py | tests/fixtures/axis_state.sample.json |
| record-report | lane-d, lane-b | tests/contract/test_record_report.py | tests/fixtures/run_record.sample.json |
| kernel-provenance | lane-e, lane-g | tests/contract/test_kernel_provenance.py | tests/fixtures/kernel_fingerprint.sample.json |

경계 테스트는 어느 레인에도 속하지 않고 팬아웃 **전에** 쓰인다.

---

## 레인별 완료 조건

### lane-a — tevatron 적재

- composite config에 `pad_token_id`를 심어 `hf_kwargs`로 넘기고, **모델이 읽지 않는 shim임을
  코드에 남긴다**
  → `infisical run --env=dev -- uv run pytest tests/test_probe.py -k tevatron`
- [pod] tevatron 3셀이 `dense_model_load`를 통과한다 (12/18 → 15/18)
  → verdict: ____ by: ____ at: ____

**범위 밖**: tevatron의 forward 시그니처 (lane-g 소유)

### lane-b — 가드·위생

- `report.py`가 캠페인 아티팩트를 mtime이 아니라 기록된 신원으로 고른다 (실측: timestamp를
  같게 두면 18칸 중 8칸이 지난 캠페인 것을 고른다)
  → `uv run pytest tests/test_report.py -k campaign`
- `report.py`가 스택이 다른 셀을 한 순위표에 넣지 않는다 (결정 4)
  → `uv run pytest tests/test_report.py -k stack`
- `sentence_transformers` 프로브가 동결 그래프를 통과시키지 않는다
  → `uv run pytest tests/test_probe.py -k sentence_transformers_frozen`
- `axes_verified`가 `all_matched:false`를 통과시키지 않는다
  → `uv run pytest tests/test_probe.py -k axes_verified`
- `check_axis_not_split`이 `framework.name`을 본다 (예외로 두든 규칙으로 표현하든, 침묵하지 않는다)
  → `uv run pytest tests/test_orchestrate.py -k axis_split`
- `entrypoint.sh`의 probe 갈래도 스키마 검증을 거친다
  → `uv run pytest tests/test_pods.py -k preflight_probe`
- `config-consumed`가 0
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- trackio 제거, 중복 validator 제거, `gradcache` 죽은 핀 처리
  → `infisical run --env=dev -- uv run pytest`

### lane-c — capture 구조

- 네 축이 config 값과 같아질 수 있다: `adamw_8bit`, `zero2/zero3`, `mxfp8/nvfp4`, `offload`
  → `uv run pytest tests/test_applied.py -k capture_matches`
- **축 소유권 상태**가 표현된다 — "미구현" / "적용됨" / **"프레임워크 소유"** 셋
  → `uv run pytest tests/contract/test_applied_axes.py`

### lane-d — 측정 계약 + bench 분해

- `bench.py`가 `trainbench/collate.py`, `trainbench/metrics/`, `trainbench/loader.py`로 쪼개지고
  동작이 바뀌지 않는다
  → `infisical run --env=dev -- uv run pytest`
- **토큰 회계 계약(B)**: 처리량의 분모가 config에 명시되고, 하네스가 프레임워크의 자체
  tokens/sec를 절대 쓰지 않고 원시 카운터에서 재계산한다. 패딩 토큰과 실토큰이 별도 필드다
  → `uv run pytest tests/test_metrics.py -k token_accounting`
- **측정 통계(D)**: warmup step 수, 반복 run 수, 타이밍 계측기(CUDA event), 집계 통계가
  전부 config knob이다
  → `uv run pytest tests/test_metrics.py -k statistics`
- **학습 유효성 게이트(A)**: run 레코드가 `grad_norm`, `trainable_params`, `loss[0]`/`loss[-1]`,
  `peak_memory_bytes`를 싣고, grad_norm이 0이거나 loss가 감소하지 않으면 그 run은 속도 결과가
  아니다
  → `uv run pytest tests/test_metrics.py -k validity_gate`
- **피크 메모리(H)**: 속도와 함께 보고되고, OOM이 "느림"이 아니라 별도 결과 범주다
  → `uv run pytest tests/test_metrics.py -k peak_memory`
- **시퀀스 길이 축(I)**: `data.max_seq_len`이 축이 되고, 결과 서술에 스코프 라벨이 강제된다
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- seed 정책과 3% 임계값은 **스키마만 만들고 값은 파드가 정한다**
  → 문서에 "파드가 답할 질문"으로 등재

**범위 밖**: 노이즈 바닥 실측, loss parity 실측 (둘 다 파드)

### lane-e — 커널 provenance

- 해석된 attention 커널의 출처(repo + revision)가 run 레코드에 남는다
  → `uv run pytest tests/contract/test_kernel_provenance.py`
- `attn_implementation`이 `AttentionMaskInterface`에 등록돼 있는지 검사하고, 아니면 **packing을
  거부한다** (미등록 = 마스크가 조용히 사라짐)
  → `uv run pytest tests/test_kernels.py -k mask_registered`
- 파드에서 네트워크 커널 fetch를 금지하고 사전 다운로드한다 (NVMe 규칙의 확장)
  → `uv run pytest tests/test_kernels.py -k no_runtime_fetch`
- `methodology.md` §9/§10.1/§10.2 정정

### lane-f — packing + 프롬프트

- Qwen 지시 프롬프트가 템플릿된 행에 **정확히 한 번** 나온다
  → `uv run pytest tests/test_prompt.py -k appears_once`
- `arch=qwen3_5`에 `cu_seq_lens_q/k` + `max_length_q/k` 넷이 전달되거나, 그 arch에서
  `packing=true`가 거부된다 (**넷 다 아니면 아무것도 아님**)
  → `uv run pytest tests/test_collate.py -k varlen`
- `create_causal_mask`가 `PackedCollate` 출력에 대해 블록 대각을 돌려준다 (CPU)
  → `uv run pytest tests/test_collate.py -k isolation`

### lane-g — 어댑터 + 지문

- 여섯 프레임워크가 `trainbench/loader.py`의 공통 진입점으로 적재되고, `bench.py`가 실제
  프레임워크 이름을 `assemble`에 넘긴다
  → `uv run pytest tests/contract/test_loader_bench.py`
- 어댑터가 **빌드된 모델 지문**을 반환한다 — 모듈 클래스명, param별 dtype, trainable param
  이름 집합, 바인딩된 attention fn identity. 프레임워크가 *요청하지 않은* 것을 무엇으로
  바꿨는지가 여기서 드러난다
  → `uv run pytest tests/test_loader.py -k fingerprint`
- tevatron의 `forward` 시그니처가 어댑터 경계에서 표현되고, `loss`/`cross_device_negatives`
  축이 그 셀에서 **프레임워크 소유**로 기록된다
  → `uv run pytest tests/test_loader.py -k tevatron_owns`
- **권장 경로 대조.** 여섯 프레임워크 각각에 대해, 그 프레임워크의 공식 문서·예제가 지목하는
  학습 진입점을 핀된 소스에서 찾아 인용하고, 우리가 쓰는 것과 다르면 그 차이를 어댑터에
  기록한다. 지금 여섯 중 다섯이 "적재만 그쪽, 학습 루프는 우리 것"이다:

  ```
  native                 AutoModel.from_pretrained        레퍼런스, 일치
  unsloth                FastVisionModel.from_pretrained  for_training() 미사용
  ms_swift               get_model_processor              자체 트레이너 미사용
  sentence_transformers  SentenceTransformer(...)         자체 손실·트레이너 미사용
  tevatron               dense.load(...)                  forward가 전체 스텝
  axolotl                ModelLoader(cfg, tok).load()     자체 Trainer 미사용
  ```

  오늘 난 세 건(unsloth `full_finetuning` 누락 → 전 파라미터 동결, axolotl `validate_config`
  건너뜀, tevatron forward 오용)이 **전부 이 격차에서 나왔고 셋 다 답이 핀된 소스 안에 있었다.**
  필드에도 같은 사례가 있다 — 어떤 재현 연구가 unsloth의 46,000 tok/s에서 grad norm 0을 관측했다.
  → `uv run pytest tests/test_loader.py -k documented_entry_point`

### lane-h — 축 구현

- `adamw_8bit`, `mxfp8/nvfp4`, `offload` 3값, `ddp/fsdp2/zero2/zero3`, `dali`, `qlora`,
  `liger`(qwen3_5)
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py` — `axis-values`에
    단일값 그룹 0
- `kernels_hub`가 스키마에서 제거되고 이유가 남는다 (결정 6)

### lane-i — 통합

- `AGENTS.md ## Commands`에 Run / Report / one-setting 세 줄 추가 (mtime 경고 포함)
- `PLAN.md`에 신설 파일 등재 → `plan-files` PASS
- `docs/open-verdicts.json` 종결 취합
- [human] `docs/support-matrix.md`가 스택이 다른 셀을 나란히 놓지 않는다
  → verdict: ____ by: ____ at: ____

---

## 전체 완료 조건

```
uv run ruff check && uv run ruff format --check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/audit_plan.py
infisical run --env=dev -- uv run python scripts/env_report.py \
  device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4
```

| 체크 | 지금 | 목표 |
|---|---|---|
| `axis-values` | 36/53, 단일값 3그룹 | 단일값 0그룹 |
| `config-consumed` | 4 | 0 |
| `verdicts-closed` | 4 open | 2 open (남는 둘은 파드) |
| Phase 0 | 12/18 | 18/18 |

**부숴서 확인하지 않은 검사는 증거가 아니다.** 각 레인은 자기가 넣은 검사를 되돌려 죽는 것을
보고 그 출력을 기록한다. 무력한 변이는 숨기지 않는다.

**핀된 소스를 읽고 나서 단언한다** (`AGENTS.md ## Verification`).

## 이 범위에서 제외

- 파드가 숫자를 만들어야 답이 나오는 것 — 노이즈 바닥과 3% 임계값 유도, 프로파일러 오버헤드,
  deterministic on/off, 데이터로딩 병목 판정, loss parity, GPU clock lock 가능 여부, 호스트 지문,
  liger 커버리지 분포, `loss-empty-pixel-slice`, `loss-gradcache-memory-and-overhead`
- 캠페인 실행 자체 (Phase 2 ablation, Phase 3 프레임워크 측정, 품질 런)
- `docs/report.md` — 산출물이고 숫자가 나와야 쓴다
