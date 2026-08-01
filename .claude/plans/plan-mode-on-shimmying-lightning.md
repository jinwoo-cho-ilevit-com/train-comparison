# Wave 2.5 (LoRA) + Wave 3 (측정 하네스)

## Context

Wave 2가 닫혔다. 축 15개에 적용 지점과 capture probe가 붙었고, 축별 패키지가 env에
들어갔고, 감사 계층이 실패 중인 체크 안의 변화를 보게 됐다(게이트 8/12, pytest 337).

**그런데 이 저장소는 아직 학습을 돌릴 수 없다.** `scripts/bench.py`도
`trainbench/metrics/`도 없다. 결과적으로:

- 축이 학습 스텝에서 켜진 적이 **0회**다. D가 검증한 것은 전부 가짜 객체 상대다.
- throughput·step time·peak VRAM을 계산하는 코드가 저장소 어디에도 없다
  (`max_memory_allocated`/`torch.cuda.Event`/`synchronize`/`torch.profiler` 호출 0건).
- `docs/methodology.md`가 두 개의 측정 부채를 "bench.py가 생기면 실행한다"로 미뤄뒀다
  — 프로파일러 오버헤드, deterministic on/off 비용. 둘 다 보고 숫자 바로 밑에 깔린다.
- `axis-values`가 12개 ablation 그룹 중 7개는 아직 비활성값 하나만 받는다고 보고한다.

그리고 `peft=lora`가 `UnappliedAxis`로 죽는다. 거부는 옳지만 `PLAN.md`의 표제 산출물
절반("full FT vs LoRA 손익분기점")이 막혀 있다.

**사용자 결정 2건 (2026-08-01)**

1. **LoRA는 Wave 3 착수 전 필수로 구현한다.**
2. **sweep은 설정마다 프로세스를 재기동한다.** 재사용된 프로세스는 autotune 캐시,
   컴파일된 그래프, 얼로케이터 단편화를 다음 설정으로 넘긴다. `kernel`·`attn` 축은
   애초에 모델 재생성이 필수라 재사용이 불가능하다.
3. **측정 단위는 `train.steps`를 유지하고 실제 소비 토큰을 측정해 기록**한다.
   `PLAN.md`가 "고정 토큰 예산 기준(고정 step 아님)"이라고 적고 있으므로 **그 서술을
   고쳐야 한다**(레인 E). 코드와 문서가 어긋난 채로 두지 않는다.

---

## Wave 2.5 — LoRA

순서가 있다. 뒤집으면 검증 없이 정의를 박게 된다.

1. **F 영역**: `peft`를 `envs/native`에 넣고 재-lock. 축 패키지들과 같은 방식.
2. **의미론 결정 — 실제 peft 모델로 확인한 뒤.** peft는 base 파라미터를 전부 얼린다.
   `freeze.ple=false` 요청이 "얼지 않음"인지 "peft가 얼린 것에 더해 얼지 않음"인지
   정해야 한다. `docs/CONTRACTS.md` §2가 이 충돌을 예고해뒀고, D는 검증할 수 없어
   거부를 택했다. 이번엔 패키지가 있으니 실물로 확인하고 정한다. 결정 자체가 리포트의
   한정 조건이 되므로 `docs/model-spec.md`나 `methodology.md`에 근거와 함께 남긴다.
3. **D 영역**: `axes._peft` 적용 + capture 확장. capture는 `peft_config`/래퍼 클래스/
   4bit 여부를 읽는 구현이 이미 있다. `configs/peft/{lora,qlora}.yaml`도 이미 있다.
4. **`axis-values`가 움직이는지 확인** — peft 1/3 → 3/3이 되어야 한다. 안 움직이면
   구현이 값을 받지 못한 것이다.

---

## Wave 3 — 측정 하네스 (순차, 병렬 금지)

### 3-0. 소유권 이관을 먼저 한다

`docs/CONTRACTS.md:42-47`이 명시한다: **Wave 3 착수 시점에 `docker/entrypoint.sh`와
`scripts/orchestrate.py`의 `RUNNABLE_PURPOSES`가 레인 C에서 G로 이관된다.** 이관을
기록하지 않고 손대면 병합된 레인의 파일을 되돌리는 일이 된다. `CONTRACTS.md` §1 표와
계약 변경 이력을 함께 갱신한다.

### 3-1. `scripts/bench.py`

`assert-called`가 요구하는 것은 AST에서 보이는 **실제 호출** 5개다
(`scripts/audit_plan.py:382`, `ENTRY_POINT_CALLS`): `patch`, `load_kwargs`, `assemble`,
`step_context`, `assert_matches`. 문자열·docstring은 만족시키지 못한다.

**함정 하나**: `trainbench/probe/steps.py:38` `verify_axes`가 `assemble`/`capture`/
`assert_matches`를 이미 호출하지만 `report.run(...)`으로 감싸서 **예외를 삼킨다.**
그대로 재사용하면 `assert-called`는 초록인데 불일치가 런을 못 멈춘다 — `CONTRACTS.md`
§2가 경고한 바로 그 실패다. bench.py는 `assert_matches`를 **직접** 호출한다.

**재사용할 것** (전부 존재, 재작성 금지):

| 이미 있는 것 | 위치 |
|---|---|
| forward + InfoNCE + backward | `trainbench/probe/steps.py:279` `infonce_backward` |
| pooling / 패딩 방향 강제 | `trainbench/embedding.py:22`, `steps.py:80` |
| 배치 구성(텍스트·이미지) | `steps.py:115`, `:140` |
| visual token 계수 — tokens/s의 분모 | `steps.py:202` `visual_token_count` |
| 레코드 골격 + 호스트 스펙 | `trainbench/record.py:156` `build_record`, `:132` `host_spec` |
| device/seed | `trainbench/device.py`, `trainbench/seed.py` |

**새로 만들 것**: 옵티마이저 스텝(`Built.optimizer`가 지금 쓰이지 않는다),
`step_context()` 래핑, grad-accum 내부 루프, MMEB 서브셋을 읽는 실제 데이터로더,
warmup 폐기 + CUDA sync를 갖춘 타이밍 루프, 그리고 지표.

`verify_env.py:36`이 `deterministic=True`를 하드코딩한다. 타이밍 런은
`config.train.deterministic`을 따라야 한다(스키마가 `purpose=timing`에서 False를 강제).

### 3-2. `trainbench/metrics/`

step time p50/p95, samples/s, tokens/s, peak VRAM. MFU는 **tokens/s를 1차 지표로 두고
격하**한다 — GDN linear attention·PLE lookup·sliding window에서 표준 FLOP 공식이 세
모델 모두 깨지므로, 모델별 공식을 유닛 테스트로 검증한 뒤에만 제시한다.

peak VRAM은 `torch.cuda.max_memory_allocated` + `reset_peak_memory_stats`로 잰다.
`prepare_data.py:792` `_peak_rss_bytes()`가 호스트 RSS에 대해 같은 일을 하고, 그
테스트 docstring이 왜 샘플링이 아니라 high-water mark여야 하는지를 이미 적어뒀다.

레코드에는 `build_record(..., metrics=...)`로 싣는다. `**extra`가 확장 지점이고
`tests/test_pods.py:669`가 `metrics` 키를 이미 전제한다.

### 3-3. entrypoint의 sweep 루프

`docker/entrypoint.sh:33-35`가 `resolved_plan.json`을 쓰지만 **아무도 읽지 않는다.**
결정대로 entrypoint가 그 목록을 읽어 **설정마다 bench.py를 새로 실행**한다.
`:107-120`의 purpose 분기에 `timing|profile|quality` 팔을 추가한다. 그 뒤
`RUNNABLE_PURPOSES`를 넓힌다 — 순서가 반대면 실행 불가능한 pod을 띄우게 된다.

### 3-4. 측정 전에 실측으로 정해야 하는 것들

이것들이 없으면 나오는 숫자를 해석할 수 없다.

- **데이터로딩 병목 선판정.** 이게 병목이면 Phase 2 전체가 무의미하다.
- **모델별 visual token 분포 실측.** 196:196:280 보정은 448 정사각 1장 가정이다.
  Qwen은 픽셀 비례, gemma-4는 280 고정이라 모델 간 토큰 예산 고정이 원리적으로
  불가능할 수 있고, 그러면 리포트 범위를 "모델 내 축 효과"로 좁혀야 한다.
- **baseline 3% 임계값 교정.** 동일 pod 동일 설정 5회 반복 편차를 먼저 재고 그 2~3배로
  잡는다. 지금 값은 근거 없이 적힌 숫자다.
- **profiler 오버헤드 / deterministic on-off 비용.** `docs/methodology.md:57-62`가
  절차까지 적어뒀다. 재기 전까지 어떤 퍼센트도 인용하지 않는다.
- **GradCache 수치 등가성** — 검증 없이 재면 GradCache 버그가 GradCache 속도 향상으로
  보고된다.

### 3-5. 남은 knob 소비

`config-consumed` 13개 중 하네스 몫: `train.steps`, `train.grad_accum`,
`train.warmup_discard_steps`, `run.profiler`, `run.trackio_*`, `model.pooling`,
`model.add_generation_prompt`, `model.instruction_prompt`.

`data.*` 4개는 **오탐이다** — `prepare_data.py:1020`의 `data = config.data` 별칭 때문에
`_reads_dotted`가 못 본다. 게이트 리뷰의 판정은 체커가 아니라 **호출부를 고치는 것**
이었다(별칭 추적은 `data`라는 이름의 함수 파라미터를 통한 읽기를 못 잡거나, 잡으려면
이미 제거된 오탐 계열을 되살린다). knob당 최소 1회를 `config.data.X`로 읽게 한다 —
레인 A, 약 4줄.

---

## 레인 E 인계 (문서, 이번에 직접 고치지 않음)

`docs/review-findings.md`의 인계 절에 추가한다. 앞의 3건은 이미 적혀 있다.

- **`PLAN.md`의 "고정 토큰 예산 기준 비교(고정 step 아님)"** — 결정과 어긋난다.
  step 기준 + 토큰 실측 기록으로 고친다.
- **`trainbench/metrics/`가 `PLAN.md` 저장소 구조 블록에 없다.** `plan-files` 체크가 그
  블록을 트리로 파싱하므로 디렉터리를 만들면 문서도 함께 고쳐야 한다.
- **LoRA 관련 서술**은 Wave 2.5가 구현하면 유지된다. 구현이 미뤄지면 그때 철회한다.
- `docs/support-matrix.md`의 "이미지 7개 중 6/7 성공" 표가 **존재하지 않는 Dockerfile**을
  서술한다(F가 base·framework를 바꾸고 native를 약 30 → 142 패키지로 키웠다).
- gemma-4의 `audio_tower` 751 텐서(2011의 37%)를 어떤 freeze 축도 건드리지 않는다.
  이미지만 쓰는 벤치마크에서 학습되지도 얼려지지도 않은 채 옵티마이저 상태를 차지하므로
  `docs/methodology.md`에 기록이 필요하다.

---

## 이번 범위에서 제외

- **B200 pod 기동.** Wave 3 완료 + audit 전항목 통과 전까지 금지다. 그 시점에 Infisical
  권한 결정이 필요하다 — pod이 필요한 시크릿은 `HF_TOKEN` 1개인데 28개에 도달하며
  코드 가드가 기동을 막고 있다.
- **프레임워크 probe 4종 API 수정**(tevatron forward 시그니처, axolotl `normalize_config`,
  unsloth `for_training`) — 이미지가 빌드돼야 검증 가능하다.
- **`tests/test_smoke_cpu.py`로 timing 경로를 검사하는 것.** CPU에서는 `adamw_fused`가
  `adamw_unfused`로 해석돼 영구 mismatch다(`CONTRACTS.md` §6). CPU 스모크는
  `purpose=probe`만 가능하다.

---

## 검증

```
uv run ruff check && uv run ruff format --check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/audit_plan.py
infisical run --env=dev -- uv run python scripts/env_report.py \
  device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4
```

게이트가 움직여야 하는 방향:

| 체크 | 지금 | Wave 3 후 |
|---|---|---|
| `assert-called` | 실패(파일 없음) | PASS |
| `axis-values` | 25/43, 7개 그룹 | peft 3/3으로 최소 1그룹 해소 |
| `config-consumed` | 13개 | 하네스 몫 소비 후 감소 |

**부숴서 확인하지 않은 검사는 증거가 아니다.** 각각에 대해:

- `bench.py`: 5개 호출 중 하나를 지우면 `assert-called`가 실패하는가
- `assert_matches`: 축 하나를 불일치시키면 `purpose=timing` 런이 **숫자를 내기 전에**
  죽는가. 이 프로젝트에서 가장 중요한 단일 안전장치이고, 지금까지 실제 런에서 한 번도
  발동한 적이 없다
- 지표: 알려진 입력(고정 step, 고정 배치)에서 손계산과 일치하는가
- peak VRAM: 의도적으로 큰 배치가 값을 올리는가

이 저장소는 검사가 자기 부재를 통과로 보고한 사례를 하루에 다섯 번 냈다(D1 행 수 게이트,
D6 소표본 스모크, D7 형식만 보는 `data-pinned`, capture 커버리지 구멍, 멤버십만 보는
baseline). Wave 3은 **처음으로 실제 숫자를 만드는 wave**이므로 같은 형태의 실패가
여기서 나오면 그 숫자가 리포트에 실린다.
