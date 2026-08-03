# 측정을 시작한다 — 막고 있던 결함 셋을 걷어내고 첫 숫자를 낸다

## 현재 위치 (2026-08-03)

이 프로젝트는 학습 속도 벤치마크인데 **속도 숫자가 아직 하나도 없다.** 파드 3대가
오늘 그 이유를 답했다: A100에서 이 파이프라인은 한 번도 스텝을 완주한 적이 없다.
첫 배치를 만들다가, 또는 첫 스텝에서 OOM으로 멈춘다.

`main = e6b5f8f`. 게이트: pytest 1203, contract 117, audit 13/15 rc=0.

### 오늘 파드가 답한 것 (전부 실측)

| 런 | 결과 |
|---|---|
| gemma-4 full FT, batch 16 | OOM, peak 83.8GB |
| gemma-4 full FT, batch 4 | OOM — 배치 문제가 아니다 |
| qwen3_5 baseline, batch 16 | OOM, PyTorch 할당 75.10GB |
| qwen3_5 baseline, batch 4 | `collate.py:330` 가드 — 이미지 4장이 4,057 토큰 |

가드는 정상이었다. `shuffle=False`(`axes.py:1583-1588`)라 행 순서는 배치 크기와
무관하고, 코퍼스 전체 스캔에서 첫 위반 구간은 `rows[136,140)`이다. batch 4에서는
배치 34(`34%8=2` → 트레이스백의 worker 2와 일치), batch 16에서는 9번째 스텝.
16은 그 전에 OOM으로 죽었다. 모순 없음.

75GB의 원인은 토큰 수가 아니라 아키텍처다 — GDN 18층의 fp32 conv state와
체크포인팅 없는 vision tower. `_baselines.yaml:44-48`이 이미 적어둔 그대로다.

### 측정을 막고 있던 결함 셋

1. **프로세서 픽셀 예산이 없다.** 워크로드가 데이터 draw의 함수가 되어 baseline의
   3% 편차 게이트를 무너뜨린다. 코퍼스에 batch 1로도 넘치는 행이 있다
   (`rows[180,184)` 16,281 토큰) — Lane I
2. **`bench.py`가 OOM 외 예외에 아티팩트를 안 남긴다** (`bench.py:1121-1123`,
   `is_oom`이 거짓이면 `raise`). 진단 가능한 실패가 "결과 없음"으로 보인다 — Lane F
3. **`infisical run`이 모든 비영 종료 코드를 1로 뭉갠다.** 그리고
   `tests/test_pods.py`의 공유 스텁이 `exec "$@"`였고 `run_entrypoint`가
   `INFISICAL_TOKEN`을 안 넘겨, 테스트가 그 버그가 존재할 수 없는 세계를 재고
   있었다 — **머지됨** (`0ecf77d`)

**넷이 아니라 셋이다.** `HF_HUB_OFFLINE` 로그 세 줄은 실패 보고가 아니라 닫은 문의
목록이고, 통합자가 그것을 오독했다. 제안했던 수정은 콜드 파드의 체크포인트·코퍼스
fetch를 죽여 모든 timing 설정을 `no_result`로 만들었을 것이다 — `bench.py:629-635`가
이미 기록한 사고다. 그 방향을 막는 가드가 `e6b5f8f`에 들어갔다.

## 확정된 결정

| 결정 | 내용 |
|---|---|
| 품질 축 | **안 한다.** Recall@k 구현 없음. 파레토는 2축(속도 × 메모리), 모델 내부 |
| GPU | **A100 통일** (`NVIDIA A100-SXM4-80GB`) |
| `precision` | **mxfp8/nvfp4 제거 완료.** `Literal["bf16"]`, transformer-engine 제거 |
| `attn` | **fa3/fa4 스키마에 남기고 측정 안 함** — Hopper/Blackwell 전용 |
| 모델 | **gemma-4 제외.** 2개 — `qwen3_vl_emb_2b`, `qwen3_5_0_8b` |
| `peft` | **qlora 제외.** `{full, lora}` |
| 픽셀 | `max_pixels=1310720`, `min_pixels=4096`, 두 모델 동일. `max_seq_len` 2048 유지 |

픽셀 값의 근거: 시각 토큰 1개 = `patch_size² × merge_size²` = 1,024 픽셀.
1,310,720은 `Qwen3-VL-Embedding-2B`가 자기 리포에 선언한 값이고(우리 산술이 아니다),
`Qwen3.5-0.8B`는 16,777,216 — **12.8배 차이가 닫으려는 confound다.** 텍스트 실측
`max=270`이므로 최악 행 = 270 + 1,280 = 1,550 / 2,048, 여유 24%.

## 레인 — 파일 소유가 겹치지 않는다

```
개발 → 리뷰 → 개선 → 머지. 리뷰어는 개발한 에이전트가 아니고, 레인의 워크트리에서
게이트를 재실행한다. 통합자는 머지 전에 네 게이트를 스스로 다시 돌린다.
레인은 커밋하지 않는다.
```

| 레인 | 소유 | 상태 |
|---|---|---|
| P | precision 제거 | **머지됨** `a8d913b` |
| X | `entrypoint.sh`, `verify_env.py`, `test_pods.py`, `test_probe.py` | **머지됨** `0ecf77d`, `e6b5f8f` |
| F | `bench.py`, `audit_plan.py`, `test_audit.py`, `test_smoke_cpu.py` | 게이트 검증 중 |
| I | `configs/{data,model}`, `config_schema.py`, `probe/*` | 진행 |
| C | `PLAN.md`, `docs/*.md`, 계약 docstring | 진행 |
| L | `trainbench/pods.py`, `scripts/orchestrate.py` | 진행 |
| G | gemma-4 + qlora 제외 | F·I·C 머지 후 |
| D | `configs/experiment/` 스윕 매니페스트 | G 후 |
| E | 잔해 제거 | 마지막 |

### Lane F — 결함 2와 만료된 전제 둘

`bench.py`에 **네 번째 기록 상태**를 만든다: 예외 타입 + 잘린 traceback,
`metrics` 블록 없음(측정 창이 완료됐다는 주장이 되므로), 기존 3/4/5/124/125/127과
겹치지 않는 종료 코드, `no_result`(= `publish_result.fallback_record` 소유)와
구별되는 어휘. 선례가 둘 있다 — `REFUSED_STATUS`/`REFUSED_EXIT`와
`oom_status`/`OOM_EXIT`.

`assert-called`의 면제는 **`loader.py`의 `owned_axes` 선언에서 파생시킨다.**
`FRAMEWORK_OWNED_STEP_RUNNERS` 등재만으로 면제하면, `loss.name` 소유를 선언하지
않은 프레임워크를 레지스트리에 추가한 순간 하네스가 인증되지 않은 손실을 계산하고
감사가 통과시킨다 — 그 검사가 막으려던 실패를 면제가 되살린다.

### Lane I — 결함 1과 곁의 둘

픽셀 상한은 `configs/data/`에 둔다. 두 모델이 **같은 값**을 쓰는 것이 핵심이고
그건 모델의 속성이 아니라 워크로드의 속성이다. 컴포즈 시점 검증
(`텍스트 상한 + ceil(max_pixels/1024) ≤ max_seq_len`)이 실제 속도 이득이다 —
왕복 16분이 랩톱 2초가 된다.

곁의 둘:
- **padded 포워드가 KV 캐시를 끄지 않는다.** `bench.py:118`(packed)은
  `use_cache=False`를 넘기고 `probe/steps.py:253`(padded, 기본이자 baseline 경로)은
  안 넘긴다. 임베딩 학습에 그 캐시는 절대 읽히지 않는다
- **세 모델 전부 `revision: null`.** 파드 로그의 `@2fc06364...`는 핀이 아니라
  가져온 뒤 기록한 것이다(`git_source: "env"`와 같은 성질). 체크포인트가 갱신되면
  같은 축의 파드가 다른 가중치를 잰다

### Lane L — 증거를 잃지 않는다

종료된 파드의 로그는 404다(직접 확인). `orchestrate.py:1024`가
`pods.terminate()`를 부르고 그 전에 아무것도 로그를 가져오지 않는다. 종료 전에
받아 결과 저장소에 남긴다. 로그 수집 실패가 종료를 막아서는 안 되고(돈이 탄다),
없는 로그와 빈 로그는 구별돼야 한다. 파싱·분류는 이 레인이 아니다.

곁: `build-images.yml`의 `paths:`에서 `trainbench/pods.py`와
`scripts/orchestrate.py`를 빼면 오케스트레이터 수정이 8~9분을 안 쓴다. 그 둘은
파드에서 실행되지 않는다.

### Lane G — 두 제외

**함정 셋, 전부 실측 확인:**
1. gemma-4 매니페스트만 먼저 지울 수 없다. `tests/test_pods.py:1737,1775,1841,1907`이
   `phase2-loss-gemma4_e2b`를 픽스처 실험 이름으로 하드코딩 — 8개를 지우면 5 failed
2. `axes.load_kwargs` 안에서 non-CUDA를 거부하는 것은 **qlora가 유일**하다
   (`axes.py:615-621`). `adamw_8bit`의 거부는 다른 함수이고 `preflight()`가 거기까지
   가지 않는다. Lane P가 오늘 `test_smoke_cpu.py` 3개를 mxfp8 → qlora로 옮겼으므로
   두 번째로 대표를 잃는다. **실제 축을 빌리는 대신 주입된 합성 거부로 바꾼다** —
   불변식은 어떤 축이 이번 주에 적용 불가인지와 무관해야 한다
3. `docs/open-verdicts.json`의 qlora 판정은 **이미 닫혀 있다.** `verdicts-closed`가
   개수 변화에 BLOCK하므로 지우면 게이트가 막히고, 지우는 게 옳지도 않다

**측정된 것은 지우지 않는다.** gemma-4의 파드 결과와 닫힌 판정은 역사다. "제외"로
표시한다. `bitsandbytes`는 `adamw_8bit`이 계속 쓰므로 남는다.

### Lane D — 스윕 매니페스트

| 축 | 값 | 파드 | 설정 |
|---|---|---|---|
| `attn` | sdpa, fa2, flex | 2 | 6 |
| `compile` | none, default, max_autotune | 2 | 6 |
| `dataloader` | torch, packed, pretokenized, packed+pretok | 2 | 8 |
| `optim` | adamw_fused, adamw_8bit, muon | 2 | 6 |
| `peft` | full, lora | 2 | 4 |
| `train.gradient_checkpointing` | none, full, selective | 2 | 6 |
| `freeze` | none, vision_tower | 2 | 4 |
| `kernel` | none, liger | qwen3_vl만 | 2 |
| | | **15** | **42** |

`kernel`이 1파드인 이유: `qwen3_5`는 `axes.FLA_ARCHS`라 `kernel=fla`가 강제다.

지켜야 할 것:
- 모든 측정 파드는 `baseline: canonical` + A100
- **qwen3_5 매니페스트는 `overrides`에 `kernel=fla`를 반드시 넣는다.** 없으면
  `axes.patch`가 거부하고 파드가 부팅해 과금하고 아무것도 측정하지 않는다
- **qwen3_5에 `kernel` 스윕을 만들지 않는다.** `axis: kernel`이 `AXIS_DECLARED`를
  추가해 기존 파드들의 `kernel=fla` 상수를 split으로 바꾼다
- `compile=max_autotune`은 `train.warmup_discard_steps=20`을 함께
- `peft` 스윕과 `freeze` 스윕은 합칠 수 없다 — 스키마가 교차를 거부한다

### Lane E — 잔해, 마지막

**착수 전에 결정해야 할 것:** `.plans/remaining-code/HAZARDS.md`가 삭제 목록에
있는데 `AGENTS.md`와 모든 레인 브리프가 "먼저 읽어라"로 지목하는 파일이다. 이
저장소가 실패에서 배운 것이 거기 모여 있다. 옮길 곳을 먼저 정한다.

`tests/contract/`의 "아무것도 단언하지 않는" 줄을 지우면 117 기준선이 내려간다.
계약 수를 줄이는 것은 Lane P의 5개처럼 **항목별 근거**가 필요하다.

## 규모와 비용

```
파드   14 (phase0 12 + phase2 2) + 15 (Lane D) = 29
런     18 + 57 = 75
```

타이밍 파드가 120스텝에 얼마나 걸리는지는 **측정 안 함** — 한 번도 완주한 적이 없다.
그 숫자가 나오는 것이 baseline 파드의 두 번째 목적이다.

### 재빌드가 필요한 기준

`build-images.yml`의 `paths:`가 1층이다: `docker/**`, `envs/**`,
`pyproject.toml`, `uv.lock`, `trainbench/**`, `scripts/**`. `configs/**`,
`docs/**`, `tests/**`는 빌드를 안 태운다.

2층은 `Dockerfile.framework`의 층 순서다:

| 바뀐 것 | 무효화 | 비용 |
|---|---|---|
| `envs/*/{pyproject.toml,uv.lock}`, 루트 락 | 10–11 | 의존성 전체 — 20분+ (오늘 실측) |
| `trainbench/**` | 62 | 8분 17초~8분 46초 (실측) |
| `scripts/**`, `entrypoint.sh` | 66–67 | COPY + chmod |
| `configs/**` | 없음 | 빌드 자체가 안 돈다 |

`COPY configs`가 없다. 그래서 축 조합 변경은 재빌드가 필요 없고, 오케스트레이터가
랩톱에서 합성해 해석된 JSON을 환경변수로 넘긴다.

빌드는 main push로 자동 트리거되고 파드는 `--tag <commit sha>`로 digest를 고정하므로
**마지막 머지의 빌드만 완료되면** 된다.

### 코드를 이미지에 굽지 않는 안 — 첫 숫자 이후

파드가 코드를 내려받으면 `trainbench/**`·`scripts/**` 변경의 8~9분이 사라진다.
`uv`가 부팅마다 로컬 `trainbench`를 다시 까므로 반영 기계는 이미 있다.

막는 것은 시크릿이다. `ALLOWED_ON_POD`는 정확히 `HF_TOKEN` 하나이고 체크는
Infisical 시크릿의 **이름만** 비교한다 — GitHub PAT의 스코프는 볼 수 없다.
`GITHUB_TOKEN`을 넣으면 안전성이 기계가 검사하는 속성에서 사람이 기억하는 속성으로
바뀐다. 반면 파드는 이미 `HF_TOKEN`으로 결과를 업로드하므로
(`publish_result.py:162,171`) **HF로 코드를 배달하면 노출이 늘지 않는다.**

함께 와야 하는 것: 코드가 다운로드로 도착하면 `image_digest`가 코드를 식별하지
않으므로 파드가 받은 것을 스스로 해시해 기록해야 한다. 지금 `git_commit`은 런처가
주장한 값(`git_source: "env"`)이고 `git_dirty: null`이다. 열린 판정
`images-carry-a-code-snapshot-nothing-checks-is-current`가 이 문제다.

## 검증

1. F·I·C·L 완료 → 통합자가 각 워크트리에서 네 게이트 재실행 → 머지 → main 재실행
2. Lane G → Lane D
3. **baseline 파드 1대. 이번에는 로그가 자동으로 남는다.** 그것이 셋을 답한다 —
   픽셀 상한이 실제로 무는지, 타이머와 metrics 블록이 도는지, 이 프로젝트 최초의
   속도 숫자
4. 최종 웨이브: `--max-concurrent 12` (18칸이 2웨이브 약 12분이었다)
5. `report.py`로 병합하고 **각 숫자를 아티팩트에서 직접 읽어 확인한다.**
   스위트가 초록인 것은 증거가 아니다

## 하지 않는 것

- 품질 축(Recall@k) — 결정으로 제외
- `mixed(bf16,fp32)`를 초록으로 만드는 시도 — 계약이 영구 불일치로 동결
- `fa3`/`fa4` 매니페스트 — A100에서 값이 없다
- `parallel.*`, `train.offload` 매니페스트 — `pods.py`가 `gpuCount: 1`로 고정해
  world ≥ 2를 만들 수 없다. GPU 등급이 아니라 파드 스펙 문제다
- `dataloader=dali` — 파이프라인 구현이 없어 영구 거부
- gemma-4, qlora — 결정으로 제외

## 규율

- **핀된 소스를 읽고 나서 단언한다.** 이 저장소의 프로브 실패 전부가 "보통 그렇게
  동작한다"에서 나왔다
- **로그를 실패 보고로 읽기 전에 그것을 출력하는 코드를 읽는다.** 오늘 통합자가
  성공 로그를 결함으로 읽고 캠페인 전체를 회귀시킬 뻔했다
- **추가한 검사마다 변이 증거.** 사보타주 → 실제 출력 인용 → 복원 → 바이트 동일 확인
- **자기가 내지 않은 숫자를 옮겨 적지 않는다.** 확인할 수 없으면 "확인 안 함"
- **파드를 띄우면 로그를 받는다.** 오늘 결함 셋 중 둘이 로그에 있었다
