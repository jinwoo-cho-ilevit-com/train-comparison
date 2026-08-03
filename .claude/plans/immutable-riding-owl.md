# 측정을 시작한다 — A100 통일, precision 제거, 6개 프레임워크

## Context

Phase 0은 끝났다 (12 OK / 6 FAIL, 2026-08-03). 그러나 **이 프로젝트는 학습 속도
벤치마크인데 속도 숫자가 하나도 없다.** A100에서 돈 59개 런의 축 설정이 전부
바이트 단위로 동일한 기본값 baseline이고, 17개 축 중 하드웨어에서 변화시킨 축이
0개다. 17개 축 중 스윕 매니페스트가 있는 축은 `loss` 하나뿐이다.

장치는 거의 다 만들어졌고 실험 계획서가 없는 상태다. 이 계획은 그것을 채운다.

### 확정된 결정 다섯

| 결정 | 내용 |
|---|---|
| 품질 축 | **안 한다.** Recall@k 구현 없음. 파레토는 2축(속도 × 메모리) |
| `mxfp8` / `nvfp4` | **완전 제거.** transformer-engine 의존성까지 |
| `fa4` | **포기.** Blackwell 전용 (TMEM, 2-CTA MMA). `fa3`도 Hopper 전용이라 자동 |
| GPU | **A100 통일.** $1.49, 가용성 HIGH, 선행작업 0 |
| 프레임워크 | **6개 전부 측정 가능하게.** 지금 타이밍은 4개만 돈다 |

`fa3`/`fa4`는 **스키마에 남긴다** — `ATTN_IMPL` 딕트의 문자열 두 개일 뿐이고 추가
패키지가 없다. 매니페스트에서 선택하지 않고 리포트에 "미측정, 하드웨어 세대 전용"
으로 적는다. `mxfp8`/`nvfp4`와 달라 지우는 이득이 없다.

---

## Step 0 — 게이트 (통합자가 직접, 레인 없음)

**A100 통일과 gemma-4 full FT 가능성을 먼저 확정한다. 이 결과가 매니페스트 23개의
내용을 바꾸므로 다른 무엇보다 먼저다.**

1. `configs/experiment/phase2-loss-*.yaml` 3개의 `gpu_type_id`를 `NVIDIA B200` →
   `NVIDIA A100-SXM4-80GB`. **이것이 먼저 없으면 어떤 A100 타이밍 매니페스트도
   추가할 수 없다** — `check_one_baseline_one_gpu`가 baseline 코호트의 GPU 혼용을
   거부하고, `load_experiments()`가 `--experiment` 필터보다 먼저 디렉터리 전체를
   검증하므로 무관한 매니페스트를 지정해도 실패한다.
2. 검증용 타이밍 매니페스트 1개 신설 — `gemma4_e2b` × `peft=full` × `run=timing`.
3. 네 게이트 → 커밋 → 파드 1대 (약 6분, $0.15).

**이 파드가 셋을 동시에 답한다:**

- gemma-4 full FT가 A100 80GB에 들어가는가. **probe는 답한 적이 없다** —
  `trainbench/probe/steps.py:476`이 `loss.backward()` 뒤에 `zero_grad`로 끝나고
  `optimizer.step()`을 밟지 않으므로 Adam 상태(5.1B × 8 = 40.8GB)와 fp32 마스터
  (20.4GB)가 **한 번도 할당된 적이 없다.** probe가 증명한 것은 가중치+그래디언트
  약 20GB뿐이고 peak GPU 메모리를 기록하지도 않는다
- 측정 파이프라인이 파드에서 실제로 도는가 — 타이머, metrics 블록,
  baseline 3% 게이트. **한 번도 돌지 않았다**
- 이 프로젝트 **최초의 속도 숫자**

**OOM이면 멈추고 다시 정한다.** GPU를 바꾸거나 gemma-4를 `peft=lora` 기준으로
재는 것을 확정해야 하고, 그것이 Lane D의 내용을 바꾼다. 밀어붙이지 않는다.

---

## Wave A — 네 레인 병렬

각 레인은 **자기 git worktree**에서 돌고, **개발 → 리뷰 → 개선**을 순서대로 밟는다.

```
개발    executor 에이전트가 구현 + 테스트 + 변이 증거
리뷰    code-reviewer 에이전트가 그 워크트리에 cd 해서 게이트를 재실행하고
        적대적으로 검증        <- 개발한 에이전트가 아니다
개선    확정된 발견만 수정. 되돌린 것은 이유와 함께 남긴다
머지    통합자가 --no-ff, 게이트 재실행
```

리뷰어에게 **새 워크트리를 주지 않는다.** 레인의 작업이 없는 트리를 재고 그것을
판정으로 보고한 전례가 있다.

### Lane P — precision 제거 (가장 큼)

```
소유  configs/precision/{mxfp8,nvfp4}.yaml          삭제
      trainbench/config_schema.py                   Literal["bf16"], Axis() 유지
      trainbench/axes.py                            약 165줄 삭제
      trainbench/applied.py                         recipe 전용 표 셋
      trainbench/record.py                          _TRACKED_PACKAGES
      tests/test_axes.py test_applied.py            21개 삭제/편집
      tests/test_smoke_cpu.py tests/test_pods.py
      tests/contract/test_applied_axes.py           3개 삭제, CONTESTED 편집
      tests/fixtures/axis_state.sample.json
      scripts/audit_plan.py                         AXIS_PACKAGES 2줄
      docs/audit-baseline.json                      --update-baseline + note 재작성
      envs/native/pyproject.toml + uv.lock          transformer-engine 제거
      docker/Dockerfile.framework                   NVTE_* ENV
      README.md:38                                  깨진 예시 명령
```

**`mixed(bf16,fp32)`를 건드리지 않는다.** `applied.py:1165-1176`의 그 경로는 recipe
를 참조하지 않는 기본 경로이고, axolotl·unsloth 6칸이 거기 걸려 있다. 계약의
`UNNAMEABLE` 표에 있는 precision 항목도 **그것에 관한 것이므로 그대로 둔다.**

**`Axis()` 마커를 유지한다.** 떼면 `axis-fields`와 `axis-wired`가 새로 깨지고, 더
중요하게 **"bf16을 요청했는데 fp32로 적재된 것"을 잡는 검사가 사라진다** — 그것이
이 축의 원래 일이고 fp8과 무관하다.

**함정 둘, 둘 다 CI에서만 조용히 터진다:**

1. `tests/test_smoke_cpu.py` 5개가 `precision=mxfp8`을 "이 환경이 적용할 수 없는
   축"의 대표로 쓴다 — `step_context` 단계에서 거부되는 유일한 사례였다.
   **대체제가 오늘 생겼다**: CPU에서 `axes.step_context`가 axolotl의
   `required_step_context`(cuda autocast)를 받으면 같은 단계에서 `UnappliedAxis`로
   거부한다. 그것으로 바꾼다.
2. `axis-values` count가 3 → 2로 **줄어 감사가 BLOCK한다**(`shrank`). 설계상
   의도된 것이므로 `--update-baseline`을 함께 돌리고 note를 사실에 맞게 다시 쓴다.
   지금 note는 TE가 import되지 않는다는 측정 기록을 담고 있어 통째로 낡는다.

**빌드 시간 이득을 숫자로 주장하지 않는다.** `docs/support-matrix.md:804-806`이
로그가 주는 것은 완료 시각이지 소요 시간이 아니라고 못박고 있고, TE는
causal-conv1d와 동시에 돈다. **측정 안 함**으로 적는다.

### Lane F — framework-owned 스텝 경로

```
소유  scripts/bench.py                              framework 소유 스텝 분기
      trainbench/collate.py                         query/passage 키 생성
      trainbench/loader.py                          필요한 만큼
      tests/test_bench*.py test_collate.py test_loader.py
```

지금 `bench.py:725`가 `step.owner != harness.owner`면 무조건 거부한다. 거부 자체는
정당하다 — 하네스 루프를 그냥 돌리면 우리 forward와 우리 손실을 재놓고 프레임워크가
인증을 면제받은 축 아래에 발행한다. **문제는 `owner=framework`를 구동하는 경로가
없다는 것**이고, 값이 둘인 enum에서 하나가 항상 거부다.

`Step`은 이미 `owner=framework`와 tevatron의 `batch_keys=("query","passage")`를
선언하고 있고, `owned_axes`가 `loss.name`을 프레임워크 소유로 적어두었다. **계약이
틀린 게 아니라 절반이 미구현이다.**

probe 쪽에 이미 두 사례가 있다 — `probe/tevatron.py`의 `_backward`(오늘 만듦)와
`probe/sentence_transformers.py:107-126`. **그 모양을 타이밍 경로로 올린다.**
sentence_transformers는 `forward(input: dict)` 위치 인자를 요구하므로 두 프레임워크의
호출 규약이 서로 다르다 — 하나로 뭉개지 않는다.

`Step`/`AdapterOut`에 필드를 더해야 하면 **`boundaryRequests`로 올리고 고치지
않는다.** `PAYLOAD_KEYS`/`ADAPTER_OUT_FIELDS`가 계약으로 동결돼 있다.

### Lane X — axolotl × qwen3_5 회귀

```
소유  trainbench/probe/axolotl.py
      tests/test_probe.py                           axolotl 구역만
      .plans/notes/axolotl-probe.md
```

`no result file after the run (exit 1)`, `peak_uptime_seconds: 1`. 같은 칸이 같은 날
앞선 웨이브에서 (같은 `kernel=fla`로) 전체 probe를 마쳤으므로 **매니페스트 변경은
면책**되고 남은 변경은 probe의 autocast 배선 하나다. gemma4·qwen3_vl에서는 같은
수정이 성공하므로 **qwen3_5 고유**다. 파이썬 예외였다면 `report.run`이 실패 체크로
기록했을 텐데 그러지 않았다 — **파이썬 레벨 위에서 죽었다.**

가설: `torch.autocast` 안에서 qwen3_5의 fla/Gated DeltaNet 경로가 프로세스를 죽인다.
**확인 안 함** — 파드가 종료돼 로그가 없다.

**첫 일은 로그 확보다.** 그 조합 1대를 띄우고 파드가 살아 있는 동안 로그를 받는다.
원인을 보기 전에 고치지 않는다.

### Lane D — 스윕 매니페스트

```
소유  configs/experiment/*.yaml                     신설 23개
```

**Step 0이 끝난 뒤 착수한다** — gemma-4의 `peft` 기준이 그 결과로 정해진다.

| 축 | 값 | 모델 | 파드 |
|---|---|---|---|
| `attn` | sdpa, fa2, flex | 3 | 3 |
| `compile` | none, default, max_autotune | 3 | 3 |
| `dataloader` | torch, packed, pretokenized, packed+pretok | 3 | 3 |
| `optim` | adamw_fused, adamw_8bit, muon | 3 | 3 |
| `peft` | full, lora, qlora | 3 | 3 |
| `train.gradient_checkpointing` | none, full, selective | 3 | 3 |
| `freeze` | vision_tower (gemma4는 ple 포함) | 3 | 3 |
| `kernel` | none, liger | qwen3_vl, gemma4만 | 2 |

**23파드 · 설정 69개 + baseline 23회 = 92런.**

지켜야 할 것:
- 모든 측정 파드는 `baseline: canonical` + `gpu_type_id: NVIDIA A100-SXM4-80GB`
- **qwen3_5 매니페스트는 `overrides`에 `kernel=fla`를 반드시 넣는다.** 없으면
  `axes.patch`가 거부한다. 같은 값이므로 `held_constant`로 통과한다
- **qwen3_5에 `kernel` 스윕 매니페스트를 만들지 않는다.** `axis: kernel`이
  `AXIS_DECLARED`를 추가해 기존 7개 파드의 `kernel=fla` 상수를 split로 바꾼다
- `compile=max_autotune`은 `train.warmup_discard_steps=20`을 함께 (기본 10)
- `peft` 스윕과 `freeze` 스윕은 **합칠 수 없다** — 어댑터와 freeze 축의 교차가
  스키마에서 거부된다
- `axis:` 라벨은 오버라이드가 실제로 움직인 것과 일치해야 한다.
  `train.gradient_checkpointing`은 **점 표기 전체**를 쓴다

---

## Wave B — Lane P 머지 후

파일이 Lane P와 겹쳐 순서를 지킨다.

### Lane C — 문서를 사실에 맞춘다

```
PLAN.md      3축 → 2축 파레토, MFU 제거, HTA·memory snapshot 제거,
             파레토를 모델 내로, 이미지 토큰 예산 고정 요구 삭제,
             fa3/fa4/precision을 "미측정 — A100 캠페인"으로
docs/methodology.md    §9 kernel_modules 측정 안 함 → 18/533
docs/support-matrix.md native·axolotl 이미지 빌드 failure → 오늘 7잡 성공
                       precision 6칸 FAIL이 의도된 결과라는 문단
tests/contract/*       "Twenty-one of these tests are deferred" 산문
```

### Lane E — 잔해 제거

```
docs/evidence/env-report-cpu-*.json   trackio 필드 — 커밋된 아티팩트가
                                      현재 스키마를 통과 못 한다. 이것만 우선
envs/*/pyproject.toml + uv.lock       trackio 잔여, kernels>=0.10 핀
.plans/review/ + remaining-code/      5,315줄
tests/contract/*                      아무것도 단언 안 하는 것 약 900줄
scripts/audit_plan.py                 plan-files 체크
```

---

## 경계 — 통합자 전용

Wave A 동안 아래는 **어느 레인도 건드리지 않는다.** 필요하면 `boundaryRequests`로
올리고 통합자가 개정 하나를 발행한다.

```
docs/CONTRACTS.md            Lane P와 Lane F가 둘 다 필요로 한다
tests/contract/* 의 구조      Lane P의 precision 항목 삭제만 예외
configs/experiment/          Lane D 전용 (Step 0의 4파일은 통합자)
```

`docs/audit-baseline.json`은 **Lane P 전용**이다 — `--update-baseline`이 필요하고
그 실행은 전체 감사를 요구한다(`--only`/`--skip`로는 거부됨).

## 레인 규율

- **핀된 소스를 읽고 나서 단언한다.** 이 저장소의 프로브 실패 전부가 "보통 그렇게
  동작한다"에서 나왔다. `AGENTS.md`와 `.plans/remaining-code/HAZARDS.md`를 먼저 읽는다
- **추가한 검사마다 변이 증거.** 사보타주 전에 `co_filename`/`co_firstlineno`로
  실제 정의 위치를 확인하고, 출력을 그대로 인용하고, 복원한다
- **자기가 내지 않은 숫자를 옮겨 적지 않는다.** 확인할 수 없으면 "확인 안 함"
- **네 게이트.** `ruff check && ruff format --check`, `pytest`,
  `pytest tests/contract -q`, `audit_plan.py`. **계약 수는 122가 기준선**이고
  Lane P만 그것을 줄인다(3개 삭제 → 119). 다른 레인이 줄이면 계약을 약화한 것이다
- **커밋하지 않는다.** 워크트리에 남기고 보고한다. 머지는 통합자가 한다
- 이 호스트에 CUDA·TE·6개 프레임워크가 없다. 파드가 답할 것을
  `.plans/notes/<lane>.md`에 축별로 한 문장씩 적는다

## 검증

랩톱에서 죽일 수 있는 것을 파드로 보내지 않는다. 하드웨어 왕복 1회가 약 16분이다
(코드만 바뀐 재빌드 8~9분 실측 + 파드 웨이브 약 7분).

1. Step 0 파드가 gemma-4 full FT와 측정 파이프라인을 동시에 답한다
2. Wave A 각 레인: 랩톱 네 게이트 + 변이 증거 + 리뷰어 재실행
3. Lane P·F는 재빌드가 필요하다 → **한 번의 push에 묶는다**
4. Lane D는 재빌드 불필요 — `configs/`는 이미지에 COPY되지 않는다
5. 최종 파드 웨이브: `--experiment 'phase3-*' --max-concurrent 12`.
   probe는 타이밍을 재지 않으므로 12 동시가 결과를 오염시키지 않고, 실측으로
   18칸이 2웨이브 약 12분이었다
6. `report.py`로 병합하고 **각 숫자를 아티팩트에서 직접 읽어 확인한다.**
   스위트가 초록인 것은 증거가 아니다

## 하지 않는 것

- 품질 축(Recall@k) 구현 — 결정으로 제외
- `mixed(bf16,fp32)`를 초록으로 만드는 시도 — 계약이 영구 불일치로 동결
- `fa3`/`fa4`/`mxfp8`/`nvfp4` 매니페스트 — A100에서 값이 없다
- `parallel.*`, `train.offload` 매니페스트 — `pods.py`가 `gpuCount: 1`로 고정해
  world ≥ 2를 만들 수 없다. GPU 등급 문제가 아니라 파드 스펙 문제다
- `dataloader=dali` — 파이프라인 구현이 없어 영구 거부
- 리뷰 major 23 + minor 54 — 범위 밖
