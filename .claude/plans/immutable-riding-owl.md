# 리뷰 발견을 반영하고 첫 속도 숫자를 낸다

## 현재 위치 (2026-08-03)

`main = bf0d0cc`. 게이트: pytest 1245, contract 117, audit 13/15 rc=0.

오늘 레인 여덟이 머지됐고, 그 결과물 전체를 모듈별 리뷰 일곱에 걸었다. 리뷰는
**측정을 시작하기 전에** 걸었고, 그 판단이 옳았다 — 확정된 발견 중 넷이 숫자를
조용히 오염시키는 종류였다.

**속도 숫자는 여전히 0개다.** 이 프로젝트는 A100에서 스텝 하나를 완주한 적이 없다.

## 확정된 결정

| | 결정 |
|---|---|
| 모델 | gemma-4 제외. `qwen3_vl_emb_2b`, `qwen3_5_0_8b` 둘 |
| `peft` | qlora 제외. `{full, lora}` |
| GPU | A100 통일 |
| `precision` | mxfp8/nvfp4 제거 완료. `Literal["bf16"]` |
| 픽셀 | `max_pixels=1310720`, `min_pixels=4096`, `text_token_ceiling=285`, `max_seq_len=2048` |
| 배치 | `train.batch_size=4`. `mini_batch`는 단위 수정 후 재확인 |
| 품질 축 | 안 한다. 파레토는 속도 × 메모리 2축, 모델 내부 |
| `optim` | `adamw_8bit`의 철자 불일치를 **고친다**. 영구 불가로 남기지 않는다 |
| `dataloader` | **축을 미룬다.** 멀티모달 packing 미구현을 결과에 기록하고, 첫 숫자 이후 구현 |
| Phase 0 | **재실행한다.** 어댑터 셋의 revision 전달과 픽셀 예산이 파드에서만 검증된다 |
| 프레임워크 | baseline 워크로드 한 점을 6개 전부에서 측정 |
| 수정 범위 | 확정 발견 **전부**를 첫 측정 파드 전에 |

### `dataloader`를 미루는 이유

`packing`은 이 코퍼스에서 캠페인 최대 효과가 될 수 있다 — 텍스트 행 p50 15토큰이
1,327토큰 이미지 행과 한 배치에 들어가면 그 행의 98.8%가 패딩이다. 그런데
`collate.py:371`(`Encode`)과 `:409`(`PackedPairs`)가 `with_images=False`라
**멀티모달 packing이 구현돼 있지 않다.**

텍스트 전용으로 재면 숫자가 나오지만 실제 워크로드로 옮겨가지 않는다. 옮겨가지
않는데 그럴듯한 숫자가 이 저장소가 최악으로 정의한 유형이다. 기록만 하고 미룬다.

재료의 절반은 있다 — `axes._split_rows`가 GradCache를 위해 만든, 행별 이미지 개수
벡터로 `pixel_values`를 자르는 프리미티브가 packing이 필요로 하는 것과 같다.
어려운 부분은 `cu_seqlens`가 두 겹이라는 것이다: LM의 문서 경계와, 비전 타워가
이미 쓰고 있는 이미지별 윈도우.

## 리뷰 발견 — 확정된 것

측정값을 오염시키는 것 넷을 먼저, 나머지를 그 뒤에.

| # | 발견 | 위치 |
|---|---|---|
| **M1** | 측정 구간 안에서 호스트 동기화 2회/배치. 편향이 `compile`·`kernel`·`attn`에 정확히 걸린다. `bench.py:287-293`이 이걸 하지 말라고 적어두었다 | `embedding.py:51,63` |
| **M2** | GradCache는 piece당 4회. `mnrl` 2회 대 `cached_mnrl` `4×ceil(b/mb)`회 — loss 축 비교가 직접 편향 | 같은 함수 |
| **M4** | `reset_peak()`은 peak을 0이 아니라 **current**로 되돌린다. "모델 생성 할당은 스텝 비용이 아니다"가 거짓. `gradient_checkpointing` 축 오독 | `metrics/__init__.py:152-159` |
| **A2** | `_capture_kernel`이 fla의 **바인딩**만 인증한다. causal-conv1d 없는 이미지에서 18/24 레이어가 torch 폴백인데 `fla`로 매치. 적용·검증 독립성이 이 축에서만 깨진다 | `applied.py:514-573` |
| M3 | `synchronize`가 cuda만 처리 — 다른 가속기에서는 벽시계가 정말 먼저 멈춘다 | `metrics/__init__.py:178` |
| M5 | OOM 레코드는 `max_memory_allocated`, 예외 메시지는 allocator 수준. 어느 쪽인지 안 적힌다 | `validity.py:171` |
| M6 | `loader` 파라미터를 모듈로 섀도잉 — 측정 루프 안 | `bench.py:370` |
| M7 | `_TRACKED_PACKAGES`에 `kernels`(attn 축 confound)와 `triton`(compile 축 실체)이 없다 | `record.py:28-50` |
| A4 | `flash-linear-attention`을 추적하는데 `fla.ops`를 싣는 배포판은 `fla-core` | `record.py:40` |
| M8 | `DataLoader`에 `worker_init_fn`/`generator`가 없어 `seed_worker`·`dataloader_generator`가 죽은 코드 | `axes.py:1415`, `seed.py:42,49` |
| M9 | deterministic 금지가 `timing`에만. `quality`도 step time을 싣는다. `measurement` 블록에 `deterministic` 필드 없음 | `config_schema.py:449` |
| M10 | refusal·failed 레코드가 매트릭스에서 "기동됐는데 결과 없음"으로 렌더 | `report.py:656-663` |
| M11 | 지표 정의 절이 없다. `tokens`·`samples`·`peak_memory_bytes` 전부 | `docs/methodology.md` |
| M12 | `step_seconds_aggregate`는 선언된 통계인데 rate는 전부 산술 평균 | `metrics/__init__.py:333` |
| M13 | `persistent_workers` 없이 에폭이 돌면 워커 8개 재생성이 측정 구간 안. `drop_last`도 없다 | `bench.py:219`, `axes.py:1415` |
| A1 | `precision.name`에 apply site가 없다. `axis-wired`는 셋의 크기만 비교해 PASS | `axes.py:227`, `audit_plan.py:348` |
| A3 | `compile.mode=regional`이 핀된 transformers에 없는 API를 요구. **테스트 스텁이 그 메서드를 손으로 정의해 은폐** | `axes.py:1043`, `tests/test_axes.py:347` |
| A5 | `_parallel` docstring이 현재 `applied.py`와 정반대 (FSDP2 캡처는 이미 읽는다) | `axes.py:1116` |
| A6 | flash_attn 없는 파드에서 fa2/3/4가 kernel 폴백 치환으로 거부됨 — 예상 목록에 필요 | 문서 |
| A7 | `honours_load_kwargs`가 native·sentence_transformers 둘뿐이라 `attn` 축이 6중 2 | `loader.py:432,473` |
| A8 | tevatron의 `cross_device_negatives` 면제가 `parallel.strategy`와 무관하게 항상 걸린다 | `loader.py:497` |
| A9 | `assert-called` 면제 파생이 `loss.name`만 본다 | `audit_plan.py:505` |
| C1 | `max_pixels`가 6중 2에만. **axolotl은 `cfg.processor_kwargs`로 가능**(확인함), unsloth는 로더가 kwargs를 골라 써서 불가, ST·ms-swift 미확인 | `probe/*.py` |
| C3 | `mini_batch <= batch_size`가 **행 대 쌍**으로 단위가 어긋남. 올바른 상한은 `2 × batch_size` | `config_schema.py:481` |
| C5 | `_lora_needs_rank`가 구조상 절대 발화 안 함 | `config_schema.py:733` |
| C6 | `baseline_tolerance` 거부의 전제가 거짓 — `report.py::declared_tolerance()`가 이미 읽는다. **첫 파드가 노이즈 플로어를 기록하는 워크플로가 실행 불가** | `config_schema.py:388` |
| C7 | revision prefix 매칭이 빈 문자열도 통과시켜 거짓 이유로 거부 | `config_schema.py:612` |
| C9 | `data.max_pixels`/`min_pixels`가 silent-fallback인데 `Axis()` 표시가 없고 `data`는 `NON_AXIS_GROUPS` | 스키마·감사 |
| R1 | `evidence-committed`가 아티팩트 0개에서 **True**를 반환 | `audit_plan.py:856` |
| R2 | `optim=adamw_8bit` 철자 불일치 | `applied.py` |

### 정정 둘 (내가 과장했던 것)

- `report.py:838-853`은 각주가 아니라 **제목 있는 절**이고 "같은 작업량을 잰 것이
  아니다"라고 명시한다. dataloader 발견의 심각도는 그만큼 낮다
- 15개 감사 검사 중 **13개가 빈-스코프 가드를 갖고 있다.** 이 저장소는 그 함정을
  체계적으로 막고 있고, 예외는 `evidence-committed` 하나다

## 레인 — 파일 소유가 겹치지 않는다

```
개발 → 리뷰 → 개선 을 blocker 가 0이 될 때까지 반복한다.
리뷰어는 개발한 에이전트가 아니고, 레인의 워크트리에서 게이트를 재실행한다.
blocker 가 남아 있으면 개선 → 재리뷰 를 다시 돈다.
통합자는 머지 전에 네 게이트를 스스로 다시 돌리고, 머지 후 감사까지 돌린 뒤 푸시한다.
레인은 커밋하지 않는다.
```

### Wave 0 — 진행 중

| 레인 | 소유 | 발견 |
|---|---|---|
| **S** | `embedding.py`, `metrics/__init__.py`, `bench.py` | M1 M2 M3 M6 |
| **G** | gemma-4 + qlora 제거 (광범위) | C4 |

### Wave 1 — G·S 머지 후, 일곱 병렬

| 레인 | 소유 | 발견 |
|---|---|---|
| **K** | `trainbench/axes.py`, `trainbench/applied.py` | A1 A2 A3 A5 A8 R2 M8 M13 |
| **Q** | `trainbench/config_schema.py` | C3 C5 C6 C7 C9 M9 |
| **P** | `trainbench/probe/*.py`, `trainbench/loader.py` | C1 A7 |
| **U** | `scripts/audit_plan.py`, `docs/audit-baseline.json` | R1 A9 A1(감사쪽) C9(감사쪽) |
| **R** | `scripts/report.py` | M10 C6(리포트쪽) |
| **M** | `trainbench/metrics/*`, `trainbench/record.py` | M4 M5 M7 M12 A4 |
| **D** | `PLAN.md`, `docs/*.md` | M11 C8 A6 A7(기록) C2(기록) |

**공유 파일 규칙:** `tests/test_smoke_cpu.py`와 `tests/contract/*`는 **Wave 1에서
한 레인만** 건드린다. 다른 레인은 boundary request 로 올리고 통합자가 순서를 정한다.
테스트 파일은 이름이 모듈과 대응하므로 나머지는 자연히 갈린다.

### Wave 2 — 캠페인

| 단계 | 내용 |
|---|---|
| 1 | Phase 0 재실행 12파드 (2웨이브, 약 15분) |
| 2 | 프레임워크 baseline 6파드 |
| 3 | 축 스윕 13파드 (dataloader 2 제외) + loss 2 |

### 미룬 것

- **멀티모달 packing 구현** — 첫 숫자 이후. 그때 `dataloader` 축을 되살린다
- **다중 GPU** — `pods.py`의 `gpuCount: 1`, 분산 런처 부재, rank 인지 하네스 부재.
  축 적용·인증 코드는 이미 있다. 첫 숫자 이후
- **코드를 이미지에 굽지 않기** — `HF_TOKEN` 경유가 노출을 안 늘린다. 첫 숫자 이후
- **Lane E 잔해 제거** — `.plans/remaining-code/HAZARDS.md`의 이관처를 먼저 정해야 한다

## 확정이 필요한 설계 하나

`_baselines.yaml:22`가 `framework=native`를 핀하는데, 같은 파일 `:12`의 주석은
"파드의 모델도 프레임워크도 언급하지 않는다"고 적는다. 모델에 대해서는 참이고
프레임워크에 대해서는 거짓이다.

프레임워크 비교 파드가 그대로 baseline 을 돌리면 호스트와 스택이 동시에 달라지고
(이미지마다 transformers 5.14.1 / 5.12.1 / 5.5.0), 3% 게이트가 스택 차이를 나쁜
호스트로 오독한다.

**제안:** baseline 항목을 하나 더 만들어 프레임워크 핀만 없앤 `per_framework` 를
두고, 비교 파드는 그것을 돌린다. 그 파드들에서 baseline 은 통제 장치가 아니라
비교 대상임을 결과에 명시한다. Wave 1 시작 전에 확정한다.

## 규율 — 오늘 실제로 당한 것들

- **새 파일은 커밋 전까지 `plan-files`에 보이지 않는다.** 레인 워크트리에서 untracked
  라 검사가 못 보고, 통합자가 커밋하는 순간 켜진다. 오늘 이걸로 main 이 약 20분간
  빨간 채 푸시돼 있었다. **머지 → 감사 → 푸시** 순서를 지킨다
- **워크트리는 로컬 main 이 아니라 `origin/main`에서 잘린다.** 머지 즉시 푸시한다
- **base drift.** 레인이 끝나면 최종 게이트 전에 `git merge main` 을 먼저 돌리고,
  숫자를 어느 기준선과 비교하는지 보고에 적는다
- **로그를 실패 보고로 읽기 전에 그것을 출력하는 코드를 읽는다.** 오늘 통합자가
  성공 로그를 결함으로 읽고 캠페인 전체를 회귀시킬 뻔했다
- **자기가 내지 않은 숫자를 옮겨 적지 않는다.** 오늘 통합자가 출처 없는 숫자를 레인에
  두 번 넘겼고(`18/533`, 한 모델만 잰 `270`) 둘 다 레인이 잡았다
- **핀된 소스는 `envs/*/.venv`에 없다.** 여섯 개 다 이 호스트에 미빌드다. 소스는
  `~/.cache/uv/`의 `archive-v0`/`wheels-v6`/`git-v0/checkouts` 에 있고, ms-swift 는
  캐시에도 없어 ctx7 또는 GitHub 이 필요하다. **읽은 버전이 lock 핀과 같은지 확인한다**
- **추가한 검사마다 변이 증거.** 사보타주 → 실제 출력 인용 → 복원 → 바이트 동일

## 검증

1. 각 레인: 랩톱 네 게이트 + 변이 증거 + 리뷰어 재실행, blocker 0 까지 반복
2. 통합자: 워크트리에서 네 게이트 재실행 → 머지 → 감사 → 푸시
3. **첫 측정 파드는 로그가 자동으로 남는 상태로 뜬다** (Lane L 머지 완료)
4. `report.py` 로 병합하고 **각 숫자를 아티팩트에서 직접 읽어 확인한다.**
   스위트가 초록인 것은 증거가 아니다

## 규모

```
Wave 1   레인 7 병렬
Wave 2   파드 12 (probe) + 6 (framework) + 15 (sweep+loss) = 33
```

타이밍 파드가 120스텝에 얼마나 걸리는지는 **측정 안 함** — 한 번도 완주한 적이 없다.
