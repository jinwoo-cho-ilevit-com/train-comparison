# 측정 방법론

`trainbench/config_schema.py`의 `_timing_runs_are_uncontaminated`와 `AGENTS.md`의
"측정 규율"이 이 문서를 근거로 참조한다. 규율을 코드로 강제하려면 그 규율이 왜
필요한지가 어딘가에 적혀 있어야 하고, 여기가 그 자리다.

**이 문서의 규칙**: 수치는 이 저장소에서 실행해 얻었거나 1차 출처가 붙은 것만
적는다. 둘 다 없으면 **미측정**이라고 쓴다. 추정치를 괄호에 넣어 적지 않는다
(컨벤션 16).

## 근거 현황

| 규율 | 근거 상태 | 코드 강제 |
|---|---|---|
| 타이밍 런에서 프로파일러 금지 | **미측정 · 출처 없음** (§1) | `purpose=timing` + `run.profiler=true` 거부 |
| 측정 중 deterministic off | **미측정** (§2) | `purpose=timing` + `train.deterministic=true` 거부 |
| 학습 데이터는 pod-local NVMe | 사양 근거 있음, 실측 없음 (§3) | 없음 |
| 같은 축은 같은 pod | 실측 대기 (§4) | 없음 |

측정 진입점(`scripts/bench.py`)이 아직 없다. §1·§2를 재려면 반복 가능한 학습 스텝이
필요하므로, 두 항목은 그 하네스가 생긴 뒤 Phase 1에서 닫힌다. **그때까지 두 숫자를
어디에도 쓰지 않는다.**

---

## 1. torch.profiler의 iteration time 부풀림 — 미측정

### 현재 상태

**측정하지 않았고, 출처도 확보하지 못했다.**

`PLAN.md` / `README.md` / `AGENTS.md` / `trainbench/config_schema.py`가 오랫동안
"20~44%"를 사실로 반복했다. 어디에도 출처가 없었다. 컨벤션 16 위반이므로 문서
3곳에서는 숫자를 제거했다.

`trainbench/config_schema.py:237`의 docstring에는 아직 남아 있다. 이 파일은
`docs/CONTRACTS.md` §1의 공유 파일이라 문서 레인이 고치지 않는다. **계약 변경으로
올려야 할 항목이다.**

### 출처 조사 기록 (2026-08-01)

재현 가능하도록 무엇을 했는지 남긴다.

| 시도 | 결과 |
|---|---|
| 웹 검색 "torch.profiler overhead iteration time inflate percent training benchmark" | 요약기가 "20%–44% in production environments"를 되돌려줬으나 **어느 문헌인지 지목하지 못했다.** 질의어를 그대로 반사한 것으로 보고 채택하지 않음 |
| 웹 검색 `"torch.profiler"` + `"20%"` / `"44%"` | 해당 수치를 담은 1차 문헌 0건 |
| ARGUS (arXiv 2606.20374) 원문 조회 | torch.profiler 오버헤드 정량 수치 없음 |
| EROICA (arXiv 2506.08528) 원문 조회 | torch.profiler를 도구로 언급만 함. 벤치마크 없음 |

조사 중 확인한 것 하나: 오버헤드는 **하드웨어와 프로파일러 옵션에 따라 크게 갈린다.**
`record_shapes` / `profile_memory` / `with_stack`을 켜면 더 커진다. 즉 단일 범위로
인용할 수 있는 상수가 아니며, 이것이 우리 스택에서 직접 재야 하는 이유다.

### 닫는 방법

`scripts/bench.py`가 생기면 다음을 실행한다.

1. 동일 config에서 `run=timing`(프로파일러 off)과 `run=profile`(on)을 각각 실행
2. warmup 폐기 후 동일 step 수, 동일 seed, 동일 데이터 순서
3. step time p50/p95를 비교해 배수를 기록
4. 모델 3종 x GPU 1종에서 재고, `run.profiler` 하위 옵션 조합도 함께 기록

**규율은 이 숫자와 무관하게 유지된다.** 프로파일러가 스텝을 느리게 만든다는 것은
프로파일러의 동작 정의이고, 얼마나 느리게 만드는지가 미확정일 뿐이다. 크기를 모르는
편향은 보정할 수 없으므로 분리가 유일한 대응이다.

---

## 2. deterministic 모드 on/off 비용 — 미측정

### 왜 끄는가

`torch.use_deterministic_algorithms(True)`는 비결정적 커널을 금지한다. cuDNN
benchmark(= 커널 autotuning)가 함께 꺼지는데, **그 autotuning이 이 벤치마크의 측정
대상 일부다.** deterministic on 상태로 잰 수치는 "autotuning이 없을 때의 속도"이지
사용자가 실제로 얻는 속도가 아니다.

컨벤션 07은 재현성 장치를 끌 때 근거를 요구한다. 그 근거가 이 절이다.

### 현재 상태

**비용을 측정하지 않았다.** 얼마나 느려지는지 모른다.

측정 진입점이 없어 로컬에서 낼 수 있는 유일한 수치는 CPU 값이고, CPU에는
cuDNN autotuning이 없으므로 이 축의 답이 되지 못한다. **CPU 수치를 대리값으로 적지
않는다.**

### 닫는 방법

1. `train.deterministic` 만 다른 두 런을 GPU에서 실행 (`purpose=probe`로 — timing은
   스키마가 deterministic을 거부한다)
2. 모델 3종 x `compile=none` / `compile=default`에서 재고, tokens/s 차이를 기록
3. 결과를 이 절에 표로 남긴다

### 적용 범위

| 경로 | deterministic |
|---|---|
| `purpose=timing` / `quality` | **off** (스키마가 on을 거부) |
| `purpose=probe` / `profile` | 임의. 테스트와 CPU 경로는 on |
| 유닛 테스트 | on (`tests/test_device_seed.py`가 고정) |

---

## 3. 학습 데이터는 pod-local NVMe에서만 읽는다

RunPod network volume은 네트워크 연결형 스토리지다. 여기에 학습 데이터를 두면
dataloader 축이 파이프라인이 아니라 볼륨 대역폭을 재게 되어 **축 하나가 통째로
무효화된다.**

이 프로젝트는 network volume을 쓰지 않기로 했으므로(`PLAN.md` "스토리지") 실측
비교는 하지 않았다. 볼륨 대역폭 수치를 인용하지 않는 이유도 같다 — 인용해봐야 우리
구성에서 잰 값이 아니다.

원칙만 남긴다: **측정 중에는 모델도 데이터도 pod-local NVMe에 있어야 한다.**

## 4. 같은 축은 같은 pod

pod이 다르면 물리 호스트가 다르다. host vCPU 수와 메모리 대역폭 차이가
데이터로딩이 CPU 바운드일 때 그대로 throughput 차이로 잡힌다.

- 한 축 안의 설정을 여러 pod에 나누지 않는다
- 모든 pod이 canonical baseline 1개를 실행한다
- pod 간 편차 **3%** 초과 시 그 pod 결과를 폐기하거나 재실행한다

**3%는 아직 근거가 없는 임계값이다.** 동일 pod에서 baseline을 5회 반복해 편차를
실측한 뒤 확정한다. 실측 편차가 3%를 넘으면 임계값이 아니라 측정 절차를 고쳐야
한다는 신호다.

---

## 재현 조건 — 게이트 통과와 재현 가능은 다르다

Wave 0 게이트는 `102 passed`로 통과했다. 그 결과는 **`configs/data/`가 로컬에
untracked로 남아 있던 체크아웃에서만 성립했다.** `.gitignore`의 앵커 없는 `data/`
패턴이 그 디렉터리를 삼켜 한 번도 커밋된 적이 없었고, 깨끗한 clone에서는 Hydra
합성 자체가 죽었다(23~24 failed / 39 errors). 같은 결함으로 `data-pinned` 체크는
검사 대상이 0건이라 **통과**했다.

따라서 측정 결과를 신뢰하기 전에 확인할 것:

- 결과를 낸 커밋에서 **깨끗한 clone**이 게이트를 통과하는가
- 통과한 체크에 **검사 대상이 실제로 있었는가**. 빈 입력에 켜지는 초록불은 부재를
  인증한다
- run 레코드의 `git_dirty`가 false인가. true면 그 수치는 커밋으로 되돌아갈 수 없다

## 수치를 쓸 때

- 측정 안 한 것은 "미측정"이라고 쓴다. 범위나 근사치를 대신 넣지 않는다
- 인용 수치에는 1차 출처 URL과 조회 시점을 붙인다. 2차 요약은 근거가 아니다
- 남의 하드웨어에서 잰 값을 우리 결과와 같은 표에 넣지 않는다. 참고 수치는 별도
  열에 두고 조건을 함께 적는다
