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

**측정 진입점(`scripts/bench.py`)은 이제 존재한다**(Wave 3, 커밋 35a9a62). §1·§2를
막고 있던 것이 그 부재였으므로 두 절차는 지금 실행 가능하고, **첫 GPU 파드의 작업**이다.
그때까지 두 숫자를 어디에도 쓰지 않는다는 규칙은 그대로다 — 하네스가 생겼다는 것은
잴 수 있게 됐다는 뜻이지 잰 것이 아니다.

두 절차 모두 CPU에서는 답이 나오지 않는다. 프로파일러 오버헤드는 CUDA 커널 추적
비용이고 deterministic 비용은 cuDNN autotuning의 부재인데, CPU에는 둘 다 없다.

---

## 1. torch.profiler의 iteration time 부풀림 — 미측정

### 현재 상태

**측정하지 않았고, 출처도 확보하지 못했다.**

`PLAN.md` / `README.md` / `AGENTS.md` / `trainbench/config_schema.py`가 오랫동안
"20~44%"를 사실로 반복했다. 어디에도 출처가 없었다. 컨벤션 16 위반이므로 문서
3곳에서는 숫자를 제거했다.

`trainbench/config_schema.py`의 `RunConfig` docstring에는 아직 남아 있다. 이 파일은
`docs/CONTRACTS.md` §1의 공유 파일이라 문서 레인이 고치지 않는다. **계약 변경으로
올렸다** (`docs/CONTRACTS.md` §5, 2026-08-02). 고칠 내용은 숫자를 지우고 "부풀림 폭은
미측정이며 `docs/methodology.md` §1이 추적한다"로 바꾸는 것 하나다 — 규율(=거부)은
숫자와 무관하므로 검증기 동작은 바뀌지 않는다.

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

하네스가 있으므로 아래는 실행 대기 상태다. 첫 GPU 파드에서 돈다.

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

막고 있던 것은 하네스의 부재였고 그것은 해소됐다. 남은 제약은 하드웨어다 — 로컬에서
낼 수 있는 유일한 수치는 CPU 값이고 CPU에는 cuDNN autotuning이 없으므로 이 축의 답이
되지 못한다. **CPU 수치를 대리값으로 적지 않는다.**

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

### 미해소 위험 — canonical baseline이 `kernel=none`을 만족하지 못할 수 있다

> **해소됨 (2026-08-02). 아래는 그 당시의 기록이고, 결론과 남은 작업은 §7에 있다.**
> 시나리오는 사실로 확인됐고 선택지 2(“config가 선언하게 한다”)로 닫았다.

**미검증. 이 레인에서 확인할 수단이 없다.** 축 검증 레인이 `kernel=none`이 6개 이미지
전부에서 불만족이라고 보고했다. 사실이라면 그 영향이 떨어지는 곳은 baseline이다.

`configs/experiment/_baselines.yaml`의 canonical은 `model=qwen3_5_0_8b` + `kernel=none`
이고, **모든 측정 pod이 이것을 돌린다.** 메커니즘은 이렇다:

1. F 레인이 `flash-linear-attention` + `causal-conv1d`를 **6개 env 전부**에 넣었다
   (`docs/support-matrix.md`). 근거는 옳다 — 없으면 Qwen3.5의 GDN 경로가 로그 한 줄만
   남기고 느린 torch 구현으로 떨어진다
2. `applied._capture_kernel`은 빌드된 모델의 **모듈 클래스가 어느 패키지에서
   왔는지**로 판정한다(`axes.KERNEL_MODULE_ROOTS`에 `fla` 포함)
3. 따라서 Qwen3.5 모델이 `fla` 패키지의 모듈을 실제로 담는다면, `kernel=none` 요청에
   `applied="fla"`가 잡혀 **영구 mismatch**가 된다

그 경우 결과는 축 하나의 손실이 아니다 — **모든 측정 pod의 baseline 런이 차단되어
pod 간 편차를 잴 수 없고, 3% 게이트가 돌지 않으며, 어느 pod의 수치도 다른 pod과 같은
표에 들어갈 수 없다.**

**Phase 0은 영향받지 않는다.** `applied.ENFORCED_PURPOSES = ("timing", "quality")`이고
Phase 0 매니페스트 18개는 전부 `run: probe`다. `assert_matches`가 probe에서 즉시
return하므로 적재 검증은 이 경로로 막히지 않는다.

**확인 방법**: transformers의 Qwen3.5 GDN이 `fla`의 *함수*를 자기 모듈 안에서
호출하는지, `fla`의 *모듈*을 인스턴스화하는지가 갈림이다. 앞이면 모듈 루트가
`transformers`라 capture는 `none`을 돌려주고 문제가 없다. 뒤면 위 시나리오가 성립한다.
`fla`는 triton에 의존하고 triton은 macOS 휠을 내지 않으므로(`support-matrix.md`)
**로컬에서는 판정할 수 없다.** fla가 설치된 이미지에서 Qwen3.5를 적재해
`_module_roots`를 찍어보는 것이 유일한 판정이고, 첫 GPU 파드의 probe에서 같이 본다.

**사실로 확인되면 선택지는 둘이고, 어느 쪽도 probe 완화가 아니다**
(`docs/CONTRACTS.md` §2: "probe가 mismatch를 낼 때 고칠 것은 probe가 아니다").

1. canonical baseline의 모델을 바꾼다 — GDN이 없는 모델이면 이 충돌이 없다. 다만
   baseline은 "host CPU와 메모리 대역폭에 민감한 가장 싼 런"이어야 하므로 대체 모델이
   그 조건을 만족하는지 따로 봐야 한다
2. `kernel` 축이 Qwen3.5에서 갖는 의미를 config가 선언하게 한다 — GDN이 fla를 쓰는
   것은 이 모델의 기본 동작이지 `kernel` 축의 값이 아니다. 이 방향이 §2의
   "이질적 적용은 표현되어야 할 상태"에 해당한다

## 5. Muon이 무엇을 최적화하는가 — 이 빌드의 한정 조건

`optim=muon`으로 나온 모든 수치는 아래 분할을 전제로 읽어야 한다. 측정 절차가 아니라
**결과 해석의 조건**이므로 코드 주석이 아니라 여기에 적는다.

### 구현 출처

`pytorch-optimizer`의 `Muon`을 쓴다. 직접 구현하지 않은 이유는 두 가지다 —
`envs/native/pyproject.toml`이 이 축을 위해 이미 그 배포판을 고정하고 있고
`scripts/audit_plan.py`의 `AXIS_PACKAGES`가 `optim/muon`을 거기에 매핑하므로 자체
구현은 두 기록을 거짓으로 만든다. 그리고 이 벤치마크가 발표할 수치는 실무자가 설치해서
얻는 수치여야지 우리가 쓴 Newton-Schulz 루프의 수치가 아니다.

py3-none-any 휠이고 의존성이 numpy + torch뿐이라 CPU에서 실제 스텝이 돈다.

**"가중치가 움직였다"는 근거가 아니다.** decoupled weight decay가 grad를 읽기 전에
모든 파라미터에 `1 - lr*wd`를 곱하므로, `p.grad`를 한 번도 읽지 않는 Muon도 그 단언을
통과한다(실제로 그런 변이가 전체 스위트 503개를 통과했다). `tests/test_axes.py`의
muon 절이 대신 고정하는 것은 두 가지다 — 같은 초기 가중치에서 grad를 준 스텝과 grad를
0으로 만든 스텝이 **다른 지점에 도달한다**는 것(grad를 읽었다는 증거), 그리고 가중치에
실제로 적용된 업데이트가 **grad의 직교화**라는 것(특이값 spread가 grad의 0.1에서
0.4 이상으로 평탄해지고, 방향이 `U @ Vt`와 코사인 0.9 이상). 두 번째가
`use_muon=False`인 Muon — 즉 Muon 이름을 단 AdamW — 을 걸러낸다.

### 파라미터 분할 — `p.ndim >= 2`

param group 2개다. `ndim >= 2`인 텐서가 Muon(Newton-Schulz) 그룹, 나머지(norm, bias)가
내부 AdamW 그룹. `applied._capture_optim`이 `param_groups: 2`로 기록하는 것이 이것이다.

**통상적인 Muon 레시피의 embedding / LM head 제외는 적용되어 있지 않다.**
`axes._optimizer`가 받는 것은 `model.parameters()` — 이름이 없는 이터러블이고,
embedding 행렬과 hidden weight 행렬을 이름 없이 구별할 방법이 없다. 이름을 넘기려면
`assemble`의 호출부를 바꿔야 하고 그건 `docs/CONTRACTS.md` §2의 호출 지점 계약에
걸리는 변경이다.

### 그래서 무엇을 읽으면 안 되는가

- 이 빌드의 Muon 수치는 **embedding 테이블까지 직교화한 Muon**의 수치다. 실무 레시피
  (Moonlight, Kimi K2 계열)와 다른 설정이며, 두 수치를 같은 이름으로 비교하면 안 된다.
- **PLAN.md의 gemma-4-E2B 가설은 이 빌드로 검증할 수 없다.** 그 가설은 파라미터의
  46% 이상인 PLE embedding 테이블이 Muon이 아니라 AdamW로 **넘어가기 때문에** Muon
  이득이 작아진다는 것인데, 여기서는 PLE 테이블이 Newton-Schulz를 통과한다 — 방향이
  반대다. 가설 검증에는 이름 기반 분할이 선행되어야 한다.
- `freeze.ple=true`와 교차하면 PLE 텐서는 param group에 그대로 들어가지만
  `requires_grad=False`라 grad가 없어 Muon이 스텝에서 건너뛰고 momentum buffer도
  만들지 않는다(실측: 얼린 텐서는 `optimizer.state`에 항목이 생기지 않는다).
  즉 **freeze on/off가 Newton-Schulz를 통과하는 텐서 수와 optimizer state 크기를
  동시에 바꾼다.** 이 빌드에서 embedding이 Muon 쪽에 있으므로 그 변동폭이 위의
  통상 레시피보다 크다 — freeze 축과 optim 축의 교차 셀은 두 축이 독립이라는 가정
  위에서 읽을 수 없다.

- `peft.mode=lora`와 교차하면 학습되는 텐서(`lora_A`/`lora_B`)가 **전부 2D**라 AdamW
  그룹에는 얼린 1D 텐서만 남는다. 즉 LoRA 팔에서는 optim 축이 사실상 순수 Muon이고,
  full finetuning 팔에서는 norm/bias가 AdamW로 빠진다 — 이 프로젝트의 표제 비교
  (full vs LoRA)에서 두 팔의 optim 구성이 같지 않다.

이 한정 조건은 `tests/test_axes.py::test_an_embedding_table_is_orthogonalised_here_rather_than_handed_to_adamw`와
`::test_under_lora_every_trained_tensor_is_on_the_muon_side`가 고정한다. 누군가 이름
기반 분할을 넣는 날 앞 테스트가 깨지고, 이 절도 같이 고쳐진다.

### 이 절이 덮지 못하는 것

- **throughput은 측정 안 함.** Muon의 주장은 수렴 속도와 optimizer state 메모리이고,
  둘 다 GPU와 실제 체크포인트가 있어야 잰다. CPU 테스트가 증명하는 것은 "스텝이 돌고
  가중치가 움직인다"까지다.
- `momentum` / `nesterov` / `ns_steps`는 `OptimConfig`에 knob이 없어 라이브러리
  기본값(0.95 / True / 5)으로 돌고 **run 레코드에 남지 않는다.**
- `pytorch-optimizer`가 `record.py`의 `_TRACKED_PACKAGES`에 없어 **Muon 구현의 버전이
  결과 JSON에 기록되지 않는다.** 프레임워크 이미지마다 스택이 다르다는 이 저장소의
  전제상 이건 남아 있는 confound다.
- `lr`은 config 값이 그대로 base가 되고 Muon이 텐서마다 재스케일한다
  (`get_adjusted_lr`). 두 그룹 모두 같은 base를 받는다 — AdamW 그룹에 별도 lr을 주면
  config에 없는 수가 측정 경로에 들어간다.
- **그 base는 AdamW의 lr이다.** `configs/optim/muon.yaml`의 `lr: 1e-5`,
  `weight_decay: 0.01`은 `configs/optim/adamw_fused.yaml`과 값이 같다. 반면
  `pytorch_optimizer.Muon.__init__`의 기본값은 `lr=0.02` / `adamw_lr=0.0003`으로
  두 경로의 lr을 애초에 분리한다. `get_adjusted_lr`이 `use_adjusted_lr=False`에서
  곱하는 `0.2*sqrt(max(dim))`를 감안해도 4096폭 행렬의 실효 lr은 약 1.3e-4로
  라이브러리 레시피보다 두 자릿수 낮다.
  **throughput 수치에는 영향이 없다**(스텝 시간은 lr과 무관하다). 그러나 PLAN.md가
  이 축에 기대하는 산출물은 throughput이 아니라 수렴 속도와 optimizer state
  메모리다. 즉 **이 config로 잰 수렴 곡선은 "AdamW의 lr로 돌린 Muon"의 곡선**이며,
  Muon의 수렴 주장에 대한 반증도 입증도 아니다. 수렴을 재려면 `optim` 그룹에
  Muon 전용 lr(과 필요하면 AdamW 쪽 별도 lr) knob을 먼저 추가해야 하고, 그것은
  `OptimConfig` 스키마 변경이다. 현재 상태로는 **측정 안 함**.

### 축이 켜지지 않는 조건 — 거부이지 대체가 아니다

세 경우 모두 `UnappliedAxis`거나 `applied=None`이며, timing 런은 숫자를 내기 전에
멈춘다. 어느 것도 AdamW 수치에 Muon 라벨을 붙이지 않는다.

- **`pytorch-optimizer`가 없는 환경.** 이 배포판은 root `native` extra와
  `envs/native`에만 있고 문서화된 셋업 명령(`uv sync --extra compose`)에는 없다.
  6개 프레임워크 이미지 중 5개도 마찬가지다. 그 환경에서 `optim=muon`은
  `UnappliedAxis`로 거부된다(예전에는 `assemble` 중간에 `ModuleNotFoundError`로
  죽었다).
- **직교화할 학습 가능한 행렬이 없는 모델.** 가드는 `p.ndim >= 2`가 아니라
  `p.ndim >= 2 and p.requires_grad`를 센다. 얼린 행렬은 grad가 없어 Muon이 스텝에서
  건너뛰므로, 2D가 전부 얼려 있으면 실제로 스텝하는 것은 1D 텐서뿐이고 그것은
  내부 AdamW다.
- **`use_muon`이 전부 False인 Muon.** 클래스 이름은 그대로 `Muon`이라 예전 레코드로는
  정직한 런과 구별할 수 없었다. `applied._capture_optim`이 이제
  `newton_schulz_tensors`(Newton-Schulz를 통과하는 **학습 가능** 텐서 수)와
  `use_muon` 분할을 detail에 기록하고, 0이면 `muon`이 아니라 undetermined로 읽는다.
  결과 JSON의 이 필드가 발행된 수치를 사후 귀속할 수 있게 하는 유일한 근거다.

## 6. gemma-4의 세 번째 타워는 어느 freeze 축도 건드리지 않는다

`freeze` 축은 `vision_tower` / `ple` 두 knob이고, gemma-4의 vision 마커는
`vision_tower.` + `embed_vision.`이다(`trainbench/axes.py`의 `VISION_PARAM_MARKERS`,
2026-08-01 각 체크포인트 헤더 실측). **`audio_tower.`는 그 목록에 의도적으로
없다** — 세 번째 타워이지 vision 타워가 아니라는 것이 그 결정의 근거이고, 축 정의로서는
옳다.

결과로 남는 것: 이 벤치마크는 **텍스트+이미지만 쓴다**(PLAN.md "모달리티: 텍스트 +
이미지, 오디오 제외"). 오디오 입력이 없으므로 `audio_tower` 파라미터는 forward에
기여하지 않는다. 그런데 어느 freeze 축도 그것을 얼리지 않으므로 **full FT 런에서
`requires_grad=True`로 남아 옵티마이저 상태를 차지한다.** AdamW 기준 파라미터당
m·v 두 벌이다.

- gemma-4-E2B의 peak VRAM 수치는 **쓰이지 않는 타워의 옵티마이저 상태를 포함한다.**
  세 모델의 메모리를 나란히 놓을 때 이 항이 gemma-4에만 있다
- PLAN.md의 gemma-4 병목 가설("옵티마이저 메모리")을 PLE 축으로만 검증하면 이 항이
  PLE 효과에 섞여 들어간다

**크기는 이 레인에서 재지 않았다.** `axes.py`의 실측 주석은 gemma-4 전체 2011 텐서 중
vision 659개까지만 적고 `audio_tower` 텐서 수는 담고 있지 않다. 같은 방식(체크포인트
safetensors 헤더)으로 재서 채운다. **재기 전까지 비율을 인용하지 않는다.**

조치 선택지는 둘이고 결정은 Phase 2 착수 전에 한다.

1. `freeze` 축에 오디오 타워를 포함하는 값을 추가한다 — 축이 하나 늘고, 세 모델에서
   같은 뜻을 갖지 않는 값이 생긴다(gemma-4 전용)
2. 오디오 타워를 축이 아니라 **모든 gemma-4 런의 고정 조건**으로 얼린다 — ablation
   표는 그대로고, 대신 "gemma-4는 오디오 타워를 얼린 상태로 측정했다"가 리포트의
   한정 조건이 된다

---

## 7. 파드의 계획이 돌 수 있는지는 파드 안에서만 답할 수 있다 — 프리플라이트

§4의 "미해소 위험"은 해소됐고, 답은 그 절이 적어둔 두 갈래 중 뒤쪽이었다.
`_baselines.yaml`의 canonical과 `phase2-loss-qwen3_5_0_8b.yaml`이 이제 `kernel=fla`를
명시한다 — Qwen3.5에서 fla는 `kernel` 축의 값이 아니라 이 아키텍처가 이 이미지들에서
갖는 기본 동작이므로, config가 그것을 선언한다.

남은 질문은 다르다: **어떤 매니페스트가 그 선언을 빠뜨렸을 때 누가 잡는가.**
`phase2-loss-qwen3_5_0_8b.yaml`이 정확히 그 상태로 커밋돼 있었고, 두 setting 모두
`axes.patch()`에서 거부되는 파드였다. 파드는 뜨고, 이미지를 받고, 체크포인트를 받고,
아무것도 측정하지 않는다.

### 이것은 감사 게이트가 될 수 없다 — 실측

`scripts/audit_plan.py`에 넣으려던 시도가 실측으로 무산됐다. 27개 계획 런을 감사
호스트에서 `axes.patch()`에 통과시키면 **답이 뒤집힌다**:

| 환경 | 거부되는 런 |
|---|---|
| 파드 이미지 (fla·causal-conv1d·CUDA 있음) | 0 / 27 |
| 감사 호스트 (셋 다 없음) | 5 / 27 — 전부 `kernel=fla`, 즉 **옳게 고친 쪽** |

감사 호스트에는 fla가 없어 `axes._fla_binding()`이 거짓이 되고, 그러면
`_environment_bound_kernel`이 빈 문자열을 돌려주므로 **실제로 죽는 `kernel=none`
setting이 통과**한다. 랩톱에서 도는 검사는 옳은 config를 빨갛게, 죽는 config를
초록으로 만든다. 이 질문의 답은 이미지의 내용물이고, 그것을 아는 것은 파드뿐이다.

### 그래서 파드가 스스로 검사한다

`docker/entrypoint.sh`가 첫 setting을 시작하기 전에 한 번,
`scripts/bench.py --preflight <plan>`을 부른다. 계획의 모든 setting을
`axes.patch` / `load_kwargs` / `step_context` — 모델 없이 답할 수 있는 세 호출
지점 — 에 통과시키고, 하나라도 거부되면 **아무것도 측정하지 않고** 모든 setting을
"측정 안 함 + 사유"로 발행한 뒤 끝난다.

`bench.py`는 원래도 `axes.patch`에서 거부한다. 프리플라이트가 더하는 것은 **시점**이다:
스윕은 두 번째 setting의 거부를 첫 번째가 끝난 뒤에야 알고, 계획 전체가 못 도는
파드는 GPU를 띄우고 이미지를 받고 체크포인트를 받은 뒤에야 안다. 프리플라이트는
파드 시간 몇 초다.

### 같은 자리에서 GPU도 본다

이미지 레인이 소스 빌드를 러너가 견디게 하려고 CUDA 아키텍처를 `80;90;100`
(A100 / H200 / B200 — `PLAN.md`가 이름 붙인 셋)으로 좁혔고, 그 목록을
`TRAINBENCH_CUDA_ARCHS`로 이미지에 넣었다. 목록 밖 GPU는 **조용히 느려지지 않는다** —
flash-attn이 `code=sm_XX`만 내보내고 PTX를 넣지 않아 JIT으로 흘러갈 경로가 없고,
`no kernel image is available for execution on the device`로 죽는다. 문제는 그 죽음이
**모델을 적재하고 첫 커널을 띄운 뒤에** 온다는 것이다. 프리플라이트가 먼저 본다.

**읽는 값은 `nvidia-smi`가 아니라 `torch.cuda.get_device_capability()`다.** 파싱할
텍스트 출력이 없고 변환 규칙이 두 벌 생기지 않는다: torch 자신이
`_get_cuda_arch_flags()`에서 capability를 `f"{major}{minor}"`로 만들어
`-gencode=arch=compute_XX,code=sm_XX`를 짓고(`torch/utils/cpp_extension.py`,
torch 2.13.0 설치본 확인), `Dockerfile.framework`의 arch 목록이 바로 그 표기다.
transformer-engine 기본값에 `89`가 있는 것이 그 표기의 증거다 — Ada는 capability 8.9이고
`8*10+9`로 읽든 `"8"+"9"`로 읽든 같은 자리를 가리키는 수는 이것뿐이다.

**변수가 없는 이미지는 거부한다.** 그 `ENV`는 프레임워크와 무관하게
`docker/Dockerfile.framework`에 있고, 이 검사를 부르는 `docker/entrypoint.sh`를
이미지에 넣는 것도 같은 파일이다. 검사를 갖고 변수를 갖지 않는 이미지는 이 저장소가
만들 수 없는 상태이고, 그런 것이 나타났다면 커버 범위를 알 수 없는 이미지다.
틀렸을 때의 대가는 변수를 넣고 한 번 다시 띄우는 것이고, 반대 방향의 대가는 모델을
적재한 뒤 죽는 파드다.

**이 검사가 실제 GPU에서 돌아본 적은 없다.** 이 저장소를 개발하는 호스트에는 NVIDIA
GPU도 `nvidia-smi`도 없다. 양방향(목록 안/밖, 변수 있음/없음, GPU 없음)은 capability를
주입해 고정했고, 실물 판정은 첫 파드다.

**이것이 증명하지 않는 것**: `assemble`과 `assert_matches`는 모델을 필요로 하므로
프리플라이트에 없다. 통과한 계획도 빌드된 모델이 요청과 다르면 setting별로 거부된다.
프리플라이트는 "이 이미지에서 이 축을 켤 수 있는가"만 답하고 "켜졌는가"는 답하지
않는다 — 뒤쪽은 `applied.capture`의 질문이다.

**이 검사는 이미지에 구워진다.** `entrypoint.sh`·`bench.py`·`trainbench/`를 고치면
이미지를 다시 빌드해야 반영된다.

## 8. gradient_checkpointing — CPU가 답하는 범위와 답하지 못하는 범위

이 축의 세 값(`none`/`full`/`selective`)은 세 개의 라벨이 아니라 세 개의 backward
pass다. 어느 쪽이 CPU에서 증명되고 어느 쪽이 GPU 파드를 기다리는지를 여기 적는다.

**철회.** 이전 수정자 보고의 "정책을 어느 방향으로 망가뜨려도 잡힌다"는 **거짓이며
철회한다.** 그 문장은 재검증 F5의 종결 근거로 쓰였다. 실제로는
`SELECTIVE_CHECKPOINT_SAVED_OPS`를 글자 그대로 둔 채 정책 함수만 목록 중 한 항목에
`PREFER_RECOMPUTE`를 돌려주게 바꾼 변이 4종(`bmm`/`_scaled_dot_product_flash_attention`/
`addmm`/`_scaled_mm`)이 전부 살아남았다 — 즉 어텐션과 bias 있는 선형층을 조용히
재계산해도 스위트는 초록이었다. `tests/test_axes.py::test_the_policy_honours_every_operator_on_its_own_save_list`가
그 구멍을 메운다(네 변이 각각 실행해 죽는 것을 확인, 2026-08-02).

### CPU에서 증명되는 것

| 질문 | 근거 |
|---|---|
| 세 값이 서로 다른 연산을 재계산하는가 | `TorchDispatchMode`로 backward의 실행 연산을 센다 (`test_the_selective_policy_saves_the_matmul_and_recomputes_the_rest`) |
| 정책이 자기 저장 목록 전체를 지키는가 | 목록의 11개 패킷 × 모든 overload에 대해 `MUST_SAVE` 단언 |
| 남의 정책이 `selective`로 읽히는가 | 같은 팩토리로 만든 `save_everything`/`save_nothing`이 undetermined로 거부됨 |
| reentrant 체크포인트가 `full`로 읽히는가 | 거부. frozen tower 입력에서 recompute 자체가 사라지는 것을 실행해 확인 |

### CPU가 구조적으로 볼 수 없는 것

- **저장 목록의 GPU 항목.** CPU에서 SDPA는 `aten._scaled_dot_product_flash_attention_for_cpu`로
  디스패치되고, 이 패킷은 목록에 아예 없다 — 즉 CPU에서 `selective`는 어텐션을 무조건
  재계산한다(torch 2.13.0, 이 호스트에서 실측 2026-08-02). 목록에 있는
  `_scaled_dot_product_flash_attention`·`_scaled_dot_product_efficient_attention`·
  `_flash_attention_forward`·`_efficient_attention_forward`·`_scaled_mm`은 위 단언이
  정책 함수를 직접 부르는 방식으로만 덮이고, **실제 모델의 backward에서 그 패킷이
  캐시에서 나오는지는 GPU 파드에서만 확인된다.**
- **활성화 메모리 — 측정 안 함.** 이 축이 재계산과 맞바꾸는 것이 그것이고, CPU에는
  잴 대상이 없다.
- **스텝 시간 — 측정 안 함.** `full`과 `selective`의 시간 차이가 이 축의 결과값이다.

### 닫는 방법

첫 GPU 파드에서 세 값 각각에 대해 `torch.cuda.max_memory_allocated`와 iteration
time을 기록한다. 그때까지 이 축에 대해 어떤 수치도 쓰지 않는다.

## 9. 커널이 모델의 얼마를 덮었는가 — `kernel_modules`, 측정 안 함

`applied._capture_kernel`은 빌드된 모델의 모듈 클래스가 어느 패키지에서 왔는지로
`kernel.name`을 판정한다. **그 판정에 임계값이 없다.** liger가 정의한 모듈이 하나뿐인
모델도 `applied="liger"`로 읽히고 `kernel` 축은 mismatch 목록에 들어가지 않는다 —
전체가 몇 개든 같은 답이다. 저장소 안의 근거는
`tests/test_axes.py::test_a_model_built_entirely_after_the_patch_is_a_kernel_run`이며,
`kernel_modules == {"liger": 1}`인 모델이 통과하는 것을 그대로 단언한다.

임계값이 없는 것은 지금으로서는 옳다. liger의 엔트리포인트는 텍스트 디코더를 바꾸고
비전 타워를 그대로 두는 것이 **문서화된 동작**이므로, 커버리지 하한을 지어내면 그
라이브러리의 정상 동작을 거부하게 된다. 정상 커버리지가 몇인지는 이 저장소가 아직
모르고, 모르는 수를 임계값으로 박는 것이 규칙 위반이다(컨벤션 16).

**대신 기록은 한다.** `_capture_kernel`의 detail이 `kernel_modules`
(라이브러리별 모듈 수)와 `modules_checked`(전체 모듈 수)를 담고,
`record.py`가 `applied.to_dict()`를 런 레코드에 그대로 쓴다. 즉 **첫 GPU 파드의
레코드에 실제 분포가 이미 들어온다** — 따로 계측을 붙일 것이 없다.

부분 적용을 실제로 막는 것은 커버리지가 아니라 `_superseded_modules`다. 자기 모듈이
이제 다른 클래스로 해석하는 이름으로 지어진 모듈이 하나라도 있으면 `partial(...)`이
되어 런이 선다. 이것은 **패치 전에 지어진 모듈**을 잡지, 라이브러리가 애초에 손대지
않는 부분을 잡지 않는다. 두 가지는 다른 상태이고, 뒤쪽이 여기서 미측정으로 남는 쪽이다.

### 오늘의 이미지 세트에서 이 구멍이 열리지 않는 이유 — 우연이 아니라 세 개의 거부

| 아키텍처 | `kernel=liger` | 근거 |
|---|---|---|
| gemma4 | 거부 | Liger-Kernel#1186 (`LIGER_UNSUPPORTED`) |
| qwen3_vl | 거부 | 기록된 엔트리포인트 없음 (`LIGER_ENTRYPOINTS`) |
| qwen3_5 | 거부 | 이미지가 fla를 바인딩한다 — `mixed(fla,liger)`가 될 런을 `patch()`가 막는다 |

셋 다 커버리지와 무관한 이유이므로, **liger의 커버리지 질문은 오늘 아무 런에서도
발생하지 않는다.** 발생하는 것은 `fla` 쪽이다: canonical baseline이 이제
`kernel=fla`를 요청하고, Qwen3.5의 모듈 중 fla가 정의하는 것이 몇 개인지는 아무도
모른다. 그 수가 작다면 지금의 판정은 "fla 패키지의 클래스가 모델 안에 하나라도
있다"에 가깝다.

### 닫는 방법 — 첫 GPU 파드

1. Qwen3.5 + `kernel=fla` 런 레코드에서 `applied.axes[kernel.name].detail`의
   `kernel_modules`와 `modules_checked`를 읽어 실제 분포를 적는다
2. 같은 값을 `kernel=none`이 가능한 이미지(fla 없는 빌드)에서 한 번 더 읽어, 이
   아키텍처에서 fla가 덮는 범위가 몇 퍼센트인지를 확정한다
3. 그 두 수가 나온 뒤에 임계값을 넣을지, 아키텍처별 기대 커버리지를 표로 둘지를
   정한다. **그 전에는 어떤 수도 여기에 쓰지 않는다.**

liger에 대해서는 위 세 거부 중 하나가 풀리는 날 — qwen3_vl 엔트리포인트가 기록되거나,
fla 없는 Qwen3.5 이미지가 생기거나, #1186이 닫히거나 — 같은 절차를 그 조합에서
반복한다.

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
