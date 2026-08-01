# Phase 0 지원 매트릭스

프레임워크 x 모델 조합의 실제 동작 여부. 셀마다 근거와 검증 버전을 남긴다.
**확인하지 못한 것은 "미확인"으로 두고 추측으로 채우지 않는다** (컨벤션 16).

이 문서 대부분은 `scripts/report.py`가 pod 실행 결과에서 생성한다. 아래
"환경 해석" 절만 로컬 실행으로 먼저 채워진 부분이다.

## 환경 해석 (2026-07-31, 로컬 macOS / uv 0.11.16 / CPython 3.13.13)

### 단일 환경 공존 불가 — 확정

프레임워크 6종을 하나의 uv 프로젝트에 `[dependency-groups]`로 묶는 시도는 실패했다.
`[tool.uv] conflicts`를 선언해도 **lockfile은 여전히 하나**이고, 공유 lockfile은
공유 패키지의 버전을 하나로 강제한다.

| 시도 | 결과 |
|---|---|
| 하한 없이 6종 그룹 + `conflicts` 선언 | 해석은 성공하나 스택 전체가 후퇴. `transformers 5.3.0`, `ms-swift 1.4.0`, `unsloth 2026.3.11`, `axolotl 0.17.0`으로 잠김 |
| 하한 명시 후 재해석 | 해석 실패 |

하한을 건 뒤 uv가 보고한 충돌 (원문):

```
Because your project depends on datasets>=5.0 and unsloth>=2026.7.1
depends on one of:
    datasets>=3.4.1,<4.0.dev0
    datasets>4.1.0,<4.4.0
we can conclude that your project and unsloth>=2026.7.1 are incompatible.
```

- **unsloth 2026.7.1~2026.7.6은 `datasets<4.4.0`을 요구한다.** 대상 모델이 필요로
  하는 `datasets>=5.0`과 양립하지 않는다
- ms-swift는 하한 없이는 **1.4.0**까지 후퇴했다. 최신은 4.4.2이며, 1.4.0에는
  Qwen3-VL/Gemma4 지원도 임베딩 학습 기능도 없다. 이 상태로 벤치마크를 돌렸다면
  "ms-swift 측정"이라는 이름의 무의미한 숫자가 나왔을 것이다

**결론**: 프레임워크마다 독립 프로젝트 + 독립 lock이 필요하다. uv workspace도
lockfile을 공유하므로 대안이 아니다. `envs/<framework>/`에 각각의 `pyproject.toml`을
두고 `trainbench`를 path 의존성으로 참조한다.

### 코어가 강제하는 핀이 곧 이미지 빌드 실패였다 — 확정

독립 프로젝트로 분리한 뒤에도 5종 중 3종이 실패했는데, 원인이 전부 프레임워크가
아니라 **`trainbench` 코어의 핀**이었다. 코어는 모든 env에 path 의존성으로 들어가므로
여기서 고정한 것이 그대로 전파된다.

| 코어 핀 | 깨진 env | 프레임워크 측 제약 |
|---|---|---|
| `torch>=2.13` | unsloth | `unsloth>=2026.7`이 `torch<2.12` 요구 |
| `huggingface-hub>=1.26` | axolotl | `axolotl==0.18.0`이 `huggingface-hub==1.17.0` 정확히 고정 |
| `pillow>=12.0` | ms-swift | `ms-swift>=4.4` -> gradio -> `pillow<12` |
| `hydra-core>=1.3` | axolotl | hydra/omegaconf가 `antlr4==4.9.*`, axolotl이 `antlr4==4.13.2` |

앞의 셋은 코어 하한을 실제 필요 수준으로 낮춰 해소했다. **hydra는 해소가
불가능하다** — 양쪽 다 antlr4를 정확히 고정하기 때문이다.

해결: **pod은 Hydra를 쓰지 않는다.** 조합은 실험을 정의하는 로컬에서 일어나고,
pod은 이미 해석된 config JSON을 받아 검증만 한다. Hydra는 `compose` extra로 빼서
프레임워크 이미지에 들어가지 않는다. 부수 효과로 이미지에서 hydra/omegaconf/antlr가
빠지고, 실제로 실행된 config가 그대로 기록된다.

교훈: 벤치마크 하네스의 코어는 **최대한 아무것도 강제하지 않아야 한다.**

### env별 의존성 해석 결과 — 6/6 성공

**`uv lock`의 해석 성공이지 설치도 실행도 아니다.** 이 표가 말하는 것은 "버전 조합이
성립한다"까지다. 설치는 아래 "이미지 빌드 결과"에서 처음 확인됐고, 실제 적재·학습은
"모델 x 프레임워크 적재 검증"의 pod 실행으로만 판정된다.

`envs/<fw>/uv.lock`에 커밋되어 있다.

| env | 패키지 수 | torch | transformers | datasets | accelerate | peft | hf-hub | numpy | pillow |
|---|---|---|---|---|---|---|---|---|---|
| native | 112 | 2.13.0 | 5.14.1 | 5.0.1 | 1.14.0 | 0.20.0 | 1.26.0 | 2.5.1 | 12.3.0 |
| unsloth | 103 | **2.11.0** | **5.5.0** | 4.3.0 | 1.14.0 | 0.20.0 | 1.26.0 | 2.5.1 | 12.3.0 |
| axolotl | 237 | **2.12.1** | 5.14.1 | 4.8.4 | 1.13.0 | 0.19.1 | **1.17.0** | 2.4.6 | 11.3.0 |
| ms-swift | 152 | 2.13.0 | 5.12.1 | 4.8.4 | 1.14.0 | 0.19.1 | 1.26.0 | 2.5.1 | 11.3.0 |
| sentence-transformers | 65 | 2.13.0 | 5.14.1 | - | - | - | 1.26.0 | 2.5.1 | - |
| tevatron | 79 | 2.13.0 | 5.14.1 | 5.0.1 | - | - | 1.26.0 | 2.5.1 | - |

프레임워크 자체 버전: unsloth 2026.7.6 / axolotl 0.18.0 / ms-swift 4.4.2 /
sentence-transformers 5.6.1 / tevatron 0.0.1(git HEAD).

**측정 유효성 경고 — Phase 3 설계에 반영 필요**

torch가 env마다 **2.11.0 / 2.12.1 / 2.13.0**으로 갈리고 transformers도
**5.5.0 ~ 5.14.1** 범위로 흩어진다. 이 상태의 프레임워크 비교는 "프레임워크 차이"가
아니라 "프레임워크 + torch 버전 차이"를 재게 된다.

두 가지 해석이 모두 성립한다.

1. **실사용 관점**: 각 프레임워크를 설치하면 실제로 저 스택을 받는다. 그것이
   사용자가 겪는 성능이다
2. **통제 관점**: 프레임워크만 분리해 비교하려면 공통 스택(최저 공통분모인
   torch 2.11.0)으로 고정해야 한다

Phase 3에서 어느 쪽을 (또는 둘 다) 채택할지 결정해야 한다. 현재는 1번 상태이며,
모든 run이 해석된 버전을 함께 기록한다.

**추가 확인 필요**: unsloth env의 transformers 5.5.0이 gemma-4-E2B(config상
`transformers_version: 5.5.0.dev0`)와 Qwen3-VL-Embedding을 실제로 지원하는지.
probe로 판정한다. tevatron이 git HEAD에서 0.0.1로 잡히는데, 이것이 논문의
Tevatron 2.0인지도 확인 대상이다.

### native 하네스 기준 환경 — 확정

`uv lock` 성공, 112 패키지.

| 패키지 | 잠긴 버전 |
|---|---|
| torch | 2.13.0 |
| transformers | 5.14.1 |
| datasets | 5.0.1 |
| accelerate | 1.14.0 |
| peft | 0.20.0 |
| hydra-core | 1.3.4 |
| trackio | 0.34.0 |
| pydantic | 2.13.4 |

### 패키지 소스 주의

- PyPI `tevatron`은 0.1.0으로, 논문의 Tevatron 2.0과 일치하지 않는다.
  git 소스(`github.com/texttron/tevatron`)를 써야 한다
- `unsloth`의 requires-python은 `>=3.9,<3.15`. 프로젝트는 3.13에 고정했다

## 모델 x 프레임워크 적재 검증

native 열은 로컬 macOS CPU 실측. 나머지는 이미지가 필요하므로 pod에서 채운다.

**native OK의 한정 조건** — 셀 하나를 다른 조건으로 읽으면 안 되므로 명시한다.

| 조건 | 값 |
|---|---|
| 하드웨어 / dtype | macOS CPU, **fp32**. GPU 아님, bf16 아님 |
| 입력 | **텍스트 위주**. 멀티모달 forward 1건은 통과했으나 vision tower는 grad를 받지 않았다 |
| 범위 | **3모델 중 2모델.** gemma-4-E2B는 미실행 |
| 의미 | 적재 + 1 step backward가 된다는 것. 속도·메모리·커널 경로에 대해서는 아무것도 말하지 않는다 |

| | Qwen3-VL-Embedding-2B | Qwen3.5-0.8B | gemma-4-E2B |
|---|---|---|---|
| native | **OK (7/7, macOS CPU fp32)** | **OK (7/7, macOS CPU fp32)** | 미확인 (CPU fp32 20GB, pod에서 확인) |
| unsloth | 미확인 | 미확인 | 미확인 |
| ms-swift | 미확인 | 미확인 | 미확인 |
| sentence-transformers | 미확인 | 미확인 | 미확인 |
| tevatron | 미확인 | 미확인 | 미확인 |
| axolotl | 미확인 | 미확인 | 미확인 |

### native probe 실측 (2026-07-31, macOS CPU / transformers 5.14.1 / torch 2.13.0)

통과 항목: processor_load, model_load, text_tokenize, visual_tokens,
text_embed_forward, infonce_backward, multimodal_embed_forward.

| | Qwen3-VL-Embedding-2B | Qwen3.5-0.8B |
|---|---|---|
| 448x448 이미지의 visual token | **196** | **196** |
| image token id | 151655 | 248056 |
| 임베딩 차원 | 2048 | 1024 |
| InfoNCE 1 step loss | 4.2736 | 3.0991 |
| grad 받은 파라미터 / 전체 | 310 / 625 | 320 / 473 |

`AutoModel`(생성 헤드 없음)로 적재하고 last-token pooling + InfoNCE로 1 step
backward까지 확인했다. 텍스트 전용 배치라 vision tower는 grad를 받지 않는다.

**gemma-4-E2B는 config상 `vision_soft_tokens_per_image: 280`이므로 같은 이미지가
Qwen 계열의 196과 다른 비용을 갖는다.** pod 실측으로 확정한다. 모델 간 속도 비교는
이 값을 보정한 뒤에만 의미가 있다.

### Qwen3.5 GDN 커널 — 확정

Qwen3.5-0.8B 적재 시 transformers가 출력한다.

```
[transformers] The fast path is not available because one of the required library
is not installed. Falling back to torch implementation. To install follow
flash-linear-attention#installation and Dao-AILab/causal-conv1d
```

`fla`와 `causal-conv1d`가 없으면 **레이어의 75%를 차지하는 Gated DeltaNet 경로가
조용히 느린 torch 구현으로 떨어진다.** 예외도 경고 반환도 아닌 로그 한 줄이라 놓치기
쉽다. 이 모델의 측정은 두 패키지의 설치 여부를 반드시 기록해야 하며, 미설치 상태의
수치는 아키텍처가 아니라 fallback을 잰 것이다.

### 프로세서 사용법 — 확정

- 멀티모달 배치는 `processor(text=..., images=...)`만으로 만들어지지 않는다. 텍스트에
  이미지 placeholder가 없으면 image token 0개 대 image feature 392개로 forward가
  실패한다. `apply_chat_template`로 이미지 블록을 넣는 것이 필수다
- 이 VLM 프로세서들은 **`torchvision`을 임포트한다**(Qwen3VLVideoProcessor 등).
  없으면 `AutoProcessor.from_pretrained`가 ImportError로 죽는다

## 남은 세부 검증 항목

pod 실행으로 판정한다.

- gemma-4-E2B 적재, PLE freeze 가능 여부, LoRA target module 인식
- gemma-4-E2B의 실제 visual token 수 (config상 280 예상)
- Unsloth 일반 VLM 경로 + 커스텀 InfoNCE에서 패칭이 깨지지 않는가
- Unsloth `FastSentenceTransformer`가 VLM 체크포인트를 실제로 거부하는가
- Axolotl의 Qwen3-VL 지원
- Tevatron 패키지의 정체 (git HEAD가 0.0.1로 보고됨) 및 세 모델 지원
- sentence-transformers가 생성형 VLM 2종에서 module layout 없이 기본 pooling으로
  떨어지는지

## 데이터 준비 실측 (2026-07-31)

- 행 수 조회는 datasets-server `size` 엔드포인트로 한다. `get_dataset_config_info`는
  config마다 데이터에 접근해 55GB 저장소에서 응답이 오지 않았다(20개 config 무응답).
  엔드포인트는 같은 정보를 **0.43초**에 돌려준다
- `MrZilinXiao/MMEB_train_with_image` `original` 스플릿 총 **1,068,472행 / 20 config**
- 2048행 비례 배분 예: VisDial 227, MSCOCO_i2t 218, ImageNet_1K 192, VOC2007 16
- 스트리밍 샘플링 실경로 확인(VOC2007 3행 + WebQA 5행, 29.7초). 컬럼
  `qry / qry_image / pos_text / mmeb_config`, 이미지가 PIL 객체로 실려온다

**주의 — Task 3에 반영 필요**: 쿼리 텍스트에 MMEB 자체 placeholder `<|image_1|>`가
들어 있다. 모델의 이미지 토큰이 아니므로 그대로 넣으면 image token 0개 대 feature N개
불일치로 forward가 실패한다. 모델별 `apply_chat_template`으로 변환해야 한다.

### 고정 서브셋 생성 완료 (2026-08-01)

| | |
|---|---|
| 저장소 | `jinwoo-cho/mmeb-subset` (private) |
| revision | `b750b9c3263e9ef5dce225fd50aa25d7c58f1d5f` |
| 행 수 | 2048 / 2048 (요청 대비 정확) |
| 기여 config | 20 / 20 (0행 config 없음) |
| sample_seed | 1234 |

revision을 `configs/data/speed.yaml`에 고정했다. 이 값이 모든 run의 데이터 버전으로
기록된다. `data/quality.yaml`은 행 수가 달라 별도 서브셋이 필요하므로 아직 미고정이다.

## 이미지 빌드 결과 (2026-08-01, GitHub Actions / linux-amd64)

베이스 1개 + 프레임워크 6개 = 이미지 7개. **6/7 성공.**

| 이미지 | 결과 |
|---|---|
| base | success |
| native | success |
| unsloth | success |
| ms-swift | success |
| sentence-transformers | success |
| tevatron | success |
| **axolotl** | **failure** |

`uv sync --frozen`이 실제 linux/CUDA 환경에서 설치까지 도달한다는 첫 증거다. 지금까지는
해석(`uv lock`)만 확인했었다.

### axolotl 실패 원인 — 베이스 이미지의 시스템 의존성 누락

```
× Failed to download and build `zstandard==0.22.0`
├─▶ Failed to install requirements from `build-system.requires`
├─▶ Failed to build `cffi==1.16.0`
    src/c/_cffi_backend.c:15:10: fatal error: ffi.h: No such file or directory
    error: command '/usr/bin/cc' failed with exit code 1
```

`libffi-dev`가 `docker/Dockerfile.base`의 apt 목록에 없다. axolotl의 의존성 트리만
`zstandard==0.22.0`을 끌어오고, 해당 버전에 cp313 휠이 없어 `cffi`를 소스 빌드하려다
헤더를 못 찾았다.

**이것은 Phase 0 결과가 아니다.** axolotl이 대상 모델을 지원하지 않는다는 증거가
아니라 우리 베이스 이미지의 결함이며, `libffi-dev` 추가로 해소된다. 계획의
"빌드되지 않는 이미지는 Phase 0 결과" 규칙을 적용할 대상이 아니므로 매트릭스의
axolotl 행을 "미지원"으로 채우면 안 된다.

**조치**: Wave 2 F 레인(이미지)에서 `libffi-dev`를 베이스에 추가하고 재빌드한다.
지금 단독 재빌드하지 않는 이유는 F 레인이 어차피 `COPY trainbench` 순서 변경과
digest 태깅으로 베이스/프레임워크 이미지를 함께 손볼 예정이기 때문이다.

### axolotl 실패 원인 — 상류 확정 (2026-08-01, Wave 2 F)

위 기록의 `libffi-dev` 부재는 **근접 원인이 맞다.** 다만 "axolotl의 의존성 트리만
`zstandard==0.22.0`을 끌어온다"는 서술이 왜 그런지까지는 밝히지 않았다. 인과 사슬을
끝까지 확인했다. 각 줄은 PyPI JSON API와 sdist 실측이다.

| 단계 | 확인한 사실 | 근거 |
|---|---|---|
| 1 | `axolotl==0.18.0`이 `zstandard==0.22.0`을 **정확히 고정** | `requires_dist`에 `zstandard==0.22.0` |
| 2 | zstandard 0.22.0(2023-11-01) 휠은 cp38~cp312뿐, **cp313 없음** | 휠 태그 `['cp310','cp311','cp312','cp38','cp39']` |
| 3 | 프로젝트가 3.13 고정이므로 sdist 빌드로 떨어짐 | `envs/axolotl/uv.lock:3448`에 `sdist`만 있고 `wheels` 목록이 없다 |
| 4 | zstandard 0.22.0 sdist가 `cffi==1.16.0`을 **빌드 요구로 정확히 고정** | sdist의 `[build-system] requires = ["cffi==1.16.0", "setuptools==68.2.2", "wheel==0.41.2"]` |
| 5 | cffi 1.16.0 휠도 cp38~cp312뿐, **cp313 없음** | 휠 태그 `['cp310','cp311','cp312','cp38','cp39']` |
| 6 | 따라서 cffi가 소스 컴파일되고 `ffi.h`가 필요해짐 | 기록된 빌드 로그 |
| 7 | `libffi-dev`가 베이스 apt 목록에 없다 (`build-essential`이 끌어오지 않는다) | `docker/Dockerfile.base` |

**왜 axolotl만인가**: ms-swift도 zstandard를 쓰지만 버전을 고정하지 않아
**0.25.0**으로 잡힌다(`envs/ms-swift/uv.lock:2328`). 0.25.0에는 cp313 휠이 있으므로
컴파일 자체가 일어나지 않는다. 즉 이 실패는 "cp313 이전에 멈춘 정확 고정 2개"가
프로젝트의 3.13 하한과 만난 지점이고, 정확 고정을 하는 프레임워크는 axolotl뿐이다.

**조치**: `docker/Dockerfile.base`에 `libffi-dev`를 추가했다. `zstandard`를
`override-dependencies`로 0.25.0까지 올리는 방법도 있으나 **택하지 않았다** —
업스트림이 정확히 고정한 값을 거스르는 것이고, GPU 없이 검증할 수 없다.

**미검증**: 이 수정으로 axolotl 이미지가 빌드된다는 것은 **확인하지 않았다.** 아래
"이번 변경의 검증 범위"를 참조한다.

## 축별 패키지 설치 (2026-08-01, Wave 2 F 레인)

`scripts/audit_plan.py`의 `axis-packages`가 12건을 보고했다 — config group이 값으로
제공하지만 어느 env에도 패키지가 없어 이름표에 그친 축들이다.

### 배치 원칙

**ablation 축 패키지는 `envs/native`에 둔다.** 근거는 추정이 아니라 매니페스트다:
`configs/experiment/phase2-*.yaml` 전부가 `framework: native`다. 축 sweep이 도는
곳이 native이므로 축을 켜는 패키지도 거기 있어야 한다. 프레임워크 env는 자기
기본 경로에 필요한 것만 받는다 — 이 규칙 덕분에 6개 중 5개 이미지가 아래의
CUDA 소스 빌드를 피한다.

**예외 하나: `flash-linear-attention` + `causal-conv1d`는 6개 env 전부.** 이것은
ablation 축이기 이전에 Qwen3.5의 Gated DeltaNet 경로다. 두 패키지가 없으면
transformers가 예외도 경고도 아닌 **로그 한 줄만 남기고** 느린 torch 구현으로
떨어지며(위 "Qwen3.5 GDN 커널" 절), 그 모델은 레이어의 75%가 GDN이다. Phase 0
매니페스트 6종이 모두 Qwen3.5를 적재하므로, 한 이미지에만 빼면 "프레임워크 차이"라는
이름으로 융합 커널 대 fallback을 비교하게 된다.

### 어디에 무엇이 들어갔나

| 축 | 패키지 | 들어간 env | 이유 |
|---|---|---|---|
| `attn/fa2,fa3,fa4` | `flash-attn>=2.8` | native | 축 sweep이 native에서 돈다 |
| `kernel/fla` | `flash-linear-attention`, `causal-conv1d>=1.6` | **6개 전부** | GDN 무음 fallback 제거 (위 예외) |
| `kernel/liger` | `liger-kernel>=0.6` | native | axolotl에만 전이 의존으로 있었다 |
| `kernel/kernels_hub` | `kernels>=0.10` | native | 위와 같음 |
| `precision/mxfp8,nvfp4` | `transformer-engine[core-cu13,pytorch]>=2.17` | native | extras 필수, 아래 참조 |
| `optim/adamw_8bit`, `peft/qlora` | `bitsandbytes>=0.48` | native | 6개 중 2개에만 있었고 native는 아니었다 |
| `optim/muon` | `pytorch-optimizer>=3.10` | native | |
| `parallel/zero2,zero3` | `deepspeed>=0.19` | native | |
| `dataloader/dali,dali_packed` | `nvidia-dali-cuda130>=2.2` | native | 이름 문제, 아래 참조 |
| `loss/cached_mnrl` | `gradcache` (git) | native | 이름 문제, 아래 참조 |

`transformer-engine`을 extras 없이 넣으면 안 된다. PyPI의 `transformer-engine`
2.17.0은 **내용이 전부 extras 뒤에 있는 shim**이다(`requires_dist`가
`transformer_engine_cu12/cu13/torch/jax`를 전부 extra 조건부로만 선언). 이름 검사는
통과시키면서 커널은 하나도 주지 않으므로 `[core-cu13,pytorch]`를 명시했다. `cu13`은
베이스 이미지의 CUDA 13 및 `cu130` torch 인덱스와 맞춘 것이다.

### 패키지 이름이 존재하지 않는 축 2건 — 계약 변경 요청

`AXIS_PACKAGES`(`scripts/audit_plan.py`, 공유 파일이라 수정하지 않았다)가 지정한
이름 중 둘은 **설치 가능한 배포판이 아니다.**

| 축 | 계약이 요구하는 이름 | 실측 |
|---|---|---|
| `dataloader/dali` | `nvidia-dali` | PyPI 0.0.1.dev5, summary가 **"A fake package to warn the user they are not installing the correct package"** — NVIDIA 본인이 올린 자리표시자다. `pypi.nvidia.com`의 `nvidia-dali`도 0.7.0/cp35~cp37(2019)에서 멈춰 있다 |
| `loss/cached_mnrl` | `grad-cache` | PyPI에서 **404**. 업스트림 `luyug/GradCache`의 `setup.py`는 `name='GradCache'`이고, 이는 PEP 503 정규화로 `gradcache`가 되지 `grad-cache`가 되지 않는다 |

실제로 설치한 것은 유지되는 배포판 쪽이다.

- `nvidia-dali-cuda130` 2.2.0 (`https://pypi.nvidia.com`, 별도 인덱스 선언).
  CUDA 13용 `nvidia-nvjpeg` / `nvidia-nvimgcodec-cu13` / `nvidia-nvtiff-cu13`를
  함께 끌어온다 — 이 축이 재려는 하드웨어 JPEG 디코드가 바로 그것이다
- `gradcache` 0.1.0을 git 소스에서. `uv.lock`이 커밋
  `906f03835fbc183132a9db32612a9e8f180ca3b4`로 고정한다 (tevatron과 같은 방식)

**따라서 `axis-packages`는 12건 → 3건으로 줄었을 뿐 PASS가 되지 않는다.** 남은 3건
(`dataloader/dali` x2, `loss/cached_mnrl`)은 설치 부재가 아니라 계약의 이름이
틀린 것이므로, `docs/audit-baseline.json`의 `axis-packages` 줄을 **지우지 않았다.**
`AXIS_PACKAGES`를 `nvidia-dali-cuda130` / `gradcache`로 고치는 것은 계약 변경이다.

### 부딪힌 충돌 — 1건

레진 결과이므로 숨기지 않고 기록한다. 원문:

```
× No solution found when resolving dependencies for split (markers:
│ python_full_version == '3.13.*' and sys_platform == 'linux'):
╰─▶ Because only axolotl<=0.18.0 is available and axolotl==0.18.0 depends
    on flash-linear-attention{platform_machine != 'aarch64'}==0.4.1, we can
    conclude that axolotl>=0.18.0 depends on flash-linear-attention==0.4.1.
    And because your project depends on axolotl>=0.18 and
    flash-linear-attention>=0.5, we can conclude that your project's
    requirements are unsatisfiable.
```

axolotl 0.18.0이 `flash-linear-attention==0.4.1`을 정확히 고정한다. `>=0.5`를 얻는
유일한 방법은 axolotl 자체를 내리는 것이므로 **강제하지 않고 axolotl의 고정을
받아들였다**(하한 없이 선언).

**결과로서의 제약**: axolotl 이미지만 GDN 커널이 **0.4.1**이고 나머지 다섯은
**0.5.2**다. 이미 기록된 torch 2.11/2.12/2.13 분산과 같은 종류의 버전 교란이며,
Phase 3에서 axolotl x Qwen3.5 수치를 다른 프레임워크와 나란히 놓을 때 이 차이를
함께 표기해야 한다.

나머지 5개 env는 충돌 없이 해석됐다. 부수 변화로 `torchvision`이 인덱스 접미사가
붙은 형태(`0.26.0+cu130` 등)로 다시 잡혔는데, 상류 인덱스 변화이고 torch/transformers
버전은 움직이지 않았다.

### 소스 빌드가 필요한 패키지 — 이미지 빌드 구조를 바꿨다

추가한 패키지 중 넷은 **휠이 없다**(전부 sdist 전용). PyPI JSON API 실측이다.

| 패키지 | 빌드 격리 | 근거 |
|---|---|---|
| `causal-conv1d` | 가능 | `[build-system] requires = ["setuptools", "wheel", "torch"]` |
| `transformer-engine-torch` | 가능 | `requires = ["setuptools>=61.0", "pip", "torch>=2.1"]` |
| **`flash-attn`** | **불가** | sdist에 `pyproject.toml`이 **없다**. `setup.py`가 `import torch` |
| **`deepspeed`** | **불가** | 같음. sdist에 `pyproject.toml`이 없고 `setup.py`가 `import torch` |

뒤의 둘은 PEP 517 격리 빌드가 setuptools 기본 requires만 보는데 `setup.py`는 최상위에서
torch를 임포트하므로 **원리적으로 격리 빌드가 불가능하다.** 그래서

- `envs/native/pyproject.toml`에 `no-build-isolation-package = ["flash-attn", "deepspeed"]`
- 격리를 끄면 빌드 의존을 환경이 직접 제공해야 하므로 `[dependency-groups] build`
  (`torch`, `setuptools`, `packaging`, `wheel`, `ninja`)를 두고 `default-groups`에 넣었다.
  기본 그룹에 넣지 않으면 2차 sync가 소스 빌드 직전에 setuptools를 지워버린다
- `docker/Dockerfile.framework`가 **3단 sync**로 바뀌었다. env 6개 전부 같은 `build`
  그룹을 선언하므로 Dockerfile에 프레임워크별 분기가 없다

## 이번 변경의 검증 범위 (2026-08-01, Wave 2 F)

**무엇이 증명됐는지에 대해 이 절이 유일한 기준이다.**

### 실행으로 확인한 것

| 항목 | 명령 | 결과 |
|---|---|---|
| env 6종 해석 | 각 `envs/<fw>`에서 `uv lock` | 6/6 성공 (native 142 / unsloth 108 / ms-swift 158 / sentence-transformers 73 / tevatron 87 / axolotl 241 패키지) |
| 워크플로 문법 | `actionlint v1.7.7` | 지적 0건 |
| 3단 sync 순서 | 최소 재현 프로젝트를 /tmp에 만들어 실행 | 아래 참조 |
| 회귀 없음 | `uv run pytest`, `ruff`, `audit_plan.py` | 281 passed / 지적 0건 / 신규 실패 0 |

3단 sync는 CUDA 없이도 검증 가능한 부분만 따로 떼어 실제로 돌렸다. path 의존성을
editable로 참조하는 최소 프로젝트를 만들고 그 소스 디렉터리를 **지운 상태에서**
1·2단이 성공하는지, 소스를 되돌린 뒤 3단이 editable 설치를 하고 `build` 그룹이
살아남는지를 확인했다. 세 단계 모두 통과했다. 이것으로 확인된 것은 **sync 순서와
`--no-install-package`의 동작**이지 CUDA 컴파일이 아니다.

### 실행으로 확인하지 **않은** 것 — 주장하지 않는다

- **어떤 이미지도 빌드하지 않았다.** 로컬은 macOS arm64이고 대상은 linux/amd64 CUDA다
- **어떤 이미지도 push하지 않았다.** GPU pod도 기동하지 않았다
- **CI가 돌지 않았다.** 로컬 main이 원격에 push된 적이 없어 이 세션에서 워크플로가
  실행될 수 없다. `actionlint` 통과는 문법이 유효하다는 뜻이지 빌드가 성공한다는
  뜻이 아니다
- **`libffi-dev` 추가로 axolotl이 빌드된다는 것은 미확인이다.** 인과 사슬 7단계는
  전부 근거가 있지만, 마지막 확인은 빌드뿐이다
- **`flash-attn` / `deepspeed` / `causal-conv1d` / `transformer-engine-torch`의 소스
  빌드가 성공하는지 미확인이다.** 이 넷은 nvcc로 CUDA 커널을 컴파일하며 시간·메모리를
  많이 쓴다. GitHub Actions 무료 러너의 6시간 잡 제한에 걸릴 가능성이 실재하고,
  걸리는지 여부는 돌려봐야 안다
- **`USE_HF=1`이 gemma-4를 찾게 해주는지 미확인이다.** 확인한 것은 ms-swift 4.4.2의
  `swift/utils/env.py`가 `strtobool(os.environ.get('USE_HF', '0'))`으로 기본값이
  ModelScope라는 사실뿐이다
- **설치·임포트·실행 어느 것도 확인하지 않았다.** `uv lock`은 해석만 증명한다

### 이미지 추적성 — `latest` 단독 태그 해소

결과에서 이미지를 되짚을 수 없던 문제(`latest`만 붙었다)를 다음으로 바꿨다.

- 모든 이미지에 `:latest`와 **`:<commit sha>`**를 함께 push한다.
  `scripts/orchestrate.py --tag <sha>`가 그 태그를 digest로 해석해 run에 기록한다
  (해당 기능은 이미 있다 — `image_digest()`)
- 프레임워크 이미지는 베이스를 **digest로** 참조한다
  (`BASE_IMAGE=<image>@sha256:...`). 태그로 받으면 베이스가 움직였을 때 어떤 베이스
  위에서 구워졌는지 알 수 없다
- 각 잡이 push된 digest를 job summary에 남긴다

### GHA 캐시 10GB 상한

`type=gha,mode=max`를 **레지스트리 캐시**로 바꿨다
(`type=registry,ref=<image>:buildcache,mode=max`). GitHub Actions 캐시는 저장소당
10GB인데 CUDA devel 베이스 + 프레임워크 6개를 `mode=max`로 올리면 한참 초과한다.
초과하면 LRU로 서로를 밀어내고, **7개 이미지가 공유하는 베이스 레이어가 가장 먼저
쫓겨난다.** GHCR에는 그 상한이 없고 어차피 push하고 있으므로 캐시를 그쪽에 둔다.
