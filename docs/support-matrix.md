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

### env별 해석 결과 — 5/5 성공

`uv lock` 기준. `envs/<fw>/uv.lock`에 커밋되어 있다.

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

| | Qwen3-VL-Embedding-2B | Qwen3.5-0.8B | gemma-4-E2B |
|---|---|---|---|
| native | **OK (7/7)** | **OK (7/7)** | 미확인 (CPU fp32 20GB, pod에서 확인) |
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
