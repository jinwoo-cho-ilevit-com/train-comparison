# Phase 0 지원 매트릭스

프레임워크 x 모델 조합의 실제 동작 여부. 셀마다 근거와 검증 버전을 남긴다.
**확인하지 못한 것은 "미확인"으로 두고 추측으로 채우지 않는다** (컨벤션 16).

**이 문서는 손으로 쓴 부분과 생성되는 부분이 마커 하나로 갈린다.** 파일 맨 아래
생성 마커(`scripts/report.py`의 `MARKER`) 아래는 pod 결과에서 만들어지고 매 실행마다
통째로 덮어쓰인다. 마커 위는 전부 손으로 쓴 것이고 병합이 보존한다.

편집할 때 지켜야 하는 것 셋. 셋 다 어기면 병합이 `exit 2`로 멈추고, **pod을 돌리고도
리포트가 나오지 않는다.**

- **마커 아래에는 아무것도 손으로 쓰지 않는다** — 다음 병합에서 사라진다
- **마커를 하나 더 만들지 않는다.** 산문에 그 문자열을 그대로 인용하는 것도 포함이다
  (`document_head`는 개수를 세지 위치를 보지 않는다)
- **마커 위에 자동 생성 표와 같은 제목을 만들지 않는다.** 경쟁하는 매트릭스 두 개를
  막는 장치이고, 실제로 그 상태로 막혀 있었다

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
맨 아래 생성 블록의 pod 실행 결과로만 판정된다.

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

## 적재 검증을 읽는 조건 (손으로 기록)

**매트릭스 표 자체는 이 문서 맨 아래 생성 블록에 있다.** `scripts/report.py`가 pod
결과에서 만들고 마커 아래를 통째로 다시 쓴다. 여기 있던 손으로 쓴 6x3 표는 지웠다 —
같은 질문에 답하는 표 두 개가 한 문서에 있으면 어느 쪽이 최신인지 읽는 사람이 구분할
수 없고, `report.py`의 `document_head`가 바로 그 상태를 거부해 병합을 멈춘다(실제로
멈춰 있었다). 아래 절들은 생성 블록이 담지 못하는 것 — 셀을 어떤 조건에서 읽어야
하는지, 그리고 pod 없이 로컬에서 확인한 것들 — 만 남긴다.

**생성 블록의 출처는 pod 아티팩트뿐이다.** 아래 "native probe 실측"의 macOS CPU 런은
pod이 아니라 로컬에서 돌았으므로 **그 표에 나타나지 않는다.** 생성 매트릭스의 native
셀이 "미시도"인데 이 문서가 통과를 기록하고 있는 상태는 모순이 아니라 출처가 다른
것이며, 그 로컬 결과는 바로 아래 절이 수치까지 보존한다.

**native OK의 한정 조건** — 셀 하나를 다른 조건으로 읽으면 안 되므로 명시한다.

| 조건 | 값 |
|---|---|
| 하드웨어 / dtype | macOS CPU, **fp32**. GPU 아님, bf16 아님 |
| 입력 | **텍스트 위주**. 멀티모달 forward 1건은 통과했으나 vision tower는 grad를 받지 않았다 |
| 범위 | **3모델 중 2모델.** gemma-4-E2B는 미실행 |
| 의미 | 적재 + 1 step backward가 된다는 것. 속도·메모리·커널 경로에 대해서는 아무것도 말하지 않는다 |

나머지 5개 프레임워크는 이미지가 필요하므로 pod에서 채워진다.

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

**같은 이미지가 모델마다 다른 비용을 갖는다.** 448x448(`PROBE_IMAGE_SIZE`)을 세
프로세서에 직접 넣어 placeholder 수를 셌다(2026-08-02, transformers 5.14.1):

| 모델 | 448x448의 soft token |
|---|---|
| Qwen3-VL-Embedding-2B | 196 |
| Qwen3.5-0.8B | 196 |
| gemma-4-E2B | 256 |

gemma-4의 280은 `max_soft_tokens` **상한**이지 이미지당 값이 아니다 — 종횡비에 따라
252~280으로 달라지며 정사각형에서는 256이 최대다(`docs/model-spec.md`). 이 문단은
`vision_soft_tokens_per_image: 280`을 이미지당 값으로 적고 있었다. 모델 간 속도 비교는
이 값을 보정한 뒤에만 의미가 있고, 보정에 쓸 수는 실측한 위 표다.

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

## 이미지 빌드 결과 (커밋 `821f8f4`, GitHub Actions / linux-amd64)

3단 sync + `libffi-dev` + CUDA 소스 빌드 4종이 들어간 뒤 처음 돈 빌드다. 앞선
(커밋 `9188fd8` 이전) 결과 표는 서술 대상 Dockerfile이 사라졌으므로 이 표로 교체했다 —
두 결과를 나란히 두지 않는다.

| 이미지 | 결과 | 소요 |
|---|---|---|
| base | success | 2분 50초 |
| unsloth | success | 29분 |
| tevatron | success | 26분 46초 |
| sentence-transformers | success | 26분 33초 |
| ms-swift | success | |
| **native** | **failure** | 28분 20초 |
| **axolotl** | **failure** | 3분 19초 |

`uv sync --frozen`이 실제 linux/CUDA 환경에서 설치까지 도달한다는 첫 증거다. 지금까지는
해석(`uv lock`)만 확인했었다. **5/7** — 실패한 둘은 서로 다른 원인이고 아래 두 절에서
각각 다룬다.

### native 실패 — 러너가 죽었고 로그는 이유를 말하지 않는다

로그의 마지막 줄들이다.

```
#12 1346.1       Built transformer-engine-torch==2.17.0
#12 1418.4       Built causal-conv1d==1.6.2.post1
#12 1418.8 Prepared 106 packages in 23m 38s
#12 1432.6    Building deepspeed==0.19.3
#12 1433.7    Building flash-attn==2.8.3.post1
#12 1446.3       Built deepspeed==0.19.3
##[error]The runner has received a shutdown signal.
```

**의존성 충돌이 아니고 시간 초과도 아니다.** 빌드는 정상 진행 중이었고, 같은 워크플로의
unsloth가 29분에 성공했다. 읽어낼 수 있는 것은 시점뿐이다.

- `transformer-engine-torch`(22분)와 `causal-conv1d`(23.6분)는 **동시에** 돌고 둘 다 통과
- `deepspeed`는 13.7초에 끝났다 — CUDA op을 굽지 않는 기본 JIT 설치이므로 부담이 아니다
- `flash-attn`이 시작하고 **86초 뒤** 러너가 죽었다

즉 부하를 만든 것은 flash-attn 하나다. 다만 **왜 죽었는지는 로그에 없다.** 메모리
고갈과 디스크 고갈이 워크플로 로그에서 같은 모습이고, 러너가 죽으면 `if: always()`
단계도 돌지 않아 아무 증거가 남지 않는다. 아래 조치는 두 가설을 모두 겨냥하고,
동시에 다음 실패가 원인을 남기도록 계측을 넣는다.

**"OOM이었다"는 아직 확정이 아니다.** 확정된 것은 러너가 flash-attn 컴파일 중에
사라졌다는 것뿐이다.

### 빌드 환경변수 — 이름을 소스에서 확인했다

각 이름은 이 lock이 고정한 바로 그 버전의 sdist를 받아 읽었다. 추정이 아니다.

| 변수 | 읽는 패키지 | 확인한 위치 | 기본값 |
|---|---|---|---|
| `FLASH_ATTN_CUDA_ARCHS` | flash-attn 2.8.3.post1 | `setup.py:70` `cuda_archs()` | `80;90;100;120` |
| `NVCC_THREADS` | flash-attn 2.8.3.post1 | `setup.py:124` `append_nvcc_threads()` | `4` |
| `MAX_JOBS` | flash-attn 2.8.3.post1 | `setup.py:513` `NinjaBuildExtension` | `min(코어/2, 여유메모리GB/9)` |
| `NVTE_CUDA_ARCHS` | transformer-engine-torch 2.17.0 | `build_tools/utils.py:255` | CUDA 13에서 `75;80;89;90;100;120` |
| `NVTE_BUILD_MAX_JOBS` | transformer-engine-torch 2.17.0 | `build_tools/utils.py:47` | 무제한 (`MAX_JOBS`가 fallback) |

**`TORCH_CUDA_ARCH_LIST`는 여기서 아무 효과가 없다.** torch의
`_get_cuda_arch_flags()`는 빌드가 자기 `-gencode`를 넘기면 즉시 `[]`를 돌려주는데
(`torch/utils/cpp_extension.py:2643`, `if 'arch' in flag: return []`),
flash-attn과 causal-conv1d 둘 다 `-gencode`를 직접 넘긴다. 이 변수를 넣었다면 30분을
태우고 아무 변화도 얻지 못했을 것이므로 **넣지 않았다.**

`flash-attn`의 `MAX_JOBS` 기본 계산에는 `import psutil`이 들어 있다. 값이 무엇이었는지
로그에 남지 않으므로 명시해서 추측을 없앤다. 그 계산 옆의 상류 주석이
"each JOB peak memory cost is ~8-9GB when threads = 4"라고 적고 있고, nvcc의
`--threads`는 gencode 사이를 병렬화하므로 **메모리 지렛대는 `MAX_JOBS`가 아니라
`NVCC_THREADS`와 arch 개수**다.

`causal-conv1d` 1.6.2.post1에는 대응하는 변수가 **없다.** `setup.py:179-199`가 CUDA 13
아래에서 9개 gencode(75/80/87/90/100/120/103/110/121)를 무조건 넣고 arch 변수를 읽지
않는다. 좁힐 수 없으므로 `MAX_JOBS`로만 묶인다.

### `TORCH_CUDA_ARCH_LIST` 대신 arch를 좁힌 결정과 그 대가

`80;90;100` = A100 / H200 / B200. `PLAN.md`가 이름을 붙인 GPU 셋 전부이고, 그 이상은
없다. Phase 0~1은 A100, Phase 2~3은 B200, "B200 확보 실패 시 H200 전면 전환"이 문서화된
분기이므로 sm_90을 빼면 그 분기가 재빌드를 요구하게 된다. 실제로 뺀 것은 flash-attn의
sm_120과 transformer-engine의 sm_75/89/120이며, 어느 것도 계획에 등장하지 않는다.

**좁히면 그 이미지는 목록 밖 GPU에서 못 돈다.** 다만 조용히 틀리지는 않는다:
flash-attn은 `code=sm_XX`만 내보내고 PTX를 넣지 않으므로 JIT으로 흘러갈 경로가 없고,
목록 밖 GPU는 `no kernel image is available for execution on the device`로 죽는다.
느린 fallback으로 떨어져 잘못된 숫자를 내는 부류의 위험이 아니다.

**시작 시점 차단이 붙었다 (2026-08-02, G 레인).** 이미지가 넣은
`TRAINBENCH_CUDA_ARCHS`를 `scripts/bench.py`의 프리플라이트가 읽고,
`docker/entrypoint.sh`가 첫 setting 전에 그것을 부른다. 목록 밖 GPU면 측정 없이
종료한다 — `docs/methodology.md` §7.

읽는 값은 `nvidia-smi`가 아니라 `torch.cuda.get_device_capability()`다. 파싱할 텍스트
출력이 없고, 변환 규칙이 두 벌 생기지 않기 때문이다: torch 자신이
`_get_cuda_arch_flags()`에서 capability를 `f"{major}{minor}"`로 만들어
`-gencode=arch=compute_XX,code=sm_XX`를 짓고(`torch/utils/cpp_extension.py`, torch
2.13.0), 위 표의 arch 목록이 바로 그 표기다 — transformer-engine 기본값에 `89`가
들어 있는 것이 그 증거다(Ada = capability 8.9).

`TRAINBENCH_CUDA_ARCHS`가 **없는 이미지는 거부한다.** 이 파일이 그 변수를 넣은
`ENV`는 프레임워크와 무관하게 `docker/Dockerfile.framework`에 있고, 그 검사를 부르는
`docker/entrypoint.sh`를 이미지에 넣는 것도 같은 파일이다. 즉 검사를 갖고 변수를 갖지
않는 이미지는 이 저장소가 만들 수 없는 상태이며, 그런 이미지가 나타났다면 커버 범위를
알 수 없는 이미지다. 비어 있으면 통과시키는 쪽이 이 저장소가 열 번 낸 실패다.

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

### axolotl 실패 원인 — 근접 원인 뒤에 하나가 더 있었다 (커밋 `821f8f4` 빌드)

`libffi-dev`는 옳았고 충분하지 않았다. 헤더가 생기자 `cffi==1.16.0`은 **컴파일에
성공하고 임포트에서 죽는다.**

```
zstandard 0.22.0 sdist:
  make_cffi.py → cffi.FFI() → import _cffi_backend
  ImportError: _cffi_backend.cpython-313-x86_64-linux-gnu.so:
               undefined symbol: _PyErr_WriteUnraisableMsg
```

위 7단계 표의 5번("cffi 1.16.0 휠도 cp312까지")에 이어지는 8번이다.

| 단계 | 확인한 사실 | 근거 |
|---|---|---|
| 8 | cffi 1.16.0의 C 소스가 `_PyErr_WriteUnraisableMsg()`를 호출한다 | sdist의 `src/c/_cffi_backend.c:6121` |
| 9 | 그 심볼이 CPython 3.13 헤더에 없다 (3.8~3.12의 비공개 API였다) | `cpython-3.13.13`의 `include/python3.13/` 전체 grep 무결과 |
| 10 | ELF 확장모듈은 미정의 심볼이 있어도 링크되므로, 실패는 컴파일이 아니라 **임포트**에서 난다 | 위 빌드 로그 |

따라서 **cffi 1.16.0은 3.13에서 동작할 수 없고**, zstandard 0.22.0 sdist가 그것을
`build-system.requires`에 정확히 고정하는 한 격리 빌드로는 길이 없다. 근접 원인이
아니라 막다른 골목이다.

**조치 — 런타임 핀은 그대로 두고 빌드 환경만 바꿨다.** `envs/axolotl`에

- `[tool.uv] no-build-isolation-package = ["zstandard"]`
- `[dependency-groups] build`에 `cffi>=1.17`

`zstandard==0.22.0`(axolotl이 요구한 값)은 손대지 않는다. 격리를 끄면 zstandard가
환경의 cffi(이 lock에서는 cryptography 경유 **2.1.0**)로 빌드되므로 sdist의 낡은 빌드
핀을 우회한다. CPython에서 zstandard의 cffi 백엔드는 런타임 선택사항이지만
`make_cffi.py`는 빌드 때 항상 돌고, 죽은 곳이 거기다.

이전 기록이 `override-dependencies`로 zstandard를 0.25.0까지 올리는 안을
"업스트림이 정확히 고정한 값을 거스른다"는 이유로 택하지 않았는데, 이 방법은 그
반대를 하지 않는다 — 상류의 런타임 핀을 존중하면서 3.13에서 성립하지 않는 **빌드**
핀만 비켜간다.

**로컬 실측 (2026-08-02, macOS arm64 / CPython 3.13.13 / cffi 2.1.0)**

실패 메커니즘이 플랫폼이 아니라 파이썬 버전에 걸린 것이라 로컬에서 재현·검증이 된다.

```
uv pip install --no-build-isolation zstandard==0.22.0
  → Built zstandard==0.22.0
  → zstandard/_cffi.cpython-313-darwin.so 가 생성됨   (make_cffi.py 통과)
  → import zstandard; backend='cext'; 압축/해제 왕복 성공
```

**생성된 `_cffi.cpython-313-*.so`가 증거다** — CI에서 죽은 바로 그 `make_cffi.py`
단계가 통과했다는 뜻이다.

**미검증**: linux/amd64에서 같은 결과가 나온다는 것. zstandard의 C 소스는 이식 가능하고
실패 원인은 파이썬 버전에 걸려 있으나, 확인은 빌드뿐이다.

**남아 있는 대안 — 쓰지 않았다**: axolotl은 0.18.0이 최신이라(PyPI 확인) 상류에서
핀이 풀린 버전으로 올라가는 길은 없다. 이 방법이 실패하면 남는 선택지는
`override-dependencies`이거나, "axolotl 0.18.0은 CPython 3.13에서 이미지가 만들어지지
않는다"를 Phase 0 결과로 기록하는 것이다.

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

## 이번 변경의 검증 범위 (2026-08-02, 이미지 빌드 수정)

위 "이번 변경의 검증 범위 (2026-08-01, Wave 2 F)"의 "**CI가 돌지 않았다**" 항목은
해소됐다 — 커밋 `821f8f4`에서 실제로 돌았고 결과가 위 표다. 나머지 항목은 그대로
유효하다. **이번 변경에 대해서는 이 절이 유일한 기준이다.**

### 실행으로 확인한 것

| 항목 | 명령/방법 | 결과 |
|---|---|---|
| 빌드 환경변수 이름 5종 | lock이 고정한 버전의 sdist를 받아 `setup.py`/`build_tools` 직접 읽기 | 위 "빌드 환경변수" 표. 파일·행 번호까지 확인 |
| `TORCH_CUDA_ARCH_LIST` 무효 | 설치된 `torch/utils/cpp_extension.py:2643` 읽기 | `-gencode`가 있으면 `[]` 반환. 넣지 않기로 결정 |
| zstandard 0.22.0 격리 없이 빌드 | py3.13.13 venv + cffi 2.1.0, `uv pip install --no-build-isolation` | 성공. `_cffi.cpython-313-darwin.so` 생성, 임포트·왕복 통과 |
| axolotl env 재해석 | `envs/axolotl`에서 `uv lock` | 241 패키지, 이전과 동수. 추가된 줄은 `cffi>=1.17` 뿐 |
| 워크플로 문법 | `actionlint v1.7.7` | 지적 0건 |
| 회귀 없음 | `uv run pytest` / `ruff` / `audit_plan.py` | 아래 "레인 게이트" |

### 실행으로 확인하지 **않은** 것 — 주장하지 않는다

- **어떤 이미지도 빌드하지 않았다.** 로컬은 macOS arm64이고 대상은 linux/amd64 CUDA다
- **native가 이번에는 통과한다는 것은 미확인이다.** 새 설정의 최대 부하가 죽은 설정보다
  낮다는 것은 두 인자(`NVCC_THREADS` 4→1, gencode 4→3)에서 따라 나오지만, 얼마나
  낮아지는지는 재지 않았다. 상류 주석의 "8-9GB @ threads=4"는 상류의 수치이지 이
  러너에서 측정한 값이 아니다
- **러너가 죽은 원인이 메모리라는 것은 미확정이다.** 디스크 회수 확대와 사후 계측을
  함께 넣은 이유가 그것이다
- **`MAX_JOBS=2`가 causal-conv1d를 얼마나 늦추는지 측정하지 않았다.** ninja 기본
  병렬도에서 내려오므로 느려지는 방향인 것은 확실하고, 23.6분이 얼마가 되는지는
  돌려봐야 안다. `timeout-minutes: 330`이 그 상한이다
- ~~**arch 불일치 차단은 만들지 못했다.**~~ 해소됨 (2026-08-02, G 레인):
  프리플라이트가 `TRAINBENCH_CUDA_ARCHS`를 읽고 목록 밖 GPU면 측정 전에 종료한다
  (위 "좁힌 결정과 그 대가"). **실제 GPU에서는 아직 안 돌았다** — 검사는 CPU
  호스트에서 capability를 주입해 양방향으로 고정돼 있고, 첫 파드가 실물 판정이다

### 레인 게이트 (2026-08-02)

| | 결과 |
|---|---|
| `uv run pytest` | 669 passed (변경 전후 동일) |
| `ruff check` / `ruff format --check` | 지적 0건 |
| `scripts/audit_plan.py` | 10/13 (`verdicts-closed` 미해소는 기존 상태) |

기준선이 659가 아니라 669인 것은 다른 레인이 테스트를 추가했기 때문이며, 변경 전에
먼저 재서 확인했다.

### env lock 6종의 stale과 `--frozen`이 하지 않던 검사 (2026-08-02 해소)

발견 당시 `uv lock --check`는 **axolotl을 제외한 5종을 stale**로 답했다. 원인은 루트
`pyproject.toml`의 `native` extra에 `pytorch-optimizer`가 들어간 뒤 env lock이
재생성되지 않은 것이다. 빌드가 깨지지 않은 이유는 Dockerfile이 `--frozen`을 썼기
때문인데, 이 플래그는 lock 신선도를 **검사하지 않는다**(검사하는 것은 `--locked`다).
주석은 그 검사를 한다고 적혀 있었다.

지금 상태:

- 5종을 `uv lock`으로 재생성했다. **핀은 하나도 움직이지 않았다** — 전체 diff가
  6줄 추가·0줄 삭제이고, 추가된 줄은 전부 lock이 기록하는 trainbench의 `requires-dist`
  선언(`pytorch-optimizer>=3.10`)이다. `envs/native`만 한 줄이 더 붙는데, 그 env의
  직접 의존성이라 `pytorch-optimizer 3.10.1`은 이미 lock에 있었다. 즉 이미지가 담는
  패키지 집합은 6종 전부 그대로다.
- Dockerfile의 세 sync 패스가 `--locked`가 됐다. 첫 패스가 싼 패스이므로 stale은
  CUDA 컴파일 전에 멈춘다.
- `scripts/audit_plan.py`의 `env-locks`가 매 게이트에서 두 가지를 함께 묻는다.

`--locked`가 이미지 안에서 실제로 어떻게 동작하는지는 **측정 안 함**. 호스트에서
확인한 것은 Dockerfile의 1·2 패스가 보는 것과 같은 트리(루트 `pyproject.toml` +
`uv.lock` + `README.md` + `envs/<framework>`, `trainbench/` 소스 없음)에서 6종 전부
`uv lock --check`가 통과한다는 것까지다 — 소스가 없어서 헛되이 실패하지는 않는다는
증거다. 남은 것은 이미지의 uv(base가 `ghcr.io/astral-sh/uv:latest`를 가져오므로
호스트 것과 다를 수 있다)가 이 lock을 같게 읽는가이고, 그건 빌드 한 번이 답한다:
`uv sync --locked --only-group build`가 몇 초 안에 통과하면 답이 나온 것이고,
`The lockfile at uv.lock needs to be updated`로 멈추면 uv 버전 차이다.

---

여기부터 아래는 `scripts/report.py`가 생성한다. 아직 pod 결과가 없으면 비어 있다.

<!-- generated: probe results -->
