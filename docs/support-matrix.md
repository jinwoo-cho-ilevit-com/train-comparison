# Phase 0 지원 매트릭스

프레임워크 x 모델 조합의 실제 동작 여부. 셀마다 근거와 검증 버전을 남긴다.
**확인하지 못한 것은 "미확인"으로 두고 추측으로 채우지 않는다** (컨벤션 16).

**gemma-4-E2B는 2026-08-03 캠페인에서 제외됐다** — full FT가 A100 80GB에
`train.batch_size=16`(83.8GB)과 4 어느 쪽에서도 들어가지 않는다는 실측이 근거다
(`PLAN.md` "gemma-4-E2B 제외"). 이 문서 안의 gemma-4 행과 실측은 **전부 제외
이전에 실제 pod에서 측정된 것**이므로 지우지 않는다 — 지우면 측정을 없었던
일로 만드는 것과 같다. 아래에서 gemma-4 관련 셀·표를 만나면 **캠페인 제외,
측정 자체는 유효**로 읽는다.

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
| pydantic | 2.13.4 |

`trackio` 는 이 표에서 빠졌다 — 결정 3 으로 스키마·config 에서 제거했다. 측정 중
네트워크 I/O 가 교란이고, 구현하려면 env lock 6종 전부에 넣고 이미지를 다시 빌드해야
했다. 루트 `pyproject.toml` 의 `tracking` extra 제거는 lock 재해석과 함께 움직이므로
통합자 몫으로 남아 있다(`.plans/notes/integrate.md`).

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
  실패한다. placeholder를 넣는 것이 필수다 — 다만 **`apply_chat_template`이 그 방법인
  것은 chat template을 가진 체크포인트에 한한다.** gemma-4-E2B에는 없고, 그쪽은 평문
  `<|image|>`가 같은 일을 한다(docs/model-spec.md 결정 5)
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
불일치로 forward가 실패한다. 모델별 프롬프트 형식으로 변환해야 한다
(`trainbench/prompt.py`; gemma-4는 chat template이 없어 평문 placeholder를 쓴다).

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

**현재 상태 (2026-08-03 정정): 두 실패 모두 이후 해소됐다.** 위 표는 이 실패가
처음 관측된 시점의 기록이고, 이 문서 자신이 이후 절에서 둘 다 근본 원인을 고치는
과정을 남긴다 — native는 flash-attn을 소스 빌드에서 미리 빌드한 휠로 바꿔
컴파일러가 죽던 지점 자체를 없앴고("flash-attn을 소스 빌드에서 직접 만든 휠로
바꿨다" 절), axolotl은 `no-build-isolation-package` + `cffi>=1.17` build 그룹으로
zstandard의 낡은 빌드 핀을 우회했다("axolotl 실패 원인 — 근접 원인 뒤에 하나가 더
있었다" 절). 그 이후 native와 axolotl 이미지는 실제로 pod에 올라가 Phase 0 probe를
실행했다 — 아래 "2차 Phase 0 캠페인" 표의 native 행(13/13, 12/12 OK)과 axolotl
행(적재·축 검증·패딩·토크나이즈까지 통과, `infonce_backward`에서 막힘)이 그
증거다. 이 시점 이후 이미지가 만들어지지 않고서는 그 probe들이 애초에 돌 수
없었다.

**확인 안 함**: native/axolotl이 정확히 어느 커밋의 CI 런에서 처음 빌드에
성공했는지, 그리고 그 빌드가 몇 분 걸렸는지. 이 문서에 그 시점의 빌드 로그가
남아 있지 않다 — 새 숫자를 지어내지 않는다.

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
| `optim/adamw_8bit`, `peft/qlora` | `bitsandbytes>=0.48` | native | 6개 중 2개에만 있었고 native는 아니었다 |
| `optim/muon` | `pytorch-optimizer>=3.10` | native | |
| `parallel/zero2,zero3` | `deepspeed>=0.19` | native | |
| `dataloader/dali,dali_packed` | `nvidia-dali-cuda130>=2.2` | native | 이름 문제, 아래 참조 |
| `loss/cached_mnrl` | `gradcache` (git) | native | 이름 문제, 아래 참조 |

`precision/mxfp8,nvfp4`는 `transformer-engine[core-cu13,pytorch]`를 native에
넣어 열었으나, 이후 캠페인이 A100(CC 8.0)으로 통일되면서 제거됐다 — 두 recipe 모두
CC 10.x 전용이라 이 스터디의 파드에서는 원리적으로 열릴 수 없었다.

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

## flash-attn을 소스 빌드에서 직접 만든 휠로 바꿨다 (2026-08-02)

위 "소스 빌드가 필요한 패키지"의 넷 중 `flash-attn` 하나가 네이티브 이미지 빌드
시간의 대부분이었다. **실측**: GitHub Actions 표준 러너(4코어 / 16GB,
`MAX_JOBS=2 NVCC_THREADS=1`)에서 이 패키지 하나가 **13,663초**, 네이티브 이미지 빌드
전체의 **88%**. 같은 컴파일이 RunPod A100 파드(cgroup 21코어 / 108GB,
`MAX_JOBS=5 NVCC_THREADS=4`)에서는 **2,323초**였다.

그래서 파드에서 한 번 빌드해 이 저장소의 GitHub Release 자산으로 올렸고,
`envs/native`는 그것을 URL로 설치한다. 상류 sdist는 그대로이므로 **소스 버전은
바뀌지 않았다** — `flash-attn 2.8.3.post1` 그대로다. 바뀐 것은 바이너리의 출처뿐이다.

### 이 변경이 측정에 대해 무엇을 바꾸고 무엇을 안 바꾸는가

**안 바꾸는 것.** 커널은 같은 소스에서 나왔고 gencode도 같다(sm_80/90/100). 지금까지
기록된 어떤 수치도 이 변경 때문에 움직이지 않는다 — 애초에 `docs/`의 모든 수치는 CPU
아니면 미측정이고, flash-attn 경로를 탄 측정은 존재한 적이 없다.

**바꾸는 것.** 바이너리 자체가 이제 상류가 아니라 우리 것이다. AGENTS.md가 "런마다
해석된 torch/framework 버전을 기록하라 — 버전은 결과에 보여야 하는 교란 변수다"라고
요구하는 대상이 하나 늘었다. 그 기록이 `docs/prebuilt-wheels.yaml`이고, 기계가 읽는
쪽이다. 누가·언제·무엇 위에서·어떤 arch로 빌드했는지, sha256과 크기가 거기 있다.

**바꾸지 않는데 바뀐 것처럼 보일 수 있는 것.** 빌드 시간은 이미지 빌드의 성질이지
측정 대상이 아니다. 13,663초는 벤치마크 결과가 아니라 CI 비용이다.

### 이 저장소가 직접 확인한 것 (2026-08-02, macOS)

| 항목 | 방법 | 결과 |
|---|---|---|
| URL이 익명으로 받아진다 | `curl -sIL` | HTTP 200, `content-length: 180912127` |
| sha256 | `uv lock`이 자산을 받아 `envs/native/uv.lock`에 기록 | `166a27d0…d2a`, 기록된 값과 일치 |
| 인터프리터 태그 | 휠의 `WHEEL` 메타데이터 | `Tag: cp313-cp313-linux_x86_64` |
| GPU arch | `flash_attn_2_cuda…so`의 `.nv_fatbin` 섹션을 직접 순회 | fatbin 72개, 각 3개씩 총 **216개 SASS 엔트리가 sm_80/90/100**, **PTX 엔트리 0개** |
| 다른 핀이 안 움직였다 | `uv lock` 전후 lock의 (name, version) 비교 | 142개 패키지 전부 동일, `flash-attn`의 `source`만 registry -> url |

PTX 엔트리가 0개라는 것은 Dockerfile 주석이 원래 주장하던 "JIT으로 흘러갈 경로가
없다"를 이 휠에 대해 실제로 확인한 것이다. 목록 밖의 arch를 가진 파드는 느려지는
것이 아니라 죽는다.

### 확인하지 **않은** 것 — 주장하지 않는다

- **이 휠이 실제로 import되는지는 측정 안 함.** CUDA GPU가 필요하고, 이 저장소에서
  GPU가 이 파일을 만진 적이 없다. 다음 이미지 빌드가 답해야 하는 것:
  `uv sync --locked`가 소스 빌드 없이(로그에 `Building flash-attn` 줄이 **없이**) 끝나고
  네이티브 이미지가 4시간이 아니라 몇 분 안에 나오는 것.
  다음 파드가 답해야 하는 것: `scripts/verify_env.py`의 native x 세 모델 프로브가
  `attn/fa2`에서 `import flash_attn` 이후 실제로 스텝을 도는 것.
- **빌드가 정말 저 파드에서 저 시간에 났는지는 보고받은 값이다.** 호스트·플래그·초는
  `docs/prebuilt-wheels.yaml`의 `verified.reported`에 그렇게 표시돼 있다.
- 휠은 `linux_x86_64` 전용이다. `envs/native`의 lock은 aarch64도 해석하므로 aarch64
  파드는 이 패키지에서 설치가 실패한다. 시끄러운 실패라 감사 대상에 넣지 않았지만,
  GH200을 쓰려면 휠을 하나 더 빌드해야 한다.

### `prebuilt-wheels` 감사 체크

URL로 고정한 휠은 리졸버가 **아무것도 검사하지 않는다.** uv는 URL 휠을 그대로
설치하므로, torch가 한 번 올라가면 어긋남은 파드 위에서 CUDA 오류로 나타난다 —
파드 시간을 다 쓴 뒤에. 그래서 주장을 여기서 대조한다. 상세 계약은 `docs/CONTRACTS.md`.

체크를 실제로 깨뜨려 보고 출력을 확인했다. 열 가지 변이 전부 이름을 불러 막았다:
lock의 torch를 2.14.0으로 올리기 / `requires-python`을 `>=3.13`으로 넓히기 /
`[tool.uv.sources]`를 지우고 재-lock(= 소스 빌드로 회귀) / 릴리스 URL의 인터프리터
태그 바꾸기 / `TRAINBENCH_CUDA_ARCHS`에 120 추가 / 기록 삭제 / sha256 한 바이트
뒤집기 / `abi.cuda`를 cu128로 / 기록이 다른 패키지를 가리켜 휠이 미기록으로 남기 /
릴리스 태그를 ABI를 말하지 않는 `v1`로 바꾸기.

### 남은 소스 빌드 — 확대하지 않고 보고만 한다

env별 lock에서 휠이 없어 소스에서 빌드되는 패키지(2026-08-02, lock 실측):

| env | 소스 빌드 |
|---|---|
| native | `causal-conv1d`, `deepspeed` |
| sentence-transformers / tevatron / unsloth | `causal-conv1d` |
| ms-swift | `causal-conv1d` 외 CUDA 아닌 것 4종 |
| axolotl | `causal-conv1d` 외 CUDA 아닌 것 7종 |

native의 소스 빌드 목록에는 2026-08-02 실측 당시 `transformer-engine-torch`도
있었다. 2026-08-03 `precision/mxfp8,nvfp4` 제거로 native 빌드에서 빠졌다 —
**빌드 시간 단축은 측정 안 함.** 아래 문단이 인용하는 `821f8f4` 빌드 로그는 완료
시각만 주고 소요 시간을 주지 않으며, `transformer-engine-torch`는
`causal-conv1d`와 동시에 빌드됐으므로 그 완료 시각 차이를 이 빌드에서 뺄 근거도
없다.

**`flash-attn`은 어느 env에도 더는 없다** — 원래 native에만 있었다. 같은 처리를
받을 다음 후보는 `causal-conv1d`다: 여섯 env 전부에 있고, arch를 좁히는 변수를
읽지 않아 gencode 아홉 개를 항상 빌드한다. 다만 **비용은 미측정이다.** 커밋
`821f8f4` 빌드 로그가 주는 것은 완료 시각뿐이고(`transformer-engine-torch` 1346.1초,
`causal-conv1d` 1418.4초), 이것은 소요 시간이 아니다. flash-attn과 달리 "네 시간"을
주장할 근거가 없으므로 이 레인은 손대지 않았다.

### 레인 게이트 (2026-08-02, wheel 레인)

| | 결과 |
|---|---|
| `uv run pytest` | 748 passed (기준선 735 + 신규 테스트 11 + 체크가 하나 늘어 `CHECKS`를 도는 parametrize 2건) |
| `ruff check` / `ruff format --check` | 지적 0건 |
| `scripts/audit_plan.py` | 12/15 (`prebuilt-wheels` 신설·통과, `verdicts-closed` 미해소는 기존 상태) |
| `scripts/env_report.py` 설정 경로 | 통과, `torch 2.13.0 / transformers 5.14.1` 기록 |

## gemma-4의 chat template 부재를 고쳤다 (2026-08-02)

1차 캠페인에서 세 프레임워크가 **같은 자리에서 같은 메시지로** 실패했다:
`native x gemma4_e2b / visual_tokens`, `native x gemma4_e2b /
multimodal_embed_forward`, `ms_swift x gemma4_e2b / visual_tokens`, `unsloth x
gemma4_e2b / visual_tokens` — 전부 `Cannot use apply_chat_template because this
processor does not have a chat template.`

세 프레임워크가 같은 실패를 낸다는 것은 프레임워크 문제가 아니다. 원인은 이 저장소가
모든 프로세서에 chat template이 있다고 가정한 것이었고, 실제로는
`google/gemma-4-E2B`(사전학습 체크포인트)에 없다. 진단과 결정은
docs/model-spec.md 결정 5, 형식 값은 `model.prompt_format`이다.

**실측한 것 (CPU, 실제 프로세서, 가중치 미적재)**

| | gemma4_e2b | qwen3_vl_emb_2b |
|---|---|---|
| `prompt_format` | `raw` | `chat_template` |
| `visual_tokens_per_image` (448x448) | 256 | 196 |
| `total_seq_len` | 265 | 221 |
| `image_token_id` | 258880 | 151655 |

같은 실행에서 `visual_token_count`가 통과한다 — pod에서 죽던 그 체크다. 이미지 목록을
행별로 묶는 변경도 함께 들어갔다(프로브 쪽. `scripts/bench.py`는 이미 그렇게 하고
있었다): 평평한 목록은 `Gemma4Processor`가 한 행의 이미지로 읽어 배치 자체를 거부한다.

**Qwen 두 모델이 재는 것은 바뀌지 않는다.** 행별 묶음 전후로 두 프로세서의 출력
텐서가 바이트 단위로 동일함을 실측했다(`input_ids` / `attention_mask` /
`pixel_values` / `image_grid_thw` / `mm_token_type_ids`).

**측정 안 함**: 이 수정이 실제로 통하는지는 pod에서만 판정된다. 여기서 실행한 것은
프로세서까지이고 모델 가중치도 GPU도 개입하지 않았다. 다음 캠페인이 초록으로
바꿔야 하는 칸:

| 칸 | 체크 |
|---|---|
| `native x gemma4_e2b` | `visual_tokens`, `multimodal_embed_forward` |
| `ms_swift x gemma4_e2b` | `visual_tokens` |
| `unsloth x gemma4_e2b` | `visual_tokens` |

`ms_swift`/`unsloth`의 gemma-4 칸은 이 체크 하나만 막고 있던 것이 아닐 수 있다 —
이 수정이 여는 것은 이 실패까지이고, 그 뒤에 무엇이 있는지는 미측정이다.

## 2차 Phase 0 캠페인 (2026-08-02, A100 18 pod, 커밋 `3ebcade`)

18개 pod 전부 결과를 올렸고(`outputs/orchestrate-phase0-rerun.json`, 18/18에
`pod_id`, `unreadable` 0), 아래 표는 그 18개 아티팩트만으로 만들었다. Phase 0은
**적재 여부만** 답한다. 속도·메모리·커널에 대해서는 아무것도 말하지 않는다.

### 어느 캠페인의 아티팩트인지부터 확인했다

결과 저장소 `jinwoo-cho/trainbench-results`에는 지금 pod 디렉터리 40개가 있다
(`results/<framework>/<model>/<pod_id>/`). 캠페인이 둘 섞이면 매트릭스는 없느니만
못하므로, 각 아티팩트의 `git_commit`을 원장의 `git_commit`과 대조해 분류했다.

| `git_commit` | pod 디렉터리 | 정체 |
|---|---|---|
| `3ebcade7` | 18 | 이번 캠페인. 원장의 `pod_id` 18개와 정확히 일치 |
| `7ede7d7e` | 19 | 1차 캠페인 |
| `a9dcc540` / `15320e7b` / `59d5908` | 각 1 | 단발 검증 pod |

**`report.py`의 `newest_per_combination`에 맡기면 안 된다.** 이 규칙의 최종
정렬키는 `timestamp`인데, 40개 `result.json` 중 `recorded_at`을 실은 것이
**0건**이라 `load_artifacts`가 파일의 로컬 mtime으로 대체한다. mtime은 pod이
올린 시각이 아니라 내려받은 시각이다. 실측: 모든 아티팩트의 timestamp를 동일하게
두고 같은 선택 규칙을 돌리면 18칸 중 **8칸이 1차 캠페인 아티팩트를 고른다**
(`axolotl x qwen3_vl_emb_2b`, `ms_swift x gemma4_e2b`,
`sentence_transformers x qwen3_5_0_8b`, `tevatron` 3칸, `unsloth x gemma4_e2b`,
`unsloth x qwen3_vl_emb_2b`). 한 번에 새로 받아오면 mtime이 전부 같아지므로,
이번에 우연히 맞은 것은 재현되지 않는다.

그래서 원장의 `pod_id` 18개에 해당하는 디렉터리만 별도 디렉터리로 복사하고
(복사할 때 각 `result.json`의 `git_commit`이 원장의 것과 같은지 단언한다),
그 디렉터리를 `--results`로 넘겼다. 아래 생성 구역이 "결과 18건, 아티팩트 18건"인
것이 그 결과다.

`report.py`가 원장을 결과 선별에 쓰지 않는 것(현재는 `launch_state`에만 쓴다)은
이 문서가 고칠 수 있는 범위 밖이다. **미해결로 남긴다.**

### 칸별 판정 — 1차와 나란히

체크 개수는 프레임워크마다 다르다(프로브가 프레임워크마다 다른 것을 묻는다). 같은
프레임워크의 1차/2차만 비교 가능하다.

| 칸 | 1차 (`7ede7d7e`) | 2차 (`3ebcade`) | 이동 |
|---|---|---|---|
| native x qwen3_vl_emb_2b | OK 12/12 | OK 12/12 | 유지 |
| native x qwen3_5_0_8b | OK 12/12 | OK 12/12 | 유지 |
| native x gemma4_e2b | FAIL 2건 (`visual_tokens`, `multimodal_embed_forward`) | OK 13/13 | **열림** |
| unsloth x qwen3_vl_emb_2b | FAIL `padding_side_alignment` | OK 9/9 | **열림** |
| unsloth x qwen3_5_0_8b | FAIL `padding_side_alignment` | OK 9/9 | **열림** |
| unsloth x gemma4_e2b | FAIL `visual_tokens` | OK 9/9 | **열림** |
| ms_swift x qwen3_vl_emb_2b | FAIL `get_model_processor` | OK 10/10 | **열림** |
| ms_swift x qwen3_5_0_8b | FAIL `get_model_processor` | OK 10/10 | **열림** |
| ms_swift x gemma4_e2b | FAIL `visual_tokens` | OK 10/10 | **열림** |
| sentence_transformers x 3종 | OK 9/9 | OK 9/9 | 유지 |
| tevatron x qwen3_vl_emb_2b | FAIL `dense_model_load` (ModuleNotFoundError) | FAIL `dense_model_load` (AttributeError) | **더 깊어짐** |
| tevatron x qwen3_5_0_8b | 같음 | 같음 | **더 깊어짐** |
| tevatron x gemma4_e2b | 같음 | 같음 | **더 깊어짐** |
| axolotl x qwen3_vl_emb_2b | FAIL `model_loader_load` (TypeError), 체크 4개 | FAIL `infonce_backward` (RuntimeError), 체크 8개 | **더 깊어짐** |
| axolotl x qwen3_5_0_8b | 같음 | 같음 | **더 깊어짐** |
| axolotl x gemma4_e2b | 같음 | 같음 | **더 깊어짐** |

18칸 중 7칸이 열렸고(native x gemma4_e2b, unsloth 3칸, ms_swift 3칸), 5칸은
1차에서도 초록이었으며(native 2칸, sentence_transformers 3칸), 나머지 6칸은
같은 이름이 아닌 **다른 실패로 이동했다**. 남은 두 프레임워크를 "여전히 실패"로
세면 무엇이 바뀌었는지가 지워진다.

**tevatron은 세 모델 전부다.** Qwen 두 개가 아니라 gemma-4까지 같은 자리에서 같은
방식으로 죽는다(`'Gemma4Config' object has no attribute 'pad_token_id'`).

### 실패 2종 — 무엇이 어디까지 갔는가

**tevatron / `dense_model_load` — AttributeError, 3칸 전부.** 핀 고정된 상류
소스(`texttron/tevatron@dd06310`, `src/tevatron/retriever/modeling/encoder.py`)에서
`DenseModel.load`는 다음 순서다.

```
166  base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
167  if base_model.config.pad_token_id is None:
168      base_model.config.pad_token_id = 0
```

1차에서는 8행의 `from peft import ...`에서 죽어 체크포인트를 건드리지도 못했다.
2차에서는 166행이 통과했다 — 즉 **가중치는 적재됐고** 167행에서 죽는다.
transformers 5.14.1의 합성 config(`Gemma4Config` / `Qwen3_5Config` /
`Qwen3VLConfig`)는 `pad_token_id`를 최상위에 두지 않아 속성 접근 자체가 예외다.
상류가 `getattr(..., None)`이 아니라 직접 접근을 쓴 것이 원인이며, 이 저장소가
의존성으로 고칠 수 있는 종류가 아니다.

`infonce_backward`는 `skipped: model did not load`로 기록된다. 나머지 세 체크
(`processor_load`, `padding_side_alignment`, `text_tokenize`)는 조기 반환으로
**등록조차 되지 않는다** — tevatron 칸의 "5 checks"는 native의 12-13과 분모가 다르다.

**axolotl / `infonce_backward` — RuntimeError, 3칸 전부.**
`expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`.
1차에서는 `model_loader_load`가 `TypeError: unsupported operand type(s) for //`로
죽어 체크가 4개뿐이었다. 2차에서는 적재·축 검증·패딩·토크나이즈까지 8개 중 7개가
통과하고 **역전파 한 스텝에서** 죽는다. 같은 실행의 `axes_verified`가
`precision.name`을 `mixed(bf16,fp32)`(요청은 `bf16`)로 기록한다 — 이 칸의 모델이
fp32와 bf16 파라미터를 섞어 들고 있다는 뜻이고, dtype 불일치와 같은 방향의 관측이다.
다만 **인과는 미확인**이다.

### `trainable_params` — 프레임워크 전체

`steps.infonce_backward`가 세는 것은 파라미터 **텐서 개수**이지 원소 수가 아니다
("아무것도 학습하지 않았는가"에 답하는 단위). `params_with_grad`는 이 스텝에서
실제로 grad가 닿은 텐서 수인데, 프로브의 배치는 텍스트 전용이므로 비전 타워가
빠지는 것이 정상이다 — `params_with_grad < trainable_params`는 결함이 아니다.

| 프레임워크 | qwen3_vl_emb_2b | qwen3_5_0_8b | gemma4_e2b |
|---|---|---|---|
| native | 625 / 625 (grad 310) | 473 / 473 (grad 320) | 988 / 988 (grad 505) |
| unsloth | 625 / 625 (grad 310) | 473 / 473 (grad 320) | 1048 / 1048 (grad 505) |
| ms_swift | 625 / 625 (grad 310) | 473 / 473 (grad 320) | 988 / 988 (grad 505) |
| sentence_transformers | 기록 없음 (grad 310) | 기록 없음 (grad 320) | 기록 없음 (grad 505) |
| tevatron | 없음 (모델 미적재) | 없음 | 없음 |
| axolotl | 없음 (체크 실패) | 없음 | 없음 |

표기는 `trainable_params / total_params`.

**이번 캠페인의 가장 중요한 숫자는 unsloth 세 칸이다.** 1차에서 세 칸 모두
`params_with_grad=0`, `trainable_params=0`으로 완전히 얼어붙은 그래프를
역전파하면서 `infonce_backward`가 통과했다. 2차에서는 세 칸 모두
`fast_vision_model_load`가 `full_finetuning: true`를 기록하고 학습 가능 텐서가
1048 / 473 / 625다. 새 가드가 붉게 만든 것이 아니라, **고쳐져서 통과했다.**

unsloth의 gemma-4만 텐서 수가 1048로 native·ms_swift의 988보다 60개 많다. unsloth의
패치가 텐서를 추가한다는 뜻이며, 그 이상은 **측정 안 함**이다.

### 이 가드가 덮지 않는 곳 — sentence_transformers

`sentence_transformers` 프로브는 `steps.infonce_backward`를 거치지 않고
자체 `_backward`를 쓴다(`trainbench/probe/sentence_transformers.py`). 그 함수는
`params_with_grad`만 반환하고 `trainable_params`를 세지도, 0인지 확인하지도 않는다.
**즉 unsloth를 잡아낸 가드가 이 프레임워크에는 없다.** 이번 세 칸의
`params_with_grad`는 310 / 320 / 505로 0이 아니니 실제로 학습은 일어났지만,
0이었다면 그대로 초록으로 지나갔을 것이다. **미해결로 남긴다.**

### 1차의 실패는 "미지원"이 아니라 배포판 누락이었다

1차에서 등급 대상 체크가 실패한 칸 중 **5칸**의 원인은 조합이 지원되지 않아서가
아니라 이미지에 배포판이 없어서였다. 매트릭스를 "미지원"으로 읽으면 안 된다.

| 1차 실패 | 칸 | 없던 배포판 |
|---|---|---|
| `dense_model_load` ModuleNotFoundError | tevatron x 3종 | `peft` |
| `get_model_processor` PackageNotFoundError | ms_swift x qwen3_vl_emb_2b, qwen3_5_0_8b | `qwen_vl_utils` |

여기에 unsloth 3칸의 `fast_sentence_transformer_accepts_vlm`이 별도로 붙는다 —
아래 "문서화된 한계가 아니었다" 참조. 등급 대상 실패는 아니었으므로 5칸과 따로 센다.

**tevatron: 상류가 선언하지 않는다.** 핀 고정된 `setup.py`의 `install_requires`는
`transformers>=4.10.0`, `datasets>=1.1.3` **둘뿐인데**, `encoder.py`는 8행에서
`from peft import LoraConfig, TaskType, get_peft_model, PeftModel`를 모듈 최상단에서
한다. 그래서 `envs/tevatron/pyproject.toml`이 `peft>=0.20`을 대신 선언한다.
`accelerate`는 따로 선언하지 않아도 `peft`의 의존성으로 따라 들어온다(lock 기준
1.14.0, pod 기록도 1.14.0). 같은 이유를 다시 캐지 않도록 여기에 남긴다.

**ms-swift: Qwen3.5(텍스트 전용)가 VL 경로로 적재된다.** 핀 고정된 소스
(`ms-swift 4.4.2`, `swift/model/models/qwen.py`)에서
`class Qwen3_5Loader(Qwen3VLLoader)`이고, `MLLMModelType.qwen3_5`로 등록되며
`requires=['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord']`다.
`Qwen3VLLoader._check_qwen_vl_utils`가 `require_version('qwen_vl_utils>=0.0.14')`를
잡히지 않는 곳에서 호출하는 것이 1차의 `PackageNotFoundError`였다. 이것은 의존성
메모가 아니라 **프레임워크의 구조적 사실이고, 뒤 단계의 처리량 교란 요인 후보다.**

교란 요인은 ms-swift만의 것이 아니다. 이번 캠페인에서 `qwen3_5_0_8b`를 적재한 칸은
전부 VL 경로를 탔다 — native는 `visual_tokens`가 이미지당 196 토큰으로 통과하고
`multimodal_embed_forward`도 통과한다(`seq_len` 209), unsloth·ms_swift는 프로세서가
`Qwen3VLProcessor`다. 이 체크포인트를 "텍스트 전용"으로 부르는 것은 config 수준에서
이미 성립하지 않는다.

**`decord`는 경고이지 실패가 아니다.** `envs/ms-swift/uv.lock`에 `decord`는 없고
(0건), 그럼에도 ms_swift 세 칸이 10/10으로 통과했다. 상류 코드가 그 이유를 말한다 —
`swift/model/model_meta.py`의 `ModelMeta.check_requires`는 `require_version`의
`ImportError`를 삼키고 `Please install the package: ...`를 `logger.warning`으로만
남긴다. `_check_qwen_vl_utils`와 달리 게이트가 아니다.

**unsloth: `sentence-transformers`는 상류 extra에서 왔다.** `unsloth[huggingface]`를
고르면 그 배포판 하나가 추가되고 새로운 제약은 생기지 않는다(`envs/unsloth/uv.lock`,
5.6.1). `FastSentenceTransformer`가 그것 없이는 import되지 않는다.

### 문서화된 한계가 아니었다 — unsloth 3칸의 예상 실패가 통과했다

`fast_sentence_transformer_accepts_vlm`은 `expected_failure=True`로 등록된
체크다. 1차에서 세 칸 모두 "예상대로 실패"했는데, 실패 내용은
`ImportError: Unsloth: To use FastSentenceTransformer, you must install
sentence-transformers.`였다. 즉 **프레임워크의 거부가 아니라 배포판 부재를
확인하고 있었다.** 배포판이 들어온 2차에서는 세 칸 모두 통과했고
(`accepted: true`, `model_class: SentenceTransformer`), 생성 구역이
"지원 매트릭스가 틀렸다"로 표시한다.

두 가지를 넘겨짚으면 안 된다.

- 위 매트릭스의 unsloth 칸은 `OK (9 checks, 문서화된 한계 1건)`로 렌더된다.
  그 "한계 1건"이 바로 **통과해버린** 체크다. 표만 읽으면 한계가 유지된 것으로
  읽힌다 — 바로 아래 절이 반대를 말한다.
- 프로브는 `FastSentenceTransformer.from_pretrained(hf_id, for_inference=True)`를
  호출한다. 통과가 말하는 것은 **추론 경로의 생성이 되더라**까지다. 학습 스텝이
  그 경로로 도는지, 1.8-3.3배 임베딩 가속이 이 연구의 모델에 적용되는지는
  **측정 안 함**이다.

### 초록이지만 그대로 믿으면 안 되는 것

- **`axes_verified`는 이제 `all_matched: false`를 거부한다.** 불일치는 결과에 그대로
  남고(`applied.axes`), 실패 메시지가 축마다 요청/적용을 이름으로 적는다. 위에 실측된
  불일치 둘 — `kernel.name`이 `none` 요청에 `fla` 적용, `precision.name`이 `bf16`
  요청에 `mixed(bf16,fp32)` 적용 — 은 사실 그대로 남고, 바뀐 것은 그것이 통과로
  읽히지 않는다는 점이다. `kernel.name`의 불일치에는 그것이 이미지에 구속된
  것인지(`environment-bound`)가 함께 적힌다 — `axes._environment_bound_kernel`이 읽는
  값이다. **다른 축에는 그 구별이 없다.** `precision.name`의 `mixed(bf16,fp32)`가
  axolotl의 fp32 유지 정책인지 잘못된 적재인지는 여기서 읽을 수 없고, 읽을 수 없는
  것을 environment-bound로 부르지 않는다.
- **CPU 호스트에서는 모든 probe 칸이 이 체크에서 빨개진다.** `steps.dtype_for`가 CUDA
  밖에서 fp32를 주는데 `configs/precision/`에 fp32 값이 없고, fused AdamW 커널은 CUDA
  전용인데 `configs/optim/`에 unfused AdamW 값이 없다. 파드에서 초록이 되는지는
  **확인 안 함**.
- **unsloth의 `padding_side_alignment`가 초록인 것은 불일치가 없다는 뜻이 아니다.**
  Qwen 두 칸의 detail이 `disagreed: [processor, tokenizer]`,
  `framework_forced: [processor, tokenizer]`를 기록한다. unsloth는 left로 오고
  체크포인트는 right이며 프로브가 right로 강제한다. 1차에서는 이 강제가 실패였고,
  지금은 "감지하고 교정했다"가 초록이다.
- **tevatron의 `framework_version`이 `version: "unknown"`으로 통과한다.** 버전이
  결과에 보여야 한다는 규칙(AGENTS.md)에 이 칸은 답하지 못한다.
- **해석 스택이 칸마다 다르다.** transformers는 5.5.0(unsloth) / 5.12.1(ms_swift) /
  5.14.1(나머지), torch는 2.11.0 / 2.12.1 / 2.13.0으로 갈린다. 적재 여부만 묻는
  Phase 0에서는 감수하지만, 처리량을 비교할 때는 프레임워크 차이와 분리되지 않는다.

### 재현

```
hf download jinwoo-cho/trainbench-results --repo-type dataset --local-dir <dir>
# 원장의 pod_id 18개에 해당하는 results/<fw>/<model>/<pod>/ 만 <stage>/results/ 로 복사
uv run python scripts/report.py --results <stage> --ledger outputs/orchestrate-phase0-rerun.json
```

복사 단계는 여전히 필요하지만, 이유가 바뀌었다. `report.py`는 더 이상 mtime으로
캠페인을 고르지 않는다 — `recorded_at`이 없는 아티팩트를 **거부하고 stderr에 이름을
적는다**. 결과 저장소에 있는 그 필드 이전의 아티팩트들은 이제 조용히 섞이는 대신
건너뛰어지므로, 복사를 빼면 옛 수치가 실리는 것이 아니라 칸이 비어 보인다.

---

## 이 표를 나란히 읽으면 안 되는 자리 (2026-08-03, wave 3 통합)

세 결정이 이 문서를 읽는 방식을 바꾼다. 셋 다 코드에 이미 들어가 있고, 여기 적는 것은
표가 침묵하는 부분이다.

### 결정 4 — 스택이 다른 칸은 나란히 놓지 않는다

해석 스택이 칸마다 다르다: transformers 5.5.0(unsloth) / 5.12.1(ms_swift) /
5.14.1(나머지), torch 2.11.0 / 2.12.1 / 2.13.0. 공유 lockfile이 버전을 하나로 강제하고
`hydra-core` ↔ axolotl 의 antlr4 정확 고정은 해소 불가라, 한 환경에 여섯을 넣는 것은
확정적으로 불가능하다. 그래서 **`report.py`가 같은 스택끼리만 줄을 세운다.** 한 표에
전부 넣고 버전을 열로 두는 안은 기각했다 — 독자는 순위를 먼저 읽고 각주를 나중에
읽는다. 스택을 못 읽은 레코드는 `스택 미상`으로 따로 선다.

**따라서 이 문서의 자동 생성 표에서 세로로 이웃한 두 칸이 비교 가능하다는 보장은
없다.** 비교 가능한 묶음은 `report.py`가 스택별로 나눠 낸 것뿐이다.

### 결정 5 의 대가 — ablation 그리드가 프레임워크마다 들쭉날쭉하다

프레임워크의 학습 스텝을 그대로 잰다(베이스 인코더만 꺼내 공통 루프에 태우면 프레임워크가
아니라 우리 루프를 재게 된다). 그 대가가 **모든 칸이 같은 축을 갖지 않는다**는 것이다.

`tevatron` 칸이 그 유일한 실례다. `DenseModel.forward`가 인코딩·풀링·정규화·스코어링·
InfoNCE·분산 게더를 한 호출 안에서 전부 하므로, 그 칸에서는 하네스의 손실이 아예 돌지
않는다. 두 축이 프레임워크 소유로 기록된다:

| 축 | tevatron 칸에서 | 다른 다섯 칸에서 |
|---|---|---|
| `loss.name` | 프레임워크 소유. `state="framework_owned"`, 값은 비어 있다 | 하네스가 적용하고 되읽는다 |
| `parallel.cross_device_negatives` | 프레임워크 소유. `is_ddp`일 때 같은 forward가 게더한다 | 하네스가 적용하고 되읽는다 |

`framework_owned`는 `undetermined`와 **다른 상태**다. 후자는 "아무도 읽지 않았다"이고
런을 막는다. 전자는 "다른 코드가 정했고 그 사실이 기록됐다"이며 런을 막지 않는다. 둘을
한 상태로 뭉개면 tevatron 칸이 통째로 측정 불가가 되거나, 반대로 읽히지 않은 축이 통과한다.
그래서 결과 JSON은 `all_determined` / `all_matched` 옆에 최상위 `framework_owned` 키를
따로 싣는다.

**이 그리드에서 `loss` 축 ablation은 다섯 칸에서만 의미가 있다.** tevatron 열의 그 자리를
빈칸이 아니라 "해당 없음"으로 읽어야 한다.

### 결정 6 — `kernel=kernels_hub` 축 값을 버렸다

`kernel` 그룹은 `none` / `liger` / `fla` 셋이다. 버린 이유가 **둘이고 서로 독립이다** —
하나가 해소돼도 다른 하나가 남는다.

1. 진입점 둘(`from_pretrained(use_kernels=True)`, `integrations.hub_kernels.kernelize(model)`)이
   **모델 객체를 요구**하는데 `axes.patch`는 모델 생성 **전에** 돈다. 적용 지점을 뒤로
   옮기는 것은 "kernel/attn은 모델 생성 전에만 바꿀 수 있다"는 설계 전제를 깨는 것이라
   기각했다(`docs/CONTRACTS.md §2`).
2. `envs/native`가 `kernels==0.16.0`을 핀하는데 transformers 5.14.1의 창은
   `0.15.2 <= v < 0.16.0`으로 상한 배타적이다. `is_kernels_available()`이 False가 되고
   `use_kernel_forward_from_hub`가 **조용히 항등 데코레이터**가 된다
   (`.plans/research/axis-libraries.md §3.1-3.2`).

`trainbench/axes.py`의 `KERNEL_MODULE_ROOTS["kernels"]`는 **남겼다.** 그 표는 적용 표가
아니라 되읽기 표이고, 어댑터가 스스로 hub dispatch를 켜면 모델은 여전히 `kernels` 패키지의
클래스로 만들어진다. 행을 지우면 그것이 `none`으로 읽혀 `kernel=none` 요청과
**일치해버린다** — 지금은 `assert_matches`가 그 런을 막는다.

### 어댑터 여섯의 진입점 대조 — 여섯 중 0이 프레임워크의 학습 진입점을 쓴다

`trainbench/loader.py::ADAPTERS`의 `documented_entry_point`를 그대로 옮긴 것이다.
`differs`는 두 진입점 문자열이 다른지를 계약이 스스로 검사한 결과다.

| 어댑터 | 프레임워크가 문서화한 학습 진입점 | 하네스가 쓰는 것 | `differs` |
|---|---|---|---|
| native | 없음 — transformers 는 LM head 없는 임베딩 모델의 학습 진입점을 문서화하지 않는다 | `AutoModel.from_pretrained` + 손으로 쓴 루프 | `false` |
| unsloth | `FastVisionModel.from_pretrained` → `for_training(model)` → TRL `SFTTrainer` | `from_pretrained(full_finetuning=...)` + 하네스 루프. `for_training()` 도 `SFTTrainer` 도 안 쓴다 | `true` |
| ms_swift | `cli_main` → `TrainerFactory 'embedding'` → `EmbeddingTrainer` + `loss_map['infonce']` | `get_model_processor` + 하네스 루프 | `true` |
| sentence_transformers | `SentenceTransformerTrainer.compute_loss` — 모델 forward 를 부르는 것은 Trainer 가 아니라 손실 객체다 | `SentenceTransformer(...)` + 하네스 루프. trainer 도 ST 손실 클래스도 안 쓴다 | `true` |
| tevatron | `EncoderModel.forward(query=, passage=)` — 인코딩부터 게더까지 한 호출 | **그 forward 자체**를 하네스 타이머로 돌린다. 손실과 cross-device negatives 는 프레임워크 소유로 기록 | `true` |
| axolotl | `axolotl.cli.main` → `axolotl.train:train` → `setup_trainer` → HF Trainer 하위클래스, accelerate 로 기동 | `ModelLoader(cfg, tokenizer).load()` + `axes.step_context` 안의 하네스 루프. trainer 도 Accelerator 도 없다 | `true` |

**여섯 중 프레임워크의 문서화된 학습 진입점을 그대로 타는 칸은 없다.** native 의
`differs=false` 는 "같은 것을 쓴다"가 아니라 "프레임워크가 문서화한 것이 없어서 하네스
자신이 기준 경로"라는 뜻이고, tevatron 은 문서화된 forward 를 쓰지만 그 위의 trainer 를
쓰지 않아 `true` 다. 그래서 이 벤치마크가 재는 것은 "프레임워크를 문서대로 썼을 때의
속도"가 아니라 **"프레임워크가 만든 모델·스텝을 같은 하네스로 돌렸을 때의 속도"** 다.
그 구별이 결과를 읽는 조건이다.

axolotl 칸에는 하나가 더 붙는다. `required_step_context`(autocast, cuda, bfloat16)로
`embed_tokens`/`lm_head` 를 fp32 로 둔 채 bf16 본체와 matmul 하게 만든다(결정 1). 따라서
**native(순수 bf16)와 axolotl(autocast)은 다른 수치 체제에서 비교된다.** 그 사실은
`documented_entry_point.differs` 와 `required_step_context` 양쪽에 남는다. autocast 를
켠 axolotl 과 끄고 잰 axolotl 의 속도 차는 **측정 안 함** — 이 호스트에 CUDA 가 없다.

### precision 6칸 FAIL은 의도된 결과다

아래 생성 구역의 "실패 상세"에 `precision.name: requested 'bf16', applied
'mixed(bf16,fp32)'` AppliedMismatch가 여섯 칸(axolotl 3종, unsloth 3종)에서
나온다. 이것은 결함이 아니라 **영구히 동결된, 의도된 불일치**다.

axolotl과 unsloth 둘 다 `embed_tokens`/`lm_head`류 파라미터를 fp32로 둔 채
나머지를 bf16으로 적재한다(위 "어댑터 여섯의 진입점 대조" 참조, axolotl은 결정
1의 autocast로 그 위에서 matmul한다). 요청은 `precision=bf16`이지만 실제로
빌드된 모델은 fp32/bf16이 섞여 있으므로 `applied._capture_precision`은 정직하게
`mixed(bf16,fp32)`로 읽는다 — `bf16`으로 뭉개면 섞인 정밀도를 순정 bf16으로
잘못 보고하는 것이 된다.

`tests/contract/test_applied_axes.py`의 `UNNAMEABLE` 테이블이 이 값을
`precision.name`의 고정 기대값으로 못박는다(`mixed(bf16,fp32)`, 다른 값으로
변할 수 없음을 계약이 보증). 즉 이 여섯 칸은 두 어댑터가 고쳐질 수 있는
버그가 아니라 **어댑터의 문서화된 로딩 방식과 순정 bf16 요청 사이의 구조적
불일치**이고, `assert_matches`가 그것을 timing 런 차단으로 정직하게 반영한다.
붉은 여섯 칸이 이 프로젝트가 원하는 상태다 — 초록으로 만들려면 어댑터가 아니라
계약을 속여야 한다.

---

여기부터 아래는 `scripts/report.py`가 생성한다. 아직 pod 결과가 없으면 비어 있다.

<!-- generated: probe results -->

## 모델 x 프레임워크 적재 검증 (자동 생성)

결과 18건, 아티팩트 18건. `미시도`는 pod을 띄운 적이 없는 조합, `결과 없음(기동됨)`는 띄웠으나 결과 파일이 올라오지 않은 조합, `미지원(문서화됨)`는 모든 체크가 문서화된 한계였던 조합이다.

| | qwen3_vl_emb_2b | qwen3_5_0_8b | gemma4_e2b |
|---|---|---|---|
| native | OK (12 checks) | OK (12 checks) | OK (13 checks) |
| unsloth | FAIL axes_verified (AppliedMismatch) | FAIL axes_verified (AppliedMismatch) | FAIL axes_verified (AppliedMismatch) |
| ms_swift | OK (10 checks) | OK (10 checks) | OK (10 checks) |
| sentence_transformers | OK (9 checks) | OK (9 checks) | OK (9 checks) |
| tevatron | OK (10 checks) | OK (10 checks) | OK (10 checks) |
| axolotl | FAIL axes_verified (AppliedMismatch) | FAIL axes_verified (AppliedMismatch) | FAIL axes_verified (AppliedMismatch) |

### 지원 매트릭스가 틀렸다 — 실패할 것으로 표시한 체크가 통과했다

문서화된 한계가 사라졌다는 뜻이므로, 해당 셀의 근거를 다시 확인해야 한다.

- **unsloth x gemma4_e2b** — fast_sentence_transformer_accepts_vlm
- **unsloth x qwen3_5_0_8b** — fast_sentence_transformer_accepts_vlm
- **unsloth x qwen3_vl_emb_2b** — fast_sentence_transformer_accepts_vlm

### 실행 환경별 해석 버전

| 조합 | torch | transformers | 프레임워크 |
|---|---|---|---|
| axolotl x gemma4_e2b | 2.12.1+cu130 | 5.14.1 | 0.18.0 |
| axolotl x qwen3_5_0_8b | 2.12.1+cu130 | 5.14.1 | 0.18.0 |
| axolotl x qwen3_vl_emb_2b | 2.12.1+cu130 | 5.14.1 | 0.18.0 |
| ms_swift x gemma4_e2b | 2.13.0+cu130 | 5.12.1 | 4.4.2 |
| ms_swift x qwen3_5_0_8b | 2.13.0+cu130 | 5.12.1 | 4.4.2 |
| ms_swift x qwen3_vl_emb_2b | 2.13.0+cu130 | 5.12.1 | 4.4.2 |
| native x gemma4_e2b | 2.13.0+cu130 | 5.14.1 | - |
| native x qwen3_5_0_8b | 2.13.0+cu130 | 5.14.1 | - |
| native x qwen3_vl_emb_2b | 2.13.0+cu130 | 5.14.1 | - |
| sentence_transformers x gemma4_e2b | 2.13.0+cu130 | 5.14.1 | 5.6.1 |
| sentence_transformers x qwen3_5_0_8b | 2.13.0+cu130 | 5.14.1 | 5.6.1 |
| sentence_transformers x qwen3_vl_emb_2b | 2.13.0+cu130 | 5.14.1 | 5.6.1 |
| tevatron x gemma4_e2b | 2.13.0+cu130 | 5.14.1 | unknown |
| tevatron x qwen3_5_0_8b | 2.13.0+cu130 | 5.14.1 | unknown |
| tevatron x qwen3_vl_emb_2b | 2.13.0+cu130 | 5.14.1 | unknown |
| unsloth x gemma4_e2b | 2.11.0+cu130 | 5.5.0 | 2026.7.6 |
| unsloth x qwen3_5_0_8b | 2.11.0+cu130 | 5.5.0 | 2026.7.6 |
| unsloth x qwen3_vl_emb_2b | 2.11.0+cu130 | 5.5.0 | 2026.7.6 |

### 실패 상세

- **axolotl x gemma4_e2b / axes_verified** — AppliedMismatch
  - `the model that was built is not the one this run asked for: precision.name: requested 'bf16', applied 'mixed(bf16,fp32)'`
- **axolotl x qwen3_5_0_8b / axes_verified** — AppliedMismatch
  - `the model that was built is not the one this run asked for: precision.name: requested 'bf16', applied 'mixed(bf16,fp32)'`
- **axolotl x qwen3_5_0_8b / infonce_backward** — RuntimeError
  - `expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`
- **axolotl x qwen3_vl_emb_2b / axes_verified** — AppliedMismatch
  - `the model that was built is not the one this run asked for: precision.name: requested 'bf16', applied 'mixed(bf16,fp32)'`
- **unsloth x gemma4_e2b / axes_verified** — AppliedMismatch
  - `the model that was built is not the one this run asked for: precision.name: requested 'bf16', applied 'mixed(bf16,fp32)'`
- **unsloth x qwen3_5_0_8b / axes_verified** — AppliedMismatch
  - `the model that was built is not the one this run asked for: precision.name: requested 'bf16', applied 'mixed(bf16,fp32)'`
- **unsloth x qwen3_vl_emb_2b / axes_verified** — AppliedMismatch
  - `the model that was built is not the one this run asked for: precision.name: requested 'bf16', applied 'mixed(bf16,fp32)'`

### 병합에서 제외한 파일

- 중복: axolotl x gemma4_e2b: ignored results/axolotl/gemma4_e2b/jmfxvky2jntn0v/result.json
- 중복: axolotl x gemma4_e2b: ignored results/axolotl/gemma4_e2b/x0yrie57seu0mm/started.json
- 중복: axolotl x gemma4_e2b: ignored results/axolotl/gemma4_e2b/jmfxvky2jntn0v/started.json
- 중복: axolotl x gemma4_e2b: ignored results/axolotl/gemma4_e2b/96ykbqpg8zhv4k/started.json
- 중복: axolotl x gemma4_e2b: ignored results/axolotl/gemma4_e2b/re17q5hfpr2qdd/started.json
- 중복: axolotl x qwen3_5_0_8b: ignored results/axolotl/qwen3_5_0_8b/3yojefsdd6hk0u/result.json
- 중복: axolotl x qwen3_5_0_8b: ignored results/axolotl/qwen3_5_0_8b/zql0z8hc4k8dlx/result.json
- 중복: axolotl x qwen3_5_0_8b: ignored results/axolotl/qwen3_5_0_8b/zql0z8hc4k8dlx/started.json
- 중복: axolotl x qwen3_5_0_8b: ignored results/axolotl/qwen3_5_0_8b/xzrx2gnudntf09/started.json
- 중복: axolotl x qwen3_5_0_8b: ignored results/axolotl/qwen3_5_0_8b/3yojefsdd6hk0u/started.json
- 중복: axolotl x qwen3_5_0_8b: ignored results/axolotl/qwen3_5_0_8b/5zjp3w6lt56d4j/started.json
- 중복: axolotl x qwen3_5_0_8b: ignored results/axolotl/qwen3_5_0_8b/pjn3jrv0dy59ql/started.json
- 중복: axolotl x qwen3_vl_emb_2b: ignored results/axolotl/qwen3_vl_emb_2b/2ounbt5px9bmh9/result.json
- 중복: axolotl x qwen3_vl_emb_2b: ignored results/axolotl/qwen3_vl_emb_2b/53lb8wroqbw4mz/started.json
- 중복: axolotl x qwen3_vl_emb_2b: ignored results/axolotl/qwen3_vl_emb_2b/2ounbt5px9bmh9/started.json
- 중복: axolotl x qwen3_vl_emb_2b: ignored results/axolotl/qwen3_vl_emb_2b/twkpqbpknu9v9w/started.json
- 중복: axolotl x qwen3_vl_emb_2b: ignored results/axolotl/qwen3_vl_emb_2b/117ldk6qywwda3/started.json
- 중복: ms_swift x gemma4_e2b: ignored results/ms_swift/gemma4_e2b/6cp180d0yom9t5/result.json
- 중복: ms_swift x gemma4_e2b: ignored results/ms_swift/gemma4_e2b/ob790ntraktfvt/started.json
- 중복: ms_swift x gemma4_e2b: ignored results/ms_swift/gemma4_e2b/6cp180d0yom9t5/started.json
- 중복: ms_swift x gemma4_e2b: ignored results/ms_swift/gemma4_e2b/lfwess4lnnkdba/started.json
- 중복: ms_swift x gemma4_e2b: ignored results/ms_swift/gemma4_e2b/106pq7lep4ndot/started.json
- 중복: ms_swift x qwen3_5_0_8b: ignored results/ms_swift/qwen3_5_0_8b/kwjo8058tcawrj/result.json
- 중복: ms_swift x qwen3_5_0_8b: ignored results/ms_swift/qwen3_5_0_8b/d8b8vedzxv0ced/result.json
- 중복: ms_swift x qwen3_5_0_8b: ignored results/ms_swift/qwen3_5_0_8b/mcq25mjvnxgcb0/started.json
- 중복: ms_swift x qwen3_5_0_8b: ignored results/ms_swift/qwen3_5_0_8b/kwjo8058tcawrj/started.json
- 중복: ms_swift x qwen3_5_0_8b: ignored results/ms_swift/qwen3_5_0_8b/d8b8vedzxv0ced/started.json
- 중복: ms_swift x qwen3_5_0_8b: ignored results/ms_swift/qwen3_5_0_8b/bjdyt8s8l7eb0r/started.json
- 중복: ms_swift x qwen3_5_0_8b: ignored results/ms_swift/qwen3_5_0_8b/rz2t3hjjctb9ir/started.json
- 중복: ms_swift x qwen3_vl_emb_2b: ignored results/ms_swift/qwen3_vl_emb_2b/3lse8mupfa1rep/result.json
- 중복: ms_swift x qwen3_vl_emb_2b: ignored results/ms_swift/qwen3_vl_emb_2b/h5ox9ep7lu2o3e/started.json
- 중복: ms_swift x qwen3_vl_emb_2b: ignored results/ms_swift/qwen3_vl_emb_2b/3lse8mupfa1rep/started.json
- 중복: ms_swift x qwen3_vl_emb_2b: ignored results/ms_swift/qwen3_vl_emb_2b/92grvnmui311st/started.json
- 중복: ms_swift x qwen3_vl_emb_2b: ignored results/ms_swift/qwen3_vl_emb_2b/ngvpq0n6jwehzk/started.json
- 중복: native x gemma4_e2b: ignored results/native/gemma4_e2b/0vr6kgfeiqptb2/result.json
- 중복: native x gemma4_e2b: ignored results/native/gemma4_e2b/ls6huw5arfzj1j/started.json
- 중복: native x gemma4_e2b: ignored results/native/gemma4_e2b/0vr6kgfeiqptb2/started.json
- 중복: native x gemma4_e2b: ignored results/native/gemma4_e2b/n2lsusgmhk45xw/started.json
- 중복: native x gemma4_e2b: ignored results/native/gemma4_e2b/oegus80eth8r75/started.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/9rmt1v6qtm4f5x/result.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/pawcygtc073uzi/result.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/9t0p0tl7o2e0n5/result.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/ooqmou59fib4du/started.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/9rmt1v6qtm4f5x/started.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/pawcygtc073uzi/started.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/9t0p0tl7o2e0n5/started.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/k9wkyvgstvnq2x/started.json
- 중복: native x qwen3_5_0_8b: ignored results/native/qwen3_5_0_8b/o7lrjq7e02enqv/started.json
- 중복: native x qwen3_vl_emb_2b: ignored results/native/qwen3_vl_emb_2b/37qsvrojng72yw/result.json
- 중복: native x qwen3_vl_emb_2b: ignored results/native/qwen3_vl_emb_2b/udt814eokzll0r/started.json
- 중복: native x qwen3_vl_emb_2b: ignored results/native/qwen3_vl_emb_2b/37qsvrojng72yw/started.json
- 중복: native x qwen3_vl_emb_2b: ignored results/native/qwen3_vl_emb_2b/cjv20cvy38c84m/started.json
- 중복: native x qwen3_vl_emb_2b: ignored results/native/qwen3_vl_emb_2b/cs5m0fd2lmcmn8/started.json
- 중복: sentence_transformers x gemma4_e2b: ignored results/sentence_transformers/gemma4_e2b/00qrf8y4rl17xa/result.json
- 중복: sentence_transformers x gemma4_e2b: ignored results/sentence_transformers/gemma4_e2b/kn2h1o4snzgr0n/started.json
- 중복: sentence_transformers x gemma4_e2b: ignored results/sentence_transformers/gemma4_e2b/00qrf8y4rl17xa/started.json
- 중복: sentence_transformers x gemma4_e2b: ignored results/sentence_transformers/gemma4_e2b/dkjq8um6a26b29/started.json
- 중복: sentence_transformers x gemma4_e2b: ignored results/sentence_transformers/gemma4_e2b/p3ffx6xcg05ksv/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/ave6fhei1uedbk/result.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/ppybhr9spj53cn/result.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/4ls03h0lctaxj9/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/ave6fhei1uedbk/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/ppybhr9spj53cn/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/98as4en2an38rs/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/ctsn3ky63mvul0/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/2i0ptlkcc5621p/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/1jscf6cxmjz72y/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/xchraazlhvqt6y/started.json
- 중복: sentence_transformers x qwen3_5_0_8b: ignored results/sentence_transformers/qwen3_5_0_8b/3tmi4ht24hs5uz/started.json
- 중복: sentence_transformers x qwen3_vl_emb_2b: ignored results/sentence_transformers/qwen3_vl_emb_2b/os2q6ynmjsgk9g/result.json
- 중복: sentence_transformers x qwen3_vl_emb_2b: ignored results/sentence_transformers/qwen3_vl_emb_2b/1jwamawidc0yql/started.json
- 중복: sentence_transformers x qwen3_vl_emb_2b: ignored results/sentence_transformers/qwen3_vl_emb_2b/os2q6ynmjsgk9g/started.json
- 중복: sentence_transformers x qwen3_vl_emb_2b: ignored results/sentence_transformers/qwen3_vl_emb_2b/adsvsynn0j2pct/started.json
- 중복: sentence_transformers x qwen3_vl_emb_2b: ignored results/sentence_transformers/qwen3_vl_emb_2b/zz21apdq19z46i/started.json
- 중복: tevatron x gemma4_e2b: ignored results/tevatron/gemma4_e2b/8uxfqkoz32isx2/result.json
- 중복: tevatron x gemma4_e2b: ignored results/tevatron/gemma4_e2b/ft9co3xj9xwysm/started.json
- 중복: tevatron x gemma4_e2b: ignored results/tevatron/gemma4_e2b/8uxfqkoz32isx2/started.json
- 중복: tevatron x gemma4_e2b: ignored results/tevatron/gemma4_e2b/j0i8cqmakhz6bk/started.json
- 중복: tevatron x gemma4_e2b: ignored results/tevatron/gemma4_e2b/dh4m30ex41frn1/started.json
- 중복: tevatron x qwen3_5_0_8b: ignored results/tevatron/qwen3_5_0_8b/p2hlma9znui9l2/result.json
- 중복: tevatron x qwen3_5_0_8b: ignored results/tevatron/qwen3_5_0_8b/afzgznxkvwhz42/result.json
- 중복: tevatron x qwen3_5_0_8b: ignored results/tevatron/qwen3_5_0_8b/4nvrf3sytmassu/started.json
- 중복: tevatron x qwen3_5_0_8b: ignored results/tevatron/qwen3_5_0_8b/p2hlma9znui9l2/started.json
- 중복: tevatron x qwen3_5_0_8b: ignored results/tevatron/qwen3_5_0_8b/afzgznxkvwhz42/started.json
- 중복: tevatron x qwen3_5_0_8b: ignored results/tevatron/qwen3_5_0_8b/wwo36vlg1onptz/started.json
- 중복: tevatron x qwen3_5_0_8b: ignored results/tevatron/qwen3_5_0_8b/n3r8t21c0clle3/started.json
- 중복: tevatron x qwen3_vl_emb_2b: ignored results/tevatron/qwen3_vl_emb_2b/55738u0yw7s2v8/result.json
- 중복: tevatron x qwen3_vl_emb_2b: ignored results/tevatron/qwen3_vl_emb_2b/2ffarhkv2n55zi/started.json
- 중복: tevatron x qwen3_vl_emb_2b: ignored results/tevatron/qwen3_vl_emb_2b/55738u0yw7s2v8/started.json
- 중복: tevatron x qwen3_vl_emb_2b: ignored results/tevatron/qwen3_vl_emb_2b/poo9rswlunjkj5/started.json
- 중복: tevatron x qwen3_vl_emb_2b: ignored results/tevatron/qwen3_vl_emb_2b/6kg59vfo5dbfpe/started.json
- 중복: unsloth x gemma4_e2b: ignored results/unsloth/gemma4_e2b/gnsyr8b60cui3b/result.json
- 중복: unsloth x gemma4_e2b: ignored results/unsloth/gemma4_e2b/11o56rwd03txyi/started.json
- 중복: unsloth x gemma4_e2b: ignored results/unsloth/gemma4_e2b/gnsyr8b60cui3b/started.json
- 중복: unsloth x gemma4_e2b: ignored results/unsloth/gemma4_e2b/xlz4cbame1awm4/started.json
- 중복: unsloth x gemma4_e2b: ignored results/unsloth/gemma4_e2b/32plsncjtkxzvv/started.json
- 중복: unsloth x qwen3_5_0_8b: ignored results/unsloth/qwen3_5_0_8b/uy5q5fuhu514wp/result.json
- 중복: unsloth x qwen3_5_0_8b: ignored results/unsloth/qwen3_5_0_8b/tdvi81oek019ll/result.json
- 중복: unsloth x qwen3_5_0_8b: ignored results/unsloth/qwen3_5_0_8b/2v0hypp3wshksa/started.json
- 중복: unsloth x qwen3_5_0_8b: ignored results/unsloth/qwen3_5_0_8b/uy5q5fuhu514wp/started.json
- 중복: unsloth x qwen3_5_0_8b: ignored results/unsloth/qwen3_5_0_8b/tdvi81oek019ll/started.json
- 중복: unsloth x qwen3_5_0_8b: ignored results/unsloth/qwen3_5_0_8b/hjqfwu95965l8p/started.json
- 중복: unsloth x qwen3_5_0_8b: ignored results/unsloth/qwen3_5_0_8b/yh32nxs19wlqv2/started.json
- 중복: unsloth x qwen3_vl_emb_2b: ignored results/unsloth/qwen3_vl_emb_2b/lexjdx5etl0hrb/result.json
- 중복: unsloth x qwen3_vl_emb_2b: ignored results/unsloth/qwen3_vl_emb_2b/a8yzn3uq22tdb7/started.json
- 중복: unsloth x qwen3_vl_emb_2b: ignored results/unsloth/qwen3_vl_emb_2b/lexjdx5etl0hrb/started.json
- 중복: unsloth x qwen3_vl_emb_2b: ignored results/unsloth/qwen3_vl_emb_2b/vfamxhetm6flkp/started.json
- 중복: unsloth x qwen3_vl_emb_2b: ignored results/unsloth/qwen3_vl_emb_2b/n75371z1ll7tz0/started.json
- 판독 불가: results/axolotl/gemma4_e2b/96ykbqpg8zhv4k/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/axolotl/gemma4_e2b/re17q5hfpr2qdd/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/axolotl/qwen3_5_0_8b/5zjp3w6lt56d4j/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/axolotl/qwen3_5_0_8b/pjn3jrv0dy59ql/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/axolotl/qwen3_vl_emb_2b/117ldk6qywwda3/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/axolotl/qwen3_vl_emb_2b/twkpqbpknu9v9w/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/ms_swift/gemma4_e2b/106pq7lep4ndot/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/ms_swift/gemma4_e2b/lfwess4lnnkdba/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/ms_swift/qwen3_5_0_8b/bjdyt8s8l7eb0r/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/ms_swift/qwen3_5_0_8b/rz2t3hjjctb9ir/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/ms_swift/qwen3_vl_emb_2b/92grvnmui311st/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/ms_swift/qwen3_vl_emb_2b/ngvpq0n6jwehzk/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/native/gemma4_e2b/n2lsusgmhk45xw/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/native/gemma4_e2b/oegus80eth8r75/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/native/qwen3_5_0_8b/k9wkyvgstvnq2x/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/native/qwen3_5_0_8b/o7lrjq7e02enqv/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/native/qwen3_vl_emb_2b/cjv20cvy38c84m/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/native/qwen3_vl_emb_2b/cs5m0fd2lmcmn8/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/gemma4_e2b/dkjq8um6a26b29/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/gemma4_e2b/p3ffx6xcg05ksv/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/qwen3_5_0_8b/1jscf6cxmjz72y/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/qwen3_5_0_8b/2i0ptlkcc5621p/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/qwen3_5_0_8b/3tmi4ht24hs5uz/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/qwen3_5_0_8b/98as4en2an38rs/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/qwen3_5_0_8b/ctsn3ky63mvul0/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/qwen3_5_0_8b/xchraazlhvqt6y/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/qwen3_vl_emb_2b/adsvsynn0j2pct/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/sentence_transformers/qwen3_vl_emb_2b/zz21apdq19z46i/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/tevatron/gemma4_e2b/dh4m30ex41frn1/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/tevatron/gemma4_e2b/j0i8cqmakhz6bk/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/tevatron/qwen3_5_0_8b/n3r8t21c0clle3/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/tevatron/qwen3_5_0_8b/wwo36vlg1onptz/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/tevatron/qwen3_vl_emb_2b/6kg59vfo5dbfpe/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/tevatron/qwen3_vl_emb_2b/poo9rswlunjkj5/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/unsloth/gemma4_e2b/32plsncjtkxzvv/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/unsloth/gemma4_e2b/xlz4cbame1awm4/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/unsloth/qwen3_5_0_8b/hjqfwu95965l8p/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/unsloth/qwen3_5_0_8b/yh32nxs19wlqv2/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/unsloth/qwen3_vl_emb_2b/n75371z1ll7tz0/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
- 판독 불가: results/unsloth/qwen3_vl_emb_2b/vfamxhetm6flkp/result.json: no `recorded_at`; the file's own clock is the download time in a clean clone, not the campaign this artifact belongs to
