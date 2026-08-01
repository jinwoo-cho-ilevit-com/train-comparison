# 모델별 규격 검증

현재 probe는 세 모델에 동일한 generic 경로를 쓴다(`AutoModel` + 자체 `last_token_pool`
+ 자체 `info_nce`). 모델이 의도한 사용법과 다르면 측정 대상이 "모델"이 아니라
"잘못 쓴 모델"이 된다. 이 문서는 그 차이를 확정한다.

**모든 항목은 HF 저장소 파일을 직접 읽거나 코드를 실행해 확인했다. 확인하지 못한
것은 "미확인"으로 둔다** (컨벤션 16).

조회 시점: 2026-08-01. transformers 5.14.1.

---

## Qwen/Qwen3-VL-Embedding-2B

**완전한 sentence-transformers 구조를 갖춘 공식 임베딩 모델이다.**
`modules.json`, `1_Pooling/`, `config_sentence_transformers.json`,
`sentence_bert_config.json`, `chat_template.jinja` 보유.

| 항목 | 공식 규격 | 근거 | 현재 구현 | 차이 |
|---|---|---|---|---|
| 모듈 구성 | Transformer -> Pooling -> Normalize | `modules.json` | AutoModel + 자체 pooling | 동등 |
| pooling | `"pooling_mode": "lasttoken"` | `1_Pooling/config.json` | last-token | **일치** |
| 임베딩 차원 | 2048 | `1_Pooling/config.json` | 2048 실측 | 일치 |
| 유사도 | cosine | `config_sentence_transformers.json` | cosine | 일치 |
| **instruction prompt** | `"Represent the user's input."` 를 기본 부착 (`default_prompt_name: "default"`) | `config_sentence_transformers.json` | **부착 안 함** | **차이** |
| `include_prompt` | `true` (prompt 토큰도 pooling 대상) | `1_Pooling/config.json` | 해당 없음 | **차이** |
| **`add_generation_prompt`** | `true` | `sentence_bert_config.json`의 `processing_kwargs.chat_template` | **`False`로 호출** | **차이** |
| 입력 형식 | `message` 모달리티가 `format: "structured"` — interleaved 이미지·텍스트를 구조화 메시지로 | `sentence_bert_config.json` | 평문 text + images | **차이** |
| 이미지 해상도 | **동적**. `min_pixels 4096`, `max_pixels 1310720`, `patch_size 16`, `merge_size 2` | `preprocessor_config.json` | 448 정사각 고정 이미지로 196 토큰 측정 | **차이** |
| MRL | 미확인 (README 미독) | - | - | 미확인 |
| `padding_side` | 미확인 | - | - | 미확인 |

### 차이의 영향

1. **instruction prompt 미부착** — 이 모델은 해당 prompt가 붙은 상태로 학습됐다.
   빼면 학습 시와 다른 입력 분포로 forward를 돌리는 것이고, 품질 가드레일이 모든
   설정에서 발화하면서 원인이 축으로 오귀속된다.
2. **`add_generation_prompt` 반대** — last-token pooling에서 이 플래그는 **마지막
   토큰이 무엇인지를 바꾼다.** pooling 방식은 맞게 골라놓고 집는 위치가 틀린 셈이라
   코드만 봐서는 드러나지 않는다.
3. **동적 해상도** — 448 정사각 1장에서 얻은 196 토큰은 그 이미지에서만 성립한다.
   실제 MMEB 이미지는 해상도가 흩어져 있으므로 모델 간 토큰 예산 고정의 근거가 될
   수 없다.

---

## google/gemma-4-E2B

단일 safetensors(10.2GB)이며 `model.safetensors.index.json`이 없다. 파라미터 이름은
config에서 meta device로 모델 골격을 구성해 열거했다(가중치 다운로드 없음).

| 항목 | 실측 | 근거 |
|---|---|---|
| **`padding_side`** | **`"left"`** | `tokenizer_config.json` |
| image token | `<|image|>` | `tokenizer_config.json` |
| processor | `Gemma4Processor` | `tokenizer_config.json` |
| 전체 파라미터 | 5.104B | meta device 실측 |
| **PLE (`per_layer`)** | **2.390B (46.8%)** | meta device 실측 |
| `embed_tokens_per_layer` 단일 텐서 | 2.349B | meta device 실측 |
| 전체 embedding | 2.751B (53.9%) | meta device 실측 |
| **non-embedding** | **2.353B** | meta device 실측 — "E2B = effective 2.3B"의 실체 |

### PLE 파라미터 실제 이름 (108개)

```
language_model.embed_tokens_per_layer.weight        <- 2.349B, 단일 최대 텐서
language_model.layers.N.per_layer_input_gate.weight
language_model.layers.N.per_layer_projection.weight
language_model.layers.N.post_per_layer_input_norm.weight
language_model.per_layer_model_projection.weight
language_model.per_layer_projection_norm.weight
```

**현재 heuristic의 절반이 죽어 있다.** `native.py:_ple_report`는
`"per_layer" in name or "altup" in name`으로 찾는데:

- `per_layer` -> **108개 매칭, 유효**
- `altup` -> **0개 매칭.** gemma-3n의 이름이며 gemma-4에는 없다
- `laurel` -> 0개 (gemma-3n 이름)

결과적으로 동작은 하지만 검증된 적이 없었고, 이름이 달랐다면
`matched_count: 0`인데 `ok: True`로 통과했을 것이다.

### 이전 주장 정정

`PLAN.md`·`docs/support-matrix.md`·커밋 메시지에서 반복한 **"5.1B 중 약 2.8B가 PLE
embedding 테이블"은 부정확하다.** 2.751B는 주 `embed_tokens`까지 포함한 전체
embedding이고, PLE만 따지면 **2.390B**다. 결론(옵티마이저 메모리가 이 모델의 지배
변수)은 유지되나 수치를 정정해야 한다.

| 미확인 | 비고 |
|---|---|
| 공식 pooling / prompt 포맷 | 생성형 모델이라 임베딩 규격이 없음. 우리가 정해야 하며 그 선택이 리포트의 한정 조건이 된다 |
| placeholder 확장 위치 (processor vs 모델 내부) | `processor_config.json` 미독 |
| `vision_soft_tokens_per_image: 280`의 실제 관측값 | probe 미실행 |
| LoRA target module 관례 | 미확인 |

---

### 이미지 처리 (`processor_config.json`)

| 항목 | 값 |
|---|---|
| `image_seq_length` / `max_soft_tokens` | **280 (고정)** |
| `patch_size` / `pooling_kernel_size` | 16 / 3 |
| 정규화 | `do_normalize: false`, mean 0 / std 1 |
| 비디오 soft token | 70 (32 프레임) |

**해상도와 무관하게 이미지당 280 토큰으로 고정된다.** 확장은 프로세서 단계에서
일어난다(`image_seq_length`는 프로세서 개념). Qwen 계열의 픽셀 비례 방식과 근본적으로
다르다.

---

## Qwen/Qwen3.5-0.8B

**생성형 VLM이며 sentence-transformers 구조가 없다.** 공식 임베딩 규격이 존재하지
않으므로 pooling·prompt를 우리가 정해야 한다.

| 항목 | 값 | 근거 |
|---|---|---|
| `padding_side` | **키 없음 -> transformers 기본값 `right`** | `tokenizer_config.json` |
| `pad_token` / `eos_token` | `<|endoftext|>` / `<|im_end|>` | `tokenizer_config.json` |
| tokenizer | `Qwen2Tokenizer` | `tokenizer_config.json` |
| image token | `<|image_pad|>` (+ `<|vision_start|>`/`<|vision_end|>`) | `extra_special_tokens` |
| 이미지 해상도 | **동적**. `shortest_edge 65536`, `longest_edge 16777216` | `preprocessor_config.json` |
| `patch_size` / `merge_size` | 16 / 2 | `preprocessor_config.json` |
| `model_max_length` | 262144 | `tokenizer_config.json` |

### `add_generation_prompt`의 함정 — 생성형 모델에서는 정반대로 작동한다

`chat_template.jinja`에서 `add_generation_prompt`가 참이면 다음이 덧붙는다:

```
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

**last-token pooling이면 이 thinking 스캐폴딩의 마지막 토큰이 임베딩이 된다.**
Qwen3-VL-Embedding은 공식 임베딩 모델이라 `true`가 규격이지만, 생성형 모델에
동일하게 적용하면 의미 없는 위치를 pooling하게 된다.

---

## 확정된 결정 (2026-08-01, 사용자)

### 관통 원칙 — 각 모델을 그 모델이 의도한 방식으로 쓴다

입력 조건을 인위적으로 동일하게 맞추는 쪽(비교 공정성)이 아니라, 공식 규격이
존재하면 그것을 따르고 없으면 최소 개입으로 간다. 아래 결정들은 전부 이 원칙의
적용이다.

**대가**: 모델 간 입력 조건이 달라지므로 **모델 간 절대 수치 비교의 근거가 약해진다.**
이는 결정 3(이미지 토큰 예산 고정 포기)에서 이미 원리적으로 불가능하다고 확인된
방향과 같다. 1차 결론을 "모델 내 축 효과"로 두는 리포트 구조와 정합한다.

### 1. instruction prompt — 모델별로 공식 규격을 따른다

| 모델 | prompt | 근거 |
|---|---|---|
| Qwen3-VL-Embedding-2B | **`"Represent the user's input."` 부착** | `config_sentence_transformers.json`의 `default_prompt_name: "default"` |
| Qwen3.5-0.8B | **무부착** | 임베딩 공식 규격 없음 |
| gemma-4-E2B | **무부착** | 임베딩 공식 규격 없음 |

생성형 두 모델에 임의의 prompt를 발명해 붙이지 않는다. 검증되지 않은 선택이 교란
변수가 되기 때문이다.

속도 측정 영향은 미미하다(~200 토큰 시퀀스에 prompt 토큰 몇 개). 품질 측면에서는
Qwen3-VL-Embedding이 학습된 대로 쓰이므로 오히려 정확해진다.

### 2. `add_generation_prompt` — 모델별로 공식/타당한 값에 맞춘다

일괄 `true`가 아니다. 위에서 확인했듯 같은 플래그가 모델 성격에 따라 반대로 작동한다.

| 모델 | 값 | 근거 |
|---|---|---|
| Qwen3-VL-Embedding-2B | **`true`** | `sentence_bert_config.json`이 명시한 공식 규격 |
| Qwen3.5-0.8B | **`false`** | 생성형. `true`면 `<think>` 스캐폴딩 끝을 pooling |
| gemma-4-E2B | **`false`** | 생성형. 임베딩 규격 없음 |

현재 코드는 전부 `False`이므로 **Qwen3-VL-Embedding만 `true`로 바꾸면 된다.**
(Wave 1 코어 정확성 레인 작업 항목)

### 3. 이미지 토큰 예산 — 모델 간 고정은 포기한다

| 모델 | 방식 |
|---|---|
| Qwen3-VL-Embedding-2B | 픽셀 비례. `min_pixels 4096` ~ `max_pixels 1310720` |
| Qwen3.5-0.8B | 픽셀 비례. `shortest_edge 65536` ~ `longest_edge 16777216` (범위가 다름) |
| gemma-4-E2B | **해상도 무관 280 고정** |

**세 모델의 토큰 예산을 동시에 고정하는 것은 원리적으로 불가능하다.** 고정 가능한
것은 입력 픽셀 수뿐이고, 그래도 gemma-4는 항상 280이며 Qwen 두 모델은 서로 다른
픽셀 범위를 갖는다.

따라서:
- 입력 이미지 픽셀 분포를 고정하고, **모델별 실제 visual token 분포(p50/p95/max)를
  실측해 기록**한다
- **모델 간 절대 throughput 비교는 리포트에서 한정한다.** 1차 결론은 "모델 내 축
  효과"이고, 모델 간 비교는 토큰 분포를 함께 제시할 때만 언급한다

이는 `PLAN.md`의 "모델 간 Pareto frontier"라는 헤드라인을 좁히는 결정이다.

### 4. 구조화 메시지 입력 — 보류 (원칙과 충돌 중)

Qwen3-VL-Embedding만 `message` 모달리티(`format: "structured"`)를 공식 입력 형식으로
갖는다(`sentence_bert_config.json`).

당초 "비교 공정성이 떨어진다"를 근거로 미채택했으나, **결정 1이 뒤집히면서 그 근거가
사라졌다.** 관통 원칙("각 모델을 의도한 방식으로")을 그대로 적용하면 채택하는 것이
일관된다.

채택하지 않을 유일한 남은 근거는 **구현 비용**이다. 모델별로 입력 구성 코드 경로가
갈라지고, `steps.py`의 공통 배치 생성 함수를 모델별 어댑터로 나눠야 한다.

| 선택지 | 결과 |
|---|---|
| 채택 | 원칙 일관. Qwen3-VL-Embedding을 완전히 공식대로 사용. 코드 경로 분기 발생 |
| 미채택 | 구현 단순. **원칙에 대한 예외를 하나 남기게 되므로 리포트에 명시 필요** |

**미결.** Wave 1 코어 정확성 레인 착수 전에 정한다.

---

## 남은 미확인

| 항목 | 비고 |
|---|---|
| Qwen3-VL-Embedding-2B `padding_side` | `tokenizer_config.json` 미독 |
| MRL(Matryoshka) 지원 차원 | 세 모델 모두 README 미독. 지원하면 임베딩 차원이 축이 될 수 있다 |
| 모델별 LoRA target module 관례 | 현재 `all-linear`는 "모델별 target module 인식" 질문을 회피한다 |
| gemma-4 placeholder 확장의 실제 관측값 | probe 미실행. `image_seq_length: 280`이 실제로 280개 토큰으로 나오는지 |
