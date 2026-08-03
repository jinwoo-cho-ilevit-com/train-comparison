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
| `padding_side` | **키 없음 -> transformers 기본값 `right`** | `tokenizer_config.json` (5404B 전문 확인) | - | 일치 |

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

**캠페인 제외 (2026-08-03).** full finetuning이 A100 80GB 한 장에 들어가지 않는다
(`train.batch_size=16`에서 peak 83.8GB, 배치 4로 낮춘 뒤에도 OOM — `PLAN.md`
"gemma-4-E2B 제외" 참조). `configs/model/gemma4_e2b.yaml`과 이 문서가 대조하던
`docs/model-spec.yaml`의 항목은 제거됐다. 아래는 제외 이전에 실측한 결과이므로
지우지 않고 그대로 남긴다 — 실측은 역사다.

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

**당시 heuristic의 절반이 죽어 있었다.** gemma-4 제외와 함께 제거된
`native.py:_ple_report`는 `"per_layer" in name or "altup" in name`으로 찾았는데:

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
| LoRA target module 관례 | 미확인 |

placeholder 확장 위치와 이미지당 토큰 수는 2026-08-02에 실측으로 해소했다 —
아래 "이미지 처리" 참조.

---

### 이미지 처리 (`processor_config.json`)

| 항목 | 값 |
|---|---|
| `max_soft_tokens` | **280 (상한)** |
| `image_seq_length` | 280 (같은 값이지만 프로세서가 읽지 않는다 — 아래) |
| `patch_size` / `pooling_kernel_size` | 16 / 3 |
| 정규화 | `do_normalize: false`, mean 0 / std 1 |
| 비디오 soft token | 70 (32 프레임) |

**280은 고정값이 아니라 상한이다** (2026-08-02 정정, transformers 5.14.1 실측).
확장은 프로세서 단계에서 일어나지만, `Gemma4Processor.replace_image_token`이 읽는
것은 `image_seq_length`가 아니라 이미지 프로세서가 이미지마다 계산한
`num_soft_tokens_per_image`다. `self.image_seq_length`는 생성자에서 대입될 뿐
어디서도 읽히지 않는다 — 이 문서와 `docs/model-spec.yaml`이 그 키를 이미지당
토큰 수로 읽은 것이 오독의 출처다.

실제 계산은 `get_aspect_ratio_preserving_size`다. 종횡비를 보존한 채
`max_patches = max_soft_tokens * pooling_kernel_size**2 = 2520` 패치 안에 들어가는
최대 크기로 리사이즈하고, 양변을 `patch_size * pooling_kernel_size = 48`의 배수로
내림한다. 토큰 수는 `(높이/48) * (너비/48)`이므로 **종횡비에 따라 달라지고**, 280에
도달하려면 그 곱이 정확히 280으로 분해되어야 한다(280 = 2^3·5·7이라 정사각형은
16x16 = 256이 최대다).

| 입력 (WxH) | soft token |
|---|---|
| 448x448 (`PROBE_IMAGE_SIZE`) | 256 |
| 64x64 / 1024x1024 / 16x16 | 256 |
| 768x256 | 252 |
| 1280x720 | 264 |
| 960x672 / 16x1120 | **280** |

16px 격자로 4096px까지 쓸어본 결과 도달 가능한 값은 138종이고 최댓값이 280이다.
Qwen 계열의 픽셀 비례 방식과 다른 것은 맞지만, **"해상도와 무관하게 고정"은 아니다.**
그래서 (제외 이전) `configs/model/gemma4_e2b.yaml`의 필드 이름은 `max_tokens_per_image`
였고, probe는 이 값과의 일치가 아니라 초과를 거부했다. gemma-4 제외와 함께 이
필드는 스키마·config·probe 시그니처에서 전부 제거됐다.

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

결정 1·2와 여기서 확인한 `padding_side`는 `docs/model-spec.yaml`에 기계 판독
가능한 형태로도 적혀 있고, `scripts/audit_plan.py`의 `model-spec` 체크가
`configs/model/*.yaml`과 **값 대 값으로** 대조한다. 이 문서와 config가 어긋나면
게이트가 막는다. (`max_tokens_per_image`는 gemma-4-E2B 전용 필드였다 — gemma-4
제외와 함께 스키마·config·이 대조표에서 모두 제거됐다. 아래 "이미지 처리" 절의
서술은 역사로 남긴다.)

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
| gemma-4-E2B | 종횡비 비례, `max_soft_tokens 280` 상한 |

**세 모델의 토큰 예산을 동시에 고정하는 것은 원리적으로 불가능하다.** 고정 가능한
것은 입력 픽셀 수뿐이고, gemma-4는 그 픽셀을 종횡비에 따라 252~280으로 접으며
Qwen 두 모델은 서로 다른 픽셀 범위를 갖는다.

(2026-08-02 정정: 이 표는 gemma-4를 "해상도 무관 280 고정"으로 적고 있었다.
결론 — 동시 고정 불가 — 은 바뀌지 않지만 근거가 틀렸었다. 위 "이미지 처리" 참조.)

따라서:
- 입력 이미지 픽셀 분포를 고정하고, **모델별 실제 visual token 분포(p50/p95/max)를
  실측해 기록**한다
- **모델 간 절대 throughput 비교는 리포트에서 한정한다.** 1차 결론은 "모델 내 축
  효과"이고, 모델 간 비교는 토큰 분포를 함께 제시할 때만 언급한다

이는 `PLAN.md`의 "모델 간 Pareto frontier"라는 헤드라인을 좁히는 결정이다.

### 4. 구조화 메시지 입력 — 채택. 전 모델의 텍스트 입력도 chat template을 탄다

Qwen3-VL-Embedding은 `message` 모달리티(`format: "structured"`)를 공식 입력 형식으로
갖는다(`sentence_bert_config.json`).

당초 미채택했던 근거 세 가지가 모두 무너졌다.

| 당초 근거 | 검토 결과 |
|---|---|
| 비교 공정성이 떨어진다 | 결정 1이 뒤집히며 **소멸**. 관통 원칙이 오히려 채택을 요구한다 |
| 코드 경로가 모델별로 갈라진다 | **사실이 아니다.** `steps.image_batch`는 이미 `apply_chat_template`에 content 블록을 넘기는 구조화 형식이다. 갈라지는 게 아니라 `text_batch`를 같은 방식으로 맞추면 두 함수가 **일관돼진다** |
| Qwen3-VL-Embedding만 지원한다 | 근거가 아니다. 공식 규격이 한 모델에만 있다는 것은 그 모델을 규격대로 쓰지 말 이유가 되지 않는다 |

추가로, **결정 2(`add_generation_prompt` 모델별)를 구현하려면 어차피 `text_batch`가
chat template을 타야 한다.** 평문 `processor(text=...)`에는 그 플래그를 줄 자리가
없다. 즉 이 채택은 별도 비용이 아니라 이미 확정된 작업에 흡수된다.

**부수 효과와 그 성격**: 생성형 두 모델의 텍스트 입력에도 chat template이 적용되어
`<|im_start|>user` 같은 토큰이 시퀀스에 붙는다. 임베딩 용도로 생성형 모델에 chat
template을 적용하는 것은 **널리 쓰이는 관행이지 공식 규격이 아니다.** 이 구분을
리포트에 유지한다 — Qwen3-VL-Embedding은 공식대로, 생성형 둘은 관행대로 사용한 것이다.

**작업 항목** (Wave 1 코어 정확성 레인):
- `steps.text_batch`를 `apply_chat_template` 기반으로 전환
- `add_generation_prompt`를 모델별 값으로 주입(결정 2)
- Qwen3-VL-Embedding에 instruction prompt 부착(결정 1)

(2026-08-02 정정: 이 결정은 **세 모델 모두 chat template을 갖고 있다**는 전제 위에
있었고, 그 전제가 틀렸다. gemma-4-E2B에는 chat template이 없다 — 결정 5. "전 모델"은
Qwen 두 모델로 좁혀진다. 남은 작업 항목의 `text_batch` 전환도 마찬가지로 gemma-4에는
적용되지 않는다.)

### 5. chat template 부재 — gemma-4는 raw 형식으로 간다

2026-08-02 실측(transformers 5.14.1, 실제 Hub 저장소):

| 모델 | `chat_template.jinja` | 프로세서의 `chat_template` |
|---|---|---|
| Qwen3-VL-Embedding-2B | 있음 | 있음 |
| Qwen3.5-0.8B | 있음 | 있음 |
| **gemma-4-E2B** | **없음** | **`None`** |

`google/gemma-4-E2B`는 사전학습 체크포인트이고, 사전학습 체크포인트에는 대화 형식이
없다 — 모델 카드의 `apply_chat_template` 예제는 전부 `google/gemma-4-E2B-it`를
적재한다(`-it`에는 `chat_template.jinja`가 있다). 즉 이것은 프레임워크 문제가 아니라
**체크포인트의 성질**이며, PLAN.md가 고정한 것은 사전학습 쪽이다.

따라서 프롬프트 형식을 모델별 config 값 `model.prompt_format`으로 선언한다.
`padding_side`와 같은 종류의 값이다 — 체크포인트에 대한 사실이고, 코드가
`model.arch`로 분기하면 결과에서 보이지 않게 된다.

| 값 | 의미 | 모델 |
|---|---|---|
| `chat_template` | 프로세서의 chat template을 탄다 | Qwen 2종 |
| `raw` | 이미지 placeholder + 텍스트, 역할/턴 마커 없음 | gemma-4-E2B |

**이것은 교란 변수이고 그렇게 읽혀야 한다.** 두 형식은 같은 프롬프트가 아니다.
`chat_template` 행은 역할·턴 마커에 감싸이고 `raw` 행은 감싸이지 않으므로, 시퀀스
길이는 두 형식 사이에서 직접 비교되지 않는다. 결정 3이 이미 좁혀 둔 "모델 간 절대
throughput 비교는 한정한다"와 같은 성격의 제약이 하나 더 붙는 것이다. 측정된
`visual_tokens` 결과에는 어느 형식으로 잰 값인지가 함께 기록된다.

`raw`가 이미지를 실을 수 있다는 것도 실측이다. gemma-4의 프로세서는 평문 안의
`<|image|>` 하나를 그 이미지의 soft token 수만큼 펼친다 — 448x448 probe 이미지에서
**256 토큰, 전체 시퀀스 265 토큰**(2026-08-02). 형식이 바꾸는 것은 placeholder를
둘러싼 마커이지 placeholder 자체가 아니다.

`trainbench/prompt.py`가 이 값을 읽는 유일한 자리이며, 선언과 체크포인트가
어긋나면 양방향 모두 런을 멈춘다 — 없는 template을 요구하는 쪽도, template이 있는
체크포인트를 raw로 자르는 쪽도 거부한다(`padding_side_alignment`와 같은 이유).

---

## 남은 미확인

| 항목 | 비고 |
|---|---|
| **모델별 vision tower 모듈 이름** | `freeze.vision_tower` 축이 이것 없이는 구현 불가. 추측으로 `visual`/`vision_tower`를 넣지 않는다 — 틀리면 0개를 얼리고 성공으로 기록된다(PLE에서 이미 겪은 실패). D 레인이 `model.safetensors.index.json`으로 확인한 뒤 구현 |
| MRL(Matryoshka) 지원 차원 | 세 모델 모두 README 미독. 지원하면 임베딩 차원이 축이 될 수 있다 |
| 모델별 LoRA target module 관례 | 현재 `all-linear`는 "모델별 target module 인식" 질문을 회피한다 |
| Qwen 두 모델의 실제 visual token 분포 | probe 미실행. gemma-4는 2026-08-02에 실측했다(위 "이미지 처리") |
