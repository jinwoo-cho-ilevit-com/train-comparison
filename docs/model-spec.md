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

## Qwen/Qwen3.5-0.8B

미확인. 생성형 모델이라 임베딩 규격이 없을 것으로 예상되나 확인하지 않았다.
`padding_side`, chat template, LoRA target 관례 전부 미조회.

---

## 결정이 필요한 사항

generic 경로를 유지할지(단순성·모델 간 비교 공정성) 모델별 공식 규격에 맞출지
(현실성)를 항목별로 정해야 한다. **이 결정 자체가 리포트의 한정 조건이 된다.**

| 항목 | 선택지 |
|---|---|
| instruction prompt | (a) Qwen3-VL만 공식 prompt 부착 -> 모델 간 입력이 달라짐 (b) 전 모델 무부착 -> Qwen3-VL을 의도와 다르게 사용 (c) 전 모델에 동일한 자체 prompt 부착 |
| `add_generation_prompt` | 공식이 `true`이고 last-token pooling에 직접 영향하므로 **맞추는 쪽을 권장** |
| 구조화 메시지 입력 | Qwen3-VL만 지원. 맞추면 코드 경로가 갈라짐 |
| 이미지 토큰 예산 | 동적 해상도(Qwen) vs 고정 280(gemma-4)이라 **모델 간 고정이 원리적으로 불가능.** 리포트 범위를 "모델 내 축 효과"로 좁히는 선택지 포함 |
