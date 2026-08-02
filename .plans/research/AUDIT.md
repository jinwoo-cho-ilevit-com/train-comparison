# 리서치 감사 기록 (2026-08-02)

브리프 7개를 각각 **다른 에이전트가** 파일을 열어 인용을 대조했다. 읽고 판단한 것이 아니라
`resolvedPath` 하위의 파일을 열어 줄범위를 확인했다.

## 판정

| 브리프 | 판정 | 확인한 인용 | 문제 |
|---|---|---|---|
| `unsloth.md` | sourced | 21 | 0 |
| `transformers-varlen-prompt.md` | sourced | 28 | 0 |
| `axis-libraries.md` | partly-sourced | 23 | 1 |
| `tevatron.md` | partly-sourced | 21 | 2 |
| `ms-swift.md` | partly-sourced | 21 | 2 |
| `axolotl.md` | partly-sourced | 35 | 4 |
| `sentence-transformers.md` | partly-sourced | 31 | 6 |

합계 인용 180건, 문제 16건.

## 문제의 성격 — 하나를 빼고 전부 색인 오류다

`partly-sourced` 는 **주장이 틀렸다는 뜻이 아니다.** 열여섯 중 열다섯은 구조화 출력의
`findingsIndex` 가 매단 줄범위가 브리프 본문이 인용한 줄과 어긋난 것이고, 감사가 매번
같은 문장으로 확인했다 — *"주장 자체는 참, 색인의 줄범위만 어긋난다"*,
*"브리프 본문은 이 줄들을 옳게 인용한다"*.

**레인 에이전트는 브리프 본문을 읽는다. 색인은 이 감사의 산물이고 커밋되지 않는다.**
그러므로 열다섯은 레인에 영향이 없다.

대표 예: `ms-swift.md` 의 `_compat_transformers5` 인용은 색인이 `register.py:68-72`
(실제로는 unsloth 적재 블록)를 가리켰고 원문은 같은 파일 388-391 에 글자 그대로 있었다.

## 본문을 고친 것 — 둘

### 1. `axolotl.md` — gemma-4 축 덮어쓰기가 무조건이 아니다 (실질 오류)

브리프가 `model_config_type ∈ {gemma4, gemma4_unified}` 이면
`gradient_checkpointing_kwargs["use_reentrant"]=False` 와 `ddp_find_unused_parameters=True` 를
**강제한다**고 적었다. 원문(`axolotl/utils/config/__init__.py:385-415`)을 직접 열어 확인한 결과
둘 다 조건부다:

- `use_reentrant=False` — `cfg.gradient_checkpointing` 이 참일 때만 (391행)
- `ddp_find_unused_parameters=True` — `cfg.ddp` 이고 그 값이 `None` 이고
  `activation_offloading` 이 `True` 가 아닐 때만 (400-415행).
  `activation_offloading=True` 면 DDP 쪽은 통째로 건너뛴다

넓게 읽으면 axes/adapters 레인이 **존재하지 않는 덮어쓰기를 전제**하게 된다. 본문을 조건과
함께 다시 썼다.

### 2. `sentence-transformers.md` — 손실 등가의 조건을 명시

"기본 설정에서 두 손실은 수치적으로 같다"에 조건 셋을 붙였다: `temperature = 1/scale`,
`partition_mode == "joint"`(기본값 — `per_direction` 이면 분모가 방향별로 쪼개져 단일 softmax
가 아니다), hard negative 없음. 손실식의 실제 줄(`multiple_negatives_ranking.py:338`)도 적었다.

## 핀 해석은 지켜졌다

받아온 핀 **15개 전부** `sha256` 이 lock 의 `hash =` 와 일치했다:
ms-swift 4.4.2, sentence-transformers 5.6.1, flash-linear-attention 0.4.1/0.5.2,
fla-core 0.4.1/0.5.2, liger-kernel 0.8.0/0.8.1, kernels 0.16.0, bitsandbytes 0.50.0,
deepspeed 0.19.3, transformer-engine 2.17.0, transformer-engine-torch 2.17.0,
causal-conv1d 1.6.2.post1, nvidia-dali-cuda130 2.2.0.

디코이도 이름으로 거부됐다 — transformers 10벌, huggingface-hub 15벌, torch 8벌,
unsloth 1벌(2026.6.9), sentence-transformers 4벌, tevatron 3벌 등.

`memoryDivergence`(파일을 열기 전 예상 vs 실제)는 브리프당 8~10건 기록됐다. 빈 것은 없다.

## 열지 못한 핀 — "확인 안 함"으로 남은 것

- **torch** (`2.11.0+cu130`, `2.13.0+cu130`) — 캐시에는 이름만 같은 macOS/PyPI 빌드뿐이다.
  torch 소스에 기댄 주장은 브리프에 하나도 넣지 않았다
- **peft 0.20.0** — tevatron 의 LoRA 경로에서 `hf_kwargs` 가 `LoraConfig.from_pretrained` 로
  그대로 넘어가는데 그 함수가 어떤 키를 거부하는지 안 열었다. 그 경로의 주장은 전부 확인 안 함
- **decord** — ms-swift 의 `ModelMeta.requires` 에 이름이 있으나 `envs/ms-swift/uv.lock` 에
  항목 자체가 없다. 확인할 대상이 없다
- **tevatron 설치본 메타데이터** — `framework_version` 을 `importlib.metadata` 로 바꿨을 때
  `direct_url.json` 에 커밋 sha 가 남는지는 리눅스 이미지에서만 답할 수 있다
