# lane-g — 어댑터 + 지문

## Scope

가장 크고 설계 결정을 품는 레인이다. `scripts/bench.py`가 native 경로를 하드코딩하고 있어
**native가 아닌 어떤 프레임워크도 수치를 낼 수 없다.**

`bench.py:936`의 `framework="native"` 리터럴은 증상이다. 원인은 그 위 세 줄:

```
:912  from transformers import AutoModel, AutoProcessor
:916  AutoProcessor.from_pretrained(...)
:920  AutoModel.from_pretrained(...)
```

`framework=unsloth`로 돌리면 native 모델을 짓고 `_capture_framework`가 `"native"`를 읽어
mismatch → `assert_matches` 거부. fail-closed이긴 하지만 Phase 3이 통째로 막혀 있다.

여섯 probe 모듈 전부 진입점이 `run(config, device, report) -> None` 하나뿐이고 적재는 그 안의
`_load` 클로저다. **재사용 가능한 `load()`를 내주는 파일이 없다.**

## Owns

- `trainbench/loader.py` (lane-d가 자리만 만든 것)
- `trainbench/probe/native.py`
- `trainbench/probe/unsloth.py`
- `trainbench/probe/ms_swift.py`
- `trainbench/probe/axolotl.py`
- `trainbench/probe/registry.py`

## 할 일

### 1. 어댑터 레지스트리

프레임워크별로 적재 결과를 돌려주는 함수를 분리한다 (~150~250줄). `bench.build_run`이 그
레지스트리를 통해 적재하고 **실제 프레임워크 이름**을 `assemble`에 넘긴다.

### 2. axolotl autocast (결정 1)

axolotl은 `embed_tokens`/`lm_head`만 fp32로 두고 나머지를 bf16으로 적재한다
(`loaders/model.py:433-436`, 복귀 분기 `:456-475`는 `adapter`/FSDP/`cut_cross_entropy` 셋 다
없으면 안 돈다 — probe는 셋 다 아니다). 상류는 HF Trainer의 autocast 안에서 돌기 때문에 문제가
아니다.

**결정: autocast로 감싸 axolotl을 그대로 잰다.** `axes.step_context`가 정밀도 컨텍스트의 유일한
자리라는 `CONTRACTS §2` 계약을 유지하려면 **프레임워크가 요구하는 컨텍스트를 그 자리로 끌어와야
한다.** 계약 변경은 lane-i가 문서에 반영하고, 이 레인은 어댑터가 그 요구를 표현하는 방법을 만든다.

native(순수 bf16)와 axolotl(autocast)이 다른 수치 체제로 비교된다는 사실이 결과에 실려야 한다.

### 3. 빌드된 모델 지문 (축 G)

**프레임워크가 *요청하지 않은* 것을 무엇으로 바꿨는지**가 지금 아무 데도 안 남는다.
`applied.py`는 요청한 축의 read-back이고, 이건 그 여집합이다.

어댑터가 반환해야 하는 것:
- 모듈 클래스명
- param별 dtype
- trainable param 이름 집합
- 실제 바인딩된 attention fn identity, mask fn 등록 여부

여섯 프레임워크에서 뽑아 diff하면 차이가 전부 confound로 드러난다. 오늘 난 것들이 정확히 이
지문으로 잡혔을 것이다 — unsloth가 전 파라미터를 얼린 것, axolotl이 두 모듈만 fp32로 둔 것,
unsloth gemma-4가 텐서를 60개 더 만든 것.

### 4. tevatron의 다른 시그니처 (결정 5)

`DenseModel.forward`(`encoder.py:52-87`)가 인코딩·풀링·정규화·스코어링·InfoNCE·분산 게더를
**전부 자기가 한다.** 우리 하네스는 그것을 `steps.encode` + `embedding.info_nce` + `axes._loss` +
`parallel.cross_device_negatives`로 나눠 갖고 있다.

**결정: 프레임워크의 학습 스텝을 그대로 잰다.** 그러므로 tevatron 셀에서 `loss`와
`parallel.cross_device_negatives`는 우리 것이 아니고, **프레임워크 소유**로 기록되어야 한다
(상태 자체는 lane-c가 만든다).

`steps.encode`는 `model(**batch)`로 `last_hidden_state`를 기대한다(`steps.py:190-195`) —
tevatron 경로는 그대로 안 통한다. **어댑터별 encode가 필요하고, 이것이 이 레인의 설계 결정이다.**

### 5. 권장 경로 대조

지금 여섯 중 다섯이 "적재만 그쪽, 학습 루프는 우리 것"이다:

```
native                 AutoModel.from_pretrained        레퍼런스, 일치
unsloth                FastVisionModel.from_pretrained  for_training() 미사용
ms_swift               get_model_processor              자체 트레이너 미사용
sentence_transformers  SentenceTransformer(...)         자체 손실·트레이너 미사용
tevatron               dense.load(...)                  forward가 전체 스텝
axolotl                ModelLoader(cfg, tok).load()     자체 Trainer 미사용
```

오늘 난 세 건 — unsloth `full_finetuning` 누락으로 전 파라미터 동결, axolotl `validate_config`
건너뜀, tevatron forward 오용 — 이 **전부 이 격차에서 나왔고 셋 다 답이 핀된 소스 안에 있었다.**
필드에도 같은 사례가 있다: 어떤 재현 연구가 unsloth의 46,000 tok/s에서 grad norm 0을 관측했다.

**각 프레임워크의 공식 문서·예제가 지목하는 학습 진입점을 핀된 소스에서 찾아 인용하고,
우리가 쓰는 것과 다르면 그 차이를 어댑터에 기록한다.**

## Completion criteria

- 여섯 프레임워크가 공통 진입점으로 적재되고 `bench.py`가 실제 프레임워크 이름을 넘긴다
  → `uv run pytest tests/contract/test_loader_bench.py`
- 어댑터가 빌드된 모델 지문을 반환한다
  → `uv run pytest tests/test_loader.py -k fingerprint`
- 지문이 오늘 난 세 건을 잡는다 — 전 파라미터 동결, 두 모듈만 fp32, 텐서 수 차이
  → `uv run pytest tests/test_loader.py -k fingerprint_catches`
- tevatron 셀에서 `loss`/`cross_device_negatives`가 **프레임워크 소유**로 기록된다
  → `uv run pytest tests/test_loader.py -k tevatron_owns`
- axolotl 경로가 autocast 컨텍스트를 요구하고 그것이 `step_context`를 통해 걸린다
  → `uv run pytest tests/test_loader.py -k axolotl_autocast`
- 여섯 프레임워크의 문서화된 학습 진입점이 인용되고, 우리 경로와의 차이가 기록된다
  → `uv run pytest tests/test_loader.py -k documented_entry_point`
- 위 각 검사를 되돌리면 죽는다
  → 변이 출력 그대로 인용
- **확인 안 함**: `Collate`의 `processor(text=..., images=...)` 규약이 sentence_transformers
  경로에서 성립하는지. sentence_transformers도 자기 손실을 갖는지. 둘 다 이 레인이 확인한다

## Out of scope

- `trainbench/probe/tevatron.py` — **lane-a** 소유 (적재 shim). forward 시그니처만 이 레인
- `trainbench/probe/sentence_transformers.py` — **lane-b** 소유 (동결 가드). 어댑터 배선은
  경계 `loader-bench`에서 맞춘다
- `trainbench/probe/steps.py` — **lane-d** 소유
- `scripts/bench.py` — **lane-d** 소유
- 축 소유권 상태의 정의 — **lane-c** 소유. 이 레인은 그것을 쓰기만 한다
- `docs/CONTRACTS.md` §2 개정 — **lane-i** 소유
