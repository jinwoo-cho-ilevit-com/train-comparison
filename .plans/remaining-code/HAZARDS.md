# HAZARDS — 이 저장소가 이미 겪은 것

**모든 레인 에이전트는 자기 브리프를 읽기 전에 이 파일을 읽는다.**

여기 적힌 것은 전부 이 저장소에서 실제로 일어났고 근거가 커밋·파일·측정에 남아 있다.
새 작업이 같은 모양으로 다시 실패하는 것을 막는 것이 이 문서의 유일한 목적이다.

인용은 전부 `file:line` 또는 커밋 sha로 추적된다. **여기 적힌 숫자도 네가 인용하려면
직접 재실행해야 한다** — 이 문서에서 옮겨 적는 것은 §2가 금지하는 바로 그 행위다.

---

## 1. 핀된 소스를 읽고 나서 단언한다

**1차 Phase 0 캠페인(2026-08-02, A100 18파드)이 찾은 모든 프로브 실패는 답이 이미 잠긴 휠
안에 있었고 아무도 그 파일을 열지 않았다.** (`AGENTS.md:154-173`)

| 실패 | 답이 있던 자리 |
|---|---|
| axolotl 3칸 `TypeError: unsupported operand type(s) for //: 'NoneType' and 'NoneType'` | axolotl 자신의 순서는 `prepare_plugins → validate_config → normalize_config`(`cli/config.py`). 우리 프로브는 validate 를 **아예 부르지 않고** normalize 를 먼저 불렀고, docstring 은 "프로젝트 자체 문서를 따른다"고 주장하고 있었다 |
| unsloth 3칸이 `params_with_grad=0, trainable_params=0` 으로 **통과** | `FastVisionModel.from_pretrained` 의 `full_finetuning=False` 기본값(`unsloth/models/loader.py`)이 `requires_grad_(False)` 로 끝나는 사슬. backward 가 안 죽은 것은 `enable_input_require_grads()` 가 임베딩 **출력**에 requires_grad 를 걸기 때문 |
| 3 프레임워크 × gemma-4 `Cannot use apply_chat_template` | `google/gemma-4-E2B` 는 base 체크포인트라 `chat_template.jinja` 가 없다. `-it` 변형에만 있다. **Hub 파일 목록 한 번이면 보였다** |
| tevatron 3칸 `ModuleNotFoundError` (1차) | 상류 `setup.py` 의 `install_requires` 가 `transformers`/`datasets` 둘뿐인데 `encoder.py` 가 최상단에서 peft 를 import 한다. **상류 미선언** |
| ms_swift 2칸 `PackageNotFoundError` | `Qwen3VLLoader._check_qwen_vl_utils` 가 잡히지 않는 곳에서 `require_version('qwen_vl_utils>=0.0.14')` 를 던진다 |
| unsloth Qwen 2칸 `padding_side_alignment` | 문구는 "model-spec 이 체크포인트와 안 맞는다"였으나 **스펙은 맞았다**. unsloth 가 `models/vision.py` 에서 padding_side 를 무조건 left 로 덮어쓴다. gemma-4 가 통과한 것은 그 스펙이 우연히 left 였기 때문 |
| tevatron 3칸 `AttributeError: 'XConfig' object has no attribute 'pad_token_id'` (2차) | `encoder.py:166-168` 이 `from_pretrained` 직후 `base_model.config.pad_token_id` 를 **getattr 아니라 직접** 접근한다. transformers 5.14.1 합성 config 는 그것을 최상위에 두지 않는다 |
| axolotl 3칸 `expected mat1 and mat2 to have the same dtype: float != c10::BFloat16` (2차) | axolotl 은 `embed_tokens`/`lm_head` 만 fp32 로 두고 나머지를 bf16 으로 적재한다(`loaders/model.py`). 복귀 분기는 adapter/FSDP/cut_cross_entropy 중 하나가 있어야 도는데 프로브는 셋 다 없다. 상류는 HF Trainer 의 autocast 안에서 돌아 문제가 안 됐다 |

이렇게 고친 것들은 **첫 파드 실행에 착지했다.** ms_swift 0/3 → 3/3, native 의 gemma-4 칸이 열렸다.
5/18 → 12/18.

**규칙**: 프레임워크 동작에 대한 주장은 lock 이 가리키는 파일을 열고 원문을 인용한 뒤에 한다.
"보통 이렇게 동작한다"는 핀된 버전과 만나면 살아남지 않는다.

핀 해석은 패키지명 glob 이 아니라 `dist-info` 로 한다 — 캐시에 **디코이가 있다**
(실측: unsloth 2026.6.9 와 2026.7.6 이 함께 있고 경로에는 어느 쪽인지 적혀 있지 않다).

---

## 2. 직접 내지 않은 숫자를 옮기지 않는다

`AGENTS.md:175-182`, 커밋 `13e000c`.

- 한 레인이 감사를 **`12/15`로 보고했으나 트리는 실제 `11/15`**였다 (`plan-files` 가 미등재
  파일 2건으로 red). 그 숫자가 검증 없이 상태 보고서에 들어갔다.
- 같은 모양으로, **다른 레인의 미커밋 작업이 올라와 있던 트리에서 읽은 테스트 수**가 인용됐다.

**규칙**: 인용하려는 게이트는 네 워크트리에서 이번 세션에 직접 재실행한다.
못 하면 숫자 대신 **"확인 안 함"** 이라고 쓴다. 측정에 대한 **"측정 안 함"** 과 같은 규칙이다.

---

## 3. 검사가 통과하면서 아무것도 보지 않는다 — 아홉 번 반복됐다

`docs/review-findings.md` D8~D13, `docs/CONTRACTS.md:1013-1019`.

| 사건 | 무엇이 비어 있었나 |
|---|---|
| `data-pinned` 가 초록 | `.gitignore` 의 앵커 없는 `data/` 가 `configs/data/` 를 삼켜 **검사할 config 가 0개**였다. "every data config pins a commit sha" 가 참이 됐다 |
| Wave 0 게이트 `102 passed` | `configs/data/` 가 untracked 였다. 그 체크아웃에서만 성립했고 clean clone 은 config 합성조차 못 했다 |
| `axis-values` 거짓 양성 | `PackedCollate.__call__` 을 `raise NotImplementedError` 로 갈아도 출력이 **바이트 단위로 같았다** |
| `axis-values` 거짓 음성 | 멀쩡한 `loss/cached_mnrl` 이 inert 로 보고됐다 |
| `doc-commands` 가 자기 문구로 거짓말 | "5 documented command(s) install what the tests need" 라 적으면서 실제로는 `--extra compose` 한 글자만 봤다. 전수 수집하니 미설치 3건 |
| `plan-files` 가 한 방향만 봄 | 블록의 언급을 줄이면 참으로 유지됐다 |
| `config-consumed` 의 `_strip_prose()` | 3가지로 뚫렸다. 실제 방어는 count 추적이었고, 자기 baseline 줄을 편집하는 레인이 그 방어를 없앨 수 있었다 |
| `axis-wired` 의 baseline note 가 blocker 를 가림 | note 는 담당 레인만 적었는데, 그 상태의 실제 의미는 `assert_matches` 가 **모든 timing 런을 거부해 측정이 하나도 불가능하다**는 것이었다. 그 위에 Wave 3 가 얹혔다 |
| `axis-values` 가 두 질문을 한 숫자로 뭉갬 | "축 기계가 받는가"와 "이 연구가 켤 수 있는가". 텍스트 전용 fixture 하나만 넘겨 `loss 2/2` 로 셌다 |

**부수적으로 더 나쁜 것**: 그 `axis-values` 사건의 재현 시도 자체가 틀렸다 — 사보타주를
**클래스 본문 앞쪽에** 끼워 넣었는데 실제 `__call__` 이 뒤쪽에 있어 **나중 정의가 이겼다.**
아무것도 안 죽었는데 "재현했다"고 보고됐다.

**규칙 셋**:
1. 게이트가 통과하면 **검사 대상이 비어 있지 않았음을 함께 보인다.** 아무것도 없는 위를
   지나간 초록은 빨강이다.
2. 추가한 검사는 **부숴서 죽는 것을 보고** 그 출력을 그대로 인용한다.
3. 사보타주를 믿기 전에 **인터프리터가 어느 정의를 잡는지 확인한다**:
   ```
   python -c "import <mod> as m; f=m.<Cls>.<meth>; print(f.__code__.co_filename, f.__code__.co_firstlineno)"
   ```
   수가 안 움직이면 그것은 재시도할 일이 아니라 **네 검사에 대한 발견**이다. `inert: true` 로 보고한다.

공허 방지 3중 잠금(`263f307`/`274fa5f`): `xfail(strict=True)` + AST 로 남은 마커를 이름으로
세는 가드 + **그 가드가 자기 마커를 세지 않는 것**. 셋 중 하나라도 빠지면 "마커가 남아 있는
초록 상태"가 생긴다.

---

## 4. 팬아웃 자체가 무너진 방식

### 4.1 base drift — 직전 팬아웃이 여기서 붕괴했다

워크트리가 `origin/main` 에서 잘려 **지정된 base 보다 6커밋 뒤**였다. 결과:

- 레인들이 main 이 이미 통일해둔 축 상태 어휘를 못 보고 옛 이름으로 작업했다
- **lane-e 가 동결된 계약 파일 `tests/contract/test_kernel_provenance.py` 를 자기가 다시 썼다** —
  동결이 막으려던 바로 그것. 그 브랜치 기준으로 통과한 완료 조건은 main 에서 성립하지 않는다
  (`BUILD_FINGERPRINT_KEY` 가 `kernel` vs `attention`, 교차 fixture 검증 테스트 유무)

**SHA 동일성 비교만으로는 부족하다.** 그 브랜치의 base 는 내부적으로 일관됐고 그냥 오래됐을
뿐이다. 진입 시 `pytest tests/contract -q` 와 `audit_plan.py` 가 그 wave 에 고정된 수치를
그대로 찍는지까지 확인해야 잡힌다.

### 4.2 레인이 고칠 수 없는 게이트에 레인을 세웠다

`.plans/remaining-code/PLAN.md:122-162` (구판), 커밋 `93c9c90`. 세 레인이 `criteria-failed` 로
멈췄고 **셋 다 정직하게 보고했으며 결함은 레인이 아니라 계획에 있었다.**

- `plan-files` 는 신설 파일에 대해 빨개지는데 `PLAN.md` 등재 권한은 통합 레인에만 있었다
- 러너가 "소유 디렉터리마다 `AGENTS.md` 를 갱신하라"고 지시했는데 이 저장소에는 디렉터리별
  `AGENTS.md` 가 없다. 만들면 동시에 도는 다른 레인의 파일을 문서화하게 되고 그것은 지어내기다

**이번 캠페인의 처리**: `PLAN.md` 레이아웃 등재와 `docs/audit-baseline.json` 갱신은
**머지 단계(메인 세션)** 가 한다. 레인은 신설 파일 목록만 보고한다.

### 4.3 동결된 계약 자체에 결함이 남아 있었다

커밋 `f102cd2` → `5971874` → `e5926bc`. 네 경계가 같은 축 상태를 **다른 문자열·다른 모양**으로
못박고 있어서 lane-c 가 어느 쪽을 구현해도 다른 계약이 빨개지는 상태였다. `record-report` 의
레코드 샘플은 lane-c 가 만들 수 없는 레코드였고 자기 자신과도 모순됐다. `kernel-provenance` 는
payload 가 build fingerprint 안에 있다고 적었는데 `loader-bench` 가 그 자리를 허용하지 않아
**lane-g 는 시작 전부터 막혀 있었다.**

**규칙**: `tests/contract/` 는 아무도 소유하지 않고 아무도 고치지 않는다. 계약이 틀렸다고
판단하면 `boundaryRequests` 로 **요청만** 하고, 통합 wave 가 **하나의** 개정본을 낸다.
양쪽이 각자 자기 편을 패치하면 그것이 위 사고의 재발이다.

---

## 5. 감사 baseline 은 양방향 래칫이다

`docs/audit-baseline.json` 은 KNOWN 실패 3건(`axis-values` count 3, `config-consumed` 4,
`verdicts-closed` 4)을 담는다. `scripts/audit_plan.py` 의 `classify()` 는 **grew 도 shrank 도
newly-fixed 도 전부 BLOCK 한다.**

이유: 멤버십만 보면 이미 실패 중인 체크 안의 변화에 눈이 먼다. 실측으로, Wave 2 캡처 레이어를
통째로 지워도 감사 출력이 **바이트 단위로 같았다.**

baseline 의 `"count": "3"` (문자열) 한 개가 TypeError 로 **13개 체크 전부의 출력을 삼킨** 적도 있다.

**규칙**: 레인은 `docs/audit-baseline.json` 을 **절대 건드리지 않는다.** 자기 몫 산문은
`.plans/notes/<lane>.md` 에 적고 머지 단계가 한 커밋으로 적용한다. `--update-baseline` 은
통합 단계만 실행한다.

---

## 6. 프레임워크별 함정 (요약 — 상세는 `.plans/research/`)

### 환경
- **단일 환경 공존 불가(확정).** 공유 lockfile 이 버전을 하나로 강제한다. 하한 없이 묶으면
  `ms-swift 1.4.0` 같은 것이 잡히고 그것으로 잰 숫자는 "ms-swift 측정"이라는 이름의 무의미다.
- `hydra-core` ↔ axolotl 의 antlr4 정확 고정은 **해소 불가** → **파드는 Hydra 를 쓰지 않는다.**
  이미 해석된 config JSON 을 받는다.
- `uv lock` 성공은 해석 성공이지 설치도 실행도 아니다.
- 해석 스택이 칸마다 다르다: transformers 5.5.0(unsloth) / 5.12.1(ms_swift) / 5.14.1(나머지),
  torch 2.11.0 / 2.12.1 / 2.13.0. **버전은 교란 요인이고 결과에 보여야 한다.**

### transformers / native
- **Qwen3.5 의 레이어 75% 인 Gated DeltaNet 은 `fla`+`causal-conv1d` 가 없으면 조용히 느린
  torch 구현으로 떨어진다.** 예외도 경고도 아니고 **로그 한 줄**이다. 그래서 두 패키지를
  6개 env 전부에 넣었다 — 한 이미지에서 빼면 "프레임워크 차이"라는 이름으로 융합 커널 대
  fallback 을 비교하게 된다.
- 부작용: `kernel=none` 요청에 `applied=fla` 가 잡혀 영구 mismatch. `_environment_bound_kernel` 로 닫았다.
- `processor(text=..., images=...)` 만으로 멀티모달 배치가 안 만들어진다. 텍스트에 이미지
  placeholder 가 없으면 forward 가 실패한다. MMEB 쿼리에는 `<|image_1|>` 이 이미 들어 있다.
- **평평한 이미지 목록은 `Gemma4Processor` 가 한 행의 이미지로 읽어 배치를 거부한다** — 행별로 묶어야 한다.
- 이미지 digest 를 고정해도 **커널은 고정되지 않는다**: flash-attn 패키지가 없으면 transformers 가
  `flash_attention_2` 요청을 Hub 저장소 이름으로 바꿔 **런 시작 중에 내려받는다.**
- 멀티모달은 `attn_implementation` 을 서브컨피그별 dict 로 받는다. 문자열 하나를 주면 **어느
  타워에 걸렸는지가 모델마다 다르다.**
- 미등록 커널 + packing = **시퀀스 격리가 조용히 사라진다** (마스크 생성을 통째로 건너뛰고
  `attention_mask=None` 을 넘긴다).
- 4D 마스크를 직접 주면 **fa2 varlen 경로가 꺼진다.** 그래서 결정 2 는 "블록 대각 마스크를
  직접 만들지 않는다"이다.

### unsloth
- `full_finetuning=False` 기본 → 전 파라미터 동결. backward 는 통과한다(§1).
- padding_side 를 무조건 left 로 덮어쓴다.
- 6개 중 가장 낮은 스택(torch 2.11.0 / transformers 5.5.0).
- 필드 사례: 어떤 재현 연구가 unsloth 46,000 tok/s 에서 **grad norm 0** 을 관측했다.
- `fast_sentence_transformer_accepts_vlm` 이 `expected_failure=True` 로 등록돼 있었는데 1차의
  실패 내용은 **프레임워크의 거부가 아니라 배포판 부재**였다. 2차엔 3칸 다 통과. 매트릭스는
  `OK (9 checks, 문서화된 한계 1건)` 으로 렌더되는데 그 "한계 1건"이 **통과해버린** 체크다.

### axolotl
- 검증 순서(§1). dtype/autocast(§1).
- `cffi 1.16.0` 은 CPython 3.13 에서 **동작할 수 없다** — 컴파일은 되고 import 에서
  `undefined symbol: _PyErr_WriteUnraisableMsg` 로 죽는다. `no-build-isolation-package` 로 비켜갔다.
- `flash-linear-attention==0.4.1` 을 정확 고정한다. 나머지 다섯은 0.5.2 — **axolotl × Qwen3.5
  수치에 이 차이를 표기해야 한다.**

### ms-swift
- `Qwen3_5Loader(Qwen3VLLoader)` — 텍스트 전용 Qwen3.5 가 VL 경로로 적재된다.
  이번 캠페인에서 `qwen3_5_0_8b` 를 적재한 칸은 **전부** VL 경로를 탔다. 이 체크포인트를
  "텍스트 전용"으로 부르는 것은 config 수준에서 이미 성립하지 않는다.
- `check_requires` 는 ImportError 를 삼키고 warning 만 남긴다 → 게이트가 아니다.
- `get_model_processor` 가 `from_pretrained` 를 소유 → `axes.load_kwargs` 가 갈 곳이 없다.

### tevatron
- 상류가 peft 를 선언하지 않는다 / `config.pad_token_id` 직접 접근(§1).
- `DenseModel.forward` 가 **인코딩·풀링·정규화·스코어링·InfoNCE·분산 게더를 전부 자기가 한다.**
  우리 하네스는 그것을 4개로 나눠 갖고 있다 → 결정 5: 프레임워크의 학습 스텝을 그대로 잰다 →
  tevatron 칸의 `loss` 와 `parallel.cross_device_negatives` 는 **프레임워크 소유**로 기록된다.
- `framework_version: "unknown"` 으로 통과한다 — 버전 기록 규칙에 이 칸은 답하지 못한다.

### sentence-transformers
- 자체 `_backward` 가 `steps.infonce_backward` 의 가드를 **거치지 않고** `params_with_grad` 만
  돌려준다. `trainable_params` 를 세지도, 0인지 확인하지도 않는다. **unsloth 를 잡은 가드가
  여기엔 없다.** 이번엔 310/320/505 였지만 **0이었다면 그대로 초록이다.**
- padding-side 정렬을 하지 않는다(설계). 생성형 VLM 2종이 ST module layout 이 없어 기본
  pooling 으로 떨어지는지는 **확인 안 함**.

### LoRA vs full
- **`peft.mode` 를 켜는 순간 모든 LoRA timing 런이 차단된다** — peft 가 base 파라미터를 전부
  얼려 `freeze.ple=false` 요청이 mismatch 가 된다. 그리고 이 프로젝트의 표제 비교가 full vs LoRA다.
  해법은 probe 완화가 아니라 `freeze.*` capture 가 peft 가 얼린 것을 기준선으로 잡는 것.
- **probe 가 mismatch 를 낼 때 고칠 것은 probe 가 아니다.** 이질적 적용은 표현되어야 할
  상태이지 숨겨야 할 잡음이 아니다.

### gradient checkpointing
- 수정자 보고의 "정책을 어느 방향으로 망가뜨려도 잡힌다"는 **거짓이며 철회됐다.** 정책 함수만
  바꾼 변이 4종이 전부 살아남았다 — 어텐션과 bias 있는 선형층을 조용히 재계산해도 스위트는 초록.
- CPU 에서 SDPA 는 저장 목록에 아예 없는 패킷으로 디스패치된다 → CPU 에서 `selective` 는
  어텐션을 무조건 재계산한다. **CPU 가 구조적으로 볼 수 없는 것이 있다.**

### 축 미구현 — 의존성 공백이 아니라 코드 공백
`optim=adamw_8bit`, `precision=mxfp8/nvfp4`, `train.offload=*`, `parallel=ddp/fsdp2/zero2/zero3`,
`dataloader.backend=dali`, `peft=qlora` 는 전부 `trainbench/axes.py` 가 거부한다.
**패키지는 이미 다 있다.** `kernel=kernels_hub` 는 두 진입점이 모델 객체를 요구하는데
`axes.patch` 는 모델 생성 전에 도므로 **축 값을 버린다**(결정 6).

---

## 7. 파드 운영 (이번 범위 밖이지만 알아야 함)

- **이미지에 코드가 구워져 있다.** 스키마를 바꾸고 재빌드 없이 파드를 띄우면 pydantic 이
  양방향으로 거부한다 — 실측 **17초 간격 40회 크래시루프, A100 12분 과금.**
- **RunPod 은 컨테이너 재시작을 막을 수 없다**(REST PodCreateInput 33개 필드에 재시작 정책 없음).
  probe 는 결정적이라 티가 안 났지만 **timing 런은 매 재시작이 새 숫자를 같은 경로에 올리고
  마지막 컨테이너가 조용히 이긴다.** `.trainbench-done` sentinel 로 닫았다.
- 크래시루프를 건강한 런으로 읽은 판정을 **세 번 고쳤고 세 번 다 실 파드에서 실패**했다.
  마지막 원인은 GraphQL 읽기가 전부 **403 Cloudflare `error code: 1010`** (urllib 기본 User-Agent 거부).
- `scripts/orchestrate.py` 의 파드 토큰 스코프 검사는 **허용목록이고 그것은 선호가 아니라 측정이다** —
  deny list 는 dev 환경이 주입하는 27개 중 22개를 통과시켰다.
- **랩톱 감사 호스트에서 도는 검사는 옳은 config 를 빨갛게, 죽는 config 를 초록으로 만든다.**
  27개 계획 런: 파드 환경 refused 0, 감사 호스트 refused 5.

---

## 8. 연구 가정 자체가 근거 없던 것

- **`AGENTS.md` 의 3% 임계값은 근거 없는 상수다.** GPU 경합만으로 표준편차 30배·평균 +21% 가
  관측된 사례가 있다. 첫 파드에서 canonical baseline 10회 반복으로 유도해야 한다.
- **고정 seed 반복은 분포가 아니라 한 점을 재측정하는 것이다.** MLPerf CLOSED 는 seed 를
  `/dev/urandom` 에서 뽑고 run 마다 기록한다. 스키마는 만들되 **값은 파드가 정한다.**

---

## 9. 재사용 후보 — 읽는 것은 허용되고 그대로 믿는 것은 아니다

병합하지 말고, 쓸 만한 것을 자기 브랜치에 다시 만들고, 완료 조건은 **직접 실행해서** 확인한다.

| ref | base | 내용 | 주의 |
|---|---|---|---|
| `preserved/wf_c5aa0913-a6d-4` | **e5926bc (현재 main)** | `applied.py` +334, `tests/test_applied.py` +235, `test_applied_axes.py` 마커 제거 | base 가 정확하다. capture 레인의 1순위 참고 |
| `preserved/agent-a95eb5c32258196cc` | 9363197 | `applied.py` +334, `axes.py` +228, `CONTRACTS.md`, 신규 `tests/test_axes.py` | main 이 그 뒤 축 상태 어휘를 통일했다 |
| `preserved/agent-a3d669f277c3ea625` | 9363197 | env lock 6종 + Dockerfile + CI (+1804줄) | **통합자 전용 파일**이다. 레인이 손대지 않는다 |
| `aborted-wave1-lane-a` | 3ebcade | tevatron `pad_token_id` shim + 테스트 | 계약과 무관. 재사용 가치 높음 |
| `aborted-wave1-lane-c` | 274fa5f | capture 구조 전체 | 계약 5개를 다 보고 작업했으나 그 뒤 main 이 어휘를 바꿨다 |
| `aborted-wave1-lane-e` | 3ebcade | `kernels.py` +437, `tests/test_kernels.py` +232, methodology | **`test_kernel_provenance.py` 를 자기가 다시 썼다. 그 파일은 버린다.** 동결본과 `BUILD_FINGERPRINT_KEY` 가 다르다(`kernel` vs `attention`) |

---

## 10. 이 캠페인에서 지킬 것 — 요약

1. 핀된 소스를 읽고 나서 단언한다. 디코이는 `dist-info` 로 거른다.
2. 자기가 낸 숫자만 옮긴다. 못 내면 "확인 안 함".
3. 게이트가 통과하면 검사 대상이 비어 있지 않았음을 함께 보인다.
4. 추가한 검사는 부숴서 죽는 것을 보고, 그 전에 인터프리터가 어느 정의를 잡는지 확인한다.
5. base 는 40자 SHA 로 확인한다. 동일성이지 ancestry 가 아니다. 계약 테스트 수치까지 대조한다.
6. `tests/contract/` 는 고치지 않는다. 요청만 한다.
7. `docs/audit-baseline.json`, `PLAN.md` 레이아웃, `envs/**`, `pyproject.toml`, `uv.lock`,
   `scripts/audit_plan.py` 는 **통합자 전용**이다.
8. 소유 밖 파일을 건드렸으면 숨기지 말고 보고한다. 머지가 어차피 더 비싸게 찾아낸다.
