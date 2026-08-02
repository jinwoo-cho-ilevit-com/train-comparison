# probe 레인 노트

## 1. `trainable_params` / `params_with_grad` / `grad_norm` 의 정의 (measure 레인과 대조용)

정의 자리: `trainbench/probe/steps.py::training_step_evidence`.
프로브 시점의 **거부 가드**이고, 측정 시점의 유효성 게이트가 아니다.

- `trainable_params` = `sum(1 for p in model.parameters() if p.requires_grad)`.
  **파라미터 텐서의 개수**이지 원소 수가 아니다. 이 수가 답하는 질문은 "무엇이라도
  학습되는가"이고, 거기에 원소 수는 틀린 단위다.
- `params_with_grad` = 위 중 `p.grad is not None` 인 것의 개수. 역시 텐서 개수다.
- `total_params` = `sum(1 for _ in model.parameters())`. 텐서 개수다.
- `loss` = `float(loss.detach())`. 유한성은 **판정하지 않는다** — 유한한 loss 가 곧
  스텝이 일어났다는 증거가 아니라는 것이 이 가드가 존재하는 이유다.

거부 조건 두 개, 이 순서로:

1. `trainable_params == 0` → 거부. 동결된 그래프가 낸 수를 "지원됨"으로 발표하는 것을 막는다.
2. `params_with_grad == 0` (그리고 `trainable_params > 0`) → 거부. pooled 임베딩이 갱신 대상
   파라미터들과 끊겨 있다는 뜻이다.

**`params_with_grad < trainable_params` 는 결함이 아니다.** 프로브 배치가 텍스트 전용이면
비전 타워 파라미터에 grad 가 붙지 않는다 (`docs/support-matrix.md:957-962`). 그래서 이
가드는 두 수의 **비율이나 차이를 보지 않고 0 만 본다.**

**언제 재는가**: `loss.backward()` 직후, `zero_grad` 직전. 옵티마이저 스텝 전이다.
`zero_grad(set_to_none=True)` 는 거부 여부와 무관하게 항상 호출된다 — 실패를 기록하고
계속 가는 호출자가 다음 체크로 grad 를 흘리지 않게 하기 위해서다.

**`grad_norm` 은 이 레인이 정의하지 않는다.** 프로브는 grad norm 을 계산하지도 기록하지도
않는다. measure 레인이 정의하는 `grad_norm` 과 대조할 프로브 쪽 항목은 없고, 겹치는 것은
`trainable_params` / `params_with_grad` 둘뿐이다. HAZARDS 가 인용한 unsloth 필드 사례
(46,000 tok/s 에서 grad norm 0)는 이 가드가 잡는 종류가 아니다 — 그것은 측정 시점 게이트의
몫이다.

## 2. 통합자에게 넘기는 변경 요청

### 2.1 `docs/support-matrix.md` — "초록이지만 그대로 믿으면 안 되는 것"

`:1059` 의 첫 항목 "**`axes_verified`는 `all_matched: false`여도 통과한다**" 는 **더 이상
사실이 아니다.** `steps._refuse_mismatch` 가 불일치를 거부한다. 불일치 자체는 그대로 남고
(`report.applied`), 바뀐 것은 그것이 통과로 읽히지 않는다는 점이다.

그 자리에 들어갈 현재 상태:

> `axes_verified` 는 `all_matched: false` 를 거부한다. 불일치는 결과에 그대로 남고
> (`applied.axes`), 실패 메시지가 축마다 요청/적용을 이름으로 적는다. `kernel.name` 의
> 불일치에는 그것이 이미지에 구속된 것인지(`environment-bound`)가 함께 적힌다 —
> `axes._environment_bound_kernel` 이 읽는 그 값이다. **다른 축에는 그 구별이 없다.**
> `precision.name` 의 `mixed(bf16,fp32)` 가 axolotl 의 fp32 유지 정책인지 잘못된 적재인지는
> 여기서 읽을 수 없고, 읽을 수 없는 것을 environment-bound 로 부르지 않는다.

또한 CPU 호스트에서는 **모든 probe 셀이 이 체크에서 빨개진다**: `steps.dtype_for` 가 CUDA
밖에서 fp32 를 주는데 `configs/precision/` 에 fp32 값이 없고, fused AdamW 커널은 CUDA
전용인데 `configs/optim/` 에 unfused AdamW 값이 없다. 파드에서 초록이 되는지는 **확인 안 함**.

### 2.2 `axes._environment_bound_kernel` 을 공개 이름으로

`trainbench/probe/steps.py::_environment_bound` 가 이 함수를 부른다. 규칙을 두 벌 두면
갈라지므로 복사하지 않았고, 대신 밑줄 이름을 가로질러 부르고 있다. axes 레인(wave 2)이
`trainbench/axes.py` 를 소유하므로, 이름을 공개형으로 바꾸는 것은 그쪽 또는 통합 단계의 몫이다.
바꾼다면 `steps.py` 의 호출 한 줄이 같이 움직여야 한다.

### 2.3 tevatron `hf_kwargs` 는 LoRA 경로에서 peft 로도 간다

`_load` 가 이제 `config=` 와 `revision=` 을 `DenseModel.load` 로 넘긴다. 이 프로브는
`lora_name_or_path` 를 주지 않으므로 지금은 안전하다. 그러나 상류의 같은 dict 가
`LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)` 로도 흘러간다
(tevatron dd063104 `retriever/modeling/encoder.py:170`, `:131`). **adapters 레인이 LoRA 축을
이 경로로 켤 때 `config` 키가 peft 로 넘어간다.** `.plans/research/tevatron.md §2` 가 그 줄을
인용하고, peft 0.20.0 이 그 키를 어떻게 처리하는지는 그 리서치에서도 **확인 안 함**이다.

### 2.4 tevatron `framework_version`

`report.add_version(tevatron)` 는 여전히 `"unknown"` 을 기록한다 — 상류 `src/tevatron/__init__.py`
가 0바이트이고 소스 전체에 `__version__` 이 없다(`.plans/research/tevatron.md §5`).
이번 레인은 이것을 건드리지 않았다. 결과에 실려야 할 값은 버전 문자열이 아니라 lock 의
커밋 sha (`envs/tevatron/uv.lock` 의 `dd063104c81a76d6a77c845f667b46b9e5abd625`)라는 것이
그 리서치의 결론이고, 그 배선은 `probe/types.py`(이 레인 소유 아님)와 report 레인에 걸쳐 있다.

## 3. 파드가 답해야 하는 질문 (여기서 확인 안 함)

- `pad_token_id` 를 심은 config 로 tevatron 3칸이 실제 체크포인트를 끝까지 적재하는가
  (12/18 → 15/18). **이번엔 실패하더라도 그것이 체크포인트에 대한 답이어야 한다** —
  상류의 누락된 `getattr` 에 대한 답이 아니라.
- 그 다음 벽은 `EncoderModel.forward(query, passage)` 시그니처다. `steps.encode` 는
  `model(**batch)` 로 `input_ids` 를 편다(`.plans/research/tevatron.md §3.3`). 이 레인은
  **적재만** 열었고 forward 는 adapters 레인(wave 2)의 몫이다. 실제 예외 문구는 확인 안 함.
- 체크포인트의 `text_config.pad_token_id` 실제 값. 여기서 본 None/None/0 은 **기본
  생성자**의 값이고 체크포인트가 무엇을 적어두었는지는 다르다.
- sentence-transformers 의 생성형 VLM 2종이 ST module layout 없이 어떤 pooling 으로
  떨어지는가. 이번 가드는 `params_with_grad`/`trainable_params` 만 다루고 pooling 은
  건드리지 않았다. 확인 안 함.
