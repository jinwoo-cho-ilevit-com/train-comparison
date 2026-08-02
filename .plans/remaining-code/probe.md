# probe — 프로브 가드 (wave 1)

> 먼저 읽는다: `HAZARDS.md`(§1 프레임워크별 함정, §3 공허한 검사), `PLAN.md`.
> 리서치: `.plans/research/tevatron.md`, `.plans/research/sentence-transformers.md`.

## 목표

프로브가 **틀린 것을 초록으로 보고하는** 세 자리를 막고, tevatron 3칸을 여는 shim 을 넣는다.

## Owns

```
trainbench/probe/steps.py
trainbench/probe/sentence_transformers.py
trainbench/probe/tevatron.py
tests/test_probe.py
```

## 작업 1 — sentence_transformers 의 동결 가드 사각지대

`trainbench/probe/sentence_transformers.py:75-89` 의 자체 `_backward` 가
`steps.infonce_backward`(가드는 `steps.py:421`, `:428`)를 **거치지 않고** `params_with_grad` 만
돌려준다. `trainable_params` 를 세지도, 0인지 확인하지도 않는다.

**unsloth 3칸을 잡은 가드가 여기엔 없다.** 이번 캠페인 값은 310/320/505 로 0이 아니었지만
**0이었다면 그대로 초록으로 통과했을 것이다.**

고친다. 그리고 **테스트가 실제 모양으로 돈다** — 전 파라미터를 얼리고 임베딩 출력에 훅을 걸어
backward 가 유한한 loss 를 내게 만든다. unsloth 가 낸 바로 그 모양이다
(`enable_input_require_grads()` 가 임베딩 **출력**에 requires_grad 를 걸어 그래프가
미분 가능하게 유지된다). 스텁으로 `trainable_params=0` 을 흉내 내는 테스트는 이 결함을
증명하지 못한다.

sentence-transformers 가 모델을 어떻게 감싸는지(`.parameters()` 가 무엇을 돌려주는지)는
`.plans/research/sentence-transformers.md` 에 있다. **핀된 소스를 읽고 나서 단언한다.**

## 작업 2 — `axes_verified` 가 `all_matched:false` 에 통과한다

지금 `axes_verified` 체크는 불일치가 있어도 통과한다. 실측된 불일치 둘:

- `kernel.name`: `none` 요청인데 **`qwen3_5_0_8b` 칸 전부**(native/ms_swift/axolotl/unsloth/
  sentence_transformers)에서 `fla` 가 적용됐다
- `precision.name`: `bf16` 요청인데 axolotl 3칸 + unsloth 3칸에서 `mixed(bf16,fp32)`

**주의 — 이것은 "불일치를 없애라"가 아니다.** `docs/CONTRACTS.md:209-215`:
*"probe 가 mismatch 를 낼 때 고칠 것은 probe 가 아니다. 이질적 적용은 표현되어야 할 상태이지
숨겨야 할 잡음이 아니다."* 요구는 **`all_matched:false` 가 초록으로 읽히지 않는 것**이다.
불일치 자체는 결과에 남아야 한다.

`fla` 는 이미 `_environment_bound_kernel` 로 patch 단계에서 처리됐다(그 이미지에서는 뺄 수
없으므로). 그 처리와 이 체크가 어떻게 맞물리는지 확인하고, **환경 구속 불일치와 진짜 불일치를
구별**한다. 구별하지 못하면 그것을 보고한다 — 지어내지 않는다.

## 작업 3 — tevatron `pad_token_id` shim

2차 캠페인에서 tevatron 3칸 전부가 이렇게 죽었다:
```
AttributeError: 'Gemma4Config'/'Qwen3_5Config'/'Qwen3VLConfig' object has no attribute 'pad_token_id'
```

핀된 `encoder.py` 가 `from_pretrained` 직후 `base_model.config.pad_token_id` 를
**getattr 이 아니라 직접** 접근한다. transformers 5.14.1 의 합성 config 는 그것을 최상위에 두지
않고 `get_text_config()` 뒤에 둔다. 정확한 줄과 원문은
`.plans/research/tevatron.md` 에 있다 — **거기서 확인하고 인용한다.**

`_load` 에서 config 를 만들고 `pad_token_id` 를 심어 `hf_kwargs` 로 넘긴다.
거기서도 `None` 이면 그대로 두어 상류의 기본값 0 이 채우게 한다. `revision` 도 함께 넘긴다.
심은 값과 심었는지 여부를 `dense_model_load` 의 detail 에 **기록한다.**

**주석 한 줄로 이것이 모델이 읽지 않는 shim 임을 남긴다** — 진짜 값은 텍스트 서브컨피그에 있고,
최상위에 심는 것은 상류의 누락된 `getattr` 을 우회하기 위해서일 뿐이다. 그 주석이 없으면
다음 사람이 같은 발굴을 다시 한다.

참고: `aborted-wave1-lane-a` 의 `1e7030f` 에 이 작업이 있다.
**병합하지 않는다.** `git show aborted-wave1-lane-a` 로 읽고 자기 브랜치에 다시 만든다.
PLAN 의 판정: "계약 파일을 못 본 채 작업했으나 tevatron shim 자체는 계약과 무관하다."

## measure 레인과의 경계 — 테스트가 없는 경계다

**measure 레인도 `grad_norm`/`trainable_params` 를 정의한다** — 측정 시점 유효성 게이트로.
이 레인은 프로브 시점 거부 가드로 정의한다. 두 정의가 어긋나면 리포트의 게이트와 프로브의
거부가 다른 말을 하게 되고 **어떤 테스트도 그것을 비교하지 않는다.**

→ `.plans/notes/probe.md` 에 네 정의를 정확히 적는다:
- `trainable_params` 는 무엇을 세는가 — 지금은 **텐서 개수**이지 원소 수가 아니다
- `params_with_grad < trainable_params` 는 결함이 아니다 — 프로브 배치가 텍스트 전용이면
  비전 타워 파라미터에 grad 가 안 붙는다 (`docs/support-matrix.md:957-962`)
- 0 의 의미는 무엇이고 언제 재는가

머지 단계가 두 노트를 대조한다.

## 완료 조건

1. sentence_transformers 프로브가 동결된 그래프를 통과시키지 않는다.
   **테스트가 실제 모양으로 돈다** →
   `uv run pytest tests/test_probe.py -k sentence_transformers_frozen`
2. `axes_verified` 가 `all_matched:false` 에 통과하지 않는다 →
   `uv run pytest tests/test_probe.py -k axes_verified`
3. 세 합성 config 전부에 `pad_token_id` 를 심은 config 가 `hf_kwargs` 로 넘어간다 →
   `infisical run --env=dev -- uv run pytest tests/test_probe.py -k tevatron`
   테스트는 스텁이 아니라 **실제 transformers config 클래스 3종**으로 세운다
4. shim 을 빼면 3번이 죽는다. 가드를 되돌리면 1·2번이 죽는다. **각각 mutation 출력 인용.**
   사보타주 전에 `co_filename`/`co_firstlineno` 확인 (`HAZARDS.md §3`)
5. 네 게이트
6. **확인 안 함**: tevatron 3칸이 실제로 `dense_model_load` 를 통과하는지(12/18 → 15/18)는
   파드가 답한다. **이번엔 실패하더라도 그것이 체크포인트에 대한 답이어야 한다** —
   상류 버그에 대한 답이 아니라. 파드 질문으로 등록한다

## 하지 않는 것

- tevatron 의 `forward` 시그니처와 그것을 어댑터 경계에서 어떻게 표현할지 — **adapters** 레인
  (wave 2). 이 레인은 **적재**만 연다
- `trainbench/probe/{native,unsloth,ms_swift,axolotl,registry}.py` — **adapters** 레인
- `trainbench/metrics/` 의 유효성 게이트 — **measure** 레인
- `envs/tevatron/` 의존성 — 이미 `peft>=0.20` 이 들어가 있다. **통합자 전용**
- `scripts/report.py` — **report** 레인
