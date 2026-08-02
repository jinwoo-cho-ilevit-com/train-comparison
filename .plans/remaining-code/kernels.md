# kernels — 커널 provenance (wave 1)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`.
> 리서치: `.plans/research/transformers-varlen-prompt.md`, `.plans/research/axis-libraries.md`.
> **파드가 런타임에 외부에서 코드를 받는 경로를 다룬다. 보안 표면이다.**

## 목표

런타임에 **실제로 바인딩된** 어텐션 커널의 신원을 결과에 남기고, 그것이 불확실할 때 거부한다.
그리고 `docs/methodology.md` 의 틀린 절을 고친다.

## Owns

```
trainbench/kernels.py        ⊕ 신설
tests/test_kernels.py        ⊕ 신설
docs/methodology.md
```

## 이 레인에는 계약 게이트가 없다 — 그래서 만든다

`tests/contract/test_kernel_provenance.py` 는 xfail 0개이고 **지금 통과한다.**
그리고 그 파일은 `trainbench/kernels.py` 를 **참조하지 않는다** — `validate_kernel_fingerprint`
가 계약 파일 안에 정의돼 있고 거기서 11번 호출된다. 저장소 전체에서 `trainbench/kernels.py` 를
언급하는 곳은 그 파일의 docstring 한 줄뿐이다.

즉 **빈 `trainbench/kernels.py` 를 만들어도 모든 게이트가 초록으로 남는다.**
그것이 이 저장소가 아홉 번 겪은 바로 그 모양이다(`HAZARDS.md §3`).

→ `tests/test_kernels.py` 가 이 레인의 유일한 게이트다. 최소 요구:
**`trainbench.kernels` 의 검증 함수가 `test_kernel_provenance.py` 자신의 validator 와
같은 판정을 내는지**를 계약이 쓰는 mutation 케이스 전부에 대해 확인한다.
둘이 갈라지면 계약은 초록인데 런타임은 다른 것을 허용한다.

## 작업 1 — 해석된 커널의 provenance 를 기록한다

transformers 의 `attention_interface` 문서가 말한다:
*"Kernels automatically register to AttentionInterface upon detection. You don't need to install
the FlashAttention package explicitly. Requesting FlashAttention by name also falls back to the
Hub kernel."*

즉 **실제 커널이 런 시작 중에 Hub 에서 해석된다.** `AGENTS.md` 의 "런마다 resolved torch/
framework 버전을 기록한다"로는 부족하다 — 같은 버전이 다른 커널을 바인딩할 수 있다.

repo + revision 이 런 레코드에 들어가야 한다. **신원은 요청이 아니라 빌드된 모델에서 되읽는다.**
revision 을 얻지 못하면 **패키지 버전으로 대체하지 말고 거부한다.**

`kernels` 라이브러리에는 `version` 지정, `revision`, `kernels.lock` + `get_locked_kernel()`
핀 메커니즘이 있고 이 저장소는 쓰지 않는다. 정확한 시그니처는
`.plans/research/axis-libraries.md` 에서 확인한다.

경계 `kernel-provenance` 가 payload 모양을 못박았다:
`tests/fixtures/kernel_fingerprint.sample.json`, 그리고 **`BUILD_FINGERPRINT_KEY` 는
`"attention"` 이다.** `"kernel"` 이 아니다 — `kernel.name` 은 다른 축이고, 한 이름이 두 축을
가리키면 `5971874` 가 없애려던 결함이 그대로 재발한다. 이 payload 는 `loader-bench` 가
동결한 어댑터 출력의 `fingerprint` 와 **같은 객체**다.

## 작업 2 — mask fn 미등록이면 packing 을 거부한다

문서:
*"If the custom `attn_implementation` name is not registered in AttentionMaskInterface,
Transformers skips mask creation and passes `attention_mask=None` to the attention layers…
those constraints can be silently dropped."*

**미등록 커널 + packing = 시퀀스 격리가 조용히 사라진다.** 거기서 `dataloader.packing=true` 를
허용하면 격리 없는 packing 수치가 나오고 그것은 다른 것을 잰 숫자다.

거부가 옳고 **거부 메시지가 이유를 말해야 한다.**
정확한 줄과 원문은 `.plans/research/transformers-varlen-prompt.md` 에서 확인한다.

## 작업 3 — 파드에서 런타임 커널 fetch 금지

이 저장소에 이미 "학습 데이터를 네트워크 볼륨에서 읽지 않는다"는 규칙이 있다.
측정 도중 네트워크로 커널을 받아오는 것은 같은 종류의 오염이다.

**이미지 digest 를 고정해도 커널은 고정되지 않는다**: flash-attn 패키지가 없으면 transformers 가
`flash_attention_2` 요청을 Hub 저장소 이름으로 바꿔 런 시작 중에 내려받는다.

환경변수 둘(`USE_HUB_KERNELS`, `HF_HUB_OFFLINE`)만으로는 부족할 수 있다 —
`hub_kernels` 가 그 변수를 **import 시점에 한 번만** 읽고 전역에 캐시하면 나중에 바꿔도
무효다. 그 사실을 `.plans/research/axis-libraries.md` 에서 확인하고, 사실이면 캐시된 전역까지
함께 닫는다. 확인하지 못하면 **그렇게 적는다.**

## 작업 4 — `docs/methodology.md` 정정

**§10.1 / §10.2 가 틀렸다.** 코드를 읽으면 답이 나온다:

- transformers 5.14.1 은 `position_ids` 가 0에서 재시작하면 **블록 대각 격리를 스스로 만든다**
- `PackedCollate` 는 그 전제 세 개를 만족한다
- arch 별로 다르다: `qwen3_vl`/`gemma4` 는 격리되고, **`qwen3_5` 는 절반**이다 —
  linear_attention(Gated DeltaNet)이 `position_ids` 를 보지 않고 `cu_seq_lens_q` 를 읽는다
- 부수 사실: sdpa + packing 은 `allow_is_causal_skip=False` 가 되어 `(1,1,total,total)` bool
  마스크를 **물리적으로 만든다**. 8×2048 이면 268,435,456 bytes. 정확성이 아니라 **성능** 때문에
  packing 은 `attn=fa2` 를 선호한다
- 멀티모달은 `attn_implementation` 을 서브컨피그별 dict 로 받는다. 문자열 하나를 주면
  **어느 타워에 걸렸는지가 모델마다 다르다**

**§9 도 함께 고친다** — `kernel_modules` 에 임계값이 없다는 것. 502개 중 liger 1개인 모델이
`applied='liger', matches=True` 로 읽힌다. 임계값을 지어내면 라이브러리의 문서화된 정상
동작을 거부하게 되므로 **"측정 안 함"으로 기록만 한다.** 멀티모달의 백본별 어텐션이 그 간극의
원인 하나다.

**모든 정정은 파일:줄을 인용한다.** 리서치 브리프의 인용을 그대로 옮기되, 브리프가
`resolvedPath` 하위 절대경로를 쓰는 것과 달리 문서에는 패키지 상대 경로로 적는다.

## 참고할 것 — 계약 파일은 버린다

`aborted-wave1-lane-e` 의 `060280d` 에 `trainbench/kernels.py`(+437),
`tests/test_kernels.py`(+232), methodology 정정(+168)이 있다.

**그러나 그 브랜치는 `tests/contract/test_kernel_provenance.py` 를 자기가 다시 썼다**
(동결이 막으려던 바로 그것). main 의 동결본과 실측 차이:

1. **`BUILD_FINGERPRINT_KEY` 가 다르다** — lane-e 는 `"kernel"`, main 은 `"attention"`
2. main 에는 `test_the_two_fixtures_that_carry_this_payload_agree_with_it` 이 있다 —
   `adapter_out.sample.json` 과 `run_record.sample.json` 을 직접 읽어 같은 payload 임을 검증
3. main 의 docstring 이 build fingerprint = `loader-bench` 어댑터 payload 의 `fingerprint`
   **같은 객체**임을 명시한다

**그 계약 파일은 버리고 나머지만 본다.** 그 브랜치 기준으로 통과한 완료 조건은 무효다.

## 완료 조건

1. 해석된 어텐션 커널의 repo + revision 이 런 레코드에 들어간다 →
   `infisical run --env=dev -- uv run pytest tests/contract/test_kernel_provenance.py -q`
   (지금도 통과한다. **그것만으로는 이 레인이 무언가 했다는 증거가 아니다** — 2번이 증거다)
2. **`trainbench.kernels` 의 검증이 계약 자신의 validator 와 같은 판정을 낸다** —
   계약이 쓰는 mutation 케이스 전부에 대해 →
   `uv run pytest tests/test_kernels.py -k agrees_with_contract`
3. `attn_implementation` 이 `AttentionMaskInterface` 에 없으면 `packing=true` 가 거부되고
   거부 문구가 "마스크가 조용히 사라진다"는 이유를 말한다 →
   `uv run pytest tests/test_kernels.py -k mask_registered`
4. 파드에서 런타임 커널 fetch 가 금지된다 →
   `uv run pytest tests/test_kernels.py -k no_runtime_fetch`
5. `methodology.md` §9/§10.1/§10.2 가 정정됐고 근거가 파일:줄로 인용됐다
6. 각 검사를 되돌리면 죽는다. mutation 출력 인용. **사보타주 전에 `co_filename` 확인**
7. 네 게이트. `plan-files` 가 `trainbench/kernels.py`, `tests/test_kernels.py` 둘로 빨간 것은
   실패가 아니다 — `newFiles` 에 적는다. **다른 이름이 함께 나오면 실패다**
8. **확인 안 함** — 파드 질문으로 등록한다:
   - fa2 varlen 경로가 실제로 도는가
   - `fa3`/`fa4` 가 `envs/native` 에서 적재되는가
   - transformers 5.5.0(unsloth)과 5.12.1(ms_swift)에 `find_packed_sequence_indices` 가 있는가

## 하지 않는 것

- `cu_seq_lens_*` 를 실제로 전달하는 것 — **packing** 레인 (`trainbench/collate.py`, wave 2)
- 어댑터가 돌려주는 빌드 지문 — **adapters** 레인. **경계 `kernel-provenance` 에서 맞춘다**
- `envs/*/uv.lock` 에 `kernels` 핀 추가 — **통합자 전용**. 필요하면
  `.plans/deps/kernels.txt` 에 적는다
- `trainbench/axes.py` 의 커널 패치 — **axes** 레인
- `AGENTS.md`, `docs/CONTRACTS.md` — **integrate** 레인.
  올라가야 할 사실은 `.plans/notes/kernels.md` 에 적어 넘긴다
