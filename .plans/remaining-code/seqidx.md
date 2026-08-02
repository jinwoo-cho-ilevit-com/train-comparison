# seqidx — packing 격리의 conv 쪽을 닫는다 (wave 3b)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`, `.plans/notes/packing.md`, `.plans/notes/wire.md`,
> `.plans/research/transformers-varlen-prompt.md`.
> integrate 레인과 동시에 돈다. 파일이 겹치지 않는다.

## 왜

`arch=qwen3_5` 의 packing 격리가 **절반만 닫혀 있다.**

- full_attention 쪽: wave 2 packing 레인이 varlen kwargs 넷
  (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`)을 전달해 닫았다
- linear_attention(Gated DeltaNet) 쪽: **열려 있다.** 그 경로는 `position_ids` 도
  `cu_seq_lens_*` 도 보지 않고 `seq_idx` 를 본다

wire 레인이 핀된 휠에서 확인한 근거(transformers 5.14.1, 이 워크트리 설치본):

- `utils/generic.py:825,839` — `TransformersKwargs.seq_idx: torch.IntTensor | None`,
  설명은 "Sequence index for each token in a flattened packed batch"
- `models/qwen3_5/modeling_qwen3_5.py:492-499` — `Qwen3_5GatedDeltaNet` 이
  `causal_conv1d_fn(..., seq_idx=kwargs.get("seq_idx"))` 로 넘긴다

**네가 직접 그 두 파일을 열어 확인한 뒤 시작한다.** 이 문서에서 옮겨 적지 않는다.

## Owns

```
tests/fixtures/microbatch.sample.json       계약 개정
tests/contract/test_collate_metrics.py      계약 개정
trainbench/collate.py
tests/test_collate.py
```

**이 레인은 계약을 고칠 권한을 명시적으로 받았다.** 그것이 이 레인이 따로 있는 이유다 —
wave 2 packing 레인은 권한이 없어 `boundaryRequests` 로 올렸고, 그 요청이 근거와 함께
`.plans/notes/wire.md` 에 있다. 계약을 고치되 **단언을 약하게 만들지 않는다.**

## 작업 1 — 계약의 all-or-nothing 범위를 좁힌다

지금 `tests/contract/test_collate_metrics.py` 가 `tensors_may_add` **전체**에
`len(present) in (0, len(MAY_ADD))` 를 건다. fixture 의 invariant 산문도 "네 개"라고 적는다.

그 규칙이 있는 이유는 varlen 경로가 **넷 다 아니면 하나도 아니기** 때문이다
(`modeling_flash_attention_utils.py` 의 `all(kwarg is not None for ...)`).
`seq_idx` 는 **다른 경로**이고 그 all-or-nothing 에 묶이지 않는다.

개정:
- fixture 가 varlen 4종과 `seq_idx` 를 **구분해서** 담는다
- 계약 테스트의 all-or-nothing 을 **varlen 4종에만** 건다
- `seq_idx` 에는 그것대로의 불변식을 건다 — 무엇이어야 하는지는 핀된 소스를 읽고 정한다
  (토큰 수와 같은 길이인가, dtype 이 무엇인가, packing 이 꺼져 있으면 없어야 하는가)

**개정 사유를 fixture 의 산문에 적는다.** 다음 사람이 왜 둘로 갈라졌는지 알아야 한다.

## 작업 2 — collate 가 `seq_idx` 를 만든다

`trainbench/collate.py` 의 packed 경로가 `seq_idx` 를 배치에 싣는다.
`PACKED_BOUNDARY_KEYS` 가 무엇을 빼고 무엇을 넘기는지 지금 구조를 읽고 맞춘다.

**arch 별로 다르다는 것을 잊지 않는다.** `qwen3_vl`/`gemma4` 는 `position_ids` 만으로
격리되고 `seq_idx` 를 쓰지 않는다. 필요 없는 곳에 넣어 모델이 거부하면 그것은 회귀다.

## 완료 조건

1. 계약이 varlen 4종에만 all-or-nothing 을 걸고 `seq_idx` 에 자기 불변식을 건다 →
   `infisical run --env=dev -- uv run pytest tests/contract/test_collate_metrics.py -q`
2. `arch=qwen3_5` 의 packed 배치가 `seq_idx` 를 싣는다 →
   `uv run pytest tests/test_collate.py -k seq_idx`
3. `seq_idx` 를 빼면 2번이 죽는다. 계약에서 `seq_idx` 불변식을 빼면 1번이 죽는다.
   **각각 mutation 출력 인용. 사보타주 전에 `co_filename`/`co_firstlineno` 확인**
4. `qwen3_vl`/`gemma4` 경로가 바뀌지 않았다 → 기존 테스트 통과로 보인다
5. 네 게이트. 신설 파일이 없으면 `plan-files` 는 초록이어야 한다
6. **확인 안 함** — `causal_conv1d_fn` 이 실제로 `seq_idx` 를 받아 경계를 지키는지는
   `causal-conv1d` 가 이 호스트에 없어 확인할 수 없다. 파드 질문으로 등록한다

## 하지 않는 것

- `docs/` 전부 — **integrate** 레인이 동시에 돈다
- `trainbench/axes.py` 의 `PackedCollate` — 이 레인은 `collate.py` 쪽만
- 다른 계약 파일 넷
