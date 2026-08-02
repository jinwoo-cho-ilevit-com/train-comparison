# packing — packing 격리 + 프롬프트 (wave 2)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`.
> **리서치 필독**: `.plans/research/transformers-varlen-prompt.md` — 이 레인의 두 작업 모두
> 그 브리프가 원문으로 확정해 두었다. 인용을 그대로 쓰되 **네가 파일을 열어 확인한다.**
> split 과 measure 가 이미 머지됐다.

## 목표

(1) Qwen instruction prompt 가 두 번 들어가는지 **토큰화로 확정**하고,
(2) `arch=qwen3_5` 에서 packing 격리를 실제로 만든다.

## Owns

```
trainbench/collate.py            split 이 만든 것. 의미는 이 레인 소유
trainbench/prompt.py
configs/model/
tests/test_prompt.py
tests/test_collate.py            ⊕ 필요하면 신설
docs/open-verdicts.json          항목 1개만
```

## 작업 1 — Qwen instruction prompt 가 두 번 들어가는가

`configs/model/qwen3_vl_emb_2b.yaml` 의 `instruction_prompt` 가 템플릿 출력 **앞에** 붙는데,
`trainbench/prompt.py` 는 user 메시지 하나만 넘긴다. 그러면 체크포인트의 `chat_template.jinja`
가 "system 메시지가 없다"고 판단해 자기 `default_system_message` 를 삽입하는데,
**그것이 같은 문자열이다.** (2026-08-02 Hub 에서 템플릿을 직접 읽어 확인)

결과 쿼리:
```
Represent the user's input.<|im_start|>system\nRepresent the user's input.<|im_end|>...
```
앞 사본은 특수 토큰 밖에 있다.

**토큰화는 하지 않았다. 실제 횟수는 미측정이다.**

사실이면 지금까지의 Qwen 프로브 전부와 앞으로의 모든 측정이 **다른 프롬프트를 잰 것**이고
시퀀스 길이가 바뀌므로 tokens/s 의 분모가 바뀐다.

세 갈래:
1. **토큰화해서 확정한다.** HF 다운로드만 필요하고 **GPU 는 필요 없다**
2. 결정한다 — 프롬프트를 system 메시지로 넘길 것인가, `self.prompt` 접두를 뺄 것인가
3. 어느 쪽이든 **Qwen 두 모델의 시퀀스 길이가 바뀐다.** 이전 Phase 0 결과 대비 델타로
   드러날 것이므로 그것을 보고한다

이것이 `docs/open-verdicts.json` 의 `qwen3-vl-query-prompt-may-go-in-twice` 다.
**owner 가 미배정이었고 이제 이 레인이다.** 앵커 테스트 이름은 원장에 적혀 있다:
`tests/test_prompt.py::test_the_query_instruction_prompt_appears_once_in_a_templated_row`.
그 이름으로 만든다.

중복이 **아니면** 그것도 답이다 — 근거를 같은 자리에 기록하고 닫는다.
닫는 것은 리뷰어 행위이고 런을 인용해야 한다. **CPU 증거로 파드가 답할 항목을 닫지 않는다.**

관련 사실 (리서치 §9): 프로세서에 `chat_template` 이 없으면 `apply_chat_template` 은
`ValueError` 를 낸다. `google/gemma-4-E2B` 는 base 체크포인트라 없다 — 그래서
`model.prompt_format` 이 `chat_template`/`raw` 두 값을 갖는다.

## 작업 2 — packing 격리, 리서치가 범위를 줄였다

**직접 블록 대각 마스크를 만들지 않는다.** transformers 5.14.1 이 `position_ids` 가 0에서
재시작하면 스스로 만든다(`masking_utils.py`, 리서치 §1). `PackedCollate` 는 그 전제 셋을
이미 만족한다. 그리고 **4D `attention_mask` 를 주면 조기 반환해서 fa2 varlen 경로가 꺼진다**
(리서치 §2). 그것이 결정 2 다.

arch 별로 다르다 (리서치 §6, §6.1, §6.2):
- `qwen3_vl` / `gemma4` — 격리된다
- **`qwen3_5` 는 절반** — full_attention 은 격리되지만 linear_attention(Gated DeltaNet)이
  `position_ids` 를 보지 않고 `kwargs.get("cu_seq_lens_q")` 를 읽는다. 우리는 그것을 넘긴 적이 없다.
  순수 torch fallback 은 `cu_seqlens` 를 `**kwargs` 로 삼키고 무시한다

**그러므로 일은 `arch=qwen3_5` 에만 있다.** `model(**tensors)` 에 넷을 넘긴다:
```
cu_seq_lens_q  cu_seq_lens_k  max_length_q  max_length_k
```

**넷 다 아니면 하나도 아니다** — `modeling_flash_attention_utils.py:765-767` 이
`all(kwarg is not None for ...)` 이다 (리서치 §5 에 원문). 네 이름은 `TransformersKwargs` 의
정식 멤버다 (`utils/generic.py:800-839`, 리서치 §5.1).

대안: 그 arch 에서 `packing=true` 를 거부한다. 어느 쪽이든 **침묵하지 않는다.**

지금 `cu_seqlens` 는 `PACKED_BOUNDARY_KEYS` 로 배치에서 빠져 pooling 으로만 간다.
그 키의 주석 "모델이 거부한다"는 **철자 `cu_seqlens` 에 대해서만 참이다.**

부수 사실 — 문서에 남긴다: sdpa + packing 은 `allow_is_causal_skip=False` 가 되어
`(1,1,total,total)` bool 마스크를 물리적으로 만든다. 8×2048 이면 268,435,456 bytes.
**정확성이 아니라 성능 때문에** packing 은 `attn=fa2` 를 선호한다.
(§10 정정 자체는 **kernels 레인**이 wave 1 에서 했다. 중복하지 않는다.)

## measure 와의 경계

`MicroBatch` 가 무엇을 싣는지는 경계 `collate-metrics`
(`tests/contract/test_collate_metrics.py`, `tests/fixtures/microbatch.sample.json`)가 못박았다.
카운트 **지점**은 네 것이고 카운트의 **정의**는 measure 의 것이다.
필드를 늘려야 하면 `boundaryRequests` 로 요청한다. **계약 파일을 고치지 않는다.**

## 완료 조건

1. Qwen instruction prompt 가 템플릿된 행에 **정확히 한 번** 나온다 →
   `uv run pytest tests/test_prompt.py -k appears_once`
2. 이중 삽입 질문이 **토큰화로 결착**되고 출력이 보고됐다 (지금까지는 소스 읽기뿐이고
   실제 횟수는 미측정이다). 결착 방향과 그것이 시퀀스 길이에 준 변화를 적는다
3. `arch=qwen3_5` 에 kwargs 넷이 전달되거나, 그 arch 에서 `packing=true` 가 거부된다 →
   `uv run pytest tests/test_collate.py -k varlen`
4. `PackedCollate` 출력에 대해 `create_causal_mask` 가 블록 대각을 돌려준다 (CPU, 작은 config) →
   `uv run pytest tests/test_collate.py -k isolation`
5. **넷 중 하나를 빼면 varlen 경로가 꺼지는 것이 테스트로 드러난다.** mutation 출력 인용
6. 각 검사를 되돌리면 죽는다. **사보타주 전에 `co_filename`/`co_firstlineno` 확인** —
   이 저장소에서 `PackedCollate.__call__` 사보타주를 클래스 본문 앞쪽에 넣어 나중 정의가 이긴
   전례가 있다 (`HAZARDS.md §3`)
7. 네 게이트. `verdicts-closed` 가 4 → 3 이 되면 `shrank` 로 BLOCK 된다. **정상이고 머지 단계가
   처리한다. `docs/audit-baseline.json` 을 건드리지 않는다**
8. **확인 안 함** — 파드 질문으로 등록: 실제 GPU 에서 fa2 varlen 경로가 도는가.
   확인 방법까지 적는다 — `attn=fa2 dataloader.packing=true` 로 forward 한 번,
   패딩 배치와 pooled embedding 을 대조

## 하지 않는 것

- `docs/methodology.md` §9/§10 정정 — **kernels** 레인 (wave 1, 이미 끝났다)
- `attn_implementation` 의 mask fn 등록 검사 — **kernels** 레인
- `scripts/bench.py` — split 이 정리했다
- `trainbench/metrics/` 의 카운터 정의 — **measure** 레인
- `trainbench/probe/*` — **probe**/**adapters** 레인
- `docs/open-verdicts.json` 의 다른 세 항목
