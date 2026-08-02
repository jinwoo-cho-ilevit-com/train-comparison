# lane-f — packing + 프롬프트

## Scope

두 가지. 하나는 **이미 난 12/18의 의미를 바꾸고**, 하나는 packing 수치의 의미를 결정한다.

## Owns

- `trainbench/collate.py` (lane-d가 `scripts/bench.py`에서 분리해 만든 것)
- `trainbench/prompt.py`
- `configs/model/`

## 할 일

### 1. Qwen 지시 프롬프트가 두 번 들어갈 수 있다

`scripts/bench.py:254`가 `self.prompt = config.model.instruction_prompt or ""`,
`:318`이 그것을 템플릿 결과 **앞에** 붙인다. 그런데 `trainbench/prompt.py:126-130`은 user 메시지
하나만 넘기므로 체크포인트의 chat_template가 system 메시지가 없다고 판단해 자기
`default_system_message`를 넣는다 — 그리고 `configs/model/qwen3_vl_emb_2b.yaml`의
`instruction_prompt`가 **같은 문자열**이다.

Hub에서 체크포인트의 `chat_template.jinja`를 직접 읽어 확인한 사실이다. **토큰화는 하지 않았으므로
실제 횟수는 측정 안 함.**

사실이면 Qwen 두 모델의 모든 probe와 앞으로의 모든 측정이 다른 프롬프트를 잰 것이다.

- 토큰화로 사실을 먼저 확정한다 (HF 다운로드만, GPU 불필요)
- 결정: 프롬프트를 system 메시지로 넘길 것인가, `self.prompt` 접두를 없앨 것인가
- 어느 쪽이든 **Qwen 두 모델의 시퀀스 길이가 바뀐다** — 그것이 앞선 Phase 0 결과와의 차이로
  드러나므로 그 사실을 보고한다

### 2. packing 격리 — 범위가 조사로 축소됐다

**`transformers` 5.14.1이 이미 블록 대각 격리를 만든다.** `position_ids`가 0에서 재시작하면
`masking_utils.py:735-764 find_packed_sequence_indices`가 세그먼트를 잡고 `:972-975`가 causal
마스크와 AND하며, `:718-728`의 매핑으로 sdpa·eager·flex·flash_attention_2/3/4 전부에 적용된다.

`PackedCollate`가 세 전제를 만족한다 (`masking_utils.py:858-867`).

| arch | 격리 | 근거 |
|---|---|---|
| `qwen3_vl` / `gemma4` | **됨** | `modeling_qwen3_vl.py:800-814`, `modeling_gemma4.py:1696-1708` |
| `qwen3_5` | **절반** | full_attention은 격리. **linear_attention(Gated DeltaNet)이 `position_ids`를 안 본다** — `modeling_qwen3_5.py:549`가 `kwargs.get("cu_seq_lens_q")`를 읽는데 우리가 안 넘긴다. 순수 torch fallback(`:248-258`)은 `cu_seqlens`를 `**kwargs`로 삼키고 무시한다 |

**블록 대각 마스크를 직접 만들면 해롭다** — 4D `attention_mask`가 non-None이면
`masking_utils.py:855-856`에서 조기 반환해 fa2 varlen 경로가 꺼진다.

**할 일은 `arch=qwen3_5`에만 있다**: `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`,
`max_length_k` 네 개를 `model(**tensors)`에 전달한다. **넷 다 아니면 아무것도 아니다**
(`modeling_flash_attention_utils.py:766`이 `all(kwarg is not None for ...)`). 또는 그 arch에서
`packing=true`를 거부한다.

지금 `cu_seqlens`는 `PACKED_BOUNDARY_KEYS`로 배치에서 빠져 pooling에만 간다
(`bench.py:484`, `:585`). `PACKED_BOUNDARY_KEYS`의 주석 "모델이 거부한다"는 `cu_seqlens`라는
**철자**에만 참이고, 정식 네 이름은 `TransformersKwargs`의 합법 멤버다
(`utils/generic.py:800-841`).

## Completion criteria

- Qwen 지시 프롬프트가 템플릿된 행에 **정확히 한 번** 나온다
  → `uv run pytest tests/test_prompt.py -k appears_once`
- 프롬프트 이중 삽입 여부를 **토큰화로 확정**하고 그 출력을 보고한다 (지금은 소스 읽기까지만
  했고 실제 횟수는 미측정)
- `arch=qwen3_5`에 네 kwarg가 전달되거나, 그 arch에서 `packing=true`가 거부된다
  → `uv run pytest tests/test_collate.py -k varlen`
- `create_causal_mask`가 `PackedCollate` 출력에 대해 블록 대각을 돌려준다 (CPU, 작은 config)
  → `uv run pytest tests/test_collate.py -k isolation`
- 네 kwarg 중 하나만 빼도 varlen 경로가 안 도는 것이 테스트로 고정된다
  → 변이 출력 그대로 인용
- 위 각 검사를 되돌리면 죽는다
  → 변이 출력 그대로 인용
- **확인 안 함**: fa2 varlen 경로가 실제 GPU에서 도는지. 파드가 답할 것으로 명시한다 —
  `attn=fa2 dataloader.packing=true` forward 하나로 padded 배치와 pooled 임베딩을 비교

## Out of scope

- `methodology.md` §10.1/§10.2 정정 — **lane-e** 소유
- `attn_implementation`의 mask fn 등록 검사 — **lane-e** 소유
- `MicroBatch`가 무엇을 싣는지 — 경계 `collate-metrics`에서 lane-d와 맞춘다
- `scripts/bench.py` — **lane-d** 소유
