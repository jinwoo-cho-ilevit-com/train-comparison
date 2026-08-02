# lane-e — 커널 provenance

## Scope

**이미지 digest를 고정해도 커널은 고정되지 않는다.** 이번 리서치에서 나온 것 중 이 스택에
가장 직접적인 위협이다.

`transformers` 공식 문서(`attention_interface`):

> Download and load compiled compute kernels directly from the Hub at runtime with the Kernels
> library... **Kernels automatically register to AttentionInterface upon detection.** You don't
> need to install the FlashAttention package explicitly. Requesting FlashAttention by name also
> falls back to the Hub kernel.

즉 실제 attention 커널이 **실행 시점에 Hub에서** 해석된다. `AGENTS.md`의 "run마다 resolved
torch/framework 버전을 기록한다" 규칙으로는 부족하다. `kernels` 라이브러리에는 `version`
specifier, `revision`, 그리고 `kernels.lock` + `get_locked_kernel()` 핀 메커니즘이 있으나
우리는 쓰지 않는다.

그리고 같은 문서가 `docs/methodology.md` §10.2의 **정확한 메커니즘**을 적어뒀다:

> If the custom `attn_implementation` name is not registered in AttentionMaskInterface,
> Transformers **skips mask creation and passes `attention_mask=None`** to the attention layers.
> Your attention function must handle causal, padding, packing, or sliding-window constraints
> itself, or **those constraints can be silently dropped**.

미등록 커널 + packing = 시퀀스 격리가 조용히 사라진다. 미측정 리스크에서 **이름 붙은 검사 가능
조건**으로 승격시킬 수 있다.

세 번째: 멀티모달은 `attn_implementation`을 sub-config별 dict로 받는다 — "Omit certain backbones
from the dict to use the default attention function (SDPA)". 문자열 하나로 `flash_attention_2`를
주면 **어느 타워에 실제로 걸렸는지가 모델마다 다르다.** `methodology.md` §9의 `kernel_modules`
미측정과 같은 자리이고, Qwen3-VL-Embedding-2B와 gemma-4-E2B가 정확히 이 케이스다.

## Owns

- `trainbench/kernels.py` (신설)
- `docs/methodology.md`

`security: true` — 파드가 런타임에 외부에서 코드를 받는 경로를 다룬다.

## 할 일

### 1. 해석된 커널의 출처를 기록한다

repo + revision이 run 레코드에 남아야 한다. 어느 커널이 실제로 바인딩됐는지가 결과의 일부다.

### 2. mask fn 등록 여부를 검사하고, 미등록이면 packing을 거부한다

`AttentionMaskInterface`에 등록되지 않은 `attn_implementation`은 마스크 생성을 건너뛴다.
그 조합에서 `dataloader.packing=true`를 허용하면 **격리 없는 packing 수치**가 나온다.
거부가 옳고, 거부 문구가 이유를 말해야 한다.

### 3. 파드에서 런타임 커널 fetch를 금지한다

이 저장소는 이미 "학습 데이터를 network volume에서 읽지 않는다"를 규칙으로 갖는다. 커널을
측정 중에 네트워크에서 받는 것은 같은 종류의 오염이다. 사전 다운로드하고 런타임 fetch를
막는다.

### 4. `methodology.md` 정정

**§10.1 / §10.2가 틀렸다.** 코드를 읽으면 답이 나오는 것이었다:

`transformers` 5.14.1이 `position_ids`가 0에서 재시작하면 블록 대각 격리를 스스로 만든다
(`masking_utils.py:735-764 find_packed_sequence_indices`, `:972-975`,
`:718-728 AttentionMaskInterface._global_mapping`). `PackedCollate`가 세 전제를 만족한다
(`masking_utils.py:858-867`: `position_ids is not None`, `attention_mask is None`,
`past_key_values is None`).

| arch | 격리 | 근거 |
|---|---|---|
| `qwen3_vl` / `gemma4` | **됨** | `modeling_qwen3_vl.py:800-814`, `modeling_gemma4.py:1696-1708` |
| `qwen3_5` | **절반** | linear_attention이 `position_ids`를 안 보고 `kwargs.get("cu_seq_lens_q")`를 읽는다 (`modeling_qwen3_5.py:549`) |

부수 사실: sdpa + packing은 `allow_is_causal_skip=False`가 되어 `(1,1,total,total)` bool 마스크를
물리적으로 만든다(`masking_utils.py:520-527`). 8×2048이면 약 268MB. **정확성이 아니라 성능**
이유로 packing에는 `attn=fa2`가 낫다.

**§9(`kernel_modules` 임계값 없음)도 함께 갱신한다** — 멀티모달 per-backbone attention이
그 미측정의 원인 중 하나다.

## Completion criteria

- 해석된 attention 커널의 repo + revision이 run 레코드에 남는다
  → `uv run pytest tests/contract/test_kernel_provenance.py`
- `attn_implementation`이 `AttentionMaskInterface`에 등록돼 있지 않으면 `packing=true`가
  거부되고, 거부 문구가 "마스크가 조용히 사라진다"는 이유를 말한다
  → `uv run pytest tests/test_kernels.py -k mask_registered`
- 파드에서 런타임 커널 fetch가 금지된다
  → `uv run pytest tests/test_kernels.py -k no_runtime_fetch`
- `methodology.md` §9/§10.1/§10.2가 정정되고, 정정 근거가 인용된 파일:줄이다
- 위 각 검사를 되돌리면 죽는다
  → 변이 출력 그대로 인용
- **확인 안 함**: fa2 varlen 경로가 실제로 도는지, `fa3`/`fa4`가 `envs/native`에서 적재되는지,
  transformers 5.5.0(unsloth)과 5.12.1(ms_swift)이 `find_packed_sequence_indices`를 갖는지.
  파드가 답해야 할 것으로 명시한다

## Out of scope

- `cu_seq_lens_*` 전달 자체 — **lane-f** 소유 (`trainbench/collate.py`)
- 어댑터가 반환하는 빌드 지문 — **lane-g** 소유. 경계 `kernel-provenance`에서 맞춘다
- `envs/*/uv.lock`에 `kernels` 핀 추가 — 어느 레인도 env를 소유하지 않는다. 필요하면 보고만 한다
