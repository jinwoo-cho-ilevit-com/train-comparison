# transformers 5.14.1 — varlen / packing / 프롬프트 경로 원문 브리프

작성 2026-08-02. 소비 레인: **split, packing, axes, kernels**.

이 문서의 모든 주장은 이 호스트에 설치된 핀의 소스에서 그대로 인용한다. 인용 블록은
번역·요약·재포맷하지 않았다. 인용에 없는 것은 이 문서에 없다.

## 0. 핀 해석 — 무엇을 읽었는지

`envs/native/uv.lock`이 고정한 세 패키지:

```
1928:name = "transformers"
1929-version = "5.14.1"
1930-source = { registry = "https://pypi.org/simple" }

1228:name = "peft"
1229-version = "0.20.0"

318:name = "datasets"
319-version = "5.0.1"
```

읽은 소스 루트: `/Users/jwcho/Codes/train-comparison/.venv/lib/python3.13/site-packages/`.
그 안의 dist-info가 lock의 버전과 일치한다:

```
/Users/jwcho/Codes/train-comparison/.venv/lib/python3.13/site-packages/datasets-5.0.1.dist-info
/Users/jwcho/Codes/train-comparison/.venv/lib/python3.13/site-packages/peft-0.20.0.dist-info
/Users/jwcho/Codes/train-comparison/.venv/lib/python3.13/site-packages/transformers-5.14.1.dist-info
```

**디코이 확인.** uv 캐시에는 transformers가 11개 버전(4.57.6 / 5.3.0 / 5.5.0 / 5.9.0 /
5.10.2 / 5.11.0 / 5.12.0 / 5.12.1 / 5.13.0 / 5.13.1 / 5.14.1), peft가 3개(0.18.1 /
0.19.1 / 0.20.0), datasets가 4개(4.3.0 / 4.8.5 / 5.0.0 / 5.0.1) 함께 있다. 경로에는
어느 쪽인지 적혀 있지 않다. 그래서 site-packages 의 파일이 **핀 버전의 휠에서 나온
것인지**를 바이트로 대조했다:

```
$ shasum -a 256 $SP/transformers/masking_utils.py \
                ~/.cache/uv/archive-v0/Kur5R2PrM3RUwEti/transformers/masking_utils.py
5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2  .../site-packages/transformers/masking_utils.py
5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2  .../archive-v0/Kur5R2PrM3RUwEti/transformers/masking_utils.py
```

`Kur5R2PrM3RUwEti` 는 `transformers-5.14.1.dist-info` 를 가진 디렉터리다. 즉
site-packages 의 `masking_utils.py` 는 5.13.1 이나 5.12.x 가 아니라 **5.14.1 의 것**이다.

---

## 1. `find_packed_sequence_indices` 와 블록 대각 격리의 전제 3개

### 1.1 함수 본문

`/Users/jwcho/Codes/train-comparison/.venv/lib/python3.13/site-packages/transformers/masking_utils.py:735-764`

```text
def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor | None:
    """
    Find the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

    Args:
        position_ids (`torch.Tensor`)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.

    Returns:
        A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
        pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].

        If the there is only one sequence in each batch item (and we don't compile), then we return `None` indicating
        no packed sequences. This is the same as [[0, 0, 0, 0, 0, 0]] for the example above.
    """
    # What separate different sequences is when 2 consecutive positions_ids are separated by more than 1. So
    # taking the diff (by prepending the first value - 1 to keep correct indexing) and applying cumsum to the result
    # gives exactly the sequence indices
    # Note that we assume that a single sequence cannot span several batch dimensions, i.e. 1 single sequence
    # cannot be part of the end of the first batch dim and the start of the 2nd one for example
    first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(-1)

    # Sadly this is a dynamic control flow, so we cannot enable this check on anything compile related
    if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
        return None

    return packed_sequence_mask
```

읽을 것: 경계는 **연속한 두 position_id 의 차가 1이 아닌 지점**이다. `cu_seqlens` 도,
`seq_lengths` 도, 어떤 별도 kwargs 도 보지 않는다. `position_ids` 하나만 본다.

### 1.2 부르는 자리 — 전제 3개가 여기 있다

`.../transformers/masking_utils.py:858-868`

```text
    # We check the position_ids for potential packed sequence format (only if the 2D attention mask is explicitly None,
    # and we don't have past_key_values, i.e. generally a training setup)
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = inputs_embeds.shape[0]
        # The position ids are sometimes just unsqueezed, without being expanded
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset
```

**전제 3개는 프롬프트가 알고 있던 그대로다. 확인됨.**

1. `position_ids is not None`
2. `attention_mask is None` — 2D 패딩 마스크를 **하나라도 넘기면** 패킹 감지가 통째로
   꺼진다. 빈 마스크도, all-ones 마스크도 안 된다. 이 분기는 `is None` 만 본다.
3. `past_key_values is None`

셋 중 하나라도 깨지면 `packed_sequence_mask` 는 `None` 으로 남고, 아래 1.3 의 격리가
만들어지지 않는다. 그때 pack 은 어텐션에게 한 개의 긴 causal 시퀀스다.

**이 저장소는 세 전제를 모두 만족한다.** `trainbench/axes.py:1405-1419` 의
`PackedCollate.__call__` 이 내는 dict 에는 `attention_mask` 키가 없고,
`position_ids` 는 시퀀스마다 0부터 다시 시작하며(`torch.arange(sequence.numel())`),
학습 forward 에는 캐시가 없다.

### 1.3 격리를 만드는 mask function

`.../transformers/masking_utils.py:182-190`

```text
def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

    return inner_mask
```

`.../transformers/masking_utils.py:971-978`

```text
    # If we detected packing format or blockwise overlay
    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False
    if block_sequence_ids is not None:
        block_sequence_ids = maybe_pad_block_sequence_ids(block_sequence_ids, attention_mask, kv_length, kv_offset)
        mask_factory_function = or_masks(mask_factory_function, blockwise_overlay(block_sequence_ids))
        allow_is_causal_skip = False
```

causal 마스크와 **AND** 되므로 결과는 하삼각 x 블록 대각, 즉 블록별 causal 이다.

### 1.4 저장소 문서와 어긋나는 곳 (packing 레인이 읽을 것)

`docs/methodology.md:520-526` 은 이렇게 적고 있다:

```
packing이 옳으려면 한 pack 안의 시퀀스가 서로의 문맥이 되지 않아야 한다. 이 저장소가
그 방향으로 하는 일은 `position_ids`를 시퀀스마다 0에서 다시 시작시키는 것 하나뿐이다.
블록 대각 어텐션 마스크를 만드는 코드가 없고, 격리를 검사하는 테스트도 없다.

따라서 경계 정보가 어텐션에 도달하지 않는 구성(`attn=sdpa`)에서는 causal 모델의 뒤
시퀀스가 앞 시퀀스를 문맥으로 읽는다.
```

앞 문단("이 저장소에 블록 대각 마스크를 만드는 코드가 없다")은 이 저장소에 대해
맞다. **뒤 문단의 결론은 transformers 5.14.1 에서는 성립하지 않는다** — 격리를
만드는 코드가 저장소에 없어도 `create_causal_mask` 가 `position_ids` 만으로 만든다
(1.2 + 1.3). "position_ids 를 0 에서 다시 시작시키는 것 하나뿐"이 정확히 이 라이브러리가
요구하는 입력의 전부다.

`docs/methodology.md:509-512` 의 "`attn=fa2/fa3/fa4`에서 transformers 5.14.1이
`position_ids`로부터 시퀀스 경계를 유도하는 경로를 타는지는 **확인하지 못했다**" 는
이제 §5 에서 원문으로 확정된다.

---

## 2. 4D 마스크가 오면 조기 반환하는 줄 — 결정 2가 걸린 자리

`.../transformers/masking_utils.py:817-819`

```text
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None, None, None
```

반환 튜플의 세 번째가 `packed_sequence_mask` 자리다. **4D 를 주면 그 자리는 무조건
`None`** 이고, 호출부는 마스크를 그대로 되돌려준다:

`.../transformers/masking_utils.py:936-940`

```text
    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask
```

읽을 것: 4D 를 직접 만들어 넘기면 라이브러리의 패킹 감지·padding 결합·`is_causal` skip
최적화가 전부 우회된다. 즉 **"블록 대각 마스크를 직접 만들지 않는다"는 결정은 이 줄에서
"직접 만들면 라이브러리가 아무것도 도와주지 않는다"로 읽힌다** — 손으로 만든 4D 가
곧 최종 마스크이며, 틀려도 아무 검사가 없다. 반대로 만들지 않으면 §1 이 자동으로 돈다.

주의: 이 분기는 `attention_mask` 가 4D 일 때만이다. 2D 를 넘기면 early exit 이 아니라
전제 2가 깨지는 경로(§1.2)로 들어간다. 두 실패는 증상이 다르다 — 4D 는 "내가 준 것이
그대로 쓰임", 2D 는 "패킹이 조용히 꺼지고 padding 마스크만 적용됨".

---

## 3. `AttentionMaskInterface._global_mapping` — 등록되지 않은 이름은 마스크를 통째로 건너뛴다

`.../transformers/masking_utils.py:718-732`

```text
class AttentionMaskInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
        "flash_attention_2": flash_attention_mask,
        "flash_attention_3": flash_attention_mask,
        "flash_attention_4": flash_attention_mask,
        "flex_attention": flex_attention_mask,
    }


# Global AttentionMaskInterface shared by all models which do not need to overwrite any of the existing ones
ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()
```

여섯 개. 그 밖의 이름이면:

`.../transformers/masking_utils.py:821-827`

```text
    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    # Note: it's not ideal to check the `_global_mapping` attribute instead of the object itself, however otherwise
    # full graph dynamo tracing (i.e. torch.export or compile with `fullgraph=True`) will fail on Python<3.11
    # with `torch._dynamo.exc.Unsupported: 'inline in skipfiles:Mapping.__contains__ | __contains__, skipped
    # according trace_rules.lookup SKIP_DIRS'` -- can be removed when we require Python>=3.11
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None, None, None
```

`early_exit=True` + `attention_mask=None` 이고, §2 의 936-940 에 따라 `create_causal_mask`
가 **`None` 을 반환한다.** 모델 레이어는 `attention_mask=None` 을 받는다.

**kernels 레인이 읽을 것 — 함정 두 개.**

1. 검사 대상이 `_global_mapping` (클래스 전역) 이지 인스턴스가 아니다. 인스턴스에
   `__setitem__` 으로 넣은 로컬 오버라이드(`_local_mapping`)는 이 `in` 검사를
   통과하지 못한다. `GeneralInterface.__setitem__` 은 로컬에만 쓴다:

   `.../transformers/utils/generic.py:1088-1104`

   ```python
       def __setitem__(self, key, value):
           # Allow local update of the default functions without impacting other instances
           self._local_mapping.update({key: value})
   ...
       @classmethod
       def register(cls, key: str, value: Callable):
           cls._global_mapping.update({key: value})
   ```

   즉 커스텀 어텐션을 붙일 때 `AttentionInterface.register(...)` 만 하고
   `AttentionMaskInterface.register(...)` 를 안 하면, **어텐션은 등록되지만 마스크는
   `None` 이 되어 causal 도 padding 도 packing 도 전부 사라진다.** 에러는 나지 않는다.

2. 커스텀 커널 이름을 `attn_implementation` 으로 주는 경로(§8 의
   `kernels-community/flash-attn2` 같은 repo id)도 이 mapping 에 없는 문자열이다.
   그 경로가 마스크를 어떻게 받는지는 이 호스트에서 확인하지 못했다(§11).

---

## 4. sdpa + packing 에서 `(1, 1, total, total)` bool 마스크가 물리적으로 만들어지는 지점

`allow_is_causal_skip` 이 `False` 로 내려가는 곳은 §1.3 의 `masking_utils.py:972-974` 다.
그 값이 `sdpa_mask` 로 전달된다:

`.../transformers/masking_utils.py:980-994`

```text
    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
        device=device,
    )
    return causal_mask
```

`sdpa_mask` 안에서 skip 은 `allow_is_causal_skip` 이 참일 때만 시도된다:

`.../transformers/masking_utils.py:490-518`

```text
    # Potentially pad the 2D mask
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Under specific conditions, we can avoid materializing the mask
    #   1. Causal masks can rely on the `is_causal` argument
    #   2. Bidirectional do not need any further processing (no bias)
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(
        padding_mask, q_length, kv_length, q_offset, kv_offset, local_size
    ):
        return None
    if allow_is_bidirectional_skip and _ignore_bidirectional_mask_sdpa(padding_mask, kv_length, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    q_arange = torch.arange(q_length, device=device) + q_offset
    kv_arange = torch.arange(kv_length, device=device) + kv_offset

    # Actual mask creation
    # Option 1: Fast non-vmap mask creation (default)
    if not use_vmap:
        # Apply mask function element-wise through broadcasting
        attention_mask = mask_function(*_non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange))
        # Expand the mask to match batch size and query length if they weren't used in the mask function
        attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
```

`.../transformers/masking_utils.py:387-389` (docstring)

```
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
```

**결론.** `PackedCollate` 가 `input_ids` 를 `.unsqueeze(0)` 로 한 행에 넣으므로
`batch_size=1`, `q_length = kv_length = total`. 따라서 sdpa + packing 은
**`(1, 1, total, total)` bool 텐서를 실제로 할당한다.** `total=8192` 이면
8192² = 67.1M bool = **64 MiB**, `total=16384` 이면 **256 MiB** — 레이어마다 재계산이
아니라 forward 당 한 번 만들어 모든 레이어가 공유하지만(마스크는 `Model.forward` 에서
한 번 만들어 루프에 넘어간다, §6), 이 텐서는 packing 을 켜는 순간 sdpa 경로에 **없던
메모리**다. 그 값은 이 호스트에서 측정 안 함 — 계산일 뿐이다.

`_non_vmap_expansion_sdpa` 는 브로드캐스트 인덱스를 만들 뿐이고(`masking_utils.py:352-368`),
`use_vmap` 은 사용자가 `or_mask_function`/`and_mask_function` 을 직접 준 경우에만
켜진다(`masking_utils.py:958-969`). packing 단독으로는 `use_vmap=False` 다.

참고로 `eager` 는 언제나 물리화한다 — `eager_mask` 가 `allow_is_causal_skip=False` 를
하드코딩해 `sdpa_mask` 를 부른 뒤 bool 을 float 로 바꾼다:

`.../transformers/masking_utils.py:588-611`

```text
    # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
    _ = kwargs.pop("allow_is_causal_skip", None)
    _ = kwargs.pop("allow_torch_fix", None)
    mask = sdpa_mask(
        ...
        allow_is_causal_skip=False,
        ...
    )
    # only bidirectional masks can be skipped, otherwise we convert bool -> float
    if mask is not None:
        min_dtype = torch.finfo(dtype).min
        # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
        mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
    return mask
```

FA 경로는 다르다 — 4D 를 만들지 않고 2D 를 그대로 돌려주거나 `None` 이다:

`.../transformers/masking_utils.py:645-654`

```text
    if attention_mask is not None:
        # Here we need to slice from the right if using sliding or chunked (for full attention, this is equivalent to doing nothing)
        attention_mask = attention_mask[:, -kv_length:]
        # We only return an actual mask if there is at least 1 padding token AND the length is the same as the kv_length (it can only
        # be smaller, if and only if we use a StaticCache, in which case we need a mask to properly slice k/v), otherwise we return
        # `None` and use `is_causal` in FA2 (note that the attention_mask is a boolean dtype here)
        if attention_mask.shape[1] == kv_length and attention_mask.all():
            attention_mask = None

    return attention_mask
```

즉 `attn=fa2` + packing 이면 마스크는 `None` 이고, 격리는 §5 의 varlen 커널이 한다.
**sdpa 는 마스크로, fa2 는 cu_seqlens 로 — 같은 격리를 서로 다른 비용으로 산다.**

---

## 5. varlen 경로가 켜지는 조건 — kwargs 4개가 **전부** 있어야 하는가

`.../transformers/modeling_flash_attention_utils.py:757-767`

```text
    # We will use `flash_varlen_fn` to prevent cross-example attention and also allow padding free approach under two cases:
    # Case 1. If position ids is provided and the position ids indicate packed sequences, see `_is_packed_sequence`.
    # Case 2. Some models pass directly pre-computed `cu_seqlens` so we don't need to infer it from position ids. It is safe to
    # use `flash_varlen_fn` knowing we already have all necessary the kwargs.
    #
    # NOTE: it is user's responsibility to take care of flattening `position_ids` if that's needed by the model.
    # See #39121 for more information.
    is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))
    is_fa_with_varlen_kwargs = all(
        kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )
```

**답: `all(...)` 이므로 네 개가 전부 있어야 `is_fa_with_varlen_kwargs` 가 참이다.
하나라도 `None` 이면 이 조건은 거짓이다.** 다만 그것이 varlen 이 꺼진다는 뜻은 아니다 —
`or` 로 묶여 있다:

`.../transformers/modeling_flash_attention_utils.py:769-820`

```text
    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_fn
        )
        ...
    # Padding free, i.e. sequences flattened into one total sequence
    elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            q, k, v, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _prepare_from_posids(
                query_states, key_states, value_states, position_ids
            )
        else:
            q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))
        ...
        out = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            **flash_kwargs(max_seqlen_q=max_length_q, max_seqlen_k=max_length_k),
        )
```

읽을 것 세 가지:

- **순서가 중요하다.** `attention_mask is not None` 이 먼저다. 2D 마스크를 같이 주면
  varlen kwargs 를 아무리 잘 채워도 `_upad_input` 이 마스크로부터 경계를 다시 만들고
  네가 준 `cu_seq_lens_*` 는 그 자리에서 덮어써진다. §1.2 의 전제 2와 같은 함정이
  FA 쪽에도 있고, 여기서는 조용히 **다른 경계**를 쓴다.
- `is_fa_with_position_ids` 만으로도 varlen 이 켜진다. 그때 `cu_seq_lens_q` 가 없으면
  `_prepare_from_posids` 가 `position_ids` 에서 만든다.
- `else` 분기(둘 다 거짓)는 `flash_fn` — 일반 dense FA 다. 패킹 격리 없음.

`_is_packed_sequence` 는 **batch_size==1 만** 인정한다:

`.../transformers/modeling_flash_attention_utils.py:534-547`

```text
def _is_packed_sequence(position_ids, batch_size):
    """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False

    increasing_position_sequences = (
        torch.arange(position_ids.shape[1], device=position_ids.device) + position_ids.min()
    )
    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()
```

`_prepare_from_posids` 가 부르는 실제 유도:

`.../transformers/modeling_flash_attention_utils.py:474-493`

```text
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.reshape(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    cu_seq_lens_k = cu_seq_lens_q

    # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
    # for some models (e.g. qwen2-vl).
    max_length_q = cu_seq_lens_q.diff().max()
    max_length_k = max_length_q
```

경계는 `position_ids == 0` 인 자리다. §1.1 의 sdpa 쪽 유도(diff != 1)와 **규칙이
다르다** — 이쪽은 0 만 본다. `PackedCollate` 처럼 매 시퀀스가 0 에서 시작하면 둘이
일치하지만, 0 에서 시작하지 않는 시퀀스가 섞이면 sdpa 와 fa2 가 다른 경계를 본다.

### 5.1 `TransformersKwargs` 에 그 이름들이 있는가

`.../transformers/utils/generic.py:800-839`

```text
class TransformersKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the forward pass of a `PreTrainedModel`.

    Attributes:
        num_items_in_batch (`Optional[torch.Tensor]`, *optional*):
            Number of items in the batch. It is recommended to pass it when you are doing gradient accumulation.
        output_hidden_states (`Optional[bool]`, *optional*):
            Most of the models support outputting all hidden states computed during the forward pass.
        output_attentions (`Optional[bool]`, *optional*):
            Turn this on to return the intermediary attention scores.
        output_router_logits (`Optional[bool]`, *optional*):
            For MoE models, this allows returning the router logits to compute the loss.
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
        position_ids (`torch.LongTensor`, *optional*)
            Indices of positions of each input sequence tokens.
        is_causal (`bool`, *optional*)
            Can be set to False to enable bi-directional attention, i.e. use decoder Attention modules as encoders.
        seq_idx (`torch.IntTensor`, *optional*):
            Sequence index for each token in a flattened packed batch.
    """

    num_items_in_batch: torch.Tensor | None
    output_hidden_states: bool | None
    output_attentions: bool | None
    output_router_logits: bool | None
    cu_seq_lens_q: torch.LongTensor | None
    cu_seq_lens_k: torch.LongTensor | None
    max_length_q: int | None
    max_length_k: int | None
    position_ids: torch.LongTensor | None
    is_causal: bool | None
    seq_idx: torch.IntTensor | None
```

**네 이름 전부 있다. 확인됨.** 덤으로 두 개가 더 있고 이 저장소에 직접 걸린다:

- `is_causal` — "Can be set to False to enable bi-directional attention, i.e. use decoder
  Attention modules as encoders." 임베딩 학습에서 양방향을 쓰려면 이 kwarg 다. config
  쪽 스위치도 있다: `masking_utils.py:917-927` 의
  `if not getattr(config, "is_causal", True): return create_bidirectional_mask(...)`.
- `seq_idx` — "Sequence index for each token in a flattened packed batch." §6 의 Qwen3.5
  conv 경로가 쓴다.

같은 네 이름이 FA 전용 TypedDict 에도 있다 — `modeling_flash_attention_utils.py:568-586`
의 `FlashAttentionKwargs`.

---

## 6. Qwen3.5 linear_attention(Gated DeltaNet)은 `position_ids` 가 아니라 `cu_seq_lens_q` 를 본다

`.../transformers/models/qwen3_5/modeling_qwen3_5.py:538-550`

```text
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                # The chunked FLA kernel takes a single `cu_seqlens` arg; for packed self-attention this matches q-side lengths.
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )
```

그리고 causal conv1d 는 `seq_idx` 를 본다:

`.../transformers/models/qwen3_5/modeling_qwen3_5.py:492-499`

```text
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=kwargs.get("seq_idx"),
                )
```

`Qwen3_5GatedDeltaNet.forward` 의 시그니처에는 `position_ids` 가 아예 없다:

`.../transformers/models/qwen3_5/modeling_qwen3_5.py:441-448`

```text
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
```

**그러므로 Qwen3.5 의 linear_attention 레이어에서 시퀀스 격리는 `position_ids` 로
자동 유도되지 않는다.** `kwargs.get("cu_seq_lens_q")` 가 `None` 이면 FLA chunked 커널은
pack 전체를 한 시퀀스로 스캔하고, `seq_idx` 가 `None` 이면 causal conv 가 경계를 넘어
컨볼브한다. 어느 쪽도 에러를 내지 않는다.

이 레이어가 받는 "마스크"는 4D 가 아니라 2D 패딩 마스크다:

`.../transformers/masking_utils.py:1447-1473`

```text
def create_recurrent_attention_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs,
) -> torch.Tensor | None:
    """Return the 2D padding mask for mamba / linear-attention layers, sized to the local sequence.

    Returns ``None`` (so the consumer skips masking entirely) when any of:
    - the input mask is missing or is already a custom 4D attention mask (no 2D padding signal);
    - the recurrent state already covers past tokens (cached forwards);
    - the mask is all-ones (un-padded batch — the masking multiply would be a no-op), skipped
      only outside trace/compile so the graph specialisation stays stable.

    Otherwise we trim the mask to the trailing ``inputs_embeds.shape[1]`` positions so it aligns
    with the current forward's local sequence and the consumer can multiply directly without
    further slicing.
    """
    if attention_mask is None or attention_mask.ndim != 2:
        return None
    if past_key_values is not None and past_key_values.has_previous_state():
        return None
    if not is_tracing(attention_mask) and torch.all(attention_mask == 1):
        return None
    # ``.contiguous()`` keeps the stride stable across decode steps so ``torch.compile`` doesn't recompile.
    return attention_mask[:, -inputs_embeds.shape[1] :].contiguous()
```

**패킹 정보는 여기 한 글자도 없다.** 이 함수는 패딩만 다룬다.

두 마스크가 레이어 타입별로 갈리는 자리:

`.../transformers/models/qwen3_5/modeling_qwen3_5.py:1193-1220`

```text
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
```

`**kwargs` 가 그대로 레이어로 내려가므로 `cu_seq_lens_q`/`seq_idx` 는 모델 forward 의
kwargs 로 넣으면 GatedDeltaNet 까지 도달한다. 마스크 dict 를 직접 넘겨 `create_*` 를
건너뛰는 길도 열려 있다(`if not isinstance(attention_mask, dict)`).

또 하나: `position_ids` 가 3D/4D 로 확장된다.

`.../transformers/models/qwen3_5/modeling_qwen3_5.py:1179-1191`

```text
        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None
```

2D 를 주면 4채널로 expand 되고 `text_position_ids = position_ids[0]` 가 마스크 생성에
쓰인다 — 즉 packing 감지에 들어가는 것은 우리가 준 2D 그대로다. **3D 를 직접 주면
`shape[0] == 4` 가 아닌 순간 `text_position_ids = None` 이 되어 packing 감지가 꺼진다.**

### 6.1 대조 — Qwen3-VL

`.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:795-815`

```text
        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
```

마스크 dict 없이 `create_causal_mask` 하나. 레이어 타입이 하나뿐이다. **Qwen3-VL 은
§1 의 자동 packing 감지가 그대로 적용되는 가장 단순한 경우다.**

Qwen3-VL 의 vision tower 는 이미 varlen kwargs 를 직접 만들어 넘긴다 — §5 의 Case 2:

`.../transformers/models/qwen3_vl/modeling_qwen3_vl.py:225-242`

```text
        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
```

vision tower 는 FA 가 아니면 **이미지마다 텐서를 split 해서 따로 돈다**. kernels 레인이
읽을 것: 비전 쪽 `attn=sdpa` 대 `attn=fa2` 의 차이는 마스크 하나가 아니라 루프냐 배치냐다.

### 6.2 대조 — gemma-4

`.../transformers/models/gemma4/modeling_gemma4.py:1694-1708`

```text
        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
```

`position_ids` 가 2D 그대로 들어간다(`.../modeling_gemma4.py:1690-1692` 에서
`position_ids.unsqueeze(0)`). 레이어 타입은 full/sliding 둘이고 **둘 다
`_preprocess_mask_arguments` 를 거치므로 packing 감지가 양쪽에 적용된다.**

세 모델의 요약:

| 모델 | 텍스트 레이어 타입 | packing 격리를 만드는 것 |
|---|---|---|
| Qwen3-VL | full_attention 하나 | `create_causal_mask` (position_ids 자동) |
| gemma-4 | full_attention + sliding_attention | 둘 다 `create_*_causal_mask` (position_ids 자동) |
| Qwen3.5 | full_attention + linear_attention | full 은 자동. **linear 은 `cu_seq_lens_q`/`seq_idx` 를 직접 넣어야 한다** |

---

## 7. `attn_implementation` 을 서브컨피그별 dict 로 받는가 — 받는다

`.../transformers/modeling_utils.py:2176-2193`

```text
    def set_attn_implementation(self, attn_implementation: str | dict, allow_all_kernels: bool = False):
        """
        Set the requested `attn_implementation` for this model.

        Args:
            attn_implementation (`str` or `dict`):
                The attention implementation to set for this model. It can be either a `str`, in which case it will be
                dispatched to all submodels if relevant, or a `dict` where keys are the sub_configs name, in which case each
                submodel will dispatch the corresponding value.
            allow_all_kernels (`bool`, optional):
                Whether to load kernels from unverified hub repos, if `attn_implementation` is a custom kernel outside
                of the `kernels-community` hub repository.
        """
        requested_implementation = (
            attn_implementation
            if not isinstance(attn_implementation, dict)
            else attn_implementation.get("", self.config._attn_implementation)
        )
```

빈 문자열 `""` 키가 **최상위(부모) 모델**의 값이다.

서브모듈 배분:

`.../transformers/modeling_utils.py:2229-2241`

```text
                else:
                    sub_implementation = requested_implementation
                    if isinstance(attn_implementation, dict):
                        for subconfig_key in self.config.sub_configs:
                            # We need to check for exact object match here, with `is`
                            if getattr(self.config, subconfig_key) is submodule.config:
                                sub_implementation = attn_implementation.get(
                                    subconfig_key, submodule.config._attn_implementation
                                )
                                break
                    # Check the module can use correctly, otherwise we raise an error if requested attention can't be set for submodule
                    sub_implementation = submodule.get_correct_attn_implementation(sub_implementation)
                    submodule.config._attn_implementation_internal = sub_implementation
```

config 쪽 setter 도 같은 규칙으로 재귀한다:

`.../transformers/configuration_utils.py:401-417`

```text
    @_attn_implementation.setter
    def _attn_implementation(self, value: str | dict | None):
        """We set it recursively on the sub-configs as well"""
        # Set if for current config
        current_attn = getattr(self, "_attn_implementation", None)
        attn_implementation = value if not isinstance(value, dict) else value.get("", current_attn)
        self._attn_implementation_internal = attn_implementation

        # Set it recursively on the subconfigs
        for subconfig_key in self.sub_configs:
            subconfig = getattr(self, subconfig_key, None)
            if subconfig is not None:
                current_subconfig_attn = getattr(subconfig, "_attn_implementation", None)
                sub_implementation = (
                    value if not isinstance(value, dict) else value.get(subconfig_key, current_subconfig_attn)
                )
                subconfig._attn_implementation = sub_implementation
```

**쓸 수 있는 키 이름 (원문):**

```
.../models/qwen3_vl/configuration_qwen3_vl.py:125:    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}
.../models/qwen3_5/configuration_qwen3_5.py:171:    sub_configs = {"vision_config": Qwen3_5VisionConfig, "text_config": Qwen3_5TextConfig}
.../models/gemma4/configuration_gemma4.py:324:    sub_configs = {
```

즉 `attn_implementation={"": "sdpa", "text_config": "flash_attention_2", "vision_config": "sdpa"}`
형태가 유효하다. **axes 레인이 읽을 것: `attn` 축을 문자열 하나로 두면 vision tower 와
text tower 가 같은 구현을 강제로 쓴다.** §6.1 이 보였듯 vision tower 는 sdpa 에서 루프를
돈다 — 그러면 `attn` 축이 텍스트 어텐션이 아니라 비전 루프를 재게 된다.

주의: `from_pretrained` 의 docstring(`modeling_utils.py:4054-4061`)은 `str` 만 적고 있다.
dict 를 받는 것은 `set_attn_implementation` 과 config setter 다. **from_pretrained 인자로
dict 를 넘기는 것이 통하는지는 이 호스트에서 실행해 확인하지 않았다(§11).**

---

## 8. flash-attn 이 없으면 `flash_attention_2` 는 Hub repo 이름으로 바뀌어 런타임에 내려받아진다

`.../transformers/modeling_flash_attention_utils.py:63-71`

```text
# Mapping from flash attention implementations to their kernel fallback repositories.

FLASH_ATTN_KERNEL_FALLBACK = {
    "flash_attention_2": "kernels-community/flash-attn2",
    "flash_attention_3": (
        "kernels-community/aiter-flash-attn" if is_rocm_platform() else "kernels-community/vllm-flash-attn3"
    ),
    "flash_attention_4": "kernels-community/flash-attn4",
}
```

`.../transformers/modeling_utils.py:1985-2036`

```text
        requested_original_flash_attn = False
        if is_flash_attention_requested(requested_attention_implementation=base_implementation):
            # If FA not installed, do not fail but use kernels instead if possible
            for fa_version in FLASH_ATTENTION_COMPATIBILITY_MATRIX.keys():
                # Check whether we have an original FA requested but not available in the env
                if (
                    base_implementation == f"flash_attention_{fa_version}"
                    and not FLASH_ATTENTION_COMPATIBILITY_MATRIX[fa_version]["general_availability_check"]()
                ):
                    requested_original_flash_attn = True
                    break

        if (
            self._supports_flash_attn
            and requested_original_flash_attn
            and is_kernels_available()
            and not is_torch_npu_available()
        ):
            applicable_attn_implementation = FLASH_ATTN_KERNEL_FALLBACK[base_implementation]

            if is_torch_xpu_available() and base_implementation == "flash_attention_2":
                # On XPU, kernels library is the native implementation
                # Disabling this flag to avoid giving wrong fallbacks on errors and warnings
                requested_original_flash_attn = False

            if is_paged:
                applicable_attn_implementation = f"paged|{applicable_attn_implementation}"

        if is_kernel(applicable_attn_implementation):
            try:
                # preload flash attention here to allow compile with fullgraph
                if is_paged:
                    lazy_import_paged_flash_attention(
                        applicable_attn_implementation, allow_all_kernels=allow_all_kernels
                    )
                else:
                    lazy_import_flash_attention(applicable_attn_implementation, allow_all_kernels=allow_all_kernels)

                # log that we used kernel fallback if successful
                if requested_original_flash_attn:
                    logger.warning_once(
                        f"You do not have `flash_attn` installed, using `{applicable_attn_implementation}` "
                        "from the `kernels` library instead!"
                    )
            except Exception as e:
                # raise the proper exception for requested flash attention
                if requested_original_flash_attn:
                    fa_version = int(base_implementation[-1])  # "flash_attention_(2|3|...)"
                    self._flash_attn_can_dispatch(flash_attn_version=fa_version, is_init_check=is_init_check)

                # error properly out if a kernel was specifically requested
                raise e
        else:
            applicable_attn_implementation = self.get_correct_attn_implementation(
                applicable_attn_implementation, is_init_check
            )
```

**kernels 레인이 읽을 것 — 이것은 측정에 직접 걸린다.**

- `flash_attn` 패키지가 없어도 `attn_implementation="flash_attention_2"` 는
  **실패하지 않는다.** `kernels` 가 설치돼 있으면 `config._attn_implementation` 이
  `"kernels-community/flash-attn2"` 라는 **문자열로 바뀐다.** 경고는 `warning_once` 뿐이다.
- 그 이름은 §3 의 `AttentionMaskInterface._global_mapping` 에 **없다.** 즉 §3 의
  826-827 이 걸려 마스크 생성이 통째로 건너뛰어질 가능성이 있다. 이것이 실제로 그렇게
  되는지는 이 호스트에서 실행해 확인하지 못했다(§11) — GPU 도 `kernels` 도 없다.
- 그러므로 **`attn` 축의 라벨이 `fa2` 인 런이 실제로 무엇을 돌렸는지는 실행 로그로만
  안다.** `AGENTS.md` 의 "Record the resolved torch/framework versions per run" 은 이
  경우 버전만으로는 부족하다 — `model.config._attn_implementation` 의 최종 문자열을
  런 레코드에 남겨야 한다. 그 값이 `"flash_attention_2"` 인지
  `"kernels-community/flash-attn2"` 인지가 다른 커널이다.
- 다운로드는 `lazy_import_flash_attention` 이 하고, 그것은 모델 로드 시점에 일어난다.
  **파드가 오프라인이거나 `HF_TOKEN` 이 없으면 여기서 예외가 난다** — 그리고 위 코드는
  `raise e` 한다. 즉 fa2 축이 통째로 죽는다.

`kernels` 없이 `flash_attn` 도 없으면 `is_kernel(...)` 이 거짓이 되어 `else` 로 가고
`get_correct_attn_implementation` 이 sdpa/eager 로 폴백한다(같은 함수 2037-2040).

---

## 9. 프로세서에 chat_template 이 없으면 `apply_chat_template` 은 `ValueError` 를 낸다

`.../transformers/processing_utils.py:2008-2024`

```text
        processor_kwargs = processor_kwargs or {}

        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify'
                    " which one to use by passing the `chat_template` argument. Available templates are: "
                    f"{', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
```

**폴백은 없다.** 토크나이저의 템플릿으로 넘어가지도, 기본 ChatML 을 쓰지도 않는다.
`AGENTS.md` 가 기록한 1차 Phase 0 실패 — `google/gemma-4-E2B` 가 base 체크포인트라
`chat_template.jinja` 가 없고 세 프레임워크가 `apply_chat_template` 에서 동일하게
죽은 것 — 이 줄이 그 지점이다.

빠져나가는 길은 하나다: **호출부에서 `chat_template=` 인자를 직접 준다.** `if chat_template
is None:` 밖으로 나가므로 프로세서에 템플릿이 없어도 된다. 즉 프롬프트 템플릿을
저장소가 소유하면(`~/.claude/rules/base/code-craft.md` 의 "Externalize prompts": LLM
프롬프트는 `.md` 파일에) base 체크포인트도 `-it` 없이 돌릴 수 있다. 이 저장소는
임베딩 학습이므로 chat 템플릿이 필요한지 자체가 설계 질문이다 — split 레인이 답할 것.

주의: `self.chat_template` 이 dict 인데 `"default"` 키가 없으면 **다른** ValueError 다.
두 실패의 메시지가 다르므로 로그로 구분된다.

---

## 10. Qwen 프로세서의 `image_grid_thw` — 이미지당 정확히 1행

`qwen3_vl` 디렉터리에는 image processor 가 없다. auto mapping 이 `qwen2_vl` 것을 쓴다:

`.../transformers/models/auto/image_processing_auto.py:138`

```
            ("qwen3_vl", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}),
```

그 클래스가 grid 를 만드는 자리:

`.../transformers/models/qwen2_vl/image_processing_qwen2_vl.py:194-230`

```text
            batch_size, channel = patches.shape[:2]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = patches.reshape(
                batch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # [batch, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
            patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)

            flatten_patches = (
                patches.unsqueeze(6)
                .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
                .reshape(
                    batch_size,
                    grid_h * grid_w,
                    channel * temporal_patch_size * patch_size * patch_size,
                )
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[1, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )
```

**확인된 것:**

- `image_grid_thw` 는 프로세서 출력에 **있다** (`model_input_names` 에도 있다:
  `.../image_processing_qwen2_vl.py:106` `model_input_names = ["pixel_values", "image_grid_thw"]`).
- 행은 `[[1, grid_h, grid_w]] * batch_size` — **이미지 한 장당 정확히 1행**이고
  t 차원은 항상 `1` 이다(이미지 경로). 즉 `image_grid_thw.shape == (num_images, 3)`.
- `grid_h = resized_height // patch_size` 이므로 **이미지마다 값이 다르다.** 크기가 같은
  이미지끼리 묶여 처리되고(`processed_images_grouped[shape]`) `reorder_images` 가 원래
  순서로 되돌린다. `pixel_values` 는 `torch.cat(..., dim=0)` 로 이미지 경계 없이 이어
  붙는다 — 경계를 아는 유일한 것이 `image_grid_thw` 다.
- `pixel_values` 의 행 수는 `sum(grid_h * grid_w)` 이지 이미지 수가 아니다.

프로세서가 텍스트에 이미지 토큰을 채우는 규칙도 같은 텐서를 쓴다:

`.../transformers/models/qwen3_vl/processing_qwen3_vl.py:76-79`

```text
    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens
```

**packing 레인이 읽을 것.** `trainbench/axes.py:183` 의
`IMAGE_PAYLOAD_KEYS = ("pixel_values", "image_grid_thw", "image_position_ids")` 는 이
구조와 맞다. 다만 `PackedCollate.__call__`(`trainbench/axes.py:1405-1419`)이 내는 dict
에는 이 키들이 **없다** — pack 은 `input_ids`/`position_ids`/`cu_seqlens`/`seq_lengths`
넷뿐이다. `docs/methodology.md:540-548` 이 적은 "packing 과 pretokenize 는 이미지를
버린다"와 일치한다. 이미지를 살리려면 `image_grid_thw` 를 이미지 순서대로 concat 하고
텍스트 쪽 이미지 토큰 수를 `prod() // merge_size**2` 로 맞춰야 한다 — 위 인용이 그
산식이다.

---

## 11. 이 호스트에서 확정하지 못한 것 — 파드/이미지가 답해야 할 질문

이 호스트는 macOS(darwin)이고 GPU 가 없다. 아래는 **소스를 읽어도 답이 나오지 않고
실행이 필요한** 것들이다. 추측은 적지 않았다.

1. `flash_attn` 이 없는 파드에서 `attn_implementation="flash_attention_2"` 로 모델을
   로드하면 `model.config._attn_implementation` 의 최종 문자열은 무엇인가 —
   `"flash_attention_2"` 인가 `"kernels-community/flash-attn2"` 인가.
2. 그 문자열이 `"kernels-community/flash-attn2"` 일 때 §3 의
   `masking_utils.py:826` 검사에 걸려 `create_causal_mask` 가 `None` 을 반환하는가.
   반환한다면 packing 격리는 §5 의 varlen 경로만이 담당하는가, 아니면 사라지는가.
3. `kernels` 가 파드 이미지에 설치돼 있는가. 없으면 fa2 요청은 sdpa/eager 로 조용히
   폴백하고, 있으면 모델 로드 시점에 Hub 다운로드가 일어난다. 어느 쪽인가.
4. 오프라인 파드(또는 `HF_TOKEN` 없는 파드)에서 3번 다운로드가 실패하면 `raise e`
   (`modeling_utils.py:2036`)로 fa2 축이 통째로 죽는가. 재시도/캐시 경로는 무엇인가.
5. `from_pretrained(..., attn_implementation={"text_config": ..., "vision_config": ...})`
   가 실제로 받아들여지는가. `set_attn_implementation` 과 config setter 는 dict 를
   받지만 `from_pretrained` 의 docstring 은 `str` 만 적는다.
6. Qwen3.5 의 linear_attention 레이어에 `cu_seq_lens_q` 를 넣지 않은 packing 런과
   넣은 런의 임베딩이 다른가. FLA chunked 커널(`fla.ops.gated_delta_rule`)이 파드에
   설치돼 있는가, 아니면 `torch_chunk_gated_delta_rule` 폴백인가 — 후자가
   `cu_seqlens` 를 받는지 이 호스트에서 확인하지 못했다.
7. `causal_conv1d_fn` 이 파드에 있는가. 있으면 `seq_idx` 없이 pack 을 돌렸을 때 conv
   가 경계를 넘는가.
8. sdpa + packing 의 `(1, 1, total, total)` bool 마스크가 실제 피크 메모리를 얼마나
   올리는가. 64 MiB(total=8192) / 256 MiB(total=16384) 는 계산이지 측정이 아니다.
9. 같은 pack 을 `attn=sdpa` 와 `attn=fa2` 로 각각 인코딩했을 때 임베딩이 일치하는가
   — sdpa 는 블록 대각 마스크로, fa2 는 cu_seqlens 로 같은 격리를 만들어야 한다.
   불일치하면 §5 의 경계 유도 규칙 차이(diff!=1 대 ==0)를 먼저 의심한다.
10. `google/gemma-4-E2B` 의 프로세서를 실제로 로드했을 때 `processor.chat_template` 이
    `None` 인가. §9 의 두 ValueError 중 어느 쪽이 나는가.
11. Qwen3-VL-Embedding-2B 의 프로세서가 실제로 `Qwen2VLImageProcessor` 로 해석되는가
    (auto mapping 은 아키텍처 `qwen3_vl` 기준이고, 체크포인트의
    `preprocessor_config.json` 이 다른 클래스를 지정할 수 있다).
