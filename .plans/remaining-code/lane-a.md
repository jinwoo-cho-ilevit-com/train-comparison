# lane-a — tevatron 적재

## Scope

Phase 0 2차에서 tevatron 3셀이 전부 같은 자리에서 죽었다:

```
'Gemma4Config' object has no attribute 'pad_token_id'
'Qwen3_5Config' object has no attribute 'pad_token_id'
'Qwen3VLConfig' object has no attribute 'pad_token_id'
```

1차의 `ModuleNotFoundError: No module named 'peft'`보다 **더 깊은 실패**다 — 가중치는 적재됐고
그다음 줄에서 죽는다.

핀 고정 소스(`envs/tevatron/uv.lock`이 `dd06310`을 지목,
`~/.cache/uv/git-v0/checkouts/af8e1386372d71f4/dd06310/src/tevatron/retriever/modeling/encoder.py`):

```python
166  base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
167  if base_model.config.pad_token_id is None:
168      base_model.config.pad_token_id = 0
```

`transformers 5.14.1`이 **composite(멀티모달) config에서만** `pad_token_id`를 최상위에서
옮겼다. 실측:

| config | 최상위 | `get_text_config()` |
|---|---|---|
| `Gemma4Config` / `Qwen3VLConfig` / `Qwen3_5Config` | 없음 | 있음 |
| `Qwen2Config` / `Qwen3NextConfig` (비교용) | 있음 | 있음 |

이 연구의 세 모델이 전부 composite다. tevatron이 다른 모델에서 도는 이유이기도 하다.

## Owns

- `trainbench/probe/tevatron.py`

## 할 일

`_load`에서 config를 만들어 `pad_token_id`를 세팅하고 `hf_kwargs`로 넘긴다. `encoder.py:166`이
`**hf_kwargs`를 그대로 전달하므로 도달한다. composite config 인스턴스에 `setattr`가 통하는 것은
실측 확인됨 (transformers 5.x `@strict`가 막지 않는다).

**모델이 읽지 않는 shim임을 코드에 남긴다** — 실제로 쓰이는 값은 text sub-config 쪽이고,
최상위에 심는 것은 상류의 `getattr` 누락을 우회하기 위한 것이다. 그 사실이 주석에 없으면
다음 사람이 왜 여기만 이러는지 다시 판다.

## Completion criteria

- composite config 세 종에 대해 `pad_token_id`가 심어진 config가 `hf_kwargs`로 전달된다
  → `infisical run --env=dev -- uv run pytest tests/test_probe.py -k tevatron`
- 그 shim을 제거하면 테스트가 죽는다 (부수기 증거를 보고한다)
  → 변이 출력 그대로 인용
- [pod] tevatron 3셀이 `dense_model_load`를 통과한다 (12/18 → 15/18). 실패한다면 이번엔
  **체크포인트에 대한 답**이어야 한다
  → verdict: ____  by: ____  at: ____

## Out of scope

- tevatron의 `forward` 시그니처 — `DenseModel.forward`가 인코딩~대조손실을 전부 한다는 사실과
  그것을 어댑터 경계에서 어떻게 표현할지는 **lane-g** 소유
- `envs/tevatron/` 의존성 — 이미 `peft>=0.20`이 들어가 있다
