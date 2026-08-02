# lane-h — 축 구현

## Scope

`axis-values` 감사가 지금 **36/53**이고 단일값 그룹이 셋이다 — `kernel 1/4`, `precision 1/3`,
`train.offload 1/4`. 축이 한 값만 받으면 그 축으로 ablation을 돌려도 라벨만 다른 같은 런이 나온다.

**패키지는 전부 이미 있다.** `bitsandbytes`(`envs/native/pyproject.toml:33`),
`transformer-engine`(:31), `deepspeed`(:39), `nvidia-dali`(:41), `liger-kernel`(:22).
**의존성 공백이 아니라 코드 공백이다.**

## Owns

- `trainbench/axes.py`
- `configs/optim/`, `configs/precision/`, `configs/train/`, `configs/parallel/`,
  `configs/dataloader/`, `configs/peft/`, `configs/kernel/`

## 선행

**lane-c가 먼저 끝나야 한다.** `applied.py`의 네 축이 config 값과 같아질 수 없는 상태에서
축을 구현하면 `assert_matches`가 런을 거부한다 — 구현했는데 측정이 안 열린다.

## 할 일

### 거부 지점 (전부 `trainbench/axes.py`)

| 축 값 | 거부 지점 | 필요한 것 |
|---|---|---|
| `optim=adamw_8bit` | `:954-960` | bitsandbytes 8-bit optimizer |
| `precision=mxfp8` / `nvfp4` | `:695-699` `step_context` | TE recipe로 forward를 감싸는 컨텍스트 |
| `train.offload=optimizer/param/both` | `:652-657` `assemble` | `deepspeed.initialize` |
| `parallel=ddp` / `fsdp2` | `:658-662` | 프로세스 그룹 + wrapper |
| `parallel=zero2` / `zero3` | `:652-657` | `deepspeed.initialize` |
| `dataloader.backend=dali` | `:1023-1027` | DALI iterator |
| `peft=qlora` | `:778-783` `_peft` + `:591-598` `load_kwargs` | 4-bit base + adapter, 게이트 둘 |
| `kernel=liger` | `:318-324` (arch별 entrypoint 미기록), `:316`(gemma4는 `LIGER_UNSUPPORTED`) | **qwen3_5만** 가능. `LIGER_ENTRYPOINTS`에 이름은 있으나 **철자가 검증 안 됨**(`:105-116`이 그렇게 적어둠) — 이미지 안에서 `dir()` 한 번이면 확정 |

`kernel=fla`는 코드 작업이 아니다 — qwen3_5 + CUDA + causal_conv1d가 있으면 자동으로 통과한다
(`:362-374`). 이미지/파드 문제다.

### `kernel=kernels_hub` 제거 (결정 6)

`:522-542`에서 무조건 거부되고 있다. transformers 5.14.1의 두 진입점이 **모두 모델 객체를
요구**하는데 `axes.patch`는 모델 생성 **전에** 돈다. 그것은 우연이 아니라 설계다 —
`scripts/bench.py`가 "`kernel`/`attn`은 모델이 존재한 뒤에는 바꿀 수 없다"고 명시한다.

**축 값을 버린다.** 스키마 Literal에서 제거하고, 버린 이유(진입점이 모델을 요구, 우리 적용
지점은 생성 전)를 남긴다. `kernel`은 `none`/`liger`/`fla` 셋이 된다.

## Completion criteria

- `axis-values`에 단일값 그룹이 0이다
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- 각 축 값에 대해, 적용 지점을 비우면 `axis-values`가 그 값을 이름으로 지목한다 (감사가
  공허해지지 않는 것을 실측으로 확인한다 — 이전에 `PackedCollate.__call__`에 raise를 넣어
  32/46 → 30/46으로 내려가는 것을 확인한 선례가 있다)
  → 변이 출력 그대로 인용
- `kernels_hub`가 스키마에서 제거되고 이유가 남는다
  → `infisical run --env=dev -- uv run pytest tests/test_config.py -k kernel`
- `kernel=liger`의 qwen3_5 entrypoint 철자가 **검증됐거나**, 검증되지 않았음이 명시된다
  → 확인 안 함이면 파드가 답할 질문으로 등재
- 위 각 검사를 되돌리면 죽는다
  → 변이 출력 그대로 인용
- **확인 안 함**: 이 호스트에 bitsandbytes / deepspeed / TE / DALI 중 어느 것도 없다. 구현은
  스텁으로만 검증되고, 실제로 도는지는 파드가 답한다. 축마다 파드가 무엇을 보여야 하는지 적는다

## Out of scope

- `trainbench/applied.py` — **lane-c** 소유. 이 레인은 capture가 이미 읽을 수 있는 상태를 전제한다
- `configs/model/` — **lane-f** 소유
- `configs/run/` — **lane-b** 소유
- `envs/*/pyproject.toml` — 어느 레인도 소유하지 않는다. 패키지는 이미 다 있다
