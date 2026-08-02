# axes — 축 구현 (wave 2)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`.
> **리서치 필독**: `.plans/research/axis-libraries.md` — 각 축의 적용 지점 시그니처와
> **이 호스트에서 확정 불가한 것 목록**이 거기 있다.
> capture 가 이미 머지됐다. 그것 없이 축을 구현하면 `assert_matches` 가 런을 거부한다 —
> 구현했는데 측정이 안 열린다.

## 목표

지금 거부되는 축 값들을 구현해서 `axis-values` 의 단일값 그룹을 0으로 만든다.
그래야 ablation 이 라벨만 다른 재실행이기를 그만둔다.

## Owns

```
trainbench/axes.py
configs/optim/  configs/precision/  configs/train/  configs/parallel/
configs/dataloader/  configs/peft/  configs/kernel/
tests/test_axes.py
```

## 기준선

`axis-values` 는 지금 **36/53**, 단일값 그룹 셋: `kernel 1/4`, `precision 1/3`,
`train.offload 1/4`.

**패키지는 전부 이미 있다** — `bitsandbytes`, `transformer-engine`, `deepspeed`,
`nvidia-dali`, `liger-kernel` 이 `envs/native/pyproject.toml` 에 들어 있다.
**의존성 공백이 아니라 코드 공백이다.**

## 거부 지점 — 전부 `trainbench/axes.py`

| 축 값 | 거부 지점 | 필요한 것 |
|---|---|---|
| `optim=adamw_8bit` | `_optimizer` | bitsandbytes 8-bit optimizer |
| `precision=mxfp8` / `nvfp4` | `step_context` | forward 를 TE recipe 로 감싸는 컨텍스트 |
| `train.offload=optimizer/param/both` | `assemble` | `deepspeed.initialize` |
| `parallel=ddp` / `fsdp2` | `assemble` | process group + wrapper |
| `parallel=zero2` / `zero3` | `assemble` | `deepspeed.initialize` |
| `dataloader.backend=dali` | `_dataloader` | DALI iterator |
| `peft=qlora` | `_peft` + `load_kwargs` | 4-bit base + adapter. **게이트가 둘이다** |
| `kernel=liger` | `LIGER_ENTRYPOINTS` | **qwen3_5 만.** gemma4 는 `LIGER_UNSUPPORTED`, qwen3_vl 은 엔트리포인트 미기록 |

정확한 줄 번호는 **네가 직접 grep 해서 확인한다.** 이 문서의 줄 번호를 옮기지 않는다 —
wave 0/1 이 파일을 바꿨을 수 있다.

- **`kernel=fla` 는 코드 작업이 아니다.** qwen3_5 + CUDA + causal_conv1d 면 자동 통과한다.
  이미지/파드 문제다
- **`kernel=liger` 의 엔트리포인트 철자는 리서치가 확정했다** — 그리고 `LIGER_ENTRYPOINTS`
  표가 **두 군데 틀렸다**고 보고했다. `.plans/research/axis-libraries.md §1` 을 읽고,
  거기 인용된 원문을 **네가 직접 열어 확인한 뒤** 고친다

## 리서치가 바꾼 것 — 읽기 전에 이것부터

`.plans/research/axis-libraries.md §8` 의 표가 축 값마다 "이 호스트에서 구현 검증 가능 /
이미지 필요 / GPU 필요"를 갈라 놓았다. 두 가지가 이 레인의 완료 조건을 바꾼다.

**1. `precision=mxfp8`/`nvfp4` 는 하드웨어가 없어서 못 켠다.**
리서치 §6.4: mxfp8 은 **compute capability 10.x 전용**, nvfp4 는 **CC ≥ 10.0 전용**이고
**지원 검사 자체가 없다**(CC 미만에서 진입하면 무엇이 일어나는지도 미확인).
이 스터디의 파드는 A100 이다. RunPod 에서 CC 10.x 를 확보할 수 있는지는 **확인 안 함**이며
리서치가 파드 질문 9번으로 등록했다.

→ **`precision` 그룹이 단일값으로 남을 수 있고, 그것은 코드 결함이 아니라 하드웨어 사실이다.**
그 둘을 뭉개지 않는다. 구현은 하되(recipe 컨텍스트는 열 수 있다), `axis-values` 가 그것을
여전히 단일값으로 세면 **그 이유를 코드 미구현이 아니라 하드웨어로 기록**하고
`.plans/notes/axes.md` 에 적어 넘긴다. 이 저장소는 note 가 "어느 레인 소관"이 아니라
**"그 구멍의 결과"** 를 적어야 한다는 규칙을 이미 갖고 있다(`HAZARDS.md §3`).

**2. `kernel=kernels_hub` 는 이미 다른 이유로도 죽어 있다.**
리서치 §3.1: `envs/native` 의 `kernels` 핀 0.16.0 을 transformers 5.14.1 이 거부한다.
결정 6(축 값을 버린다)은 그대로이되, **이유가 하나 더 있고 그것이 독립적이다.**
제거 사유를 기록할 때 둘 다 적는다.

## `kernel=kernels_hub` 를 제거한다 (결정 6)

지금 `axes.py` 가 무조건 거부한다. transformers 5.14.1 의 두 진입점이 **모두 모델 객체를
요구하는데** `axes.patch` 는 모델 생성 전에 돈다. 그것은 버그가 아니라 설계다 —
`scripts/bench.py` 가 kernel/attn 은 모델이 생긴 뒤에는 바꿀 수 없다고 적고 있다.

**축 값을 버린다.** `configs/kernel/kernels_hub.yaml` 을 지우고 스키마 Literal 에서 빼고
이유를 기록한다. `kernel` 은 `none`/`liger`/`fla` 가 된다.

`kernels_hub` 는 아홉 파일에 흩어져 있다. 그중 **네 소유가 아닌 것들**:
- `trainbench/config_schema.py` 의 Literal — **measure 레인 소유** (wave 1 에 끝났다)
- `scripts/audit_plan.py` 의 `axis-packages` 표 항목 — **통합자 전용**
- `docs/CONTRACTS.md`, `docs/support-matrix.md`, 루트 `PLAN.md` — **integrate** 레인
- `tests/test_pods.py`, `tests/test_smoke_cpu.py` — **report** / split 소유

→ 네 소유(`configs/kernel/`, `trainbench/axes.py`, `tests/test_axes.py`)만 고치고,
**나머지 전부를 `.plans/notes/axes.md` 에 파일:줄로 적어 넘긴다.** 머지 단계가 적용한다.
이 목록이 빠지면 머지 후 스키마와 config 가 어긋나 게이트가 죽는다.

## 이 레인은 로컬에서 실행 검증이 안 된다 — 그래서 게이트가 다르다

bitsandbytes / deepspeed / Transformer Engine / DALI 중 **어느 것도 이 호스트에 없다.**
구현은 스텁으로만 검증되고, 실제로 도는지는 파드가 답한다.

그러므로 "부숴서 확인한다"를 **감사에 대고** 한다:
적용 지점을 비우면 `axis-values` 가 그 값을 **이름으로 지목해야 한다.**
선례: `PackedCollate.__call__` 에 raise 를 넣으면 32/46 → 30/46 으로 내려가고
`dataloader/torch_packed` 를 이름으로 지목했다.

**축마다 파드가 무엇을 보여야 하는지 한 문장씩 적는다.**

## 완료 조건

1. `axis-values` 의 단일값 그룹이 0 →
   `infisical run --env=dev -- uv run python scripts/audit_plan.py`
   **예외**: `precision` 이 하드웨어 이유(CC 10.x 부재)로 단일값으로 남을 수 있다.
   남으면 그것을 **미구현이 아니라 하드웨어로** 기록하고 `.plans/notes/axes.md` 에 넘긴다.
   남은 이유를 코드 미구현으로 적으면 그것이 이 저장소가 아홉 번 겪은 "note 가 blocker 를
   가린" 모양이다
2. **축 값마다** 적용 지점을 비우면 `axis-values` 가 그 값을 이름으로 지목한다.
   감사가 공허해지지 않는 것을 실증한다. **축마다 mutation 출력 인용**
3. `kernels_hub` 가 config 에서 제거되고 이유가 기록됐다.
   스키마·감사·문서 쪽 변경 목록이 `.plans/notes/axes.md` 에 파일:줄로 있다
4. `kernel=liger` 의 qwen3_5 엔트리포인트 철자가 **확정됐거나, 확정되지 않았음이 명시됐다**
5. 각 검사를 되돌리면 죽는다. **사보타주 전에 `co_filename`/`co_firstlineno` 확인** —
   이 저장소에서 클래스 본문 앞쪽에 사보타주를 넣어 나중 정의가 이긴 전례가 있다
6. 네 게이트. `axis-values` 의 count 가 3 → 0 이 되면 `newly fixed` 로 BLOCK 된다.
   **정상이고 머지 단계가 처리한다. `docs/audit-baseline.json` 을 건드리지 않는다**
7. **확인 안 함** — 축별로 파드가 답할 것을 적었다.
   구현은 스텁 검증뿐이고 그것을 숨기지 않는다

## capture 와의 경계

capture 레인이 wave 1 에서 네 축의 되읽기를 열었다:
`optim=adamw_8bit`, `parallel=zero2/zero3`, `precision=mxfp8/nvfp4`, `train.offload=*`.
그 되읽기는 **엔진/옵티마이저/recipe 객체에서 읽는다** — config 에서 읽지 않는다.

그러므로 네가 구현하는 객체가 그 속성을 **내줘야 한다.** 예: deepspeed 엔진에서 zero stage 와
offload target 이 읽히지 않으면 capture 는 `undetermined` 를 돌려주고 `assert_matches` 가
런을 거부한다. **구현했는데 측정이 안 열리는 것이 정확히 그 모양이다.**

`Built` 에 필드가 더 필요하면 (capture 가 `owned_axes`/`precision_recipe` 를 이미 더했다)
**경계 `applied-axes` 에서 맞춘다** — 새 필드는 기본값을 갖게 하고, 계약이 표현하지 못하면
`boundaryRequests` 로 요청한다. **계약 파일을 고치지 않는다.**

## 하지 않는 것

- `trainbench/applied.py` — **capture** 레인
- `configs/model/` — **packing** 레인
- `configs/run/` — **report** 레인
- `trainbench/collate.py` — **packing** 레인
- `envs/*/pyproject.toml` — 아무도 소유하지 않는다. 패키지는 이미 있다.
  더 필요하면 `.plans/deps/axes.txt`
- `scripts/audit_plan.py`, `docs/audit-baseline.json`, 루트 `PLAN.md` — **통합자 전용**
