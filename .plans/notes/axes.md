# axes — 머지 단계로 넘기는 것

이 레인이 소유하지 않은 파일에 필요한 변경, 그리고 이 호스트에서 답이 나오지 않아
파드로 넘어가는 질문. 숫자는 전부 이 워크트리에서 이번에 직접 낸 것이다.

---

## 1. `kernels_hub` 제거의 남은 지점 (결정 6)

`configs/kernel/kernels_hub.yaml` 은 지웠다. `kernel` 그룹은 `none`/`liger`/`fla` 셋이다.
`trainbench/axes.py` 는 `KERNEL_PATCHERS["kernels_hub"]` 를 **남겨 두었다** — 스키마
Literal 이 아직 그 값을 내주고, `scripts/bench.py::preflight` 는 Hydra 가 아니라 스키마에서
만든 config 를 받으므로 그 경로로는 여전히 도달한다. 거부 메시지가 두 이유를 다 적는다.

머지 단계가 적용할 것:

| 파일:줄 | 변경 | 왜 |
|---|---|---|
| `trainbench/config_schema.py:174` | `Literal["none", "liger", "fla", "kernels_hub"]` → `Literal["none", "liger", "fla"]` | 축 값이 사라졌다. 이것이 빠지면 스키마가 config 에 없는 값을 계속 내준다 |
| `scripts/audit_plan.py:794` | `"kernel/kernels_hub": ("kernels",),` 삭제 | `axis-packages` 는 config 파일을 돌므로 지금은 그냥 안 쓰이는 항목이다. 죽은 표 항목 |
| `docs/CONTRACTS.md:164` | `` `kernel.name` (liger/fla/kernels_hub) `` → `(liger/fla)` | §2 호출 지점 표 |
| `docs/support-matrix.md:504` | `` | `kernel/kernels_hub` | `kernels>=0.10` | native | 위와 같음 | `` 행 삭제 | 축 값이 없다 |
| `PLAN.md:414` | `# none, liger, fla, kernels_hub` → `# none, liger, fla` | 레이아웃 블록 주석 |
| `PLAN.md:573` | `` `kernel=liger` / `fla` / `kernels_hub` `` 행에서 `kernels_hub` 제거 | 같은 이유 |

`config_schema.py` 의 Literal 이 사라지면 `trainbench/axes.py` 의 `_patch_kernels_hub` 와
`KERNEL_PATCHERS` 항목, `KERNEL_MODULE_ROOTS["kernels"]`, 그리고
`tests/test_axes.py::test_kernels_hub_from_the_schema_is_still_refused_with_both_reasons`
도 같이 지워야 한다. `tests/test_axes.py::test_every_kernel_the_schema_offers_routes_to_a_patcher`
가 그 짝을 강제하므로 한쪽만 지우면 빨개진다.

`tests/test_smoke_cpu.py:1499,1608` 과 `tests/test_pods.py:2781` 은 스키마에서 만든 config 에
`kernel.name="kernels_hub"` 를 넣어 preflight 가 거부하는지 본다. Literal 이 사라지는
시점에 그 셋도 다른 거부 사유로 바꿔야 한다 — 지금은 그대로 통과한다(실행 확인함).

**제거 사유는 둘이고 서로 독립이다.** 하나가 해소돼도 다른 하나가 남는다.
(1) 두 진입점(`from_pretrained(use_kernels=True)`, `integrations.hub_kernels.kernelize(model)`)이
모델 객체를 요구하는데 `axes.patch` 는 모델 생성 전에 돈다(docs/CONTRACTS.md §2).
(2) `envs/native` 가 `kernels==0.16.0` 을 핀하는데 transformers 5.14.1 의 창은
`0.15.2 <= v < 0.16.0` 상한 배타적이다 — `is_kernels_available()` 이 False 가 되고
`use_kernel_forward_from_hub` 가 **조용히 항등 데코레이터**가 된다
(`.plans/research/axis-libraries.md` §3.1-3.2).

---

## 2. capture 쪽 공백 셋 — `trainbench/applied.py` (capture 레인)

전부 "축은 적용됐는데 되읽기가 못 본다" 모양이다. 이 레인은 `applied.py` 를 건드리지
않았고, 셋 다 계약(`tests/contract/test_applied_axes.py`)이 표현하지 못하는 것이 아니라
capture 구현이 아직 모르는 것이다 — 그래서 `boundaryRequests` 가 아니라 여기 적는다.

### 2.1 `PARALLEL_WRAPPERS` 가 FSDP2 를 못 읽는다 — **`parallel=fsdp2` 를 막는다**

`applied.py:821-825` 는 `"FullyShardedDataParallel": "fsdp2"` 로 클래스 이름을 본다.
그것은 **FSDP1 의 래퍼 클래스**다. torch 2.13.0 의 FSDP2(`fully_shard`)는 래핑하지 않고
제자리에서 클래스를 갈아끼운다 — `type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)`,
`module.__class__ = new_cls`
(`.venv/.../torch/distributed/fsdp/_fully_shard/_fsdp_init.py:404-430`, 이 워크트리의 설치본을 열어 확인).

결과: `axes._parallel` 이 `fully_shard` 를 부르면 축은 적용되지만
`_capture_parallel_strategy` 는 `unwrapped(world_size=N)` 를 돌려주고 `assert_matches` 가
런을 거부한다. **`parallel=fsdp2` 는 측정이 열리지 않는다.**

고칠 곳: `applied.PARALLEL_WRAPPERS` 대신(또는 함께) MRO 를 본다 —
`torch.distributed.fsdp.FSDPModule` 의 인스턴스인지. 클래스 이름 접두사 `FSDP` 만 보는 것은
약하다(사용자 클래스가 `FSDPBlock` 이면 걸린다).
`tests/test_axes.py::test_fsdp2_shards_in_place_and_the_capture_cannot_yet_see_it` 이 이 상태를
그대로 고정해 두었으므로, 고치는 쪽이 그 마지막 단언을 뒤집으면 된다.

### 2.2 deepspeed 엔진 아래의 옵티마이저가 기록에 안 남는다

`axes._deepspeed` 는 `deepspeed.initialize` 가 돌려준 래퍼가 아니라 **넘긴 torch 옵티마이저**를
`Built.optimizer` 에 남긴다. 래퍼를 남기면 `_capture_optim` 이 `OPTIM_CLASS_AXIS` 에 없는
클래스명을 소문자로 돌려주고(`applied.py:363`), `optim.name` 이 ZeRO 와 무관한 축인데도
모든 ZeRO 런이 막힌다.

결과: 결과 JSON 에 두 층 중 한 층만 남는다. 실제로 스텝하는 것은 deepspeed 의 래퍼이고
그것이 우리 인스턴스에 위임한다. capture 가 `engine.optimizer.optimizer` 를 벗겨 두 이름을
다 적으면 이 손실이 사라진다. **측정을 막지는 않는다.**

### 2.3 `dataloader.packing` 은 collate 로만 읽힌다 — `dali_packed` 를 막는다

`_capture_dataloader_packing` 은 `dataloader.collate_fn` 을 본다. DALI 이터레이터에는
`collate_fn` 이 없다. 그래서 `configs/dataloader/dali_packed.yaml` 은 backend 가 구현돼도
packing 이 undetermined 로 돌아와 런이 막힌다. DALI 구현 전에 정해져야 하는 것이라
아래 §4 에 함께 적는다.

---

## 3. `train.offload` 는 감사가 원리적으로 볼 수 없다 — `AXIS_VALUE_COMPANIONS` 필요

`scripts/audit_plan.py` 는 통합자 전용이므로 요청만 한다.

`offload_optimizer` / `offload_param` 은 deepspeed 의 `zero_optimization` **안의 섹션**이다
(`engine.zero_offload_optimizer()` 가 `self._config.zero_config.offload_optimizer` 를 읽는다,
`.plans/research/axis-libraries.md` §5.2). 그래서 ZeRO stage 없이는 offload 가 없고,
stage 를 지정하는 스키마 필드는 `parallel.strategy` 하나뿐이다. `axes._deepspeed` 는
`train.offload != none` 인데 strategy 가 zero2/zero3 가 아니면 거부한다 — 여기서 stage 를
고르면 어떤 config 에도 적히지 않은 설정이 측정 경로에 들어간다.

`axis-values` 는 그룹 하나씩 합성하므로 `train.offload=optimizer` 를 기본
`parallel=single` 과 함께 시도하고, **파드에서도** 거부된다. 즉 이 그룹은 코드가 다 맞아도
`1/4` 로 남는다. 고치려면 `AXIS_VALUE_COMPANIONS` 에:

```python
"train.offload/optimizer": ("parallel=zero2",),
"train.offload/param": ("parallel=zero3",),
"train.offload/both": ("parallel=zero3",),
```

`offload_param` 이 stage 2 에서도 실제로 동작하는지는 **확인 안 함** — deepspeed 가 이
호스트에 없고 리서치도 stage 별 요구를 인용하지 않았다. `zero2 + param` 을 거부하지
않은 것은 그 때문이다(근거 없는 규칙을 지어내지 않는다). 파드 질문 §4 에 넣었다.

이 호스트 실측: 이 브랜치에서 `axis-values` 는
`36/52 applicable ... 3 group(s) offering one usable value: kernel 1/3, precision 1/3, train.offload 1/4`.
감사 전체는 `12/15 passing, 0 new failure(s), 0 newly fixed, 0 grew, 0 shrank, 0 unreadable`.

---

## 4. `axis-values` 의 단일값 그룹 셋이 왜 남았는가 — 셋 다 이유가 다르다

**이 구별을 뭉개면 `HAZARDS.md §3` 의 "note 가 blocker 를 가린" 모양이 된다.**

| 그룹 | 이 브랜치 | 남은 이유 | 파드에서 움직이는가 |
|---|---|---|---|
| `kernel 1/3` | none 만 적용 | **이미지**. liger 는 triton macOS 휠이 없어 import 불가, fla 는 `is_torch_cuda_available()` 게이트 | **움직인다.** 거부가 이제 환경 조건부다 |
| `precision 1/3` | bf16 만 적용 | **하드웨어**. mxfp8 은 CC 정확히 10.x, nvfp4 는 CC >= 10.0. 이 스터디 파드는 A100(8.0) | **A100 에서는 안 움직인다.** CC 10.x 확보 가능 여부는 확인 안 함 |
| `train.offload 1/4` | none 만 적용 | **체크의 합성 방식**(위 §3). 코드가 아니다 | `AXIS_VALUE_COMPANIONS` 를 넣기 전에는 안 움직인다 |

이전 baseline note 의 "거부가 `axes.py` 의 무조건 'not implemented' 라 이미지를 빌드해도
이 수는 안 움직인다" 는 **이 커밋으로 kernel 에 대해 더 이상 사실이 아니다.**
`kernel=liger` 는 세 아키텍처 전부 엔트리포인트가 있고, 패키지가 있는 이미지에서 실제
`apply_liger_kernel_to_*()` 를 부른다. `precision` 과 `train.offload` 에 대해서는 여전히
사실이 아니게 됐지만 — 구현은 들어갔다 — 수가 움직이지 않는 이유가 코드에서
하드웨어/체크 합성으로 옮겨갔을 뿐이다.

---

## 5. 축마다 파드가 무엇을 보여야 하는가

구현은 전부 스텁 검증뿐이다. 아래는 파드가 답해야 확정된다.

| 축 값 | 파드가 찍어야 할 것 |
|---|---|
| `kernel=liger` × qwen3_vl | `apply_liger_kernel_to_qwen3_vl()` 을 인자 없이 부른 뒤 예외 없이 돌아오는가. 임베딩 모델 클래스가 `Qwen3VLForConditionalGeneration` 이 아닐 수 있다 |
| `kernel=liger` × gemma4 | native(0.8.1)에서 `apply_liger_kernel_to_gemma4` 가 있는가, 그리고 `google/gemma-4-E2B` 가 어느 클래스로 로드되는가 |
| `kernel=liger` (전부) | `fused_linear_cross_entropy=True` 가 기본인데 이 스터디 손실은 InfoNCE 라 LM head 를 안 지난다. `_capture_kernel` 이 무엇이 갈렸다고 읽는지 |
| `kernel=fla` | `Qwen3_5GatedDeltaNet` 인스턴스의 `self.causal_conv1d_fn` / `self.chunk_gated_delta_rule` / `type(self.norm)` 각각 무엇인지. `fla-core` 가 실제로 깔려 있는지(`pip show fla-core`) |
| `optim=adamw_8bit` | `type(optimizer).__name__` (`AdamW8bit` 인지 `PagedAdamW8bit` 인지 — `applied.OPTIM_CLASS_AXIS` 가 둘 다 담고 있으나 어느 쪽인지는 확인 안 함). `min_8bit_size=4096` 문턱에 걸려 32bit 로 남는 텐서 수와 비율 |
| `peft=qlora` | `from_pretrained` 후 `model.is_loaded_in_4bit` 와 `config.quantization_config` 중 어느 쪽이 실제로 세팅되는지. 4bit base 의 부동소수 파라미터 dtype 집합(`_capture_precision` 이 `mixed` 를 읽으면 런이 막힌다) |
| `parallel=zero2/zero3` | `type(engine).__name__`, `engine.zero_optimization_stage()`, `type(engine.optimizer).__name__` |
| `train.offload=*` | `engine.zero_offload_optimizer()` / `engine.zero_offload_param()` 의 실제 반환값. **그리고 `zero2 + offload=param` 이 실제로 파라미터를 옮기는지** — 엔진 config 는 섹션을 그대로 돌려주므로 무시돼도 되읽기는 통과한다 |
| `parallel=ddp` | `type(model).__name__ == "DistributedDataParallel"` 확인. world >= 2 필요 |
| `parallel=fsdp2` | 위 §2.1 이 고쳐지기 전에는 측정 불가 |
| `precision=mxfp8/nvfp4` | `is_mxfp8_available()` / `is_nvfp4_available()` 의 **반환 arity** — bool 인지 `(bool, str)` 인지. 그리고 CC 10.x GPU 를 RunPod 에서 확보할 수 있는지 |
| `compile` × `parallel` | `_parallel` 을 `_compile` 앞에 두었다. 반대 순서의 비용은 측정 안 함 |

---

## 6. `dataloader.backend=dali` 는 구현하지 않았다

거부 메시지에 이유를 넣었다. 요약: `DALIGenericIterator` 의 시그니처는 리서치가 확정했지만
(`.plans/research/axis-libraries.md` §7) **그것이 도는 파이프라인**은 어느 핀된 소스에도
읽히지 않았다 — `datasets` 행을 `fn.external_source` 로 넣는 파이프라인을 기억으로 쓰는 것이
`AGENTS.md` 가 금지하는 바로 그 행위다. 그리고 §2.3 의 `dali_packed` 문제가 먼저 정해져야
한다. 두 가지가 정해지면 `_dataloader` 한 곳만 고치면 된다.

`dataloader` 그룹은 지금 4/6 이고 단일값이 아니므로 완료 조건을 막지 않는다.

---

## 7. `.plans/deps/axes.txt` — 없음

새 패키지를 요구하지 않는다. bitsandbytes / deepspeed / transformer-engine / nvidia-dali 는
`envs/native/pyproject.toml` 에 이미 있다. 이 호스트(root `pyproject.toml` 의 `native` extra)에는
넷 다 없고, 그것이 이 레인의 검증이 스텁뿐인 이유다 — 넣어 달라고 요청하지 않는다.
넷 중 셋은 macOS 휠이 없거나 CUDA 확장이라 root lock 에 들어가면 `uv sync` 가 깨진다.
