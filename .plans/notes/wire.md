# wire — 머지 단계로 넘기는 것

base `da44388549d73a27b7cebd74fc818e2173f8e164`, 브랜치 `wave3a-wire`.
여기 적힌 숫자는 전부 이 워크트리에서 이번 세션에 직접 실행해 얻었다.

---

## 1. `docs/audit-baseline.json` — `config-consumed` 가 0이 됐다

```
infisical run --env=dev -- uv run python scripts/audit_plan.py
  FIXED config-consumed     all config leaves are read by code
  13/15 passing, 0 new failure(s), 1 newly fixed, 0 grew, 0 shrank, 0 unreadable
  BLOCKED: baseline is stale, these now pass: config-consumed
```

브리프가 예고한 정상 상태다. trackio 두 leaf 를 `configs/run/*.yaml` 4개와
`config_schema.RunConfig` 에서 **한 커밋으로** 뺐다. baseline 의 `config-consumed`
note 는 이제 전부 과거형이다 — 갱신할 때 통째로 다시 쓴다.

## 2. trackio 의 `pyproject.toml` 몫은 **되돌렸다** — 여섯 env lock 이 흔들린다

브리프 작업 4 가 `tracking` extra 도 이 레인이 하라고 했고, 실제로 해봤고, **되돌렸다.**
실측:

```
# tracking = ["trackio>=0.34"] 삭제 + uv lock
Resolved 154 packages in 526ms
Removed gradio-client v2.6.0 / orjson v3.11.9 / trackio v0.34.0

infisical run --env=dev -- uv run python scripts/audit_plan.py --only env-locks
NEW   env-locks  envs/axolotl/uv.lock is stale: regenerate with `uv lock`;
      envs/ms-swift/uv.lock ...; envs/native/uv.lock ...;
      envs/sentence-transformers/uv.lock ...; envs/tevatron/uv.lock ...;
      envs/unsloth/uv.lock is stale
  0/1 passing, 1 new failure(s)

# 같은 명령, pyproject.toml/uv.lock 만 stash 한 상태
PASS  env-locks  every lock agrees with its pyproject.toml and every image sync asserts it
```

여섯 env 가 루트 `trainbench` 를 경로로 의존하므로 optional-dependency 그룹 하나를
빼면 여섯 lock 이 전부 stale 이 된다. `envs/**` 는 통합자 전용이라 이 레인이
재생성할 수 없다.

**머지 단계가 할 것**: 루트 `pyproject.toml` 의 `tracking` extra 삭제 + `uv lock` +
`envs/*/uv.lock` 여섯 개 `uv lock` 재생성을 **한 커밋**으로. 스키마·yaml 쪽은 이미
빠져 있으므로 config 합성은 어느 순서로도 깨지지 않는다 — 이번엔 한쪽만으로 착지
가능하다(report/measure 레인이 못 했던 것과 다른 점이다).

## 3. 소유 밖이라 넘기는 한 줄짜리들

| 파일:줄 | 변경 | 왜 |
|---|---|---|
| `docs/support-matrix.md:128` | `\| trackio \| 0.34.0 \|` 행 삭제 | 결정 3 |
| `docs/support-matrix.md:504` | `\| kernel/kernels_hub \| ...` 행 삭제 | 결정 6 |
| `PLAN.md:414` | `# none, liger, fla, kernels_hub` → `# none, liger, fla` | 결정 6 |
| `PLAN.md:576` | `kernel=liger / fla / kernels_hub` 에서 `kernels_hub` 제거 | 결정 6 |
| `docs/evidence/env-report-cpu-qwen3_5_0_8b-native.json:74-75` | `trackio_*` 두 키를 담고 있다 | 스키마가 더 이상 그 필드를 갖지 않는다. 재생성하거나 그 아티팩트가 어느 커밋의 것인지 명시한다 |

`docs/CONTRACTS.md §2` 는 이 레인이 했다(호출 지점 표 + `step_context` 문단).

## 4. 계약 개정 요청 — `tests/fixtures/microbatch.sample.json` 의 `seq_idx`

**근거는 확인됐다. 그런데 fixture 한 줄로는 닫히지 않아서 넣지 않았다.**

핀된 휠에서 확인(transformers 5.14.1, 이 워크트리 설치본):

- `transformers/utils/generic.py:825,839` — `TransformersKwargs.seq_idx: torch.IntTensor | None`,
  문서 문장은 "Sequence index for each token in a flattened packed batch."
- `transformers/models/qwen3_5/modeling_qwen3_5.py:492-499` —
  `self.causal_conv1d_fn(x=..., weight=..., bias=..., activation=..., seq_idx=kwargs.get("seq_idx"))`
- 같은 파일 `:500-501` — `causal_conv1d_fn` 이 None 이면(패키지 부재) 폴백은
  `F.silu(self.conv1d(mixed_qkv)[...])` 이고 **`seq_idx` 를 받는 자리가 아예 없다.**
  즉 패키지 없는 이미지에서는 `seq_idx` 를 넘겨도 conv 격리가 생기지 않는다.

**넣지 못한 이유** (실측):

```
# tensors_may_add 에 "seq_idx" 만 추가하고
infisical run --env=dev -- uv run pytest -q tests/contract/test_collate_metrics.py
FAILED test_every_invariant_holds_for_a_real_batch[packed]
E  packed.tensors carries ['cu_seq_lens_q','cu_seq_lens_k','max_length_q','max_length_k']
   of the varlen kwargs. ... all of [... 'seq_idx'] are non-None ...
E  assert 4 in (0, 5)
```

`test_collate_metrics.py:320-324` 가 `tensors_may_add` **전체**에 all-or-nothing 을
건다. 그 규칙은 어텐션 varlen 4종에 대해서는 옳다(넷이 다 있어야 fast path 를 탄다).
`seq_idx` 는 성질이 다르다 — 아키텍처 조건부이고 `Qwen3_5GatedDeltaNet` 의 conv 만
읽는다. 그래서 같은 집합에 넣으면 `qwen3_vl`/`gemma4` 의 packed 배치가 넷만 싣고 빨개진다.

개정본이 해야 할 것: `tensors_may_add` 에 `seq_idx` 를 넣되 **all-or-nothing 집합을
어텐션 varlen 4종으로 좁힌다**(fixture 의 invariant 문장과 `test_collate_metrics.py`
의 `MAY_ADD` 사용처 둘 다). 그 뒤 packing 레인이 `trainbench/collate.py` 에서
`seq_idx` 를 만든다. 양쪽이 각자 자기 편을 패치하는 것이 `HAZARDS §4.3` 이므로
이 레인은 fixture 를 원상 복구하고 요청만 남겼다.

## 5. axes 레인 노트와 다르게 한 것 하나 — `KERNEL_MODULE_ROOTS["kernels"]` 는 남겼다

`.plans/notes/axes.md §1` 은 `_patch_kernels_hub`·`KERNEL_PATCHERS` 항목과 함께
`KERNEL_MODULE_ROOTS["kernels"]` 도 지우라고 적었다. 앞의 둘은 지웠고 이것은 남겼다.

이유: 그 표는 **적용 표가 아니라 되읽기 표**다. 어댑터가 스스로 hub dispatch 를 켜면
모델은 여전히 `kernels` 패키지의 클래스로 만들어지고, 행을 지우면 그것이 `none` 으로
읽혀 `kernel=none` 요청과 **일치해버린다.** 매핑되는 값 `kernels_hub` 가 이제 어떤
설정도 아니라는 점은 의도된 것이고 `_capture_kernel` 의 `mixed(...)`/`partial(...)` 과
같은 장치다 — `assert_matches` 가 그것으로 런을 막는다. 코드에 그 문장을 적어 두었다.
`tests/test_axes.py::test_every_kernel_the_schema_offers_routes_to_a_patcher` 가 강제하는
짝은 스키마 Literal ↔ `KERNEL_PATCHERS` 뿐이고, 이 표는 거기 들어가지 않는다(확인함).

## 6. `native_binding` 은 지웠다 (작업 7 판단)

`trainbench/probe/native.py::load` 가 같은 빌드를 하고 그쪽이 유지되는 정의다
(`AutoProcessor` + `AutoModel.from_pretrained(dtype=steps.dtype_for(device))` +
`model.to(device)`; padding-side 정렬은 `loader.Adapter.aligns_padding_side` 가 한다).
`load_framework` 의 `ModuleNotFoundError` 폴백도 함께 지웠다 — 이미지에서
`trainbench/loader.py` 가 빠지는 것은 패키징 결함이고, 폴백은 그것을 "어느 코드 경로가
숫자를 냈는가"의 조용한 변경으로 바꾼다.

## 7. 이 레인이 하지 않은 것 — 다음 레인/파드로

- **`measurement.repeats` 반복 루프.** `.plans/notes/measure.md §3` 이 요구한 다섯 줄 중
  `metrics.repeat_seeds(...)` 를 반복마다 기록하는 것만 하지 않았다. 지금 `train()` 은
  런 하나를 한 번 잰다. 반복 루프는 이 레인이 잇는 배선이 아니라 새 끝이고, 결과 레코드의
  모양(스칼라 vs 반복별 목록)이 `record-report` 계약과 `scripts/report.py` 양쪽을 건드린다.
  `measurement` 블록은 이미 레코드에 실리므로 `repeats: 1` 이 결과에 보인다.
- **`_capture_dataloader_packing` 의 DALI 공백**(`.plans/notes/axes.md §2.3`). 고치지 않았다.
  `axes._dataloader` 가 `backend=dali` 를 아직 거부하므로 되읽을 객체가 이 저장소에
  존재한 적이 없고, `DALIGenericIterator` 가 무엇을 노출하는지는 어느 핀된 소스에서도
  읽지 않았다. 이 호스트에 `nvidia-dali` 가 없다. DALI 구현과 같은 레인에서 정해야 한다.
- `parallel=fsdp2` 되읽기(§2.1)와 deepspeed 옵티마이저 기록(§2.2)은 닫았다.

## 8. 파드가 답해야 하는 것 — 축별로 한 문장 (완료 조건 9)

이 호스트에 deepspeed·bitsandbytes·transformer-engine·nvidia-dali·fla·causal-conv1d·
liger-kernel 이 **하나도 없다**(2026-08-03 `uv run python -c "import ..."` 로 확인).
아래 배선은 전부 스텁 위에서만 검증됐고, 실제 객체에서 무엇을 읽는지는 파드가 답한다.

| 축 / 배선 | 파드가 찍어야 할 한 가지 |
|---|---|
| `parallel=fsdp2` | `fully_shard` 후 `isinstance(model, torch.distributed.fsdp.FSDPModule)` 와 `type(model).__name__` — MRO 로 읽는 새 경로가 실물에서 맞는지 |
| `parallel=zero2/zero3` | `type(engine.optimizer).__name__` — `_engine_optimizer_class` 가 `engine.optimizer` 를 getattr 로만 읽는다. 이름이 다르면 detail 이 조용히 비어 있는다 |
| `train.offload=*` | `engine.zero_offload_optimizer()` / `zero_offload_param()` 의 실제 반환값 (capture 레인 노트와 같은 질문) |
| `optim=adamw_8bit` | `type(optimizer).__name__` 이 `AdamW8bit` 인지 paged 변형인지 |
| `peft=qlora` | `load_kwargs` 의 CUDA 게이트가 통과한 뒤 `model.is_loaded_in_4bit` 가 실제로 서는지 |
| `precision=mxfp8/nvfp4` | CC 10.x GPU 를 확보할 수 있는가. 확보 못 하면 이 축은 이 스터디에서 측정 불가로 확정된다 |
| axolotl `required_step_context` | `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 안에서 `torch.is_autocast_enabled("cuda")` 가 True 이고 axolotl 의 fp32 임베딩 × bf16 본체 matmul 이 죽지 않는지. **autocast on/off 의 속도 차는 측정 안 함** (CUDA 없음) |
| `kernel=fla` × packing | `causal_conv1d_fn` 이 설치돼 있는지. 없으면 폴백 conv 라 `seq_idx` 를 넘겨도 경계 격리가 생기지 않는다(§4) |
| OOM 경로 | `metrics.is_oom` 이 실제 `torch.OutOfMemoryError` 를 잡고 `OOM_EXIT`(5)로 나가는지. 이 호스트에서는 주입한 예외로만 확인했다 |

## 9. `.plans/deps/wire.txt` — 없음

새 패키지를 요구하지 않는다.
