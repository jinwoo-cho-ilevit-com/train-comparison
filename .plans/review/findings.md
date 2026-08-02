# 리뷰 발견 — 실행으로 확정된 것 (2026-08-03)
base `fa5a325aa2d55112b9e1ef3d2e4022f3507e1607` .. head `d372af722df95f982bb88f7858ee731871245f38`. micro 9 + macro 4 단위, 검증 에이전트가 발견마다 실행으로 확정했다.
확정 40 / 반박 1 / 실행불가 0 / 미검증 0 / minor 30

## 확정 발견
### `loader-bench-step-owner-never-consumed` — blocker / contract-split
- 단위: macro:contracts
- 위치: `scripts/bench.py:305`

**주장**: `loader-bench` 의 생산자 쪽은 `step.owner=framework` / `batch_keys=("query","passage")` 를 선언하는데, 소비자 쪽(`scripts/bench.py::train`)은 `binding.step` 을 한 번도 읽지 않고 모든 프레임워크에 하네스 스텝을 강제한다.

**실패 시나리오**: `framework=tevatron run=timing` 으로 파드에서 bench.py 를 돌리면, 모델 적재와 `assert_matches` 까지는 통과한다(`loss.name`/`parallel.cross_device_negatives` 가 framework_owned 라 거부되지 않는다). 그리고 첫 스텝에서 `pooled_embeddings` 가 `model(**tensors)` 를 `{input_ids, attention_mask}` 로 호출한다. 계약 fixture 와 `trainbench/loader.py:461-476` 이 그 forward 는 `query=`/`passage=` 만 받는다고 적어둔 바로 그 호출이다. `metrics.is_oom` 이 아니므로 `main` 이 그대로 re-raise 하고 결과 파일이 하나도 안 남는다 — 적재까지 다 치른 파드-시간이 '측정 실패'가 아니라 '아무 기록 없음'으로 끝난다. 계약의 element 1(`_validate_step`, `tests/contract/test_loader_bench.py:129-153`)이 존재하는 이유가 정확히 이것인데, 그 계약은 fixture 만 검사하고 소비자가 그 필드를 쓰는지는 아무도 검사하지 않는다. AdapterOut 여덟 필드 중 `step`·`fingerprint`·`documented_entry_point` 셋이 생산만 되고 소비되지 않는다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && uv run python -c "import sys; sys.path[:0]=['.','tests/contract']; import test_collate_metrics as c; from trainbench import loader; mb=c._micro_batch('padded_text_only'); print('collate keys', sorted(mb.tensors)); print('tevatron batch_keys', list(loader.ADAPTERS['tevatron'].step.batch_keys)); print('bench reads binding.step:', 'binding.step' in open('scripts/bench.py').read())"  →  collate keys ['attention_mask','input_ids'] / tevatron batch_keys ['query','passage'] / bench reads binding.step: False. 변이 확인: tests/test_loader.py 에 `assert set(loader.ADAPTERS[n].step.batch_keys) <= set(mb.tensors)` 를 넣으면 tevatron 에서 죽는다.
```

**검증** (reproduced):
```text
cd /Users/jwcho/Codes/train-comparison && uv run python scratchpad/repro.py   # (1) 소비자 지점 재현
# repro.py: scripts/bench.py 를 경로 로드 -> pooled_embeddings 의 co_filename/co_firstlineno 출력 ->
#   tests/contract/test_collate_metrics._micro_batch('padded_text_only') 의 실제 collate 텐서를
#   upstream 시그니처(`def forward(self, query=None, passage=None)`, .plans/research/tevatron.md:451,
#   encoder.py:52) 를 그대로 가진 nn.Module 스텁에 bench.pooled_embeddings 로 흘림
# (2) 변이(기존 검사가 아무것도 안 봄): tests/test_loader.py 에 아래를 임시 추가 후
#   uv run pytest tests/test_loader.py -k mutation_consumer -x -q
#   assert set(adapter.step.batch_keys) <= set(mb.tensors)

---
[재현 1 — 소비자 경로]
LIVE DEF: /Users/jwcho/Codes/train-comparison/scripts/bench.py 86
collate keys: ['attention_mask', 'input_ids']
tevatron step: Step(owner='framework', callable='tevatron.retriever.modeling.DenseModel.forward', batch_keys=('query', 'passage'))
TypeError: DenseModelStub.forward() got an unexpected keyword argument 'input_ids'

[재현 2 — 정적 소비 여부]
bench reads binding.step: False
bench mentions .step.owner: False

[변이 — 기존 스위트에 이 검사가 없음]
E           AssertionError: ('tevatron', ('query', 'passage'), ['attention_mask', 'input_ids'])
E           assert {'passage', 'query'} <= {'attenti
```

### `framework-owned-step-has-no-consumer` — blocker / contract-split
- 단위: macro:axis-pipeline
- 위치: `scripts/bench.py:305`

**주장**: `AdapterOut.step`(owner=framework, batch_keys=("query","passage"))은 어디에서도 읽히지 않아 tevatron 칸은 `assert_matches` 를 두 축 면제로 통과한 뒤 timing 루프 안에서 하네스 스텝을 돈다 — 결정 5 와 `docs/support-matrix.md` 가 서술하는 동작이 코드에 없다.

**실패 시나리오**: `framework=tevatron purpose=timing` 런. `loader.ADAPTERS['tevatron']` 은 `Step(owner='framework', callable='tevatron.retriever.modeling.DenseModel.forward', batch_keys=('query','passage'))`(loader.py:454-458)과 `owned_axes={loss.name, parallel.cross_device_negatives}` 를 선언한다. `build_run` 은 `owned_axes` 만 `axes.assemble` 로 넘기고(bench.py:554) `binding.step` 은 읽지 않는다(AST 로 확인: `binding.step` 참조 0건). 그래서 capture 는 두 축을 `state='framework_owned'` 로 면제하고 `assert_matches` 가 런을 통과시킨다. 그 다음 `train()` 은 `step.owner` 를 보지 않고 무조건 `pooled = pooled_embeddings(built.model, tensors, side, micro.cu_seqlens)` → `loss = built.loss_fn(pooled[:half], pooled[half:])`(bench.py:305-311)를 돈다. `tensors` 의 키는 collate 가 만드는 `input_ids`/`attention_mask` 이고(저장소 전체에서 'query'/'passage' 배치 키를 만드는 코드 0건), `DenseModel.forward` 는 `query=`/`passage=` 만 받는다 — 어댑터 자신의 `documented_entry_point.source` 가 'a signature this forward has no argument for' 라고 적어둔 그 시그니처다. 결과: `assert_matches` 를 통과하고 타이머가 열린 뒤 step 0 에서 TypeError, `main` 의 `except` 가 OOM 이 아니므로 re-raise, 결과 파일 없음. 소모된 파드 시간이 결과 0건이 되고, 레코드가 남았다면 그 레코드의 최상위 `framework_owned` 는 '돌지도 않은 tevatron 이 손실을 계산했다' 고 주장한다. `.plans/notes/adapters.md §5.3` 은 '실제 배치를 넣었을 때의 예외/성공' 을 파드 질문으로 남겼으나, **부를 자리 자체가 없다**는 것은 어느 노트에도 없다.

**재현**:
```text
infisical run --env=dev -- uv run python -c "import ast,pathlib; t=ast.parse(pathlib.Path('scripts/bench.py').read_text()); print([ast.unparse(n) for n in ast.walk(t) if isinstance(n,ast.Attribute) and n.attr=='step'])"  # -> ['built.optimizer.step'] 뿐, binding.step 없음
grep -rn 'batch_keys' --include='*.py' trainbench scripts   # -> trainbench/loader.py 안에서만. 소비자 없음
grep -rn "'query'\|\"query\"" --include='*.py' trainbench/collate.py trainbench/embedding.py trainbench/probe/steps.py   # -> 배치 키 생산 0건
infisical run --env=dev -- uv run python -c "from trainbench import loader; a=loader.ADAPTERS['tevatron']; print(a.step.owner, a.step.callable, a.step.batch_keys, sorted(a.owned_axes))"
# 변이 확인: scripts/bench.py:305-311 을 `raise AssertionError('framework step')` 로 갈아도 tevatron 경로를 도는 테스트가 하나도 죽지 않는다
```

**검증** (mutation-killed-nothing):
```text
(1) AST 소비자 스캔: uv run python -c "import ast,pathlib; ... n.attr=='step'" over trainbench/**/*.py + scripts/**/*.py
(2) 변이: trainbench/loader.py:456-457 의 tevatron Step 을 callable="mutation.does.not.exist", batch_keys=("mutation_bogus_key",) 로 교체 후 infisical run --env=dev -- uv run pytest -q -p no:randomly (전체 스위트, tests/test_zz_verify_repro.py 제외 — 수집 단계에서 config_mapping 픽스처 부재로 에러, 이 발견과 무관)
(3) 실행 재현: scripts/bench.py 를 import 해 tevatron dd06310 forward 시그니처(query=, passage=)를 가진 nn.Module 스탠드인에 pooled_embeddings(model, {"input_ids","attention_mask"}, "left", None) 호출
(4) 복구: git checkout -- trainbench/loader.py; git status --porcelain
---
(1) 소비자 스캔 결과 — .step 속성 접근 전부:
('trainbench/loader.py', 532, 'adapter.step')   # AdapterOut 생성
('trainbench/loader.py', 200, 'self.step')      # to_dict 직렬화
('trainbench/loader.py', 358, 'self.step')      # owner/owned_axes 정합 검사
('trainbench/loader.py', 364, 'self.step')      # 같은 검사
('scripts/bench.py', 326, 'built.optimizer.step')
-> scripts/bench.py 안에 binding.step / built.step 참조 0건. 선언을 읽고 분기하는 코드 없음.

(2) 변이 후 전체 스위트 (결정적 순서):
1 failed, 113 passed  (대상 파일들)
FAILED tests/test_loader.py::test_tevatron_owns_the_axes_its_forward_subsumes
E       AssertionError: assert 'mutation.does.not.ex
```

### `build-fingerprint-never-reaches-the-record` — blocker / emptiness
- 단위: macro:axis-pipeline
- 위치: `scripts/bench.py:851`

**주장**: `loader.load` 이 계산한 build fingerprint 는 `Binding.fingerprint` 로 들어온 뒤 아무도 읽지 않아, `kernel-provenance` 계약이 목적지로 못박은 `build_fingerprint` 키가 실제 런 레코드에 한 번도 실리지 않는다.

**실패 시나리오**: `framework=native attn=fa2 purpose=timing` 파드 런. `loader.describe`(loader.py:524)가 `kernels.read_fingerprint` 로 repo_id+revision 을 읽고 `AdapterOut.fingerprint` 에 담는다. `build_run`(bench.py:533)이 그것을 `binding.fingerprint` 로 받는다. 그리고 끝이다 — `main` 은 `build_record(config, device, applied=state, metrics=summary, applied_axes=applied)`(bench.py:851)과 OOM 경로(bench.py:840), 거부 경로(bench.py:450) 셋 다 fingerprint 를 넘기지 않는다. 결과 JSON 에 `build_fingerprint` 키가 없고, `scripts/report.py` 에도 그 키를 읽는 코드가 0건이다. 그래서 flash-attn 없는 이미지가 런 시작 중에 Hub 에서 내려받은 커널의 repo+revision 도, module_classes/parameter_dtypes/trainable_parameter_names 도 아티팩트에 남지 않는다. 두 파드가 서로 다른 revision 의 fa2 커널을 바인딩해도 결과 파일만으로는 구별 불가이고, 그것을 잡으라고 만든 `loader.fingerprint_diff` 는 읽을 데이터가 없다(프로덕션 호출자도 0건). 계약(`tests/contract/test_kernel_provenance.py:546-547`)은 `tests/fixtures/run_record.sample.json` 만 검증하므로 122 passed 로 초록이다 — HAZARDS §3 의 '검사가 통과하면서 아무것도 보지 않는다' 와 같은 모양이고, kernels 레인 노트가 'integrate 레인이 루트 문서에 올릴 것'으로 넘긴 것은 문서 문장이지 이 배선이 아니다.

**재현**:
```text
infisical run --env=dev -- uv run python -c "import ast,pathlib; t=ast.parse(pathlib.Path('scripts/bench.py').read_text()); print('binding.fingerprint 를 읽는 곳:', [ast.unparse(n) for n in ast.walk(t) if isinstance(n,ast.Attribute) and n.attr=='fingerprint'])"  # -> []  (선언 scripts/bench.py:485 뿐)
grep -rn 'build_fingerprint' scripts/ trainbench/record.py   # -> 0건
grep -rn 'RUN_RECORD_KEY' --include='*.py' . | grep -v '\.venv' | grep -v '^tests/'   # -> trainbench/kernels.py:48 정의 하나뿐, 소비자 없음
infisical run --env=dev -- uv run pytest tests/contract/test_kernel_provenance.py -q   # 초록. fixture 만 본다
```

**검증** (reproduced):
```text
임시 테스트(tests/test_zzverify_fingerprint.py, 실행 후 삭제)로 pod_setting 픽스처를 통해 scripts/bench.py::main 을 CPU 에서 끝까지 돌리고 결과 JSON 의 최상위 키를 출력: infisical run --env=dev -- uv run pytest tests/test_zzverify_fingerprint.py -q -s
---
qwen3_vl_emb_2b x native: 2 steps
EXIT 0
RECORD TOP-LEVEL KEYS: ['applied', 'applied_axes', 'config', 'device', 'git_commit', 'git_dirty', 'git_source', 'host', 'image', 'image_digest', 'metrics', 'packages', 'recorded_at']
HAS build_fingerprint: False
AssertionError: assert 'build_fingerprint' in {...}

보조 확인:
- grep -n 'fingerprint' scripts/bench.py -> scripts/bench.py:485 (Binding.fingerprint 선언) 한 줄뿐, 읽는 곳 0건
- grep -c 'fingerprint' scripts/report.py -> 0
- grep -rn 'RUN_RECORD_KEY' --include='*.py' . (venv 제외) -> trainbench/kernels.py:48 정의 + tests/ 만; 프로덕션 소비자 0건
- infisical run --env=de
```

### `precision-recipe-preempts-dtype-refusals` — blocker / correctness
- 단위: capture
- 위치: `trainbench/applied.py:1105`

**주장**: `built.precision_recipe` 가 있으면 `_capture_precision` 이 즉시 반환하므로 그 아래 dtype 기반 거부 네 개(`swapped`, `not base`, adapter 마커 누락, `mixed(...)`)가 하나도 실행되지 않고, 요청한 fp8 값이 무조건 인증된다.

**실패 시나리오**: config `precision.name=mxfp8`, `run.purpose=timing`. base 파라미터가 bf16 1개 + fp32 1개인 반쯤 변환된 모델에 `Built(model=m, precision_recipe=MXFP8BlockScaling())` 을 주면 capture 가 `applied='mxfp8'`, `matches=True` 를 돌려주고 `assert_matches` 가 통과한다. 같은 모델을 recipe 없이 주면 `applied='mixed(bf16,fp32)'`, `matches=False` 다. 파라미터가 전부 fp32 인 모델도 똑같이 `mxfp8` / `matches=True` 로 인증된다(recipe 없으면 `'fp32'` / False). 즉 `tests/contract/test_applied_axes.py:385-388` 이 "어떤 setting 에도 속하지 않는 런"으로 못박은 `mixed(bf16,fp32)` 와, `_capture_precision` docstring 이 "mismatch 여야 한다"고 적은 fp32 적재가 둘 다 요청한 라벨로 발행된다.

**재현**:
```text
tests/contract/test_repro_prec.py 를 만들어 실행:
```python
import torch
from tests.contract.test_applied_axes import bench, model, weights, tensor, instance, axis, config_mapping  # noqa: F401
from trainbench.applied import Built, capture

def test_recipe_preempts_every_dtype_reading(config_mapping):
    config = bench(config_mapping, **{"precision.name": "mxfp8", "run.purpose": "timing"})
    half = model(params=[*weights(1), ("head.weight", tensor(torch.float32))])
    allf32 = model(params=weights(dtype=torch.float32))
    for label, m in (("mixed", half), ("all-fp32", allf32)):
        e = axis(capture(Built(model=m, precision_recipe=instance("MXFP8BlockScaling")), config), "precision.name")
        e2 = axis(capture(Built(model=m), config), "precision.name")
        print(label, e.applied, e.matches, "| no recipe:", e2.applied, e2.matches)
```
`uv run pytest tests/contract/test_repro_prec.py -q -s -p no:randomly`
실측 출력: `mixed mxfp8 True | no recipe: mixed(bf16,fp32) False` / `all-fp32 mxfp8 True | no recipe: fp32 False`
```

**검증** (reproduced):
```text
uv run pytest tests/contract/test_repro_prec.py -q -s -p no:randomly
---
mixed mxfp8 True | no recipe: mixed(bf16,fp32) False
all-fp32 mxfp8 True | no recipe: fp32 False
.
1 passed in 0.81s
```

### `zero-engine-never-driven` — blocker / measurement-validity
- 단위: axes
- 위치: `trainbench/axes.py:1326`

**주장**: `_deepspeed` hands back an engine that nothing in this repository ever drives, and its docstring asserts deepspeed's delegation behaviour without reading the pinned source — so `parallel=zero2/zero3` and `train.offload=*` are certified from the engine's config while the measured step goes through `loss.backward()` and the pre-`initialize` torch optimizer.

**실패 시나리오**: A 2-rank pod run with `parallel=zero2 train.offload=optimizer`: `assemble` builds the engine, `capture` reads `engine.zero_optimization_stage()==2` and `engine.zero_offload_optimizer()=={"device":"cpu"}` (both are `_config.zero_config` reads, not behaviour), `assert_matches` passes, and `train()` then executes `(loss/grad_accum).backward()` (scripts/bench.py:315) and `built.optimizer.step()` (scripts/bench.py:326) on the exact torch optimizer instance handed to `initialize` — `tests/test_axes.py:2226` pins that identity. `engine.backward` / `engine.step` are never called anywhere in the repo. The docstring's justification — "What steps is still this instance — deepspeed's wrapper holds it and delegates" — is a claim about a framework whose wheel is installed nowhere in this checkout (`.plans/notes/axes.md` §7 says so), which is the assertion-without-the-pinned-source shape HAZARDS §1 is about. If the delegation does not hold, the pod publishes a plain single-process step time under the `zero2 + offload=optimizer` label with every gate green. `.plans/notes/axes.md` §5 lists what the pod must print for these axes and this is not on it.

**재현**:
```text
grep -rn "engine\.backward\|engine\.step\|\.backward(\|\.step()" scripts trainbench --include=*.py   # only scripts/bench.py:315 and :326
infisical run --env=dev -- uv run pytest tests/test_axes.py::test_the_optimizer_on_built_is_the_one_deepspeed_was_given -v   # pins built.optimizer is the pre-initialize instance
# then read trainbench/axes.py:1313-1358 and scripts/bench.py:256-326 side by side
```

**검증** (reproduced):
```text
grep -rn "engine\.backward|engine\.step|\.backward\(|\.step\(\)" scripts trainbench --include="*.py"  # only scripts/bench.py:315,326 (+probe/audit prose)
# then, tests/test_zz_repro.py (temporary, since removed): fake `deepspeed.initialize` returning a class named DeepSpeedEngine whose backward()/step() append to a DRIVEN log; parallel.strategy=zero2 + train.offload=optimizer, world_size=2 stubbed; axes.assemble -> applied.capture -> assert_matches(state, config) -> scripts/bench.py::train(built, built.dataloader, config, CPU)
infisical run --env=dev -- uv run pytest tests/test_zz_repro.py -v -s
---
APPLIED_NAMES ['optim.name', 'parallel.strategy', 'train.offload', 'loss.name', 'dataloader.backend', 'framework.name']
zero stage axis: zero2 zero2
offload axis: optimizer optimizer
assert_matches: PASSED (gate green)
built.optimizer is the pre-initialize instance: True
deepspeed config handed: {'stage': 2, 'offload_optimizer': {'device': 'cpu'}}
steps_measured: 4
engine methods driven during the measured loop: []
PASSED
============================== 1 passed in 0.79s ===============================

grep result:
scripts/bench.py:315:                    (loss / grad_accum).backward()
scripts
```

### `packed-isolation-dies-under-the-default-cache` — blocker / correctness
- 단위: collate-prompt
- 위치: `trainbench/collate.py:485`

**주장**: 팩 격리는 `past_key_values is None` 일 때만 생기는데, 팩을 먹이는 유일한 자리(`scripts/bench.py:101`)는 `use_cache=False` 를 넘기지 않아 세 모델 전부 `config.use_cache=True` 로 캐시를 만들고, qwen3_vl/gemma4 는 팩 전체에 대한 단일 인과 삼각형(격리 없음)을 얻고 qwen3_5 는 forward 가 죽는다.

**실패 시나리오**: `dataloader.packing=true`, `attn=sdpa`, 4개 시퀀스가 `cu_seqlens=[0,3,5,6,10]` 로 팩된 배치. 측정 루프의 `pooled_embeddings` 가 `model(**tensors, output_hidden_states=False)` 를 부른다 → `masking_utils._preprocess_mask_arguments` 의 `position_ids is not None and attention_mask is None and past_key_values is None` 이 캐시 때문에 거짓 → `find_packed_sequence_indices` 를 건너뛴다. 실측 마스크는 10x10 하한삼각 전부 1: 토큰 3(두 번째 시퀀스 시작)이 토큰 0~2(첫 시퀀스)를 읽는다. 즉 packing 런의 pooled embedding·InfoNCE 손실·어텐션 비용이 전부 격리되지 않은 것인데 런은 `dataloader.packing=True` 로 인증된다. arch=qwen3_5 는 대신 `ValueError: get_seq_length can only be called on Attention layers, and the current Cache seem to only contain LinearAttention layers` 로 죽어 packed 런이 시작조차 못 한다. `use_cache=False` 는 저장소 전체에서 `tests/test_collate.py:394,396,429,430` 네 줄에만 있고 프로덕션 코드에는 한 번도 없다 — 즉 `test_packing_isolation_is_block_diagonal_for_this_pack` 이 초록인 이유는 그 테스트가 `create_causal_mask(..., past_key_values=None)` 을 직접 부르기 때문이고, 실제 경로는 그 전제를 만족하지 않는다.

**재현**:
```text
uv run python /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/repro_packing_isolation.py  (스크립트가 없으면: 팩 배치를 `build_collate(...packing=True)(rows)` 로 만들고 `transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask` 를 감싼 뒤 `Qwen3VLTextModel(...).train()` 에 `model(**mb.tensors, output_hidden_states=False)` 를 부른다. 출력: `past_key_values present: True`, 마스크가 블록대각이 아닌 완전 하한삼각. 이어서 같은 호출을 `Qwen3_5TextModel` 에 하면 ValueError. `use_cache=False` 를 더하면 둘 다 정상.)  대조: grep -rn use_cache trainbench scripts tests → tests/test_collate.py 4줄뿐
```

**검증** (reproduced):
```text
uv run python /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/repro_packing_isolation.py ; uv run python /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/repro_pooled.py
---
$ uv run python .../repro_packing_isolation.py
cu_seqlens: [0, 3, 5, 6, 10]
past_key_values present: True
tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)
BLOCK-DIAGONAL: False
qwen3_5 RAISED: ValueError `get_seq_length` can only
```

### `refusals-have-no-caller` — blocker / emptiness
- 단위: kernels
- 위치: `trainbench/kernels.py:243`

**주장**: 완료 조건 3·4가 지키기로 한 두 거부(`assert_packing_is_isolated`, `assert_no_runtime_kernel_fetch`/`forbid_runtime_kernel_fetch`)는 `tests/test_kernels.py` 밖에서 호출하는 곳이 저장소 전체에 없다 — 런 경로에서 아무것도 거부하지 않는다.

**실패 시나리오**: `attn=fa3`(마스크 미등록 커널) + `dataloader.packing=true` 로 런을 돌린다. `loader.build_fingerprint` 는 `read_fingerprint` 만 부르므로 지문에 `mask_registered=false` 가 기록되지만 예외는 나지 않고, 스텝이 돌아 throughput 숫자가 나온다. 그 숫자는 pack 안의 시퀀스가 서로의 문맥이 된 배치의 숫자다 — `assert_packing_is_isolated` 가 막기로 한 바로 그 상태인데, 그 함수를 부르는 코드가 없다. 같은 런에서 `HF_HUB_OFFLINE` 이 안 걸려 있어도 `assert_no_runtime_kernel_fetch` 를 부르는 곳이 없어 커널이 런 시작 중에 내려받아진다. `.plans/notes/kernels.md` 는 adapters 레인(`read_fingerprint`)과 packing 레인(`cu_seqlens` 배선)만 요청하고 이 두 거부의 미배선은 적지 않았다 — 정직하게 보고된 미완이 아니다.

**재현**:
```text
grep -rn "assert_packing_is_isolated\|assert_no_runtime_kernel_fetch\|forbid_runtime_kernel_fetch" --include="*.py" trainbench/ scripts/ | grep -v tests/  → 정의 3줄 외에 0건. 대조군: grep -rn "read_fingerprint" trainbench/ scripts/ → trainbench/loader.py:258 에 실제 호출이 있다.
```

**검증** (reproduced):
```text
uv run python /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/repro_nocaller.py
---
attn.name = fa3 | attn.impl = flash_attention_3 | dataloader.packing = True
resolved impl in mask registry: False
open_fetch_doors() = ["$HF_HUB_OFFLINE=None, want '1'", "$USE_HUB_KERNELS=None, want 'NO'", 'huggingface_hub.constants.HF_HUB_OFFLINE=False, want True — cached at import, so the environment variable was set too late to reach it']
describe() returned with NO refusal
  resolved.attn_implementation = kernels-community/vllm-flash-attn3
  resolved.mask_registered     = False
  backbones                    = {'text_config': {'attn_implementation': 'kernels-community/vllm-flash-attn3', 'm
```

### `kernel-guards-have-no-caller` — blocker / emptiness
- 단위: macro:emptiness
- 위치: `trainbench/kernels.py:243`

**주장**: 이번 캠페인이 추가한 거부 셋 `assert_packing_is_isolated` / `forbid_runtime_kernel_fetch` / `assert_no_runtime_kernel_fetch` 은 자기 테스트 말고는 부르는 곳이 하나도 없어서 런타임 검사 대상이 0건인데, `docs/methodology.md:618,626-627` 은 이 셋이 조합을 거부하고 런을 세운다고 적는다.

**실패 시나리오**: `dataloader=torch_packed` + `attn=fa2` 를 flash-attn 없는 이미지에서 돌리면 transformers 가 요청을 `kernels-community/flash-attn2` 로 다시 써서 `AttentionMaskInterface._global_mapping` 에 없는 이름이 된다. `loader.build_fingerprint` -> `kernels.read_fingerprint` 는 그 상태를 `mask_registered: false` 로 기록하고 `validate_fingerprint` 는 그 payload 를 **정상으로 통과시킨다**(fixture 의 `fa3_hub_kernel_mask_unregistered_qwen3_vl` 이 valid sample 이다). 그 뒤 `scripts/bench.py` 는 아무 거부도 거치지 않고 팩을 그대로 잰다 — 시퀀스들이 서로의 컨텍스트인 배치의 throughput 이 `dataloader.packing=True` applied 로 인증되어 발행된다. `scripts/report.py` 에는 `mask_registered` 를 읽는 코드가 한 줄도 없어 결과 문서에도 표시되지 않는다. fetch 쪽도 같다: `docker/entrypoint.sh` 어디에도 `HF_HUB_OFFLINE`/`USE_HUB_KERNELS` 를 세우는 줄이 없고 `assert_no_runtime_kernel_fetch` 도 불리지 않으므로 측정 중 커널이 네트워크로 도착할 수 있다. 덤으로 `kernels.packing_isolation_holds({'backbones': {}})` 는 `True` 를 돌려준다 — backbones 가 비면 격리가 성립한다고 답한다.

**재현**:
```text
grep -rn "assert_packing_is_isolated\|forbid_runtime_kernel_fetch\|assert_no_runtime_kernel_fetch" scripts trainbench docker   # 정의 3줄 + docstring 1줄만 나오고 호출자는 없다
grep -rn "HF_HUB_OFFLINE\|USE_HUB_KERNELS" docker/entrypoint.sh   # 아무것도 안 나온다
uv run python -c "import json,pathlib;from trainbench import kernels;s=json.loads(pathlib.Path('tests/fixtures/kernel_fingerprint.sample.json').read_text())['samples'];u=s['fa3_hub_kernel_mask_unregistered_qwen3_vl'];kernels.validate_fingerprint(u);print('validate OK, isolation=',kernels.packing_isolation_holds(u));print('empty backbones ->',kernels.packing_isolation_holds({'backbones':{}}))"
# -> validate OK, isolation= False / empty backbones -> True
```

**검증** (reproduced):
```text
grep -rn "assert_packing_is_isolated\|packing_isolation_holds\|forbid_runtime_kernel_fetch\|assert_no_runtime_kernel_fetch" --include="*.py" --include="*.sh" . | grep -v "^./docs/"
grep -n "isolat\|UnsafePacking\|RuntimeKernelFetch" scripts/bench.py scripts/report.py trainbench/loader.py trainbench/applied.py docker/entrypoint.sh
grep -rn "HF_HUB_OFFLINE\|USE_HUB_KERNELS" docker/
uv run python -c "import json,pathlib;from trainbench import kernels;print(kernels.assert_packing_is_isolated.__code__.co_filename, kernels.assert_packing_is_isolated.__code__.co_firstlineno);s=json.loads(pathlib.Path('tests/fixtures/kernel_fingerprint.sample.json').read_text())['samples'];u=s['fa3_hub_kernel_mask_unregistered_qwen3_vl'];kernels.validate_fingerprint(u);print('validate OK, isolation=',kernels.packing_isolation_holds(u));print('empty backbones ->',kernels.packing_isolation_holds({'backbones':{}}))"
---
$ grep -rn "assert_packing_is_isolated|packing_isolation_holds|forbid_runtime_kernel_fetch|assert_no_runtime_kernel_fetch" --include="*.py" --include="*.sh" . | grep -v "^./docs/"
trainbench/kernels.py:14:what `assert_packing_is_isolated` refuses on.
trainbench/kernels.py:233:def packing_isolation_holds(fingerprint: Mapping[str, Any]) -> bool:
trainbench/kernels.py:243:def assert_packing_is_isolated(fingerprint: Mapping[str, Any]) -> None:
trainbench/kernels.py:479:def forbid_runtime_kernel_fetch(
trainbench/kernels.py:500:def assert_no_runtime_kernel_fetch(
tests/test_kernels.py:266,276,291,3
```

### `kernel-fetch-guard-has-no-caller` — blocker / measurement-validity
- 단위: macro:measurement
- 위치: `trainbench/kernels.py:479`

**주장**: `forbid_runtime_kernel_fetch` 와 `assert_no_runtime_kernel_fetch` 는 프로덕션 경로에 호출자가 하나도 없는데 `docs/methodology.md §11` 은 둘이 실제로 문을 닫고 런을 세운다고 현재형으로 단언한다.

**실패 시나리오**: flash-attn 이 없고 `kernels` 가 있는 파드에서 `attn=fa2` 로 timing 런을 시작하면 transformers 가 요청을 Hub 저장소 이름으로 바꿔(`modeling_utils.py:1997-2003`) 런 시작 중에 커널을 내려받는다. 이 저장소에는 그것을 막는 코드가 실행되지 않으므로 스텝 시간에 네트워크 fetch 가 섞이고 이미지 digest 가 커널을 고정하지 못한다. 실측: 프로덕션 적재 경로(`trainbench.loader`)와 `transformers` 를 import 한 직후 `kernels.open_fetch_doors()` 가 `["$HF_HUB_OFFLINE=None, want '1'", "$USE_HUB_KERNELS=None, want 'NO'", 'huggingface_hub.constants.HF_HUB_OFFLINE=False, want True — cached at import...']` 세 개를 돌려준다. `docker/`, `scripts/`, `configs/` 어디에도 그 두 변수를 세팅하는 곳이 없다.

**재현**:
```text
grep -rn "forbid_runtime_kernel_fetch\|assert_no_runtime_kernel_fetch" trainbench scripts docker configs   # 정의 3줄 외 호출자 0
grep -rn "HF_HUB_OFFLINE\|USE_HUB_KERNELS" docker/ scripts/ configs/                                        # 출력 없음
uv run python -c "import os; os.environ.pop('HF_HUB_OFFLINE',None); os.environ.pop('USE_HUB_KERNELS',None); import trainbench.loader, transformers; from trainbench import kernels; print(kernels.open_fetch_doors())"
# -> 문 3개가 열린 채로 나온다. docs/methodology.md:622-627 은 닫힌다고 적혀 있다.
```

**검증** (reproduced):
```text
grep -rn "forbid_runtime_kernel_fetch\|assert_no_runtime_kernel_fetch\|open_fetch_doors" trainbench scripts docker configs tests; grep -rn "HF_HUB_OFFLINE\|USE_HUB_KERNELS" docker/ scripts/ configs/; uv run python -c "import os; os.environ.pop('HF_HUB_OFFLINE',None); os.environ.pop('USE_HUB_KERNELS',None); import trainbench.loader, transformers; from trainbench import kernels; print(kernels.open_fetch_doors())"
---
grep(1): 호출자는 정의부(trainbench/kernels.py:448/479/500, 483/491/505 내부 참조)와 tests/test_kernels.py(328-389, 587-589)뿐. trainbench/(loader 등)/scripts/docker/configs 어디에도 호출자 0.
grep(2): docker/, scripts/, configs/ 에서 HF_HUB_OFFLINE / USE_HUB_KERNELS 히트 없음 (exit 1).
python: /Users/jwcho/Codes/train-comparison/trainbench/kernels.py 448
["$HF_HUB_OFFLINE=None, want '1'", "$USE_HUB_KERNELS=None, want 'NO'", 'huggingface_hub.constants.HF_HUB_OFFLINE=False, want True — cached at import, so the environment variable was set too late to reach it']
docs/methodology.md:626-627: "`kernels.forbid_runtime_kernel
```

### `unsloth-lora-refused-before-peft-attaches` — blocker / correctness
- 단위: loader-probe
- 위치: `trainbench/loader.py:307`

**주장**: `_refuse_a_build_the_fingerprint_condemns` runs at load time, before `axes.assemble` attaches LoRA, so every unsloth run with `peft.mode=lora`/`qlora` dies with AdapterRefusal on a frozen graph that is the expected intermediate state.

**실패 시나리오**: framework=unsloth, peft.mode=lora. `unsloth.load` passes `full_finetuning=False` (loader.py:548 -> probe/unsloth.py:42). Pinned source: `FastVisionModel.from_pretrained` calls `post_patch_model` (unsloth 2026.7.6 models/vision.py:1751, :2122), which calls `prepare_model_for_training(full_finetuning=False)`; unsloth_zoo 2026.7.7 training_utils.py:376-412 then does `param.requires_grad_(False)` for every name without `.lora_A.`/`.lora_B.` — and no LoRA module exists yet, because `axes._peft`/`get_peft_model` (axes.py:1077-1091) only runs later, inside `assemble` (bench.py:548). So `fingerprint['trainable_parameter_names'] == []` and `describe` raises `AdapterRefusal: unsloth built a model with no trainable parameter among N`. Result: unsloth x LoRA — half of the study's headline full-vs-LoRA comparison for one of six frameworks — cannot produce a number, and because AdapterRefusal is not caught by `refusing()` it produces no result file at all, only entrypoint's fallback record. peft.mode=full loads fine, so the guard fires on exactly the branch it was not written for.

**재현**:
```text
uv run python -c 'import json,sys,types,torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from trainbench.compose import resolve
from trainbench.config import to_bench_config
from trainbench.device import get_device
from transformers import Qwen3VLConfig
class B(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.config=Qwen3VLConfig(); s.config._attn_implementation="sdpa"; s.lin=torch.nn.Linear(2,2).to(torch.bfloat16)
def mk():
    m=types.ModuleType("unsloth"); m.__version__="0"
    class F:
        @staticmethod
        def from_pretrained(hf_id, load_in_4bit=False, full_finetuning=False, dtype=None):
            b=B()
            if not full_finetuning: [p.requires_grad_(False) for n,p in b.named_parameters() if ".lora_A." not in n]
            return b, types.SimpleNamespace(padding_side="right")
    m.FastVisionModel=F; return m
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    mp=resolve(compose(config_name="config", overrides=["run=probe","device=cpu"]))[1]
sys.modules["unsloth"]=mk()
from trainbench import loader
for mode in ("full","lora"):
    m=json.loads(json.dumps(mp)); m["framework"]["name"]="unsloth"; m["peft"]["mode"]=mode
    if mode!="full": m["peft"]["r"]=32
    try: print(mode, "OK", len(loader.load(to_bench_config(m), get_device("cpu")).fingerprint["trainable_parameter_names"]))
    except Exception as e: print(mode, type(e).__name__+":", str(e)[:90])'

Observed: `full OK trainable 2` / `lora AdapterRefusal: unsloth built a model with no trainable parameter among 2`. The freeze branch is verbatim from `sed -n '376,412p' /Users/jwcho/.cache/uv/archive-v0/PM921ZbVZCUP68sU/unsloth_zoo/training_utils.py`.
```

**검증** (reproduced):
```text
uv run python /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/repro.py   # stub-unsloth 주입 후 loader.load(peft.mode=full|lora)
---
live def: /Users/jwcho/Codes/train-comparison/trainbench/loader.py 295
full OK trainable 2
lora AdapterRefusal: unsloth built a model with no trainable parameter among 2; a step over a frozen graph still produces a number, and that
```

### `sentence-transformers-model-is-not-a-processor` — blocker / contract-split
- 단위: loader-probe
- 위치: `trainbench/probe/sentence_transformers.py:43`

**주장**: The ST adapter returns `(model, model)`, so `AdapterOut.processor` is a `SentenceTransformer`, but `scripts/bench.py:572` hands that object to `build_collate`, which calls it as an HF processor — every ST timing batch raises before a step runs.

**실패 시나리오**: framework=sentence_transformers, any model. `bench.py:572` does `built.dataloader.collate_fn = build_collate(processor, config)` where `processor is binding.model`. `trainbench/collate.py:326` then calls `self.processor(text=[...], return_tensors='pt', padding=True, truncation=True, max_length=N)`. Pinned sentence-transformers 5.6.1 defines `def forward(self, input: dict[str, Tensor], **kwargs)` at base/model.py:496 and `sentence_transformer/model.py` declares no override, so the call raises `TypeError: forward() missing 1 required positional argument: 'input'`. With `model.prompt_format=chat_template` (the default for qwen3_vl_emb_2b) it fails one step earlier, in `format_prompt`, because `SentenceTransformer` exposes no `chat_template` attribute (grep for `chat_template` in base/model.py and sentence_transformer/model.py returns nothing). Either way the sentence_transformers cell can never produce a number. `.plans/notes/adapters.md` §4 records the opposite conclusion — "processor(text=..., images=...) 규약을 받는가 -> 받는다" — but the citation (base/modules/transformer.py:1258-1290) is about the HF processor held *inside* the `Transformer` module, not about the `SentenceTransformer` the adapter hands out.

**재현**:
```text
uv run python -c 'import json, torch
from pathlib import Path
from hydra import compose, initialize_config_dir
from trainbench.compose import resolve
from trainbench.config import to_bench_config
from trainbench.collate import build_collate
class ST(torch.nn.Sequential):
    chat_template = "{{ messages }}"
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False): return "text"
    def forward(self, input, **kwargs): return {"sentence_embedding": torch.zeros(2)}
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    mp = resolve(compose(config_name="config", overrides=["run=probe","device=cpu"]))[1]
mp = json.loads(json.dumps(mp)); mp["framework"]["name"] = "sentence_transformers"
collate = build_collate(ST(torch.nn.Linear(2,2)), to_bench_config(mp))
try: collate([{"query":"q","positive":"p"}])
except Exception as e: print(type(e).__name__+":", e)'

Observed: `TypeError: ST.forward() missing 1 required positional argument: 'input'`. Drop the two `chat_template` lines from the stub to see the earlier `ValueError: model.prompt_format is 'chat_template' but this processor has none`. Signature check: grep -n 'def forward' /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/pins/sentence-transformers-5.6.1/sentence_transformers/base/model.py
```

**검증** (reproduced):
```text
uv run --with scikit-learn python /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/repro_st3.py   # (raw path)  및  .../repro_st2.py  (chat_template path). 두 스크립트 모두 sys.path 에 pins/sentence-transformers-5.6.1 을 넣고 실제 SentenceTransformer 클래스 인스턴스를 build_collate 에 넘긴다.
---
### raw path (model=gemma4_e2b, prompt_format=raw), 실제 SentenceTransformer 인스턴스
prompt_format: raw | real ST: True
SentenceTransformer.forward @ /private/tmp/.../pins/sentence-transformers-5.6.1/sentence_transformers/base/model.py 496
Traceback (most recent call last):
  File ".../repro_st3.py", line 25, in <module>
    collate([{"query":"q","positive":"p"}])
  File "/Users/jwcho/Codes/train-comparison/trainbench/collate.py", line 326, in __call__
    encoded = self.processor(**kwargs)
  File ".../torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
Ty
```

### `probe-preflight-renders-a-deliberate-refusal-as-a-lost-pod-hour` — major / correctness
- 단위: report-orchestrate
- 위치: `docker/entrypoint.sh:225`

**주장**: probe 갈래에 붙인 preflight 는 스키마 불일치뿐 아니라 `axes.patch`/`load_kwargs` 의 축 거부에도 발화하므로, 하네스가 의도적으로 거부한 조합이 probe 판정 대신 `결과 없음(기동됨)` 으로 렌더된다 — `scripts/report.py` 모듈 docstring 이 이 파일의 존재 이유로 적어둔 바로 그 구분이 뒤집힌다.

**실패 시나리오**: `configs/experiment/phase0-native-gemma4_e2b.yaml` 의 `overrides` 에 `peft=qlora`(또는 `precision=mxfp8`, `parallel=ddp` 등 `axes.py` 가 아직 거부하는 값)를 넣어 "이 프레임워크가 이 축을 받는가"를 묻는 probe 파드를 세운다. preflight 가 exit 4 로 거부하고 entrypoint 는 `verify_env.py` 를 아예 부르지 않은 채 `--mode fallback` 만 올린다. 그 레코드는 `status: no_result` 이므로 `report.cell` 이 `artifact.produced_result` 에서 걸려 `결과 없음(기동됨)` 을 낸다 — 파드가 뜨고 죽은 것과 구별되지 않고, `미지원(문서화됨)` 도 `FAIL <check>` 도 나오지 않는다. probe 목적은 `ENFORCED_PURPOSES` 에 없어서(`trainbench/applied.py:45`) 축 거부가 원래 probe 를 막지 않았다는 점이 이 갈래에서만 사라진다. 현재 등재된 18개 phase0 매니페스트는 전부 기본값이라 아직 발화하지 않는다.

**재현**:
```text
거부 확인: `uv run python -c "import sys,json,tempfile;sys.path.insert(0,'scripts');from pathlib import Path;import orchestrate as o,bench;cfg=o.resolved_config(['framework=native','model=gemma4_e2b','run=probe','data.limit=8','train.batch_size=8','peft=qlora']);p=Path(tempfile.mkdtemp())/'plan.json';p.write_text(json.dumps([{'name':'phase0-native-gemma4_e2b','role':'experiment','overrides':[],'config':cfg}]));print('exit',bench.preflight(p))"` -> `preflight REFUSED ... UnappliedAxis: peft.mode=qlora ...`, `exit 4`. 셀 확인: `uv run python -c "import sys,json,tempfile;sys.path.insert(0,'scripts');from pathlib import Path;import publish_result,report;cfg={'framework':{'name':'native'},'model':{'name':'gemma4_e2b'},'run':{'purpose':'probe'}};rec=publish_result.fallback_record(cfg,'preflight refused this pod config (exit 4)');root=Path(tempfile.mkdtemp());d=root/'results'/'native'/'gemma4_e2b'/'podA';d.mkdir(parents=True);(d/report.RESULT_NAME).write_text(json.dumps(rec));a,s=report.load_artifacts(root);l=report.split_lanes(a);c,_=report.newest_per_combination(l.matrix);print(report.cell(c.get(('native','gemma4_e2b')),[{'pod_id':'podA'}],None,{}))"` -> `결과 없음(기동됨)`.
```

**검증** (reproduced):
```text
(1) TRAINBENCH_CUDA_ARCHS=80 uv run python -c "import sys,json,tempfile;sys.path.insert(0,'scripts');from pathlib import Path;import orchestrate as o,bench;[print('OVERRIDES',e,'-> exit',bench.preflight((lambda p,c: (p.write_text(json.dumps([{'name':'phase0-native-gemma4_e2b','role':'experiment','overrides':e,'config':c}])), p)[1])(Path(tempfile.mkdtemp())/'plan.json', o.resolved_config(['framework=native','model=gemma4_e2b','run=probe','data.limit=8','train.batch_size=8']+e)))) for e in ([],['peft=qlora'])]"   (2) uv run python -c "import sys,json,tempfile;sys.path.insert(0,'scripts');from pathlib import Path;import publish_result,report;cfg={'framework':{'name':'native'},'model':{'name':'gemma4_e2b'},'run':{'purpose':'probe'}};rec=publish_result.fallback_record(cfg,'preflight refused this pod config (exit 4)');print('status=',rec.get('status'));root=Path(tempfile.mkdtemp());d=root/'results'/'native'/'gemma4_e2b'/'podA';d.mkdir(parents=True);(d/report.RESULT_NAME).write_text(json.dumps(rec));a,s=report.load_artifacts(root);l=report.split_lanes(a);c,_=report.newest_per_combination(l.matrix);print('CELL=',report.cell(c.get(('native','gemma4_e2b')),[{'pod_id':'podA'}],None,{}))"
---
(1) OVERRIDES [] ->
preflight: phase0-native-gemma4_e2b OK
preflight REFUSED this pod's GPU: no CUDA device is visible, but this pod was launched to measure on one (the image compiled kernels for sm_80).
exit 4
OVERRIDES ['peft=qlora'] ->
preflight REFUSED this pod's GPU: no CUDA device is visible, ...
preflight REFUSED phase0-native-gemma4_e2b: UnappliedAxis: peft.mode=qlora needs a 4-bit base and bitsandbytes quantises on CUDA; device=mps would load the base in full precision and measure that under the qlora label.
preflight: 1 of the 1 setting(s) it could compose, and this pod's GPU, cannot
```

### `baseline-note-attributes-precision-to-hardware-that-was-never-reached` — major / correctness
- 단위: audit
- 위치: `docs/audit-baseline.json:4`

**주장**: `axis-values` note 가 `precision 1/3` 을 "**하드웨어다**"(CC 10.x vs A100)로 단정하지만, 이 수를 만든 실행은 하드웨어 검사에 도달한 적이 없다 — transformer-engine 이 이 호스트에 없어 import 단계에서 거부된다. 같은 문장의 "감사가 원리적으로 못 보는 축 값" 목록도 두 방향으로 틀렸다.

**실패 시나리오**: 상태: 이 랩톱에 `transformer_engine` 미설치. `axis-values` 가 `precision=mxfp8` 를 시도하면 `axes._precision_supported` 가 `_import_or_refuse(TE_QUANTIZATION_MODULE, ...)` 에서 `UnappliedAxis: ... transformer_engine.pytorch.quantization is not importable here (No module named ...)` 로 끝난다. `is_mxfp8_available()` / CC 비교는 한 번도 실행되지 않는다. 잘못된 출력: baseline note 가 이 수를 "코드 결함이 아니라 하드웨어 사실" 로 확정하고 "recipe 컨텍스트는 구현했고 되읽기도 열렸다" 를 그 수의 근거인 것처럼 붙인다 — 이 게이트는 그 어느 쪽도 증명하지 않았다. 같은 note 의 마지막 문장 "패키지가 없는 축 값(adamw_8bit / qlora / liger / fla / dali)은 감사가 원리적으로 못 본다" 는 동일하게 패키지 부재로 못 보는 `mxfp8`/`nvfp4` 를 빠뜨리고, 반대로 `fla` 를 패키지 부재로 분류한다(실제로는 위 발견대로 arch 고정이라 패키지가 있어도 안 보인다). 결과: 독자가 "precision 은 다 됐고 A100 만 문제" 로 읽고 멈춘다 — 이 baseline 이 이미 한 번 기록한 오귀속("이전 note 가 적어둔 F(이미지 빌드)는 오귀속이었다")과 같은 모양이다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && uv run python -c "import transformer_engine"  # -> ModuleNotFoundError
cd /Users/jwcho/Codes/train-comparison && uv run python -c "
import sys; sys.path.insert(0,'.')
from trainbench import axes
for p,(cls,chk) in axes.TE_PRECISIONS.items():
    try: axes._precision_supported(p, chk); print(p,'supported')
    except Exception as e: print(p,'->',type(e).__name__,':',str(e)[:150])"
실측 출력: mxfp8/nvfp4 둘 다 `UnappliedAxis: precision=... has to be gated on the device that executes it, and transformer_engine.pytorch.quantization is not importable here`. 하드웨어 문장(`... is not executable on this device ... compute capability 10.x`)은 나오지 않는다.
```

**검증** (reproduced):
```text
cd /Users/jwcho/Codes/train-comparison && uv run python -c "
import sys; sys.path.insert(0,'.')
import torch
from hydra import compose, initialize_config_dir
from scripts.audit_plan import CONFIGS, AXIS_VALUE_BASE_OVERRIDES, AXIS_VALUE_COMPANIONS, _AxisValueRows, _AxisValueRowsWithImages
from trainbench import axes
from trainbench.compose import resolve
class T(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.block=torch.nn.Linear(4,4); s.block.gradient_checkpointing=False
    def gradient_checkpointing_enable(s,**k): pass
for v in ['precision/mxfp8','precision/nvfp4','kernel/fla']:
    g,n=v.split('/')
    ov=[f'{g}={n}',*AXIS_VALUE_BASE_OVERRIDES,*AXIS_VALUE_COMPANIONS.get(v,())]
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        c=resolve(compose(config_name='config', overrides=ov))[0]
    for shape,rows in [('text',_AxisValueRows),('image',_AxisValueRowsWithImages)]:
        try:
            axes.patch(c); axes.load_kwargs(c)
            ds=rows()
            if c.dataloader.pretokenize: ds=axes.pretokenize(ds, lambda r: dict(r))
            built,_=axes.assemble(T(), c, torch.device('cpu'),'native',dataset=ds)
            if built.dataloader is not None: next(iter(built.dataloader))
            with axes.step_context(c): pass
            print(v,shape,'APPLIED')
        except Exception as e: print(v,shape,'->',type(e).__name__,':',str(e)[:200])
"
---
precision/mxfp8 text -> UnappliedAxis : precision=mxfp8 runs the step inside a Transformer Engine recipe and has to be gated on the device that executes it, and transformer_engine.pytorch.quantization is not importable here (No module named
precision/mxfp8 image -> UnappliedAxis : precision=mxfp8 runs the step inside a Transformer Engine recipe and has to be gated on the device that executes it, and transformer_engine.pytorch.quantization is not importable here (No module named
precision/nvfp4 text -> UnappliedAxis : precision=nvfp4 runs the step inside a Transformer Engine recipe and has to b
```

### `stale-repo-line-citations` — major / measurement-validity
- 단위: kernels
- 위치: `docs/methodology.md:514`

**주장**: §10.1 이 근거로 인용한 저장소 내부 파일:줄 두 개가 리뷰 대상 트리에서 틀렸다 — 둘 다 무관한 docstring 을 가리킨다.

**실패 시나리오**: §10.1 의 결론('세 전제를 모두 만족하므로 격리는 라이브러리가 만든다', 'Qwen3.5 는 linear 레이어에서 격리 없이 돈다')을 확인하려는 독자가 `trainbench/axes.py:1405` 로 가면 Muon param_group docstring 이 나오고(실제 `PackedCollate.__call__` 은 1917행, 클래스는 1729행 — 512줄 차이), `trainbench/collate.py:429`(:537행에서 인용) 로 가면 `axis_packing` docstring 이 나온다(실제 `batch.pop(PACKED_BOUNDARY_KEYS)` 는 451행). :1405 는 base 커밋 fa5a325 에서만 맞았고 그 사이 axes.py 가 512줄 밀렸으며, collate.py 는 fa5a325 에 아예 존재하지 않았다. 레인 브리프가 '모든 정정은 파일:줄을 인용한다'를 요구한 근거가 검증 불가능해지고, 같은 잘못된 줄이 `.plans/notes/kernels.md:33` 으로 packing 레인에 그대로 전달된다.

**재현**:
```text
sed -n '1405p' trainbench/axes.py; grep -n 'class PackedCollate' trainbench/axes.py; grep -n 'def __call__' trainbench/axes.py | sed -n '/191/p'; sed -n '429p' trainbench/collate.py; grep -n 'PACKED_BOUNDARY_KEYS}' trainbench/collate.py  → 1405/429 는 docstring, 실제는 1917/451
```

**검증** (reproduced):
```text
sed -n '1405p' trainbench/axes.py; grep -n 'class PackedCollate' trainbench/axes.py; awk 'NR>=1729 && NR<=1960 && /def __call__/ {print NR": "$0}' trainbench/axes.py; sed -n '429p;451p' trainbench/collate.py; grep -n 'axes.py:1405\|collate.py:429' docs/methodology.md .plans/notes/kernels.md
---
    way of telling an embedding matrix from a hidden weight matrix needs the names
---
trainbench/axes.py:1729:class PackedCollate:
1917:     def __call__(self, rows: list[Any]) -> dict[str, torch.Tensor]:
---
    `applied._capture_dataloader_packing` asks, and two answers is how one of them
        boundaries = {key: batch.pop(key) for key in axes.PACKED_BOUNDARY_KEYS}
---
docs/methodology.md:514:`axes.PackedCollate.__call__`(`trainbench/axes.py:1405`)이 내는 dict의 키는
docs/methodology.md:537:그리고 `trainbench/collate.py:429`가 `axes.PACKED_BOUNDARY_KEYS`로 `cu_seqlens`와
.plans/notes/kernels.md:33:- 
```

### `axis-values-fla-has-no-companion-so-it-can-never-be-counted` — major / measurement-validity
- 단위: audit
- 위치: `scripts/audit_plan.py:1603`

**주장**: `AXIS_VALUE_COMPANIONS` has no entry for `kernel/fla`, so `axis-values` composes it against the default model `qwen3_vl` and gets refused on architecture — the value is uncountable in every environment, including an image that has fla and CUDA.

**실패 시나리오**: 입력: `configs/config.yaml` 의 기본 `model: qwen3_vl_emb_2b` (arch `qwen3_vl`) + `kernel=fla` override, 즉 `axis-values` 가 실제로 만드는 조합. `axes.FLA_ARCHS == frozenset({'qwen3_5'})` (trainbench/axes.py:138) 이므로 `patch()` 가 `axes.py:396` 에서 `UnappliedAxis: kernel=fla on arch=qwen3_vl: transformers takes no fla kernel path` 로 거부한다 — 패키지·CUDA 질문에 도달하기 전이다. 잘못된 출력: `kernel` 그룹은 fla 를 설치한 native 이미지에서도 영원히 `1/3` 또는 `2/3` 이고 `kernel/fla` 는 절대 applicable 로 세어지지 않는다. 즉 `axes._patch_fla` 를 통째로 비워도 이 게이트의 수는 어느 환경에서도 움직이지 않는다. `freeze/ple`·`freeze/vision_and_ple` 가 `model=gemma4_e2b` 동반값을 받는 이유(주석: "miscounted as inapplicable for a reason that has nothing to do with whether axes.py can apply it")와 정확히 같은 상황인데 fla 만 빠져 있다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && uv run python - <<'EOF'
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import audit_plan as ap
from hydra import compose, initialize_config_dir
from trainbench import axes
from trainbench.compose import resolve
for extra in ([], ["model=qwen3_5_0_8b"]):
    with initialize_config_dir(config_dir=str(ap.CONFIGS), version_base=None):
        c = resolve(compose(config_name="config", overrides=["kernel=fla", *ap.AXIS_VALUE_BASE_OVERRIDES, *extra]))[0]
    try:
        axes.patch(c); print(extra or "default", "-> applied")
    except Exception as e:
        print(extra or "default", "->", type(e).__name__, str(e)[:110])
EOF
실측 출력(2026-08-03, 이 트리): default -> UnappliedAxis kernel=fla on arch=qwen3_vl: transformers takes no fla kernel path... / ['model=qwen3_5_0_8b'] -> UnappliedAxis kernel=fla cannot take the Gated DeltaNet fast path here: fla not installed.
변이: scripts/audit_plan.py:1603 의 AXIS_VALUE_COMPANIONS 에 `"kernel/fla": ("model=qwen3_5_0_8b",),` 를 넣고 `infisical run --env=dev -- uv run python scripts/audit_plan.py --only axis-values` 를 다시 돌리면 거부 사유가 arch 고정에서 환경 의존으로 바뀐다(이 랩톱에서는 여전히 1/3 이지만 이유가 달라진다).
```

**검증** (reproduced):
```text
cd /Users/jwcho/Codes/train-comparison && uv run python - <<'EOF'
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import audit_plan as ap
from hydra import compose, initialize_config_dir
from trainbench import axes
from trainbench.compose import resolve
print("fla entry:", ap.AXIS_VALUE_COMPANIONS.get("kernel/fla"))
for extra in ([], ["model=qwen3_5_0_8b"]):
    with initialize_config_dir(config_dir=str(ap.CONFIGS), version_base=None):
        c = resolve(compose(config_name="config", overrides=["kernel=fla", *ap.AXIS_VALUE_BASE_OVERRIDES, *extra]))[0]
    try:
        axes.patch(c); print(extra or "default", "-> applied")
    except Exception as e:
        print(extra or "default", "->", type(e).__name__, str(e)[:140])
EOF
# then: mutation A (empty axes._patch_fla) and mutation B (add "kernel/fla": ("model=qwen3_5_0_8b",) to AXIS_VALUE_COMPANIONS), each followed by
infisical run --env=dev -- uv run python scripts/audit_plan.py --only axis-values
---
fla entry: None
default -> UnappliedAxis kernel=fla on arch=qwen3_vl: transformers takes no fla kernel path for this architecture; only ['qwen3_5'] import fla. The run would carry t
['model=qwen3_5_0_8b'] -> UnappliedAxis kernel=fla cannot take the Gated DeltaNet fast path here: fla not installed. transformers falls back to the torch implementation with one lo

# baseline gate
KNOWN axis-values  36/52 applicable on both text-only/image-carrying data; 3 group(s) offering one usable value: kernel 1/3, precision 1/3, train.offload 1/4

# mutation A — axes._patch_fla emptied to `return ["kernel.na
```

### `adapter-refusal-escapes-refusing` — major / correctness
- 단위: bench
- 위치: `scripts/bench.py:418`

**주장**: `refusing()` 이 `axes.UnappliedAxis` 와 `AppliedMismatch` 두 가지만 잡아서, adapters 레인이 새로 만든 세 번째 거부 `loader.AdapterRefusal` 이 `main` 을 통째로 빠져나가고 결과 파일이 하나도 남지 않는다.

**실패 시나리오**: `framework=unsloth peft.mode=full` 파드 런. `loader.load` → `describe` → `_refuse_a_build_the_fingerprint_condemns`(`trainbench/loader.py:305`)가 `trainable_parameter_names == []` 를 보고 `AdapterRefusal("unsloth built a model with no trainable parameter among N ...")` 을 던진다. 이것은 2026-08-02 캠페인이 실제로 겪은 상태이고(HAZARDS §1, `full_finetuning=False` 기본값), 이 스터디가 리포트에 실어야 할 **결과**다. 그런데 `AdapterRefusal(RuntimeError)` 은 `UnappliedAxis` 도 `AppliedMismatch` 도 아니므로 `with refusing("load_kwargs")`(bench.py:527)를 그대로 통과하고, `main` 의 `except RefusedSetting`(bench.py:800)에도 걸리지 않고, OOM `try` 는 그보다 뒤에서 시작하므로 아예 도달하지 않는다. 프로세스가 traceback 과 함께 exit 1 로 죽고 `--out` 이 안 써진다. `docker/entrypoint.sh:365` 가 `[[ -s ${RESULT_PATH} ]]` 실패로 `--mode fallback --reason "no result file after the run (exit 1)"` 을 올리고, 리포트에는 "unsloth 가 전부 얼렸다"가 아니라 "아무도 시도하지 않은 조합"으로 렌더된다. bench.py 의 모듈 docstring(:23-26)이 "A setting those calls refuse still writes --out" 이라고 적어둔 바로 그 성질이 새 거부 타입에 대해 깨져 있다.

**재현**:
```text
```
cat > /tmp/adapterrefusal_plugin.py <<'PY'
import trainbench.loader as L
def _refuse(framework, fingerprint, context):
    raise L.AdapterRefusal(f"{framework} built a model with no trainable parameter among 3")
L._refuse_a_build_the_fingerprint_condemns = _refuse
PY
PYTHONPATH=/tmp infisical run --env=dev -- uv run pytest tests/test_smoke_cpu.py -q -p adapterrefusal_plugin -k gradient_norm_and_the_parameter_counts
```
실측 결과: `E trainbench.loader.AdapterRefusal: native built a model with no trainable parameter among 3` 로 테스트가 FAILED — `main()` 이 예외를 그대로 올렸다는 뜻이고, REFUSED_EXIT 도 레코드도 없다. 비교용으로 `grep -n "AdapterRefusal" scripts/bench.py` → 0건.
```

**검증** (reproduced):
```text
tests/test_smoke_cpu.py 에 임시 테스트를 붙여 main() 을 엔드투엔드로 돌림(pod_setting 픽스처, CPU): monkeypatch로 trainbench.loader._refuse_a_build_the_fingerprint_condemns 가 AdapterRefusal 을 던지게 한 뒤 `infisical run --env=dev -- uv run pytest tests/test_smoke_cpu.py -q -s -k TEMP_adapter_refusal_escapes`. 테스트 본문: with pytest.raises(loader_mod.AdapterRefusal): pod_setting(timing_config(config_mapping), rows(8)); print(pod_setting.out.exists()). 대조: `grep -c "AdapterRefusal" scripts/bench.py` -> 0
---
1차 실행(예외 미포획 버전):
scripts/bench.py:799: in main
    built, applied, state, binding = build_run(config, device)
scripts/bench.py:533: in build_run
    binding = load_framework(config, device)
trainbench/loader.py:552: in load
    return describe(adapter, model, processor, config, **fingerprint_kwargs)
trainbench/loader.py:525: in describe
    _refuse_a_build_the_fingerprint_condemns(
E       trainbench.loader.AdapterRefusal: native built a model with no trainable parameter among 3
FAILED tests/test_smoke_cpu.py::test_TEMP_adapter_refusal_escapes

2차 실행(raises 로 감싼 버전):
ESCAPED: AdapterRefusal -
```

### `owned-axes-exempts-without-routing-the-step` — major / measurement-validity
- 단위: bench
- 위치: `scripts/bench.py:554`

**주장**: `binding.owned_axes` 는 `assemble` 로 전달해 축을 검증에서 면제시키면서 `binding.step`(누가 스텝을 도는지)은 `scripts/bench.py` 어디에서도 읽지 않아, 하네스가 자기 loss 로 잰 런이 "프레임워크가 계산했다"로 기록된다.

**실패 시나리오**: tevatron 어댑터는 `step=Step(owner="framework", callable="tevatron.retriever.modeling.DenseModel.forward", batch_keys=("query","passage"))` 와 `owned_axes={"loss.name": ...}` 를 함께 선언한다(`trainbench/loader.py:454-460`, 결정 5). `build_run` 은 `owned_axes` 만 `axes.assemble` 로 넘기고 `step` 은 버린다 — `grep -c "binding.step" scripts/bench.py` = 0. 그래서 `train()` 은 owner 와 무관하게 언제나 `built.loss_fn` 으로 InfoNCE 를 계산하고 `built.optimizer.step()` 을 돌면서, 레코드에는 `loss.name` 이 `state: framework_owned`, `applied: null`, `detail.reason: "<fw> computes loss.name inside its own training step, so this run's value is not ours to read back"` 로 남는다. 실측(아래 재현): 어댑터가 framework-owned step 을 선언한 상태로 `main()` 을 끝까지 돌렸더니 exit 0, 완전한 `metrics` 블록, `applied.framework_owned == ['loss.name']` 이 동시에 나왔다. 즉 `owned_axes` 는 `assert_matches` 의 면제만 주고 스텝 라우팅은 아무도 하지 않는다 — 면제는 실효, 그 면제가 정당하다고 말하는 선언은 무효. 오늘 tevatron 칸은 `steps.encode` 가 `input_ids`/`attention_mask` 를 넘겨 `DenseModel.forward(query=,passage=)` 에서 TypeError 로 죽을 가능성이 크지만(그것도 §2 처럼 결과 파일 없이 죽는다), collate 가 `query`/`passage` 키를 만들게 되는 순간 이 조합은 **크래시가 아니라 잘못 라벨된 숫자**가 된다 — HAZARDS 가 `loss=cached_mnrl` 로 이미 한 번 겪은 모양 그대로.

**재현**:
```text
```
cat > /tmp/framework_step_plugin.py <<'PY'
import dataclasses
import trainbench.loader as L
native = L.ADAPTERS["native"]
L.ADAPTERS["native"] = dataclasses.replace(
    native,
    step=L.Step(owner=L.FRAMEWORK,
                callable="tevatron.retriever.modeling.DenseModel.forward",
                batch_keys=("query", "passage")),
    owned_axes={"loss.name": "DenseModel.forward computes it"},
)
PY
cat > /tmp/probe_plugin.py <<'PY'
import trainbench.record as R
_orig = R.write_json
def write_json(path, payload):
    ap = payload.get("applied") or {}
    print("HAS_METRICS:", "metrics" in payload)
    print("FRAMEWORK_OWNED:", ap.get("framework_owned"))
    return _orig(path, payload)
R.write_json = write_json
PY
PYTHONPATH=/tmp infisical run --env=dev -- uv run pytest tests/test_smoke_cpu.py -q -p framework_step_plugin -p probe_plugin -k gradient_norm_and_the_parameter_counts -s
```
실측 출력: `HAS_METRICS: True`, `FRAMEWORK_OWNED: ['loss.name']`, `1 passed` — 하네스 루프가 loss 를 계산했는데도 레코드는 프레임워크 소유라고 적는다.
```

**검증** (reproduced):
```text
git archive HEAD | tar -x -C $S  (깨끗한 HEAD 사본; 공유 트리에 다른 레인의 변이가 실시간으로 들어와 있어 격리 사본에서 실행)
# tests/test_smoke_cpu.py 말미에 임시 테스트 추가: 기존 adapter_binding() 헬퍼로
# framework="tevatron", step=Step(owner=FRAMEWORK, callable="tevatron.retriever.modeling.DenseModel.forward",
# batch_keys=("query","passage")), owned_axes={"loss.name": ...} 를 선언한 Binding 으로 main() 을 끝까지 실행
cd $S && infisical run --env=dev -- uv run --project $S pytest tests/test_smoke_cpu.py -q -k ZZZ_verifier -s
grep -c "binding.step" $S/scripts/bench.py
---
qwen3_vl_emb_2b x native: 2 steps
  step p50 0.0005s  p95 0.0101s
  samples/s 754.87  rows/s 1509.75  tokens/s 33969.34081203512
wrote .../result.json
EXIT: 0
HAS_METRICS: True
GRAD_NORM: 2.1050214699476224e-06
FRAMEWORK_OWNED: ['loss.name']
LOSS_AXIS: [{'applied': None, 'axis': 'loss.name', 'detail': {'reason': "tevatron computes loss.name inside its own training step, so this run's value is not ours to read back"}, 'determined': False, 'matches': False, 'owner': 'tevatron', 'requested': 'mnrl', 'state': 'framework_owned'}]
.
1 passed, 67 deselected in 1.62s
GREP binding.step: 0
```

### `bench-record-drops-build-fingerprint` — major / contract-split
- 단위: bench
- 위치: `scripts/bench.py:851`

**주장**: `loader.load` 가 매 런 계산하는 build fingerprint 를 `main` 이 버려서, 결과 레코드에 `build_fingerprint` 키가 존재한 적이 없다.

**실패 시나리오**: 파드에서 `framework=native attn=fa2` 를 잰다. `trainbench/loader.py:258` 이 `kernels.read_fingerprint(...)` 로 `attention` 블록(resolved.attn_implementation, identity.repo_id/revision, mask_registered)을 만들어 `AdapterOut.fingerprint` 에 담고, `build_run` 이 그 binding 을 `main` 으로 돌려준다. 그런데 `main` 은 `build_record(config, device, applied=state, metrics=summary, applied_axes=applied)` 만 부른다 — `binding.fingerprint` 는 어디에도 전달되지 않는다. 실측: 실제 `main()` 런이 쓴 레코드의 최상위 키는 ['applied','applied_axes','config','device','git_commit','git_dirty','git_source','host','image','image_digest','metrics','packages','recorded_at'] 13개이고 `build_fingerprint` 가 없다. 반면 `tests/fixtures/run_record.sample.json` 은 그 키를 싣고 있고(`attention`/`module_classes`/`parameter_dtypes`/`buffer_dtypes`/`trainable_parameter_names`), `trainbench/kernels.py:48` 과 `tests/contract/test_kernel_provenance.py:61` 이 `RUN_RECORD_KEY = "build_fingerprint"` 로 그 자리를 못박는다. 결과: fa2 요청이 Hub 커널로 내려갔는지, `mask_registered` 가 서서 packing 격리가 실제로 생겼는지, 프레임워크가 파라미터 dtype 을 바꿨는지가 **어떤 아티팩트에도 남지 않는다.** `kernel_provenance` 계약의 `packing_isolation_holds` 는 실측 런에 대해 평가될 수 없고, `fingerprint_diff` 로 프레임워크 간 confound 를 비교하겠다는 설계가 입력을 못 받는다. 계약 테스트가 이것을 못 잡는 이유는 `tests/contract/test_record_report.py:363` 의 `unseen = set(produced) - set(payload)` 가 **한 방향만** 보기 때문이다(HAZARDS §3 의 `plan-files` 와 같은 모양).

**재현**:
```text
(a) 산출물 대조: `python3 -c "import json;print('build_fingerprint' in json.load(open('tests/fixtures/run_record.sample.json')))"` → True, `grep -c build_fingerprint scripts/bench.py trainbench/record.py` → 둘 다 0. (b) 실제 런에서 확인 — 스크래치패드에 플러그인을 만들고
```
cat > /tmp/probe_plugin.py <<'PY'
import trainbench.record as R
_orig = R.write_json
def write_json(path, payload):
    print("RECORD_TOPLEVEL_KEYS:", sorted(payload))
    return _orig(path, payload)
R.write_json = write_json
PY
PYTHONPATH=/tmp infisical run --env=dev -- uv run pytest tests/test_smoke_cpu.py -q -p probe_plugin -k gradient_norm_and_the_parameter_counts -s
```
→ `RECORD_TOPLEVEL_KEYS:` 에 `build_fingerprint` 없음. (c) 변이: `tests/test_smoke_cpu.py::test_a_probe_record_carries_the_gradient_norm_and_the_parameter_counts` 에 `assert "build_fingerprint" in record` 한 줄 추가 후 같은 -k 로 실행 → AssertionError.
```

**검증** (reproduced):
```text
PYTHONPATH=<scratchpad> infisical run --env=dev -- uv run pytest tests/test_smoke_cpu.py -q -p probe_plugin -k gradient_norm_and_the_parameter_counts -s   # then mutation: assert "build_fingerprint" in record
---
RECORD_TOPLEVEL_KEYS: ['applied', 'applied_axes', 'config', 'device', 'git_commit', 'git_dirty', 'git_source', 'host', 'image', 'image_digest', 'metrics', 'packages', 'recorded_at']
qwen3_vl_emb_2b x native: 2 steps
1 passed, 66 deselected in 1.18s

# mutation run
>       assert "build_fingerprint" in record
E       AssertionError: assert 'build_fingerprint' in {'applied': {'all_determined': False, 'all_matched': False, 'axes': [{'applied': 'sdpa', 'axis': 'attn.name', 'detail'...': False, ...}, 'dataloader': {'backend': 'torch', 'packing': False, 'pretokenize': False}, ...}, 'device': 'cpu', 
```

### `build-fingerprint-never-written-to-record` — major / contract-split
- 단위: macro:contracts
- 위치: `scripts/bench.py:851`

**주장**: `kernel-provenance` 페이로드를 담는 build fingerprint 는 `loader.describe` 가 매 런 계산해 `Binding.fingerprint` 로 넘겨받지만, `build_record` 호출 셋 중 어느 것도 그것을 싣지 않아 실제 결과 JSON 에는 `build_fingerprint` 키가 존재하지 않는다.

**실패 시나리오**: 파드에서 timing 런이 성공하면 `build_record(config, device, applied=state, metrics=summary, applied_axes=applied)` 가 기록을 쓴다. 여기에 `build_fingerprint=binding.fingerprint` 가 없으므로, 어떤 커널이 바인딩됐는지(`attention.resolved.identity` 의 repo_id+revision), 파라미터 dtype 분포, trainable 집합이 파드가 사라지는 순간 전부 소실된다. 그런데 `tests/fixtures/run_record.sample.json:188` 은 그 키를 싣고 있고 `test_kernel_provenance.py::test_the_two_fixtures_that_carry_this_payload_agree_with_it` 이 `record["build_fingerprint"]["attention"]` 을 검증한다 — 즉 계약은 초록인데 검사 대상이 fixture 뿐이고 생산자에는 그 키가 없다(HAZARDS §3 의 '아무것도 안 보는 초록'). `test_the_stored_sample_is_the_shape_the_producer_writes` 는 `set(produced) - set(payload)` 한 방향만 보므로 생산자가 빠뜨린 키를 구조적으로 못 잡는다. 결과: transformers 가 `flash_attention_2` 요청을 런 시작 중에 Hub repo 로 바꿔 받아오는 경우(HAZARDS §6) 어떤 커널로 잰 숫자인지 사후에 답할 수 없다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && grep -n 'build_record(' scripts/bench.py && grep -c 'binding.fingerprint' scripts/bench.py  →  build_record 호출 3곳 모두 build_fingerprint 인자 없음, binding.fingerprint 참조 0. 변이 확인: `python -c "import json;d=json.load(open('tests/fixtures/run_record.sample.json'));del d['build_fingerprint'];json.dump(d,open('tests/fixtures/run_record.sample.json','w'))"` 뒤 `uv run pytest tests/contract/test_kernel_provenance.py -q` → 죽는다. 그 키를 살려두는 것이 fixture 하나뿐임을 보여준다(되돌리기: git checkout tests/fixtures/run_record.sample.json).
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python scripts/env_report.py device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4 ; uv run python -c "import json,glob;p=sorted(glob.glob('outputs/qwen3_5_0_8b-native-20260803-*/env_report.json'))[-1];d=json.load(open(p));print(sorted(d));print('build_fingerprint present:', 'build_fingerprint' in d)" ; uv run python - <<'PY' (ast scan of every build_record call in scripts/bench.py) ; python -c "import json;p='tests/fixtures/run_record.sample.json';d=json.load(open(p));del d['build_fingerprint'];json.dump(d,open(p,'w'),indent=2)" ; infisical run --env=dev -- uv run pytest tests/contract/test_kernel_provenance.py tests/contract/test_record_report.py -q ; git checkout -- tests/fixtures/run_record.sample.json
---
outputs/…/env_report.json top-level keys: ['applied', 'config', 'device', 'git_commit', 'git_dirty', 'git_source', 'host', 'image', 'image_digest', 'packages', 'recorded_at'] / build_fingerprint present: False — 그리고 fixture에서 그 키를 지우면 tests/contract/test_kernel_provenance.py:547 KeyError: 'build_fingerprint' (1 failed, 39 passed), 반대로 test_the_stored_sample_is_the_shape_the_producer_writes 는 초록 유지.
```

### `report-reimplements-the-library-training-verdict` — major / duplication
- 단위: report-orchestrate
- 위치: `scripts/report.py:384`

**주장**: `scripts/report.py::training_verdict` is a second production implementation of `trainbench/metrics/validity.py::training_verdict`, which this same merge added and which no production code calls — the speed table's validity gate and the library's gate can diverge with nothing red.

**실패 시나리오**: measure 레인이 `validity.training_verdict` 를 조인다 (예: `steps_measured > 0` 를 추가하거나 `last >= first` 를 허용오차로 완화한다). `tests/test_metrics.py` 는 라이브러리를 부르므로 따라가고 초록을 유지한다. `scripts/report.py` 는 자기 사본을 그대로 들고 있으므로 병합 리포트는 라이브러리가 거부하는 런을 순위표에 계속 싣거나(전자), 라이브러리가 통과시키는 런을 `학습하지 않은 런` 으로 떨어뜨린다(후자). 두 경우 다 아무 테스트도 실패하지 않는다 — `tests/test_report.py` 와 `tests/contract/test_record_report.py` 는 각자 자기 정의만 본다. 부수적으로 `trainbench.metrics.training_verdict` 는 지금 상태로 프로덕션 호출자가 0이다(`grep -rn training_verdict --include=*.py` 기준 호출은 tests/ 뿐).

**재현**:
```text
1) 어느 정의를 잡는지 먼저 확인: `uv run python -c "from trainbench.metrics.validity import training_verdict as t; print(t.__code__.co_filename, t.__code__.co_firstlineno)"` -> `trainbench/metrics/validity.py 156`. 2) 그 함수 첫 문장으로 `return False, ["MUTANT: library verdict refuses everything"]` 를 넣는다. 3) `infisical run --env=dev -- uv run pytest tests/test_report.py -q` -> 이 세션 실측 `25 passed`, 속도 표는 그대로 렌더된다. 4) 원복. (`scripts/report.py` 에는 `import trainbench` 가 한 줄도 없다.)
```

**검증** (mutation-killed-nothing):
```text
python3 - (insert `return False, ["MUTANT: library verdict refuses everything"]` as first statement of trainbench/metrics/validity.py::training_verdict) && uv run python -c "from trainbench.metrics.validity import training_verdict as t; print(t.__code__.co_filename, t.__code__.co_firstlineno); print(t({'grad_norm':1.0,'trainable_params':1,'total_params':1,'loss_first':2.0,'loss_last':1.0}, peft_mode='full', device='cuda:0'))" && infisical run --env=dev -- uv run pytest tests/test_report.py -q && infisical run --env=dev -- uv run pytest tests/contract/test_record_report.py -q && infisical run --env=dev -- uv run pytest tests/test_metrics.py -q
---
live-definition check:
/Users/jwcho/Codes/train-comparison/trainbench/metrics/validity.py 156
(False, ['MUTANT: library verdict refuses everything'])

tests/test_report.py:
.........................                                                [100%]
25 passed in 0.46s

tests/contract/test_record_report.py:
.....................                                                    [100%]
21 passed in 0.63s

tests/test_metrics.py (mutated):
FAILED tests/test_metrics.py::test_the_validity_gate_agrees_with_the_record_it_will_be_applied_to
10 failed, 25 passed in 0.55s

after revert, tests/test_me
```

### `report-verdict-third-copy-uncompared` — major / duplication
- 단위: macro:emptiness
- 위치: `scripts/report.py:384`

**주장**: `training_verdict` 구현이 세 벌인데 `scripts/report.py` 의 것과 `trainbench/metrics/validity.py` 의 것을 같은 payload 로 대조하는 테스트가 하나도 없고(`tests/test_report.py` 는 `report.training_verdict` 를 직접 부르지도 않는다), 두 구현은 이미 `peft.mode` 해석에서 갈라진다.

**실패 시나리오**: `config` 블록에 `peft` 키가 없는 레코드(구 캠페인 아티팩트, 또는 프레임워크 소유 스텝이 채운 레코드)를 주면 `scripts/report.py::training_verdict` 는 `mode=None` 이 되어 peft 분기를 통째로 끄고 `(True, [])` 를 돌려주는 반면, 동결된 계약 사본은 `payload['config']['peft']['mode']` 로 직접 접근해 `KeyError` 로 죽는다. 즉 `docs/report.md` 를 실제로 찍는 구현이 세 벌 중 가장 조용한 쪽이고, 그 침묵을 잡을 대조가 없다. kernels 레인은 같은 상황(계약 validator vs 런타임 validator)에 대해 `tests/test_kernels.py::test_the_runtime_validator_agrees_with_contract` 를 만들어 두고 모듈 docstring 에 "that comparison is the only thing keeping them one rule" 이라고 적었는데, measure/report 쌍에는 그 대조가 없다.

**재현**:
```text
uv run python - <<'PY'
import importlib.util,json,sys,pathlib
sys.path.insert(0,'.')
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
c=load('_c','tests/contract/test_record_report.py'); r=load('_r','scripts/report.py')
p=json.loads(pathlib.Path('tests/fixtures/run_record.sample.json').read_text())
p['config'].pop('peft'); p['metrics']['trainable_params']=12
print('report.py:',r.training_verdict(p))
try: print('contract :',c.training_verdict(p))
except Exception as e: print('contract : raised',type(e).__name__)
PY
# -> report.py: (True, [])   contract : raised KeyError
```

**검증** (reproduced):
```text
uv run python scratchpad/repro.py  # 리뷰어 스크립트 + 각 정의의 co_filename/co_firstlineno 출력. 보강: scripts/report.py:393 앞에 `return True, []` 변이 후 uv run pytest -q tests/test_report.py tests/contract/test_record_report.py tests/test_metrics.py
---
report.py DEF /Users/jwcho/Codes/train-comparison/scripts/report.py 384
contract DEF /Users/jwcho/Codes/train-comparison/tests/contract/test_record_report.py 296
validity DEF /Users/jwcho/Codes/train-comparison/trainbench/metrics/validity.py 156
report.py: (True, [])
contract : raised KeyError 'peft'
validity : raised TypeError training_verdict() missing 2 required keyword-only arguments: 'peft_mode' and 'device'

(변이 실행) 1 failed, 80 passed in 0.81s
FAILED tests/contract/test_record_report.py::test_a_run_that_trained_nothing_is_not_published_as_a_speed_result
```

### `baseline-tolerance-declared-but-report-hardcodes-3pct` — major / duplication
- 단위: macro:measurement
- 위치: `scripts/report.py:99`

**주장**: 이번 캠페인이 신설한 `measurement.baseline_tolerance` 는 레코드에만 실리고, 파드 판정을 실제로 내리는 것은 `report.py` 에 그대로 박혀 있는 `BASELINE_DEVIATION_LIMIT = 0.03` 이다 — 같은 임계값의 진실 소스가 둘이다.

**실패 시나리오**: 첫 파드가 노이즈 바닥을 재고 `+measurement.baseline_tolerance=0.081 +measurement.baseline_tolerance_calibrated=true` 로 캠페인을 돌리면(`tests/test_config.py:306` 이 이미 그 경로를 고정한다) 모든 run 레코드의 `metrics.measurement` 는 `baseline_tolerance: 0.081, baseline_tolerance_status: "calibrated"` 를 싣는다. 그런데 `baseline_gate` 의 `deviation > BASELINE_DEVIATION_LIMIT` 은 여전히 0.03 이고 리포트는 `임계값 3%` 를 출력한다. 편차 5% 파드가 레코드상 통과인데 리포트에서는 `POD_INVALID` 로 버려지고, 그 파드의 모든 측정치가 표에서 빠진다. HAZARDS §8 이 '3% 는 근거 없는 상수이고 첫 파드가 유도해야 한다' 고 적은 바로 그 값이 유도돼도 아무 데도 도달하지 못한다.

**재현**:
```text
grep -rn "baseline_tolerance" --include='*.py' . | grep -v '\./\.venv'   # config_schema 정의 + metrics 기록 + 테스트뿐, 소비자 0
grep -n "BASELINE_DEVIATION_LIMIT" scripts/report.py                    # 99, 551, 897
# 변이: trainbench/config_schema.py:282 의 기본값을 0.5 로 바꾸고
uv run pytest -q tests/test_report.py    # 전부 초록 — 리포트의 판정은 이 값을 읽지 않는다
```

**검증** (reproduced):
```text
uv run python scratchpad/repro_tolerance_gate.py   # 그리고 변이: config_schema.py:282 default 0.03 -> 0.5 후 uv run pytest -q tests/test_report.py
---
config.measurement.baseline_tolerance = 0.081
config.measurement.tolerance_status   = calibrated
record metrics.measurement = {"declared": true, "repeats": 1, "instrument": "wall_clock", "aggregate": "mean", "trim_fraction": 0.0, "seed_policy": "fixed", "throughput_denominator": "tokens", "baseline_tolerance": 0.081, "baseline_tolerance_status": "calibrated"}
reference=1.0 slowpod deviation=0.0500 status=무효
report.BASELINE_DEVIATION_LIMIT = 0.03
REPORT: | attn-fa3 | slowpod | native x gemma4_e2b | timing | 2.0000 | 2.0000 | 2.0000 | 16.00 | 4096.0 | 20.00 | 12/2/10 | 무효 (편차 5.00%) |
REPORT: 임계
```

### `deepspeed-config-keys-unasserted` — major / emptiness
- 단위: axes
- 위치: `tests/test_axes.py:2206`

**주장**: Two of the four keys `_deepspeed_config` hands `deepspeed.initialize` are asserted by nothing, and the one key that is asserted cannot tell a micro-batch from a total batch because every composed config in this repo has `grad_accum == 1`.

**실패 시나리오**: Three independent edits to `trainbench/axes.py:1374-1382` each leave the whole suite at 1147 passed: (a) `"train_micro_batch_size_per_gpu": config.train.batch_size * config.train.grad_accum` — the assertion at tests/test_axes.py:2206 compares against `config.train.batch_size` and `grad_accum` is 1 in `configs/train/default.yaml`, the only place it is set, so the multiplication is invisible; on a pod with `train.grad_accum=4` deepspeed would derive a 16x `train_batch_size` and partition/accumulate for a workload the config never asked for; (b) `"gradient_accumulation_steps": 1` hardcoded — the harness still feeds 4 micro-batches per timed step while deepspeed thinks each one is a step boundary; (c) deleting `"bf16": {"enabled": True}` — every ZeRO run then falls to deepspeed's default numeric regime while the run is labelled `precision=bf16`, and `applied._capture_precision` reads the weights' dtype, not the engine's regime, so nothing catches it.

**재현**:
```text
git worktree add /tmp/mut d372af7 && ln -s $PWD/.venv /tmp/mut/.venv
# in /tmp/mut/trainbench/axes.py apply any one of:
#   "train_micro_batch_size_per_gpu": config.train.batch_size  ->  ... * config.train.grad_accum
#   "gradient_accumulation_steps": config.train.grad_accum      ->  1
#   delete the line  "bf16": {"enabled": True},
cd /tmp/mut && infisical run --env=dev -- ./.venv/bin/python -m pytest -q   # 1147 passed for all three (measured this session)
```

**검증** (mutation-killed-nothing):
```text
git worktree add $SCRATCH/mut d372af7 && ln -s /Users/jwcho/Codes/train-comparison/.venv $SCRATCH/mut/.venv; cd $SCRATCH/mut; # baseline, then each of the three mutations to trainbench/axes.py:1378-1381, each followed by:
infisical run --env=dev -- ./.venv/bin/python -m pytest -q
---
baseline (unmutated d372af7): "1147 passed, 14 warnings in 105.11s (0:01:45)"
(a) "train_micro_batch_size_per_gpu": config.train.batch_size * config.train.grad_accum -> "1147 passed, 14 warnings in 99.62s (0:01:39)"
(b) "gradient_accumulation_steps": 1 -> "1147 passed, 14 warnings in 102.39s (0:01:42)"
(c) delete "bf16": {"enabled": True}, -> "1147 passed, 14 warnings in 101.49s (0:01:41)"
grep -rn "gradient_accumulation_steps|\"bf16\"|train_micro_batch|train_batch_size" tests/ trainbench/ 에서 deepspeed config 딕셔너리를 보는 단언은 tests/test_axes.py:2206 (`assert handed["train_micro_batch_size_per_gpu"
```

### `shared-load-test-passes-on-steps-load-kwargs` — major / emptiness
- 단위: loader-probe
- 위치: `tests/test_loader.py:526`

**주장**: `test_the_probe_and_the_harness_take_the_same_load` matches `\bload(?:_\w+)?\(config`, which `steps.load_kwargs(config, report)` satisfies, so for native and sentence_transformers the test stays green even if `run` stops calling the shared `load` entirely — the drift it exists to prevent.

**실패 시나리오**: The regex hits in each run body are: native `['load_kwargs(config', 'load_processor(config', 'load_model(config']`, sentence_transformers `['load_kwargs(config', 'load(config']`. Replace `model, _ = load(config, device, load_kwargs)` in `sentence_transformers.run` with an inline `SentenceTransformer(...)` construction — i.e. reintroduce exactly the second definition the diff removed — and the run body still contains `steps.load_kwargs(config, report)`, so the test passes. The same holds for native: inline `AutoModel.from_pretrained` in `run` and `load_kwargs(config` alone keeps it green. Per HAZARDS §3 this is a check that passes while looking at nothing, and its docstring claims to guard "two definitions of `full_finetuning=` or of axolotl's validate/normalize order" — the class of bug that cost a whole campaign.

**재현**:
```text
uv run python -c "import re, pathlib
pat = re.compile(r'\\bload(?:_\\w+)?\\(config')
for n in ('native','sentence_transformers','unsloth','ms_swift','axolotl','tevatron'):
    body = pathlib.Path('trainbench/probe/%s.py' % n).read_text().partition('\ndef run(')[2]
    print(n, [m.group(0) for m in pat.finditer(body)])"

Then the mutation, in memory (no repo edit):
uv run python -c "import re, pathlib
s = pathlib.Path('trainbench/probe/sentence_transformers.py').read_text()
mut = s.replace('model, _ = load(config, device, load_kwargs)', 'from sentence_transformers import SentenceTransformer as _ST; model = _ST(config.model.hf_id)')
print('still green:', bool(re.search(r'\\bload(?:_\\w+)?\\(config', mut.partition('\ndef run(')[2])))"
-> still green: True
```

**검증** (mutation-killed-nothing):
```text
uv run python -c "import re,pathlib; pat=re.compile(r'\bload(?:_\w+)?\(config'); [print(n,[m.group(0) for m in pat.finditer(pathlib.Path('trainbench/probe/%s.py'%n).read_text().partition('\ndef run(')[2])]) for n in ('native','sentence_transformers','unsloth','ms_swift','axolotl','tevatron')]"  # then on-disk mutation of trainbench/probe/sentence_transformers.py:62 (`model, _ = load(config, device, load_kwargs)` -> inline `_ST(config.model.hf_id)`), followed by: uv run pytest tests/test_loader.py -q  AND  infisical run --env=dev -- uv run pytest -q  ; restored with: git checkout -- trainbench/probe/sentence_transformers.py
---
$ regex hits per run body (unmutated)
native ['load_kwargs(config', 'load_processor(config', 'load_model(config']
sentence_transformers ['load_kwargs(config', 'load(config']
unsloth ['load(config']
ms_swift ['load(config']
axolotl ['load(config']
tevatron ['load_dense_model(config']

$ after mutation, grep -n "load(config\|_ST(" trainbench/probe/sentence_transformers.py
27:def load(config: BenchConfig, device: torch.device, load_kwargs: dict[str, Any]) -> tuple[Any, Any]:
63:        model = _ST(config.model.hf_id)

$ uv run python -c "import trainbench.probe.sentence_transformers as m; ..."
mo
```

### `empty-system-turn-guard-is-untested` — major / emptiness
- 단위: collate-prompt
- 위치: `tests/test_prompt.py:274`

**주장**: `test_a_row_with_no_instruction_prompt_sends_no_system_turn` 은 `_Templated.apply_chat_template` 가 `messages[-1]` 만 렌더하기 때문에 system 턴이 붙었는지를 볼 수 없고, 그것이 유일하게 그 가드를 주장하는 테스트다.

**실패 시나리오**: `trainbench/prompt.py:146` 의 `if instruction_prompt:` 를 `if True:` 로 바꾸면 `instruction_prompt: null` 인 두 모델(`qwen3_5_0_8b`, `gemma4_e2b`)과 모든 positive 행이 빈 system 턴(`<|im_start|>system\n<|im_end|>\n`)을 하나씩 더 싣게 되어 모든 qwen3.5 행의 시퀀스 길이가 늘고 tokens/s 의 분모가 바뀐다. 그런데 전체 스위트 1147개가 전부 통과한다 — 이 테스트의 docstring 이 "an empty system turn is not the same row as no system turn" 이라고 적어둔 바로 그 구별을 스텁이 지워버린다. `_QwenTemplated`(같은 파일 :39)는 `messages[0]['role']` 을 보므로 그것을 쓰면 죽는다.

**재현**:
```text
cp trainbench/prompt.py /tmp/p.bak; uv run python -c "import pathlib;p=pathlib.Path('trainbench/prompt.py');s=p.read_text();p.write_text(s.replace('    if instruction_prompt:\n','    if True:\n'))"; uv run python -c "from trainbench.prompt import format_prompt; print(format_prompt.__code__.co_filename, format_prompt.__code__.co_firstlineno)"; uv run pytest -q | tail -3; cp /tmp/p.bak trainbench/prompt.py   → 관측: co_firstlineno=100 (변이본이 잡힌다), `1147 passed`
```

**검증** (mutation-killed-nothing):
```text
cp trainbench/prompt.py /tmp/p.bak; uv run python -c "import pathlib;p=pathlib.Path('trainbench/prompt.py');s=p.read_text();n=s.replace('    if instruction_prompt:\n','    if True:\n');assert n!=s;p.write_text(n)"; uv run python -c "from trainbench.prompt import format_prompt as f; import inspect; print(f.__code__.co_filename, f.__code__.co_firstlineno); print('MUTANT-LIVE' if 'if True:' in inspect.getsource(f) else 'MUTANT-NOT-IN-LIVE-DEF')"; infisical run --env=dev -- uv run pytest -q tests/test_prompt.py; infisical run --env=dev -- uv run pytest -q; cp /tmp/p.bak trainbench/prompt.py
---
grep -n "if True:" trainbench/prompt.py
146:    if True:

/Users/jwcho/Codes/train-comparison/trainbench/prompt.py 100
MUTANT-LIVE

infisical run --env=dev -- uv run pytest -q tests/test_prompt.py
.............                                                            [100%]
13 passed in 0.01s

infisical run --env=dev -- uv run pytest -q   (mutation live, full suite)
FAILED tests/contract/test_applied_axes.py::test_a_framework_owned_axis_does_not_block_a_reportable_run
FAILED tests/test_applied.py::test_a_disclaimed_axis_lets_the_run_measure_and_says_who_has_it
FAILED tests/test_axes.py::test
```

### `precision-capture-is-a-mirror-of-its-own-construction` — major / measurement-validity
- 단위: macro:axis-pipeline
- 위치: `trainbench/applied.py:1105`

**주장**: `_capture_precision` 은 recipe 가 있으면 `axes.assemble` 이 같은 config 에서 방금 만든 recipe 객체의 클래스 이름만 되읽어 축 값으로 돌려주고, 모델 쪽 증거(`PRECISION_MODULE_ROOTS` 스캔)를 계산해 놓고 그 분기에서 버린다 — 요청을 그대로 되돌려주는 거울이다.

**실패 시나리오**: CC 10.x 파드에서 `precision=mxfp8`. `axes.assemble` 은 `_recorded_precision_recipe(config)`(axes.py:835)로 config 에서 `MXFP8BlockScaling()` 를 만들어 `Built.precision_recipe` 에 싣는다(axes.py:685-687, 714). 파이프라인 어디에서도 모델의 모듈을 Transformer Engine 모듈로 교체하지 않는다 — `_apply_to_model`(axes.py:1007-1016)은 freeze/peft/checkpointing/parallel/compile 뿐이다. 그런데 `_capture_precision` 은 `swapped = set(_module_roots(model)) & PRECISION_MODULE_ROOTS` 를 계산한 직후, recipe 분기(applied.py:1105-1115)에서 **그것을 보지 않고** `PRECISION_RECIPE_AXIS['MXFP8BlockScaling'] -> 'mxfp8'` 를 돌려준다. 실측(이 호스트): 전부 torch 모듈 + bf16 파라미터인 `nn.Linear` 에 recipe 스텁 하나를 붙이면 `applied='mxfp8'`, `matches=True`, detail 은 `{'base': {'bf16': 2}, 'adapter': {}, 'recipe': 'MXFP8BlockScaling'}` — 'transformer_engine 모듈 0개' 라는 사실이 detail 에도 판정에도 남지 않는다. 즉 이 축의 되읽기는 `config.precision.name` 을 한 바퀴 돌려준 것과 정보량이 같고, `applied.py` 자기 모듈 docstring(:8-9)이 이 모듈의 존재 이유로 적은 'fp8 recipes no-op on unsupported hardware' 를 이 축에 대해서만 잡지 못한다. (완화: 이 스터디 파드는 A100 이라 `_precision_supported` 가 mxfp8/nvfp4 를 거부하므로 이번 캠페인에서는 도달하지 않는다. 그래서 blocker 가 아니라 major 다.)

**재현**:
```text
infisical run --env=dev -- uv run python -c "
import sys; sys.path.insert(0,'.')
import torch
from hydra import compose, initialize_config_dir
from trainbench import applied
from trainbench.compose import resolve
with initialize_config_dir(config_dir='configs', version_base=None):
    cfg = resolve(compose(config_name='config', overrides=['run=timing','device=cpu','model=qwen3_5_0_8b','framework=native','precision=mxfp8','data.limit=4','train.batch_size=4']))[0]
class MXFP8BlockScaling: pass
m = torch.nn.Linear(4,4).to(torch.bfloat16)
print('TE/torchao 모듈:', sorted(set(applied._module_roots(m)) & set(applied.PRECISION_MODULE_ROOTS)))
b = applied.Built(model=m, precision_recipe=MXFP8BlockScaling(), framework='native')
print(applied._capture_precision(b, cfg))"
# 출력: TE/torchao 모듈: []   ('mxfp8', {'base': {'bf16': 2}, 'adapter': {}, 'recipe': 'MXFP8BlockScaling'})
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python -c "import sys; sys.path.insert(0,'.'); import torch; from hydra import compose, initialize_config_dir; from trainbench import applied; from trainbench.compose import resolve; print('def site:', applied._capture_precision.__code__.co_filename, applied._capture_precision.__code__.co_firstlineno); ..." (전문: configs 로 initialize_config_dir 후 overrides=['run=timing','device=cpu','model=qwen3_5_0_8b','framework=native','precision=mxfp8','data.limit=4','train.batch_size=4'] 로 resolve, class MXFP8BlockScaling: pass, m=torch.nn.Linear(4,4).to(torch.bfloat16), applied._module_roots(m) 교집합 출력, applied.Built(model=m, precision_recipe=MXFP8BlockScaling(), framework='native') 로 applied._capture_precision(b, cfg) 출력)
---
def site: /Users/jwcho/Codes/train-comparison/trainbench/applied.py 1069
requested: mxfp8
TE/torchao modules: []
('mxfp8', {'base': {'bf16': 2}, 'adapter': {}, 'recipe': 'MXFP8BlockScaling'})
```

### `optim-kind-lower-fallback-bypasses-table` — major / correctness
- 단위: capture
- 위치: `trainbench/applied.py:390`

**주장**: `return kind.lower(), detail` 는 `OPTIM_CLASS_AXIS` 를 우회하는 두 번째 매핑이며 config 가 제공하는 `muon` 을 만들어낼 수 있어, `kind == "Muon"` 의 newton-schulz 게이트를 건너뛴 채 축을 인증한다. 덤으로 `OPTIM_CLASS_AXIS["Muon"]` 은 이 폴백에 가려 죽은 항목이다.

**실패 시나리오**: config `optim.name=muon`. 프레임워크 어댑터가 클래스명 `MUON` 인 옵티마이저를 만들면 `kind != "Muon"` 이라 `newton_schulz_tensors` 검사가 통째로 생략되고, 표에도 없으니 `kind.lower()` == `"muon"` 이 되어 `applied='muon'`, `matches=True` 로 인증된다. 즉 use_muon 그룹에 학습 가능한 텐서가 하나도 없어 전부 내부 AdamW 경로로 도는 옵티마이저가 Muon 의 처리량으로 발행된다 — `_capture_optim` docstring 이 "다른 Muon 은 undetermined 가 맞다"고 적어둔 바로 그 경우. 실측: `MUON -> applied=muon matches=True newton=None`, `Muon -> applied=None matches=False newton=0`, `SomeVendorMuon -> applied=somevendormuon matches=False`.

**재현**:
```text
(a) 인증 재현 — tests/contract/test_repro_optim.py:
```python
import pytest
from tests.contract.test_applied_axes import bench, model, optimizer, weights, axis, config_mapping  # noqa: F401
from trainbench.applied import Built, capture

@pytest.mark.parametrize("cls", ["MUON", "Muon", "SomeVendorMuon"])
def test_casing(config_mapping, cls):
    config = bench(config_mapping, **{"optim.name": "muon", "run.purpose": "timing"})
    e = axis(capture(Built(model=model(params=weights()), optimizer=optimizer(cls, params=weights())), config), "optim.name")
    print(cls, e.applied, e.matches, e.detail.get("newton_schulz_tensors"))
```
`uv run pytest tests/contract/test_repro_optim.py -q -s -p no:randomly`
(b) 폴백 무검사 — `trainbench/applied.py:390` 을 `return None, detail` 로 바꾸고 `uv run pytest tests/ -q -p no:randomly` → 1147 passed (변화 없음)
(c) 죽은 표 항목 — `    "Muon": "muon",` 줄을 지우고 같은 명령 → 1147 passed (변화 없음)
```

**검증** (reproduced):
```text
uv run pytest tests/contract/test_repro_optim.py -q -s -p no:randomly  # (a) 재현; 이어서 (b) applied.py:390 -> `return None, detail`, (c) `"Muon": "muon",` 삭제 후 각각 `uv run pytest tests/ -q -p no:randomly`
---
$ uv run pytest tests/contract/test_repro_optim.py -q -s -p no:randomly
MUON muon True None
.Muon None False 0
.SomeVendorMuon somevendormuon False None
.
3 passed in 0.82s

기준선: $ uv run pytest tests/ -q -p no:randomly
1147 passed, 14 warnings in 99.08s (0:01:39)

(b) applied.py:390 -> `return None, detail`:
1147 passed, 14 warnings in 97.93s (0:01:37)

(c) `    "Muon": "muon",` 삭제:
1147 passed, 14 warnings in 97.53s (0:01:37)
git diff --stat -> trainbench/applied.py | 1 -
```

### `ddp-device-ids-guard-unverified` — major / emptiness
- 단위: axes
- 위치: `trainbench/axes.py:1301`

**주장**: The `device_ids` derivation whose comment says a guessed index "would put every rank's replica on device 0" is covered by no assertion — the only ddp test stubs `torch.nn.parallel.DistributedDataParallel` with a lambda that discards the argument.

**실패 시나리오**: Replace line 1301 with `ids = [0]` and the full suite stays at 1147 passed. On a multi-GPU pod every rank then constructs its DDP replica pinned to `cuda:0`, which is precisely the harm the comment claims to prevent; and the half that *is* testable on this CPU host — a model whose parameters have no device index must yield `device_ids=None`, because torch's DDP rejects `device_ids` for a CPU module — is never asserted either, since the stub at tests/test_axes.py:2075 never records what it was passed.

**재현**:
```text
grep -rn device_ids tests trainbench   # only trainbench/axes.py:1302 and the stub signature tests/test_axes.py:2075
git worktree add /tmp/mut d372af7 && ln -s $PWD/.venv /tmp/mut/.venv
# /tmp/mut/trainbench/axes.py:1301 -> `        ids = [0]`
cd /tmp/mut && infisical run --env=dev -- ./.venv/bin/python -m pytest -q   # 1147 passed (measured this session)
```

**검증** (mutation-killed-nothing):
```text
git worktree add $SCRATCH/mut d372af7 && ln -s /Users/jwcho/Codes/train-comparison/.venv $SCRATCH/mut/.venv && perl -i -pe 's/^        ids = \[device\.index\].*$/        ids = [0]/' $SCRATCH/mut/trainbench/axes.py && cd $SCRATCH/mut && infisical run --env=dev -- ./.venv/bin/python -m pytest -q
---
1147 passed, 14 warnings in 102.14s (0:01:42)
```

### `repeats-knob-recorded-as-applied-without-a-loop` — major / measurement-validity
- 단위: metrics-schema
- 위치: `trainbench/config_schema.py:263`

**주장**: `measurement.repeats` 는 소비자가 없는데 레코드에는 선언값 그대로 실린다 — 같은 모듈이 `trim_fraction` 에 대해 금지한 "적용된 것으로 기록되면서 아무것도 바꾸지 않는 knob" 이다.

**실패 시나리오**: `+measurement.repeats=10` 으로 런을 돌리면 `scripts/bench.py` 에 repeat 루프가 없으므로(문자열 `repeats` 가 파일에 0회 등장) 런은 한 번 돈다. 그런데 `metrics.measurement.repeats == 10` 이 레코드에 실리고, `step_seconds_stdev` 는 한 런 내부의 스텝 분산이지 10회 반복의 분산이 아니다. 실측: `summarise(..., config=compose_cfg('+measurement.repeats=10'))` 가 `{repeats: 10, declared: true}` 와 `steps_timed: 6` 을 동시에 낸다. `MeasurementConfig._the_trim_and_the_aggregate_agree` 는 같은 모양(`trim_fraction` under `mean`)을 ValidationError 로 거부하는데 `repeats` 에는 그 가드가 없다. `.plans/notes/wire.md §7` 은 루프 미구현을 정직하게 적었으나 "`measurement` 블록은 이미 레코드에 실리므로 `repeats: 1` 이 결과에 보인다" 고 안심시키며, 그 서술은 기본값에서만 참이다.

**재현**:
```text
grep -c 'repeats' scripts/bench.py   # 0
uv run python -c "
import json
from tests.test_config import compose_cfg
from trainbench.metrics import summarise
s = summarise([0.1]*6, discard=2, rows_per_step=8, tokens_per_step=1000, padded_tokens_per_step=2000, config=compose_cfg('+measurement.repeats=10'))
print(s['measurement']['repeats'], s['steps_timed'], s['step_seconds_stdev'])"
```

**검증** (reproduced):
```text
grep -rn "repeats" --exclude-dir=.venv --exclude-dir=.git . ; uv run python -c "from tests.test_config import compose_cfg; from trainbench.metrics import summarise; s=summarise([0.1]*6, discard=2, rows_per_step=8, tokens_per_step=1000, padded_tokens_per_step=2000, config=compose_cfg('+measurement.repeats=10')); print(s['measurement'], s['steps_timed'])"
---
$ grep -c 'repeats' scripts/bench.py
0

$ uv run python -c "... summarise(..., config=compose_cfg('+measurement.repeats=10')) ..."
{
 "declared": true,
 "repeats": 10,
 "instrument": "wall_clock",
 "aggregate": "mean",
 "trim_fraction": 0.0,
 "seed_policy": "fixed",
 "throughput_denominator": "tokens",
 "baseline_tolerance": 0.03,
 "baseline_tolerance_status": "uncalibrated"
}
steps_timed 6 stdev 0.0

$ uv run python -c "compose_cfg('+measurement.trim_fraction=0.2')"
ValidationError 1 validation error for BenchConfig
measurement
  Value error, measurement.trim_fraction=0.2 is set under aggrega
```

### `baseline-tolerance-knob-has-no-consumer` — major / duplication
- 단위: metrics-schema
- 위치: `trainbench/config_schema.py:282`

**주장**: `measurement.baseline_tolerance` 를 읽는 소비자가 없다 — 파드 판정은 `scripts/report.py:99` 의 하드코딩된 `BASELINE_DEVIATION_LIMIT = 0.03` 로 내려진다.

**실패 시나리오**: 파드가 노이즈 바닥을 재고 `+measurement.baseline_tolerance=0.081 +measurement.baseline_tolerance_calibrated=true` 로 런을 돌리면, 레코드의 `metrics.measurement` 는 `baseline_tolerance: 0.081, baseline_tolerance_status: "calibrated"` 를 싣는다. 그런데 `scripts/report.py:551` 의 `over = deviation > BASELINE_DEVIATION_LIMIT` 는 여전히 0.03 을 쓰므로 편차 5% 인 파드가 무효 처리되고, 같은 표 옆에 `BASELINE_DEVIATION_SOURCE` 가 "미교정 임계값이다" 를 출력한다 — 발행된 판정이 그 판정을 계산했다고 주장하는 레코드와 정면으로 모순된다. 브리프 작업 6 은 "config 에서 읽고" 를 요구했고 스키마 쪽만 착지했다. `.plans/notes/*.md` 어디에도 이 단절이 적혀 있지 않다.

**재현**:
```text
grep -n 'baseline_tolerance' scripts/report.py   # 0 hits
grep -n 'BASELINE_DEVIATION_LIMIT' scripts/report.py  # 99, 551
uv run python -c "from tests.test_config import compose_cfg as c; print(c('+measurement.baseline_tolerance=0.081','+measurement.baseline_tolerance_calibrated=true').measurement.baseline_tolerance)"
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python scratchpad/repro_tol.py   # composes +measurement.baseline_tolerance=0.081 +measurement.baseline_tolerance_calibrated=true, writes 3 baseline artifacts (1.00/1.00/1.05 s), runs report.baseline_gate + report.render
---
config tolerance: 0.081 calibrated
record measurement block: {'baseline_tolerance': 0.081, 'baseline_tolerance_status': 'calibrated'}
podA: deviation=0.0 verdict=OK note=
podB: deviation=0.0 verdict=OK note=
podC: deviation=0.050000000000000044 verdict=무효 note=임계값 초과 — 이 파드의 결과는 다른 파드와 같은 표에 들어갈 수 없다
report.BASELINE_DEVIATION_LIMIT = 0.03
임계값 3% — **미교정 임계값이다.** docs/methodology.md §4가 근거 없는 값이라고 명시한다 — 동일 pod에서 baseline을 5회 반복해 편차를 실측한 뒤 확정한다. 실측 편차가 이 값을 넘으면 임계값이 아니라 측정 절차를 고쳐야 한다는 신호다. 아래 판정은 그 교정 전의 잠정 판정이다.
| podC | 1.0500 | 5.00% | 무효 | 임계값 초과 — 이 파드의 결과는 다른 파드와 같은 표에 들어갈 수 없다 |
- podC: 
```

### `packing-isolation-refusal-has-no-caller` — major / dead-code
- 단위: macro:axis-pipeline
- 위치: `trainbench/kernels.py:243`

**주장**: `kernels.assert_packing_is_isolated` / `packing_isolation_holds` — 계약 파일이 'lane-e 의 거부 입력' 이라고 못박은 그 게이트 — 는 프로덕션 호출자가 0건이라, 지문을 읽는 유일한 자리인 `loader.describe` 가 packing 안전성을 보지 않는다.

**실패 시나리오**: `attn=fa3 dataloader=torch_packed purpose=timing`, flash-attn 없는 이미지에서 transformers 가 요청을 Hub repo 로 바꿔 바인딩한 모델. 실측(이 호스트, 재현 스크립트 첨부): `loader.describe(ADAPTERS['native'], model, ..., cfg)` 가 `resolved.mask_registered=False`, `packing_isolation_holds=False` 인 지문을 만들고 **거부 없이 AdapterOut 을 돌려준다**. 지문을 읽는 유일한 프로덕션 함수 `_refuse_a_build_the_fingerprint_condemns`(loader.py:295-319)는 `parameter_dtypes` 와 `trainable_parameter_names` 둘만 보고 `attention` 블록을 보지 않는다 — 그 함수 docstring 은 자기를 '지문이 막으려는 두 상태를, 그것을 읽는 자리에서' 라고 적지만 kernels 모듈이 만든 상태는 셋이다. 그래서 HAZARDS §6 의 '미등록 커널 + packing = 시퀀스 격리가 조용히 사라진다' 를 잡는 코드는 존재하고 테스트도 5개 있으나 어느 실행 경로에서도 도달하지 않는다. 완화: transformers 5.14.1 의 마스크 레지스트리는 정확히 {eager, fa2, fa3, fa4, flex, sdpa} 이므로 `mask_registered=False` 이면 resolved impl 이 그 여섯 중 하나가 아니고, 따라서 `_capture_attn` 이 `config.attn.impl` 과 불일치를 내 timing 런이 어차피 거부된다 — 오늘은 잘못된 숫자가 아니라 죽은 코드다. 그러나 그 완화는 설계가 아니라 문자열 동일성의 부수효과이고, unsloth(5.5.0)/ms_swift(5.12.1) 스택의 레지스트리 내용은 kernels 노트가 파드 질문 3 으로 남긴 확인 안 함이다.

**재현**:
```text
grep -rn 'assert_packing_is_isolated\|packing_isolation_holds' --include='*.py' . | grep -v '\.venv' | grep -v '^tests/'   # -> trainbench/kernels.py 의 정의 두 줄뿐
infisical run --env=dev -- uv run python -c "
import sys; sys.path.insert(0,'.')
import torch
from hydra import compose, initialize_config_dir
from trainbench import loader, kernels
from trainbench.compose import resolve
from transformers import Qwen3VLConfig
HUB='kernels-community/vllm-flash-attn3'
class T(torch.nn.Module):
    def __init__(s): super().__init__(); s.proj=torch.nn.Linear(2,2).to(torch.bfloat16)
class B(torch.nn.Module):
    def __init__(s):
        super().__init__(); c=Qwen3VLConfig(); c._attn_implementation='sdpa'; c._attn_implementation=HUB
        s.config=c; s.visual=T(); s.language_model=T()
with initialize_config_dir(config_dir='configs', version_base=None):
    cfg=resolve(compose(config_name='config', overrides=['run=timing','device=cpu','model=qwen3_vl_emb_2b','framework=native','attn=fa3','dataloader=torch_packed','data.limit=4','train.batch_size=4']))[0]
out=loader.describe(loader.ADAPTERS['native'], B(), object(), cfg, revision_resolver=lambda r:'abc123')
fp=out.fingerprint[loader.BUILD_FINGERPRINT_KEY]
print('packing=', cfg.dataloader.packing, 'mask_registered=', fp['resolved']['mask_registered'], 'isolation=', kernels.packing_isolation_holds(fp), '-> describe() 는 거부하지 않았다')"
# 출력: packing= True mask_registered= False isolation= False -> describe() 는 거부하지 않았다
```

**검증** (reproduced):
```text
grep -rn 'assert_packing_is_isolated\|packing_isolation_holds' --include='*.py' . | grep -v '\.venv'  # + infisical run --env=dev -- uv run python -c "<첨부 재현 스크립트>"
---
$ grep -rn 'assert_packing_is_isolated\|packing_isolation_holds' --include='*.py' . | grep -v '\.venv'
trainbench/kernels.py:14:what `assert_packing_is_isolated` refuses on.
trainbench/kernels.py:233:def packing_isolation_holds(fingerprint: Mapping[str, Any]) -> bool:
trainbench/kernels.py:243:def assert_packing_is_isolated(fingerprint: Mapping[str, Any]) -> None:
tests/test_kernels.py:266,276,291,306,443,498,584,585
tests/contract/test_kernel_provenance.py:220,251,308,408,434

$ infisical run --env=dev -- uv run python -c "...(재현 스크립트)..."
2026-08-03T03:36:16+09:00 INF Injecting 27 Infisical 
```

### `packing-isolation-guard-has-no-caller` — major / measurement-validity
- 단위: macro:measurement
- 위치: `trainbench/kernels.py:243`

**주장**: `assert_packing_is_isolated` 도 호출자가 없어서, 마스크 레지스트리에 없는 구현 + `dataloader.packing=true` 조합이 거부되지 않고 그대로 측정된다 — `docs/methodology.md:618` 은 거부된다고 적는다.

**실패 시나리오**: `dataloader=torch_packed attn=fa2` 로 파드에서 돌리면 resolved 구현이 `AttentionMaskInterface._global_mapping` 에 없을 때 `create_causal_mask` 가 None 을 돌려주고 팩 안의 시퀀스들이 서로의 컨텍스트가 된다. 예외도 경고도 없고, 레코드는 `dataloader.packing=True` 와 정상 throughput 숫자를 함께 싣는다. 지문은 이미 `trainbench/loader.py:258` 이 만들어 손에 들고 있고 `config.dataloader.packing` 도 같은 자리에 있는데 `loader.py` 는 `packing` 이라는 단어를 한 번도 읽지 않는다 (`grep -n packing trainbench/loader.py` 출력 0줄). `_refuse_a_build_the_fingerprint_condemns` 가 거부하는 것은 동결 그래프와 혼합 dtype 둘뿐이다.

**재현**:
```text
grep -rn "assert_packing_is_isolated\|packing_isolation_holds" trainbench scripts docker | grep -v "^trainbench/kernels.py"   # 출력 없음
grep -n "packing\|mask_registered" trainbench/loader.py                                                          # 출력 없음
# 변이: trainbench/kernels.py 의 assert_packing_is_isolated 본문 첫 줄을 `raise AssertionError("dead")` 로 바꾸고
uv run pytest -q --ignore=tests/test_kernels.py    # 여전히 전부 초록 = 프로덕션 경로가 이 함수를 거치지 않는다
```

**검증** (mutation-killed-nothing):
```text
grep -rn "assert_packing_is_isolated\|packing_isolation_holds" trainbench scripts docker  # 정의 3줄(kernels.py 14/233/243)만, 호출자 0
grep -n "packing\|mask_registered" trainbench/loader.py  # 출력 0줄 (exit 1)
grep -rn "assert_packing_is_isolated\|UnsafePacking" tests/  # tests/test_kernels.py 만 호출
# 변이: trainbench/kernels.py:244 에 raise AssertionError("dead") 삽입
uv run python -c "import trainbench.kernels as m; f=m.assert_packing_is_isolated; print(f.__code__.co_filename, f.__code__.co_firstlineno)"
infisical run --env=dev -- uv run pytest -q --ignore=tests/test_kernels.py
git checkout -- trainbench/kernels.py
---
$ grep -rn "assert_packing_is_isolated\|packing_isolation_holds" trainbench scripts docker
trainbench/kernels.py:14:what `assert_packing_is_isolated` refuses on.
trainbench/kernels.py:233:def packing_isolation_holds(fingerprint: Mapping[str, Any]) -> bool:
trainbench/kernels.py:243:def assert_packing_is_isolated(fingerprint: Mapping[str, Any]) -> None:

$ grep -n "packing\|mask_registered" trainbench/loader.py
(no output, exit 1)

$ uv run python -c "import trainbench.kernels as m; ..."
/Users/jwcho/Codes/train-comparison/trainbench/kernels.py 243

$ infisical run --env=dev -- uv run pytest -q
```

### `top-level-dict-branch-false-claim` — major / correctness
- 단위: kernels
- 위치: `trainbench/kernels.py:366`

**주장**: `_targeted` 의 '"" 키는 부모에게 물어 모든 서브컨피그로 전파된다'는 주석은 transformers 5.14.1 에 대해 거짓이고, 그 주석이 정당화하는 분기는 항상 무관한 메시지로 거부된다.

**실패 시나리오**: `read_fingerprint(model, axis='attn.name', value='eager', requested={'': 'eager'})` 는 `_targeted` 가 모든 백본을 반환한 뒤 `_one([])` 에 걸려 `UnidentifiedKernel: the backbones the request reached do not agree on what was asked: []. One axis value bound more than one kernel...` 을 낸다 — 커널을 하나도 안 물었는데 '두 개를 물었다'고 말한다. 실제 setter 는 `value.get(subconfig_key, current_subconfig_attn)`(configuration_utils.py:415)이라 "" 키는 서브컨피그에 전파되지 않으므로 주석의 사실 주장 자체가 틀렸다. 같은 경로가 dict 키가 백본 이름과 하나도 안 겹칠 때도 열린다: 서브컨피그가 없는 Qwen3.5(백본 키 = model_type)에 `requested={'text_config': impl}` 을 넘기면 동일한 오도 메시지로 죽는다 — adapters 레인이 멀티모달용 dict 를 모든 모델에 그대로 넘기면 바로 밟는다.

**재현**:
```text
uv run python -c "from trainbench import kernels; from transformers import Qwen3VLConfig\nclass M:\n def __init__(s,c): s.config=c\nc=Qwen3VLConfig(); c._attn_implementation='sdpa'; c._attn_implementation={'':'eager'}\nprint(c.text_config._attn_implementation)  # 'sdpa' — 전파 안 됨\nkernels.read_fingerprint(M(c), axis='attn.name', value='eager', requested={'':'eager'})"  → UnidentifiedKernel 'do not agree on what was asked: []'
```

**검증** (reproduced):
```text
uv run python /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/repro.py   (Qwen3VLConfig, requested={"": "eager"})  및  .../repro2.py  (Qwen3Config, requested={"text_config": "eager"})
---
repro.py (transformers 5.14.1):
/Users/jwcho/Codes/train-comparison/trainbench/kernels.py 361   # _targeted 정의 위치
/Users/jwcho/Codes/train-comparison/trainbench/kernels.py 383   # read_fingerprint 정의 위치
parent: eager
text_config: sdpa
vision_config: sdpa
backbones: {'vision_config': 'sdpa', 'text_config': 'sdpa'}
targeted: ['text_config', 'vision_config']
RAISED UnidentifiedKernel: the backbones the request reached do not agree on what was asked: []. One axis value bound more than one kernel and the frozen payload records a single identity, so this build cannot be described rather than being d
```

### `cached-switch-name-vacuum` — major / emptiness
- 단위: kernels
- 위치: `trainbench/kernels.py:470`

**주장**: `getattr(module, attribute, want)` 가 속성이 없는 모듈을 '닫힘'으로 읽고, `CACHED_FETCH_SWITCHES` 의 이름이 실제 설치본에 존재한다고 주장하는 테스트가 하나도 없다 — 라이브러리가 전역을 개명하면 커널 fetch 검사가 조용히 공허해진다.

**실패 시나리오**: transformers 가 `_TRANSFORMERS_USE_HUB_KERNELS` 를 다른 이름으로 옮긴 파드(이 저장소는 envs 에서 5.5.0/5.12.1/5.14.1 을 함께 쓴다)에서, 살아 있는 스위치가 'YES'(= Hub 커널 허용)인데도 `assert_no_runtime_kernel_fetch` 가 예외 없이 통과한다. 측정 중 커널이 네트워크로 도착할 수 있는 상태를 '문 0개'로 보고하는 것이고, 이 레인의 게이트는 합성 `types.ModuleType` 스텁만 검사하므로 그 개명을 한 번도 보지 못한다.

**재현**:
```text
uv run python -c "import sys, transformers.integrations.hub_kernels as hk, huggingface_hub.constants as c; from trainbench import kernels; hk._kernels_enabled=False; hk.RENAMED=hk._TRANSFORMERS_USE_HUB_KERNELS; del hk._TRANSFORMERS_USE_HUB_KERNELS; c.HF_HUB_OFFLINE=True; kernels.assert_no_runtime_kernel_fetch(dict(kernels.RUNTIME_FETCH_ENV), sys.modules); print('PASSED while the live switch is', hk.RENAMED)"  → 'PASSED while the live switch is YES' 출력(실행 확인)
```

**검증** (reproduced):
```text
uv run python -c "import sys, transformers.integrations.hub_kernels as hk, huggingface_hub.constants as c; from trainbench import kernels; print('live switch before:', getattr(hk,'_TRANSFORMERS_USE_HUB_KERNELS','<absent>')); print('defn:', kernels.assert_no_runtime_kernel_fetch.__code__.co_filename, kernels.assert_no_runtime_kernel_fetch.__code__.co_firstlineno); hk._kernels_enabled=False; hk.RENAMED=hk._TRANSFORMERS_USE_HUB_KERNELS; del hk._TRANSFORMERS_USE_HUB_KERNELS; c.HF_HUB_OFFLINE=True; kernels.assert_no_runtime_kernel_fetch(dict(kernels.RUNTIME_FETCH_ENV), sys.modules); print('PASSED while the live switch is', hk.RENAMED)"
---
live switch before: YES
defn: /Users/jwcho/Codes/train-comparison/trainbench/kernels.py 500
PASSED while the live switch is YES
```

### `fetch-door-absent-attribute-reads-closed` — major / emptiness
- 단위: macro:emptiness
- 위치: `trainbench/kernels.py:470`

**주장**: `open_fetch_doors` 가 `getattr(module, attribute, want)` 로 읽어 **속성이 없는 모듈을 닫힌 문으로 읽고**, `forbid_runtime_kernel_fetch` 는 그 이름을 `setattr` 로 새로 만들어 붙인 뒤 `assert_no_runtime_kernel_fetch` 가 그 조작된 속성을 근거로 통과한다. 네 개의 `(module, attribute)` 쌍이 실제 설치본에 존재하는지 확인하는 테스트가 하나도 없다 — 모든 테스트가 `types.ModuleType` 스텁에 이름을 직접 심어 놓고 그것을 읽는다.

**실패 시나리오**: 이 표는 transformers 5.14.1 원문에서 읽은 것인데(`docs/methodology.md:623-624`), 파드 이미지 여섯 중 unsloth 는 5.5.0, ms_swift 는 5.12.1 이다(`HAZARDS.md §6`). 그 버전에서 스위치 이름이 `_kernels_enabled` 가 아니면 `open_fetch_doors` 는 `[]` 를 돌려주고 `assert_no_runtime_kernel_fetch` 는 통과한다. 그 뒤 `forbid_runtime_kernel_fetch` 가 transformers 가 읽지 않는 새 속성 `_kernels_enabled=False` 를 모듈에 만들어 붙이고, 살아 있는 진짜 스위치는 켜진 채로 남는다 — 측정 중 커널이 네트워크로 도착할 수 있는 상태를 "닫혔다" 로 인증한다. 커널 레인 노트의 '확인 안 함' 6항목에 5.5.0/5.12.1 의 `find_packed_sequence_indices` 는 있지만 이 네 이름의 버전 의존성은 없다.

**재현**:
```text
uv run python - <<'PY'
import types
from trainbench import kernels
old=types.ModuleType('transformers.integrations.hub_kernels')
old.USE_HUB_KERNELS='YES'; old.kernels_enabled=True   # 살아 있는 스위치, 다른 이름, 켜짐
mods={'transformers.integrations.hub_kernels':old}; env=dict(kernels.RUNTIME_FETCH_ENV)
print('open doors ->',kernels.open_fetch_doors(env,mods))
kernels.assert_no_runtime_kernel_fetch(env,mods); print('assert PASSED')
print('forbid was_open ->',kernels.forbid_runtime_kernel_fetch(env,mods))
print('live switch after forbid:',old.USE_HUB_KERNELS,old.kernels_enabled)
print('fabricated attrs:',old._TRANSFORMERS_USE_HUB_KERNELS,old._kernels_enabled)
PY
# -> open doors -> []  / assert PASSED / live switch after forbid: YES True / fabricated attrs: NO False
```

**검증** (reproduced):
```text
uv run python - <<'PY'
import types
from trainbench import kernels
old=types.ModuleType('transformers.integrations.hub_kernels')
old.USE_HUB_KERNELS='YES'; old.kernels_enabled=True
mods={'transformers.integrations.hub_kernels':old}; env=dict(kernels.RUNTIME_FETCH_ENV)
print('open doors ->',kernels.open_fetch_doors(env,mods))
kernels.assert_no_runtime_kernel_fetch(env,mods); print('assert PASSED')
print('forbid was_open ->',kernels.forbid_runtime_kernel_fetch(env,mods))
print('live switch after forbid:',old.USE_HUB_KERNELS,old.kernels_enabled)
print('fabricated attrs:',old._TRANSFORMERS_USE_HUB_KERNELS,old._kernels_enabled)
print('co:',kernels.open_fetch_doors.__code__.co_filename,kernels.open_fetch_doors.__code__.co_firstlineno)
PY
---
open doors -> []
assert PASSED
forbid was_open -> []
live switch after forbid: YES True
fabricated attrs: NO False
co: /Users/jwcho/Codes/train-comparison/trainbench/kernels.py 448
```

### `build-fingerprint-has-no-consumer` — major / dead-code
- 단위: loader-probe
- 위치: `trainbench/loader.py:239`

**주장**: The build fingerprint (the lane's axis-G deliverable) is computed on every load and then discarded — no result file carries it, and `fingerprint_diff`/`tensor_count` have no production caller — while `tests/contract/test_kernel_provenance.py` states the run record carries it under `build_fingerprint`.

**실패 시나리오**: A completed timing run writes `build_record(config, device, applied=state, metrics=summary, applied_axes=applied)` (bench.py:850). `build_record` (trainbench/record.py:157-192) takes `**extra`, and `binding.fingerprint` is never passed: `grep -n fingerprint scripts/bench.py` returns exactly one hit, the `Binding` field declaration at :485; `grep -rn fingerprint scripts/report.py trainbench/record.py trainbench/applied.py` returns nothing. So the confound the fingerprint exists to expose — unsloth freezing every parameter, axolotl leaving two modules in fp32, unsloth x gemma-4 carrying 60 extra tensors — is invisible in every published result, and `loader.fingerprint_diff` (loader.py:269), whose docstring promises the six-way diff, is only ever called from tests/test_loader.py. Meanwhile `test_kernel_provenance.py:521` asserts the round-trip on a record dict the test builds itself, and `test_the_two_fixtures_that_carry_this_payload_agree_with_it` reads `run_record.sample.json` — a fixture — so no check notices that the key has no producer. This gap is not in `.plans/notes/adapters.md` §3, which lists every other cross-lane wiring the lane needed.

**재현**:
```text
grep -n fingerprint scripts/bench.py   # one hit: line 485, the Binding field
grep -rn --include='*.py' -e fingerprint_diff -e tensor_count -e RUN_RECORD_KEY scripts/ trainbench/   # no production caller; RUN_RECORD_KEY only defined, never used
grep -n 'def build_record' -A 40 trainbench/record.py   # no build_fingerprint key
Mutation: replace `loader.build_fingerprint`'s body with `return {}` and run `uv run pytest -q` — only tests/test_loader.py and tests/contract/test_loader_bench.py react; no record-writing or report test does.
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python scratchpad/probe.py   # build_record output + AST of every build_record call in scripts/bench.py
# then: tensor_count -> `return 0`, fingerprint_diff -> `return {}`, and `build_fingerprint` popped from tests/fixtures/run_record.sample.json
infisical run --env=dev -- uv run pytest -q tests/contract/test_record_report.py tests/contract/test_kernel_provenance.py tests/test_report.py tests/test_metrics.py tests/test_kernels.py tests/test_loader.py
---
1) 실제 레코드 생산자 실행 (scripts/bench.py:851과 동일한 인자 형태):

record top-level keys:
[
 "applied", "applied_axes", "config", "device", "git_commit", "git_dirty",
 "git_source", "host", "image", "image_digest", "metrics", "packages", "recorded_at"
]
build_fingerprint in record: False
bench.py:450 build_record kwargs=['applied', 'status', 'refusal']
bench.py:851 build_record kwargs=['applied', 'metrics', 'applied_axes']
bench.py:840 build_record kwargs=['applied', 'applied_axes', None]
fixture run_record.sample.json has build_fingerprint: True

-> bench.py의 build_record 호출 3곳(정상 완료 851, OOM 840, axis-ref
```

### `adapter-refusals-escape-the-refusing-block` — major / contract-split
- 단위: loader-probe
- 위치: `trainbench/loader.py:58`

**주장**: `AdapterRefusal` and `kernels.KernelProvenanceError` are both raised inside `bench.py`'s `with refusing("load_kwargs")` region, but `refusing` catches only `UnappliedAxis` and `AppliedMismatch`, so every refusal the loader lane added produces no result file instead of a refusal record.

**실패 시나리오**: `scripts/bench.py:526-532` wraps `load_framework` in `refusing('load_kwargs')`, which at :418 catches `(axes.UnappliedAxis, AppliedMismatch)` only; its own docstring says everything else "passes through untouched and leaves no result file". `loader.AdapterRefusal` (loader.py:58) and `kernels.KernelProvenanceError` (kernels.py:83) both subclass `RuntimeError`, neither is caught. Two reachable triggers: (a) the unsloth+LoRA frozen build above; (b) attn.name=fa2 on an image without the flash-attn package — transformers rewrites the request to a Hub repo id, and `kernels._identify` (kernels.py:315-321) raises `UnidentifiedKernel` because `loader.build_fingerprint`'s `revision_resolver` parameter is never supplied by any production caller (grep shows `revision_resolver` only in loader.py, kernels.py and tests/test_kernels.py). In both cases the pod writes no `--out` file, `main` exits through the broad `except`, and the setting is filed by `docker/entrypoint.sh` as a crash rather than as a declined setting — the exact difference `.plans/notes/adapters.md` §3.3 argued mattered for `UnappliedAxis`, applied to nothing else.

**재현**:
```text
Run the finding-1 repro; it raises AdapterRefusal straight out of `loader.load`. Then confirm the catch list: `sed -n '406,420p' scripts/bench.py` (catches `(axes.UnappliedAxis, AppliedMismatch)`), and `uv run python -c "from trainbench.loader import AdapterRefusal; from trainbench.kernels import KernelProvenanceError; from trainbench.axes import UnappliedAxis; from trainbench.applied import AppliedMismatch; print(issubclass(AdapterRefusal,(UnappliedAxis,AppliedMismatch)), issubclass(KernelProvenanceError,(UnappliedAxis,AppliedMismatch)))"` -> `False False`.
```

**검증** (reproduced):
```text
tests/test_zz_verify_repro.py (임시, 검증 후 삭제): pod_setting 픽스처로 bench_entry.main()을 실제 실행하며 trainbench.loader.load 만 각각 loader.AdapterRefusal / kernels.UnidentifiedKernel 을 던지게 바꾼 뒤 `infisical run --env=dev -- uv run pytest tests/test_zz_verify_repro.py -q`. 라이브 정의 확인: `uv run python -c "...; g=m.refusing.__wrapped__; print(g.__code__.co_filename, g.__code__.co_firstlineno)"` 및 서브클래스 확인 one-liner.
---
$ infisical run --env=dev -- uv run pytest tests/test_zz_verify_repro.py -q
..                                                                       [100%]
2 passed in 1.64s

(두 테스트의 단언: `pytest.raises(loader.AdapterRefusal)` / `pytest.raises(kernels.KernelProvenanceError)` 가 main() 밖으로 그대로 올라오고 `assert not pod_setting.out.exists()` — 즉 --out 결과 파일이 전혀 쓰이지 않음)

$ uv run python -c "... g=m.refusing.__wrapped__ ..."
/Users/jwcho/Codes/train-comparison/scripts/bench.py 405

$ uv run python -c "print(issubclass(AdapterRefusal,(UnappliedAxis,AppliedMismatch)), issubclass(KernelProvenanceError,(Unap
```

### `throughput-denominator-changes-no-published-figure` — major / emptiness
- 단위: macro:measurement
- 위치: `trainbench/metrics/__init__.py:345`

**주장**: `measurement.throughput_denominator` 는 어떤 값을 골라도 결과 문서의 처리량 수치를 바꾸지 않는다 — `summarise` 는 이름 문자열만 싣고, `report.py` 는 `tokens_per_second` 를 무조건 출력한다.

**실패 시나리오**: `dataloader.packing` 축을 재면서 `+measurement.throughput_denominator=padded_tokens` 를 선언하면(패딩이 배치의 89% 까지 가고 이 선택이 packing 축 순위를 뒤집는다는 것이 이 knob 의 존재 이유다) 레코드는 `throughput_denominator: "padded_tokens"` 를 싣지만 `_figure_table` 의 `tokens/s` 열은 여전히 비패딩 토큰 기준이고, `padded_tokens_per_second` 는 `_figure_table` 에도 `_counts_table` 에도 렌더되지 않는다(`_counts_table` 이 내는 것은 `padded_tokens_per_step`, 즉 rate 가 아니다). 결과: 선언한 분모와 순위표가 쓰는 분모가 다르고, 리포트만 읽는 독자에게는 그 사실이 보이지 않는다.

**재현**:
```text
grep -rn "throughput_denominator" scripts/ docs/CONTRACTS.md tests/contract/ tests/fixtures/   # 출력 없음 (소비자·계약 모두 0)
grep -n "tokens_per_second\|padded_tokens_per_second" scripts/report.py   # tokens_per_second 만 나온다
# 변이: trainbench/config_schema.py:280 의 기본값을 "padded_tokens" 로 바꾸고
infisical run --env=dev -- uv run pytest -q    # 전부 초록, 리포트 출력 한 글자도 안 바뀐다
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python scratchpad/repro_denom.py  # summarise(config=+measurement.throughput_denominator={tokens,padded_tokens}) -> scripts/report.py _figure_table / _counts_table 렌더 비교
---
summarise def: /Users/jwcho/Codes/train-comparison/trainbench/metrics/__init__.py 246
_figure_table def: /Users/jwcho/Codes/train-comparison/scripts/report.py 675
_counts_table def: /Users/jwcho/Codes/train-comparison/scripts/report.py 701
declared: tokens / padded_tokens
tokens_per_second: 2000.0 / 2000.0
padded_tokens_per_second: 8000.0 / 8000.0
--- figure table (tokens denominator) ---

| 런 | 파드 | 프레임워크 x 모델 | 목적 | step p50 (s) | p95 (s) | mean (s) | samples/s | tokens/s | peak mem (GiB) | steps 계측/폐기/측정 | 파드 판정 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (단일 런) | unknown | native
```

### `gradnorm-float64-materializes-two-full-copies` — major / measurement-validity
- 단위: metrics-schema
- 위치: `trainbench/metrics/validity.py:106`

**주장**: `gradient_norm` 이 파라미터마다 full-size float64 사본을 두 개 만들며, 그 할당이 `peak_memory_bytes` 를 읽기 전에·OOM 으로 파일링되는 블록 안에서 일어난다.

**실패 시나리오**: `total + grad.to(torch.float64).pow(2).sum()` 는 grad 원소당 8바이트 텐서를 두 번 실체화한다. Qwen3.5 임베딩 grad 하나(151936x2048 bf16, 622MB)에 대해 이 호스트에서 max-RSS 델타 4,980,031,488 바이트를 실측했다(같은 값을 내는 `torch.linalg.vector_norm(..., dtype=torch.float64)` 는 2,490,449,920). 호출 지점은 `scripts/bench.py:333` 로 (a) `scripts/bench.py:341` 의 `metrics.peak_memory_bytes(device)` 보다 **먼저** 돌고 `reset_peak_memory` 이후이므로 이 임시 할당이 `torch.cuda.max_memory_allocated` 에 들어간다 — 활성화 피크가 이 5GB 여유보다 낮은 설정(작은 배치·짧은 seqlen·gradient_checkpointing=full)에서는 보고되는 peak memory 가 학습 스텝의 피크가 아니라 norm 계산의 피크가 된다. (b) `scripts/bench.py:833` 의 `except BaseException ... if metrics.is_oom(exc)` 안이므로, 측정 창을 정상 완료한 런이 norm 계산에서 OOM 나면 `status: oom` 으로 기록된다 — `oom_status` 자신의 docstring 이 금지하는 "우리 코드의 결함을 하드웨어 한계로 발행" 이 바로 그 경로다.

**재현**:
```text
for mode in current fused; do uv run python - "$mode" <<'PY'
import sys, resource, torch
mode = sys.argv[1]
g = torch.randn(151936, 2048, dtype=torch.bfloat16)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
v = float(g.detach().to(torch.float64).pow(2).sum().cpu()) if mode=="current" else float(torch.linalg.vector_norm(g.detach(), 2, dtype=torch.float64)**2)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(mode, v, peak-base)
PY
done
# 그리고 호출 순서: sed -n '330,345p' scripts/bench.py (gradient_norm 이 peak_memory_bytes 보다 앞)
```

**검증** (reproduced):
```text
for mode in current fused; do uv run python - "$mode" <<'PY'
import sys, resource, torch
mode = sys.argv[1]
g = torch.randn(151936, 2048, dtype=torch.bfloat16)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
v = float(g.detach().to(torch.float64).pow(2).sum().cpu()) if mode=="current" else float(torch.linalg.vector_norm(g.detach(), 2, dtype=torch.float64)**2)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(mode, v, peak-base)
PY
done
# plus: grep -n "gradient_norm\|peak_memory_bytes\|reset_peak" scripts/bench.py
---
current 311156748.301987 4980015104
fused 311146505.004231 2490433536

grep -n:
scripts/bench.py:259:            metrics.reset_peak_memory(device)
scripts/bench.py:333:    validity["grad_norm"] = metrics.gradient_norm(built.model)
scripts/bench.py:347:        peak_bytes=metrics.peak_memory_bytes(device),
scripts/bench.py:845:            **metrics.oom_status(exc, peak_bytes=metrics.peak_memory_bytes(device)),
```

### `training-gate-total-params-absent` — major / emptiness
- 단위: macro:emptiness
- 위치: `trainbench/metrics/validity.py:187`

**주장**: `training_verdict` 의 peft-mode 정합성 검사는 `total_params` 가 없거나 int 가 아니면 이유 한 줄 없이 통째로 건너뛰고 (True, []) 를 돌려주는데, 같은 파일 `GATE_FIELDS` 는 `total_params` 를 포함하며 그 주석은 "Absence is a refusal rather than a pass" 라고 적는다. 세 벌(`validity.py:187`, `scripts/report.py:411`, `tests/contract/test_record_report.py:319`) 이 전부 같은 구멍을 갖는다.

**실패 시나리오**: `peft.mode=full` 런이 파라미터 텐서 12개만 학습했는데(HAZARDS 가 인용한 unsloth 동결 계열의 부분 동결) 레코드의 `metrics` 에 `total_params` 가 없으면 세 게이트가 모두 `(True, [])` 를 낸다. 이번 캠페인이 바로 이 경우를 잡으려고 추가한 "peft.mode=full but 12 of N parameter tensors train; a full finetune that froze part of the model is a different workload" 문장이 비교 대상이 없어 한 번도 평가되지 않고, 그 런의 step time 이 full finetune 속도로 발행된다. `peft.mode=lora` 쪽 "the adapter did not narrow anything" 도 같은 조건에서 침묵한다. 계약의 mutation 표(`tests/contract/test_record_report.py:477-511`, `tests/test_metrics.py:407-437`) 는 필드 값을 나쁘게 바꾸는 케이스만 있고 `total_params` 를 **삭제**하는 케이스가 하나도 없어서 이 공허가 테스트되지 않는다.

**재현**:
```text
uv run python -c "from trainbench.metrics.validity import training_verdict,GATE_FIELDS;m={'grad_norm':1.0,'trainable_params':12,'loss_first':2.0,'loss_last':1.0,'peak_memory_bytes':1024};print(GATE_FIELDS);print('full :',training_verdict(m,peft_mode='full',device='cuda:0'));print('lora :',training_verdict(m,peft_mode='lora',device='cuda:0'))"
# -> full : (True, [])   lora : (True, [])
# 대조: {**m,'total_params':100} 을 넣으면 두 경우 다 (False, [...]) 가 된다
```

**검증** (reproduced):
```text
uv run python -c "from trainbench.metrics.validity import training_verdict,GATE_FIELDS;m={'grad_norm':1.0,'trainable_params':12,'loss_first':2.0,'loss_last':1.0,'peak_memory_bytes':1024};print(GATE_FIELDS);print('full :',training_verdict(m,peft_mode='full',device='cuda:0'));print('lora :',training_verdict({**m,'trainable_params':100},peft_mode='lora',device='cuda:0'));print('full+total:',training_verdict({**m,'total_params':100},peft_mode='full',device='cuda:0'));print('lora+total:',training_verdict({**m,'trainable_params':100,'total_params':100},peft_mode='lora',device='cuda:0'))"
---
/Users/jwcho/Codes/train-comparison/trainbench/metrics/validity.py 156   (training_verdict.__code__)
('grad_norm', 'trainable_params', 'total_params', 'loss_first', 'loss_last', 'peak_memory_bytes')
full : (True, [])
lora : (True, [])
full+total: (False, ['peft.mode=full but 12 of 100 parameter tensors train; a full finetune that froze part of the model is a different workload'])
lora+total: (False, ['peft.mode=lora but 100 of 100 parameter tensors train; the adapter did not narrow anything'])
```


## 반박된 발견 — 지우지 않고 남긴다
- `[refuted]` **pytest-gate-is-not-reproducible-at-head** (major, macro:measurement) — 동일 커밋·동일 워크트리에서 `pytest` 를 반복 실행하면 실패 집합이 매번 달라진다 — 같은 명령이 `1147 passed` 도 내고 1·2·3·5·8건 실패도 낸다. 캠페인 완료 조건이 정확한 통과 수를 인용하는데 그 수가 재현되지 않는다.
  - 반박 근거: === run 1 ===
1147 passed, 14 warnings in 99.27s (0:01:39)
=== run 2 ===
1147 passed, 14 warnings in 96.50s (0:01:36)
=== run 3 ===
1147 passed, 14 warnings in 96.70s (0:01:36)
=== run 4 ===
1147 passed, 14 warnings in 97.45s (0:01:37)
=== run 5 ===
1147 passed, 14 warnings in 96.13s (0:01:36)
=== r

## minor — 수정 패스에 넣지 않았다
- `engine-optimizer-class-unobserved` (capture) `trainbench/applied.py:308` — `_engine_optimizer_class` 의 반환값(`detail["engine_optimizer"]`)을 확인하는 테스트도 픽스처도 없어, 주석이 약속한 "결과가 여전히 어느 파티셔너가 돌았는지 말한다"가 검증되지 않는다.
- `stale-fsdp2-claims-in-axes` (capture) `trainbench/axes.py:1287` — `_parallel` docstring 과 `.plans/notes/axes.md` 가 아직 "capture 가 FSDPModule 을 읽기 전까지 fsdp2 런은 거부된다 / `parallel=fsdp2` 는 측정이 열리지 않는다"고 적고 있으나, `applied._is_fsdp2` 가 지금 그것을 읽는다.
- `duplicate-literal-values-helper` (capture) `tests/test_applied.py:978` — `literal_values` 가 두 파일에 서로 다른 시그니처로 정의돼 있고, `tests/test_applied.py` 사본에는 bool 축용 `or {"True","False"}` 폴백이 없다.
- `stale-ddp-refusal-claims-in-owned-files` (axes) `configs/parallel/single_cross_device.yaml:4` — Two files this lane owns still state that `assemble` refuses every strategy but `single` and that DDP is not implemented — both made false by this same commit.
- `ddp-test-docstring-denies-its-own-stub` (axes) `tests/test_axes.py:2066` — `test_ddp_wraps_the_model_and_reads_back_as_ddp`'s docstring says the wrapper is torch's own and that "a stand-in named `DistributedDataParallel` would satisfy the capture while proving nothing", and the next four lines monkeypatch `torch.nn.parallel.DistributedDataParallel` into exactly that stand-in.
- `axes-notes-false-against-the-tree` (axes) `.plans/notes/axes.md:11` — Two statements in this lane's handoff note are false against the tree it is being merged into: `KERNEL_PATCHERS["kernels_hub"]` was not left in, and `parallel=fsdp2` is no longer unreadable by the capture side.
- `model-rejects-boundary-keys-is-false` (collate-prompt) `trainbench/collate.py:36` — 새로 추가된 주석이 `axes.PACKED_BOUNDARY_KEYS`(`cu_seqlens`/`seq_lengths`)를 "the model does reject those" 라고 단언하지만 모델은 조용히 삼킨다 — 같은 레인의 `test_an_undeclared_kwarg_is_swallowed_just_as_quietly` 가 증명하는 규칙과 정반대다.
- `varlen-gate-citation-is-stale` (collate-prompt) `trainbench/collate.py:469` — `modeling_flash_attention_utils.py:761-763` 이라고 두 곳에 적혀 있으나 핀된 휠에서 `is_fa_with_varlen_kwargs = all(` 은 765행이다 — 레인 브리프(`.plans/remaining-code/packing.md:78`)가 이미 765-767 로 적어둔 것을 fixture 의 옛 숫자로 되돌려 옮겨 적었다.
- `arch-forward-test-omits-the-varlen-four` (collate-prompt) `tests/test_collate.py:360` — `test_every_arch_takes_the_packed_batch_this_collate_builds` 는 collate 가 만든 7키 배치 중 `input_ids`/`position_ids`/`seq_idx` 3키만 forward 에 넘긴다 — varlen 네 kwarg 은 저장소의 어떤 테스트에서도 실제 모델에 전달된 적이 없다.
- `fingerprint-claims-an-attn-request-never-made` (loader-probe) `trainbench/loader.py:263` — `build_fingerprint` always passes `requested=config.attn.impl` to `kernels.read_fingerprint`, but four of six adapters have `honours_load_kwargs=False` and never hand any `attn_implementation` to `from_pretrained`, so the frozen provenance payload records a request that did not happen.
- `st-backward-measured-in-eval-mode` (loader-probe) `trainbench/probe/sentence_transformers.py:86` — The ST probe's `_backward` never calls `model.train()`, and the `encode` check that runs just before it leaves the model in eval mode, so `mnrl_backward` answers "does a training step run" under a different regime from every other framework's `infonce_backward`.
- `duplicate-never-imported-test` (kernels) `tests/test_kernels.py:387` — `test_no_runtime_fetch_ignores_modules_that_were_never_imported` 의 단언은 `test_no_runtime_fetch_accepts_a_closed_environment` 첫 줄과 문자 그대로 같아서, docstring 이 주장하는 '`kernels` 없는 파드' 를 검사하지 않는다.
- `trimmed-mean-window-validator-branch-unreachable` (metrics-schema) `trainbench/config_schema.py:433` — `_the_aggregate_has_samples_to_aggregate` 의 `trimmed_mean` 분기는 어떤 스키마-유효 입력으로도 발화할 수 없다.
- `gate-fields-has-no-reader` (metrics-schema) `trainbench/metrics/validity.py:51` — `GATE_FIELDS` 는 정의와 재수출 말고 읽는 곳이 하나도 없는데, 주석은 그것이 게이트의 전제 조건이라고 주장한다.
- `throughput-denominator-never-reaches-a-published-figure` (metrics-schema) `trainbench/config_schema.py:280` — `measurement.throughput_denominator` 는 선언·기록되지만 발행되는 어떤 수치도 그 선언을 따르지 않는다.
- `drift-test-runs-one-of-the-two-copies` (metrics-schema) `tests/test_metrics.py:439` — 두 사본의 드리프트를 막는다고 주장하는 테스트가 사본 하나만 호출한다 — 한 방향만 보는 검사다.
- `binding-namedtuple-never-constructed` (bench) `scripts/bench.py:464` — `Binding` NamedTuple 은 프로덕션 경로에서 한 번도 생성되지 않는다 — `load_framework` 는 `AdapterOut` 을 그대로 돌려주면서 반환 타입만 `Binding` 이라고 적는다.
- `stale-prose-native-binding-and-oom` (bench) `scripts/bench.py:528` — 지워진 심볼과 바뀐 배선을 가리키는 주석·docstring 이 세 곳 남아, 다음 독자가 존재하지 않는 코드 경로를 근거로 판단하게 된다.
- `manifest-axis-mismatch-message-names-an-override-that-does-not-exist` (report-orchestrate) `scripts/orchestrate.py:243` — `pod_overrides` 가 `axes_touched` 에 합류하면서 `moved` 는 항상 `framework` 를 담게 됐고, 그 결과 `load_experiment` 의 축 라벨 검사가 오버라이드가 하나도 없는 매니페스트에 대해 "its overrides move framework" 라고 진단한다 — 존재하지 않는 오버라이드를 고치라고 지시한다.
- `measurement-summary-counts-overlap-and-do-not-partition-the-runs` (report-orchestrate) `scripts/report.py:785` — `측정 결과` 머리말이 네 범주를 런 수의 분할처럼 나열하지만 `학습하지 않은 런` 은 `수치를 낸 것` 의 부분집합이라, 합이 전체 런 수를 넘는 문장이 생성된다.
- `open-verdicts-anchor-changed-evidence-is-truncated-mid-command` (report-orchestrate) `docs/open-verdicts.json:531` — `images-carry-a-code-snapshot-nothing-checks-is-current` 의 `anchor_changed` 문자열이 명령 중간에서 잘려 있다 — 되돌린 근거로 인용된 실행 명령과 그 출력이 파일에 남아 있지 않다.
- `stale-comment-claims-a-baseline-entry-that-no-longer-exists` (audit) `scripts/audit_plan.py:104` — `CONFIG_OBJECT_NAMES` 위 주석이 `prepare_data.py` 의 `data = config.data` 별칭 오탐을 "a known false alarm, tracked in the baseline" 이라고 적지만, 그 별칭은 이미 제거됐고 `config-consumed` 는 이 diff 직전 커밋(01b8a83)에서 baseline 에서 빠졌다.
- `verdicts-closed-ratchet-hides-offsetting-changes` (audit) `docs/audit-baseline.json:8` — note 가 "넷이면 grew, 둘이면 shrank 로 BLOCK 되어 어느 쪽이든 갱신이 의도적 행위가 된다" 고 적지만, `count` 는 `len(open_ids) + len(problems)` 한 개의 합이라 상쇄되는 변화 한 쌍은 게이트를 통과한다 — note 가 막는다고 주장하는 바로 그 상황(새 판정이 여기 묻히는 것)이다.
- `native-env-keeps-a-dependency-whose-only-stated-axis-was-deleted` (audit) `envs/native/pyproject.toml:23` — 결정 6 이 `kernel/kernels_hub` 를 없애고 이 diff 가 `AXIS_PACKAGES` 의 짝 항목을 지웠는데, `envs/native` 는 그 축을 유일한 정당화로 적은 주석과 함께 `kernels>=0.10` 를 직접 의존성으로 계속 들고 있다.
- `applied-axes-second-name-and-unproducible-value` (macro:contracts) `tests/fixtures/run_record.sample.json:187` — 레코드가 '적용된 축'을 `applied`(AppliedState dict)와 `applied_axes`(list[str]) 두 이름으로 싣는다. 후자는 읽는 곳이 하나도 없고, 동결된 샘플의 값 `[]` 는 생산자가 만들 수 없는 값이다.
- `axis-state-sample-packing-detail-contradicts-capture` (macro:contracts) `tests/fixtures/axis_state.sample.json:35` — `applied-axes` 샘플의 `dataloader.packing` 항목이 `applied: "False"` 이면서 근거 detail 로 `{"collate": "PackedCollate"}` 를 든다 — 그 detail 을 낳는 capture 는 같은 입력에서 `"True"` 를 돌려준다.
- `adapter-sample-cites-a-file-that-does-not-exist` (macro:contracts) `tests/fixtures/adapter_out.sample.json:5` — `adapter_out.sample.json` 의 `values_are` 가 '이 경계가 기대는 실측은 전부 여기 인용돼 있다'며 `.plans/remaining-code/lane-g.md` 를 가리키는데 그 파일이 없다(레인이 역할 이름으로 개명됐고 해당 파일은 `adapters.md` 다).
- `loader-load-docstring-misstates-the-wiring` (macro:axis-pipeline) `trainbench/loader.py:543` — `loader.load` 의 docstring 이 '`scripts/bench.py` … reads the eight fields off the result' 라고 적지만 bench.py 는 다섯만 읽는다 (`step`, `fingerprint`, `documented_entry_point` 는 `Binding` 에 담기고 끝이다).
- `aggregate-statistic-reaches-no-published-figure` (macro:measurement) `trainbench/metrics/__init__.py:331` — `measurement.aggregate` / `trim_fraction` 가 만드는 `step_seconds_aggregate` 는 레코드에만 있고 리포트의 어떤 열도, 파드 판정도 그것을 읽지 않는다.
- `deterministic-allowed-for-quality-runs-the-report-ranks` (macro:measurement) `trainbench/config_schema.py:357` — `_timing_runs_are_uncontaminated` 는 `purpose=timing` 에만 걸리는데 `scripts/report.py` 는 `quality` 런의 스텝 시간을 `timing` 과 같은 순위표(`_ranked_by_stack(timed, ...)`)에 넣고, deterministic 여부는 레코드의 metrics 에도 리포트에도 표시되지 않는다.
