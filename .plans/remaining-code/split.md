# split — `bench.py` 분해 (wave 0, 단독)

> 먼저 읽는다: `.plans/remaining-code/HAZARDS.md`, `.plans/remaining-code/PLAN.md`.
> 이 레인이 머지되기 전에는 다른 어떤 레인도 돌지 않는다. 여기서 만드는 seam 위에
> wave 2 의 packing 과 adapters 가 통째로 얹힌다.

## 목표

`scripts/bench.py`(1231줄)에서 collate 블록을 `trainbench/collate.py` 로 **순수 이동**하고,
프레임워크 어댑터가 나중에 꽂힐 seam 을 남긴다. **동작은 바뀌지 않는다.**

## Owns

```
scripts/bench.py
trainbench/collate.py            ⊕ 신설
tests/test_smoke_cpu.py
trainbench/record.py
tests/contract/test_record_report.py    마커 1개 제거만
tests/contract/test_loader_bench.py     마커 1개 제거만
```

이 밖의 파일은 전부 남의 것이다. 건드렸으면 `outOfBounds` 에 적는다.

## 작업 1 — collate 블록 이동

`scripts/bench.py:85–564` 를 `trainbench/collate.py` 로 옮긴다:

| 줄 | 심볼 |
|---|---|
| 85 | `MMEB_IMAGE_MARKER` |
| 88–122 | `class MicroBatch(NamedTuple)` |
| 124–138 | `class PairTexts(NamedTuple)` |
| 140–166 | `class PairDataset(Dataset)` |
| 168–192 | `def load_pairs` |
| 194–215 | `def _group_by_row` |
| 217–384 | `class Collate` |
| 385–419 | `class Encode` |
| 420–450 | `class PackedPairs` |
| 451–497 | `class PackedBatches` |
| 498–538 | `class PretokenizedCollate` |
| 539–564 | `def build_collate` |

**모듈 이름은 협상 대상이 아니다.** `tests/contract/test_collate_metrics.py:63` 이
`("trainbench.collate", "trainbench.metrics", "trainbench.axes")` 순서로 payload 를 찾는다.

`bench.py` 는 이들을 import 해서 쓴다. `build_run` 이 `load_pairs`/`Encode`/`build_collate` 를
쓰고(`bench.py:928`, `:935`, `:953`), `pooled_embeddings`/`train` 이 `MicroBatch` 를 쓴다.

### 테스트 재배선 — 실측 26곳

`tests/test_smoke_cpu.py` 만 영향을 받는다 (이 세션에서 직접 셈):

```
Collate 14 · PairDataset 5 · build_collate 3 · MicroBatch 2 · Encode 1 · PackedBatches 1
```

`PairTexts`, `load_pairs`, `_group_by_row`, `PackedPairs`, `PretokenizedCollate`,
`MMEB_IMAGE_MARKER` 는 테스트에서 `bench_entry.` 로 참조되지 않는다.

`monkeypatch.setattr(bench_entry, …)` 는 `train`(1346)과 `current_gpu_arch`(1470/1593/1602)에만
걸려 있고 **둘 다 이동하지 않는다.** 이동하는 심볼에 걸린 monkeypatch 는 없다.

`bench_entry` 는 `tests/test_smoke_cpu.py:42` 에서 `importlib.util.module_from_spec` 으로
**파일 경로로** 적재된다. 재배선은 `from trainbench import collate` 를 추가하고 이동한
26곳을 `collate.` 로 바꾸는 것이다. `bench.py` 에 재-export 를 남겨 테스트를 그대로 두는 방식은
**쓰지 않는다** — `tests/contract/test_collate_metrics.py:90-92` 가 `bench.py` 를 파일 경로로
적재하는 이유가 "깨진 bench.py 가 멀쩡한 collate.py 를 가리지 못하게" 하려는 것이고,
재-export 는 그 분리를 되돌린다.

### 순수 이동의 증명

동작이 바뀌지 않았다는 증거는 **이동 전후의 `pytest` 수가 같다는 것**이다.
이동 커밋에서 테스트 로직을 함께 고치지 않는다. 재배선(import 경로)만 허용된다.

## 작업 2 — 어댑터 seam

지금 `build_run`(`bench.py:903-962`)은 프레임워크를 세 곳에서 못박는다:

```text
912    from transformers import AutoModel, AutoProcessor
916    processor = AutoProcessor.from_pretrained(config.model.hf_id, revision=...)
920    model = AutoModel.from_pretrained(config.model.hf_id, revision=..., dtype=..., **load_kwargs)
936    built, applied = axes.assemble(model, config, device, framework="native", dataset=dataset)
```

`framework="native"` 리터럴 때문에 `framework=unsloth` 로 돌려도 native 모델이 만들어지고
`_capture_framework` 가 `"native"` 를 읽어 mismatch → `assert_matches` 가 거부한다.
**fail-closed 이지만 Phase 3 가 통째로 막혀 있다.**

해야 할 것: `bench.py` 안에 바인딩을 만들고 `assemble` 에 **비상수**로 넘긴다.

```text
built, applied = axes.assemble(model, config, device, framework=binding.framework, dataset=dataset)
```

`tests/contract/test_loader_bench.py:576-586` 의 검사는 AST 로
`node.func.attr == "assemble"` 의 `framework` 키워드가 `ast.Constant` 인지만 본다.
상수가 아니면 통과한다.

### 바인딩의 필드 이름은 `AdapterOut` 과 정확히 같게

`tests/contract/test_loader_bench.py:38-51` 이 고정한 여덟 개:

```
framework  model  processor  step  owned_axes  required_step_context
fingerprint  documented_entry_point
```

split 은 앞의 셋(`framework`, `model`, `processor`)만 채우고 나머지는 `None`/빈 값으로 둔다.
adapters 레인이 `trainbench/loader.py` 의 `AdapterOut` 을 그 이름 그대로 만들면 **그대로 꽂힌다.**

### 해석은 있으면 쓰고 없으면 native 로 떨어진다

```text
# trainbench.loader 가 아직 없으므로, 없으면 지금의 native 경로를 그대로 쓴다.
# adapters 레인이 그것을 만들면 bench.py 는 한 줄도 바뀌지 않고 그것을 탄다.
```
`importlib.import_module("trainbench.loader")` 를 시도하고 `ModuleNotFoundError` 면 fallback.
fallback 은 지금의 912–926 을 그대로 옮긴 것이어야 한다 — **동작이 바뀌면 안 된다.**

### 절대 하지 않는 것 — `trainbench/loader.py` 를 만들지 않는다

`test_loader_serves_every_framework_through_one_entry_point`(`test_loader_bench.py:558`)는
`xfail(strict=True)` 이고 다음 셋만 본다:

```text
loader = importlib.import_module("trainbench.loader")
assert set(loader.ADAPTERS) == FRAMEWORKS          # 6개
assert callable(loader.load)
assert {f.name for f in dataclass_fields(loader.AdapterOut)} == ADAPTER_OUT_FIELDS
```

`NotImplementedError` 를 던지는 6키 스텁이어도 **세 단언이 전부 통과한다.**
그러면 XPASS → strict → **빨강**. 그 마커는 adapters 의 것이고, 지우려면 adapters 가 실제
어댑터를 만들어야 한다. 스텁을 만들면 adapters 는 자기가 하지 않은 일에 대해 마커를 지우거나
빨간 트리를 물려받는다. **둘 다 이 계약이 막으려던 것이다.**

### 깨면 안 되는 두 불변

지금 통과하는 테스트가 고정하고 있다:

- `axes.assemble` 의 `framework` 파라미터는 **기본값 없이** 남고 `Built` 는 `framework` 필드를
  유지한다 (`test_loader_bench.py:520-528`)
- `scripts/bench.py` 는 `patch`/`load_kwargs`/`assemble`/`step_context`/`assert_matches` 다섯
  호출과 `built.loss_fn` 에서 온 `loss` 바인딩을 **텍스트로** 갖고 있어야 한다
  (`scripts/audit_plan.py` 의 `assert-called` 체크, `BENCH_ENTRY_POINT` 가 `scripts/bench.py` 로
  하드코딩돼 있다). 이 호출들을 `collate.py` 로 밀어내면 게이트가 죽는다.
  `bench.py:16-21` 의 docstring 이 `assert_matches` 를 `steps.verify_axes` 를 통하지 않고
  직접 불러야 하는 이유를 적어두었다 — `report.run(...)` 이 raise 를 삼킨다.

## 작업 3 — `recorded_at`

`trainbench/record.py:156 build_record` 는 `recorded_at` 을 쓰지 않는다.
이 세션에서 직접 확인: 저장소 전체에서 `recorded_at` 을 **쓰는** 곳은
`scripts/publish_result.py:101` 하나뿐이고, 그것은 발행 래퍼이지 런 레코드가 아니다.
읽는 곳은 `scripts/report.py:232-235` 이고 없으면 `path.stat().st_mtime` 으로 떨어진다.

결과: 결과 저장소의 아티팩트 40개 중 `recorded_at` 을 실은 것이 **0건**이고,
타임스탬프가 같아지면 18칸 중 8칸이 이전 캠페인 아티팩트를 고른다.

`build_record` 가 `recorded_at` 을 찍게 한다. **어떤 시계를 쓰는지 주석 한 줄로 적는다** —
`report.py` 가 정렬 키로 쓰므로 단조성이 결과 선별의 정확성이다.

지우는 마커: `tests/contract/test_record_report.py` 의
`test_the_producer_stamps_the_identity`.

> mtime fallback 을 **없애는** 것은 report 레인의 일이다(`test_an_artifact_without_the_identity_
> _is_refused_rather_than_dated_by_its_file`). 이 레인은 생산 쪽만 한다.

## 완료 조건

각 항목은 명령을 직접 돌리고 원문 tail 을 보고한다.

1. **순수 이동이 증명된다**
   `infisical run --env=dev -- uv run pytest` → 캠페인 base 와 같은 통과 수
   (base: `877 passed, 38 xfailed`. 마커 2개를 지우므로 **`879 passed, 36 xfailed`** 가 되어야 한다)
2. **collate 계약이 새 모듈을 본다**
   `infisical run --env=dev -- uv run pytest tests/contract/test_collate_metrics.py -q`
   그리고 `python -c "import trainbench.collate as m; print(m.MicroBatch, m.build_collate)"`
3. **framework 리터럴이 사라졌다**
   `infisical run --env=dev -- uv run pytest tests/contract/test_loader_bench.py::test_bench_takes_the_framework_name_from_the_adapter -q`
   (마커를 지운 뒤 단독 실행)
4. **`recorded_at` 이 찍힌다**
   `infisical run --env=dev -- uv run pytest tests/contract/test_record_report.py::test_the_producer_stamps_the_identity -q`
5. **`trainbench/loader.py` 가 없다**
   `test -e trainbench/loader.py && echo VIOLATION || echo ok`
   그리고 `pytest tests/contract/test_loader_bench.py -q` 가 xfail 1개를 남긴 채 초록
6. **`assert-called` 가 살아 있다**
   `infisical run --env=dev -- uv run python scripts/audit_plan.py --only assert-called`
   (부분 실행은 wave 게이트가 아니다 — 확인용이고, 최종 판정은 전체 실행이다)
7. **네 게이트**
   `plan-files` 는 `trainbench/collate.py` 하나로 빨갛다. 그것은 이 레인의 실패가 아니다 —
   `newFiles` 에 적어 보고하면 머지 단계가 등재한다. **다른 이름이 함께 나오면 그것은 실패다.**
8. **부숴서 확인**
   - seam 을 상수로 되돌리면(`framework="native"`) 3번이 죽는가
   - `build_record` 의 `recorded_at` 을 지우면 4번이 죽는가
   - `collate.py` 의 `build_collate` 를 `raise` 로 갈면 무엇이 죽는가.
     **사보타주를 믿기 전에** `python -c "import trainbench.collate as m; f=m.build_collate; print(f.__code__.co_filename, f.__code__.co_firstlineno)"` 로
     인터프리터가 잡는 정의가 네가 고친 자리인지 확인한다 (`HAZARDS.md §3`)

## 남길 것

- `.plans/notes/split.md` — `PLAN.md` 레이아웃에 등재할 신설 파일 목록,
  다른 레인이 알아야 할 seam 의 정확한 이름과 필드
- `boundaryRequests` — 계약이 seam 을 표현하지 못하면 여기에. **계약 파일을 고치지 않는다.**
- `notMeasured` — 못 낸 숫자

## 하지 않는 것

- 토큰 회계·측정 통계·유효성 게이트·피크 메모리 — **measure** 레인 (wave 1)
- packing 의미·varlen kwargs·프롬프트 — **packing** 레인 (wave 2)
- 어댑터 구현·빌드 지문 — **adapters** 레인 (wave 2)
- `trainbench/metrics/` 내용, `config_schema.py` — **measure**
- `PLAN.md`, `docs/audit-baseline.json`, `envs/**`, `pyproject.toml`, `scripts/audit_plan.py` —
  **통합자 전용**
