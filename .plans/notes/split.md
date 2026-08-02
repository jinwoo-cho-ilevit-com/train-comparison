# split — 머지 단계로 넘기는 것

## PLAN.md 레이아웃에 등재할 신설 파일

`trainbench/` 블록은 자식을 전부 열거하므로 한 줄이 필요하다. `record.py` 다음 자리:

```
│   ├── collate.py             # 행 -> MicroBatch. collate-metrics 경계가 이 이름을 찾는다
```

이 한 줄이 없으면 `plan-files` 가 `trainbench/collate.py` 하나로 빨갛다.

## seam 의 정확한 이름과 필드 — adapters 레인이 볼 것

`scripts/bench.py` 는 이제 프레임워크를 리터럴로 부르지 않는다.

```
Binding(NamedTuple)          scripts/bench.py
native_binding(config, device) -> Binding
load_framework(config, device) -> Binding
```

- `Binding` 의 필드는 `AdapterOut` 여덟 개와 이름이 같다:
  `framework, model, processor, step, owned_axes, required_step_context,
  fingerprint, documented_entry_point`. split 은 앞의 셋만 채우고 나머지는 `None`.
- `load_framework` 는 `importlib.import_module("trainbench.loader")` 를 시도하고
  `ModuleNotFoundError` 면 `native_binding` 으로 떨어진다. 그 fallback 은 이동 전
  `build_run` 안에 있던 912-926 을 그대로 옮긴 것이다.
- **adapters 레인이 맞춰야 할 호출 규약**: `loader.load(config, device)` 가
  `Binding` 과 같은 필드를 가진 객체를 돌려주면 `bench.py` 는 한 줄도 바뀌지 않는다.
  계약(`test_loader_bench.py:563`)은 `callable(loader.load)` 까지만 고정하고 인자를
  고정하지 않는다. 이 두 인자가 `bench.py` 가 실제로 넘기는 것이다.
- `axes.assemble(..., framework=binding.framework, ...)` 이므로 어댑터가 자기 이름을
  적어 넣으면 `_capture_framework` 가 그것을 읽는다.

## `Binding` 이 dataclass 가 아닌 이유 — 다른 레인이 다시 밟지 않게

`scripts/bench.py` 는 `sys.modules` 등록 없이 파일 경로로 exec 되는 경로가 있다
(`tests/test_pods.py` 의 `FAKE_BENCH` 가 preflight 를 그렇게 부른다). 그 상태에서
`@dataclass` 는 `sys.modules.get(cls.__module__)` 이 `None` 이라
`AttributeError: 'NoneType' object has no attribute '__dict__'` 로 죽는다.
이 세션에서 실제로 그렇게 죽었고 `tests/test_pods.py` 12개가 빨갰다.
**`scripts/bench.py` 안에 dataclass 를 두지 않는다.** NamedTuple 은 안전하다.

## `recorded_at` 의 시계

`trainbench/record.py::build_record` 가 `time.time()` 을 찍는다. monotonic 이 아닌
이유는 `scripts/report.py` 가 서로 다른 파드의 아티팩트를 한 줄에 세우기 때문이다 —
monotonic 은 한 프로세스 안에서만 비교 가능하다. NTP 보정만큼의 오차는 남는다.

mtime fallback 제거는 report 레인의 일이다
(`test_an_artifact_without_the_identity_is_refused_rather_than_dated_by_its_file`).

## 테스트 재배선

`tests/test_smoke_cpu.py` 의 `bench_entry.<moved symbol>` 26곳이 `collate.` 로 바뀌었다
(이 세션에서 셈: `Collate` 14, `PairDataset` 5, `build_collate` 3, `MicroBatch` 2,
`Encode` 1, `PackedBatches` 1). 부수적으로 지역변수 `collate` 5개가 모듈 이름을 가려
`collate_fn` 으로 개명됐다 — 로직은 그대로다.

`monkeypatch.setattr(bench_entry, "load_pairs", ...)` 는 그대로 둔다.
`bench.py` 가 `from trainbench.collate import load_pairs` 로 모듈 전역에 이름을 두므로
그 setattr 은 여전히 `build_run` 이 보는 이름을 바꾼다.
