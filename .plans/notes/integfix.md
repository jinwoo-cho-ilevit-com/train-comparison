# integfix — 파티션을 가로지르는 잔여

이 레인이 이 워크트리에서 직접 실행한 것만 적는다. 재실행하지 않은 수는 "확인 안 함".

## 1. 통합자가 적용해야 할 것

- **`docs/CONTRACTS.md:268`** 이 `loader-bench` 행에서 아직
  `_refuse_a_build_the_fingerprint_condemns`(사설 이름)를 부른다. 실제 이름은 공개
  `refuse_a_build_the_fingerprint_condemns`(`trainbench/loader.py:320`)이고 이 레인이
  `tests/test_smoke_cpu.py` 쪽을 공개 이름으로 맞췄다. `docs/CONTRACTS.md` 는 wave 3
  소유라 손대지 않았다.
- **`.plans/notes/kernels.md:33`** 이 `trainbench/axes.py:1405` 를 전달하고 있다. 그 줄은
  Muon param_group docstring 이다. `PackedCollate` 는 `trainbench/axes.py:1736`,
  `__call__` 은 `:1924`. `docs/methodology.md` 쪽은 이 레인이 줄 번호를 빼고
  심볼 인용으로 바꿨다(줄은 밀리고 심볼은 안 밀린다).
- **`.plans/notes/axes.md §2.2`** 에 "실제로 스텝하는 것은 deepspeed 의 래퍼이고 그것이
  우리 인스턴스에 위임한다"는 문장이 남아 있다. `trainbench/axes.py::_deepspeed` 의
  docstring 에서는 이미 걷어냈고 `확인 안 함` 으로 바뀌었다. 노트만 옛 주장을 들고 있다.
- **`trainbench/config_schema.py`** 의 `_no_knob_is_declared_ahead_of_the_code_that_would_apply_it`
  이 `measurement.baseline_tolerance != 0.03` 또는 `baseline_tolerance_calibrated=true`
  를 거부하며, 그 이유로 "pod validity is decided by scripts/report.py's own
  BASELINE_DEVIATION_LIMIT" 를 적는다. **그 전제는 이제 거짓이다** —
  `report.declared_tolerance` 가 baseline 레코드의
  `metrics.measurement.baseline_tolerance` 를 읽어 판정한다. 그 절(그리고 짝인
  `tests/test_config.py` 의 단언)을 걷어내는 것은 measure 레인/통합자 몫이다. 걷어내기
  전까지 교정된 임계값을 실은 config 는 스키마가 막으므로, 새 경로는
  `tests/test_report.py` 가 만든 레코드로만 실행된다.

## 2. 하지 않기로 한 것과 이유

### `parallel=zero2/zero3` 를 측정 루프에 진짜로 배선하기 (선택지 (a))

**하지 않았다.** axes 레인의 (b)(단언을 낮추고 docstring 을 사실에 맞게)를 유지한다.

- 실측: `importlib.util.find_spec("deepspeed")` 가 이 워크트리에서 `None` 이다.
  `engine.backward`/`engine.step` 이 무엇을 하는지 핀된 소스로 확인할 수 없고,
  확인 없이 쓰는 것이 HAZARDS §1 이 금지하는 바로 그 모양이다.
- 배선은 `scripts/bench.py::train` 의 backward/step 두 줄을 분기시키는 일인데, 그 분기는
  이 호스트에서 한 번도 실행될 수 없다. 커버되지 않는 분기를 측정 루프 한복판에 넣는 것은
  ZeRO 아닌 런까지 위험에 넣는다.
- `tests/test_axes.py::test_the_measured_step_never_drives_the_engine` 이 그 간극을
  열어둔 채로 못박고 있다. 지금 상태는 "모르는 것을 모른다고 적은 상태"다.

**파드가 답할 질문** (deepspeed 가 있는 이미지에서, 2랭크):

1. `parallel=zero2 train.offload=none` 로 `deepspeed.initialize` 가 돌려준 옵티마이저
   래퍼의 `type(...).__name__` 과 `engine.optimizer.optimizer is built.optimizer` 여부.
2. 우리 루프의 `loss.backward()` + `built.optimizer.step()` 만으로 실제 파라미터가
   갱신되고 파티셔닝이 일어나는가 — `engine.backward`/`engine.step` 없이. 판정 근거는
   스텝 전후 파라미터 델타와 랭크별 옵티마이저 상태 텐서 수.
3. 2가 거짓이면 `zero2/zero3` 와 `train.offload` 행은 **측정 불가**로 확정하고 축을
   닫는다. 참이면 `_deepspeed` docstring 의 `확인 안 함` 을 그 출력으로 대체한다.

### `measurement.repeats` 반복 루프

**하지 않았다.** metrics 레인의 선택("레코드에 적용된 것처럼 싣지 않는다")이 유지된다:
`trainbench/config_schema.py:330` 이 `repeats != 1` 을 거부하고 그 메시지가
`scripts/bench.py runs one` 이라고 적는다. record-report 계약은 건드리지 않았다.

### `kernels.KernelProvenanceError` 를 `AppliedMismatch` 상속으로 바꾸기

**하지 않았다.** 실측: `AdapterRefusal.__mro__` 는 `AppliedMismatch` 를 포함하고
`KernelProvenanceError.__mro__` 는 포함하지 않는다. 그런데 둘 다 이미
`scripts/bench.py::refusal_types()` 가 명시적으로 나열하므로 `refusing()` 안에서 잡힌다
(`tests/test_smoke_cpu.py` 의 `UnsafePacking` 케이스가 그것을 end-to-end 로 못박는다).
상속을 붙이면 `except AppliedMismatch` 를 쓰는 다른 자리들이 커널 출처 실패를 축 불일치로
삼키게 되고, 레코드의 `refusal.kind` 가 구별하는 두 발견이 하나가 된다.

**부수 발견 (mutation `inert`)**: `refusal_types()` 에서 `loader.AdapterRefusal` 줄을
지워도 `test_an_adapter_refusal_is_filed_as_a_result_instead_of_escaping_main` 이
통과한다 — 상속 때문에 `AppliedMismatch` 항목이 이미 잡는다. 그 줄은 방어적으로 옳지만
게이트가 아니다. 실제 게이트는 `refusing()` 의 `except refusal_types()` 자체이고, 그것을
`(axes.UnappliedAxis,)` 로 좁히면 테스트가 죽는다.

## 3. 이 호스트가 답할 수 없는 것 — 확인 안 함

CUDA·deepspeed·transformer-engine·DALI·fla·causal-conv1d 가 이 워크트리에 없다.

- `precision=mxfp8/nvfp4` 가 **하드웨어** 때문에 막히는지. 실측(2026-08-03):
  `transformer_engine` 이 `ModuleNotFoundError` 이고 `axes._precision_supported` 가 둘 다
  `transformer_engine.pytorch.quantization is not importable here` 로 끝난다 —
  `is_mxfp8_available()` 도 CC 비교도 실행되지 않는다. `docs/audit-baseline.json` 의
  `axis-values` note 를 그 사실에 맞게 다시 썼다(count 는 건드리지 않았다).
- `kernel=fla` 가 fla 있는 이미지에서 실제로 적용되는지. 동반값
  `model=qwen3_5_0_8b` 를 넣어 거부 사유가 아키텍처 고정에서 환경 의존으로 바뀐 것까지가
  이 호스트가 낼 수 있는 전부다. 이 호스트의 `kernel` 그룹 수는 여전히 1/3.
- profile 목적 런의 커널 출처. `close_kernel_fetch_doors` 와
  `assert_no_runtime_kernel_fetch` 는 `ENFORCED_PURPOSES`(timing/quality)에서만 돈다.
  `docs/methodology.md §11` 에 그대로 적었다.
- 실 이미지 안에서 probe preflight 가 발화하는지 (`verdicts-closed` 의
  `images-carry-a-code-snapshot-nothing-checks-is-current`). 이 레인은 preflight 의
  **목적별 갈래**만 바꿨고 파드는 띄우지 않았다.
