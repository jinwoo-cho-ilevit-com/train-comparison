# capture — 넘기는 것

## 신설 파일

없음. `plan-files` 는 이 레인만으로는 빨개지지 않는다.

## 다른 레인·통합자가 해야 할 변경

- `docs/CONTRACTS.md:104-124` 의 `applied.py` 블록이 옛 모양이다. 지금은
  `Built` 에 `precision_recipe` / `owned_axes` 가 있고, `AxisState` 에 `owner` 와
  `state` 가, `AppliedState` 에 `framework_owned()` 가 있다. 레코드 dict 에는
  최상위 `framework_owned` 키가 는다. **integrate 레인 몫.**
- `Built.owned_axes` 는 **멤버십만** 읽는다. `test_loader_bench.py` 가 못박은
  `AdapterOut.owned_axes` 는 축→사유 매핑이므로, adapters 레인은 그 매핑을 그대로
  넘겨도 된다. 지금 capture 가 붙이는 사유는 자기가 만든 한 문장이며, 어댑터의
  사유를 대신 싣고 싶으면 그것은 계약 개정 요청이지 이 레인의 변경이 아니다.
- `axes` 레인: `train.offload` 와 `parallel.strategy` 의 되읽기 경로는 이제
  `applied.zero_stage` / `applied.offload_targets` 두 함수다. `axes.assemble` 이
  `deepspeed.initialize` 를 부르게 되면 그 엔진을 `Built.model` 트리 안에 두어야
  이 둘이 찾는다(`applied._deepspeed_engine`, 클래스 이름 `DeepSpeedEngine` 로 매칭).
  `precision` 은 `Built.precision_recipe` 에 Transformer Engine recipe 객체를
  실어야 읽힌다 — dtype 으로는 영원히 undetermined 다.

## 파드가 답해야 할 것 — 축별로 한 문장

이 호스트에 bitsandbytes / deepspeed / transformer-engine 이 셋 다 없다
(2026-08-02 `uv run python -c "import ..."` 로 확인). 아래는 전부 **확인 안 함**이며,
표로 몰아둔 이유는 파드가 한 번에 고칠 수 있게 하기 위해서다.

- `optim.name` — `optim=adamw_8bit` 로 만들어진 옵티마이저의
  `type(optimizer).__name__` 이 `AdamW8bit` 인가, 아니면 paged 변형인가
  (`applied.OPTIM_CLASS_AXIS` 의 키).
- `precision.name` — `axes.step_context` 가 mxfp8/nvfp4 로 감쌀 때 만드는 recipe 의
  `type(recipe).__name__` 이 `MXFP8BlockScaling` / `NVFP4BlockScaling` 인가
  (`applied.PRECISION_RECIPE_AXIS` 의 키).
- `parallel.strategy` — `deepspeed.initialize` 가 돌려준 엔진에서
  `engine.zero_optimization_stage()` 가 zero2/zero3 config 에 대해 각각 무엇을
  돌려주는가 (`applied.zero_stage`).
- `train.offload` — 같은 엔진에서 `engine.zero_offload_optimizer()` 와
  `engine.zero_offload_param()` 이 offload 를 켠 config 와 끈 config 에 대해 각각
  무엇을 돌려주고, 껐을 때가 `None` 인지 `.device == "none"` 인지
  (`applied.offload_targets`).

넷 다 답이 안 나오면 축은 undetermined 로 남고 timing 런이 거부된다 — 틀린 라벨이
붙는 것이 아니라 멈춘다.

## 감사 baseline

`axis-values` 의 `train.offload 1/4` 는 이 레인이 움직일 수 없다. 되읽기는 열렸지만
`axes.assemble` 이 여전히 `none` 외 세 값을 거부하므로 적용 지점이 없다. `precision`
도 같다. 둘 다 axes 레인 몫이고, baseline note 수정도 그 뒤의 일이다.
