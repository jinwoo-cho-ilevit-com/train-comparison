# adapters 레인 노트 — 머지 단계로 넘기는 것

## 1. 신설 파일 (PLAN.md 레이아웃 등재)

- `trainbench/loader.py` — 어댑터 레지스트리, 빌드 지문, 여섯의 선언
- `tests/test_loader.py` — 이 레인의 게이트

`trainbench/` 블록은 자식을 전부 열거하므로 `kernels.py` 다음 자리에 한 줄이 필요하다:

```
│   ├── loader.py               # 프레임워크 -> AdapterOut. loader-bench 경계가 이 이름을 찾는다
```

## 2. 소유 밖 변경 — 두 파일

`trainbench/probe/tevatron.py` 와 `trainbench/probe/sentence_transformers.py` 에
`load(config, device, load_kwargs)` 를 하나씩 넣었다. **probe 레인 소유다.**

넣어야 했던 이유는 `scripts/audit_plan.py::_per_image_adapters` 다. `doc-commands` 는
`trainbench/` 아래 모든 서드파티 import 를 루트 lock 에 요구하고 **`trainbench/probe/<env>.py`
여섯 개만 면제**한다. 프레임워크 적재를 `loader.py` 에 두면 `unsloth`/`swift`/`axolotl`/
`tevatron`/`sentence_transformers` 다섯이 루트 lock 요구가 되고, 그것은 `envs/` 가 존재하는
이유인 해석 불가능한 잠금이다. **실측: 처음에 `loader.py` 에 두었을 때
`tests/test_audit.py::test_a_lazy_third_party_import_in_the_package_is_demanded_of_the_documented_setup`
가 `assert 'unsloth' not in modules` 로 빨개졌다.**

그래서 여섯 적재는 여섯 프로브 모듈에 남고 `loader.Adapter.load` 가
`importlib.import_module(f"trainbench.probe.{name}").load(...)` 로 늦게 잡는다.
`tests/test_loader.py::test_the_framework_imports_stay_out_of_this_module` 가 그 성질을 잠근다.

두 파일에서 한 것은 **추가와 추출뿐**이다:

- `tevatron.py`: `run` 의 `_load` 클로저 안에 있던 적재를 `load_dense_model(config)` 로
  꺼내고 `_load` 가 그것을 부른다. `load()` 는 그 위에 `model.to(device)` + `AutoProcessor` 를
  얹는다. `plant_pad_token_id` 는 손대지 않았다
- `sentence_transformers.py`: `_load` 클로저의 `SentenceTransformer(...)` 를 `load()` 로
  꺼내고 `_load` 가 그것을 부른다. **padding-side 정렬을 하지 않는다는 설계는 유지했다** —
  `Adapter.aligns_padding_side=False` 가 그것을 loader 쪽에 그대로 옮긴 것이다

## 3. 다른 레인에 필요한 변경 (요청만 한다)

### 3.1 axes 레인 — `Built.owned_axes` 로 가는 길이 없다 (막힘)

`applied.Built.owned_axes` 는 존재하고 `applied._owned` 가 그것을 읽는다. 그런데
**아무도 채우지 않는다.** `axes.assemble()` 은 `Built(...)` 를 만들면서 `owned_axes` 를
넘기지 않고, `scripts/bench.py::build_run` 도 `binding.owned_axes` 를 전달하지 않는다.

결과: tevatron 칸의 `loss.name` / `parallel.cross_device_negatives` 는 어댑터가 선언해도
레코드에서 `undetermined` 로 남고, `assert_matches` 가 timing 런을 거부한다.
**이 레인이 만든 선언은 아직 어디에도 도착하지 않는다.**

필요한 것 (둘 다 이 레인 소유 밖):
- `axes.assemble(..., owned_axes: Mapping[str, str] = ())` → `Built(owned_axes=...)`
- `scripts/bench.py::build_run` 이 `owned_axes=binding.owned_axes` 를 넘긴다

### 3.2 axes 레인 — `step_context` 가 어댑터의 요구를 받지 않는다 (막힘)

`axes.step_context(config)` 는 config 만 받고 bf16 이면 `nullcontext()` 를 돌려준다.
axolotl 어댑터가 내놓는 `required_step_context`(kind=autocast, cuda, bfloat16)를 받을
파라미터가 없다. 계약(`test_loader_bench.py:191-217`)은 어댑터가 **요구만** 하고
`axes.step_context` 가 **세운다**고 못박으므로, 어댑터가 자기 `with torch.autocast(...)` 를
여는 것은 금지다. 그래서 이 레인은 요구만 만들었다.

필요한 것: `axes.step_context(config, required=None)` 가 `required.kind == "autocast"` 일 때
`torch.autocast(device_type=required.device_type, dtype=getattr(torch, required.dtype))` 를
돌려준다. 그리고 `scripts/bench.py` 가 `axes.step_context(config, binding.required_step_context)`
로 부른다.

**측정 안 함**: autocast 를 켠 axolotl 과 끄고 잰 axolotl 의 차이. CUDA 가 없다.

### 3.3 split/bench — `refusing("load_kwargs")` 태그가 사라진다

`scripts/bench.py::native_binding` 은 `with refusing("load_kwargs"): axes.load_kwargs(config)`
안에서 부른다. `trainbench.loader` 가 생긴 지금 `load_framework` 는 그 fallback 을 타지 않고
`loader.load` 를 부르며, `loader.load` 안의 `axes.load_kwargs` 는 `refusing` 밖이다.
`UnappliedAxis` 가 나가면 `RefusedSetting` 으로 태그되지 않아 `refusal_record` 대신
`main` 의 광의 `except` 로 떨어진다.

필요한 것: `build_run` 의 `binding = load_framework(config, device)` 를
`with refusing("load_kwargs"):` 로 감싼다. **런은 어느 쪽이든 멈추므로 blocking 은 아니다** —
바뀌는 것은 결과 파일이 정돈된 거부 레코드인가 아닌가다.

같은 자리에서, `native_binding` 은 이제 도달 불가 경로다(`loader.py` 가 항상 있다).
지우는 것은 통합 단계의 판단이다 — 남겨두면 `trainbench.loader` 가 없는 체크아웃에서의
안전망이고, 동시에 native 적재의 두 번째 정의다.

### 3.4 integrate — `docs/CONTRACTS.md` §2 에 들어갈 두 줄

1. **axolotl 의 autocast 배선.** 어댑터는 `AdapterOut.required_step_context` 로 요구를
   선언하고 `axes.step_context` 가 그것을 세운다. 정밀도 컨텍스트를 세우는 자리는 계속
   하나다. native(순수 bf16)와 axolotl(autocast)이 **다른 수치 체제**라는 사실은
   `documented_entry_point.differs` 와 이 필드 둘 다에 남는다.
2. **`loader-bench` 규칙의 구현이 둘인 이유.** 계약 파일(`tests/contract/test_loader_bench.py`)
   은 fixture 를 import 없이 검증하고, `trainbench/loader.py` 의 `__post_init__` 들과
   `_refuse_a_build_the_fingerprint_condemns` 는 **살아 있는 객체**를 검증한다.
   `tests/test_loader.py::test_a_live_adapter_out_passes_the_frozen_contract_validator` 가
   두 판정이 같다는 것을 증명한다. `trainbench/kernels.py` 가 `attention` 블록에 대해
   같은 구조를 이미 갖고 있다.

## 4. 경계 개정 요청 — `tests/fixtures/adapter_out.sample.json`

**고치지 않았다.** fixture 의 `sentence_transformers.documented_entry_point` 는
`differs: null`, `source: "확인 안 함 — ... lane-g settles them inside the ST image;
nothing on this host can."` 로 적혀 있다.

**이 호스트에서 답이 나왔다.** `.plans/research/sentence-transformers.md` 가 핀된 휠을 열어
둘 다 인용했다:

- 자체 loss 가 있는가 → 있다. `SentenceTransformerTrainer(BaseTrainer)` 는 `training_step`
  을 오버라이드하지 않고(휠 전체에서 `def training_step` 0건) 진입점은 `compute_loss` 이며
  모델 forward 를 부르는 것은 Trainer 가 아니라 **손실 함수**다
  (`base/trainer.py:76`, `:459-509`, `sentence_transformer/trainer.py:36`)
- `processor(text=..., images=...)` 규약을 받는가 → **받는다. transformers v5 규약으로.**
  `base/modules/transformer.py:1258-1290` `_call_multimodal_processor`,
  `base/modality_types.py:59-67` 의 `MODALITY_TO_PROCESSOR_ARG`,
  분기 스위치 `transformer.py:113` 이 `transformers > 4.56.1` 에서 v5 경로를 고른다

그래서 이 레인의 live 어댑터는 `differs=True` 로 인용과 함께 선언한다. **fixture 와 live 가
이 한 칸에서 어긋난다.** 계약 테스트는 fixture 만 검증하므로 지금은 초록이지만,
`test_the_sample_exercises_every_branch_the_contract_has` 가 `differs is None` 인 항목이
하나 있기를 요구하므로 **fixture 를 고치려면 그 브랜치를 어디로 옮길지 함께 정해야 한다.**
통합 wave 가 하나의 개정본을 낸다.

**남아 있는 확인 안 함**(엔트리포인트와 다른 질문이다): 우리 세 체크포인트에서 ST 의
`modality_config` 에 `text` / `("image","text")` 가 실제로 들어가는지. 리서치 §5 가
"이 호스트에서 확정 불가"로 남겼다. gemma-4-E2B 는 `chat_template.jinja` 가 없어
`"message"` 경로로 강제되지 않는다는 것까지가 소스 독해의 끝이다.

## 5. 이 호스트에서 확인 안 함 — 파드 질문으로 등록

1. 여섯 프레임워크가 실제로 체크포인트를 적재하는가. 이 체크아웃에는 여섯 중 어느
   프레임워크도 설치돼 있지 않고, 이 레인의 테스트는 전부 스텁 모듈 트리 위에서 돈다
2. `kernel=fla` 자동 통과가 CUDA + `causal_conv1d` 에 달려 있는가
3. `EncoderModel.forward(query=, passage=)` 로 실제 배치를 넣었을 때의 예외/성공.
   이 레인은 `step.batch_keys=("query","passage")` 로 **모양만** 고정했다 —
   그 키를 만드는 것은 collate 쪽이고 `trainbench/collate.py` 는 packing 레인 소유다
4. tevatron LoRA 경로에서 `load_kwargs` 를 `DenseModel.load` 로 밀면 `LoraConfig.from_pretrained`
   가 어떤 키를 거부하는가 (`encoder.py:131`, `:170`). 그래서 이 어댑터는
   `honours_load_kwargs=False` 다
5. unsloth 의 `get_peft_model` 이 새 객체를 돌려주는가. `module_classes["model"]` 이
   그것을 잡도록 계약이 요구하지만(`fingerprint taken from a pre-peft object`), 어느 쪽인지는
   이미지 안에서만 답이 나온다

## 6. `docs/audit-baseline.json`

건드리지 않았고 필요한 변경도 없다. 이 레인은 축을 적용하지 않는다.
