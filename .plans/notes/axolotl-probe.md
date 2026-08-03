# axolotl-probe 레인 노트 — 머지 단계로 넘기는 것

## 1. 무엇을 고쳤나

`trainbench/probe/axolotl.py::run` 의 `infonce_backward` 가 `steps.infonce_backward`를
직접 불렀다. `bench.py:278`(`with timer, axes.step_context(config, required_context):`)은
어댑터의 `required_step_context`를 안에서 실행하는데, 프로브 경로는 `loader.load()`를
아예 거치지 않아 그 요구가 전달될 곳이 없었다 — `grep -rn "autocast|step_context"
trainbench/probe/` 는 이 수정 전까지 0건이었다(직접 확인).

axolotl 0.18.0은 `embed_tokens`/`lm_head`/`*norm*`을 fp32로 남기고 나머지를 bf16으로
적재한다(`loaders/model.py:1025-1047`, 복귀 조건 4개가 이 cfg에서 전부 거짓). fp32
임베딩 출력이 0번째 블록 `q_proj`의 bf16 가중치와 만나 `F.linear`에서
`RuntimeError: expected mat1 and mat2 to have the same dtype`로 죽는다(2026-08-03,
실측 A100). InfoNCE 자체의 문제가 아니다.

`_infonce_backward`를 `axes.step_context(config, required)` 안에서 돌게 바꿨다.
`required`는 `trainbench.loader.ADAPTERS["axolotl"].required_step_context` 하나에서만
읽는다 — 이미 `loader.py:513-536`이 유일한 선언이고, `describe()`가 그것을
`AdapterOut.required_step_context`로 실어 `scripts/bench.py`에 전달하는 것과 같은
경로다. 프로브 쪽에 두 번째 `StepContext(...)`를 새로 짓지 않았다: `code-craft.md`의
"find before adding"과, 이미 `trainbench/probe/sentence_transformers.py:22`가
`from trainbench.loader import AdapterRefusal`로 이 모듈을 참조하는 선례를 따른 것.

## 2. CPU 경로 — 의도한 동작

`axes._autocast_step_context`는 `required.device_type`("cuda")가 `get_device(config.device)`와
다르면 `UnappliedAxis`를 던진다(`axes.py:856-884`). 이 호스트는 CUDA가 없으므로
`infonce_backward` 체크는 **항상 실패한다** — `expected_failure`로 표시하지 않았다:
이것은 프레임워크의 한계가 아니라 호스트의 한계이고, `expected_failure`를 붙이면
"axolotl은 이걸 못 한다"는 잘못된 주장이 된다. 같은 자리에서 이미 확립된 패턴을
따랐을 뿐이다 — `axes.load_kwargs`가 CPU에서 `peft.mode=qlora`를 거부할 때도
`expected_failure` 없이 그냥 실패한 체크로 기록된다
(`tests/test_probe.py::test_a_refused_load_axis_does_not_read_as_a_model_that_will_not_load`).
`report.run`이 모든 예외를 잡으므로(`ProbeReport.run`, `probe/types.py`) 프로브
프로세스는 죽지 않고, 이유가 있는 실패 체크 하나가 남는다.

## 3. 다른 프로브 모듈은 손대지 않았다 — 확인함

`trainbench/loader.py`의 `Adapter(...)` 호출 6개 중 `required_step_context`를
선언하는 것은 axolotl 하나뿐이다(`grep -n "required_step_context=StepContext"
trainbench/loader.py` → 1건, line 516). unsloth/ms_swift/sentence_transformers/
tevatron/native는 전부 `None`(기본값). `axes.step_context(config, None)`은
`_precision_recipe(config)`로 빠지고 bf16 precision에서는 `contextlib.nullcontext()`를
돌려주므로(`axes.py:952-955` 부근), 이 다섯 프로브가 지금 하는 대로 `steps.*_backward`를
직접 부르는 것과 결과가 같다 — 바꿀 이유가 없다.

## 4. accelerate와의 두 가지 확인된 차이 (상류 대비)

axolotl 0.18.0은 이 호스트에 없어 axolotl 쪽 코드는 직접 못 열었다(확인 안 함 —
axolotl이 accelerate의 `prepare_model`을 실제로 그대로 쓰는지, 아니면 자체 래퍼를
얹는지는 axolotl 소스 확인 없이는 모른다). accelerate는 axolotl 이미지가 고정하는
버전 그대로 `uv cache`에서 열어 확인했다(`envs/axolotl/uv.lock` accelerate==1.13.0,
uv 캐시 `archive-v0/fEtyCRRqLx-Vto5O/accelerate/accelerator.py`가 그 배포판 원문).

- **accelerate는 `model.forward`만 autocast로 감싼다, loss나 `backward()`는 아니다**
  (`Accelerator.prepare_model`, accelerate 1.13.0 `accelerator.py:1807-1818`: `if
  self.native_amp: ... model.forward = convert_outputs_to_fp32(autocast_context(
  model_forward_func))`, 원문 그대로 실측). 이 저장소의 `axes.step_context`는
  `scripts/bench.py:278`에서 `with timer, axes.step_context(...):` 로 zero_grad부터
  backward까지 스텝 전체를 감싸 — forward + loss + backward가 한 autocast 리전 안에
  있다. 두 규칙이 감싸는 범위가 다르다.
- **accelerate는 추가로 `convert_outputs_to_fp32`로 forward 출력을 fp32로 되돌린다**
  (같은 줄, `accelerator.py:1813`). `axes._autocast_step_context`(`axes.py:856-884`)는
  `torch.autocast(device_type=..., dtype=...)`만 반환하고 그런 변환이 없다 — InfoNCE가
  보는 logit의 dtype이 상류와 이 하니스에서 다를 수 있다.

두 차이 모두 코드로 흡수하지 않았다(범위 밖 — 브리프가 "기록만" 요청). 실제 수치
영향은 측정 안 함.

## 5. 게이트 (직접 재실행, 이 세션)

`ruff check`, `ruff format --check`, 전체 `pytest`, `pytest tests/contract`,
`audit_plan.py` 결과는 보고 본문에 원문 그대로 인용.
