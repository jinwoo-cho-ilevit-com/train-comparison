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

## 6. qwen3_5 회귀 — `scripts/verify_env.py`의 가드는 이 칸을 고치지 않는다

2026-08-03, A100 파드 `zql0z8hc4k8dlx`/`x2i12l0tyqzf2a`에서
`phase0-axolotl-qwen3_5_0_8b`(`kernel=fla`, `run=probe`, 커밋 `7a0f712`)가
`no_result`/`"no result file after the run (exit 1)"`로 죽었다. `report.run`이
`_infonce_backward`(§1의 그 지점)를 감싸므로 파이썬 레벨 예외였다면 실패 체크로
남았을 것이다 — 남지 않았다.

**실측(통합 세션, 파드 `x2i12l0tyqzf2a`, 라이브 로그):**
```
Loading weights: 100%|██████████| 473/473 [00:00<00:00, 3344.37it/s]
failed to wait for command termination: exit status 139
```
`139 = 128 + 11 = SIGSEGV`. 이것은 신호에 의한 하드 프로세스 종료이고, 파이썬
레벨에서 잡을 수 있는 예외가 아니다.

**`scripts/verify_env.py::main()`에 추가한 `try/except` 가드(같은 브리프, 같은
세션)는 이 칸을 고치지 않는다.** 그 가드는 `run_probe()`가 던지는 파이썬 예외
(`report.run()`의 개별 체크 밖에서 escape하는 것)를 잡아 `probe_process` 체크로
기록하는 것이 목적이고, 실측으로 다른 실패 모드(`run_probe`가 `MemoryError`를
던지게 한 변이)에서는 정확히 그렇게 동작한다 — 하지만 SIGSEGV는 파이썬
인터프리터에 도달하지 않으므로 그 `except`가 아예 실행되지 않는다. 이 가드는
**escape하는 파이썬 예외에 대한 안전망으로는 유효하고 유지한다** — 이 칸의 실패
원인이 아니다. 다음에 이 노트를 읽는 사람이 "가드를 넣었으니 닫혔다"로 읽지 않게
여기 못박는다.

가드의 경계는 세 방향으로 좁다. 신호사는 위 이유로 못 잡고, `load_bench_config`/
`get_device`/`set_seed`는 `try` **밖**이므로 config를 못 읽는 이미지는 여전히
레코드를 남기지 않으며(그 질문은 `bench.py --preflight`가 앞에서 답한다), `Exception`만
잡으므로 `SystemExit`은 통과한다 — 고의로 고른 종료 코드를 이 함수 자신의 1로
납작하게 만드는 것은 `run_with_secrets`가 없애려는 그 laundering과 같은 것이다.

### 원인 재평가 — 자동캐스트 리전이 아니라 모델 구성 시점으로 보였으나, HF 결과 저장소 실측 diff가 그 판단을 다시 좁혔다

로그의 타이밍("Loading weights" 완료 직후 ~2초 뒤 죽음, 어떤 체크도 결과에 남지
않음)만 보면 죽는 지점이 `axes.step_context`/autocast 리전(§1) **안**이 아니라
모델 **구성** 시점처럼 보인다. 그러나 체크는 메모리에 쌓였다가 `write_json` 한
번에만 직렬화되므로(`scripts/verify_env.py:38-40`), 총 프로세스 죽음에서는 어느
체크가 실제로 돌았는지 로그 타이밍만으로 알 수 없다 — "체크가 하나도 안 남았다"는
것이 "axes_patch도 안 돌았다"의 증거가 아니다.

**HF 결과 저장소(`jinwoo-cho/trainbench-results`)에서 세 파드의 `started.json`/
`result.json`을 직접 받아 diff했다(이번 세션, `hf_hub_download`):**

| pod | git_commit (기록값) | image tag (실제 실행 코드) | image digest | 결과 |
|---|---|---|---|---|
| `xzrx2gnudntf09` | `f9d9b0e` | **`db8337a`** | `sha256:8405479c...` | 완료, 2칸 실패 |
| `zql0z8hc4k8dlx` | `7a0f712` | `7a0f712` | `sha256:c904dddc...` | no_result (exit 1 laundered) |
| `x2i12l0tyqzf2a` | `88007ee` | `7a0f712` | `sha256:c904dddc...` | no_result, SIGSEGV(139) 실측 |

**결정적 사실: "완료된" 비교 파드 `xzrx2gnudntf09`는 `db8337a` 이미지로 돌았다 —
`7a0f712`(step_context 배선)보다 다섯 커밋 전이다.** 그 파드의 `infonce_backward`
실패 메시지를 직접 읽으면:
```
"expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16"
```
이것은 §1이 이미 확정한 **구판(자동캐스트 없음) 실패**다 — layer 0 `q_proj`에서
fp32 활성이 bf16 가중치를 만나 죽는 그 경로다. 즉 **"같은 배선이 완료됐다"는 브리프의
원래 전제가 틀렸다**: `xzrx2gnudntf09`는 `_infonce_backward`를 `axes.step_context`
없이 직접 불렀던 구판 코드로 돌았고, 그 구판은 `Qwen3_5GatedDeltaNet.forward`(layer
1의 `linear_attention` 블록)에 도달하기 훨씬 전, layer 0의 평범한 `nn.Linear`에서
이미 죽는다. **fla/causal_conv1d 커널이 forward에서 실제로 호출된 적은 이 저장소
역사상 한 번도 없다** — 구판은 거기 도달하지 못했고, 신판(7a0f712, autocast로
layer 0을 통과시킴)은 두 번 다(`zql0z8hc4k8dlx`, `x2i12l0tyqzf2a`) 도달하자마자
SIGSEGV로 죽었다.

`envs/axolotl/uv.lock`/`pyproject.toml`은 `db8337a`→`7a0f712` 사이에 **변경 없음**
(`git diff --stat db8337a 7a0f712 -- envs/axolotl/uv.lock envs/axolotl/pyproject.toml`
→ 출력 없음, 이번 세션 직접 확인) — 두 이미지의 `fla-core`/`flash-linear-attention`
==0.4.1, torch 2.12.1+cu130 pin은 동일하다. `trainbench/probe/axolotl.py`의 유일한
diff는 §1의 `_infonce_backward` 배선(`git diff db8337a 7a0f712 -- trainbench/probe/axolotl.py`,
11+/4-, 이번 세션 직접 확인) — `load()`/`verify_axes`/모델 구성 경로는 바이트
단위로 동일하다.

**재평가된 순위:**
1. (유력) autocast 리전 안, `Qwen3_5GatedDeltaNet.forward`가 `causal_conv1d_fn`/
   `chunk_gated_delta_rule`/`FusedRMSNormGated`(fla==0.4.1 컴파일 커널, 자동캐스트가
   손대지 않는 외부 호출)를 **이 저장소 역사상 처음으로 실제 호출**하는 지점에서
   SIGSEGV. dtype 불일치(§1의 fp32-norm 규칙이 `Qwen3_5GatedDeltaNet.norm`에도
   적용될 가능성)와 fla==0.4.1 고유 pin(다른 다섯 이미지는 0.5.2) 둘 다 후보이고
   **어느 쪽인지는 구분 안 됨** — 파드에 있는 fla/causal_conv1d 소스를 직접 열지
   않고는 이 이상 좁힐 수 없다(이 호스트에 없음, 확인 안 함).
2. 모델 구성 자체(가중치 적재, `FusedRMSNormGated.__init__` 등)가 fla==0.4.1/torch
   2.12.1 조합에서 원래 위험하다는 가설은 **약화됨** — 그 정확히 같은 구성 코드가
   `xzrx2gnudntf09`에서 **성공적으로 완료**됐다(18개 fla kernel_modules 전부 포함,
   `applied.axes` 실측). 같은 패키지 스택에서 구성 자체는 안전하다는 것이 실측으로
   있다.

**다음 판별 실험(제안, 이 노트 담당자는 파드를 띄우지 않음):** 같은 셀을
`kernel=none`으로 강제해(순수 파이썬 `torch_chunk_gated_delta_rule`/
`Qwen3_5RMSNormGated` 폴백 경로로 떨어짐) `7a0f712` 배선으로 한 번 더 돌린다.
성공하면 fla/causal_conv1d 컴파일 커널이 원인이라는 것이 좁혀지고, 여전히 SIGSEGV면
autocast 리전 자체나 다른 무언가가 원인이다.

## 7. `docker/entrypoint.sh` — exit code laundering 확인 및 수정

**측정(실제 `infisical` 0.43.116, `--env=dev`):** `infisical run --env=dev -- <cmd>`는
`<cmd>`가 무엇으로 죽든 — SIGSEGV든 평범한 `sys.exit(42)`든 — infisical 자신의
종료 코드를 **항상 1로 납작하게 만든다.** 진짜 코드는 stderr 텍스트로만 남고, 그
텍스트도 두 갈래다: 정상 종료한 자식은 `"failed to wait for command termination:
exit status 42"`로 숫자가 남지만, **직접 신호사한 자식은 `"...: signal:
segmentation fault"`로 숫자가 아예 없다.** §6의 두 파드가 `exit status 139`를 남긴
것은 `timeout`을 거쳤기 때문이다 — `timeout`이 자식의 신호사를 자기 자신의 정상
종료 139로 바꿔 주므로 infisical에게는 평범한 비영 종료로 보인다. 이 저장소가
기록한 모든 신호사(死)가 "exit 1"로 적혔던 이유다.

**수정**: `docker/entrypoint.sh::run_with_secrets`가 실제 커맨드를 `infisical run`
안에서 한 겹 더 안쪽의 POSIX `sh -c`로 감싸고, 그 셸 자신의 `$?`(신호사면
128+signal을 그대로 보고 — `sh`/`bash`는 infisical과 달리 이것을 납작하게 만들지
않는다, 실측)를 `${RESULT_DIR}/.last-exit` 파일에 적은 뒤 그 값으로 `exit`한다.
`run_with_secrets`는 그 파일이 있으면 자신의 리턴값으로 그 값을 쓰고, 없으면(예:
infisical 자체가 커맨드 실행 전에 인증 실패 등으로 죽은 경우) infisical이 보고한
값으로 폴백한다. `publish_result.py`/`entrypoint.sh`의 기존 fallback 메시지
(`"no result file after the run (exit ${run_status})"`)는 손대지 않았다 —
`run_status=$?`가 이제 올바른 값을 담으므로 메시지가 자동으로 정확해진다.

**이 버그가 살아남은 이유는 테스트 하네스 쪽에 둘 있었고, 둘 다 고쳤다.**
`tests/test_pods.py::stub_bin`의 `infisical` 스텁이 `exec "$@"`였다 — 실 코드를
그대로 통과시키므로 파드가 결코 볼 수 없는 종료 코드를 20여 개 테스트가 읽고
있었다. 그리고 `run_entrypoint`가 `INFISICAL_TOKEN`을 아예 설정하지 않아
`run_with_secrets`의 **파드가 절대 밟지 않는 무시크릿 갈래**만 실행됐다
(`orchestrate.pod_env`는 `{PATH, HOME, INFISICAL_TOKEN}`을 넘긴다). 스텁을 실측
모양(비영 종료 → 자기 자신의 1, 진짜 코드는 stderr 텍스트로만)으로 바꾸고 토큰을
넣었다. 기존 197개는 그대로 통과했으므로, 새는 코드에 기대고 있던 단정은 없었다.

검증은 실제 `infisical` 바이너리(위 측정)와 그 스텁 둘 다로 했다. 체크는
`tests/test_pods.py`에 있다 — 별도 파일을 두지 않았다. 스텁 자체가 실 바이너리처럼
평탄화하는지 재는 체크가 하나 있고(그 스텁이 모든 다른 체크의 전제이므로),
`run_with_secrets` 단위 다섯(SIGSEGV 139, `sys.exit(42)` 42, 성공 0, 앞 setting의
캡처 파일이 다음 것으로 새지 않음, `INFISICAL_TOKEN` 미설정 갈래는 원래도 정상),
그리고 sweep 전체를 통과해 결과 저장소에 올라간 레코드의 사유 문자열이
`exit 139`인지 재는 종단 하나.

변이 증거(이 세션 실측, 각각 복원 후 바이트 동일):

| 변이 | 결과 |
|---|---|
| inner `sh -c` 래퍼 제거(구판 복귀) | 4 failed, 3 passed |
| 폴백을 `return 0`으로 (거짓 성공) | 2 failed, 5 passed |
| `rm -f` 한 줄 제거 (setting 간 누수) | 1 failed, 6 passed |
| 스텁을 `exec "$@"`로 되돌림 | 2 failed, 5 passed |
| `except Exception` → `BaseException` | 1 failed (`SystemExit` 삼킴) |

`str(exc)` 절단 상수만 변이 생존이다 — 길이 상한을 단정하는 체크는 이 저장소에
없고(다른 두 곳도 없다) 새로 만들지 않았다.
