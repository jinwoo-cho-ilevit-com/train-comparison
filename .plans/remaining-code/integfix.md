# integfix — 파티션을 가로지르는 잔여 (단독)

> 먼저 읽는다: `HAZARDS.md`, `.plans/review/findings.md`, `.plans/notes/*.md`.
> **혼자 돈다.** 여러 파티션의 소유 경계를 가로지르거나 파티션들이 서로를 깨뜨린 것만 모았다.

## 진입 상태 — 트리가 빨갛다

수정 패스 6개를 머지한 직후이고 **의도적으로 빨간 채로 넘긴다.** 세 실패는 전부
**파티션 상호작용**이지 어느 레인의 결함이 아니다. 그것을 닫는 것이 이 레인의 첫 일이다.

```text
infisical run --env=dev -- uv run pytest -q
  3 failed, 1180 passed, 14 warnings

FAILED tests/test_axes.py::test_an_fp8_recipe_wraps_the_step_and_is_what_the_run_is_read_by[mxfp8-MXFP8BlockScaling]
FAILED tests/test_axes.py::test_an_fp8_recipe_wraps_the_step_and_is_what_the_run_is_read_by[nvfp4-NVFP4BlockScaling]
FAILED tests/test_smoke_cpu.py::test_an_adapter_refusal_is_filed_as_a_result_instead_of_escaping_main

infisical run --env=dev -- uv run pytest tests/contract -q   ->  122 passed
infisical run --env=dev -- uv run python scripts/audit_plan.py
  ->  13/15 passing, 0 new failure(s), 0 newly fixed, 0 grew, 0 shrank, 0 unreadable
uv run ruff check / format --check   ->  초록, 108 files
```

## 작업 1 — 이름이 갈린 자리 (BLOCKING)

`tests/test_smoke_cpu.py:2024` 가 `loader._refuse_a_build_the_fingerprint_condemns` 를
monkeypatch 하는데 그 이름은 없다. `loader-probe` 파티션이 같은 발견을 고치면서
`trainbench/loader.py:320` 의 **공개** `refuse_a_build_the_fingerprint_condemns` 로 개명했다.

두 파티션이 `adapter-refusals-escape-the-refusing-block` 을 양쪽 끝에서 고쳤고 이름이 갈렸다.
**공개 이름 하나로 맞춘다.** 테스트가 진짜로 무엇을 주장하려던 것인지 읽고 고친다 —
이름만 바꿔 초록으로 만들면 그 테스트가 무엇을 보는지 아무도 다시 확인하지 않는다.

## 작업 2 — 테스트가 결함을 못박고 있었다 (BLOCKING)

`tests/test_axes.py:4654` `test_an_fp8_recipe_wraps_the_step_and_is_what_the_run_is_read_by`.

그 docstring 은 이렇게 적는다:
> These recipes keep **bf16 parameters** and cast inside the step, so the weights
> cannot say which one ran.

그런데 테스트가 쓰는 모델은 `plain_model()` — **fp32 이고 TE 모듈이 0개다.**
`capture` 파티션이 고친 것이 정확히 이것이다: recipe 가 있으면 dtype 거부 넷이 통째로
건너뛰어져 **fp32 순수 torch 모델이 mxfp8 로 인증**됐다. 이제 올바르게 undetermined 가 된다.

즉 **이 테스트는 결함을 단언하고 있었다.** 고칠 것은 capture 가 아니라 이 테스트다.
docstring 이 서술하는 체제(bf16 파라미터 + recipe 가 실제로 감싼 스텝)를 **모델이 실제로
만족하게** 만들어 테스트가 진짜를 단언하게 한다.

`.plans/notes/capture.md` 와 `.plans/review/findings.md` 의
`precision-recipe-preempts-dtype-refusals` 항목이 근거다.

## 작업 3 — findings.md 의 남은 확정 발견 여섯

파티션이 소유하지 않아 남은 것들이다.

| key | 위치 |
|---|---|
| `report-reimplements-the-library-training-verdict` | `scripts/report.py:384` |
| `report-verdict-third-copy-uncompared` | `scripts/report.py:384` |
| `baseline-tolerance-declared-but-report-hardcodes-3pct` | `scripts/report.py:99` |
| `probe-preflight-renders-a-deliberate-refusal-as-a-lost-pod-hour` | `docker/entrypoint.sh:225` |
| `axis-values-fla-has-no-companion-so-it-can-never-be-counted` | `scripts/audit_plan.py:1603` |
| `stale-repo-line-citations` | `docs/methodology.md:514` |

`training_verdict` 는 구현이 **세 벌**이다 — `trainbench/metrics/validity.py`,
`scripts/report.py`, `tests/contract/test_record_report.py`. 라이브러리 하나로 모으고,
계약 파일의 사본은 계약이 그 자리에 있어야 할 이유가 있는지 읽고 판단한다.
**계약 파일을 고칠 때는 단언을 약화하지 않는다.**

## 작업 4 — 문서가 코드보다 앞서 나간 자리

`docs/methodology.md §11`(:618, :622-627)이 커널 거부 셋이 문을 닫고 런을 세운다고
**현재형으로** 단언한다. 수정 패스 전에는 프로덕션 호출자가 0건이었다.
`bench-consumers` 파티션이 호출자를 붙였으므로 **지금은 참일 수 있다** —
`git grep` 으로 확인하고, 참이면 그대로 두고 참이 아닌 부분만 고친다.
**확인하지 않고 고치지도 않고 두는 것이 가장 나쁘다.**

`docs/audit-baseline.json` 의 `axis-values` note 는 `precision 1/3` 을 **"하드웨어다"**
(CC 10.x vs A100)로 단정하는데, 그 수를 만든 실행은 하드웨어 검사에 도달한 적이 없다 —
transformer-engine 이 이 호스트에 없어 import 단계에서 거부된다. `baseline-note-attributes-
precision-to-hardware-that-was-never-reached` 가 그것이다. **note 를 사실에 맞게 다시 쓴다.**
하드웨어 사실 자체는 리서치가 핀에서 확정했으므로 지우지 말고, **이 호스트의 이 수가 그것
때문이라는 인과를 주장하지 않는다.**

> `docs/audit-baseline.json` 은 평소 통합자 전용이나 **이 레인에는 그 항목 하나에 대해
> 권한을 준다.** count 는 건드리지 않는다 — note 산문만이다.

## 작업 5 — 파티션들이 `forSerialPass` 로 넘긴 것

각 파티션의 보고가 `.plans/notes/` 에 있다. 특히:

- `kernels.KernelProvenanceError` 가 아직 `refusing()` 밖인지 확인한다.
  `loader.AdapterRefusal` 은 `AppliedMismatch` 를 상속하도록 바뀌었다 — 같은 결정을 할지,
  `scripts/bench.py` 의 catch 목록을 넓힐지 정한다
- `parallel=zero2/zero3` 의 진짜 배선(측정 루프가 `engine.backward`/`engine.step` 을 쓰는 것)은
  `scripts/bench.py` 가 필요하다. axes 파티션이 선택지 (b)(단언을 낮추고 docstring 을 사실에
  맞게 고침)를 골랐다. **(a)를 여기서 할지 판단한다.** 하지 않기로 하면 그 이유와 파드가
  답할 질문을 `.plans/notes/integfix.md` 에 적는다
- `.plans/notes/axes.md §2.2` 에 `axes.py` 에서 걷어낸 것과 **같은 위임 단언**이 남아 있다
- `measurement.repeats` 반복 루프는 record-report 계약을 건드린다. 하지 않기로 했다면
  metrics 파티션이 고른 "레코드에 적용된 것처럼 싣지 않는다"가 유지되는지 확인한다

## Owns

```
tests/test_axes.py
tests/test_smoke_cpu.py
scripts/report.py
scripts/audit_plan.py
scripts/bench.py
docker/entrypoint.sh
docs/methodology.md
docs/audit-baseline.json          note 산문만. count 금지
trainbench/loader.py
trainbench/kernels.py
tests/contract/test_record_report.py    training_verdict 사본에 한해
tests/test_report.py
tests/test_pods.py
tests/test_audit.py
```

## 완료 조건

1. **네 게이트 전부 초록** — `pytest` 실패 0, `audit_plan.py` exit 0,
   `ruff check && ruff format --check`, `env_report.py`
2. `pytest tests/contract -q` → 122 passed 유지 (줄어들면 계약을 약화한 것이다)
3. 작업 3의 발견 여섯이 각각 **재현 명령을 다시 돌려 다른 출력**을 낸다.
   `findings.md` 에서 그 명령을 찾아 그대로 돌린다
4. `docs/methodology.md` 와 `docs/audit-baseline.json` 의 주장이 **`git grep` 으로 확인된
   현재 코드**와 맞는다
5. 추가·수정한 검사마다 mutation 증거. **사보타주 전에 `co_filename`/`co_firstlineno` 확인**
6. 고치지 못한 것은 `notFixed` 에 키와 이유. 숨기지 않는다
7. **확인 안 함** — 이 호스트에 CUDA·deepspeed·TE·DALI·fla 가 없다. 파드가 답할 것을
   `.plans/notes/integfix.md` 에 적는다

## 하지 않는 것

- `docs/audit-baseline.json` 의 `count` — 통합자 전용
- 루트 `PLAN.md` 레이아웃, `AGENTS.md`, `README.md`, `docs/support-matrix.md` — 통합자
- `envs/**`, `uv.lock`, `pyproject.toml`
- minor 30건 — 수정 대상이 아니다
