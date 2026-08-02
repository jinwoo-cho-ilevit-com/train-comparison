# report — 리포트·오케스트레이션 위생 (wave 1)

> 먼저 읽는다: `HAZARDS.md`(특히 §3 "검사가 통과하면서 아무것도 보지 않는다"), `PLAN.md`.
> 이 레인의 주제가 정확히 그것이다.

## 목표

결과를 읽고 발행하는 경로에서 **아무것도 보지 않고 통과하는 검사**를 없앤다.

## Owns

```
scripts/report.py
scripts/orchestrate.py
scripts/prepare_data.py
docker/entrypoint.sh
configs/run/
tests/test_report.py
tests/test_pods.py
tests/contract/test_record_report.py    마커 5개 제거만
docs/open-verdicts.json                 항목 1개만
```

`docker/entrypoint.sh` 는 시크릿 주입 경계이고 `orchestrate.py` 는 파드 토큰 스코프 검사를
소유한다. **둘 다 보안 표면이다.** 스코프 검사의 허용목록 성격을 약화하지 않는다 —
`AGENTS.md` 가 그것이 선호가 아니라 측정임을 기록해두었다(deny list 는 27개 중 22개를 통과시켰다).

## 지우는 마커 5개

```
tests/contract/test_record_report.py
  test_an_artifact_without_the_identity_is_refused_rather_than_dated_by_its_file
  test_a_run_that_trained_nothing_is_not_published_as_a_speed_result
  test_oom_is_its_own_result_category
  test_two_stacks_are_never_ranked_in_one_table
  test_an_axis_the_framework_owns_is_visible_in_the_report
```

여섯 번째(`test_the_producer_stamps_the_identity`)는 **split 이 wave 0 에서 지웠다.**

**중요**: `tests/fixtures/run_record.sample.json` 은 이미 완전하다 — `recorded_at`,
`applied.framework_owned`, `metrics.grad_norm`, `metrics.trainable_params`, `build_fingerprint`
를 다 싣고 있다. 다섯 마커는 **`scripts/report.py` 만으로 지울 수 있다.** 다른 레인을 기다리지
않는다. `FRAMEWORK_OWNED` 도 `test_record_report.py:141` 의 지역 리터럴이지
`trainbench.applied` 에서 import 하는 것이 아니다.

## 작업 1 — 캠페인 아티팩트 선별이 mtime 으로 떨어진다

`scripts/report.py:235`:
```python
timestamp=float(payload.get("recorded_at") or path.stat().st_mtime),
```

실측: 결과 저장소의 아티팩트 40개 중 `recorded_at` 을 실은 것이 **0건**. clean clone 은 전부
한꺼번에 내려받으므로 mtime 이 같아지고, 그러면 **18칸 중 8칸이 이전 캠페인 아티팩트를 고른다.**
2차 캠페인 판정 때 원장의 `pod_id` 18개를 손으로 복사해 넘겨서 피했다.

고칠 방향은 둘 중 하나이고 **어느 쪽이든 mtime fallback 에 의존하지 않는 신원**이어야 한다:
(a) 기록된 `recorded_at` 이 없으면 **거부한다**(split 이 생산 쪽을 이미 고쳤다),
(b) 원장(`outputs/orchestrate-*.json`)으로 거른다.
`report.py` 가 원장을 결과 선별에 쓰지 않는 것은 알려진 미해결이다.

## 작업 2 — 두 스택을 한 표에 세우지 않는다 (결정 4)

해석 스택이 칸마다 다르다:

```
native / sentence_transformers / tevatron / axolotl   transformers 5.14.1 + torch 2.13.0
ms_swift                                              transformers 5.12.1 + torch 2.13.0
unsloth                                               transformers 5.5.0  + torch 2.11.0
```

이 분기는 uv 충돌 때문에 **해소 불가**로 문서화돼 있다. `report.render_measurements` 는 지금
프로파일 안 걸린 런을 전부 한 표에 넣어 5.5.0 과 5.14.1 을 나란히 줄 세운다.
스택이 다르면 **행을 나누고 각자의 스택을 함께 렌더한다.**

## 작업 3 — OOM 은 별도 범주

OOM 레코드는 metrics 도 probe 도 없어서 `report.cell`(`report.py:457-491`)이
`launch_state` 로 떨어지고 **시도조차 안 한 조합처럼 렌더된다.**

## 작업 4 — 프레임워크 소유 축이 표에 보인다

`scripts/report.py` 는 `applied` 를 읽지 않는다. 그래서 **자기 프레임워크에 축을 넘겨준 칸이
그 축을 돌린 칸처럼 보인다.** 결정 5 의 대가는 ablation 그리드가 프레임워크마다 들쭉날쭉해지는
것이고, **리포트가 그것을 숨기면 안 된다.**

상태 어휘(`framework_owned`, 밑줄)는 **capture 레인의 것**이다. 이 레인은 쓰기만 한다.
`tests/fixtures/run_record.sample.json` 이 모양을 정한다.

## 작업 5 — sentence_transformers 동결 가드 사각지대

`trainbench/probe/sentence_transformers.py:75-89` 의 자체 `_backward` 가
`steps.infonce_backward` 의 가드를 거치지 않고 `params_with_grad` 만 돌려준다.
`trainable_params` 를 세지도, 0인지 확인하지도 않는다. **unsloth 를 잡은 가드가 여기엔 없다.**
이번엔 310/320/505 였지만 **0이었다면 그대로 초록이다.**

> **이 파일은 probe 레인 소유다.** 여기 적는 이유는 결함의 출처가 리포트 신뢰성이기 때문이다.
> 이 레인은 손대지 않는다. probe 레인 브리프에 들어 있다.

## 작업 6 — `axes_verified` 가 `all_matched:false` 에 통과한다

실측 불일치 둘: `kernel.name` 이 `none` 요청인데 **qwen3_5 칸 전부에서 `fla` 적용**,
`precision.name` 이 `bf16` 요청인데 axolotl 3칸 + unsloth 3칸에서 `mixed(bf16,fp32)`.

> 이것도 **probe 레인**(`trainbench/probe/steps.py`)이다. 여기 적는 것은 리포트가 그 결과를
> 어떻게 읽는지 때문이다 — 초록인 칸이 초록이 아닐 수 있다는 것이 `docs/support-matrix.md` 의
> "초록이지만 믿으면 안 되는 것" 절이다.

## 작업 7 — `check_axis_not_split` 이 `framework.name` 을 보지 못한다

`scripts/orchestrate.py:272-284` 의 `axes_touched` 는 `exp.overrides` 와 `exp.settings` 만 읽는데,
`framework`/`model` 은 **매니페스트 최상위 필드**이고 `plan_runs`(`:413`) 안에서 덮어써진다.
그래서 `framework.name` 축이 **구조적으로 보이지 않는다.**
Phase 0 의 18파드가 그 축을 모델당 여섯 갈래로 쪼갰는데 가드는 침묵했다.

설계 결정이 필요하다: 명시적 예외("다른 이미지는 한 파드를 공유할 수 없다")로 둘 것인가,
규칙을 코드에 넣을 것인가. **어느 쪽이든 침묵하지 않아야 한다.**
고른 쪽과 그 이유를 `.plans/notes/report.md` 에 적는다.

## 작업 8 — probe 분기도 preflight 를 탄다

`scripts/bench.py` 의 `preflight` 는 존재하지만 `docker/entrypoint.sh` 가
`timing`/`profile`/`quality` 분기에서만 부른다. `purpose == "probe"` 는 곧바로
`verify_env.py` 로 간다. **크래시루프를 낸 카나리가 probe 파드였다** — 이미지-스키마 불일치로
17초 간격 40회 재시작, A100 12분 과금.

이것이 열린 판정 `images-carry-a-code-snapshot-nothing-checks-is-current` 의 앵커를 만든다.
앵커 테스트 이름은 원장에 적혀 있다:
`tests/test_pods.py::test_a_pod_whose_image_predates_the_config_it_is_handed_does_not_measure`.
**그 이름으로 만든다.** 앵커가 생긴 뒤에만 `docs/open-verdicts.json` 의 그 항목을 닫는다.
그 전에 닫으면 원장이 "검사하지 않는 검사"가 되고, 실제로 그렇게 닫으려다 게이트에 잡힌 선례가
있다(`1a7b7c7`).

## 작업 9 — 위생

- `config-consumed` 오탐 2건: `scripts/prepare_data.py` 가 `data = config.data` 별칭으로 읽어서
  탐지기가 못 본다(`audit_plan.py:103` 이 이 한계를 문서화해 두었다). 호출부 4줄.
  **수가 실제 구멍보다 2 크면 진짜 미소비 knob 이 목록에 묻힌다**
- trackio 제거(결정 3): **네 몫은 `configs/run/*.yaml` 4개뿐이다.**
  스키마 쪽은 **measure 레인**이 뺀다 — **경계에서 맞춘다**, 한쪽만 빠지면 config 합성이 깨진다.
  루트 `pyproject.toml` 의 `tracking` extra 는 **통합자 전용**이다(`uv.lock` 재해석을
  `env-locks`/`doc-commands` 가 검사하고, 이 호스트에서 재해석할 수 없는 lock 이 여섯 중
  다섯이다). **`.plans/notes/report.md` 에 "루트 pyproject.toml 의 tracking extra 제거"라고
  적어 넘긴다.** 직접 지우지 않는다
- `gradcache` 죽은 핀: `envs/native/pyproject.toml` 이 고정하는데 import 0건이고
  `axes._loss` 의 `gradcache_backward` 는 손으로 짠 것이다. `optim=muon` 이 세운 원칙
  ("라이브러리를 쓰고 손으로 짜지 않는다")과 어긋난다.
  **`envs/**` 는 통합자 전용이다. 판단만 `.plans/notes/report.md` 에 적고 손대지 않는다**

## 완료 조건

1. `pytest tests/contract/test_record_report.py -q` → 마커 5개 제거 후 **17 passed, 0 xfailed**
2. 캠페인 아티팩트를 기록된 신원으로 고른다 → `uv run pytest tests/test_report.py -k campaign`
3. **모든 아티팩트의 타임스탬프가 같아도** 이번 캠페인 것만 고른다 (그것이 결함의 측정된 재현
   조건이다) → `uv run pytest tests/test_report.py -k equal_timestamps`
4. 스택이 다른 칸을 한 순위표에 넣지 않는다 → `uv run pytest tests/test_report.py -k stack`
5. `check_axis_not_split` 이 `framework.name` 에 침묵하지 않는다 →
   `uv run pytest tests/test_pods.py -k axis_split`
6. `entrypoint.sh` 의 probe 분기도 스키마 검증을 탄다 →
   `uv run pytest tests/test_pods.py -k preflight_probe`
   (앵커 이름은 위 작업 8 의 것을 쓴다)
7. `config-consumed` 가 0 →
   `infisical run --env=dev -- uv run python scripts/audit_plan.py`
   **주의**: 0 이 되면 `newly fixed` 로 BLOCK 된다. 그것은 정상이고 머지 단계가 baseline 을
   고친다. **`docs/audit-baseline.json` 을 건드리지 않는다.** 감사 마지막 줄이 무엇을 이름으로
   지목하는지만 보고한다
8. 각 검사를 되돌리면 죽는다. mutation 출력 인용. 사보타주 전에 `co_filename` 확인
9. **[human]** `gradcache` 핀 처리에 대한 판단을 `.plans/notes/report.md` 에 적었다

## 하지 않는 것

- `trainbench/probe/sentence_transformers.py`, `trainbench/probe/steps.py` — **probe** 레인
- `trainbench/record.py`, `trainbench/metrics/` — split(끝남) / **measure** 레인
- `docs/support-matrix.md`, `AGENTS.md` — **integrate** 레인
- `envs/**`, `docs/audit-baseline.json`, `scripts/audit_plan.py`, 루트 `uv.lock` — **통합자 전용**
- `docs/open-verdicts.json` 의 다른 세 항목 — 하나(`images-carry-a-code-snapshot…`)만 네 것이다
