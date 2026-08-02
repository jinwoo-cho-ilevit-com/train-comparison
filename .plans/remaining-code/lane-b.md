# lane-b — 가드·위생

## Scope

이 저장소가 반복해온 실패 형태는 **검사가 통과하면서 아무것도 보지 않는 것**이다. 오늘만
세 건이 더 나왔고, 그중 둘은 실제로 결과를 오염시킬 수 있었다.

## Owns

- `scripts/report.py`
- `scripts/orchestrate.py`
- `scripts/prepare_data.py`
- `docker/entrypoint.sh`
- `trainbench/probe/sentence_transformers.py`
- `configs/run/`
- `pyproject.toml`

`security: true` — `entrypoint.sh`가 시크릿 주입 경계이고, `orchestrate.py`가 파드 토큰 스코프
검사를 소유한다.

## 할 일

### 1. `report.py`가 캠페인을 mtime으로 고른다

`scripts/report.py:235`:

```text
timestamp=float(payload.get("recorded_at") or path.stat().st_mtime)
```

결과 저장소 40건 중 `recorded_at`을 실은 것이 **0건**이다. 실측: 모든 아티팩트 timestamp를
같게 두면(= 깨끗한 클론에서 한 번에 내려받은 상태) **18칸 중 8칸이 지난 캠페인 것을 고른다.**
문서화된 명령을 그대로 돌리면 다음 캠페인에서 섞인 매트릭스가 나온다.

프로브가 `recorded_at`을 쓰게 하거나 `report.py`가 원장으로 거르게 한다. 둘 중 어느 쪽이든
**mtime 폴백에 의존하지 않는 신원**이어야 한다.

### 2. `report.py`가 스택이 다른 셀을 한 순위표에 넣는다

여섯 이미지의 스택이 갈린다(통일 불가, `docs/support-matrix.md`에 uv 충돌 원문):

```
native, sentence_transformers, tevatron, axolotl   transformers 5.14.1, torch 2.13.0
ms_swift                                           transformers 5.12.1, torch 2.13.0
unsloth                                            transformers 5.5.0,  torch 2.11.0
```

**결정 4**: 같은 스택끼리만 나란히 줄 세운다. 스택이 다른 셀을 한 순위표에 넣는 것을 거부하고,
다른 스택은 자기 스택과 함께 별도 행으로 낸다.

### 3. `sentence_transformers` 프로브에 동결 가드가 없다

`trainbench/probe/sentence_transformers.py:75-89`의 자체 `_backward`가
`steps.infonce_backward`(가드는 `steps.py:421`, `:428`)를 거치지 않고 `params_with_grad`만
반환한다. `trainable_params`를 세지도, 0인지 확인하지도 않는다.

unsloth를 잡은 가드의 사각지대다. 이번 캠페인 값(310/320/505)은 0이 아니었지만, **0이었다면
그대로 초록으로 지나갔다.**

### 4. `axes_verified`가 `all_matched:false`에도 통과한다

실측 불일치 둘:
- `kernel.name`: `none` 요청인데 `qwen3_5_0_8b` 칸 **전부**에서 `fla` 적용
- `precision.name`: `bf16` 요청인데 axolotl 3칸 + unsloth 3칸에서 `mixed(bf16,fp32)`

Phase 0 판정은 안 바뀌지만 뒤 단계 교란 요인이 이미 보인다.

### 5. `check_axis_not_split`이 `framework.name`을 못 본다

`orchestrate.py:272-284`의 `axes_touched`는 `exp.overrides`와 `exp.settings`만 읽는데,
`framework`/`model`은 **매니페스트 최상위 필드**이고 오버라이드로 바뀌는 것은 `plan_runs`
안(`:413`)이다. 그래서 `framework.name` 축은 **원리적으로 보이지 않는다.** Phase 0의 18파드가
이 축을 모델당 6갈래로 쪼갰고 가드는 침묵했다.

**설계 결정**: 명시적 예외로 둘 것인가("이미지가 다르면 한 파드에 담을 수 없다"), 아니면 그
규칙 자체를 코드로 표현할 것인가. 어느 쪽이든 **침묵하지 않아야** 한다.

### 6. `entrypoint.sh`의 probe 갈래에 preflight가 없다

`scripts/bench.py:1123 preflight`는 있는데 `entrypoint.sh`가 `timing/profile/quality`
갈래에서만 부른다. `purpose == "probe"` 갈래는 `verify_env.py`를 바로 실행한다.
**크래시루프를 낸 카나리가 probe 파드였다.**

### 7. 위생

- `config-consumed` 오탐 2건 — `prepare_data.py`가 `data = config.data` 별칭으로 읽어
  검출기가 못 본다(`audit_plan.py:103`이 이 한계를 명시). 호출부 4줄. count가 진짜 구멍보다
  2 크면 새 미소비 knob이 목록에 묻힌다
- **trackio 제거** (결정 3) — `configs/run/*.yaml` 4개, `pyproject.toml`의 `tracking` extra.
  스키마 쪽은 lane-d 소유이므로 **경계에서 맞춘다**
- `gradcache` 죽은 핀 — `envs/native/pyproject.toml`이 핀하는데 import 0건.
  `axes._loss`의 `gradcache_backward`는 직접 구현이다. **결정 필요**: 라이브러리로 갈아탈
  것인가 핀을 지울 것인가. `optim=muon`이 세운 원칙("직접 구현하지 않고 라이브러리를 쓴다")과
  정반대 상태다. 핀은 `envs/` 소유가 아니므로 **판단만 보고하고 실행은 하지 않는다**

## Completion criteria

- `report.py`가 캠페인 아티팩트를 mtime이 아니라 기록된 신원으로 고른다
  → `uv run pytest tests/test_report.py -k campaign`
- 아티팩트 timestamp를 전부 같게 둔 상태에서도 이번 캠페인 것만 고른다 (그것이 이 결함의 실측
  재현 조건이다)
  → `uv run pytest tests/test_report.py -k equal_timestamps`
- `report.py`가 스택이 다른 셀을 한 순위표에 넣지 않는다
  → `uv run pytest tests/test_report.py -k stack`
- `sentence_transformers` 프로브가 동결 그래프를 통과시키지 않는다. 테스트는 **실제 모양**으로
  돈다 — 파라미터를 전부 얼리고 임베딩 출력에 훅을 걸어 backward가 유한한 손실을 내는 그래프
  → `uv run pytest tests/test_probe.py -k sentence_transformers_frozen`
- `axes_verified`가 `all_matched:false`를 통과시키지 않는다
  → `uv run pytest tests/test_probe.py -k axes_verified`
- `check_axis_not_split`이 `framework.name`에 대해 침묵하지 않는다
  → `uv run pytest tests/test_orchestrate.py -k axis_split`
- `entrypoint.sh`의 probe 갈래도 스키마 검증을 거친다
  → `uv run pytest tests/test_pods.py -k preflight_probe`
- `config-consumed`가 0
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- 위 검사 각각을 되돌리면 죽는다 (부수기 출력을 그대로 보고한다)
- [human] `gradcache` 핀 처리 방향에 대한 판단을 보고했다
  → verdict: ____  by: ____  at: ____

## Out of scope

- `trainbench/config_schema.py` — lane-d 소유. trackio 스키마 제거는 경계에서 맞춘다
- `envs/*/pyproject.toml`, `envs/*/uv.lock` — 어느 레인도 소유하지 않는다. `gradcache` 핀은
  판단만 보고한다
- `docs/open-verdicts.json` — lane-i가 마지막에 모은다
