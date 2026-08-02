# lane-c — capture 구조

## Scope

`trainbench/applied.py`는 "요청한 축이 실제로 걸렸는가"를 되읽는 유일한 자리다. 그런데 네 축은
**config 값과 절대 같아질 수 없는 값**을 돌려준다. 축을 구현해도 `assert_matches`가 런을
거부하므로, **capture가 축 구현의 선행 조건이다.**

| 축 | 지금 돌려주는 값 | config Literal | 크기 |
|---|---|---|---|
| `optim=adamw_8bit` | `adamw8bit` (`:265-266`, `kind.lower()`는 밑줄을 못 만든다) | `adamw_8bit` | ~5줄 |
| `parallel=zero2/zero3` | `deepspeed` (`:724-728`, 엔진 클래스 하나가 두 stage를 다 나타낸다) | `zero2`/`zero3` | stage 판별 |
| `precision=mxfp8/nvfp4` | **영원히 `None`** (`:773`, `:819-825`) | `mxfp8`/`nvfp4` | **구조 변경** |
| `train.offload=optimizer/param/both` | `none` / `offloaded(<dev>)` (`:914-915`) | 4값 | **구조 변경** |

뒤 둘의 docstring이 스스로 인정한다 — precision은 "bf16을 fp8 런의 가중치에서 읽어 인증하느니
모른다고 하겠다"(설계상 옳다), offload는 "deepspeed 자기 config에 있고 여기서 못 읽는다".
그러므로 `step_context`에 recipe를 구현하는 것만으로는 부족하고, **dtype이 아니라 recipe를,
optimizer가 아니라 engine config를 읽는 경로**가 함께 필요하다. `Built`에 필드가 붙는
구조 변경이다.

## Owns

- `trainbench/applied.py`
- `tests/test_applied.py`

`tests/test_applied.py`가 이 레인에 들어온 이유는 실측이다 — 경계 에이전트가 프로토타입으로
전체 스위트를 돌려 `test_applied_state_serialises_for_the_record`가 축별 dict를 **정확히**
못박고 있어 `owner`/`state` 추가로 깨지는 것을 확인했다. 명세 초안에 소유자가 없었다.

## 할 일

### 1. 네 축이 config 값과 같아질 수 있게 한다

`_capture_optim`의 명시적 매핑, `zero2/zero3`의 stage 판별, precision recipe 경로,
offload의 engine config 읽기.

**주의**: 이 모듈의 원칙은 "요청을 되읽는다"이지 "요청을 반복한다"가 아니다. 매핑을 넣되
요청값을 그대로 돌려주는 지름길이 되지 않게 한다 — 그러면 capture가 아무것도 검증하지 않는다.
의도적 비매칭 값(`mixed(...)`, `partial(...)`, `adamw_unfused`, `unwrapped(...)`, `qlora(...)`)은
설계대로이므로 건드리지 않는다.

### 2. 축 소유권 상태를 신설한다

**결정 5**: 프레임워크의 학습 스텝을 그대로 잰다. tevatron `DenseModel.forward`
(`encoder.py:52-87`)가 인코딩·풀링·정규화·스코어링·InfoNCE·분산 게더를 전부 자기가 하므로,
그 셀에서 `loss`와 `parallel.cross_device_negatives`는 **우리 것이 아니다.**

지금 축 상태는 "미구현"과 "적용됨" 둘뿐이라, 그 셀에서 `loss=mnrl`을 요청하면 우리 손실이
안 걸린 것을 불일치로 읽고 런을 거부한다.

**필요한 것**: "이 축은 이 프레임워크가 소유한다"를 표현하는 세 번째 상태. 그것이 결과
레코드에도 실려야 하고, `report.py`가 그 셀에 그 축이 없음을 표에 드러낼 수 있어야 한다.
ablation 그리드가 프레임워크마다 들쭉날쭉해지는 것이 결정 5의 대가이고, 리포트가 그것을
숨기면 안 된다.

## Completion criteria

- 네 축이 config 값과 같아질 수 있다
  → `uv run pytest tests/test_applied.py -k capture_matches`
- 축 소유권 상태가 표현된다 — 미구현 / 적용됨 / **프레임워크 소유** 셋, 그리고 각각이 결과
  레코드에서 구분된다
  → `uv run pytest tests/contract/test_applied_axes.py`
- 계약이 아무것도 미루지 않는다 — 경계가 건 `xfail` 마커 21개가 전부 사라진다. 마커가
  남아 있으면 계약 테스트가 초록이어도 **아무것도 증명하지 않는다**
  → `infisical run --env=dev -- uv run pytest tests/contract/test_applied_axes.py --runxfail -k the_contract_defers_nothing`
  (오늘 exit 1, 21개를 이름으로 열거. `-k`가 필수다 — 파일 전체에 `--runxfail`을 걸면
  이미 아는 실패 21개를 벗겨낼 뿐 "마커가 없다"는 판정하지 못한다)
- 매핑을 "요청을 그대로 돌려주기"로 바꾸면 테스트가 죽는다 (capture가 공허해지는 것을 막는
  검사가 있어야 한다)
  → 변이 출력 그대로 인용
- 각 capture 경로를 되돌리면 그 축의 테스트가 죽는다
  → 변이 출력 그대로 인용
- **확인 안 함**: 실제 bitsandbytes/deepspeed/TE 객체에서 이 경로가 무엇을 읽는지. 이 호스트에
  셋 다 없다. 파드가 답해야 할 것을 명시한다

## Out of scope

- 축 자체의 구현 — `trainbench/axes.py`는 **lane-h** 소유. 이 레인은 "구현되면 읽을 수 있게"
  까지만 한다
- `Built`가 어디서 만들어지는지 — `axes.assemble`은 lane-h 소유. 필드 추가는 경계
  `applied-axes`에서 맞춘다
