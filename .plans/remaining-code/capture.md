# capture — 되읽기 구조 (wave 1)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`.
> 이 레인이 xfail 38개 중 **30개**를 지운다. 캠페인에서 가장 큰 단일 산출이고,
> wave 2 의 axes 와 adapters 가 이것을 기다린다.

## 목표

`trainbench/applied.py` 가 (a) config 값과 같아질 수 없는 네 축을 되읽을 수 있게 하고,
(b) **세 번째 축 상태 `framework_owned`** 를 도입한다.

## Owns

```
trainbench/applied.py
tests/test_applied.py
tests/contract/test_applied_axes.py     마커 제거만. 단언을 약화하지 않는다.
```

`tests/test_applied.py` 를 함께 갖는 이유: `test_applied_state_serialises_for_the_record` 가
축별 dict 를 **정확 일치**로 못박고 있어 `owner`/`state` 를 더하면 깨진다.

## 계약이 완료를 정의한다

```
infisical run --env=dev -- uv run pytest tests/contract/test_applied_axes.py -q
```
지금 26 passed, 30 xfailed. **끝나면 56 passed, 0 xfailed.**

마지막 마커는 `test_the_contract_defers_nothing` 이다. 그것은 이 파일을 AST 로 스캔해
살아남은 `pytest.mark.xfail` 을 이름으로 세고, 자기 자신은 세지 않는다. **가장 마지막에 지운다.**
진행 확인:
```
infisical run --env=dev -- uv run pytest tests/contract/test_applied_axes.py \
  --runxfail -k the_contract_defers_nothing
```
`-k` 는 필수다. `--runxfail` 을 파일 전체에 걸면 이미 알려진 실패 30개를 벗겨낼 뿐
"마커가 없다"를 판정하지 못한다.

## 작업 1 — 네 축이 config 값과 같아질 수 있게

| 축 | 지금 돌려주는 값 | config Literal | 지점 |
|---|---|---|---|
| `optim=adamw_8bit` | `adamw8bit` — `kind.lower()` 는 밑줄을 만들 수 없다 | `adamw_8bit` | `applied.py:266` |
| `parallel=zero2/zero3` | `deepspeed` — 엔진 클래스 하나가 두 stage 를 대표한다 | `zero2`/`zero3` | `applied.py:727`, `:731` |
| `precision=mxfp8/nvfp4` | **항상 `None`** | `mxfp8`/`nvfp4` | `applied.py:786` |
| `train.offload=optimizer/param/both` | `none` / `offloaded(<dev>)` | 4값 | `applied.py:846` |

- `optim` — `_capture_optim` 에 명시적 매핑 표(`OPTIM_CLASS_AXIS`)
- `parallel` — 엔진에서 **stage 를 분별**한다. config 가 아니라 엔진에서 읽는다
- `precision` — **recipe 읽기 경로**. dtype 이 아니다. `Built` 에 `precision_recipe` 필드가 는다.
  지금 docstring 이 스스로 인정한다: fp8 런의 가중치는 bf16 이라 거기서 인증하면 틀린다
- `offload` — **엔진 config 읽기 경로**. optimizer 가 아니다. `applied.py:878` 이
  "deepspeed 가 자기 config 안에서 정하고 여기서는 읽을 수 없다"고 적어두었다

**이 모듈의 원칙은 "요청을 되읽는다"이지 "요청을 반복한다"가 아니다.**
매핑이 요청을 그대로 돌려주는 것으로 퇴화하면 capture 는 아무것도 검증하지 않는다.
`test_the_optimizer_class_table_only_maps_to_values_the_config_offers` 와
`test_the_precision_recipe_table_only_maps_to_values_the_config_offers` 가 그것을 잡는다.

**건드리지 않는 값** — 일부러 일치하지 않게 만든 것들이다:
`mixed(...)`, `partial(...)`, `adamw_unfused`, `unwrapped(...)`, `qlora(...)`.

## 작업 2 — 세 번째 상태 `framework_owned`

결정 5(프레임워크의 학습 스텝을 그대로 잰다)의 대가다. tevatron 의 `DenseModel.forward` 는
인코딩·풀링·정규화·스코어링·InfoNCE·분산 게더를 전부 자기가 한다. 그 칸에서 `loss` 와
`parallel.cross_device_negatives` 는 **우리 것이 아니다.** 지금은 상태가 둘뿐이라
`loss=mnrl` 요청이 mismatch 로 읽히고 런이 거부된다.

필요한 것:
- `AXIS_STATES` — 세 이름. 철자는 **`framework_owned`** (밑줄). `record-report` 가 한때
  `framework-owned`(하이픈)로 적어 두 계약이 갈라졌고 `f102cd2`/`5971874`/`e5926bc` 가 그것을
  수습했다. 파이썬 식별자에 하이픈이 못 들어가는 것이 어휘 소유가 `applied-axes` 에 있는 이유다
- `FRAMEWORK_OWNABLE` — 소유를 주장할 수 있는 축의 **선언된 집합**.
  `framework.name` 은 여기 넣지 않는다: 자기 자신을 부인할 수 있는 어댑터는 그 아래 전부를
  부인할 수 있다
- `AxisState.owner` / `AxisState.state`
- `Built.owned_axes` — **소유권은 config 가 아니라 `Built` 에서 온다.**
  `framework=tevatron` 요청만으로 축이 남의 것이 되면 모든 셀이 가장 안 걸릴 축부터 스스로
  면제한다
- `AppliedState.framework_owned()` 와 레코드의 `owner`/`state`/`framework_owned` 키

레코드에서의 모양은 `tests/fixtures/axis_state.sample.json` 과
`tests/fixtures/run_record.sample.json` 이 못박았다. **샘플이 정답이다** — 소유 축 엔트리는
`owner` 최상위 + `applied` null + `determined` false 다.

## 참고할 것 — 믿지는 않는다

| ref | base | 내용 |
|---|---|---|
| `preserved/wf_c5aa0913-a6d-4` | **e5926bc (= 캠페인 base)** | `applied.py` +334, `tests/test_applied.py` +235, 마커 제거. **base 가 정확하다** |
| `aborted-wave1-lane-c` | 274fa5f | 같은 설계. 계약 5개를 다 보고 작업했으나 그 뒤 main 이 어휘를 통일했다 |
| `preserved/agent-a95eb5c32258196cc` | 9363197 | `applied.py` + `axes.py` 동시 수정. **`axes.py` 는 남의 것이다** |

`git show preserved/wf_c5aa0913-a6d-4` 로 읽는다. **병합하지 않는다.** 쓸 만한 것을 자기
브랜치에 다시 만들고 완료 조건은 직접 실행해서 확인한다.

## 완료 조건

1. `pytest tests/contract/test_applied_axes.py -q` → **56 passed, 0 xfailed**
2. 네 축이 config 값과 같아질 수 있다 →
   `uv run pytest tests/test_applied.py -k capture_matches`
3. 세 상태가 **레코드만 보고** 구별된다 →
   `pytest tests/contract/test_applied_axes.py -k three_states_are_distinguishable`
4. **매핑을 "요청 되돌려주기"로 바꾸면 테스트가 죽는다** — 공허해지는 것을 막는 검사가 있어야
   한다. mutation 출력 인용
5. 각 capture 경로를 되돌리면 그 축의 테스트가 죽는다. 경로마다 mutation 출력 인용
6. 네 게이트. `pytest` 전체는 `879 → 909 passed, 36 → 6 xfailed` 근처가 되어야 한다
   (정확한 수는 **네가 직접 돌려서** 보고한다. 이 문서의 수를 옮기지 않는다)
7. **확인 안 함**: 이 경로들이 실제 bitsandbytes / deepspeed / Transformer Engine 객체에서
   무엇을 읽는지. 셋 다 이 호스트에 없다. **파드가 답해야 할 것을 축별로 한 문장씩 적는다.**

## 하지 않는 것

- 축 자체의 구현 — `trainbench/axes.py` 는 **axes** 레인 (wave 2). 이 레인은
  "구현되면 읽을 수 있게"까지만 한다
- `Built` 가 어디서 만들어지는지 — `axes.assemble` 은 axes 레인. **필드 추가는 경계
  `applied-axes` 에서 맞춘다**: 새 필드는 기본값을 갖고, `axes.assemble` 의 생성 호출이
  바뀌지 않아도 되게 한다 (지금 `Built` 의 필드가 전부 `None` 기본값을 갖는 것과 같은 방식)
- `report.py` 가 소유 축을 어떻게 보여주는지 — **report** 레인
- 계약 파일의 단언 수정 — 마커 제거만
