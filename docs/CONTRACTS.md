# 공유 계약 (Wave 0 확정)

Wave 1~2의 모든 워크트리 레인이 이 문서를 계약으로 삼는다. **여기 정의된 인터페이스를
레인이 임의로 바꾸지 않는다.** 변경이 필요하면 직접 고치지 말고 계약 변경으로 올린다.

병렬 개발에서 각 레인이 서로 다른 스키마 위에 코드를 쌓으면 병합이 불가능해진다.
Wave 0을 순차 구간으로 둔 이유가 이것이다.

---

## 1. 파일 소유권

레인은 자기 소유 파일만 수정한다. 이 표가 유일한 기준이며, **저장소의 모든 파일이
정확히 한 곳에 속한다.** 표에 없는 파일이 생기면 그것이 계약 변경 대상이다.

| 레인 | 워크트리 | 소유 파일 |
|---|---|---|
| A 데이터 | `wt-data` | `scripts/prepare_data.py`, `configs/data/`, `tests/test_data.py` |
| B 코어정확성 | `wt-core` | `trainbench/embedding.py`, `trainbench/device.py`, `trainbench/seed.py`, `trainbench/probe/` 전체(`types.py` 제외), `scripts/verify_env.py`, `scripts/env_report.py`, `configs/model/`, `tests/test_embedding.py`, `tests/test_device_seed.py`, `tests/test_probe.py` |
| C 오케스트레이션 | `wt-orch` | `trainbench/pods.py`, `scripts/{orchestrate,publish_result,report}.py`, `configs/experiment/`, `configs/run/`, `docker/entrypoint.sh`, `docs/evidence/`, `tests/test_pods.py` |
| D 축구현 | `wt-axes` | `trainbench/axes.py`, `trainbench/applied.py`의 `_CAPTURES`·`_REQUESTED_OVERRIDES`·capture 함수들, `configs/{attn,kernel,precision,compile,optim,freeze,dataloader,parallel,peft,loss}/`, `configs/train/`, `tests/test_axes.py` |
| E 문서 | `wt-docs` | `PLAN.md`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/methodology.md`, `docs/support-matrix.md`, `docs/model-spec.md` |
| F 이미지 | `wt-images` | `envs/*/`(pyproject + lock), `docker/Dockerfile.*`, `.github/workflows/`, `pyproject.toml`, 루트 `uv.lock`, `.pre-commit-config.yaml` |

**공유(수정 금지)**: `trainbench/config_schema.py`, `trainbench/config.py`,
`trainbench/compose.py`, `trainbench/record.py`, `trainbench/probe/types.py`,
`trainbench/applied.py`의 인터페이스(데이터클래스·`capture`·`assert_matches`),
`scripts/audit_plan.py`, `scripts/compose_config.py`, `docs/CONTRACTS.md`,
`docs/model-spec.yaml`, `tests/{conftest,test_config,test_applied,test_audit}.py`,
`.env.example`, `.infisical.json`.

### `docs/audit-baseline.json` — 공유하되 한 줄씩만

이 파일은 6개 레인 중 5개가 반드시 건드린다(§6이 "통과하기 시작해도 차단"이므로
자기 항목을 해소한 레인은 자기 게이트에서 막힌다). 그래서 규칙을 좁게 고정한다.

- 레인은 **자기 항목 한 줄만 삭제**한다. 다른 줄은 읽지도 고치지도 않는다
- 레인에서 **`--update-baseline`을 실행하지 않는다.** 전체 실행이 아니면 도구가
  거부하고, 전체 실행이면 다른 레인의 미완 항목까지 자기 상태로 덮어쓴다
- 항목 추가는 계약 변경이다. 새 실패는 baseline이 아니라 수정으로 해소한다

레인별 담당 항목:

| 항목 | 해소 레인 |
|---|---|
| `data-pinned` | A |
| `evidence-committed` | C |
| `doc-commands`, `plan-files` | E |
| `axis-packages` | F |
| `axis-wired`, `config-consumed` | D (+ 잔여 knob은 Wave 3 G) |

---

## 2. `trainbench/axes.py` + `trainbench/applied.py` — 요청과 실제

이 프로젝트에서 가장 중요한 단일 안전장치다. 없으면 sdpa로 폴백된 런이
"FA3 1.4배"로 리포트에 실린다.

**둘로 나뉜 이유**: 축을 켜는 코드와 켜졌는지 확인하는 코드가 같은 곳에 있으면,
"적용했다고 주장하는 것"과 "적용된 것"이 같은 근거를 갖게 된다. 분리해야 대조가 된다.

```python
# axes.py — 축을 켜는 유일한 지점
IMPLEMENTED: frozenset[str]                     # 실제로 적용 가능한 축
def load_kwargs(config) -> dict                 # from_pretrained에 넘길 것
def apply(model, config) -> list[str]           # 적재 후 적용, 적용한 축 반환

# applied.py — 켜졌는지 읽는 유일한 지점
@dataclass(frozen=True)
class AxisState:
    axis: str            # "attn.name" 같은 dotted knob
    requested: str
    applied: str | None  # None = 확인 불가
    detail: dict

@dataclass(frozen=True)
class AppliedState:
    axes: tuple[AxisState, ...]
    def mismatched(self) -> list[AxisState]
    def undetermined(self) -> list[AxisState]
    def missing(self) -> list[str]        # 스키마에 있는데 상태에 없는 축

def capture(model, config: BenchConfig) -> AppliedState
def assert_matches(state: AppliedState, config: BenchConfig) -> None  # AppliedMismatch
```

**불변식**

- `applied=None`(미확인)은 불일치와 **동일하게** `purpose in ("timing","quality")`를
  차단한다. "확인 못 했다"가 "괜찮다"로 읽히면 이 장치는 장식이 된다
- **상태가 비어 있거나 축이 빠져 있어도 차단한다.** 축 0개는 "전부 정상"이 아니라
  "capture가 돌지 않았다"이다
- `capture`는 절대 예외를 던지지 않는다. 읽기 실패는 `applied=None` + `detail.reason`.
  config가 이상해도, probe가 던져도 마찬가지다
- `assert_matches`는 **config를 받는다.** purpose 문자열을 받으면 `"Timing"` 오타
  하나로 전체 검증이 조용히 통과한다. 스키마에 없는 purpose는 `ValueError`
- `purpose`가 `probe`/`profile`이면 차단하지 않는다
- **축 집합은 스키마에서 유도된다.** `config_schema.py`에서 `Axis()`로 표시한 필드가
  곧 축이다. 손으로 적은 목록은 fail-open이다 — 목록에서 빠진 축은 "미확인"이 아니라
  아예 존재하지 않게 되고, 한 줄 지워도 아무 테스트도 실패하지 않는다

**D 레인의 작업**: `axes.py`에 적용을, `applied.py`의 `_CAPTURES`에 확인을 **쌍으로**
추가한다. capture 시그니처는 `(model, config) -> tuple[str | None, dict]`. 한쪽만
추가하면 `audit_plan.py`의 `axis-wired`가 막는다. `applied.py`의 데이터클래스와
`capture`/`assert_matches` 본문은 건드리지 않는다.

**Wave 3 G의 의무**: 측정 진입점(`scripts/bench.py`)은 `assert_matches`를 호출한다.
`audit_plan.py`의 `assert-called`가 호출자 존재를 강제한다.

---

## 3. `trainbench/probe/types.py` — 체크 결과

```python
@dataclass
class Check:
    name: str
    ok: bool
    expected_failure: bool = False   # 실패가 곧 답인 체크
    detail: dict
    error / error_type / traceback: str | None

@dataclass
class ProbeReport:
    ...
    applied: AppliedState | None = None   # 모델을 만든 어댑터가 채운다
```

- `ProbeReport.all_ok`는 `ok or expected_failure`로 계산한다
- `expected_failure`인데 **통과한** 체크는 `unexpected_passes`로 드러난다. 문서화된
  한계가 사라지면(예: Unsloth가 VLM을 받기 시작하면) support-matrix가 틀린 것이고,
  그것을 아는 곳은 그 런뿐이다. `all_ok`는 이걸 말할 수 없다
- `run()`/`skip()`에 `expected_failure=`를 넘길 수 있다. 이 플래그 하나 때문에
  `Check`를 손으로 조립하지 않는다
- probe는 **어떤 실패에도 예외를 밖으로 내보내지 않는다.** 모든 실패는 `Check`가 된다
- `report.run(name, fn)`이 반환하는 dict는 그대로 `detail`이 되므로 **텐서를 넣지
  않는다**

---

## 4. `trainbench/record.py` — 실행 기록

모든 run이 남기는 필드. 프레임워크 이미지마다 스택이 다르므로 버전은 결과와 함께
이동해야 한다.

| 필드 | 출처 |
|---|---|
| `git_commit` / `git_dirty` / `git_source` | `TRAINBENCH_GIT_COMMIT` 환경변수 우선, 없으면 `git rev-parse` + `git status --porcelain` |
| `image` / `image_digest` | `TRAINBENCH_IMAGE` / `TRAINBENCH_IMAGE_DIGEST` |
| `applied` | `build_record(config, device, applied=...)`. **없으면 결과 JSON은 요청만 담고 실제를 담지 않는다** |
| `packages` | `_TRACKED_PACKAGES`의 설치 버전 |
| `host` | `cpu_count_host` / `cpu_count_process` / `cpu_quota`(cgroup v2+v1) / `cpu_model` / `memory_total_gb` / `cuda_runtime` / `gpu` / `runpod_pod_id` |
| `config` | 검증된 `BenchConfig`의 전체 덤프 |

**C 레인의 작업**: 오케스트레이터가 `TRAINBENCH_GIT_COMMIT`,
`TRAINBENCH_IMAGE_DIGEST`, `INFISICAL_TOKEN`을 pod env로 주입한다.

`write_json`은 temp -> fsync -> `os.replace` 원자적 쓰기이며 `default=str`로 직렬화
실패가 결과 파일 전체를 잃지 않게 한다. fsync가 없으면 rename이 내용보다 먼저 도달할
수 있고, 위협 모델이 "pod이 쓰는 도중 사라진다"인 이상 그건 순서 보장이 아니다.

---

## 5. `trainbench/config_schema.py` — 확정된 스키마

**수정 금지.** 필드 추가가 필요하면 계약 변경으로 올린다.

모델별 사용 규격은 코드가 아니라 config에 있다. 기계 판독 가능한 형태는
`docs/model-spec.yaml`이고, `audit_plan.py`의 `model-spec`이 **값 대 값으로** 대조한다
(문자열 존재 확인은 true를 false로 뒤집어도 통과한다).

| 모델 | `add_generation_prompt` | `instruction_prompt` | `padding_side` | `tokens_per_image` |
|---|---|---|---|---|
| qwen3_vl_emb_2b | `true` | `"Represent the user's input."` | `right` | `null`(픽셀 비례) |
| qwen3_5_0_8b | `false` | `null` | `right` | `null`(픽셀 비례) |
| gemma4_e2b | `false` | `null` | **`left`** | `280`(고정) |

`padding_side`가 config에 있는 이유: gemma-4만 left이고, 그것이 `last_token_pool`
결함이 드러나는 유일한 모델이다. 코드가 `arch`로 분기하면 그 사실이 pooling 코드를
읽는 사람 눈에 보이지 않는다.

`attn.impl`은 **config에 없다.** `attn.name`에서 `ATTN_IMPL`로 유도된다. 둘을 따로
적을 수 있으면 `name: fa3 / impl: sdpa`가 가능해지고, 그 런은 fa3로 라벨링된 채
applied.py에서 sdpa-요청 sdpa-적용으로 **일치 판정**을 받는다.

실행 전 차단하는 검증기가 있다. 측정 규율을 산문이 아니라 코드로 만든 것이므로
**우회하지 않는다** — 검증기에 걸리면 config를 고치지 검증기를 고치지 않는다.

---

## 6. 매 wave 종료 게이트

```
infisical run --env=dev -- uv run ruff check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/audit_plan.py
```

`audit_plan.py`는 **회귀 추적기**다. 완료 기준이 아니다.

- `docs/audit-baseline.json`의 알려진 실패는 `KNOWN`으로 통과시킨다. 각 항목에 해소
  wave가 적혀 있어 baseline이 변명이 아니라 일정표가 된다
- **새 실패**가 생기면 차단한다
- baseline 항목이 **통과하기 시작해도 차단한다.** 낡은 baseline은 이후 파손에 조용히
  면죄부를 주기 때문이다. 해당 레인이 자기 한 줄을 삭제한다(§1)
- `--only`/`--skip`을 쓴 실행은 게이트가 아니다. 도구가 `PARTIAL RUN`을 출력하고
  `--update-baseline`을 거부한다

추가로 각 wave는 **작성자와 분리된 리뷰 레인**을 통과해야 한다(컨벤션 09). 2개 이상
모듈이나 인터페이스를 건드리면 3레인.
