# 공유 계약 (Wave 0 확정)

Wave 1~2의 모든 워크트리 레인이 이 문서를 계약으로 삼는다. **여기 정의된 인터페이스를
레인이 임의로 바꾸지 않는다.** 변경이 필요하면 직접 고치지 말고 계약 변경으로 올린다.

병렬 개발에서 각 레인이 서로 다른 스키마 위에 코드를 쌓으면 병합이 불가능해진다.
Wave 0을 순차 구간으로 둔 이유가 이것이다.

---

## 1. 파일 소유권

레인은 자기 소유 파일만 수정한다. 이 표가 유일한 기준이다.

| 레인 | 워크트리 | 소유 파일 |
|---|---|---|
| A 데이터 | `wt-data` | `scripts/prepare_data.py`, `configs/data/`, `tests/test_data.py` |
| B 코어정확성 | `wt-core` | `trainbench/embedding.py`, `trainbench/probe/{steps,native,registry}.py`, `trainbench/seed.py`, `scripts/verify_env.py`, `tests/test_embedding.py` |
| C 오케스트레이션 | `wt-orch` | `trainbench/pods.py`, `scripts/{orchestrate,publish_result,report}.py`, `configs/experiment/`, `docker/entrypoint.sh` |
| D 축구현 | `wt-axes` | `trainbench/applied.py`의 `_CAPTURES`, `configs/{attn,kernel,precision,compile,optim,freeze,dataloader,parallel}/` |
| E 문서 | `wt-docs` | `PLAN.md`, `README.md`, `AGENTS.md`, `docs/methodology.md`, `docs/support-matrix.md` |
| F 이미지 | `wt-images` | `envs/*/pyproject.toml`, `docker/Dockerfile.*`, `.github/workflows/` |

**공유(수정 금지)**: `trainbench/config_schema.py`, `trainbench/config.py`,
`trainbench/compose.py`, `trainbench/record.py`, `trainbench/probe/types.py`,
`trainbench/applied.py`의 인터페이스, `scripts/audit_plan.py`.

---

## 2. `trainbench/applied.py` — 요청 대 실제 적용

이 프로젝트에서 가장 중요한 단일 안전장치다. 없으면 sdpa로 폴백된 런이
"FA3 1.4배"로 리포트에 실린다.

```python
@dataclass(frozen=True)
class AxisState:
    axis: str            # "attn", "kernel", ...
    requested: str       # config에서 읽은 값
    applied: str | None  # 실제 적용값. None = 확인 불가
    detail: dict

@dataclass(frozen=True)
class AppliedState:
    axes: tuple[AxisState, ...]
    def mismatched(self) -> list[AxisState]
    def undetermined(self) -> list[AxisState]

def capture(model, config: BenchConfig) -> AppliedState
def assert_matches(state: AppliedState, purpose: str) -> None  # AppliedMismatch
```

**불변식**

- `applied=None`(미확인)은 불일치와 **동일하게** `purpose in ("timing","quality")`를
  차단한다. "확인 못 했다"가 "괜찮다"로 읽히면 이 장치는 장식이 된다
- `capture`는 절대 예외를 던지지 않는다. 읽기 실패는 `applied=None` + `detail.reason`
- `purpose`가 `probe`/`profile`이면 차단하지 않는다

**D 레인의 작업**: `_CAPTURES` 딕셔너리에 축별 probe를 추가한다. 시그니처는
`(model, config) -> tuple[str | None, dict]`. **다른 부분은 건드리지 않는다.**
축은 누군가 실제로 들여다보는 probe를 쓴 뒤에만 인증된다.

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
```

- `ProbeReport.all_ok`는 `ok or expected_failure`로 계산한다
- probe는 **어떤 실패에도 예외를 밖으로 내보내지 않는다.** 모든 실패는 `Check`가 된다
- `report.run(name, fn)`이 반환하는 dict는 그대로 `detail`이 되므로 **텐서를 넣지
  않는다**

---

## 4. `trainbench/record.py` — 실행 기록

모든 run이 남기는 필드. 프레임워크 이미지마다 스택이 다르므로 버전은 결과와 함께
이동해야 한다.

| 필드 | 출처 |
|---|---|
| `git_commit` | `TRAINBENCH_GIT_COMMIT` 환경변수 우선, 없으면 `git rev-parse` |
| `image` / `image_digest` | `TRAINBENCH_IMAGE` / `TRAINBENCH_IMAGE_DIGEST` |
| `packages` | `_TRACKED_PACKAGES`의 설치 버전 |
| `host` | `cpu_count_host` / `cpu_count_process` / `cpu_quota` / `cpu_model` / `memory_total_gb` / `cuda_runtime` / `gpu` / `runpod_pod_id` |
| `config` | 검증된 `BenchConfig`의 전체 덤프 |

**C 레인의 작업**: 오케스트레이터가 `TRAINBENCH_GIT_COMMIT`,
`TRAINBENCH_IMAGE_DIGEST`, `INFISICAL_TOKEN`을 pod env로 주입한다.

`write_json`은 temp -> `os.replace` 원자적 쓰기이며 `default=str`로 직렬화 실패가
결과 파일 전체를 잃지 않게 한다.

---

## 5. `trainbench/config_schema.py` — 확정된 스키마

**수정 금지.** 필드 추가가 필요하면 계약 변경으로 올린다.

모델별 사용 규격이 config에 있다(`docs/model-spec.md`의 결정 1·2):

| 모델 | `add_generation_prompt` | `instruction_prompt` |
|---|---|---|
| qwen3_vl_emb_2b | `true` | `"Represent the user's input."` |
| qwen3_5_0_8b | `false` | `null` |
| gemma4_e2b | `false` | `null` |

실행 전 차단하는 검증기 10종이 있다. 측정 규율을 산문이 아니라 코드로 만든 것이므로
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
  면죄부를 주기 때문이다. 확인 후 `--update-baseline`으로 갱신한다

추가로 각 wave는 **작성자와 분리된 리뷰 레인**을 통과해야 한다(컨벤션 09). 2개 이상
모듈이나 인터페이스를 건드리면 3레인.
