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
| D 축구현 | `wt-axes` | `trainbench/axes.py`, `trainbench/applied.py`의 `_CAPTURES`·`_REQUESTED_OVERRIDES`·capture 함수들, `configs/{attn,kernel,precision,compile,optim,freeze,dataloader,parallel,peft,loss,framework}/`, `configs/train/`, `tests/test_axes.py` |
| E 문서 | `wt-docs` | `PLAN.md`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/methodology.md`, `docs/support-matrix.md`, `docs/model-spec.md` |
| F 이미지 | `wt-images` | `envs/*/`(pyproject + lock), `docker/Dockerfile.*`, `.github/workflows/`, `pyproject.toml`, 루트 `uv.lock`, `.pre-commit-config.yaml`, `.gitignore`, `.python-version` |
| G 하네스 (Wave 3) | 순차 | `scripts/bench.py`, `trainbench/metrics/`, `tests/{test_metrics,test_smoke_cpu}.py` |

**공유(수정 금지)**: `trainbench/config_schema.py`, `trainbench/config.py`,
`trainbench/compose.py`, `trainbench/record.py`, `trainbench/probe/types.py`,
`trainbench/applied.py`의 인터페이스(데이터클래스·`capture`·`assert_matches`),
`scripts/audit_plan.py`, `scripts/compose_config.py`, `docs/CONTRACTS.md`,
`docs/model-spec.yaml`, `tests/{conftest,test_config,test_applied,test_audit}.py`,
`.env.example`, `.infisical.json`, `configs/config.yaml`,
`trainbench/__init__.py`, `tests/__init__.py`.

`configs/config.yaml`이 공유인 이유: `audit_plan.py`의 `_composed_groups()`가 이
파일의 `defaults`에서 config group 집합을 유도하므로, 이 한 파일이 `config-groups` ·
`config-consumed` · `axis-fields` 세 체크의 **검사 범위를 결정한다.** 축 그룹을 추가하는
것은 D의 작업이지만 `defaults`에 한 줄을 추가하는 것은 계약 변경이다.

**기록 문서(수정 금지, 추가만)**: `docs/review-findings.md`, `.claude/plans/`.
무엇이 왜 틀렸는지의 기록이므로 고쳐서 맞게 만들지 않는다. 해소는 항목에 표시로 남긴다.

### Wave 3 시작 시 이관

G는 `docker/entrypoint.sh`(C 소유)를 반드시 고쳐야 한다 — pod 진입점이 `bench.py`를
호출해야 하고, 지금은 `probe` 외 purpose에 분기가 없다. **Wave 3 착수 시점에
`docker/entrypoint.sh`와 `scripts/orchestrate.py`의 `RUNNABLE_PURPOSES`가 C에서 G로
이관된다.** 이관 없이 G가 손대면 병합된 레인의 파일을 되돌리는 일이 된다.

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
# axes.py — 축을 켜는 유일한 지점. 이 4개가 bench.py의 호출 지점이다
IMPLEMENTED: frozenset[str]                            # 실제로 적용 가능한 축
def patch(config) -> list[str]                         # 모델 생성 "이전"
def load_kwargs(config) -> dict                        # from_pretrained kwargs
def assemble(model, config, device, framework, dataset=None) -> tuple[Built, list[str]]
def step_context(config) -> AbstractContextManager     # 스텝을 감싸는 컨텍스트
class UnappliedAxis(RuntimeError)                      # 구현 없음 -> 기본값 대체 금지

# applied.py — 켜졌는지 읽는 유일한 지점
@dataclass(frozen=True)
class Built:                          # 런이 만든 것. 축은 모델에만 있지 않다
    model / optimizer / dataloader / loss_fn / framework

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

def capture(built: Built, config: BenchConfig) -> AppliedState
def assert_matches(state: AppliedState, config: BenchConfig) -> None  # AppliedMismatch
```

**호출 지점 4개를 Wave 0에서 고정하는 이유**: 이걸 호출할 하네스(`scripts/bench.py`)는
Wave 3에 다른 레인이 만든다. 형태가 없으면 축을 추가하는 레인과 하네스를 짓는 레인이
각자 다른 인터페이스를 가정하게 되고, 그게 정확히 이 문서가 막으려는 것이다.

`assemble`이 조각별 빌더가 아니라 **한 번에 전부** 만드는 이유는 조각을 따로 만들 수
없는 축이 있기 때문이다. 두 가지를 문서로 확인했다:

| 근거 | 사실 | 영향 |
|---|---|---|
| DeepSpeed 튜토리얼(cifar-10, bert-pretraining) | `deepspeed.initialize(model=, model_parameters=, training_data=)` -> `(engine, optimizer, dataloader, scheduler)` | `parallel.strategy=zero2/3`와 `train.offload`를 독립 훅 3개로 쪼갤 수 없다 |
| DALI 문서 | "replacing the standard DataLoader with DALIClassificationIterator" | `dataloader.backend=dali`는 kwargs가 아니라 로더 자체를 교체한다 |
| Liger README | `apply_liger_kernel_to_llama()` 다음에 `# 2. Instantiate patched model` | `kernel.name`은 생성 이전에 적용된다. 그래서 `patch`가 별도 호출 지점이다 |

`assemble`이 모델을 **반환**하는 이유도 같다. `torch.compile`과 `get_peft_model`은
새 객체를 돌려주고 FSDP/DeepSpeed는 감싼다. in-place 변형만 표현하는 시그니처로는
이들을 담을 수 없다.

`step_context`가 따로 있는 이유: precision은 구성 시점만의 선택이 아니다. fp8 recipe는
forward를 감싼다. 갈 곳 없는 축은 검증되지 않는 어딘가에서 적용된다.

`capture`가 `Built`를 받는 이유: `optim.name`은 옵티마이저가, `dataloader.*`는
데이터로더가, `loss.name`은 손실이, `framework.name`은 실제로 실행된 어댑터가
결정한다. 모델만 보는 capture는 이들을 영원히 미확인으로 두거나, 더 나쁘게는 추측하도록
넓혀진다.

`framework`는 config가 아니라 **어댑터가 리터럴로** 넘긴다. config는 요청이고, 요청이
실행의 증거가 아니라는 것이 이 모듈의 존재 이유다.

### 17축이 4개 호출 지점에 전부 들어가는가

`assemble` 내부는 D가 자유롭게 나눈다(`_apply_to_model` / `_optimizer` / `_dataloader` /
`_loss`가 현재 구성이다). 계약은 **호출 지점 4개**이지 내부 구조가 아니다. 다만 순서
하나는 고정한다 — **모델 변형이 옵티마이저 생성보다 먼저다.** FSDP2는 샤딩된 파라미터
위에 옵티마이저가 만들어져야 하고, 순서가 뒤집히면 옵티마이저가 원본 파라미터를 잡는다.

| 호출 지점 | 담는 축 |
|---|---|
| `patch` | `kernel.name` (liger/fla/kernels_hub) |
| `load_kwargs` | `attn.name`, (qlora 양자화 config, precision의 적재 dtype) |
| `assemble` -> 모델 | `freeze.vision_tower`, `freeze.ple`, `compile.mode`, `peft.mode`, `train.gradient_checkpointing`, `precision.name`의 모듈 교체(torchao) |
| `assemble` -> 옵티마이저 | `optim.name`, `train.offload` |
| `assemble` -> 데이터로더 | `dataloader.backend/packing/pretokenize` |
| `assemble` -> 손실 | `loss.name`, `parallel.cross_device_negatives` |
| `assemble` -> 공동 초기화 | `parallel.strategy` (FSDP2/DDP는 모델 래핑, ZeRO는 모델+옵티마이저+로더 동시) |
| `assemble` -> `framework` 인자 | `framework.name` |
| `step_context` | `precision.name`의 fp8 autocast |

빠진 축이 없다. D가 이 표에서 벗어나는 축을 만나면 그것은 계약 변경이다.

`framework.name`은 훅이 적용하는 것이 아니라 **어댑터가 자기 이름을 리터럴로 넘겨서**
결정된다. `IMPLEMENTED`에 있지만 거짓 등재가 아니다 — 그 리터럴을 쓴 파일이 곧 어느
코드 경로가 돌았는지의 증거다.

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
추가한다. capture 시그니처는 `(built, config) -> tuple[str | None, dict]`. 한쪽만
추가하면 `audit_plan.py`의 `axis-wired`가 막는다. `applied.py`의 데이터클래스와
`capture`/`assert_matches` 본문은 건드리지 않는다.

새 knob을 추가하면 `Axis()` 마커를 붙이거나 `audit_plan.py`의 `NOT_AN_AXIS`에 올려야
한다. 둘 다 아니면 `axis-fields`가 막는다 — 마커가 opt-in이라 잊으면 그 필드만 검증
밖으로 빠지고, 그건 축 전체가 빠지던 것과 같은 모양의 구멍이다.

**Wave 3 G의 의무**: 측정 진입점(`scripts/bench.py`)이 `assert_matches`를 호출한다.
`audit_plan.py`의 `assert-called`는 **그 파일이 호출하는지**를 본다. "어딘가에 호출자가
있으면 통과"로는 부족하다 — probe도 호출하지만 `purpose=probe`는 즉시 return하므로,
호출 없는 하네스가 초록불 아래에서 무검증 측정을 돌릴 수 있다.

### probe가 mismatch를 낼 때 고칠 것은 probe가 아니다

vision tower가 구조적으로 FA를 못 받아 transformers가 개별 강등하면 `attn` 요청은
항상 `mixed(...)` -> mismatch가 되어 timing이 영구 차단된다. 이때 **probe를 완화하지
않는다.** 완화는 C2가 고친 결함을 그대로 되돌리는 것이다. config가 기대하는 per-module
구성을 선언하게 하고, capture는 그 기대와 대조한다. 이질적 적용은 표현되어야 할 상태이지
숨겨야 할 잡음이 아니다.

### probe 어댑터의 축 중복 (B/D 경계)

`trainbench/probe/native.py`의 `_lora_attach`는 축 적용 지점이 **아니다.** peft는
`axes.py`에 구현이 없고(LoRA가 모든 base 파라미터를 얼려 `freeze.ple` 판정과 충돌한다 —
freeze 축이 "얼림"인지 "peft가 얼린 것에 더해 얼림"인지는 축을 구현하는 레인이 정한다),
`_lora_attach`는 모델을 in-place로 재작성하므로 **마지막에 실행되고 그 뒤에 어떤 체크도
오지 않는다.**

**D가 `peft.mode`를 켜는 순간 모든 LoRA timing 런이 차단된다.** 이건 열린 설계 질문이
아니라 확정된 결과다: peft가 base 파라미터를 전부 얼리므로 `freeze.ple=false` 요청이
`applied="True"`와 mismatch를 낸다. 그리고 이 프로젝트의 표제 비교가 full finetuning
대 LoRA다 — 축을 켜자마자 스터디의 절반이 멈춘다. **해법은 probe 완화가 아니라
`freeze.*` capture가 peft가 얼린 것을 기준선으로 잡고 그 위의 차분을 재는 것이다.**
D는 축을 구현하기 전에 이걸 정하고 들어간다. PLE 파라미터 판별은 `axes.ple_parameters` 하나뿐이다 — native.py가 갖고
있던 두 번째 정의는 제거했다(이미 죽은 `altup` 조건으로 드리프트해 있었다).

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

계약 변경 이력:

- 2026-08-01 `CORRUPT_DATA_REVISIONS` + `_no_run_reads_a_corrupt_subset` 추가.
  손상 서브셋 revision을 핀한 config는 **purpose와 무관하게** 구성 자체가 실패한다.
  `scripts/prepare_data.py`는 이 목록을 import하며 자체 정의를 두지 않는다 —
  "손상"의 정의가 두 곳에 있으면 그게 D1(컬럼 목록 2개)의 재현이다. 필드 추가가
  아니라 검증기 추가이므로 config 스키마 자체는 그대로다.
- 2026-08-01 **`DataConfig.source_revision` 추가**(필드 추가, 기본값 없음).
  샘플러가 미러의 HEAD를 스트리밍하는 동안 `prepare_data.py` docstring과 `PLAN.md`는
  커밋을 고정한다고 서술하고 있었다 — 코드가 하지 않는 일을 문서가 주장하는 상태였고,
  shard 캐시가 얹히면서 업스트림 변경이 재실행으로도 드러나지 않게 됐다. 이 값은
  shard 캐시 키의 일부이기도 하다. `configs/data/*.yaml` 양쪽에 기입 필요.
- 2026-08-01 **`tests/test_applied.py::test_an_axis_without_a_probe_blocks_a_timing_run`이
  합성 축을 쓴다**(공유 테스트 변경, 스키마 변경 아님). 이 테스트는 "probe 없는 축"의
  예시로 `compile.mode`를 리터럴로 박아두고 있었다. D 레인의 작업이 정확히 그 축에
  probe를 붙이는 것이므로, **어떤 축을 배선하든 이 테스트가 깨진다.** 리터럴을 아직
  미배선인 다른 축으로 바꾸는 것은 같은 일이 다음 레인에서 반복되게 하고, 그 반복은
  "실패를 읽지 않고 이름만 고친다"로 굳는다. 어떤 축이 미배선인지는 `audit_plan.py`의
  `axis-wired`가 이미 추적하므로 두 곳에서 단언할 이유도 없다. 테스트는
  `AppliedState`에 `synthetic.unwired` 축 하나를 덧붙여 **메커니즘만** 검사한다 —
  미확인 축은 undetermined이고 timing을 차단한다. `axis_knobs() - _CAPTURES`에서
  동적으로 유도하지 않는 이유는 그 형태가 전 축이 배선된 순간 검사 대상 0개로
  조용히 통과하기 때문이다(§6의 "빈 입력은 통과가 아니라 실패다").
- 2026-08-01 **§2가 열어둔 freeze x peft 충돌을 닫는다** — `config_schema.py`에
  검증기 2건 + `tests/test_config.py` 테스트 3건.
  결정이 아니라 측정으로 닫았다(peft 0.20.0): `get_peft_model`은 base 파라미터를 전부
  얼리고, **freeze 축이 먼저 돌았든 아니든 결과가 같다.** 즉 어댑터 아래에서
  `freeze.ple=true`와 `false`는 같은 모델을 만든다 — 정의할 합성 의미가 없다.
  그래서 "얼린 것에 더해"인지 "얼린 것"인지를 고르는 대신 **조합 자체를 config 단계에서
  거부**한다. 허용하면 ablation 표에 라벨만 다른 동일 모델 두 행이 생긴다.
  `peft.r=0`도 거부한다 — 학습 가능한 어댑터 파라미터가 0개인 LoRA 런이다.
- 2026-08-01 **`audit_plan.py` 감사 계층 수리 3건 + `tests/test_audit.py`.**
  Wave 2 게이트 리뷰가 찾은 것: 병합 트리에서 D의 capture 계층을 통째로 되돌려도
  `7/11 passing, 0 new failure(s), 0 newly fixed`가 바이트 단위로 동일했다. 게이트가
  판정에 쓰는 줄이 이번 wave 산출물 전체를 보지 못했다.
  - `Result.count` + baseline 형식 `{체크: {note, count}}`. 이미 실패 중인 체크
    **안에서의** 크기 변화가 양방향으로 차단된다. 문자열 항목은 note로 읽어 크기
    비교만 비활성화한다(옛 baseline 호환).
  - **`axis-values` 신설.** `axis-wired`는 knob 이름의 멤버십 검사라 축이 비활성값
    하나만 받아도 통과한다 — Wave 2 후 12→2로 내려갔지만 12개 ablation 그룹 중
    7개가 여전히 기본값만 받는다. 새 체크는 각 variant를 실제로 4개 호출 지점에
    통과시켜 적용 가능한 값을 센다(현재 25/43). 두 체크는 서로 다른 질문이고 어느
    쪽도 다른 쪽을 대신하지 않는다.
  - **`AXIS_PACKAGES` 이름 정정.** `nvidia-dali`는 NVIDIA가 오설치를 경고하려고
    올린 자리표시자이고 `grad-cache`는 404다 — 올바른 설치가 영원히 체크를 만족시킬
    수 없었다. `nvidia-dali-cuda130`/`gradcache`로 고치고, 통과 문구를 "어느 락엔가
    이름이 있다"는 실제 의미로 바꿨다. 빌드·설치·import를 증명하지 않는다.
  `tests/test_audit.py`(공유)는 옛 시그니처를 쓰고 있어 함께 고쳤고, 개수 회귀가
  차단되는지 검사하는 테스트를 추가했다.
- 2026-08-01 **레인 경계 침범 기록**. 아래 파일들을 레인 A 작업이 수정했다. 기술적
  위험은 낮지만(아래 근거) 소유 레인이 모르는 채로 남지 않도록 여기 남긴다.
  - `pyproject.toml`, `uv.lock` (레인 F 소유) — `compose` extra에 `pyarrow>=21.0`
    추가. `scripts/prepare_data.py`의 shard 캐시가 필요로 하는데 지금까지는
    `native` extra의 `datasets`를 통해서만 딸려왔다. `envs/*/pyproject.toml`에는
    `pyarrow` 고정이 하나도 없어 이미지 해석에 영향이 없고, `uv.lock` 변화는 2줄이다.
  - `tests/test_config.py` (공유) — 위 두 검증기의 테스트 추가. 스키마 변경과 같은
    커밋에 들어가지 않으면 검증기가 테스트 없이 랜딩한다.

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

**빈 입력은 통과가 아니라 실패다.** 집합을 순회하는 체크는 그 집합이 비면 실패한다.
Wave 1 착수 직후 세 레인이 동시에 발견한 것이 이 형태였다 — `.gitignore`의 앵커 없는
`data/`가 `configs/data/`를 삼켜 한 번도 커밋된 적이 없었고, `data-pinned`가
"every data config pins a commit sha"로 통과했다. config가 0개라서 참이었다. 동시에
깨끗한 clone에서는 Hydra 합성이 불가능해 Wave 0 게이트가 재현되지 않았다.
`tests/test_audit.py::test_a_check_with_nothing_to_examine_fails`가 5개 체크 전부를
고정한다.

- `docs/audit-baseline.json`의 알려진 실패는 `KNOWN`으로 통과시킨다. 각 항목에 해소
  wave가 적혀 있어 baseline이 변명이 아니라 일정표가 된다
- **새 실패**가 생기면 차단한다
- baseline 항목이 **통과하기 시작해도 차단한다.** 낡은 baseline은 이후 파손에 조용히
  면죄부를 주기 때문이다. 해당 레인이 자기 한 줄을 삭제한다(§1)
- `--only`/`--skip`을 쓴 실행은 게이트가 아니다. 도구가 `PARTIAL RUN`을 출력하고
  `--update-baseline`을 거부한다

추가로 각 wave는 **작성자와 분리된 리뷰 레인**을 통과해야 한다(컨벤션 09). 2개 이상
모듈이나 인터페이스를 건드리면 3레인.

### CPU에는 timing 경로가 없다

`optim.name`은 `fused=device.type=="cuda"`이므로 CPU에서는 `adamw_fused` 요청에
`adamw_unfused`가 적용되어 영구 mismatch다. 의도된 설계다 — CPU 수치는 어차피 보고
대상이 아니다. 다만 **CPU 통합 테스트는 `purpose=probe`로만 짤 수 있다.**
`tests/test_smoke_cpu.py`(Wave 3 G)가 여기 부딪힌다.
