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
| C 오케스트레이션 | `wt-orch` | `trainbench/pods.py`, `scripts/{orchestrate,publish_result,report}.py`(`RUNNABLE_PURPOSES` 제외), `configs/experiment/`, `configs/run/`, `docs/evidence/`, `tests/test_pods.py` |
| D 축구현 | `wt-axes` | `trainbench/axes.py`, `trainbench/applied.py`의 `_CAPTURES`·`_REQUESTED_OVERRIDES`·capture 함수들, `configs/{attn,kernel,precision,compile,optim,freeze,dataloader,parallel,peft,loss,framework}/`, `configs/train/`, `tests/test_axes.py` |
| E 문서 | `wt-docs` | `PLAN.md`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/methodology.md`, `docs/support-matrix.md`, `docs/model-spec.md` |
| F 이미지 | `wt-images` | `envs/*/`(pyproject + lock), `docker/Dockerfile.*`, `.github/workflows/`, `pyproject.toml`, 루트 `uv.lock`, `.pre-commit-config.yaml`, `.gitignore`, `.python-version` |
| G 하네스 (Wave 3) | 순차 | `scripts/bench.py`, `trainbench/metrics/`, `tests/{test_metrics,test_smoke_cpu}.py`, `docker/entrypoint.sh`, `scripts/orchestrate.py`의 `RUNNABLE_PURPOSES` |

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

**이관 완료 (2026-08-01, Wave 3 착수).** 위 §1 표를 갱신했다. 이관 대상은
`docker/entrypoint.sh` 전체와 `RUNNABLE_PURPOSES` 하나뿐이며, `scripts/orchestrate.py`의
나머지는 C 소유로 남는다.

### `docs/audit-baseline.json` — 공유하되 한 줄씩만

이 파일은 6개 레인 중 5개가 반드시 건드린다(§6이 "통과하기 시작해도 차단"이므로
자기 항목을 해소한 레인은 자기 게이트에서 막힌다). 그래서 규칙을 좁게 고정한다.

- 레인은 **자기 항목 한 줄만 삭제**한다. 다른 줄은 읽지도 고치지도 않는다
- 레인에서 **`--update-baseline`을 실행하지 않는다.** 전체 실행이 아니면 도구가
  거부하고, 전체 실행이면 다른 레인의 미완 항목까지 자기 상태로 덮어쓴다
- 항목 추가는 계약 변경이다. 새 실패는 baseline이 아니라 수정으로 해소한다
- **note는 어느 레인 소관인지가 아니라 그 구멍의 결과를 적는다.** 이건 스타일 규칙이
  아니라 이 저장소에서 여섯 번 반복된 실패의 수리다. `axis-wired`의 note는
  "Wave 2 (D: axes) - one apply site and one capture probe per axis"였는데,
  그 상태의 실제 의미는 **`assert_matches`가 모든 `purpose=timing` 런을 거부해
  측정이 하나도 불가능하다**는 것이었다. 감사는 사실을 보고하고 있었고 note가
  그 함의를 가렸으며, 그 위에 Wave 3이 얹혔다. note를 읽는 사람은 다음 레인이고,
  그가 알아야 하는 것은 담당자가 아니라 지금 무엇이 안 되는가다

`--update-baseline`은 note를 보존하지만 **count는 이번 실행 값으로 덮어쓴다.**
그래서 다른 레인이 고친 것이 내 count에 흡수될 수 있다. 실행하지 않는 이유가
하나 더 있는 셈이고, count가 움직였는데 내가 한 일이 아니면 baseline이 아니라
보고로 올린다.

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
| `patch` | `kernel.name` (liger/fla) |
| `load_kwargs` | `attn.name`, (qlora 양자화 config, precision의 적재 dtype) |
| `assemble` -> 모델 | `freeze.vision_tower`, `freeze.ple`, `compile.mode`, `peft.mode`, `train.gradient_checkpointing`, `precision.name`의 모듈 교체(torchao) |
| `assemble` -> 옵티마이저 | `optim.name`, `train.offload` |
| `assemble` -> 데이터로더 | `dataloader.backend/packing/pretokenize` |
| `assemble` -> 손실 | `loss.name`, `parallel.cross_device_negatives` |
| `assemble` -> 공동 초기화 | `parallel.strategy` (FSDP2/DDP는 모델 래핑, ZeRO는 모델+옵티마이저+로더 동시) |
| `assemble` -> `framework` 인자 | `framework.name` |
| `step_context` | `precision.name`의 fp8 autocast, 그리고 어댑터의 `required_step_context` |

빠진 축이 없다. D가 이 표에서 벗어나는 축을 만나면 그것은 계약 변경이다.

`step_context`는 축 하나가 아니라 **정밀도 컨텍스트를 세우는 유일한 자리**다. 프레임워크가
상류에서 이미 다른 수치 체제로 학습하면(axolotl 은 `embed_tokens`/`lm_head` 를 fp32 로 두고
나머지를 bf16 으로 적재하므로 `torch.autocast` 없이는 matmul 이 죽는다) 어댑터는
`AdapterOut.required_step_context` 로 **요구만** 하고 자기 `with` 를 열지 않는다.
`axes.step_context(config, required)` 가 그것을 세운다 — `scripts/bench.py` 가 넘긴다.
둘이 동시에 요구되면 거부한다: fp8 recipe 와 bf16 autocast 는 같은 질문에 대한 두 답이고,
겹쳐 켜면 한 라벨 아래 두 체제로 잰 숫자가 된다.

그 결과 **native(순수 bf16)와 axolotl(autocast)은 다른 수치 체제에서 비교된다.** 이 사실은
`documented_entry_point.differs` 와 `required_step_context` 양쪽에 남고, 결과를 읽는 쪽이
프레임워크 차이로 오해하지 않도록 결과에 실려야 한다.

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
- 2026-08-01 **`config-consumed`가 산문에 만족하던 것을 고친다** — `audit_plan.py`에
  `_strip_prose()` 추가. 이 체크는 정규식이라 knob을 **언급만** 해도 통과했다.
  `scripts/bench.py` docstring이 `config.data.subset_rows`를 적었을 뿐인데 소비된 것으로
  보고됐다 — 이 파일 헤더가 경고하는 실패이고 `assert-called`가 AST 파싱인 이유인데,
  이 체크는 그 결함을 그대로 갖고 있었다. 주석과 docstring만 제거한다: 부분식
  `config["data"]["key"]`도 문자열이라 문자열 전체를 지우면 그 형태를 못 잡는다.
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
- 2026-08-02 **레인 경계 침범 기록 (35a9a62, Wave 3 G -> 레인 A).**
  `scripts/prepare_data.py`(§1에서 A 소유)를 G의 커밋이 수정했다(+3 -9):
  자체 `_percentile`을 지우고 `trainbench.metrics.percentile`을 import한다.
  같은 커밋의 C->G 이관은 위 §"Wave 3 시작 시 이관"에 꼼꼼히 남겼는데 이건 남기지
  않았다. **레인 A는 이 변경을 모른 채였다.** 중복 제거 자체는 컨벤션 01대로 옳다 —
  nearest-rank 정의가 두 곳에 있으면 그게 D1(컬럼 목록 2개)의 재현이다.
  **의존 방향은 남겨둔다(이동하지 않는다), 근거는 아래.** `trainbench/metrics/`의
  docstring은 자기를 "What a timing run reports, and how it is measured"라고
  적고 있고 `percentile`은 측정 개념이 아니라 산술 유틸이므로, 데이터 준비
  스크립트가 타이밍 보고 모듈에 의존하는 모양이 된 것은 사실이다. 그럼에도
  옮기지 않는 이유는 (a) 중립 모듈을 새로 만들면 함수 하나짜리 모듈이 생기고
  §1의 소유 표에 또 한 줄이 필요해진다, (b) 두 소비자가 같은 것을 원하는 이유가
  실제로 같다 — `percentile`의 docstring이 말하는 "보간하면 아무 step도 걸리지
  않은 시간을 보고하게 된다"가 prepare_data의 행 길이 분포에도 그대로 적용된다,
  (c) 이 방향은 순환이 아니고 `trainbench.metrics`는 torch 외에 아무것도 import하지
  않는다. **재검토 조건**: `trainbench/metrics/`가 타이밍 런 상태(step 시간 버퍼,
  device 핸들)를 들고 있게 되는 순간 이 근거는 무효다. 그때 산술만 중립 모듈로
  분리한다.
- 2026-08-02 **`config-consumed`를 정규식에서 AST로 옮긴다** + **`plan-files`가
  양방향이 된다**(`audit_plan.py`, `tests/test_audit.py`). 리뷰가 두 체크를 각각
  무력화했다.
  - `_strip_prose()`는 자기가 막으려던 구멍의 절반만 막고 있었다. 문자열을 이름에
    대입하면(`_NOTE = "config.data.subset_rows"`) `ast.Expr`가 아니라 `ast.Assign`이라
    살아남았고, 속성 정규식에 앵커가 없어 `anything.data.subset_rows`도 읽기로
    셌으며, `if False:` 블록 안의 접근도 셌다. 셋 다 미읽음 수를 5에서 3~4로
    떨어뜨렸다. 산문 제거를 고치는 대신 **읽기 자체를 AST로 판정**한다 —
    `config_schema.py`가 검증기 에러 메시지에 knob 이름을 관용적으로 넣으므로
    문자열을 금지하는 방향은 쓸 수 없다. 앵커는 **그룹 바로 앞 이름**이라
    `self.config.model.add_generation_prompt`는 읽기로 잡히고
    `anything.data.subset_rows`는 아니다. 덤으로 fail-closed 버그도 사라진다 —
    주석 제거가 `line.split("#", 1)[0]`이라 문자열 안의 `#` 뒤가 잘려나갔다.
    **잔여 한계**: 상수 조건만 접는다. `if 1 == 2:`는 접지 않고, 읽은 값이
    쓰이는지도 보지 않는다(그건 `axis-values`의 질문이다).
  - `plan-files`는 **적혀 있는데 없는 것**만 봤다. 반대 방향이 없어서
    `trainbench/metrics/`, `scripts/bench.py`, 테스트 6개가 전부 구조 블록에 없는
    채로 통과했다 — 블록은 언급을 줄이면 참으로 유지된다. 그리고 확장자 있는
    항목만 검사 대상이라 `configs/nonexistent/`가 블록에 있어도 통과했다.
    이제 디렉터리 존재도 확인하고 **저장소에 있는데 블록에 없는 것도 막는다.**
    범위는 트리에서 유도한다: **자식이 적힌 디렉터리는 완전한 목록을 주장하는
    것으로 보고 그 안을 검사하고, 자식 없이 이름만 적힌 디렉터리**(`docker/`,
    `envs/`, `trainbench/probe/`, 각 config 그룹)**는 통째로 문서화된 것으로 보고
    안을 보지 않는다.** dot 항목과 `__init__.py`는 제외이며, 이 범위는 체크가
    통과할 때도 자기 출력에 적는다 — 무엇을 안 봤는지가 보여야 한다.

- 2026-08-02 **`config_schema.py`의 `RunConfig` docstring에서 출처 없는 "20~44%"를
  제거한다** (공유 파일 변경 요청, 스키마 변경 아님). `PLAN.md`/`README.md`/`AGENTS.md`
  세 곳에서는 이미 제거했고 `docs/methodology.md` §1이 조사 기록과 함께 "미측정"으로
  추적하고 있는데, 이 파일에만 숫자가 남아 **코드가 문서보다 강하게 주장하는** 상태다.
  스키마를 읽는 사람이 마지막으로 보는 것이 그 숫자다. 고칠 것은 한 줄 — 숫자를 지우고
  "부풀림 폭은 미측정, `docs/methodology.md` §1" 로 바꾼다. **검증기 동작은 바뀌지
  않는다**: `_timing_runs_are_uncontaminated`의 거부는 폭이 아니라 분리 규율에서
  나오고, 그 규율은 숫자와 무관하게 유지된다(프로파일러가 스텝을 느리게 만든다는 것은
  프로파일러의 동작 정의다). 소유 레인이 없어 아무도 손대지 못한 채 세 wave를 지났다.
- 2026-08-02 **계획이 측정 인프라와 측정 대상 기법을 같은 것으로 취급했다** (`PLAN.md`
  Task 3.5 신설, 레인 E). `Task 4 — Phase 2 ablation`이 "`configs/experiment/`에 축
  그룹별 조합 정의 후 실행"만 적고 있었고, 그 축들을 **구현하는** 작업이 어느 Task에도
  없었다. Liger 패칭, Muon, GradCache, Transformer Engine recipe, DALI, FSDP2/DDP/ZeRO —
  전부 작업 항목이 아니었다.
  원인은 이 프로젝트 자신의 규칙이다. `AGENTS.md`와 `PLAN.md`의 "새 실험 변형은 Hydra
  config 조합에서 나오지 코드 변경에서 나오지 않는다"는 축을 *변화시키는 방법*으로는
  맞지만 조용히 **축이 이미 존재한다**는 전제를 깔았다. ms-swift나 axolotl을 쓴다면
  맞는 전제다 — 우리는 `native` 하네스를 자체로 만들고 있고 거기서는 축 하나가 config
  한 줄이 아니라 구현 하나다. 그 결과 `axis-values`가 그 구멍을 정확히 보고하고 있었는데
  (`26/43`, 6개 그룹이 비활성값 하나만 수용) 계획에는 그것을 닫는 작업이 없어서,
  감사가 말하는 것과 계획이 하는 일이 서로를 참조하지 않았다.
  **다음 레인이 읽어야 할 것**: 축 하나를 켜는 일은 `axes.py`의 적용 지점 + `applied.py`의
  capture 확장 **두 개가 한 커밋**이다. `applied=None`은 불일치와 동일하게 timing을
  차단하므로(§2 불변식), 구현만 랜딩한 축은 "구현됐지만 영구히 측정 불가"라는 상태로
  들어간다. 현재 그 상태가 예정된 것 3건(`optim=adamw_8bit`의 철자 불일치,
  `train.offload`의 deepspeed undetermined, `precision=mxfp8/nvfp4`의 step-내 캐스팅)이며
  `PLAN.md` Task 3.5가 표로 추적한다.

- 2026-08-02 **파드 시크릿 가드가 금지 목록에서 허용 목록으로 바뀐다**
  (`scripts/orchestrate.py`의 `ALLOWED_ON_POD` 신설, 레인 C). 결정이 아니라 측정으로
  바꿨다: `infisical run --env=dev` 아래 `os.environ`을 맨 셸과 diff한 결과 `dev`가
  **27개 이름**을 주입하고, `FORBIDDEN_ON_POD`이 아는 것은 4개, 파드가 쓰는 것은
  `HF_TOKEN` 1개다. **나머지 22개가 가드를 통과해 파드에 도달**하고 있었다 — 클라우드·
  데이터베이스·모델 제공자 자격증명들이며 목록에 올릴 생각을 아무도 하지 않은 것들이다.
  가드 자신의 에러 메시지가 이미 "lengthening FORBIDDEN_ON_POD cannot reach this"라고
  적어두고 있었는데, 정작 검사는 그 목록으로 하고 있었다.
  허용 목록의 근거는 `.env.example`이다 — "Experiment pods: model/dataset pull, result
  push, Trackio Space sync"로 `HF_TOKEN` 하나를 지목하고 나머지 셋을 오케스트레이터·
  빌드 전용으로 명시한다. `TRAINBENCH_*`와 `INFISICAL_*`은 Infisical이 아니라 파드 env
  dict로 건네지므로 이 목록에 들어가지 않는다.
  **양방향 거부다.** 초과분(보안)과 `HF_TOKEN` 부재(측정 타당성) 둘 다 막는다. 후자는
  이 저장소에 이미 기록된 실패다 — 토큰이 없어 게이트 모델이 401을 냈고 그 조합이
  "미지원"으로 기록됐다. 빈 스코프도 같은 이유로 통과가 아니다(§6 "빈 입력은 통과가
  아니라 실패다").
  `FORBIDDEN_ON_POD`은 남는다: `pod_env`의 dict 검사와, 거부 메시지에서 어느 초과분이
  account-wide인지 지목하는 용도다.
- 2026-08-02 **스코프 검사가 파드와 다른 Infisical 환경을 보고 있었다**
  (`scripts/orchestrate.py`, 레인 C). `pod_env`는 파드에 `args.infisical_env`를 주는데
  `pod_reachable_secret_names`는 `os.environ["INFISICAL_ENV"]`로 검사하고 있었다. 둘 다
  우연히 `dev`라 발동하지 않았을 뿐, `--infisical-env pod`으로 띄우는 순간 **가드가
  파드가 쓰지 않을 환경을 검사**한다. 그리고 그 분리는 가드 자신이 권하는 해법이므로,
  조언을 따르는 순간 가드가 눈을 감는 구조였다. `env`를 인자로 넘기도록 고쳤다.
  이것이 이 저장소의 "검사는 돌고 통과하는데 검사 대상이 실제 대상이 아니다" 패턴의
  아홉 번째이며, 처음으로 보안 가드에서 나왔다.

- 2026-08-02 **가드를 실물에 물려보니 두 가지가 더 나왔다** (`scripts/orchestrate.py`,
  레인 C). 둘 다 스텁으로는 드러나지 않았고 실제 Infisical 환경 두 개에 대고 돌려서
  나왔다. 이 저장소의 "검사가 옳은 질문을 던지는데 잘못된 양을 재고 있다" 패턴이다.
  - **프로브가 시크릿이 아니라 자식 프로세스의 환경변수를 셌다.** `clean` 세 개만
    빼고 있었는데 OS가 자식에 붙이는 것(macOS의 `LC_CTYPE`,
    `__CF_USER_TEXT_ENCODING`)이 남아, 시크릿 1개짜리 올바른 `pod` 환경이 **3개를
    가졌다는 이유로 거부**됐다. Infisical 자신은 "Injecting 1 secret"이라고 말하고
    있었다. 이제 같은 sanitised 환경에서 `infisical run` 있이/없이 두 번 돌려 그
    **차집합**을 취한다 — 주입 전후의 차이가 곧 주입된 것이다. 금지 목록일 때는 로케일
    변수가 목록에 없어 조용히 통과했으므로, 이 결함을 드러낸 것은 허용 목록 전환이다.
  - **문서화된 실행 방식이 파드에 dev 바인딩 토큰을 넘기고 있었다.** `infisical_token()`이
    주변 `INFISICAL_TOKEN`을 우선했는데, 이 저장소가 문서화한 실행 방식
    (`infisical run --env=dev -- python scripts/orchestrate.py`)이 바로 그 자리에
    dev 저장 서비스 토큰을 넣는다. 그 토큰은 `--env`를 **무시한다**(실측: `dev`·`pod`·
    존재하지 않는 환경 모두에 같은 26개). 즉 환경을 분리해도 파드는 dev 전체를 읽을 수
    있는 토큰을 받았다. 이제 유니버설 인증으로 발급하며, 명시적으로 넘기려면
    `TRAINBENCH_POD_INFISICAL_TOKEN`을 쓴다 — `INFISICAL_TOKEN`은 호출자의 것이다.
    추가로 **그 속성을 가정하지 않고 잰다**: 존재할 수 없는 환경을 요청해 답이 돌아오면
    바인딩된 토큰이므로 거부한다. 이 검사가 없으면 `dev`가 정리되는 날 바인딩 결함이
    조용히 통과한다(지금은 초과 시크릿 때문에 우연히 걸릴 뿐이다).

- 2026-08-02 **레인 경계 침범 기록 (`optim=muon`, 레인 D).** 위 2026-08-01 레인 A
  항목과 같은 형식으로 남긴다. 세 파일이 D 소유가 아니다.
  - `pyproject.toml`, 루트 `uv.lock` (레인 F 소유) — `native` extra에
    `pytorch-optimizer>=3.10` 추가. `envs/native/pyproject.toml`이 이미 이 축을 위해
    같은 배포판을 고정하고 있고, 이 줄은 루트 락이 같은 버전을 해석하게 하는 용도다.
    **결정이 필요한 것이 하나 남아 있고 그것은 F의 몫이다**: 이 배포판은 `native`에만
    있어서 문서화된 셋업 명령(`uv sync --extra compose`)도, 6개 프레임워크 이미지 중
    5개도 설치하지 않는다. `peft`가 똑같은 모양이며 그쪽은 테스트가 무조건 import한다
    (선행 구멍이고 이 레인이 만든 것이 아니다). D가 한 조치는 `axes._optimizer`가
    import 실패를 `UnappliedAxis`로 감싸는 것까지다 — 없는 환경에서 축이 **거부**되지
    `assemble` 중간에 `ModuleNotFoundError`로 죽지 않는다. extra를 옮길지는 이미지
    해석에 영향을 주므로 F가 정한다.
  - `docs/methodology.md` (레인 E 소유) — §5 "Muon이 무엇을 최적화하는가". 코드 주석에
    적을 수 없는 것이라 여기 있다. 이번에 추가한 것은 lr 한정 조건(config의 `lr: 1e-5`가
    AdamW의 값 그대로이고 라이브러리 기본값은 `lr=0.02`/`adamw_lr=3e-4`로 두 경로를
    분리한다 — throughput은 무관하지만 **수렴 곡선은 이 config로 못 잰다**), 축이
    거부되는 세 조건, 그리고 "가중치가 움직였다"가 근거가 아닌 이유다.
- 2026-08-02 **`_capture_optim`이 `use_muon` 분할을 기록한다** (`trainbench/applied.py`,
  레인 D — §1에서 capture 함수는 D 소유지만 파일이 공유이므로 계약 변경으로 올린다).
  **레코드 스키마가 넓어진다**: `optim.name`의 detail에 `newton_schulz_tensors`(Newton-Schulz를
  통과하는 학습 가능 텐서 수)와 `use_muon`(그룹별 bool)이 추가되고, `newton_schulz_tensors == 0`이면
  `applied`가 `"muon"`이 아니라 **`None`(undetermined)** 이다. 기존 필드
  (`class`/`fused`/`param_groups`)와 AdamW 경로의 동작은 그대로다.
  이유: `use_muon`은 param group 플래그이고 클래스 이름은 어느 쪽이든 `Muon`이라,
  모든 그룹을 `use_muon=False`로 빌드한 런 — 즉 **Muon 이름을 단 AdamW** — 의
  레코드가 정직한 런과 바이트 단위로 같았다(`{'class': 'Muon', 'fused': False,
  'param_groups': 2}`). 발행된 수치를 사후에 어느 옵티마이저의 것으로도 귀속할 수
  없다는 뜻이다. 이 저장소가 아홉 번 낸 "검사는 통과하는데 검사한 것이 없다"의
  optim 판이며, 판정서가 머지 전 최소 조치로 지목했다.
  **다른 레인이 알아야 할 것**: `optim=muon` 런의 결과 JSON에 필드 두 개가 늘어난다.
  `scripts/report.py`가 detail을 열거하고 있다면 새 키를 만난다.

- 2026-08-02 **`--infisical-env` 기본값이 `dev`에서 `pod`으로 바뀐다**
  (`scripts/orchestrate.py`의 `POD_INFISICAL_ENV`, 레인 C). 이 인자가 이름하는 것은
  **파드가 읽을 환경**이지 오케스트레이터가 읽을 환경이 아니다 — 오케스트레이터의
  시크릿은 이 인자가 아니라 자기를 감싼 `infisical run --env=dev`에서 온다. 따라서
  올바른 기본값은 처음부터 파드 환경이었고 `dev`는 파드 환경이 없던 시절의 잔재다.
  **어느 쪽 기본값이든 fail-closed다** — 스코프 검사가 보증 못 하는 것을 거부하므로.
  다른 것은 어느 실수가 조용한가다: `dev`가 기본이면 플래그를 잊었을 때 캠페인이
  멈추고, `pod`이 기본이면 잊었을 때 옳게 동작한다.
  기본값이 된 이상 문서의 실행 예시에서 이 플래그를 빼야 한다 — 예시마다 기본값을
  반복하면 나중에 기본값이 바뀌어도 아무도 눈치채지 못한다. `AGENTS.md`를 그렇게
  정리했다. 고정 테스트는 `test_a_launch_with_no_flag_hands_the_pod_its_own_environment`
  이며, 리터럴 `"pod"`을 단언한다(`POD_INFISICAL_ENV`를 단언하면 상수를 `dev`로
  되돌려도 통과한다).

- 2026-08-02 **레인 경계 침범 기록 (`dataloader` 축, 레인 D).** 판정서 F9가 지적한
  누락분(이전 라운드 3건)과 이번 라운드 1건을 함께 남긴다.
  - `scripts/audit_plan.py`, `tests/test_applied.py`, `trainbench/embedding.py`
    (이전 라운드) — `embedding.py`는 `packed_last_token_pool` 추가(레인 B 소유 파일,
    packed 배치는 `last_token_pool`의 계약을 정면으로 깬다), 나머지 둘은 새 축의
    등재·고정이다.
  - `scripts/audit_plan.py` (이번 라운드) — `AXIS_NEEDS_NOTHING`에
    `dataloader/torch_packed_pretokenized` 한 줄. 신규 config 값은 분류되지 않으면
    `axis-packages`가 실패하므로, config 추가와 같은 커밋에 들어가야 한다.
  **다른 레인이 알아야 할 것 1 — `configs/dataloader/torch_packed_pretokenized.yaml`이
  생겼다.** `packing`이 토크나이저 없이 동작하는 유일한 조합(패딩 없는 행별 ids)이
  `pretokenize=true`이고, 그 조합을 표현하는 config가 없었다(AGENTS.md: 새 변형은
  config 조합에서 나온다). `axis-values`의 분모가 45 -> 46이 된다.
  **알아야 할 것 2 — `axes.PackedCollate`의 시그니처가 `(tokenize, pad_id)`로 넓어졌고,
  `tokenize`를 주면 `pad_id`가 필수다.** 하네스가 프로세서로 배치를 토크나이즈하는
  경로가 이것 하나뿐인데, 배치 토크나이저는 기본이 패딩이다. 패킹된 PAD는 tokens/s가
  실토큰으로 세고 `packed_last_token_pool`이 어떤 시퀀스의 임베딩으로 읽는 반면
  `dataloader.packing=True` 인증은 그대로 나간다 — 즉 죽는 런이 아니라 잘못 라벨된
  숫자다. 이제 세 지점에서 **행동으로** 막는다: `tokenize`가 2-D 텐서(=패딩된 배치)를
  돌려주면 거부, 시퀀스에 `pad_id`가 하나라도 있으면 거부, 행이 `attention_mask`를
  들고 오면 그 마스크를 읽어 0이 있으면 거부. pad id와 eos id가 같은 체크포인트는 이
  방식으로 검사할 수 없고 그때는 거부가 정답이다 — `pretokenize`로 행마다 따로
  토크나이즈하면 패딩이 애초에 쓰이지 않는다.
- 2026-08-02 **`axis-values`가 dataloader 축에 대해 vacuous였다** (`audit_plan.py`,
  `tests/test_audit.py`). 레인 D가 넘긴 것을 감사 레인에서 확인하고 고쳤다.
  이 체크는 `axes.assemble(...)`을 **dataset 없이** 불렀고, `axes._dataloader`는
  `if dataset is None`에서 packing/pretokenize에 닿기 전에 반환한다. 그래서
  `PackedCollate.__call__`을 `raise NotImplementedError`로 갈아버려도 출력이 바이트
  단위로 같았다.
  - **양방향으로 틀렸다.** 거짓 양성만이 아니다 — `loss/cached_mnrl`(GradCache)은
    구현이 있는데도 inert로 보고되고 있었다. 거부 사유가 "이 런의 dataset이 None"
    이었기 때문이다. 감사가 자기 fixture를 안 준 탓에 멀쩡한 축이 미구현으로 보였다.
  - 고친 내용: 합성 dataset(`_AxisValueRows`)을 넘기고, **배치를 한 개 뽑는다.**
    dataset만 넘기면 절반만 닫힌다 — collate는 배치를 뽑기 전에는 호출되지 않고
    packing은 전부 collate 안에 있다. `dataloader.pretokenize=true`면
    `scripts/bench.py`가 하는 대로 `axes.pretokenize`를 먼저 부른다(`_dataloader`는
    토큰 ids가 있는지 **보기만** 하지 만들지 않으므로, 그러지 않으면 이미 토크나이즈된
    fixture가 축을 대신 인증한다).
  - 모든 variant에 `data.num_workers=0`을 건다. 워커 수는 런의 속성이지 축이
    적용되는지의 속성이 아니고, `configs/data/*.yaml`은 8을 요구한다.
  - **여전히 증명하지 않는 것**: packing이 *올바른지*. 잘못 이어붙이는 collate도 이
    체크에는 적용된 것으로 보인다. 등가성은 `tests/test_axes.py`와 capture probe의
    질문이다.
- 2026-08-02 **레인 경계 침범 기록 (`loss`/`parallel.cross_device_negatives` 축, 레인 D).**
  판정서 발견 8이 지적한 이전 라운드 누락분이다. 두 파일 모두 §1의 "공유(수정 금지)"에
  있고, 변경 자체는 축 등재이므로 config·구현과 같은 커밋에 들어가야 한다.
  - `scripts/audit_plan.py` — `AXIS_PACKAGES`에서 `loss/cached_mnrl`을 빼고
    `AXIS_NEEDS_NOTHING`으로 옮겼다(`axes._loss`가 `gradcache` 라이브러리를 import하지
    않고 `probe/steps.py::encode` + `embedding.py::info_nce`로 직접 구현한다. 그 패키지는
    `envs/native` 락에만 있어 import하면 나머지 이미지에서 ImportError가 된다).
    `parallel/single_cross_device`도 같은 목록에 넣었다 — all-gather는 torch.distributed
    자신의 것이라 설치되는 것이 없다.
  - `tests/test_applied.py` — `UNIMPLEMENTED_AXES`에서 `loss.name: cached_mnrl`과
    `parallel.cross_device_negatives: True` 두 줄을 지웠다(구현됐으므로). 남은 거부는
    무조건이 아니라 조건부이고 `tests/test_axes.py`로 옮겼다.
- 2026-08-02 **다른 레인이 알아야 할 것 — `loss=cached_mnrl`이 `assemble`에서
  데이터에 따라 거부된다** (`trainbench/axes.py`, 레인 D. 소유 파일이라 계약 변경은
  아니고, 두 레인의 fixture에 닿으므로 남긴다).
  판정서 발견 5: 이미지가 섞인 MMEB 서브셋에서는 `_split_rows`가 매 배치를 거부하므로
  이 축은 **한 배치도 못 돈다**. 그런데도 `assemble`은 적용됐다고 이름을 돌려주고
  `assert_matches`가 통과했다 — 스텝 1에서 죽을 런을 그 전에 "적용 가능"으로 세고
  있었다. 이제 `axes._gradcache_needs_splittable_data(dataset)`가 `assemble`에서
  먼저 거부한다: 행에 이미지가 있으면(`datasets.Image` 피처 선언 또는
  `qry_image`/`pos_image` 값이 실제로 있는 행) 거부, dataset이 없거나 컬럼을 읽을 수
  없으면(=모르면) 거부.
  - **하네스 레인(G)**: `scripts/bench.py::PairDataset`이 진짜 서브셋 행을 담으면
    거부된다. 이것이 의도다 — 그 데이터에서 GradCache는 측정 불가다.
    `tests/test_smoke_cpu.py`의 `rows()`는 이미지 값이 `None`이라 그대로 통과한다
    (컬럼 이름이 아니라 **행의 값**을 본다. 20개 MMEB config 중 4개는 `qry_image`가,
    13개는 `pos_image`가 없으므로 이름만으로 거부하면 텍스트 전용 draw까지 막힌다).
  - **감사 레인**: `axis-values`의 `_AxisValueRows`는 텍스트 전용이라 이 값은 여전히
    applicable로 센다. 그 숫자는 "axes.py가 적용할 수 있다"이지 "이 연구의 측정 런이
    켤 수 있다"가 아니다. 후자는 지금 **거짓**이며 `configs/loss/cached_mnrl.yaml`
    주석에 적어뒀다. 두 질문을 한 숫자로 읽지 않도록 하는 것이 남은 과제다.

- 2026-08-02 **`doc-commands`의 설치 확인이 문자열 검사에서 import 검사로 바뀐다**
  (`audit_plan.py`, `tests/test_audit.py`). optim(muon) 레인이 넘긴 판정서 #4를
  감사 레인에서 확인하고 고쳤다.
  이 체크는 `uv sync` 줄이 `--extra compose`를 달았는지만 정규식으로 봤고, 근거가
  "tests import hydra"였다. hydra가 테스트의 유일한 의존이 아니게 된 지 오래인데
  문구는 `install what the tests need`라고 주장하고 있었다.
  - **놓친 것 3건**: `peft`, `datasets`, `transformers`가 전부 root `native` extra에만
    있고 문서화된 명령 중 `--extra native`를 쓰는 것이 없다. 제보는 `peft` 하나였으나
    import를 전수 수집하니 3건이었다.
  - **제보의 메커니즘 서술은 정정한다.** "`tests/test_axes.py`가 `peft`를 무조건
    import해 collection에서 실패한다"가 아니다. 세 건 다 **함수 안** import이며
    (`test_applied.py:570`, `test_axes.py:2119`, `test_axes.py:1318`), collection은
    통과하고 해당 테스트만 `ImportError`로 죽는다. `importorskip`이나 try/except로
    감싸여 있지도 않으므로 skip이 아니라 error다. 결론(문서대로 설치하면 스위트가
    녹색이 아니다)은 그대로다.
  - 새 방식: `tests/`의 서드파티 top-level import를 AST로 전수 수집하고, 설치
    메타데이터로 배포판 이름을 얻어(`import yaml` -> `pyyaml`, `import hydra` ->
    `hydra-core`. 소스 어디에도 안 적혀 있다), 문서화된 `uv sync`가 만드는 lock
    (`uv export --frozen`)에 그 배포판이 있는지 본다. **함수 안 import도 센다** —
    질문이 "collection을 넘기는가"가 아니라 "문서대로 설치하면 스위트가 도는가"이고,
    실제로 3건 다 함수 안이었다. 손으로 적은 목록은 잊을 목록이다.
  - `--extra compose` 정규식은 **삭제했다.** 새 체크가 포함한다 — extra가 빠지면
    hydra가 lock에 없어 그대로 잡힌다.
  **이 체크는 지금 실패하며, 해소는 감사 레인 몫이 아니다.** 둘 중 하나다:
  (E) `README.md:22` / `AGENTS.md:15`의 명령을 `uv sync --extra compose --extra native`
  로 고치거나, (F) `native` extra의 해당 패키지를 `compose`로 옮긴다. 둘 다 확인했고
  전자는 실측으로 통과시켰다(`--all-extras`도 통과). baseline 항목 추가는 계약 위반
  이므로(§1 "새 실패는 baseline이 아니라 수정으로 해소한다") 항목을 넣지 않았다.

- 2026-08-02 **`axis-values`가 데이터 모양에 따라 갈리는 값을 하나로 뭉개고 있었다**
  (`audit_plan.py`, `tests/test_audit.py`). loss 레인 제보(판정서 발견 5).
  `loss=cached_mnrl`은 `axes._split_rows`가 `pixel_values`를 실은 배치를 전부 거부하고,
  이 연구의 두 서브셋은 모두 이미지를 싣는 MMEB 드로우다. 즉 **구현은 있으나 이 연구가
  설정한 어떤 런에서도 켤 수 없다.** 그런데 감사가 텍스트 전용 fixture 하나만 쓰고 있어
  `loss 2/2`로 셌고, 그 숫자는 "GradCache 측정 준비됨"으로 읽힌다.
  - **감사 레인의 이전 보고를 정정한다.** 직전 라운드에 `loss` 축의 3->2 이동을
    "진짜 개선"이라고 보고했고 baseline note가 그 문장 위에 쓰였다. 절반만 맞았다 —
    dataset이 None이라 거부되던 것은 감사의 결함이 맞지만, 이미지 데이터에서의 거부는
    **사실**이다. note에 정정을 남기고 count를 회수했다.
  - 수리: variant마다 **행이 이미지를 싣는지만 다른 두 fixture**로 시도하고, 한쪽에서만
    통과하는 값은 `applicable`로 세지 않고 `data-dependent`로 이름을 부른다. 두 질문을
    가르는 방법으로 fixture를 이미지 쪽으로 **교체**하지 않은 이유는 그러면 같은 fixture를
    쓰는 dataloader 축의 의미가 함께 바뀌기 때문이다(제보가 지적한 그대로). 두 개를 두고
    **차이를 관측**하면 "이 값은 데이터에 의존한다"가 선언이 아니라 발견이 된다 — MMEB가
    이미지를 싣는다는 지식을 체크에 하드코딩하지 않아도 된다.
  - 실측: 두 fixture가 실제로 갈라놓는 값은 `loss/cached_mnrl` 하나뿐이다. dataloader 4종
    (torch / packed / pretokenized / packed_pretokenized)과 optim·peft는 양쪽 동일했다.
    fixture의 이미지 값은 `None`이 아니라 텐서다 — `image_columns`가 `None`을 "컬럼이 비었다"로
    읽고 torch 기본 collate가 스택하지 못해, 두 fixture가 두 가지로 달라져 버린다.
  - baseline `axis-values` 2->4는 **전부 이 레인 몫이고 퇴화가 아니다.** loss가 1/2로
    돌아오고(+1 inert) cached_mnrl이 data-dependent로 잡힌다(+1). 이전 2가 실제보다
    낙관적이었던 것이다.

- 2026-08-02 **`loss=cached_mnrl`이 이미지 배치에서 돈다 — 행->픽셀 매핑**
  (`trainbench/axes.py`, `scripts/bench.py`, loss 레인). 위 두 항목이 기록한
  "이 연구가 설정한 어떤 런에도 적용되지 않는다"를 해소한다.
  - **실측이 근거다.** 세 체크포인트의 프로세서를 실제로 돌려 텐서 이름과 leading
    dim이 무엇을 세는지 읽었다(transformers 5.14.1, 행별 이미지 수를 1/1/1/1,
    1/0/2/1, 2/1로 바꿔가며). Qwen 두 모델은 `Qwen3VLProcessor`로 같고
    `pixel_values`가 **패치**를(grid `[1,4,4] [1,6,8] [1,8,10] [1,12,12]` -> `(288,
    1536)`), `image_grid_thw`가 **이미지**를 센다. gemma-4는 `Gemma4Processor`이고
    `pixel_values` `(images, 2520, 768)`, `image_position_ids` `(images, 2520, 2)`로
    둘 다 이미지를 센다. `mm_token_type_ids`는 셋 다 `(rows, seq)`라 행 정렬이다.
    `axes.IMAGE_PAYLOAD_KEYS` 위 주석이 이 표를 담고 있다.
  - **계약 변경 1건**: `built.loss_fn.gradcache_backward(...)`가 키워드 인자
    `images_per_row`를 받는다. `scripts/bench.py::MicroBatch`에 같은 이름의 필드가
    생겼고 `Collate`가 채운다(행 순서 = 배치 행 순서 = 쿼리 전부, 그다음 positive
    전부). 넘기지 않으면 픽셀을 실은 배치는 **종전대로 거부된다** — 이 작업은 거부를
    없앤 것이 아니라 좁힌 것이다.
  - **여전히 거부하는 것**: 귀속 근거가 없는 텐서(행 수도 아니고 이 배치의
    `images_per_row`가 설명하는 수도 아닌 leading dim), 텐서가 아닌 값, 그리고
    `image_columns(dataset)`가 `None`인 dataset(컬럼을 선언하지 않거나 행이 매핑이
    아니어서 collate가 행별 이미지 수를 셀 수 없는 경우).
  - **같이 고친 것 — `Collate`가 프로세서에 이미지를 행별로 묶어 넘긴다.** 읽다가
    실측으로 확인했다: `Gemma4Processor`는 평평한 리스트를 **거부한다**
    (`make_nested_list_of_images`가 한 행의 이미지로 읽고 `validate_inputs`가
    "Received inconsistently sized batches of images (1) and text (4)"로 raise).
    행마다 이미지가 하나씩일 때도 거부한다. 즉 **gemma-4는 이미지가 실린 배치를
    지금까지 하나도 만들 수 없었다** — GradCache만의 문제가 아니라 모든 loss가
    그렇다. Qwen 두 프로세서는 두 형태를 다 받고 행당 1장인 경우 텐서가 바이트 단위로
    같음을 확인했으므로, 묶은 형태가 셋 다 받는 유일한 모양이다. 묶는 데 쓰는 벡터는
    `images_per_row` 그 자체다 — 지도를 두 벌 두지 않으려는 것이고, 그래서 잘못된
    벡터는 이제 프로세서 자신이 거부하는 배치가 된다.
    `tests/test_smoke_cpu.py::FakeProcessor`도 같은 등식을 검사하도록 바꿨다.
  - **다른 레인이 알아야 할 것**: `axes.image_columns`의 *어느* 컬럼이 이미지인지를
    가리는 부분은 이제 어떤 런의 동작도 결정하지 않는다. `None`인지 아닌지만 본다.
    그 함수의 `datasets.Image` 피처 경로는 `scripts/audit_plan.py`의 두 fixture를
    가르는 데만 쓰이므로, 정리는 감사 레인 몫이다.
  - **레인 경계 침범 2건**(§1 "공유", 축 등재라 같은 커밋에 들어간다):
    `tests/test_audit.py` — `test_a_value_applicable_to_only_one_data_shape_is_named_not_counted`가
    실물 대신 `_gradcache_needs_splittable_data`를 monkeypatch해 data-dependent
    케이스를 **만들어** 검사하도록 바꿨다(살아 있는 사례가 없어졌기 때문이고, 지우면
    그 분기가 무검사로 남는다). 반대 방향을 보는
    `test_gradcache_is_counted_applicable_on_both_data_shapes`를 추가했다.
    `configs/loss/cached_mnrl.yaml` — 주석이 "두 서브셋 모두에서 거부된다"고
    적혀 있어 거짓이 됐으므로 고쳤다.
  - baseline `axis-values` 4->2 (loss 1/2 -> 2/2, 31/46 -> 32/46). note에 적었다.
  - **GPU 없이 확인 못 한 것**: 조각 하나가 실제 체크포인트에 `pixel_values` 없이
    들어갈 때(그 조각의 행이 이미지를 하나도 안 실은 경우) 무엇이 일어나는지는
    측정 안 함이다. 텍스트 전용 배치와 같은 모양이라 받아들여질 것이라는 판단으로
    빈 이미지 텐서를 조각에서 **뺐다**. 실제 모델에서의 확인은 첫 pod 몫이다.
    GradCache의 메모리 절감량과 오버헤드 자체도 측정 안 함이다 — 이 저장소에 GPU가
    없고, 여기서 잰 것은 autograd의 saved tensor 개수(테스트 텐서 기준)뿐이다.
  - **다른 레인에 넘기는 제보 — gemma-4의 `tokens_per_image` 280이 실측과 다르다.**
    `docs/model-spec.yaml`은 `processor_config.json`의 `image_seq_length: 280`을
    적어뒀지만, transformers 5.14.1의 `Gemma4Processor`는 이미지마다
    `num_soft_tokens_per_image`를 종횡비에서 계산하고 280은 `max_soft_tokens`
    상한이다. 실측(2026-08-02): 64x64 -> 256, 768x256 -> 252, 1024x1024 -> 256.
    `trainbench/probe/steps.py::visual_token_count`는 선언값과 다르면 raise하므로
    gemma-4 probe는 이 지점에서 죽는다. `docs/model-spec.yaml`도
    `probe/steps.py`도 이 레인 소유가 아니라 손대지 않았다.

- 2026-08-02 **`model.tokens_per_image`가 `model.max_tokens_per_image`로 바뀐다**
  (`config_schema.py` 필드 이름·의미 변경 + 검증기 1건, 위 제보를 받은 레인).
  값이 아니라 **의미가** 틀렸다. `processor_config.json`의 `image_seq_length: 280`은
  이미지당 토큰 수가 아니라 `max_soft_tokens` 상한이고, 그나마
  `Gemma4Processor`는 `self.image_seq_length`를 생성자에서 대입만 하고 **어디서도
  읽지 않는다**(transformers 5.14.1). 실제 확장은
  `replace_image_token`이 이미지 프로세서의 `num_soft_tokens_per_image`를 읽어
  하고, 그 수는 `get_aspect_ratio_preserving_size`가 종횡비에서 계산한
  `(높이/48) * (너비/48)`이다. 재현 실측(이 레인, transformers 5.14.1):
  448x448 -> 256, 768x256 -> 252, 1024x1024 -> 256, 1280x720 -> 264,
  960x672 -> **280**. 16px 격자로 4096px까지 쓸면 138종의 값이 나오고 최댓값이 280이다
  — 즉 280은 도달 가능하지만 특정 종횡비에서만이고, 정사각형은 256이 최대다.
  `PROBE_IMAGE_SIZE`가 448x448이므로 gemma-4 probe는 항상 256을 재고, 280과의
  **일치**를 요구하던 `visual_token_count`는 올바른 측정에 대고 죽고 있었다.
  **바뀐 것 3가지**:
  - 스키마 필드 `model.tokens_per_image` -> `model.max_tokens_per_image`
    (`configs/model/*.yaml` 3개, `docs/model-spec.yaml` 양쪽 키 동시 변경 —
    `audit_plan.py`의 `model-spec`이 값 대 값으로 대조하므로 한쪽만 고치면 막힌다).
    이름을 그대로 두고 비교만 바꾸는 선택지도 있었지만, `tokens_per_image: 280`이라는
    **이름 자체가** 이번 오독을 초래했다(컨벤션 01: 의미를 반영하는 이름).
  - 검증기 `_fixed_image_tokens_are_gemma4_only` ->
    `_an_image_token_cap_is_gemma4_only`. 방향(gemma4만 선언, 나머지는 금지)은 그대로다
    — gemma-4만 토큰 상한을 선언하고 Qwen 둘은 픽셀 범위만 선언한다는 사실이 바뀌지
    않았기 때문이다. 에러 문구가 바뀌므로 `tests/test_config.py`의 match 문자열도 바뀐다.
  - `visual_token_count`의 4번째 게이트가 `count != 선언값`에서
    `count > 상한`으로 바뀐다. **게이트 수는 줄지 않았고 나머지 셋은 그대로다.**
    `null`(Qwen)일 때의 동작도 그대로다. 완화 방향이므로 pad-id 게이트가 상한 아래의
    pad 수를 여전히 잡는지 테스트에 상한을 붙여 고정했다.
  **레코드 스키마가 좁게 바뀐다**: probe의 `visual_tokens` detail에서
  `declared_tokens_per_image` -> `declared_max_tokens_per_image`. 값 자체
  (`visual_tokens_per_image`, `visual_tokens_per_sample`)는 그대로다.
  **경계 침범**: `config_schema.py`·`docs/model-spec.yaml`·`tests/{test_config,test_applied}.py`는
  §1의 "공유(수정 금지)", `docs/model-spec.md`는 레인 E 소유다. 필드 이름 변경은
  한 커밋에 다 들어가지 않으면 어느 쪽도 합성되지 않으므로 나눌 수 없다.

- 2026-08-02 **파드가 첫 측정 전에 자기 계획을 검사한다 — `bench.py --preflight`**
  (`scripts/bench.py`, `docker/entrypoint.sh`, 둘 다 레인 G 소유. 소유 파일이지만
  **이미지 재빌드가 필요하고** 다른 레인의 fixture에 닿으므로 남긴다).
  `configs/experiment/phase2-loss-qwen3_5_0_8b.yaml`이 `kernel`을 오버라이드하지 않아
  두 setting 모두 `axes.patch()`에서 거부되는 파드가 매니페스트로 커밋돼 있었다.
  이 질문을 `audit_plan.py`에 넣으려던 시도는 실측으로 무산됐다 — 27개 계획 런을
  감사 호스트에서 통과시키면 `kernel=fla` 5건이 거부되고(호스트에 fla 부재) 실제로
  죽는 `kernel=none`은 통과한다. **답이 이미지의 내용물이라 파드만 답할 수 있다.**
  - `scripts/bench.py --preflight <plan>`: 계획의 모든 항목을 `to_bench_config` ->
    `patch`/`load_kwargs`/`step_context`에 통과시키고 하나라도 거부되면
    `PREFLIGHT_EXIT=4`. `--config`/`--out`은 이제 required가 아니며, 둘 다 없고
    `--preflight`도 없으면 `parser.error`다. **`assemble`/`assert_matches`는 모델이
    필요하므로 빠져 있다** — 통과한 계획도 setting별로 거부될 수 있다.
  - `docker/entrypoint.sh`가 스윕 루프 전에 한 번 부른다. 거부되면 모든 setting이
    `preflight refused ...` 사유와 함께 발행되고 `timeout`은 한 번도 호출되지 않는다.
    개별 항목에 config가 없는 경우는 거부가 아니라 "검사하지 않음"이다 — 그 setting
    하나만 막는 기존 동작(`test_a_plan_item_without_a_resolved_config_stops_only_that_setting`)을
    뒤집지 않기 위해서다.
  - **다른 레인이 알아야 할 것**: `tests/test_pods.py`의 `Sweep` 네임드튜플에
    `preflight` 필드가 생겼고(`Sweep(proc, bench, preflight, uploads, calls)`),
    `sweep_pod(..., pod_image=True)`가 기본이다. 스윕 테스트를 추가하는 레인은
    `pod_image=False`가 곧 "fla도 CUDA도 없는 호스트"이며 canonical baseline이
    거부된다는 뜻임을 알아야 한다. FAKE_BENCH는 `--preflight`를 스텁하지 않고
    저장소의 진짜 `bench.py`를 로드해 실행한다.
  - **같은 프리플라이트가 GPU도 본다** (2026-08-02, 이미지 레인 `ed5b323`의 인계).
    그 커밋이 CUDA arch를 `80;90;100`으로 좁히고 `TRAINBENCH_CUDA_ARCHS`를 이미지에
    넣었으나 읽는 쪽이 없었다. 이제 `bench.py`의 `gpu_refusal`이 읽는다.
    - 값은 `nvidia-smi`가 아니라 `torch.cuda.get_device_capability()`에서 온다.
      변환은 `device_arch()` **한 곳**이고 규칙은 torch에서 읽었다:
      `_get_cuda_arch_flags()`가 capability를 `f"{major}{minor}"`로 만들어
      `code=sm_XX`를 짓는다(`torch/utils/cpp_extension.py`, 2.13.0). 이 저장소를
      개발하는 호스트에 NVIDIA GPU가 없어 `nvidia-smi`의 출력 형식은 **실측하지
      못했고**, 그래서 쓰지 않았다.
    - **변수가 없으면 거부한다(fail-closed).** 그 `ENV`와 이 검사를 부르는
      `entrypoint.sh`를 이미지에 넣는 것이 같은 파일(`Dockerfile.framework`)이므로,
      검사를 갖고 변수를 갖지 않는 이미지는 이 저장소가 만들 수 없다. 다른 프레임워크
      이미지도 예외가 아니다 — `ENV`는 `ARG FRAMEWORK`와 무관하게 걸린다.
    - **다른 레인이 알아야 할 것**: `sweep_pod`에 `gpu_arch="80"` /
      `cuda_archs="80;90;100"` 인자가 생겼다. 스윕 테스트는 이제 GPU를 가진 파드를
      기본으로 흉내 내고, `gpu_arch`를 바꾸면 목록 밖 GPU가 된다.
      `tests/test_smoke_cpu.py`의 `pod_gpu` fixture가 단위 테스트 쪽 대응물이다.
    - `docs/support-matrix.md`(레인 E 소유)의 "읽는 코드가 없다" 두 곳을 갱신했다.
  - 근거와 한계는 `docs/methodology.md` §7(레인 E 소유 파일, 경계 침범).
    같은 절에서 §4의 "미해소 위험"에 해소 표시를 달았다 — 그 절은 미검증이라고 적힌
    채였고, 저장소는 이미 그 시나리오가 사실이라는 전제로 `_baselines.yaml`을
    고친 상태였다.

- 2026-08-02 **`pods.observe`가 문자열이 아니라 `Reading`을 돌려주고, 재시작을
  판정한다** (`trainbench/pods.py`, 레인 C 소유. **원장 스키마가 넓어지므로** 남긴다).
  첫 A100 카나리에서 컨테이너가 17초마다 40번 재시작하는 10분 내내 `observe()`가
  `running`을 돌려줬고 `orchestrate`는 오지 않을 결과를 데드라인까지 기다렸다.
  RunPod은 재시작 횟수를 **어디에도 내주지 않는다**(실계정 실측: REST `Pod` 스키마
  32개 속성에 없음, GraphQL `Pod`/`PodRuntime`에 후보 이름 전부 부재,
  `Pod.uptimeSeconds`는 50대 전부 0). 유일한 관측 가능한 것은
  `runtime.uptimeInSeconds`이고, 실행 중인 21대로 그것이 **컨테이너 시계**임을
  확인했다 — 자세한 수치는 `pods.py` 모듈 docstring.
  - `observe(pod_id, get_pod, previous_uptime=None) -> Reading(status, uptime_seconds)`.
    시계가 `UPTIME_JITTER_SECONDS`(1초) 넘게 떨어지면 새 상태 `RESTARTING`.
  - `PodWatch`가 시계를 pod별로 들고 있고, 재시작 한 번에 감시가 끝난다
    (`REASON_RESTARTED`). 봐주지 않는 이유는 entrypoint가 계획을 처음부터 다시 돌려
    이미 발행한 setting을 다시 발행하고 다시 과금하기 때문이다.
  - **원장 스키마**: `PodOutcome.to_dict()`에 `uptime_seconds`가 추가된다.
    `scripts/orchestrate.py`가 `entries[...]["outcome"]`에 그대로 넣으므로 원장
    JSON에 필드 하나가 는다. 17초짜리 컨테이너와 40분짜리 컨테이너를 구별하는 것이
    이 필드의 용도다.
  - **미확인으로 남는 것**: 컨테이너가 exit 0으로 정상 종료한 뒤에도 RunPod이
    재시작하는지는 재지 못했다(카나리 파드는 사라졌고 새 파드를 만들지 않았다).
    그렇다면 성공한 파드도 `restarted`로 감시가 끝나며, 어느 쪽이든 데드라인까지
    기다리는 것보다 낫다. 첫 캠페인의 `uptime_seconds`가 바로 답한다.

- 2026-08-02 **`doc-commands`의 import 수집이 `tests/`에서 패키지와 엔트리포인트까지
  넓어진다** (`audit_plan.py`, `tests/test_audit.py`. optim 레인, 판정서
  `optim-muon-silently-absent-from-the-documented-setup`).
  위 2026-08-02 항목이 세운 새 방식은 `tests/`만 훑었고, 그 답이 더 넓은 질문의
  답으로 쓰였다. `optim=muon`의 `pytorch-optimizer` import는 `trainbench/axes.py`
  안 함수 레벨에 단 하나 있고 어떤 테스트도 그것을 import하지 않는다 — 그래서 어떤
  스캔에도 안 잡혔고, clean clone은 optim 축이 스스로 거부하는 ablation을 받는
  동안 이 체크는 계속 "문서화된 명령이 필요한 것을 전부 설치한다"고 보고했다.
  - `_test_imports` -> `_demanded_imports`. 수집 대상이 `tests/` + `trainbench/` +
    `scripts/`이고, 보고에 붙는 출처가 파일명이 아니라 저장소 상대 경로가 된다.
  - **프레임워크 프로브 어댑터는 제외한다.** `_per_image_adapters()`가 `envs/`의
    디렉터리 이름에서 `trainbench/probe/<framework>.py`를 유도한다. 각 어댑터는
    자기 이미지 안에서만 import되고(`probe/registry.py`) `envs/<framework>/uv.lock`이
    핀하므로, 루트 lock에 `unsloth`를 요구하면 결함이 아닌 것을 결함으로 만든다.
    대가는 어댑터 안에 새로 생기는 지연 import를 이 체크가 못 본다는 것이고,
    결과 문자열이 몇 개를 건너뛰었는지 그대로 말한다.
  - 문서 쪽(`README.md:22` / `AGENTS.md:15`)은 이미 `--extra compose --extra native`로
    고쳐져 있었다. 이 변경은 그 사실을 되돌릴 수 없게 만드는 쪽이다 — 셋업 명령을
    `--extra compose`로 되돌리면 체크가 `pytorch_optimizer (imported by
    trainbench/axes.py)`를 이름으로 지목하며 red가 된다(실측).

- 2026-08-02 **`classify`가 잘못된 baseline `count`에 죽지 않고 보고한다**
  (`audit_plan.py`, `tests/test_audit.py`. optim 레인이 등재하고 닫았다).
  `docs/audit-baseline.json`은 웨이브 사이에 손으로 편집되는데, 따옴표 붙은 숫자
  하나(`"count": "3"`)가 `TypeError: '>' not supported between instances of 'int'
  and 'str'`를 내고 게이트 전체를 세웠다 — 그 항목과 무관한 12개 체크까지 아무
  결과도 못 냈다.
  - **반환 개수가 4에서 5로 는다**: `classify(...) -> (regressions, fixed, grew,
    shrank, unreadable)`. 정수가 아닌 `count`가 다섯 번째에 이름으로 들어가고,
    `main`이 `BLOCKED: baseline entries with an uncomparable count: ...`로 호명하며
    1로 끝난다. 요약 줄에도 `N unreadable`이 붙는다.
  - `None`은 그대로 '기록된 count 없음'이라 크기 비교만 꺼진다(옛 문자열 전용 항목).
    `bool`은 `int`의 하위 클래스이고 `True == 1`이라 명시적으로 걸러낸다.

- 2026-08-02 **새 체크 `env-locks` — 이미지가 설치하는 lock이 최신이고, 빌드가 그것을
  주장한다** (`audit_plan.py`, `tests/test_audit.py`, `docker/Dockerfile.framework`.
  images 레인, 판정서 `env-locks-are-stale-and-the-dockerfile-claims-a-check-it-does-not-make`).
  `uv sync --frozen`은 lock으로만 설치할 뿐 **lock이 최신인지 묻지 않는다**(묻는 것은
  `--locked`다). 그런데 Dockerfile 주석은 "A stale lock fails the build instead"라고
  적혀 있었고, 그 주석 아래에서 6종 중 5종의 env lock이 stale이었다. 이미지가 담는
  것과 `pyproject.toml`이 선언한 것이 갈라져도 아무것도 말해주지 않는 상태였고,
  버전은 이 연구가 보이게 유지해야 하는 교란 변수다.
  - 체크의 두 반쪽은 한 성질이다: `envs/*/uv.lock`과 루트 lock을 `uv lock --check`로
    전부 재보고, `docker/Dockerfile.framework`의 `uv sync` 호출이 전부 `--locked`를
    다는지 본다. lock만 갱신하면 다음 드리프트를 다시 아무도 안 묻고, 플래그만 바꾸면
    5종 빌드가 즉시 실패한다.
  - uv가 다른 이유로 실패하면 통과가 아니라 `unverified`다. 답에 닿지 못한 감사는
    답을 얻은 것이 아니다.
  - 주석 줄의 `uv sync`는 세지 않는다. 세면 아무 sync도 실행하지 않는 Dockerfile이
    빈 집합 가드를 만족시킨다.
  - **다른 레인이 알아야 할 것**: `envs/*/pyproject.toml`이나 루트 `pyproject.toml`의
    의존성을 건드리면 해당 env에서 `uv lock`을 돌려야 게이트가 녹색이 된다.

- 2026-08-02 **새 체크 `prebuilt-wheels` — URL로 고정한 바이너리 휠은 lock이 해석하는
  ABI와 같아야 한다** (`audit_plan.py`, `tests/test_audit.py`,
  `docs/prebuilt-wheels.yaml`, `envs/native/pyproject.toml`,
  `docker/Dockerfile.framework`. wheel 레인).
  `envs/native`가 `flash-attn`을 소스에서 빌드하는 대신 이 저장소가 직접 빌드해
  릴리스로 올린 휠을 URL로 설치한다(실측 근거와 무엇이 미측정인지는
  `docs/support-matrix.md`). **URL 휠은 리졸버가 아무것도 검사하지 않는다** — uv는
  받아서 그대로 넣고, 어긋남은 파드 위에서 CUDA 오류로 나타난다. 그래서 그 휠이
  무엇에 대해 빌드됐는지를 `docs/prebuilt-wheels.yaml`에 적고 매 게이트에서 대조한다
  (`docs/model-spec.yaml`과 같은 모양: 산문은 문서, 기계가 비교하는 값은 YAML).
  - 대조 대상 넷: **아티팩트 자신의 이름**(PEP 427 파일명의 버전·인터프리터 태그,
    릴리스 태그의 `torch<ver>`/`cu<ver>`), **lock이 해석한 것**(`torch`의 버전과 로컬
    CUDA 태그, `source = { url = … }`와 sha256), **lock의 `requires-python`**,
    **이미지가 선언한 arch**(`TRAINBENCH_CUDA_ARCHS`).
  - `requires-python`은 포함 관계만으로는 부족하다. `>=3.13`은 3.13도 3.14도 담고,
    cp313 바이너리를 3.14가 로드하는 것이 막으려는 실패다. 그래서 "3.13을 담고 이웃
    마이너는 담지 않는가"를 묻는다.
  - 양방향이다. 기록은 있는데 lock이 URL로 설치하지 않으면 **조용히 소스 빌드로
    되돌아간 것**이고(네이티브 이미지 빌드에 13,663초가 다시 붙는다), URL로 설치하는데
    기록이 없으면 **출처를 아무도 안 적은 바이너리**다.
  - 릴리스 태그가 ABI를 말하지 않으면 통과가 아니라 실패다. 휠 파일명은 torch 버전을
    실을 수 없으므로 태그가 그것이 적히는 유일한 자리다.
  - **다른 레인이 알아야 할 것**: `envs/native`의 torch가 올라가면 이 체크가 막는다.
    막히는 것이 옳다 — 그 휠은 그 torch에 대해 다시 빌드해야 하고, 릴리스와
    `docs/prebuilt-wheels.yaml`을 함께 갱신해야 한다. `TRAINBENCH_CUDA_ARCHS`를 넓히는
    것도 같다: 휠에 없는 arch는 느린 파드가 아니라 죽은 파드다.

- 2026-08-02 **`pods.get`이 자기 GraphQL 문서를 보내고, 재시작한 컨테이너는 파드의
  계획을 다시 돌리지 않는다** (`trainbench/pods.py`, `docker/entrypoint.sh`.
  레인 C 소유. **원장의 `uptime_seconds`가 이제 실제로 채워지므로** 남긴다).
  위 `pods.observe` 항목이 세운 재시작 판정은 **실 파드에서 한 번도 동작한 적이
  없다.** 판정의 입력인 `runtime.uptimeInSeconds`를 `runpod.get_pod`가 요청하지 않기
  때문이다 — SDK의 문서는 `runtime { ports { … } }`만 고른다. 그래서 `runtime`은
  truthy이고(전부 `running`으로 읽힘) 시계는 항상 `None`이었다. 첫 실 파드
  (`phase0-sentence_transformers-qwen3_5_0_8b`)가 4분간 프로브를 9번 다시 돌리는
  동안 감시자는 계속 기다렸고, 원장에는 `uptime_seconds: null`이 남았다.
  실측(2026-08-02, 같은 파드 `0dw2kaljoo8pio`, 같은 분): SDK 문서의 `runtime` 키는
  `['ports']`, 이 저장소 문서는 `uptimeInSeconds: 1188465`.
  - `pods.GRAPHQL_URL` / `pods.POD_QUERY` / `pods.read_request(pod_id)`가 생긴다.
    `pods.get(pod_id, transport=send)`로 두 번째 인자가 늘었다(`create`와 같은 모양).
    선택 필드는 `id` / `desiredStatus` / `runtime { uptimeInSeconds }` **뿐이다** —
    SDK 문서가 함께 가져오던 `env`에는 파드의 Infisical 토큰이 들어 있다.
  - GraphQL은 실패한 문서에 **HTTP 200 + `errors`**로 답한다. `get`이 봉투를 벗기고
    `errors`면 예외를 던진다(→ `observe`가 `unknown`). 벗기지 않으면 그 봉투가
    `desiredStatus`도 `runtime`도 없는 파드로 읽혀 **데드라인까지 `pending`**이다.
  - `uptimeInSeconds`는 **30초 단위로 갱신된다**(실행 중 3대 x 5회, 22초 간격 실측:
    증가폭 29~31초, 3대 모두 한 폴링 동안 정지). 떨어지면 증거이고 **안 오르는 것은
    증거가 아니다** — 이 시계 위에 liveness 검사를 세우면 정상 파드를 죽인다.
  - **파드 쪽**: `entrypoint.sh`가 `${RESULT_DIR}/.trainbench-done`을 EXIT trap으로
    쓰고, 그 파일이 있으면 아무것도 하지 않고 exit 0 한다. RunPod은 종료 코드와
    무관하게 컨테이너를 다시 띄우고 그러지 말라고 할 방법이 없다(`PodCreateInput`
    33개 필드에 재시작 정책 없음, REST OpenAPI 실측). 감시자만 고치면 **두 번째
    컨테이너가 이미 다시 측정한 뒤**에야 감시가 끝난다 — 프로브는 결정적이라 티가
    안 나지만 timing 런은 매 재시작마다 새 숫자를 덮어쓰고 마지막 것이 조용히 남는다.
  - **미확인으로 남는 것**: RunPod의 컨테이너 재시작이 컨테이너 디스크를 보존하는지
    재지 못했다. 보존하지 않으면 이 sentinel은 무력하고 동작은 오늘과 같아진다.

- 2026-08-02 **재시작 판정이 두 읽기의 비교에서 벽시계 대비 바닥으로 바뀐다**
  (`trainbench/pods.py`, 레인 C 소유. **원장 스키마가 또 한 칸 넓어지므로** 남긴다).
  위 두 항목이 세운 판정은 실 파드에서 **여전히 발화하지 않았다.** 시계는 이제
  코드에 도착하지만(`341ff62`가 고친 부분), 그 시계가 이런 모양이기 때문이다 —
  실측 2026-08-02, 컨테이너가 재시작 중이던 파드 `xchraazlhvqt6y`, 약 12초 간격:

      05:20:53 RUNNING 0    05:21:06 -11    05:21:18 -11    05:21:31 -9    05:21:43 -9

  `uptimeInSeconds`는 **음수로 간다**(캐시된 `now`가 방금 쓰인 `startedAt`보다
  뒤에 있다). 그리고 **단조 감소하지 않는다** — `0 -> -11`은 하락이고 `-11 -> -9`는
  상승이라, 직전 읽기와 비교하는 규칙은 폴링이 어디에 떨어지느냐가 결정한다.
  오케스트레이터는 6분 넘게 지켜보고 아무것도 결정하지 못했고, 파드는 손으로
  지워졌다(연속 두 번째).
  - `outran_its_clock(anchor, elapsed_seconds, current)`가 생긴다. **이 감시가 처음
    본 시계**를 앵커로 잡고, 이후 모든 읽기는 `anchor + 경과시간 - 캐시 허용치`
    이상을 보고해야 한다. 컨테이너 시계는 초당 1초씩 오르므로 이것은 성질이지
    임계값이 아니다. 입력이 읽기 하나와 벽시계라서 **폴링이 재시작과 증거 사이에
    떨어질 수 없다.**
  - 앵커를 **파드 임대 나이로 잡지 않은 이유**: 임대와 컨테이너 시작 사이에는 이미지
    풀이 있고(계정의 다른 파드에서 10~140초, 이 저장소의 수 GB 이미지에 대해서는
    **측정 안 함**), 안전한 허용치는 이 바닥보다 넓어서 탐지도 더 늦다.
    `lastStartedAt`은 대안이 못 된다 — 50대 전부 `createdAt`과 같았다(재시작에
    움직이지 않는다).
  - `observe(pod_id, get_pod, previous_uptime=None, anchor_uptime=None,
    since_anchor_seconds=0.0)`. 인자 두 개가 는다.
  - **`UPTIME_JITTER_SECONDS`가 1초에서 `CLOCK_REFRESH_SECONDS`(31초)로 오른다.**
    같은 실측이 두 곳에 쓰인다: 필드가 30초 캐시에서 나오므로 한 리프레시만큼
    떨어진 스냅샷 두 개가 컨테이너와 무관하게 최대 31초 차이를 만든다. 1초 임계값은
    그 캐시 잡음을 크래시루프로 읽고 **정상 파드를 런 도중에 종료시킨다.**
    대가는 첫 A100 카나리 모양(17초 주기)이 하락 검사가 아니라 바닥에 걸린다는 것
    — 테스트에서 100초 만에 걸린다.
  - **원장 스키마**: `PodOutcome.to_dict()`에 `peak_uptime_seconds`가 는다.
    크래시루프에서는 마지막 읽기가 무의미하다(끝낸 컨테이너는 몇 초짜리이거나
    음수다). 최고값은 그 감시가 본 가장 오래 산 컨테이너를 가리키고, **발행하고 나서
    재시작한 파드**와 **한 번도 끝내지 못한 파드**를 원장에서 가르는 것이 그 필드다.
    둘 다 계획이 끝났다는 뜻은 아니다 — 그것은 업로드된 결과가 말한다.
  - **오케스트레이터 쪽은 이미 맞다.** `main`은 어떤 outcome에도 파드를 종료한다.
    이번에 생긴 것은 그 경로를 `restarted` outcome으로 고정하는 테스트다.
  - **미확인으로 남는 것**: 실 파드에 대고 이 규칙을 돌려보지 못했다(파드는
    삭제됐고 새로 만들지 말라는 지시). 다음 파드 발사가 보여야 하는 것 — 원장의
    `peak_uptime_seconds`가 채워질 것, 그리고 컨테이너가 재시작하면 발사 후 몇 분
    안에 `reason: restarted`로 감시가 끝날 것.

**새 레인 의무 — 파일을 추가하면 `PLAN.md` 구조 블록에 한 줄 추가한다.**
위 반대 방향 때문에 생긴다. 열거되는 디렉터리(저장소 루트, `configs/`,
`trainbench/`, `scripts/`, `tests/`, `docs/`)에 추적되는 파일이나 디렉터리를 새로
만들면 **추가한 레인의 게이트가 막힌다.** `PLAN.md`는 E 소유지만 이 한 줄은 파일을
만든 레인이 직접 넣는다 — 매번 계약 변경으로 올리게 하면 이 체크가 우회 대상이 된다.
`docker/`, `envs/`, config 그룹 안쪽은 해당 없다(통째로 문서화된 디렉터리다).
`PLAN.md`의 "미작성" 표는 이 체크가 보지 않으므로 여전히 손으로 관리한다 — 구조
블록은 뒤처질 수 없고 그 표만 뒤처질 수 있다.

모델별 사용 규격은 코드가 아니라 config에 있다. 기계 판독 가능한 형태는
`docs/model-spec.yaml`이고, `audit_plan.py`의 `model-spec`이 **값 대 값으로** 대조한다
(문자열 존재 확인은 true를 false로 뒤집어도 통과한다).

| 모델 | `add_generation_prompt` | `instruction_prompt` | `padding_side` | `max_tokens_per_image` |
|---|---|---|---|---|
| qwen3_vl_emb_2b | `true` | `"Represent the user's input."` | `right` | `null`(상한 미선언) |
| qwen3_5_0_8b | `false` | `null` | `right` | `null`(상한 미선언) |
| gemma4_e2b | `false` | `null` | **`left`** | `280`(상한, 고정값 아님) |

`padding_side`가 config에 있는 이유: gemma-4만 left이고, 그것이 `last_token_pool`
결함이 드러나는 유일한 모델이다. 코드가 `arch`로 분기하면 그 사실이 pooling 코드를
읽는 사람 눈에 보이지 않는다.

`padding_side_alignment`가 이 값을 대조하는 상대는 **체크포인트의
`tokenizer_config.json`**이지 프레임워크가 돌려준 객체가 아니다. 위 표의 `source`가
가리키는 파일이 그것이고, 적재 후 프레임워크가 옮겨놓은 값은 체크포인트에 대한
근거가 아니다 — unsloth는 `from_pretrained` 끝에서 무조건 left로 바꾼다. 프레임워크가
옮긴 사실은 `framework_forced`로 기록되고 실패로 채점되지 않는다.

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
