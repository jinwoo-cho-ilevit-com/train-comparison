# Task 1+2 — 저장소 부트스트랩 + Phase 0 지원 매트릭스

## Context

Qwen3-VL-Embedding-2B / Qwen3.5-0.8B / gemma-4-E2B 세 모델의 임베딩 학습 속도
최적화를 실측 비교하는 연구다. 전체 설계와 근거는 `PLAN.md`에 있다.

지금 필요한 것은 **측정을 시작할 수 있는 상태**다. 그런데 그 앞에 판정해야 할 것이
있다. 세 모델은 config상 요구 transformers 버전이 갈리고(4.57.1 / 4.57.0.dev0 /
5.5.0.dev0, 현재 안정판 5.14.1), 프레임워크 6종의 지원 여부도 문서로는 확인되지
않는다. Unsloth의 임베딩 경로는 encoder-only로 보이고, Axolotl의 Qwen3-VL 지원은
문서에 없으며, Tevatron의 세 모델 지원도 미확인이다.

이 판정 없이 하네스를 만들면 만든 뒤에 전제가 무너진다. 따라서 이번 범위는
**저장소 골격(Task 1) + 프레임워크 x 모델 적재 검증(Task 2)**이고, 산출물은
`docs/support-matrix.md`다. 이 매트릭스가 이후 모든 실험 설계의 입력이 된다.

## 핵심 설계 결정

### 1. 스토리지 — network volume 미사용

RunPod network volume은 통상 200~400 MB/s의 네트워크 연결형 스토리지다(HIGH
PERFORMANCE 티어가 최대 3배 처리량/4배 IOPS). 여기에 학습 데이터를 두면
**Phase 1의 데이터로딩 병목 판정이 파이프라인이 아니라 볼륨을 측정하게 되어
실험 축 하나가 통째로 무효화된다.** 따라서 쓰지 않는다.

원칙: 소스가 무엇이든 **측정 중에는 모든 것이 pod-local NVMe에 있어야 한다.**

**의존성 = Docker 이미지 (공통 베이스 + 프레임워크별 6개)**

- 베이스 이미지: CUDA + torch + transformers. 6개 프레임워크 이미지가 이 레이어를
  공유하므로 레지스트리 저장량과 호스트 캐시가 재사용된다 — 증분만 얇게 쌓인다
- 프레임워크 6종은 한 환경에 공존이 불가능하다. **이미지 경계로 격리**한다
- 셋업 편차가 사라져 벤치마크 신뢰도가 오른다. 이것이 이미지를 쓰는 진짜 이유다
- 레지스트리는 GHCR. RunPod의 container registry auth에 한 번 등록하면 pod마다
  자격증명을 넘길 필요가 없다
- **빌드는 amd64 네이티브에서 한다.** macOS arm64에서 QEMU 에뮬레이션으로 CUDA
  이미지를 굽는 것은 비현실적이다. RunPod CPU pod에서 빌드·푸시하는 것을 기본으로
  하고, GitHub Actions는 무료 러너 디스크 용량을 먼저 확인한 뒤에만 쓴다

**모델·데이터 = HF Hub → pod-local NVMe**

- 모델 3종은 이미 공식 HF repo에 있으므로 그대로 받는다
- 전처리한 MMEB 고정 서브셋은 **private dataset repo**로 올린다. repo revision이
  곧 데이터 버전이 되어 컨벤션 07의 "데이터 버전 기록" 요건을 자연히 충족한다
- `HF_XET_HIGH_PERFORMANCE=1`을 쓴다. **`HF_HUB_ENABLE_HF_TRANSFER`는 현재
  huggingface_hub에서 무시되므로 쓰지 않는다** (hf_xet이 대체, 적응형 동시성 최대
  64 스트림)
- 결과 JSON과 프로파일 산출물도 같은 private repo의 pod별 경로로 push

**부수 효과: DC 종속이 사라진다**

network volume이 pod을 특정 DC에 묶는 유일한 요인이었다. 제거하면:

- B200 재고가 있는 3개 DC(EU-RO-1, US-NC-2, US-NE-1)를 **모두** 쓸 수 있어 18개
  pod 확보 확률이 오른다
- A100(Phase 0~1) -> B200(Phase 2~3) 전환 시 볼륨 재구축이 불필요하다
- H200 fallback이 실제로 즉시 전환 가능해진다

`PLAN.md`의 스토리지·DC 제약·리스크 절을 Task 1에서 이에 맞춰 갱신한다.

단, **pod 간 하드웨어 편차 규칙은 그대로 유효하다.** 그것은 호스트 CPU/메모리
대역폭 차이지 스토리지 문제가 아니다. canonical baseline 게이트는 유지한다.

### 2. 프레임워크별 의존성 그룹 분리

- 베이스 환경 = native 하네스 (torch + transformers + peft)
- 프레임워크마다 `[dependency-groups]`의 별도 그룹, 이미지 하나당 그룹 하나
- 공존 불가 자체가 Phase 0의 유효한 결과이므로 매트릭스에 기록한다

이 구조가 "프레임워크 x 모델 = 18 pod" 분할과 정확히 맞물린다.

### 3. deterministic 모드 — 컨벤션과의 충돌 해소

컨벤션 07은 `torch.use_deterministic_algorithms(True)` + `cudnn.benchmark=False`를
기본으로 요구한다. 그런데 이는 커널 자동선택과 cudnn autotuning을 끄는 것이라
**속도 벤치마크의 측정 대상 자체를 왜곡한다.**

컨벤션이 규정한 탈출구를 따른다: "성능 비용이 병목으로 측정되면 끄고, 그 사실을
기록하라."

- `seed.py`는 `deterministic: bool` config 필드로 양쪽을 지원
- Phase 1에서 deterministic on/off 비용을 **1회 측정해 기록** (이것이 컨벤션이
  요구하는 근거)
- 본 캠페인은 deterministic off + `cudnn.benchmark=True`로 실행하고, 결정과
  측정치를 `docs/methodology.md`에 남긴다
- 유닛 테스트와 CPU 스모크는 deterministic on 유지

같은 문서가 "GPU 세대·배치 크기·병렬 구성·프레임워크 버전이 다르면 bit-exact
재현은 보장되지 않는다"고 명시한다. 이것이 `PLAN.md`의 GPU 혼용 금지 규칙과
pod별 canonical baseline 규칙의 근거다.

### 4. 실험 추적 — Trackio + HF Space

`trackio.init(project=..., space_id=...)`로 다수 pod이 중앙 Space에 기록한다.
백그라운드 스레드가 배치를 푸시하고, 실패 시 로컬 SQLite에 보관 후 재시도하므로
pod 손실·네트워크 단절에 강하다. 모든 run은 resolved config + git hash + 데이터
repo revision과 연결해 기록한다(컨벤션 07).

### 5. 시크릿 — Infisical 주입 (컨벤션 13)

`infisical init` 완료 상태(`.infisical.json`, workspaceId
`a95d3bb8-6fe8-4993-9f32-cf7a92d444a9`). `defaultEnvironment`가 비어 있으므로
환경 이름을 확인해 채우거나 매번 `--env=`를 넘긴다.

- **평문 `.env`를 만들지 않는다.** 저장소에는 `.infisical.json`(민감정보 없음,
  커밋 대상) + `.env.example`(키 이름만)만 둔다
- 모든 실행 명령을 주입 래퍼로 감싼다: `infisical run --env=dev -- uv run ...`
- 코드는 평소대로 `os.environ[...]`로 읽는다. SDK 직접 조회는 쓰지 않는다
- `.gitignore`에 `.env`, `.env.*`, `!.env.example`
- gitleaks를 pre-commit과 CI 양쪽에서 실행

**최소 권한 설계**

| 주체 | 필요한 시크릿 | 근거 |
|---|---|---|
| 로컬 오케스트레이터 | `RUNPOD_API_KEY`, `HF_TOKEN` | pod 기동/종료, 데이터 repo 준비 |
| 이미지 빌드 pod | `GHCR_TOKEN` | 레지스트리 푸시 |
| 실험 pod (18개) | `HF_TOKEN`만 | 모델·데이터 pull, 결과 push, Trackio Space 푸시 |

**`RUNPOD_API_KEY`는 실험 pod에 절대 올리지 않는다.** pod이 자기를 기동한 계정의
전권 키를 들고 있을 이유가 없다. GHCR 자격증명도 RunPod의 container registry auth에
등록하므로 pod env에는 들어가지 않는다.

실험 pod은 비대화형 컨테이너이므로 **machine identity(Universal Auth)**로 인증한다.
읽기 전용·dev 범위 identity를 만들고, RunPod pod env로 `INFISICAL_TOKEN`을 주입한
뒤 엔트리포인트에서 `infisical run -- ...`이 `HF_TOKEN`을 가져오게 한다. `HF_TOKEN`
값 자체를 18개 pod 설정에 뿌리지 않기 위한 것이다. 컨벤션 13의 argv 노출 경고에
따라 client-secret을 명령행 인자로 넘기지 않는다.

## 생성할 파일

### Task 1 — 부트스트랩 (GPU 불필요)

| 파일 | 내용 |
|---|---|
| `pyproject.toml` | `~/Codes/develop-convention/templates/pyproject.toml` 기반. py3.13, ruff(E,F,I,UP,B, line-length 100). torch platform marker 주석 해제 — non-linux는 `pytorch-cpu`, linux는 `pytorch-cuda`(cu130, B200 드라이버에 맞춰 Task 2에서 재확인). 프레임워크별 `[dependency-groups]` 6종 |
| `.python-version`, `uv.lock` | `uv.lock`은 커밋 |
| `AGENTS.md` | `templates/AGENTS.md` 기반. ML 블록 유지, LLM API 블록과 docsync 블록 삭제. `[CONVENTION_PATH]`에 `~/Codes/develop-convention` |
| `CLAUDE.md` | `@AGENTS.md` |
| `.env.example` | `RUNPOD_API_KEY`, `HF_TOKEN`, `GHCR_TOKEN` — 키 이름만, 값 없음. Trackio Space id와 HF repo id는 시크릿이 아니므로 `configs/`에 둔다 |
| `.infisical.json` | 이미 존재. `defaultEnvironment` 확인 후 채움 |
| `.gitignore`, `.pre-commit-config.yaml` | `.env`/`.env.*`/`!.env.example` + `astral-sh/ruff-pre-commit` + gitleaks |
| `trainbench/device.py` | `get_device()` — `torch.accelerator` 기반 단일 헬퍼, config로 override 가능. 인라인 `.cuda()` 금지 |
| `trainbench/seed.py` | `set_seed(seed, deterministic)` — random/numpy/torch/CUDA + DataLoader `worker_init_fn`·`generator` |
| `trainbench/config_schema.py` | Pydantic 모델. 잘못된 조합은 실행 전 fail-fast |
| `configs/` | `config.yaml` + 그룹 골격 (`model/`, `data/`, `attn/`, `kernel/`, `precision/`, `compile/`, `optim/`, `loss/`, `peft/`, `freeze/`, `dataloader/`, `parallel/`, `framework/`, `experiment/`) |
| `tests/test_config.py` | 잘못된 조합이 실행 전에 죽는지 |

config group 상세는 `PLAN.md`의 "저장소 구조" 참조. 상호배타 변형이 3개 이상인
축만 group으로 만들고, 단일 플래그(gradient checkpointing 등)는 `train.yaml` 필드다.

### Task 2 — 이미지·데이터 준비 + Phase 0 검증 (A100 18 pod)

| 파일 | 내용 |
|---|---|
| `docker/Dockerfile.base` | CUDA + torch + transformers + uv. 6개 이미지가 공유할 레이어 |
| `docker/Dockerfile.<framework>` | 베이스 위에 `uv sync --group <framework>` 만 얹는 얇은 레이어 6개 |
| `docker/entrypoint.sh` | `infisical run --` 로 `HF_TOKEN` 주입 -> 모델·데이터를 local NVMe로 pull -> `verify_env.py` 실행 -> 결과 push |
| `scripts/build_images.py` | RunPod CPU pod에서 6개 이미지 빌드 후 GHCR 푸시 |
| `scripts/prepare_data.py` | MMEB 고정 서브셋 생성(분포 보존 무작위 샘플) -> private HF dataset repo push. revision을 기록 |
| `trainbench/probe/` | 프레임워크별 probe 어댑터 6종. 공통 인터페이스로 "모델 적재 -> 1 step 학습 -> 결과 JSON"만 수행 |
| `scripts/verify_env.py` | Hydra 진입점. `framework=` x `model=` 한 조합을 검증하고 결과 JSON을 atomic save 후 HF repo에 push |
| `trainbench/pods.py` | RunPod REST API 얇은 래퍼 (pod 생성/상태/종료, 이미지 지정, `INFISICAL_TOKEN` 주입) |
| `scripts/orchestrate.py` | 조합 목록을 받아 pod 기동, 확보 실패분은 **큐잉**(전량 순차 폴백 아님), 결과 수집, 종료. DC 종속이 없으므로 재고 있는 DC 아무 곳에나 배치 |
| `scripts/report.py` | 결과 JSON 병합 -> `docs/support-matrix.md` |
| `docs/methodology.md` | 측정 규율 기록 시작 (deterministic 결정 포함) |

**검증 항목** (`PLAN.md` Phase 0 체크리스트와 동일)

- 세 모델이 동일 transformers 5.14.x에서 로드되는가
- sentence-transformers v5.5 x transformers v5 호환
- Qwen3.5 GDN 레이어가 `fla` 없이 학습되는가 / `fla` 설치 시 커널 경로
- gemma-4-E2B PLE의 freeze 가능 여부, LoRA target module 인식
- Unsloth 일반 VLM 경로 + 커스텀 InfoNCE에서 패칭이 깨지지 않는가
- Unsloth `FastSentenceTransformer`가 VLM 체크포인트를 실제로 거부하는가
- Axolotl의 Qwen3-VL 지원
- Tevatron 2.0의 세 모델 지원
- 모델별 동일 이미지의 실제 visual token 수

**기록 규칙**: 셀마다 근거(로그 경로 또는 URL) + 검증한 버전. 확인 못 한 것은
"미확인"으로 남기고 추측으로 채우지 않는다 (컨벤션 16).

## 실행 순서

1. Task 1 전체를 로컬에서 완료하고 lint/test/CPU 스모크 통과
2. `PLAN.md`의 스토리지·DC·리스크 절을 network volume 미사용에 맞춰 갱신
3. 하네스 코드를 작성자와 분리된 레인에서 리뷰 (컨벤션 09)
4. Infisical에 `RUNPOD_API_KEY`/`HF_TOKEN`/`GHCR_TOKEN` 등록 + pod용 machine
   identity 발급 + RunPod container registry auth 등록
5. `prepare_data.py`로 MMEB 서브셋 생성 -> private HF dataset repo push
6. CPU pod에서 베이스 이미지 1개 + 프레임워크 이미지 6개 빌드·푸시
7. **pod 1개로 entrypoint -> pull -> verify -> push 전 경로 검증**
8. 18개로 확장 -> `docs/support-matrix.md` 생성 -> 결과를 보고 Task 3 설계 조정

7단계를 건너뛰면 18개 분의 비용을 날린다.

## 검증 방법

**Task 1 완료 조건** (전부 로컬 macOS CPU에서)

```
infisical run --env=dev -- uv run ruff check
infisical run --env=dev -- uv run ruff format --check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/verify_env.py \
    device=cpu model=qwen3_5_0_8b framework=native data.limit=4
```

Hydra 진입점이므로 컨벤션 04의 `--limit N`은 `data.limit` config 필드로 만족시킨다.
별도 argparse 플래그를 두면 "값은 전부 중앙 config" 규칙(02)과 충돌한다.

- 마지막 명령이 macOS CPU에서 끝까지 돌아야 한다. GPU 없이 실행 불가면 컨벤션 03 위반
- `test_config.py`에서 잘못된 config 조합이 학습 시작 전에 죽는 것을 확인
- 평문 `.env` 파일이 생성되지 않았고, gitleaks pre-commit이 도는 것을 확인

**Task 2 완료 조건**

- 6개 프레임워크 이미지가 베이스 레이어를 공유하는 것을 확인 (`docker history`로
  증분 크기 확인 — 공유가 깨졌으면 빌드 순서가 잘못된 것)
- pod 1개에서 모델·데이터가 **local NVMe 경로**로 내려받아졌는지 확인
  (network volume 마운트가 없는지 함께 확인)
- 18개 조합 실행 후 `docs/support-matrix.md`에 18셀이 채워지고, 미확인 셀이
  "미확인"으로 표기됨
- Trackio Space에 18개 run이 config + git hash + 데이터 repo revision과 함께 기록됨
- pod env에 `RUNPOD_API_KEY`와 `GHCR_TOKEN`이 없고 `INFISICAL_TOKEN`만 있는 것을 확인
- 모든 pod이 종료되어 과금이 멈춘 것을 확인 (`list-pods`)

**완료 주장 금지 조건**: TODO/stub/skip이 남아 있거나, 실행 로그 없이 통과를
주장하는 경우 (컨벤션 06)

## 이번 범위에서 제외

- 측정 하네스 본체 (throughput/MFU/VRAM, 타이밍-프로파일 분리, canonical
  baseline 게이트) -> Task 3
- DALI 도입 판단 -> Task 3의 데이터로딩 병목 선판정 결과에 따름
- 실제 ablation -> Task 4
- GitHub Actions 빌드 파이프라인 -> CPU pod 빌드로 충분하면 만들지 않는다
