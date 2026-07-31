# Phase 0 지원 매트릭스

프레임워크 x 모델 조합의 실제 동작 여부. 셀마다 근거와 검증 버전을 남긴다.
**확인하지 못한 것은 "미확인"으로 두고 추측으로 채우지 않는다** (컨벤션 16).

이 문서 대부분은 `scripts/report.py`가 pod 실행 결과에서 생성한다. 아래
"환경 해석" 절만 로컬 실행으로 먼저 채워진 부분이다.

## 환경 해석 (2026-07-31, 로컬 macOS / uv 0.11.16 / CPython 3.13.13)

### 단일 환경 공존 불가 — 확정

프레임워크 6종을 하나의 uv 프로젝트에 `[dependency-groups]`로 묶는 시도는 실패했다.
`[tool.uv] conflicts`를 선언해도 **lockfile은 여전히 하나**이고, 공유 lockfile은
공유 패키지의 버전을 하나로 강제한다.

| 시도 | 결과 |
|---|---|
| 하한 없이 6종 그룹 + `conflicts` 선언 | 해석은 성공하나 스택 전체가 후퇴. `transformers 5.3.0`, `ms-swift 1.4.0`, `unsloth 2026.3.11`, `axolotl 0.17.0`으로 잠김 |
| 하한 명시 후 재해석 | 해석 실패 |

하한을 건 뒤 uv가 보고한 충돌 (원문):

```
Because your project depends on datasets>=5.0 and unsloth>=2026.7.1
depends on one of:
    datasets>=3.4.1,<4.0.dev0
    datasets>4.1.0,<4.4.0
we can conclude that your project and unsloth>=2026.7.1 are incompatible.
```

- **unsloth 2026.7.1~2026.7.6은 `datasets<4.4.0`을 요구한다.** 대상 모델이 필요로
  하는 `datasets>=5.0`과 양립하지 않는다
- ms-swift는 하한 없이는 **1.4.0**까지 후퇴했다. 최신은 4.4.2이며, 1.4.0에는
  Qwen3-VL/Gemma4 지원도 임베딩 학습 기능도 없다. 이 상태로 벤치마크를 돌렸다면
  "ms-swift 측정"이라는 이름의 무의미한 숫자가 나왔을 것이다

**결론**: 프레임워크마다 독립 프로젝트 + 독립 lock이 필요하다. uv workspace도
lockfile을 공유하므로 대안이 아니다. `envs/<framework>/`에 각각의 `pyproject.toml`을
두고 `trainbench`를 path 의존성으로 참조한다.

**측정 유효성에 대한 함의**: 프레임워크별 이미지는 서로 다른 torch/transformers
버전을 갖게 된다. 이는 제거 불가능한 교란 변수이므로, 모든 run이 해석된
torch/transformers/프레임워크 버전을 함께 기록하고 리포트에 노출한다.

### native 하네스 기준 환경 — 확정

`uv lock` 성공, 112 패키지.

| 패키지 | 잠긴 버전 |
|---|---|
| torch | 2.13.0 |
| transformers | 5.14.1 |
| datasets | 5.0.1 |
| accelerate | 1.14.0 |
| peft | 0.20.0 |
| hydra-core | 1.3.4 |
| trackio | 0.34.0 |
| pydantic | 2.13.4 |

### 패키지 소스 주의

- PyPI `tevatron`은 0.1.0으로, 논문의 Tevatron 2.0과 일치하지 않는다.
  git 소스(`github.com/texttron/tevatron`)를 써야 한다
- `unsloth`의 requires-python은 `>=3.9,<3.15`. 프로젝트는 3.13에 고정했다

## 모델 x 프레임워크 적재 검증

미실행. `scripts/orchestrate.py` 실행 후 `scripts/report.py`가 채운다.

| | Qwen3-VL-Embedding-2B | Qwen3.5-0.8B | gemma-4-E2B |
|---|---|---|---|
| native | 미확인 | 미확인 | 미확인 |
| unsloth | 미확인 | 미확인 | 미확인 |
| ms-swift | 미확인 | 미확인 | 미확인 |
| sentence-transformers | 미확인 | 미확인 | 미확인 |
| tevatron | 미확인 | 미확인 | 미확인 |
| axolotl | 미확인 | 미확인 | 미확인 |

## 세부 검증 항목

전부 미확인. `PLAN.md` Phase 0 체크리스트 참조.

- 세 모델이 동일 transformers 5.14.x에서 로드되는가
- sentence-transformers 5.6.x x transformers v5 호환
- Qwen3.5 GDN 레이어가 `fla` 없이 학습되는가 / `fla` 설치 시 커널 경로
- gemma-4-E2B PLE의 freeze 가능 여부, LoRA target module 인식
- Unsloth 일반 VLM 경로 + 커스텀 InfoNCE에서 패칭이 깨지지 않는가
- Unsloth `FastSentenceTransformer`가 VLM 체크포인트를 실제로 거부하는가
- Axolotl의 Qwen3-VL 지원
- Tevatron 2.0의 세 모델 지원
- 모델별 동일 이미지의 실제 visual token 수
