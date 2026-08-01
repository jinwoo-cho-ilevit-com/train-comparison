# docs/evidence/

커밋된 실행 레코드. `.gitignore`가 `outputs/`를 제외하므로, 저장소 안에서 어떤 주장을
실행으로 되짚을 수 있는 유일한 자리다.

`audit_plan.py`의 `evidence-committed`는 여기에 파싱되고 `git_commit`과 `config`를 가진
JSON이 하나라도 있으면 통과한다. **통과가 곧 근거는 아니다.** 그 체크는 "추적 가능한
레코드가 커밋되어 있다"만 보고, 그 레코드가 무엇을 뒷받침하는지는 보지 않는다. 그래서
파일마다 무엇을 근거하고 무엇을 근거하지 않는지 여기에 적는다.

## env-report-cpu-qwen3_5_0_8b-native.json

`scripts/env_report.py`가 쓴 레코드.

**근거하는 것**

- 설정 경로가 끝까지 돈다: Hydra 합성 -> 스키마 검증 -> device 해석(`cpu`) -> 시드 ->
  원자적 JSON 쓰기
- 그 레코드가 `git_commit`(`28875a2`), 검증된 `config` 전체, `packages`
  (torch 2.13.0 / torchvision 0.28.0)를 담은 채 커밋되어 있다

**근거하지 않는 것 — `docs/support-matrix.md`의 어떤 셀도 이 파일에서 나오지 않는다**

- `applied`이 `null`이고 `probe` 키가 없다. 이 실행은 **모델을 적재한 적이 없다**
- 따라서 support-matrix의 `native x qwen3_5_0_8b = OK (7/7)`, visual token 196,
  InfoNCE 1 step loss 4.2736 같은 값은 이 파일이 아니라 **커밋되지 않은 실행**에서
  나왔다. 이 저장소 안에는 그 수치를 되짚을 곳이 없다
- 속도·메모리·커널 경로에 대해서는 아무것도 말하지 않는다. `host`의 `cpu_model`,
  `memory_total_gb`, `gpu`, `cuda_runtime`이 전부 `null`이다

## 빠져 있는 근거: probe 결과 아티팩트

`scripts/verify_env.py`의 출력이 하나도 커밋되어 있지 않다. support-matrix의 native
열이 근거로 삼는 것이 바로 그것이다.

이 워크트리에서 재현을 시도했고, 왜 안 되는지까지 확인했다(2026-08-01, macOS):

| 시도 | 결과 |
|---|---|
| 루트 환경(`uv run`) | probe가 `probe_import` 체크에서 `ModuleNotFoundError: transformers`로 끝난다 |
| `envs/native`(`uv run --project envs/native`) | `uv`가 거부한다 — 락파일이 `environments = ["sys_platform == 'linux'"]` |

즉 native probe는 이 저장소의 락으로 **linux에서만** 재현된다. macOS에서 나온
2026-07-31 수치는 그 락 밖의 환경에서 나온 것이고, 그래서 되짚을 아티팩트가 없다.

해소 경로는 Phase 0 pod 실행이다. pod은 `docker/entrypoint.sh`를 통해 결과를 결과
저장소에 올리고, 그중 하나를 여기에 커밋하면 이 항목이 닫힌다. 그 전까지 native 열의
수치는 실행으로 확인되지 않은 상태로 읽어야 한다.
