# 남은 코드 작업 전부 (파드 수치가 필요 없는 것)

## 왜

Phase 0가 두 번 돌았다. **5/18 → 12/18**이고, 네 프레임워크가 세 모델을 모두 적재하며
같은 파라미터 수를 학습한다(625 / 473 / 988). 프레임워크 간 속도 비교의 전제가 처음 섰다.

남은 것은 대부분 코드다. 조사 셋이 이 계획의 근거다 — 저장소 전수 조사(파일:줄), varlen
도달 가능성 조사, 크로스 프레임워크 벤치마크 방법론의 현재 표준 조사. 마지막 것이 축 목록의
큰 구멍 여섯을 찾았고 앞선 가정 둘을 반박했다.

**범위**: 파드 없이 쓰고 검증할 수 있는 것 전부.

## 결정

| # | 결정 | 선택 | 기각한 대안과 이유 |
|---|---|---|---|
| 1 | axolotl dtype | autocast로 감싸 axolotl을 그대로 잰다 | 임베딩을 bf16으로 되돌리기 — native와 같은 체제가 되지만 axolotl이 실제로 학습할 모델이 아니게 된다 |
| 2 | packing 격리 | `cu_seqlens`를 varlen 커널로 전달 | 블록 대각 마스크 직접 생성 — 조사 결과 transformers가 이미 만들고, 4D 마스크를 주면 fa2 varlen 경로가 꺼진다 |
| 3 | trackio | 스키마에서 제거 | 구현 — 6개 env lock 전부에 넣고 이미지 재빌드, 측정 중 네트워크 I/O가 교란 |
| 4 | 버전 교란 | `report.py`가 같은 스택끼리만 줄 세운다 | 한 표에 다 넣고 버전을 열로 — 독자가 순위를 먼저 읽고 각주를 나중에 읽는다 |
| 5 | 어댑터 경계 | 프레임워크의 학습 스텝을 그대로 잰다 | 베이스 인코더만 꺼내 공통 루프 — 프레임워크가 아니라 우리 루프를 재게 된다 |
| 6 | `kernel=kernels_hub` | 축 값을 버린다 | 적용 지점을 모델 생성 후로 이동 — "kernel/attn은 모델 생성 전에만 바꿀 수 있다"는 설계 전제가 깨진다 |
| 7 | 방법론 보강 범위 | 리서치가 찾은 새 축까지 전부 | 나중 배치 — 어댑터 인터페이스를 두 번 설계하게 된다 |

## 축 표 (커버리지 기록)

| 축 | 상태 | 근거 |
|---|---|---|
| 프레임워크 간 측정 의미론 | decided | 결정 1·4·5 |
| 축 적용 검증 계약 (capture) | decided | lane-c |
| 어댑터 경계 | decided | 결정 5 + 빌드 지문 |
| 공허한 검사 | decided | lane-b |
| 아티팩트 신원 | decided | lane-b (image digest는 이미 있음) |
| 레인 경계·파일 소유권 | decided | 아래 레인표. 외부 표준 없음 — null result |
| 범위 경계 (코드 vs 파드) | decided | 아래 "제외" |
| 증거 기준 | decided | 각 레인 완료 조건 |
| A 학습 유효성 게이트 | decided | lane-d |
| B 토큰 회계 계약 | decided | lane-d |
| D 측정 통계 | decided | lane-d |
| G 빌드 지문 + 커널 provenance | decided | lane-e, lane-g |
| H 피크 메모리 / OOM 범주 | decided | lane-d |
| I 시퀀스 길이 축 + 스코프 라벨 | decided | lane-d |
| C 수렴 등가성 (loss parity) | **open — 파드** | 설계는 lane-d, 측정은 파드 |
| E GPU clock/power lock | **open — 파드** | RunPod 컨테이너에서 `nvidia-smi -lgc` 가능한지 미확인 |
| F 호스트 지문 | **open — 파드** | 같은 A100 이름이 같은 기계가 아님 |
| K 에너지 | not applicable | MLPerf Power는 optional, 2026 필수 근거 미확인 |

## 리서치가 반박한 가정 둘 — 문서에 올리고 값은 파드가 정한다

**seed 고정.** MLPerf CLOSED는 seed를 `/dev/urandom`에서 뽑고 run마다 기록하며 "no other run
can log the same seed on the same line"을 요구한다. 고정 seed로 반복하면 분포가 아니라 한 점을
재측정하는 것이다. lane-d가 스키마를 만들되 정책 변경은 노이즈 바닥 측정 후다.

**3% 임계값.** GPU 경합만으로 표준편차 30배·평균 +21%가 관측된 사례가 있다. `AGENTS.md`의 3%는
근거 없는 상수이고, 첫 파드에서 canonical baseline 10회 반복으로 유도해야 한다.

## 새 병목 — `scripts/bench.py`

프롬프트·packing·토큰 회계·측정 통계·유효성 게이트·피크 메모리·어댑터가 전부 이 한 파일을
건드린다. 한 파일은 한 레인만 소유하므로, **lane-d가 먼저 모듈로 쪼갠다.**

```
scripts/bench.py        얇은 진입점            lane-d 소유
trainbench/collate.py   Collate/PackedBatches  lane-d가 만들고 lane-f 소유
trainbench/metrics/     토큰회계·통계·메모리    lane-d 소유
trainbench/loader.py    어댑터 레지스트리       lane-d가 자리만, lane-g 소유
```

## 레인표

| lane | owns | security |
|---|---|---|
| lane-a | `trainbench/probe/tevatron.py` | |
| lane-b | `scripts/report.py`, `scripts/orchestrate.py`, `scripts/prepare_data.py`, `docker/entrypoint.sh`, `trainbench/probe/sentence_transformers.py`, `configs/run`, `pyproject.toml` | true |
| lane-c | `trainbench/applied.py` | |
| lane-d | `scripts/bench.py`, `trainbench/metrics`, `trainbench/probe/steps.py`, `trainbench/config_schema.py` | |
| lane-e | `trainbench/kernels.py`, `docs/methodology.md` | true |
| lane-f | `trainbench/collate.py`, `trainbench/prompt.py`, `configs/model` | |
| lane-g | `trainbench/loader.py`, `trainbench/probe/native.py`, `trainbench/probe/unsloth.py`, `trainbench/probe/ms_swift.py`, `trainbench/probe/axolotl.py`, `trainbench/probe/registry.py` | |
| lane-h | `trainbench/axes.py`, `configs/optim`, `configs/precision`, `configs/train`, `configs/parallel`, `configs/dataloader`, `configs/peft`, `configs/kernel` | |
| lane-i | `AGENTS.md`, `PLAN.md`, `docs/CONTRACTS.md`, `docs/support-matrix.md`, `README.md`, `docs/open-verdicts.json` | |

lane-i는 마지막에 돈다 — 루트 문서와 판정 원장은 여러 레인이 건드릴 대상이라 한 번에 모은다.

## 경계표

| name | lanes | test | sample |
|---|---|---|---|
| collate-metrics | lane-d, lane-f | tests/contract/test_collate_metrics.py | tests/fixtures/microbatch.sample.json |
| loader-bench | lane-d, lane-g | tests/contract/test_loader_bench.py | tests/fixtures/adapter_out.sample.json |
| applied-axes | lane-c, lane-h | tests/contract/test_applied_axes.py | tests/fixtures/axis_state.sample.json |
| record-report | lane-d, lane-b | tests/contract/test_record_report.py | tests/fixtures/run_record.sample.json |
| kernel-provenance | lane-e, lane-g | tests/contract/test_kernel_provenance.py | tests/fixtures/kernel_fingerprint.sample.json |

## 전체 완료 조건

- 모든 레인의 완료 조건이 충족된다
  → 각 `lane-*.md`
- 게이트 넷이 통과한다
  → `uv run ruff check && uv run ruff format --check`
  → `infisical run --env=dev -- uv run pytest`
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
  → `infisical run --env=dev -- uv run python scripts/env_report.py device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4`
- `axis-values`에 단일값 그룹이 0이다
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- `config-consumed`가 0이다
  → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
- [pod] Phase 0가 18/18이다
  → verdict: ____  by: ____  at: ____

## 규율

**부숴서 확인하지 않은 검사는 증거가 아니다.** 각 레인은 자기가 넣은 검사를 되돌려 죽는 것을
보고 그 출력을 보고한다. 무력한 변이는 숨기지 않는다.

**핀된 소스를 읽고 나서 단언한다.** Phase 0가 찾은 모든 프로브 실패는 답이 이미 잠긴 휠 안에
있었다 (`AGENTS.md ## Verification`).

**직접 내지 않은 숫자를 옮기지 않는다.** 확인할 수 없으면 "확인 안 함"이라고 쓴다.

## 제외

- 파드가 숫자를 만들어야 답이 나오는 것 — 노이즈 바닥과 3% 임계값 유도, 프로파일러 오버헤드,
  deterministic on/off, 데이터로딩 병목 판정, loss parity, GPU clock lock 가능 여부, 호스트 지문,
  liger 커버리지 분포, `loss-empty-pixel-slice`, `loss-gradcache-memory-and-overhead`
- 캠페인 실행 자체 (Phase 2 ablation, Phase 3 프레임워크 측정, 품질 런)
- `docs/report.md` — 산출물이고 숫자가 나와야 쓴다
