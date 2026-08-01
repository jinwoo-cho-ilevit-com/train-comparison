# 축 구현 + 측정 개시 (A~D + Phase 0/3)

## Context

Wave 3 게이트가 닫혔다. 하네스는 정직해졌다 — 타이밍 창이 데이터 파이프라인을 담고,
capture probe가 17축을 검증하고, sweep이 잰 것을 올린다. 게이트는 403 passed / 감사 10/12.

**그런데 이 저장소는 아직 측정을 시작할 수 없다.** 탐색으로 확인한 사실:

- **축 12개 중 6개가 비활성값 하나만 받는다.** kernel·precision·optim·dataloader·loss·parallel.
  전부 `axes.py`의 무조건적 `not implemented`이고 import 실패가 아니라, 이미지를 빌드해도
  움직이지 않는다. 저장소의 유일한 timing 매니페스트 3개가 하필 `loss` 축인데
  `cached_mnrl`이 즉사한다.
- **`PLAN.md`가 이 구현들을 일정에 넣은 적이 없다.** Task 4는 "축 그룹별 조합 정의 후 실행"뿐이다.
  "새 변형은 config 조합에서 나온다"는 규칙이 조용히 "축이 이미 존재한다"를 전제했다.
- **`scripts/report.py`가 timing 결과를 읽지 못한다.** `payload["probe"]["checks"]`만 보는데
  `bench.py` 레코드에는 `probe` 키가 없다(`record.py:156`). 측정에 성공한 런이
  `결과 없음(기동됨)`으로 렌더된다(`report.py:194`). 그리고 **3% 편차를 계산하는 코드가
  저장소 전체에 0줄이다** — `_baselines.yaml`의 canonical은 계획·구성·전송·실행되지만
  아무도 그 결과를 되읽지 않는다.
- **Phase 0가 한 번도 실행된 적이 없다.** probe 매니페스트 18개(프레임워크 6 x 모델 3)는
  오늘 그대로 돌고 `report.py`가 정확히 렌더하는데, 파드가 뜬 적이 없어 결과가 0건이다.
- **파드 기동이 `assert_pod_scope_is_safe`(`orchestrate.py:462`)에 막혀 있다.** 파드 토큰이
  `RUNPOD_API_KEY` 등 4개를 읽을 수 있어서 거부된다. 가드는 옳고, 사용자가 Infisical
  스코프를 고치기로 했다(2026-08-02).

**결정 (2026-08-02)**: 파드를 지금 띄워 GPU 트랙과 CPU 트랙을 병행한다. 구현은 workflow
팬아웃 + 적대 검증으로 진행한다. Phase 3도 함께 진행한다.

**이 계획이 끝나면 "측정을 시작할 수 있는 상태"가 된다.** 캠페인 실행과 `docs/report.md`는
그 다음이다.

---

## P0 — 파드 가드 해제 (선행, 일부는 사용자 작업)

1. **사용자**: Infisical 머신 아이덴티티를 `HF_TOKEN`만 읽도록 스코프한다(별도 환경 또는
   경로 스코프 중 편한 쪽). `pod_reachable_secret_names`(`orchestrate.py:423`)가 읽는 것은
   토큰의 가시 범위이므로 둘 다 통한다.
2. `orchestrate`가 그 환경을 쓰도록 `--infisical-env` 기본값/문서를 맞춘다.
3. **`assert_pod_scope_is_safe`에 테스트가 없다.** 금지 시크릿이 보이는 토큰으로 기동이
   거부되는지, 스코프된 토큰으로 통과하는지 둘 다 건다. 이 가드는 사용자 제약을 강제하는
   유일한 코드인데 지금 아무도 지키지 않는다.

부수 확인: dirty tree(`:552`)와 미해결 이미지 digest(`:636`)도 기동을 막는다. 둘 다 우회
플래그가 있으므로 차단은 아니지만 첫 기동 전에 상태를 확인한다.

---

## 트랙 1 — `report.py` (모든 측정 해석의 선행 조건)

`scripts/report.py`, `docs/support-matrix.md` 생성 블록.

1. **timing 레코드를 렌더한다.** `cell()`이 `probe` 키 부재를 "결과 없음"으로 읽지 않게 하고,
   `metrics` 키를 가진 레코드를 별도 표로 낸다: step p50/p95, samples/s, tokens/s,
   peak memory, `steps_discarded`, `profiled`, 그리고 `METRIC_DEFINITIONS`.
2. **baseline 편차를 계산한다.** 각 파드의 `baseline:canonical` 결과를 모아 기준선과 비교하고
   3% 게이트를 적용한다. 임계값은 상수 하나로 두고 **교정 전까지 "미교정"으로 표시**한다
   (`methodology.md:127` — 3%는 근거 없는 값이다).
3. **baseline 결과 충돌을 푼다.** 지금은 모든 파드의 canonical이 `(native, qwen3_5_0_8b)`
   셀로 들어가 phase0 probe와 겹치고 중복으로 버려진다(`newest_per_combination:136`).
   baseline은 지원 매트릭스 셀이 아니라 파드 귀속으로 파일링되어야 한다.

**부술 것**: 편차 초과 파드를 만들어 게이트가 실제로 그 파드를 무효로 표시하는지.
timing 레코드에서 `metrics`를 빼고 렌더가 그걸 조용히 통과시키지 않는지.

---

## 트랙 2 — CPU에서 구현·검증 가능한 축 (workflow 팬아웃)

각 축이 `axes.py`의 **다른 함수**를 건드리므로 병렬 레인이 성립한다. workflow는
`pipeline(축, 구현, 적대검증)` 형태 — 축마다 구현→검증이 다른 축을 기다리지 않는다.

| 축 | 손대는 곳 | 핵심 |
|---|---|---|
| `loss=cached_mnrl` | `axes._loss` | **최우선** — 유일한 timing 매니페스트 3개를 살린다 |
| `optim=muon` | `axes._optimizer` | Muon 가설을 연다 |
| `dataloader.pretokenize` | `axes._dataloader` | 새 config 변형 파일 필요 |
| `dataloader.packing` | `axes._dataloader` + collate | `last_token_pool`이 packed 배치를 거부한다 |
| `gradient_checkpointing=selective` | `axes._apply_to_model` | **capture 확장 필수** |
| `cross_device_negatives` | `axes._loss` | gloo 2프로세스로 검증 |
| `kernel` dispatch | `axes.patch` | 실제 패치는 GPU, 분기는 스텁 검증 |
| `peft=qlora`의 `load_kwargs` 절반 | `axes.load_kwargs` | 4bit 적재는 GPU |

**축별로 놓치면 안 되는 것 (탐색에서 확인됨):**

- **GradCache**: `gradcache` 0.1.0은 의존성 없는 순수 파이썬이다. `steps.encode` + `info_nce`를
  그대로 재사용한다. `config.loss.mini_batch`는 이미 존재하고 검증된다(`config_schema.py:304`).
  capture는 손댈 게 없다 — 클로저에 `axis_value`/`axis_cross_device_negatives`만 달면 된다.
  **구조적 간극**: `Built.loss_fn`은 `(queries, documents) -> tensor`인데 GradCache는 모델과
  원본 입력이 필요하고, 측정 루프는 `built.loss_fn`을 **한 번도 부르지 않는다**
  (`bench.py:373`이 `info_nce`를 직접 호출). 이 계약을 먼저 정해야 한다.
  **수치 등가성 테스트가 별도로 요구된다**(`review-findings.md:155`) — 없이 재면 GradCache
  버그가 GradCache 속도 향상으로 보고된다.
- **Muon**: `pytorch-optimizer` 3.10.1이 이미 `envs/native`에 잠겨 있고 `py3-none-any`,
  의존성은 numpy+torch뿐이다. 로컬 dev 환경에 추가만 하면 CPU에서 실제 스텝이 돈다.
  capture도 손댈 게 없다(`type(opt).__name__.lower()` -> `"muon"`).
  **설계 결정 하나**: Muon은 통상 embedding을 제외하고 AdamW에 넘긴다. gemma-4는 PLE가
  파라미터의 46% 이상이라 param-group 분할 방식이 `_capture_optim`의 group 수 및
  `freeze.ple`과 상호작용한다. 이건 리포트의 한정 조건이 되므로 근거와 함께 문서에 남긴다.
- **packing**: `last_token_pool`(`embedding.py:22`)이 행당 시퀀스 1개 + 연속 패딩을 단언하고
  **아니면 raise한다.** packed 배치는 이 계약을 정면으로 위반하므로 packed 풀링 경로가
  같이 필요하다. `MicroBatch`의 토큰 회계도 함께 바뀐다.
- **selective checkpointing**: `create_selective_checkpoint_contexts`는 로컬 torch 2.13.0에
  이미 있다. 그런데 `_capture_gradient_checkpointing`(`applied.py:392`)은 `none`/`full`/`partial`
  밖에 반환할 수 없어서, **동작해도 영구 불일치가 된다.** probe 확장이 구현과 한 세트다.
- **config 파일이 없는 값 3개**: `gradient_checkpointing=selective`, `dataloader.pretokenize=true`,
  `cross_device_negatives=true`. 변형 파일을 만들면 `AXIS_PACKAGES` 또는 `AXIS_NEEDS_NOTHING`
  (`audit_plan.py:703`, `:737`)에 등록해야 `axis-packages`가 통과한다.

---

## 트랙 3 — GPU가 있어야 검증되는 축 (첫 파드 이후)

`precision=mxfp8/nvfp4`, `optim=adamw_8bit`, `parallel=ddp/fsdp2/zero2/zero3` + `train.offload`,
`dataloader.backend=dali`.

**capture 쪽 작업이 apply 쪽만큼 중요하다** — 아래 넷은 구현해도 probe가 못 읽으면
런이 보고 불가다:

- `train.offload`: `_capture_offload`(`applied.py:576`)가 deepspeed 아래에서 **의도적으로**
  undetermined를 반환한다. undetermined는 불일치와 똑같이 timing 런을 막는다(`applied.py:803`).
  즉 오늘은 deepspeed가 완벽히 동작해도 결과를 낼 수 없다.
- `precision=mxfp8/nvfp4`: `_capture_precision`이 transformer_engine/torchao 모듈이 보이면
  undetermined를 반환한다(설계상 옳다 — 가중치의 bf16을 읽어 fp8 런을 인증하면 안 된다).
  recipe가 실제로 걸렸는지 읽는 경로가 따로 필요하다.
- `optim=adamw_8bit`: `_capture_optim`이 `AdamW8bit` -> `"adamw8bit"`을 반환하는데 config 값은
  `adamw_8bit`이다. **영구 불일치**.
- `parallel=ddp/fsdp2`: capture는 이미 완성돼 있다(`PARALLEL_WRAPPERS`, `applied.py:453`).
  gloo로 CPU 2프로세스 검증이 가능하므로 트랙 2에서 배선까지 끝내고 측정만 GPU로 넘긴다.

---

## 트랙 4 — 첫 파드 (A100): Phase 0 -> 측정 부채

**첫 작업은 Phase 0다.** 매니페스트 18개가 오늘 그대로 돌고, `report.py`가 이미 정확히
렌더한다. 코드 수정 없이 산출물이 나온다 — 어느 프레임워크가 어느 모델을 적재하는지,
axolotl 이미지가 실제로 빌드되는지(`9188fd8`이 빌드 실패를 기록해뒀다), tevatron·unsloth의
API 문제가 무엇인지. **그게 Phase 3의 선행 조건 전부다.**

그 다음, 같은 파드에서 `methodology.md`가 절차까지 적어둔 측정 부채를 닫는다:

1. **3% 임계값 교정** — 동일 파드에서 canonical baseline 5회 반복, 편차 실측(§4).
   트랙 1이 먼저 있어야 잴 수 있다. 실측이 3%를 넘으면 임계값이 아니라 측정 절차를 고친다.
2. **프로파일러 오버헤드**(§1) — 동일 config로 `run=timing` vs `run=profile`, 동일 step/seed/
   데이터 순서, p50/p95 배수 기록. 모델 3종.
   부수: 출처 없는 "20~44%"가 아직 `config_schema.py:237` docstring에 남아 있다. 공유 파일이라
   계약 변경으로 올린다.
3. **deterministic on/off 비용**(§2) — `purpose=probe`로 GPU에서, `compile=none`/`default` 교차.
4. **데이터로딩 병목 판정** — 이게 병목이면 Phase 2 전체가 무의미하다(`PLAN.md:258`).
5. **모델별 visual token 분포** — 196:196:280 보정은 448 정사각 1장 가정이다.
6. **GradCache 수치 등가성 GPU 확인** + Qwen 픽셀 비례 이미지가 `max_seq_len: 2048`을 넘는지.
   넘으면 `Collate`가 멈춘다 — 설계된 정지이고, 픽셀 예산 knob이 필요하다는 신호다.
7. `.item()` 동기화 크기, fork/CUDA 재초기화 경로, `StepTimer` sync 순서 — Wave 3에서
   "측정 안 함"으로 남긴 것들.

---

## 트랙 5 — Phase 3 (프레임워크)

1. **`bench.py:213`이 `framework="native"`를 리터럴로 넘긴다.** `framework=unsloth` 런은
   `assert_matches`에 막힌다. 어댑터가 자기 이름을 넘기는 경로가 필요하다.
2. **프레임워크별 실제 학습 경로 어댑터.** 지금 있는 건 적재+1스텝 probe뿐이다.
3. probe API 수정 4종(tevatron forward 시그니처, axolotl `normalize_config`,
   unsloth `for_training`) — Phase 0 결과가 나와야 무엇을 고칠지 정해진다.

---

## 트랙 6 — 남은 항목

- **`attn` 축을 아무 매니페스트도 쓰지 않는다.** 21개 전부 `sdpa`다. 유일하게 5/5 적용
  가능한 축인데 한 번도 스윕되지 않는다. Phase 2 매니페스트를 만들 때 첫 후보다.
- `config-consumed` 4: `data.push_subset`/`data.subset_rows`는 오탐(레인 A, 호출부 4줄).
  `run.trackio_*`는 진짜 미소비 — **스윕 도중 죽은 파드가 흔적을 안 남긴다**.
- `axes._dataloader`의 `drop_last` 미설정 — `data.limit` 오버라이드 시 p50/p95 잡음.
- `axes.assemble`에 collate를 넘길 경로가 없어 사후 대입이 유일하다(CONTRACTS §2 계약 변경).
- **GPU 타입 혼용 금지에 코드가 0줄이다.** 이미 매니페스트가 섞여 있다(18 A100 / 3 B200).
- **축 분할 금지 가드에 구멍**: `settings` 없이 `overrides`에 값을 직접 넣고 `axis`를 선언하지
  않으면 `check_axis_not_split`(`orchestrate.py:224`)이 보지 못한다.
- 문서: `methodology.md`와 `AGENTS.md:33`이 "`bench.py`가 아직 없다"고 적고 있다(낡음).
  `support-matrix.md`의 "6/7 성공" 표가 존재하지 않는 Dockerfile을 서술한다.
  gemma-4 `audio_tower` 751텐서(37%)가 어떤 freeze 축에도 안 걸린다.
- **`PLAN.md`에 축 구현 작업을 추가한다** — 사용자가 처음 요청한 항목이다. Task 4가
  "조합 정의 후 실행"만 담고 있어서 이 계획서의 트랙 2·3이 통째로 누락돼 있었다.

---

## 순서

```
P0 (Infisical) ─┬─> 트랙 4 첫 파드: Phase 0 ──> 측정 부채 ──> 트랙 3 GPU 축
                │                          └──> 트랙 5 Phase 3
트랙 1 (report.py) ──────────────────────────┘  (3% 교정의 선행)
트랙 2 (CPU 축, workflow 팬아웃) ── 병렬, P0와 무관
트랙 6 ── 아무 때나
```

트랙 2는 P0를 기다리지 않는다. 트랙 1은 측정 부채 1번의 선행이다.

---

## 검증

기존 게이트를 매 단계 유지한다.

```
uv run ruff check && uv run ruff format --check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/audit_plan.py
infisical run --env=dev -- uv run python scripts/env_report.py \
  device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4
```

게이트가 움직여야 하는 방향:

| 체크 | 지금 | 트랙 2 후 | 트랙 3 후 |
|---|---|---|---|
| `axis-values` | 26/43, 6그룹 | loss 2/2, optim 2/3, dataloader 3/4 | 43/43 목표 |
| `config-consumed` | 4 | 2 (trackio는 로거를 붙일 때) | |

**부숴서 확인하지 않은 검사는 증거가 아니다.** 이 저장소는 검사가 자기 부재를 통과로
보고한 사례를 여덟 번 냈고, 여덟 번째는 그 패턴을 잡으려고 만든 변이 하네스 안에서 나왔다.
각 축 구현마다:

- 구현을 되돌리면 새 테스트가 실패하는가 (실패 출력을 기록한다)
- capture probe를 불일치시키면 `purpose=timing` 런이 **숫자를 내기 전에** 죽는가
- `axis-values`가 실제로 움직이는가 — 안 움직이면 구현이 값을 받지 못한 것이다
- GPU 없이 검증 불가능한 부분은 **"측정 안 함"으로 명시한다.** CPU에서 통과하는 vacuous
  테스트를 쓰지 않는다

## 이 범위에서 제외

- **캠페인 실행 자체** (Phase 2 12~18파드, Phase 3 18파드, 품질 런). 이 계획은 그걸 시작할
  수 있는 상태를 만드는 것까지다.
- **`docs/report.md`** — 이 프로젝트의 산출물이고, 숫자가 나와야 쓸 수 있다.
- 첫 파드가 낼 새 결함. 지금까지 매 wave가 그랬듯 실물 실행은 결함을 드러낸다.
