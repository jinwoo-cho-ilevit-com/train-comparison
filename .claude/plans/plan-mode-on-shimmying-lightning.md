# 리뷰 후속 수정 — 워크트리 병렬 개발 계획

## Context

3레인 병렬 리뷰(module / architecture / critic)에서 치명 결함 5건이 나왔고, 전부
코드를 직접 실행해 확정했다. 상세는 `docs/review-findings.md`.

핵심은 **측정을 시작할 수 있는 상태가 아니라는 것**이다. 이미지는 빌드되고 config는
통과하고 리포트는 숫자를 뱉지만, (a) 학습 데이터의 positive 54.6%가 중복이고 쿼리
이미지 31.4%가 결측이며, (b) 12개 ablation 축 중 8개를 코드가 한 줄도 읽지 않는다.
지금 Phase 2를 돌리면 이름만 다른 동일 실험이 나오고, 그 표는 이 프로젝트의 가설을
구조적으로 확증한다.

이 계획의 목표는 **유효한 측정이 가능한 상태**에 도달하는 것이다. 작업량이 커서
워크트리로 파일을 격리해 병렬 진행하되, 공유 계약을 먼저 얼리고 매 단계마다 전체
계획과의 정합을 기계적으로 점검한다.

확정된 설계 결정 두 가지 (2026-08-01, 사용자):
- **SFT 대조군을 두지 않는다.** 주장 범위를 "임베딩 학습 내부의 축별 효과 + 모델별
  병목"으로 좁히고 `PLAN.md`에서 SFT 비교 주장을 철회한다.
- **8개 축을 전부 구현한 뒤** Phase 2를 시작한다.

---

## 파동 구조 — 왜 이렇게 나누는가

컨벤션 09: 워크트리는 **파일 격리 수단**이므로 파일 편집이 겹칠 때만 도입하고,
**공유 계약은 실행 전에 얼린다**. 따라서 병렬화 이전에 순차 구간이 하나 필요하다.

```
Wave 0 (순차, 병렬 금지)  공유 계약 확정
        │
        ├─ Wave 1 (워크트리 4개 병렬)  A 데이터 / B 코어정확성 / C 오케스트레이션 / E 문서
        │
        ├─ Wave 2 (워크트리 2개 병렬)  D 축구현 / F 이미지·env
        │
        └─ Wave 3 (순차)  G 하네스 + baseline 게이트 + 품질 가드레일
```

Wave 0을 병렬화하면 안 되는 이유: `config_schema.py`와 신설 `applied.py`가 이후 모든
레인의 입력이다. 이걸 각자 고치면 4개 워크트리가 서로 다른 스키마 위에서 개발하게
되고 병합이 불가능해진다.

Wave 3을 병렬화하지 않는 이유: 하네스는 B(pooling)와 D(축 적용)의 결과물을 모두
소비하고, baseline 게이트는 이후 모든 결과가 통과하는 지점이라 마지막에 한 번에
확정해야 한다.

---

## Wave 0 — 공유 계약 확정 (순차, 메인 워크트리)

**이 구간이 끝나기 전에는 어떤 워크트리도 만들지 않는다.**

| 산출물 | 내용 |
|---|---|
| `trainbench/config_schema.py` | 8개 축의 누락 필드 추가: `TrainConfig.offload`, `DataloaderConfig.pretokenize`, `ParallelConfig.cross_device_negatives`. 검증기 추가: `warmup_discard_steps < steps`, `purpose=profile`인데 `profiler=false` 금지, `batch_size <= data.limit`, `quality.yaml`의 `revision: null` 금지 |
| `trainbench/applied.py` (신설) | **요청값 vs 실제 적용값** 계약. `AppliedState` 데이터클래스 + `capture(model, config) -> AppliedState` + `assert_matches(requested, applied)` 시그니처만 확정. 구현은 Wave 2(D) |
| `trainbench/probe/types.py` | `Check.expected_failure: bool` 추가. unsloth의 `fast_sentence_transformer_accepts_vlm`처럼 실패가 예상 결과인 체크가 셀 전체를 FAIL로 만드는 문제 해소 |
| `trainbench/record.py` | 레코드 스키마 확정: `applied`, `image_digest`, `git_commit`(env 우선), 호스트 스펙(`os.process_cpu_count()` + cgroup, `/proc/meminfo`, `torch.version.cuda`), `_TRACKED_PACKAGES`에 `flash-attn`/`causal-conv1d`/`bitsandbytes`/`deepspeed`/`torchvision` 추가 |
| `docs/CONTRACTS.md` (신설) | 위 인터페이스를 문서로 고정. Wave 1~2의 모든 레인이 이 파일을 계약으로 삼는다 |
| `docs/model-spec.md` (신설) | 아래 "모델별 규격 검증"의 산출물. B와 D의 입력 |

### 모델별 규격 검증 (Wave 0에 포함, HuggingFace MCP 사용)

현재 probe는 세 모델에 **동일한 generic 경로**를 쓴다 — `AutoModel` + 자체
`last_token_pool` + 자체 `info_nce`. 모델이 의도한 사용법과 다르면 측정 대상이
"모델"이 아니라 "잘못 쓴 모델"이 된다. Wave 0에 넣는 이유는 B(pooling)와 D(freeze
대상)가 이 결과를 입력으로 받기 때문이다.

HF MCP(`hub_repo_details`, `hf_fs`)로 저장소 파일을 직접 읽어 확인하고, **추측으로
채우지 않는다**(컨벤션 16). 확인하지 못한 항목은 "미확인"으로 남긴다.

| 질문 | 읽을 아티팩트 |
|---|---|
| 공식 pooling이 last-token인가 | `modules.json`, `1_Pooling/config.json`, `config_sentence_transformers.json`, 모델 카드 |
| 쿼리/문서에 붙는 instruction·prompt 포맷 | 모델 카드 사용 예시, `chat_template`, `config_sentence_transformers.json`의 prompts |
| 정규화·유사도 함수·temperature | 모델 카드, ST config |
| MRL(Matryoshka) 지원 차원 | 모델 카드, ST config. 지원하면 임베딩 차원이 축이 될 수 있다 |
| gemma-4 PLE 파라미터 **실제 이름** | `model.safetensors.index.json`의 weight map. 현재 `per_layer`/`altup` 문자열 매칭은 추측이며, 틀리면 `matched_count: 0`인데 `ok: True`로 통과한다 |
| 이미지 placeholder 확장 위치(processor vs 모델 내부) | `preprocessor_config.json`, `processor_config.json`. 후자면 visual token 카운트가 1 같은 값으로 나와 모델 간 정규화 상수가 오염된다 |
| 동적 해상도 범위 | `preprocessor_config.json`의 `min_pixels`/`max_pixels`(Qwen) vs `vision_soft_tokens_per_image`(gemma) |
| `padding_side` | `tokenizer_config.json`. gemma-4는 `left`로 확인됨 — `last_token_pool` 결함이 노출되는 유일한 모델 |
| LoRA target module 관례 | 모델 카드/공식 레시피. 현재 `all-linear`는 "모델별 target module 인식" 질문을 회피한다 |

**산출물**: `docs/model-spec.md` — 항목마다 근거 파일 경로와 revision을 남긴다.
generic 경로와 공식 규격이 다른 항목은 **차이를 명시**하고, 차이를 수용할지
(단순성·비교 공정성) 모델별로 맞출지(현실성)를 결정해 기록한다. 이 결정 자체가
리포트의 한정 조건이 된다.

**완료 조건**: `uv run pytest` 통과 + 스키마 변경으로 기존 config 53개가 전부 해석됨
+ `docs/model-spec.md`의 모든 행이 근거 또는 "미확인"으로 채워짐.

---

## Wave 1 — 병렬 워크트리 4개

각 레인은 자기 파일만 만진다. 겹침 없음을 아래 표로 보장한다.

### A. 데이터 재생성 (`wt/data`)

가장 심각한 결함이고 다른 모든 것의 선행 조건.

| 파일 | 변경 |
|---|---|
| `scripts/prepare_data.py` | `SUBSET_COLUMNS`를 config별 기대 스키마로 대체. `pos_image` 보존. 기대 컬럼이 없으면 **예외**(현재는 `row.get()`이 조용히 None) |
| `configs/data/speed.yaml`, `quality.yaml` | 손상된 revision 고정 해제, 재생성 후 새 revision 고정. `quality.yaml`도 서브셋 생성 |
| `tests/test_data.py` (신설) | `proportional_quota(total < len(counts))` 경계, 컬럼 스키마 검증 |

manifest에 추가할 품질 지표 — **이게 없어서 손상을 놓쳤다**:
`rows_without_query_image`, `rows_without_positive_content`, `distinct_pos_text_count`,
`duplicate_pos_text_ratio`, config별 시퀀스 길이·이미지 해상도 분포(p50/p95).
**임계값 초과 시 push 거부.**

`pos_text`가 `<|image_1|>` 같은 MMEB placeholder를 담고 있으므로 모델별
`apply_chat_template` 변환이 필요하다는 점도 여기서 문서화(구현은 Wave 3).

### B. 코어 정확성 (`wt/core`)

| 파일 | 변경 |
|---|---|
| `trainbench/embedding.py` | `last_token_pool` left padding 수정. Qwen 공식 구현처럼 `attention_mask[:, -1].sum() == batch`로 left를 감지해 분기. **거짓 주석 제거** |
| `trainbench/probe/steps.py` | `visual_token_count`에 `image_token_id` -> `image_token_index` -> `get_text_config()` fallback. 반환값 타당 범위(10~2000) 검사. `_tokenize` 클로저 5중 복제를 헬퍼로 흡수 |
| `trainbench/probe/native.py` | `_ple_report`가 `matched_count == 0`이면 **실패**로 기록(현재는 아무것도 못 찾아도 `ok: True`). `requires_grad` 스냅샷 후 복원 |
| `trainbench/probe/registry.py` | report를 registry가 만들어 `module.run(config, device, report)`로 전달 → 부분 결과 보존 |
| `trainbench/seed.py`, `scripts/verify_env.py` | `set_seed(warn_only=)` 추가. probe는 `warn_only=True` — 결정적 구현이 없는 연산이 "프레임워크 미지원"으로 오기록되는 것 방지 |
| `tests/test_embedding.py` | **left padding 테스트**(현재 right만 검증해 결함을 통과시켰다), `is_finished`, registry 부분 실패 보존 |

### C. 오케스트레이션 견고화 (`wt/orch`)

| 파일 | 변경 |
|---|---|
| `trainbench/pods.py` | `is_finished`를 "runtime이 non-null -> null로 **전이**" 또는 `desiredStatus == EXITED`로 수정. `get()` 예외를 `unknown` 센티널로. pod별 개별 deadline |
| `scripts/orchestrate.py` | `pod_env`에 `INFISICAL_TOKEN`·`TRAINBENCH_GIT_COMMIT`·이미지 digest 추가. `run=probe` 하드코딩 제거 → `configs/experiment/` 소비. launch 직후 ledger 증분 기록 |
| `configs/experiment/*.yaml` (신설) | 파일 1개 = pod 1개 작업(모델 + 축그룹 + override + **필수 `baseline:`**). 컨벤션 02 §3 "재실행 가능해야 실험이다" |
| `docker/entrypoint.sh` | 결과 파일 없으면 fallback 레코드 기록 후 publish. `cd ... \|\| exit`. 빈 `--projectId` 플래그 제거. `timeout Nm` 자살 장치 |
| `scripts/publish_result.py` | `create_repo(exist_ok=True)`, 백오프 재시도, probe 시작 전 `started.json` |
| `scripts/report.py` | 타임스탬프 최신 우선 병합 + 중복 경고, 파싱 실패 파일 스킵, `expected_failure` 제외, "기동했으나 결과 없음"을 미확인과 구분 |

### E. 문서 정정 (`wt/docs`)

코드와 겹치지 않아 안전하게 병렬 가능.

- `PLAN.md`: **SFT 비교 주장 철회**, 핵심 가설을 "임베딩 학습 내부 축별 효과 + 모델별
  병목"으로 재작성. Liger를 "무력화"가 아니라 "FLCE 경로만 정의상 비활성"으로 정정.
  저장소 구조의 미존재 파일 정리. 데이터 출처 표기 수정
- `README.md`: `uv sync --group dev` → `--extra compose` (**현재 문서대로 하면 pytest가
  hydra 부재로 실패**)
- `AGENTS.md`: `env_report.py`를 "smoke"로 지칭한 부분 정정(모델 적재도 step도 없음)
- `docs/methodology.md` **신설**: `config_schema.py`와 `AGENTS.md`가 참조하는데 부재.
  "torch.profiler 20~44%"의 출처 명기 또는 미측정 표기 (현재 4곳에 출처 없이 사실로
  반복 — 컨벤션 16 위반)
- `docs/support-matrix.md`: "env 5/5 성공"에 "`uv lock` 성공이며 설치·빌드·실행 아님"
  한정 추가. native 셀에 "macOS CPU fp32, 텍스트 위주, 3모델 중 2모델" 조건 명시

---

## Wave 2 — 병렬 워크트리 2개 (Wave 1 병합 후)

### D. 8개 축 구현 (`wt/axes`)

Wave 0의 `applied.py` 계약을 구현하고 축을 실제로 배선한다.

축별로 **적용 지점 + 검증 방법 + 검증 가능 GPU**를 함께 정의한다. FA4/NVFP4는 B200
전용이라 A100에서는 검증조차 불가능하므로, config 검증기가 GPU와 축의 조합을 거부해야
한다.

| 축 | 적용 | 검증(applied) | 필요 패키지 |
|---|---|---|---|
| attn | `attn_implementation=` | `model.config._attn_implementation` | `flash-attn` (fa2/3/4) |
| kernel liger | Liger 패치 | 패치된 심볼 목록 | `liger-kernel` |
| kernel fla | 설치 여부 | GDN fast path 실사용 | `flash-linear-attention` + `causal-conv1d` |
| precision | TE / torchao | 활성 recipe | `transformer-engine` |
| compile | `torch.compile` | `torch._dynamo.utils.counters` graph break 수 | - |
| optim | 옵티마이저 생성 | 클래스명 + param group | `bitsandbytes`, Muon 구현체 |
| freeze | `requires_grad` | 실제 학습 파라미터 수 | - |
| dataloader | packing/DALI/pretokenize | 실제 경로 | `nvidia-dali` |
| parallel | FSDP2/ZeRO/all-gather | world size + 전략 | `deepspeed` |

**`purpose=timing`에서 요청값 ≠ 적용값이면 런 실패.** 이것이 이 프로젝트에서 가장
중요한 단일 안전장치다 — 없으면 sdpa로 폴백된 런이 "FA3 1.4배"로 리포트에 실린다.

### F. 이미지·env 갱신 (`wt/images`)

- `envs/*/pyproject.toml`에 축별 패키지 추가 후 재-lock. **패키지 간 충돌은 Phase 0
  결과로 기록**(예: unsloth의 torch<2.12가 특정 flash-attn과 양립 불가)
- `Dockerfile.framework`: `COPY trainbench`를 `uv sync` **뒤로** 이동 — 현재는 소스
  한 줄 수정이 axolotl 237패키지 sync를 매번 재실행시킨다
- `build-images.yml`: 이미지에 digest 태그 부여(현재 `latest`만이라 어떤 이미지가
  그 숫자를 냈는지 사후 특정 불가). GHA 캐시 10GB 상한 대응
- **axolotl 빌드 실패 원인 규명** (미확인 상태)
- `USE_HF=1` (ms-swift가 기본 ModelScope hub라 gemma-4를 못 찾을 위험)

---

## Wave 3 — 하네스 (순차, 메인 워크트리)

### G. 측정 하네스 + 게이트

- `scripts/bench.py` + pod 내부 **sweep 러너**: 모델 1회 적재 후 축 sweep. 현재 진입점은
  프로세스당 모델 1회 적재라 5B 모델에서 오버헤드가 측정을 지배한다. PLAN의 "35h"는
  이 러너를 전제해야 성립하는데 요구사항으로 적혀 있지도 않았다
- `trainbench/metrics/`: throughput, peak VRAM, step p50/p95. **MFU는 tokens/s를 1차
  지표로 격하** — GDN linear attention / PLE lookup / sliding window에서 표준 FLOP
  공식이 세 모델 모두 깨진다. 모델별 공식을 유닛 테스트로 검증한 뒤에만 제시
- **baseline 게이트**: 3% 임계값을 **동일 pod 동일 설정 5회 반복 편차를 실측한 뒤**
  그 2~3배로 교정. 현재 값은 멀티테넌트 클라우드 편차보다 타이트할 수 있고, 그러면
  재실행이 비용의 지배 항이 된다
- **데이터로딩 병목 선판정**: 이것이 병목이면 Phase 2 전체가 무의미
- **품질 가드레일**: Recall@k 이전에 **축별 수치 등가성 검사**(같은 seed로 N step 후
  baseline 대비 loss/grad norm tolerance). GradCache는 전용 등가성 테스트 필수 —
  검증 없이 재면 GradCache 버그가 GradCache 속도 향상으로 리포트된다
- **모델별 visual token 분포 실측**: 합성 이미지 1장 기반 196:196:280 보정은 448 정사각
  1장에서만 성립. Qwen은 동적 해상도(픽셀 비례), gemma-4는 280 고정이라 모델 간
  토큰 예산 고정이 원리적으로 불가능. 실제 서브셋에서 분포를 재고, 불가능하면 리포트
  범위를 "모델 내 축 효과만 비교"로 명시적으로 좁힌다

---

## 매 단계 정합 점검 — 기계화

사용자 요구: "개발이 마칠 때마다 전체 계획에서 누락/오류가 없는지 항상 점검".
사람이 체크리스트를 읽는 방식은 이번에 이미 실패했다(서브셋 손상을 검증 지표가
완벽 통과로 보고했다). **가능한 만큼 기계로 만든다.**

### `scripts/audit_plan.py` (신설, Wave 0에서 만들고 매 wave 끝에 실행)

기계적으로 검증 가능한 불변식만 담는다. 실패하면 non-zero.

1. **config 손잡이가 전부 소비되는가** — 모든 config leaf 필드가 코드 어딘가에서
   읽히는지. 안 읽히면 실패. (D4를 잡았을 검사)
2. **PLAN.md에 적힌 파일이 실제로 존재하는가** — 저장소 구조 절의 경로 전수 확인
3. **support-matrix의 수치가 커밋된 아티팩트를 참조하는가** — 현재 `.gitignore`가
   `outputs/`를 제외해 실측 로그가 저장소에 하나도 없다. `docs/evidence/`를 만들고
   결과 JSON을 커밋 대상으로
4. **축 ↔ 패키지 정합** — config group에 있는 축의 필요 패키지가 해당 env lock에
   존재하는지
5. **문서 명령이 실제로 도는가** — README/AGENTS의 명령을 dry-run
6. **데이터 revision이 null이 아닌가**, manifest 품질 지표가 임계 내인가
7. **모델별 규격 정합** — `docs/model-spec.md`에 기록된 각 모델의 공식 규격(pooling,
   prompt 포맷, PLE 파라미터 이름, placeholder 확장 위치, `padding_side`)이 HF에
   호스팅된 현재 파일과 여전히 일치하는가. HF MCP로 재조회해 대조하고, 업스트림이
   바뀌었으면 실패시킨다 — 모델 저장소는 갱신되며 우리 구현은 그 시점 스냅샷 위에
   서 있다

### 각 wave 종료 게이트 (사람 + 에이전트)

1. `uv run pytest` + `ruff` + `scripts/audit_plan.py` 전부 통과
2. **작성자와 분리된 리뷰 레인 1개** (컨벤션 09). 변경이 2+ 모듈이거나 인터페이스를
   건드리면 3레인
3. `docs/review-findings.md`의 해당 항목을 해소 표시하고, **새로 발견된 것을 추가**
4. 통합 검증: 워크트리 병합 후 전체 테스트 1회

---

## 워크트리와 team mode의 역할 분담

둘은 대안이 아니라 직교한다. **워크트리는 파일 격리**, **team mode는 공유 task list
기반 조율**이다. 병용한다.

| Wave | 워크트리 | team mode | 근거 |
|---|---|---|---|
| 0 계약 확정 | 미사용 | 미사용 | 병렬화 자체가 금지 구간 |
| 1 (A/B/C/E) | 사용 | 사용 | 파일 겹침 방지 + 진행 상태를 오케스트레이터 컨텍스트 밖에 유지 |
| 2 (D/F) | 사용 | 사용 | 위와 동일 |
| 3 하네스 | 미사용 | 미사용 | baseline 게이트가 이후 모든 결과의 통과 지점이라 한 번에 확정 |

**team mode를 쓰는 진짜 이유는 처리량이 아니라 상태 외부화다.** Wave가 4개, 레인이
6개이므로 진행 상태를 오케스트레이터의 컨텍스트에 두면 중간에 유실된다. task list가
그것을 대신한다.

**team mode가 풀지 못하는 것**: 이번에 발견된 결함 5건은 전부 정확성·판단 실패였고
처리량 부족이 아니었다. 손이 더 많아도 잡히지 않았을 것이고, 실제로 잡은 것은
독립적인 검증 관점이었다. 따라서 병렬 인원보다 **매 wave 게이트의 분리된 리뷰
레인**이 우선한다. 이 우선순위를 뒤집지 않는다.

## 워크트리 운용

```
.claude/worktrees/wt-data     A
.claude/worktrees/wt-core     B
.claude/worktrees/wt-orch     C
.claude/worktrees/wt-docs     E
.claude/worktrees/wt-axes     D   (Wave 2)
.claude/worktrees/wt-images   F   (Wave 2)
```

- 각 워크트리는 Wave 0 병합 커밋에서 분기
- **파일 소유권은 위 표가 유일한 기준.** 다른 레인의 파일을 고쳐야 하면 직접 고치지
  말고 계약 변경으로 올린다
- 병합 순서: A → B → C → E (충돌 최소 순). 각 병합 후 전체 테스트
- `configs/`는 A(data), C(experiment), D(axes)가 서로 다른 하위 디렉터리만 만진다

---

## 이번 범위에서 제외

- 실제 Phase 2 측정 실행 — Wave 3 완료 후 별도
- 프레임워크 probe 4종의 API 수정(tevatron forward 시그니처, axolotl `normalize_config`,
  unsloth `for_training`) — 이미지가 빌드돼야 검증 가능하므로 Wave 2(F) 이후 별도 wave
- B200 pod 기동 — Wave 3 완료 및 audit 통과 전까지 금지

---

## 검증

각 wave 종료 시:
```
infisical run --env=dev -- uv run ruff check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/audit_plan.py
```

Wave 3 종료 시 추가:
```
# 데이터 무결성
infisical run --env=dev -- uv run python scripts/audit_plan.py --check data-quality

# 축 적용 검증이 실제로 막는가 (패키지 없는 축을 요청하면 실패해야 함)
infisical run --env=dev -- uv run python scripts/bench.py device=cpu attn=fa4 run=timing
# -> 실패해야 정상. 성공하면 축 검증 계층이 동작하지 않는 것

# CPU 소수 샘플 E2E
infisical run --env=dev -- uv run python scripts/bench.py device=cpu data.limit=4 train.steps=2
```

**완료 주장 금지**: TODO/stub/skip 잔존, 실행 로그 없는 통과 주장, `audit_plan.py`
실패 상태에서의 다음 wave 착수.
