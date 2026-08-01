# Wave 2 게이트 마감 — 병합, 감사 계층 수리, LoRA 선행 작업

## Context

Wave 2를 D(축 구현)/F(이미지·env) 두 워크트리로 병렬 진행해 둘 다 완료했다.

```
axis-wired      12축 미배선 → 2축      (D)
axis-packages   12건 → 3건             (F)
config-consumed 16개 → 13개            (D)
pytest          281 → 331
```

3레인 게이트 리뷰(module / architecture / critic)가 이 숫자들이 **보이는 것만큼을
뜻하지 않는다**는 것을 찾아냈고, 세 건은 직접 재현했다.

- **감사 요약줄이 이번 wave 전체를 보지 못한다.** 병합 트리에서 D의 `applied.py`를
  통째로 되돌려도 `7/11 passing, 0 new failure(s), 0 newly fixed`가 **바이트 단위로
  동일**하다. 상세줄은 바뀌지만(축 2개→12개), 게이트가 판정에 쓰는 줄은 눈이 멀어
  있다. baseline이 `{체크: 메시지}`뿐이라 이미 실패 중인 체크 안의 크기 변화를
  양방향 모두 못 본다.
- **`axis-wired`는 knob 이름의 멤버십 검사다.** 축이 기본값 말고 아무 값도 적용하지
  못해도 통과한다. 44개 config 값을 실제로 `axes`에 통과시킨 결과 **26개만 적용
  가능**하고, ablation 축 본체는 19/37이다. 12개 축 중 7개가 값 하나만 받는다.
  `review-findings.md` D4가 "이 상태로 Phase 2를 돌리면 이름만 다른 동일 실험이
  나온다"고 적은 바로 그 상태인데, 그것을 잡으라고 만든 체크의 숫자가 내려갔다.
- **`AXIS_PACKAGES`가 존재하지 않는 배포판을 가리킨다.** `nvidia-dali`는 NVIDIA가
  오설치를 경고하려고 올린 자리표시자이고 `grad-cache`는 404다. 올바른 설치가
  영원히 체크를 만족시킬 수 없고, 감사는 지금 **패키지가 설치된 축에 대해 없다고**
  출력한다.

내가 지시한 계약 변경이 만든 결함도 하나 있었다. `tests/test_applied.py`를 합성 축
형태로 바꾸면서 `capture`의 "probe 없는 축 → undetermined" 분기 커버리지가 사라졌고,
한 단어 변이로 `precision.name`/`train.offload`가 "적용됨"으로 인증되는데 325개
테스트가 전부 통과했다. D가 고쳤다(집합에서 유도하되 **빈 집합이면 실패**).

사용자 결정(2026-08-01): **LoRA는 Wave 3 착수 전 필수로 구현한다.**

---

## 1. 병합 (D + F)

두 레인의 파일 집합은 완전히 disjoint이고, 실제 병합 워크트리에서 스위트를 돌려
확인했다(325 passed 당시 4회 연속, ruff clean, audit 0 new failure). D가 그 뒤
6개 테스트를 더 추가해 331이 됐으므로 병합 후 다시 돌린다.

논리 단위로 커밋을 나눈다: D의 축 배선 / F의 env·이미지 / 계약 변경 기록.

**병합 후 확인**: `pytest` 331 passed, `audit_plan.py` `axis-wired` 2 + `axis-packages` 3,
`ruff` clean. 숫자가 다르면 병합이 무언가를 삼킨 것이다.

## 2. 감사 계층 수리 — 공유 파일, 계약 변경으로 기록

세 건 모두 `scripts/audit_plan.py`(수정 금지)와 `docs/audit-baseline.json`을 건드린다.
`docs/CONTRACTS.md` §5에 이력을 남긴다.

### 2-1. baseline이 개수를 기록하게 한다 (가장 먼저)

`{체크: 메시지}` → `{체크: {note, count}}`. 감사가 현재 개수를 baseline과 대조해
**"N new failure(s)"에 더해 크기 증감**을 출력한다. 이게 없으면 나머지 수리의 효과도
같은 방식으로 보이지 않는다.

이 변경 자체를 검증하는 방법: D의 `applied.py`를 되돌린 트리에서 감사가 **회귀를
보고하는지** 확인한다. 지금은 안 한다.

### 2-2. `axis-values` 신설

`axis-wired`는 그대로 둔다 — "축 하나에 적용 지점 하나, capture probe 하나"라는
질문은 여전히 유효하고 baseline 항목이 그것을 추적한다. 대신 **각 config 그룹의
variant를 전부 `axes.patch`/`load_kwargs`/`assemble`/`step_context`에 통과시켜
적용 가능한 값의 개수를 세는** 체크를 새로 만든다.

critic 레인이 같은 열거를 이미 스크립트로 돌렸다(`enum_axes.py`, 26/44). 그 접근을
가져오되 저장소 코드로 옮긴다. 스키마 검증기가 최소 override를 거부하는 조합이
있으므로 그룹별 정상 동반값이 필요하다.

현재 개수로 baseline을 잡고, 값이 늘면 줄어들게 한다.

### 2-3. `AXIS_PACKAGES` 이름 정정 + 통과 문구 정직화

`nvidia-dali` → `nvidia-dali-cuda130`, `grad-cache` → `gradcache`.

**`axis-packages` baseline 줄은 유지한다.** 이름을 고치면 `dataloader/dali`가 PASS로
바뀌는데 `axes._dataloader`는 `backend=dali`를 여전히 거부한다 — 아무 런도 켤 수 없는
축에 초록불이 켜진다. 통과 메시지를 "어느 락엔가 이름이 있다"는 실제 의미로 바꾸고,
빌드·설치·import를 증명하지 않는다는 것을 체크 docstring에 적는다.

## 3. LoRA — Wave 3 선행 (사용자 결정)

`peft=lora`/`qlora`가 지금 `UnappliedAxis`로 죽는다. 거부 자체는 옳다 — 대안은 LoRA
라벨을 단 full finetuning이고, D가 근거를 정확히 적었다(peft를 어느 env에도 설치하지
않아 `freeze.*` x peft 의미론을 실제 모델로 검증할 수 없었다).

순서가 있다.

1. **F 영역**: `peft`를 축 패키지들과 같은 방식으로 env에 넣고 재-lock.
2. **의미론 결정**: peft는 base 파라미터를 전부 얼린다. `freeze.ple=false` 요청이
   "얼지 않음"인지 "peft가 얼린 것에 더해 얼지 않음"인지 정해야 한다. 실제 peft
   모델로 확인한 뒤 정한다 — 추측으로 정하면 그 정의가 측정 결과의 해석을 바꾼다.
3. **D 영역**: `_peft` 적용 + capture. capture는 `peft_config`/래퍼 클래스/4bit
   여부를 읽는 기존 구현이 있으므로 확장한다.
4. `configs/peft/{lora,qlora}.yaml`은 이미 있다.

**Wave 3은 이것이 끝나기 전에 착수하지 않는다.**

## 4. 레인 E 인계 (이번에 직접 고치지 않음)

`docs/support-matrix.md`와 `PLAN.md`는 레인 E 소유다. `docs/review-findings.md`의
인계 절에 추가한다.

- **"이미지 7개 중 6/7 성공" 표가 존재하지 않는 Dockerfile을 서술한다.** F가
  `Dockerfile.base`(libffi-dev)와 `Dockerfile.framework`(3단 sync)를 바꾸고
  `envs/native`를 약 30 → 142 패키지로 키웠다. 그 표는 9363197 이전 정의에 대한
  측정값이고, 진부화 표시가 없다.
- **gemma-4의 `audio_tower` 751 텐서(전체 2011의 37%)를 어떤 freeze 축도 건드리지
  않는다.** 이미지만 쓰는 벤치마크에서 이 타워는 학습되지도 얼려지지도 않은 채
  옵티마이저 상태를 차지한다. gemma-4의 메모리·optimizer 축 수치가 이것을 포함하므로
  `docs/methodology.md`에 기록이 필요하다.
- F가 `docs/support-matrix.md`(E 소유)에 198줄을 쓰고 계약 위반을 기록하지 않았다.
  더 나쁜 것은 **F의 계약 변경 요청이 그 파일 안에만 있다**는 점 — `CONTRACTS.md`를
  읽는 사람은 못 본다. §5에 기록한다.

## 5. 기록만 하고 지금 고치지 않는 것

전부 지금은 활성 결함이 아니지만 다음 레인이 축을 켜는 순간 살아난다.
`docs/review-findings.md`에 남긴다.

| | 내용 |
|---|---|
| D-3 | `mnrl.axis_value`/`axis_cross_device_negatives` 리터럴이 config 복사로 바뀌어도 테스트가 안 죽는다. 지금은 `_loss`가 먼저 raise해서 등가 변이다. cached_mnrl이나 cross-device가 구현되는 순간 capture가 요청 에코가 된다 |
| D-8 | 계약 §2가 고정한 유일한 순서 불변식(모델 변형 → 옵티마이저)에 테스트가 없다. FSDP2가 붙는 순간 load-bearing이 된다 |
| D-11 | `PARALLEL_WRAPPERS`가 FSDP**1** 클래스 이름을 `fsdp2` 값에 매핑한다. torch FSDP2는 래퍼를 만들지 않는다. fail-safe 방향이지만 이름 검증은 DDP 하나뿐이다 |
| D-12 | `_capture_kernel`은 **클래스 정의**만 본다. liger는 메서드도 붙이므로 그 경우 `kernel=none`으로 읽힌다 |
| F8 | kernel/peft/parallel capture는 실물 객체를 한 번도 보지 못했다. 전부 `SimpleNamespace`/가짜다. Wave 3에서 GPU pod의 `verify_env.py`로 실물 대조가 필요하다 |
| F3 | `axis-packages`는 락 문자열 검사다. flash-attn/deepspeed/transformer-engine/DALI는 아무도 빌드한 적이 없고 넷 다 nvcc 소스 빌드다. GHA 6시간 잡 제한 저촉 여부는 첫 빌드에서만 알 수 있다 |

## 6. 검증

```
uv run ruff check && uv run ruff format --check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/audit_plan.py
infisical run --env=dev -- uv run python scripts/env_report.py \
  device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4
```

**새 체크와 baseline 변경은 통과만으로 완료로 치지 않는다.** 각각에 대해:

- `axis-values`: 축 값 하나를 적용 가능하게 만들었을 때 개수가 오르는지, 거부로
  되돌렸을 때 내려가는지
- baseline 개수 기록: D의 `applied.py`를 되돌린 트리에서 **회귀로 보고되는지**
  (지금은 보고되지 않는다는 것을 이미 확인했다)
- `AXIS_PACKAGES`: 이름을 되돌리면 다시 3건이 되는지

이 저장소는 검사가 자기 부재를 통과로 보고한 사례를 오늘 하루에만 네 번 냈다
(D1 행 수 게이트, D6 소표본 스모크, D7 형식만 보는 `data-pinned`, 그리고 내가 만든
capture 커버리지 구멍). **통과하는 검사는 그 대상을 부숴서 실패시켜 본 뒤에만
증거다.**
