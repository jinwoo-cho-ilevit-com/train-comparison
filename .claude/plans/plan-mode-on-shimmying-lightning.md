# Wave 1 레인 A 게이트 마감 — 3레인 리뷰 findings 수정

## Context

D1(서브셋 손상) 수정 도중 D6(전 행을 디코딩된 픽셀로 누적 → SIGKILL)을 실행으로
발견해 고쳤고, 두 서브셋을 재생성·재핀했다. 그 변경에 대해 module / architecture /
critic 3레인 병렬 리뷰를 돌렸다.

**리뷰가 찾아낸 것 중 가장 중요한 사실: 오늘 만든 speed 핀 `f4363029`는 읽을 수
없다.** `load_dataset(...)`이 CastError로 실패한다. 저장소 README의
`dataset_info`가 여전히 **D1 시절의 4컬럼 스키마**(`pos_image` 없음)를 선언하고
있고, `Dataset.push_to_hub`는 기존 `dataset_info`가 있는 저장소에 덮어쓸 때 새
features를 쓰지 않는다(`datasets/arrow_dataset.py:6953`, `info_to_dump = repo_info`).
D1의 컬럼 손실이 parquet에서는 두 번 고쳐졌는데 저장소 메타데이터에서는 살아남았다.

이 결함이 통과한 경로가 이 수정의 핵심이다: **push가 성공하고 revision이 나온 것을
완료로 취급했고, push된 산출물을 다시 읽어보지 않았다.** 이 저장소가 반복해서 겪은
패턴(검사가 자기 부재를 통과로 보고하는 것)의 또 다른 사례다.

목표는 findings를 닫고, **재-push 후 산출물을 실제로 읽어 게이트를 재계산하는
검증 단계**를 절차에 넣는 것이다.

여기 적힌 결함은 전부 실행으로 재현됐다. 재현하지 못한 것은 "이번에 하지 않는 것"에
있다.

---

## 1. 재-push 전에 반드시 고칠 것

이걸 고치지 않고 다시 올리면 같은 결함을 다시 담는다.

### F1. `push_to_hub`가 features 메타데이터를 갱신하도록 (`scripts/prepare_data.py`)

`build_dataset(...).push_to_hub(...)` → `DatasetDict({"train": ds}).push_to_hub(...)`.
`DatasetDict.push_to_hub`는 `remove_other_splits=True`를 넘기고(`dataset_dict.py`
소스 확인), 그러면 else 분기를 타 features를 새로 쓴다.

### F2. 캐시 히트에서도 업스트림 스키마를 대조 (`scripts/prepare_data.py`)

현재 `check_columns`는 `stream_rows` 안에만 있고 `sample_config`의 캐시 경로는
그것을 건너뛴다. **재현 완료**: 업스트림이 `pos_image`를 떨어뜨린 상황을 만들면
스트리밍 경로는 RuntimeError를 내지만 캐시 경로는 1602행을 조용히 반환한다.
manifest에는 검사한 적 없는 스키마가 검사한 것처럼 기록된다 — D1과 같은 모양이다.

`sample_config`가 캐시 히트 여부와 무관하게 매 실행 `stream.features`를 읽어
`check_columns`를 돌린다. 행을 받지 않으므로 비용은 config당 HTTP 한 번이다.

### F3. `_image_size`가 픽셀까지 검증 (`scripts/prepare_data.py`)

`PIL.Image.open`은 lazy라서 IHDR만 멀쩡하면 IDAT이 깨져도 크기를 돌려준다. D6
이전에는 `datasets`가 매 행 `load()`를 호출해 상류에서 터졌으므로,
`MAX_ROWS_WITH_UNREADABLE_IMAGE = 0`은 **"픽셀이 디코딩된다"**를 뜻했다. 지금은
"헤더가 파싱된다"를 뜻한다. D6이 만든 회귀다.

`with` 블록 안에서 `load()`를 호출한다. 이미지를 보유하지 않으므로 메모리는
한 장 단위로 유지되고 D6이 없앤 누적은 돌아오지 않는다.

**비용 실측**: speed draw 2320장 디코딩에 2.2초, 실패 0건. quality는 약 65초로
추정되며 캐시된 shard에서 돌므로 네트워크가 필요 없다. 게이트 판정은 바뀌지 않는다.

### F4. `DecompressionBombError`를 판독 불가로 계산 (`scripts/prepare_data.py`)

`Exception` 파생이라 `except (OSError, ValueError)`를 빠져나가 런을 죽인다. 그리고
캐시가 있으므로 **재시도마다 같은 자리에서 반복**한다 — kill을 견디려고 만든 캐시가
실패를 영구화한다. 예외 목록에 추가해 이상 이미지로 계산한다.

### F5. push 후 산출물을 다시 읽어 게이트를 재계산 (`scripts/prepare_data.py`)

이 계획에서 가장 중요한 항목이다. 깨진 핀을 완료로 보고한 원인을 닫는다.

push 직후 `load_dataset(repo_id, revision=<새 revision>)`으로 되읽어:

- 컬럼 집합이 `SUBSET_COLUMN_NAMES`와 일치하는가 (F1의 회귀 방지)
- 행 수가 quota 합과 일치하는가
- `rows_without_positive_content` / `rows_without_query_image` /
  `rows_with_unreadable_image`를 **아티팩트에서** 재계산해 manifest와 일치하는가
- 이미지 positive config의 `max_single_positive_share`를 재계산해 일치하는가

`pos_image_path`는 push되지 않으므로 아티팩트에서는 **`pos_image` 바이트 해시**를
identity로 쓴다. critic 레인이 speed draw 7개 config 전부에서 경로↔해시가 1:1이고
manifest 수치가 자릿수까지 재현됨을 실측했다. manifest에 대리 키를 썼다는 사실을
명시적으로 기록한다 — 1:1은 이 draw에서 측정된 것이지 보장이 아니다.

불일치하면 non-zero 종료하고, config에 핀하지 말라고 출력한다.

### F6. 업스트림 커밋 고정 (`trainbench/config_schema.py`, `configs/data/*.yaml`)

`load_dataset(...)`에 `revision=`이 없어 미러의 HEAD를 스트리밍하는데,
`prepare_data.py` docstring과 `PLAN.md`는 "커밋을 고정하고 기록한다"고 주장한다.
코드가 하지 않는 일을 문서가 주장하는 상태다. 캐시가 얹히면서 업스트림 드리프트가
재실행으로도 드러나지 않게 됐다.

`DataConfig.source_revision`을 추가하고(계약 변경 — `docs/CONTRACTS.md` §5에 기록),
`load_dataset`에 넘기고, shard 캐시 키에 포함한다.

**재추출은 불필요하다.** 미러 HEAD는 `9d0fd31789c12a007442de52fe22509e46e49e7d`
(2025-06-17, 21개 커밋 중 최신)이고, 두 draw의 다운로드 URL이 전부 그 sha를
가리킨다(로그 확인). 지금 뽑아둔 행은 그 커밋의 산출물임이 증명된다.

---

## 2. 재-push와 재핀

**두 서브셋 모두 다시 올린다.** quality의 README는 정상이고 행도 동일하지만, 두 핀이
같은 최종 코드 경로에서 나오고 둘 다 F5 검증을 통과해야 설명이 필요 없는 상태가
된다. 캐시된 shard가 있으므로 재추출은 없고 조립·업로드만 든다(speed 약 1분,
quality 약 8분).

핀한 뒤 `configs/data/*.yaml` 주석의 **거짓 서술 두 가지를 정정**한다.

- **"두 서브셋은 같은 스트림을 다른 깊이로 읽은 것"은 20개 중 12개 config에서
  거짓이다.** `shuffle_buffer(want) = min(2000, max(64, want*20))`이라 speed의 작은
  `want`가 2000을 채우지 못해 셔플 자체가 달라진다. 직접 확인: 버퍼가 같은 8개는
  prefix가 성립하고 다른 12개는 성립하지 않는다(HatefulMemes 320, VOC2007 300,
  OK-VQA 340 …). 두 draw는 각각 유효하지만, quality.yaml이 그 대응관계를 **공동
  재핀의 근거로 내세운 것**은 틀렸다.
- **speed.yaml의 "최대 단일 image positive 3.3%"는 게이트가 평가하지 않은 값이다.**
  그 3.3%는 NIGHTS(30행)이고, WebQA(33행)와 함께 `MIN_ROWS_FOR_SHARE_GATE = 50`
  미달로 붕괴 게이트가 아예 돌지 않는다. speed draw의 image-positive 7개 중 5개만
  게이트를 받고, 인용된 최댓값은 안 받은 쪽에서 나왔다.

---

## 3. 독립적으로 고칠 것 (재-push와 무관)

| # | 결함 | 수정 |
|---|---|---|
| F7 | 원자성 테스트가 `write_shard`를 호출조차 안 함 — 원자성을 제거해도 4개 shard 테스트 전부 통과(변이 실험으로 증명) | `write_shard`를 실제로 호출하고 중간에 실패시켜 최종 경로가 생기지 않음을 검사 |
| F8 | `read_shard`가 파일에 없는 컬럼을 null로 backfill — `take_row`가 막던 것을 재개 경로가 되살림 | 읽은 파일의 컬럼 집합이 스키마와 다르면 예외 |
| F9 | 동시 `write_shard`가 고정 `.partial` 이름에서 충돌해 `FileNotFoundError` | staging 이름에 pid 포함 |
| F10 | 손상된 shard를 `read_shard`가 그대로 터뜨려 재시도가 영구 실패 | `sample_config`에서 읽기 실패 시 shard 삭제 후 재추출 |
| F11 | 붕괴 게이트가 건너뛴 config가 어디에도 기록되지 않음 | manifest와 `report()`에 "게이트 미평가 config" 명시 |
| F12 | 메모리 증거가 120초 간격 `ps` 샘플링뿐 — 진짜 피크는 미측정 | `resource.getrusage(RUSAGE_SELF).ru_maxrss`를 manifest에 기록 |
| F13 | 계약 위반 4건 중 1건만 기록 (`pyproject.toml`·`uv.lock`은 레인 F, `tests/test_config.py`는 공유) | `docs/CONTRACTS.md`에 나머지 기록 |
| F14 | `docs/review-findings.md`에 이번 findings 미기재 | D7(읽을 수 없는 핀) 이하 추가, D6 항목에 회귀(F3) 명시 |

F12 관련: "재생성 런이 수정을 증명한다"는 주장은 과했다. 방어 가능한 형태는
**"120초 해상도 샘플링에서 14.2 GB를 넘지 않았고 런이 완주했다"**이다. 누적은 단조인데
RSS가 40377행 14.2 GB에서 65536행 8.6 GB로 **떨어졌다**는 것 자체가 두 샘플이
누적이 아니라 할당 노이즈를 쟀다는 증거다. 반면 17배 비율(독립 재측정 17.6배)과
그로부터 나오는 예측 — 40377행 × 1.18 MiB/행 ≈ 49.8 GB, 실제 SIGKILL 지점 —
은 성립한다. 메커니즘 증거는 살아 있고 결과 증거가 약했다.

---

## 4. 이번에 하지 않는 것

- **`MIN_ROWS_FOR_SHARE_GATE = 50` 자체의 재설계.** 소표본에서 점유율이 draw
  크기에 좌우된다는 원래 논거는 여전히 옳다. 지금 필요한 것은 임계값 변경이 아니라
  **건너뛴 사실의 기록**(F11)이다. 임계값을 바꾸려면 근거 측정이 선행해야 하고,
  그건 이 수정의 범위가 아니다.
- **`fsync` 추가.** `trainbench/record.py`의 `write_json`과 맞추는 것이 일관되지만,
  shard는 재생성 가능한 캐시이고 손상 시 F10이 재추출한다.
- **`build_dataset`의 조립 단계 피크 메모리 최적화.** 실측 12.7 GB로 48 GB 대비
  여유가 크다. F12가 정확한 피크를 기록하기 시작하면 그때 판단한다.

---

## 5. 실행 순서

의존 관계가 있어 순서를 지킨다. 어기면 재-push를 두 번 하게 된다.

1. F1~F4, F6 코드 수정 + 테스트
2. F5 검증기 구현 + 테스트
3. F7~F10 수정 + 테스트
4. `ruff` / `pytest` / `audit_plan.py` 통과 확인
5. speed 재생성·push (캐시 사용, 재추출 없음) → **F5 검증 통과 확인** → 재핀
6. quality 재생성·push → **F5 검증 통과 확인** → 재핀
7. F11~F14 문서·기록
8. 전체 게이트 재실행 후 논리 단위로 커밋 분할

---

## 6. 검증

```
uv run ruff check && uv run ruff format --check
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/audit_plan.py
infisical run --env=dev -- uv run python scripts/env_report.py \
  device=cpu model=qwen3_5_0_8b framework=native data.limit=4 train.batch_size=4
```

재-push 후 반드시 (F5가 자동으로 하지만 독립적으로도 확인):

```
infisical run --env=dev -- uv run python -c "
from datasets import load_dataset
ds = load_dataset(<repo>, revision=<new sha>, split='train', streaming=True)
print(sorted(next(iter(ds))))
"
```

`pos_image`가 컬럼 목록에 있어야 한다. 이것이 오늘 통과했다고 보고했으나 실제로는
실패했던 검사다.

**완료 주장 금지**: F5 검증을 통과하지 않은 revision을 핀하지 않는다. push 성공과
revision 출력은 완료의 증거가 아니다 — 오늘 그렇게 취급해서 읽을 수 없는 핀을
완료로 보고했다.
