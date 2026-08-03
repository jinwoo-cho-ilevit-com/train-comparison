# blockerfix — 2차 라운드 blocker 5건 (단독)

> 먼저 읽는다: `HAZARDS.md`, **`.plans/review/findings-round2.md`** (다섯 항목의 실패
> 시나리오·재현 명령·검증 출력이 거기 그대로 있다).
> **범위는 blocker 다섯뿐이다.** major 23건과 minor 24건은 이번에 손대지 않는다 —
> 1차 수정이 40건을 고치면서 28건을 만든 전례가 있어 범위를 좁혔다.

## 진입 상태 — 게이트는 전부 초록이다

```text
uv run ruff check && uv run ruff format --check   ->  통과, 110 files
infisical run --env=dev -- uv run pytest          ->  1200 passed, 0 failed
infisical run --env=dev -- uv run pytest tests/contract -q  ->  122 passed
infisical run --env=dev -- uv run python scripts/audit_plan.py
  ->  13/15 passing, 0 new failure(s), 0 newly fixed, 0 grew, 0 shrank  (exit 0)
```

**초록이 이 다섯을 보지 못한다는 것이 발견의 내용이다.** 게이트를 초록으로 유지하는 것은
완료 조건이지 증거가 아니다.

## Owns

```
scripts/bench.py
trainbench/kernels.py
trainbench/loader.py
tests/fixtures/run_record.sample.json      B3 에 한해
tests/test_smoke_cpu.py
tests/test_kernels.py
tests/test_loader.py
tests/test_report.py
```

`tests/contract/*.py` 는 고치지 않는다. B3 은 **fixture** 를 고치는 것이고 계약 테스트가
아니다. fixture 를 고칠 때 **계약이 그 자리에서 무엇을 단언하는지 먼저 읽는다.**

---

## B1 (blocker) — 타이밍 런이 체크포인트·코퍼스를 받기 전에 오프라인이 된다

`scripts/bench.py:680` `close_kernel_fetch_doors(config)` 가 `HF_HUB_OFFLINE=1` 과
`huggingface_hub.constants.HF_HUB_OFFLINE=True` 를 켠 **뒤에** 체크포인트와 코퍼스를 Hub 에서
읽는다. 캐시가 빈 파드(HF_HOME 비어 있고 볼륨 없고 Dockerfile 에 사전 다운로드 없음)에서
`AutoModel.from_pretrained` 가 `OSError("We couldn't connect to ...")`, 이어서 `load_pairs` 가
`ConnectionError` 로 죽는다. **모든 timing/quality 설정이 결과 파일 없이 exit 1 이다.**

## B2 (blocker) — 그리고 캐시가 있으면 더 나쁘다: 고정 revision 이 조용히 버려진다

같은 원인, 다른 결과. 오프라인에서 `datasets` 는 `load_dataset(..., revision=...)` 요청에
**로컬 캐시에 이미 있는 빌드를 내준다.** 검증자가 실측한 그것이 `CORRUPT_DATA_REVISIONS` 가
거부하려고 존재하는 **D1 손상 revision** 이었다. 그런데 런 레코드에는 고정 revision 이
그대로 실린다 — 손상된 데이터로 잰 숫자가 옳은 데이터의 라벨을 달고 발행된다.

`data-pinned` 감사 체크가 존재하는 이유가 정확히 이것이다.

### B1·B2 를 고칠 때 지킬 것

1차 라운드가 이 호출자를 붙인 이유는 **`kernel-fetch-guard-has-no-caller`** 였다 —
거부 함수 셋에 프로덕션 호출자가 0건인데 `docs/methodology.md §11` 이 문을 닫는다고
현재형으로 단언했다. **그 요구를 되돌리지 마라.** 측정 중 커널을 네트워크에서 받아오는 것은
네트워크 볼륨에서 데이터를 읽는 것과 같은 종류의 오염이고 `AGENTS.md` 가 금지한다.

고칠 것은 **순서와 범위**다: 체크포인트와 코퍼스는 고정된 신원으로 받아야 하고,
커널 문은 모델이 만들어지기 전에 닫혀야 한다. 그 둘이 양립하는 지점을 찾는다.

**되돌린 것이 아님을 테스트로 보인다** — 문이 닫힌 뒤 커널 fetch 가 여전히 거부되는 것과,
문이 닫히기 전에 체크포인트·코퍼스가 고정된 revision 으로 온 것 **둘 다** 단언한다.
한쪽만 보이면 다음 라운드가 반대쪽을 다시 찾는다.

---

## B3 (blocker) — 경계의 동결 레코드를 자기 스키마가 거부한다

`tests/fixtures/run_record.sample.json` 의 `config.run` 이 `trackio_project`/`trackio_space_id`
를 싣고 있는데 그 둘은 결정 3 으로 `RunConfig` 에서 제거됐고 `BenchConfig` 는
`extra="forbid"` 다. `BenchConfig.model_validate(sample["config"])` 가 `ValidationError` 로 죽는다.

즉 **`record-report` 경계가 두 레인을 검증하는 기준 레코드가 어떤 런도 만들 수 없는
레코드다.** 계약의 fixture 는 "계약의 진실은 산문이 아니라 이 파일"이라는 자리에 있다 —
그것이 생산 불가능하면 그 경계는 아무것도 고정하지 않는다.

두 키를 빼고, **fixture 가 자기 스키마를 통과하는지 검사하는 테스트를 둔다.**
그런 테스트가 없어서 이 상태가 생겼다.

---

## B4 (blocker) — `read_fingerprint` 가 동결 샘플 셋 중 둘을 만들지 못한다

`trainbench/kernels.py:433`. `config.attn.impl` 은 스키마 표에서 **항상 평문 문자열**만 낸다 —
dict 를 넘길 경로가 없다. 문자열이면 `_requested_by_backbone` 이 모든 백본을 `landed` 로
돌려주고 `_one([...])` 가 서로 다른 값 둘을 보고 `UnidentifiedKernel` 로 거부한다.

그런데 동결된 지문 `fa2_hub_fallback_qwen3_vl` 과
`fa3_hub_kernel_mask_unregistered_qwen3_vl` 자체가 **문자열 요청 + 백본별로 다른 구현**이다.
즉 계약이 동결한 세 샘플 중 둘을 이 런타임은 만들어낼 수 없다.

멀티모달이 `attn_implementation` 을 서브컨피그별로 해석한다는 것은 리서치가 핀에서 확정했다
(`.plans/research/transformers-varlen-prompt.md §7`). **거기서 다시 읽고** 문자열 요청에서
백본별로 다른 구현이 바인딩되는 것이 정상인지 확정한 뒤 고친다.

---

## B5 (blocker) — 하네스 스텝이 `SentenceTransformer.forward` 를 구동하지 못한다

`trainbench/loader.py:471`. 어댑터가 `step=HARNESS_STEP` 을 선언하는데 핀된 5.6.1 의
`SentenceTransformer.forward(self, input: dict, **kwargs)` 는 **위치 인자 하나**를 요구하고
dict 를 돌려준다. 하네스 루프의 `model(**batch, output_hidden_states=False)` 가 스텝 0 에서
`TypeError` 로 죽는다.

1차 수정이 collate 앞단(프로세서를 올바른 객체로 돌려주기)만 고치고 스텝은 그대로 두었다.

`.plans/research/sentence-transformers.md` 에 `forward` 의 원문과 module layout 이 있다.
**핀된 소스를 읽고 나서** 고친다 — 어댑터별 step 을 선언할지, `HARNESS_STEP` 의 계약을
넓힐지는 `loader-bench` 계약이 무엇을 고정하는지 읽고 정한다. 계약이 표현하지 못하면
`boundaryRequests` 로 올리고 고치지 않는다.

---

## 완료 조건

1. **다섯 각각의 재현 명령을 `findings-round2.md` 에서 그대로 돌려 다른 출력**을 낸다.
   `fixed[].reproNowShows` 에 그 출력을 적는다. 스위트가 초록인 것은 증거가 아니다
2. B1·B2 는 **양방향을 함께 보인다** — 커널 fetch 는 여전히 거부되고, 체크포인트·코퍼스는
   고정된 신원으로 온다
3. 재발을 막는 가드마다 mutation 증거. **사보타주 전에 `co_filename`/`co_firstlineno` 확인**
4. 네 게이트 전부 초록. `pytest tests/contract -q` → **122 passed 유지**
   (줄어들면 계약을 약화한 것이다)
5. 고치지 못한 것은 `notFixed` 에 키와 이유. 숨기지 않는다
6. **확인 안 함** — 이 호스트에 CUDA·deepspeed·TE·DALI·fla·여섯 프레임워크가 없다.
   파드가 답할 것을 `.plans/notes/blockerfix.md` 에 적는다

## 하지 않는 것

- major 23건, minor 24건 — 범위 밖이다. 눈에 띄어도 고치지 않고 `notFixed` 에도 적지 않는다
- `docs/audit-baseline.json`, 루트 `PLAN.md`, `AGENTS.md`, `docs/methodology.md` — 통합자
- `tests/contract/*.py` — 계약 테스트. fixture 는 B3 에 한해 허용
- `envs/**`, `uv.lock`, `pyproject.toml`
