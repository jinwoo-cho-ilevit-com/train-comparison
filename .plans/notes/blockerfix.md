# blockerfix — 2차 라운드 blocker 5건

base `1c2581c8ff3031d0e44f2c9e2b7ffbef3ac10f1d`. 이 워크트리에서 직접 실행한 것만 적는다.

## 고친 것

| 키 | 자리 | 무엇을 바꿨나 |
|---|---|---|
| `timing-run-goes-offline-before-the-checkpoint-and-corpus-are-fetched` | `scripts/bench.py::build_run` | 고정 입력을 먼저 받고 그 다음에 문을 닫는다 |
| `offline-doors-defeat-the-data-pin` | 같은 자리 | 코퍼스를 문 닫기 전에 **읽는다**(워밍이 아니다) |
| `run-record-sample-config-rejected-by-its-own-schema` | `tests/fixtures/run_record.sample.json` | trackio 두 키 제거 + 자기 스키마 통과 검사 신설 |
| `read-fingerprint-cannot-produce-two-of-three-frozen-samples` | `trainbench/kernels.py::read_fingerprint` | 요청이 **바인딩된** 백본만 resolved 로 센다 |
| `st-harness-step-cannot-drive-sentencetransformer-forward` | `scripts/bench.py` | 스텝 0 `TypeError` 를 타이머 전 거부로 바꿨다 (아래 경계 요청 참조) |

### B1·B2 순서

`fetch_the_pinned_inputs` → `close_kernel_fetch_doors` → `load_framework` 순이고 그
순서 자체가 게이트다(`tests/test_smoke_cpu.py`, 양방향 한 런에서 읽는다).

- `from_pretrained` 는 내려받기와 조립을 한 호출로 하므로 그 둘 사이에 문을 닫을 수 없다.
  그래서 `prefetch_checkpoint` 가 스냅샷을 먼저 당기고, 조립은 디스크에서 읽는다.
- 코퍼스는 워밍이 아니라 읽기다. 오프라인 `datasets` 는 고정 revision 요청에 캐시에 있는
  빌드를 내주고 **로그 한 줄만** 남긴다. 답이 먼저 나오면 치환될 자리가 없다.

### 이 순서를 되돌리려는 충동은 파드 로그에서 온다 — `kernel fetch door closed`

파드 로그는 setting 마다 이 셋을 찍는다:

```
kernel fetch door closed: $HF_HUB_OFFLINE=None, want '1'
kernel fetch door closed: $USE_HUB_KERNELS=None, want 'NO'
kernel fetch door closed: huggingface_hub.constants.HF_HUB_OFFLINE=False, want True
  — cached at import, so the environment variable was set too late to reach it
```

**이것은 실패 보고가 아니라 닫은 문의 목록이다. 과거형이다.**
`forbid_runtime_kernel_fetch` 는 열려 있던 문을 **반환하고 전부 닫으며**,
`scripts/bench.py:603` 이 반환값을 그대로 찍는다. 실측(이 세션, 프로덕션 적재 경로 +
transformers import 후): 호출 뒤 `open_fetch_doors()` 는 `[]`, env 는 `1`, const 는
`True`, `assert_no_runtime_kernel_fetch()` 통과. 세 번째 줄은 캐시된 상수 **도** 닫혔다는
기록이지 안 닫혔다는 뜻이 아니다.

세 줄을 "문이 안 닫힌다"로 읽으면 다음에 손이 가는 곳은 `docker/entrypoint.sh` 최상단의
`export HF_HUB_OFFLINE=1` 이다. **그것이 위 B1·B2 를 정확히 되돌린다.** 실측(빈 `HF_HOME`,
프로세스 시작부터 `HF_HUB_OFFLINE=1`):

```
prefetch_checkpoint  -> LocalEntryNotFoundError ... outgoing traffic has been disabled
load_pairs           -> ConnectionError Couldn't reach 'jinwoo-cho/mmeb-subset' on the Hub (OfflineModeIsEnabled)
```

둘 다 `refusal_types()` 가 아니라서 exit 1 + 결과 파일 없음이 되고, 모든 timing/quality
설정이 `no_result` 로 올라간다 — B1 이 없애려던 그 증상이다.

읽는 순서를 바꿔 주는 사실 하나: **문이 열린 채로는 측정이 진행되지 않는다.**
`build_run` 이 모델을 지은 **뒤** 다시 읽고(`scripts/bench.py:792-793`,
`ENFORCED_PURPOSES` 한정) 하나라도 열려 있으면 거부한다. 그러므로 "커널 fetch 가 조용히
타이밍 숫자에 섞였다" 는 이 배선에서 성립하지 않는다 — 섞였다면 숫자가 아니라 refusal 이
나온다.

`tests/test_smoke_cpu.py` 의 순서 체크 다섯은 이것을 못 잡는다. 두 헬퍼가 시작 상태를
고정하려고 `RUNTIME_FETCH_ENV` 를 `monkeypatch.delenv` 하므로(`:1948`, `:2220`),
**환경이 그 변수를 들고 도착하는 경우가 원리적으로 안 보인다** — ambient `HF_HUB_OFFLINE=1`
로 다섯을 돌려도 그대로 통과하는 것을 실측했다. 그 구멍은
`tests/test_pods.py::test_the_pod_hands_bench_an_environment_with_the_fetch_doors_still_open`
이 막는다: 엔트리포인트가 `bench.py` 에 넘긴 환경을 setting 마다 읽어 두 이름이 모두
비어 있는지 본다.

**파드 실측 하나(이 세션에서 확인 안 함 — 파드 로그 보유자 판독):** A100 파드
`aib8xamhmrb312` 는 `train()` 안에서 죽었고(`scripts/bench.py:288`, collate 의
`RuntimeError` 재발생) `train()` 은 `:793` 보다 뒤이므로, 그 setting 에서
`assert_no_runtime_kernel_fetch()` 는 **실 하드웨어에서 실행되어 통과했다.** 오프라인
게이트가 랩톱 밖에서도 성립한다는 첫 근거다. 위 "파드가 답해야 하는 것" 2번은 이것으로
닫히지 않는다 — 그 항목은 프레임워크 넷의 자체 로더가 오프라인에서 캐시를 집는지를 묻고,
이 파드는 `native` 한 셀이다.

## 파드가 답해야 하는 것 — 이 호스트에서 확인 안 함

1. **`prefetch_checkpoint` 의 실제 다운로드량.** `snapshot_download` 를 필터 없이 부른다.
   세 체크포인트 repo 에 safetensors 말고 무엇이 더 있는지, 그래서 파드가 여분으로
   얼마를 받는지 재지 않았다. 크면 `allow_patterns` 가 아니라 **repo 파일 목록을 보고**
   정해야 한다 — 형식을 추측하면 `.bin` 만 있는 repo 에서 적재가 깨진다.
2. **오프라인 `from_pretrained` 가 프리페치된 스냅샷을 집는지.** 여섯 프레임워크 중
   `native`/`sentence_transformers` 만 `from_pretrained` 를 우리 경로로 부른다. 나머지
   넷은 자기 로더가 부르고, 그 경로가 `HF_HUB_OFFLINE=1` 아래서 캐시를 읽는지는
   실 이미지에서만 답이 나온다.
3. **`configs/model/*.yaml` 의 `revision: null`.** 세 모델 다 고정돼 있지 않다. 지금은
   `prefetch_checkpoint` 가 해석된 sha 를 stdout 에 찍을 뿐 레코드에 넣지 않는다.
   범위 밖이라 손대지 않았고, 축·레코드 스키마를 쥔 레인이 정할 일이다.
4. **fa2 샘플의 `mask_registered`.** 동결 샘플은 `true` 인데 이 호스트는 `false` 를 낸다 —
   `kernels`/`flash-attn` 이 없어 커널이 바인딩되지 않기 때문이고, 계약이
   `test_mask_registration_is_a_read_back_...` 에서 같은 말을 한다. 나머지 필드는
   전부 일치하고 `fa3` 샘플은 바이트 단위로 재현된다.

## 통합자에게

- 신설 파일 없음. `plan-files` 는 이 레인 때문에 빨개지지 않는다.
- `docs/audit-baseline.json`, 루트 `PLAN.md`, `AGENTS.md`, `docs/methodology.md` 를
  건드리지 않았다.
- `docs/evidence/env-report-cpu-qwen3_5_0_8b-native.json` 에 `trackio_project`/
  `trackio_space_id` 가 아직 남아 있다. 그 파일은 이 레인 소유가 아니고 과거 실행
  아티팩트라 그대로 뒀다. 다시 만들 때 사라진다.
