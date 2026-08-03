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
