# kernels 레인 노트 — 머지 단계로 넘기는 것

## 신설 파일 (PLAN.md 레이아웃 등재)

- `trainbench/kernels.py` — 바인딩된 어텐션 커널의 신원, packing 거부, 런타임 fetch 차단
- `tests/test_kernels.py` — 이 레인의 유일한 게이트

`docs/methodology.md`는 기존 파일이고 §9/§10.1/§10.2를 고쳤으며 §11을 새로 넣었다.

## 루트 문서에 올라가야 할 사실 (integrate 레인)

`AGENTS.md`의 "Record the resolved torch/framework versions per run"은 커널에 대해
부족하다. `flash-attn`이 없는 환경에서 `flash_attention_2` 요청은 Hub 저장소 이름으로
바뀌고(`transformers/modeling_utils.py:2003`) 그 커널은 런 시작 중에 내려받아진다.
같은 버전이 다른 커널을 바인딩할 수 있으므로 런 레코드에 남아야 하는 것은
**repo + revision**이고, revision을 못 읽으면 거부다. 근거와 인용은
`docs/methodology.md §11`에 있다.

`docs/CONTRACTS.md`에 추가할 한 줄: `kernel-provenance` payload의 유일한 런타임 구현은
`trainbench/kernels.py::validate_fingerprint`이고, 계약 파일의 validator와 같은 판정을
낸다는 것은 `tests/test_kernels.py -k agrees_with_contract`가 증명한다. 두 구현이
존재하는 이유는 계약이 import-free여야 하기 때문이며, 그 사실이 문서에 없으면 다음
레인이 한쪽을 지운다.

## 다른 레인 파일에 필요한 변경 (요청만 한다)

- **adapters 레인**: 어댑터의 build fingerprint에서 `BUILD_FINGERPRINT_KEY`
  (`"attention"`) 자리는 `kernels.read_fingerprint(model, axis=..., value=...,
  requested=...)`의 반환값을 그대로 쓴다. 그 함수는 반환 전에 스스로 검증하므로
  어댑터 쪽에 검증을 한 번 더 두지 않는다. Hub 커널을 쓰는 칸은
  `revision_resolver`를 넘겨야 한다 — 이 호스트에 `kernels`가 없어 기본 해석기를
  쓰지 않았고, 인라인 `@revision` 접미사와 주입된 해석기 두 경로만 테스트했다.
- **packing 레인**: `trainbench/collate.py:429`가 `cu_seqlens`/`seq_lengths`를 모델에
  넘기기 전에 배치에서 빼낸다. Qwen3.5의 linear_attention 레이어는 `position_ids`가
  아니라 `cu_seq_lens_q`/`seq_idx`를 읽으므로(`models/qwen3_5/modeling_qwen3_5.py:549`,
  `:498`) 지금 Qwen3.5 + packing 런은 그 레이어에서 격리 없이 돈다. 상세는
  `docs/methodology.md §10.1`.
- **axes 레인**: `docs/methodology.md §9`의 표는 `kernel=liger` x `qwen3_vl`이
  "기록된 엔트리포인트 없음"으로 거부된다고 적는다. `.plans/research/axis-libraries.md
  §1.2`는 liger 0.8.1에 `apply_liger_kernel_to_qwen3_vl`이 있다고 원문으로 확정했다.
  즉 그 거부는 라이브러리의 한계가 아니라 `LIGER_ENTRYPOINTS`의 누락이다. 표를 고치면
  §9의 그 행도 함께 고쳐야 한다.

## `docs/audit-baseline.json`

이 레인은 건드리지 않았고 필요한 변경도 없다. `axis-values`의 `kernel 1/4`는 이 레인의
범위 밖이다 — 이 레인은 커널을 적용하지 않고 신원만 읽는다.

## 확인 안 함 — 파드 질문으로 등록

1. fa2 varlen 경로가 실제로 도는가. 이 체크아웃에 `flash_attn`도 `kernels`도 없다.
2. `fa3`/`fa4`가 `envs/native`에서 적재되는가.
3. transformers 5.5.0(unsloth)과 5.12.1(ms_swift)에 `find_packed_sequence_indices`가
   있는가. `docs/methodology.md §10`의 인용은 전부 5.14.1 원문이다.
4. `flash_attn` 없는 파드에서 fa2 요청의 최종 `config._attn_implementation` 문자열이
   `"flash_attention_2"`인가 `"kernels-community/flash-attn2"`인가.
5. 그 문자열이 repo id일 때 `create_causal_mask`가 실제로 `None`을 돌려주는가.
   소스 독해로는 그렇게 읽히지만(`masking_utils.py:826`) 실행으로 확인하지 못했다.
6. `kernels`가 설치된 환경에서 기본 revision 해석기가 무엇을 돌려주는가.
