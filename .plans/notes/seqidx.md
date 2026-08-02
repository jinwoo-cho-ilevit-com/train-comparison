# seqidx — 머지 단계로 넘기는 것

base `d0d0aa0395ce209acfc818a408198f1135e535c3`, 브랜치 `wave3b-seqidx`.
여기 적힌 것은 전부 이 워크트리에서 이번 세션에 직접 실행하거나 핀된 휠을 열어 확인했다.

## 1. 신설 파일 — 없음

`plan-files` 는 초록이다(실행함). `PLAN.md` 레이아웃에 등재할 것이 없다.
수정한 파일은 소유 넷뿐이다: `trainbench/collate.py`, `tests/test_collate.py`,
`tests/contract/test_collate_metrics.py`, `tests/fixtures/microbatch.sample.json`.

## 2. `docs/methodology.md` §10.1 이 과거형이 됐다 — integrate 레인 몫

그 절은 `linear_attention` 레이어가 격리 없이 돈다고 적고 있고, 근거로
`trainbench/collate.py` 가 경계를 배치에서 빼내기만 한다는 것을 든다. 두 갈래 다
닫혔다:

- chunked 커널의 `cu_seq_lens_q` — wave 2 packing 레인
- causal conv 의 `seq_idx` — 이 레인

같은 절의 표("qwen3_5: full만 자동, linear은 아니다")도 함께 다시 써야 한다. 격리가
이제 자동이 아니라 **collate 가 두 kwarg 를 실어 보내서** 생긴다는 것이 바뀐 문장이다.

`PackedBatches` 를 가리키는 줄 번호도 옮겨갔다(그 절은 `collate.py:429` 를 인용한다).

## 3. 파드가 답할 것 — 이 호스트에서 확인 안 함

`causal-conv1d` 가 이 랩톱에 없다. 그래서 `Qwen3_5GatedDeltaNet` 의
`self.causal_conv1d_fn` 이 None 이고 폴백 분기
(`modeling_qwen3_5.py:500-501`, `F.silu(self.conv1d(...))`)에는 `seq_idx` 를 받는
자리가 아예 없다. 실측 2026-08-03, CPU, 이 체크아웃: `Qwen3_5TextModel` 에
`seq_idx` 를 주고 안 주고가 hidden state 를 **바이트 단위로 같게** 만든다.

즉 이 호스트는 "커널이 경계를 지키는가"를 원리적으로 볼 수 없다. 파드가 찍어야 할 것:

| 질문 | 무엇을 찍는가 |
|---|---|
| 커널이 설치됐는가 | `Qwen3_5GatedDeltaNet.causal_conv1d_fn is not None` |
| 경계를 지키는가 | `dataloader.packing=true` 로 forward 한 번, 같은 행을 패딩 배치로 돌린 것과 pooled embedding 대조. 일치하면 conv 격리가 패딩 격리와 같은 답을 냈다는 뜻 |

`.plans/notes/packing.md` 의 파드 질문 2번과 `.plans/notes/wire.md §8` 의
`kernel=fla × packing` 행이 같은 질문이고, 이제 배선은 있고 관측만 남았다.

## 4. 이 레인이 찾은 것 — "모델이 받았다"는 증거가 아니다

실측 2026-08-03, CPU: `Qwen3VLTextModel` 은 `not_a_real_kwarg=...` 를 넘겨도
예외 없이 통과하고 hidden state 도 그대로다. `**kwargs` 를 받는 forward 는 자기가
읽지 않는 이름을 조용히 삼킨다.

그래서 "세 arch 가 `seq_idx` 를 받는다"는 forward 테스트 하나만으로는 공허하다.
`tests/test_collate.py` 에 통제군을 같이 넣었다 —
`test_an_undeclared_kwarg_is_swallowed_just_as_quietly` 가 그 삼킴을 고정하고,
`test_only_qwen3_5_reads_seq_idx_among_the_arches_this_study_measures` 가 핀된 휠에서
**누가 그 이름을 읽는지**를 본다(qwen3_5 만 읽고 qwen3_vl/gemma4 는 읽지 않는다).
후자가 arch 무관 방출이 안전한 이유이고, 버전 업이 그것을 바꾸면 거기서 터진다.

## 5. `.plans/deps/seqidx.txt` — 없음

새 패키지를 요구하지 않는다.
