# packing 레인 노트 (wave 2)

머지 단계가 처리할 것, 그리고 이 레인이 소유 밖이라 손대지 않고 넘기는 것.

## 통합자에게 — 반드시 처리

### 1. `docs/audit-baseline.json` — `verdicts-closed` 4 -> 3

`qwen3-vl-query-prompt-may-go-in-twice` 를 닫았으므로 감사가 `shrank` 로 BLOCK 한다.
이 워크트리에서 실측:

```
12/15 passing, 0 new failure(s), 0 newly fixed, 0 grew, 1 shrank, 0 unreadable
BLOCKED: baseline is stale, these shrank: verdicts-closed 4->3
```

브리프가 예고한 정상 상태다. `--update-baseline` 은 통합 단계만 실행한다.
baseline 의 `verdicts-closed` note 첫 문장이 "판정 넷이 열려 있고 셋으로 갈린다" 이고
그 (1) 항목이 이제 닫혔으므로, 갱신할 때 그 문장도 남은 셋 기준으로 다시 쓴다.

### 2. `PLAN.md` 레이아웃 — 신설 파일 1개

```
tests/test_collate.py
```

이 레인이 만든 유일한 신설 파일이다. 커밋 시점의 `plan-files` 는 이것 하나만으로 빨갛다.

## 소유 밖이라 넘기는 것

### 3. `docs/model-spec.md` — "instruction prompt 는 쿼리 쪽만 싣는다" 가 이 체크포인트에선 거짓

측정 2026-08-03, `AutoProcessor.from_pretrained('Qwen/Qwen3-VL-Embedding-2B')`,
transformers 5.14.1. 이 체크포인트의 `chat_template.jinja` 는 system 턴이 없는 행에
`default_system_message` 를 스스로 넣고 그 문자열이 `instruction_prompt` 와 같다.
그래서 **positive 행도 instruction 을 1회 싣는다**:

```
[query 0]    len=24 prompt_occurrences=1
[query 1]    len=24 prompt_occurrences=1
[positive 2] len=25 prompt_occurrences=1
[positive 3] len=23 prompt_occurrences=1
```

이 레인의 변경으로 생긴 것이 아니다 — 고치기 전에도 positive 는 같았고, 바뀐 것은 쿼리뿐이다
(30 -> 24, occurrences 2 -> 1). 문서가 주장하는 쿼리/positive 비대칭이 이 템플릿 아래에서
성립하지 않는다는 사실만 기록한다. 없애려면 빈 system 턴을 명시적으로 넘겨야 하는데,
그것은 "이 체크포인트가 학습된 적 없는 프롬프트"를 만드는 쪽이라 설계 결정이고
`docs/model-spec.md` 소유 레인의 몫이다.

### 4. 이전 Phase 0 의 Qwen 수치와 앞으로의 수치는 직접 비교되지 않는다

Qwen3-VL-Embedding-2B 의 **쿼리** 행이 6토큰 짧아졌다(측정: 쿼리 2행 모두 delta=-6).
tokens/s 의 분모가 바뀌므로, 이전 결과를 인용하는 문서는 이 델타를 함께 적어야 한다.
`qwen3_5_0_8b` 와 `gemma4_e2b` 는 `instruction_prompt: null` 이라 영향이 없다.

## 파드가 답할 질문 — 이 호스트에서 확인 안 함

1. **실제 GPU 에서 fa2 varlen 경로가 실제로 도는가.** 확인 방법:
   `attn=fa2 dataloader.packing=true` 로 forward 를 한 번 돌리고, 같은 행들을 패딩 배치로
   돌린 것과 pooled embedding 을 대조한다. 일치하면 varlen 격리가 패딩 격리와 같은 답을 냈다는
   뜻이고, 불일치하면 리서치 §5 의 경계 유도 규칙 차이(sdpa 는 `diff != 1`, fa2 는
   `position_ids == 0`)를 먼저 의심한다. 이 호스트에는 GPU 도 flash-attn 도 없어
   `is_fa_with_varlen_kwargs` 분기가 한 번도 실행된 적이 없다.
2. **`seq_idx` 없이 돌린 pack 에서 Qwen3.5 의 causal conv 가 경계를 넘는가.**
   아래 `boundaryRequests` 항목과 같은 질문의 실측판이다. `causal_conv1d_fn` 이 파드에
   설치돼 있어야 관측된다 — 없으면 폴백 경로라 애초에 conv 가 다르게 돈다.
