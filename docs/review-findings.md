# 3레인 병렬 리뷰 결과와 수정 계획 (2026-08-01)

컨벤션 09에 따라 module / architecture / critic 3레인을 병렬 실행하고, 작성자와
분리된 상태에서 fan-in 했다. 아래 "실행으로 확정"은 전부 코드를 직접 돌려 확인한
것이다.

## 실행으로 확정한 치명 결함

| # | 결함 | 검증 방법 |
|---|---|---|
| D1 | **고정 서브셋 손상** | 푸시된 revision 실측 |
| D2 | **`is_finished`가 pull 중인 pod을 완료 판정** | `is_finished({'runtime':None,...}) -> True` |
| D3 | **`last_token_pool`이 left padding에서 오답** | mask `[0,0,1,1]` -> index 1 반환(정답 3) |
| D4 | **12축 중 8축을 코드가 읽지 않음 + 패키지 부재** | grep 0건, `envs/*/uv.lock` 전수 |
| D5 | `INFISICAL_TOKEN` 미주입 | `pod_env()`에 부재 |

### D1 상세 — 가장 심각

`jinwoo-cho/mmeb-subset` @ `b750b9c3` 실측:

```
rows                              : 2048
qry_image is None                 : 644  (31.4%)
distinct pos_text                 : 1003 (49%)
most common pos_text              : 466 rows (22.8%)
  -> '<|image_1|>\nRepresent the given image.\n'
rows sharing a duplicated pos_text: 1119 (54.6%)
```

원인: `SUBSET_COLUMNS`가 `pos_image`를 버렸다. positive가 이미지인 config에서
positive 내용이 사라지고 placeholder 문자열만 남는다. InfoNCE는 배치 내 나머지를
negative로 보므로, 동일 positive 466개는 같은 문서를 서로 구분하라는 gradient가 되어
제거 불가능한 loss 하한을 만든다. 쿼리 이미지 31.4% 결측으로 속도도 낙관 편향된다.

**탐지 불가였던 이유**: 검증 지표가 행 수와 config 커버리지뿐이라
`taken 2048/2048, 20/20 config`으로 완벽 통과했다. 커밋 `9ac6531`의 Result가 이
지표를 인용해 "완료"라고 적었다.

### D4 상세

`attn` / `kernel` / `precision` / `compile` / `optim` / `freeze` / `dataloader` /
`parallel` — 스키마·테스트 외 참조 0건. 필요 패키지도 어느 env에도 없다:
`flash-attn`, `causal-conv1d`, `transformer-engine`, `deepspeed`, `nvidia-dali`,
Muon 구현체. `liger-kernel`/`bitsandbytes`는 일부 env에만.

이 상태로 Phase 2를 돌리면 이름만 다른 동일 실험이 나오고, 그 표는 "임베딩 학습에서
최적화가 대부분 무효"를 지지한다. 그것이 이 프로젝트의 가설이라 더 위험하다.

## 확정된 설계 결정 (2026-08-01)

### 1. SFT 대조군을 두지 않는다 -> 주장 범위를 좁힌다

동일 조건 SFT arm 없이는 "SFT와 우선순위가 다르다"를 반증 가능한 형태로 검증할 수
없다. 공개 제3자 수치는 하드웨어·데이터·시퀀스가 달라 대조군이 아니다.

- `PLAN.md`의 핵심 가설에서 SFT 비교 주장을 **철회**한다
- 남는 주장: **임베딩 학습 내부의 축별 효과 + 모델별 병목 차이**
- Phase 4의 "공개 주장 대비 실측"은 비교가 아니라 참고 수치로 격하
- Liger는 FLCE 외 RMSNorm/SwiGLU/RoPE 커널도 제공하므로 "무력화"가 아니라
  "FLCE 경로만 정의상 비활성"으로 정정

### 2. 8개 축을 전부 구현한 뒤 Phase 2를 시작한다

"돌아가는데 아무 일도 하지 않는" 축을 남기지 않는다. 축별로 검증 가능한 GPU를
명시한다 (FA4/NVFP4는 B200 전용, FA3는 Hopper 이상, A100에서는 검증 불가).

## 수정 순서

선행 의존 관계 순. 앞 단계가 끝나기 전에 뒤로 가면 결과가 폐기된다.

1. **데이터 재생성** — config별 기대 컬럼 선언 + 없으면 예외, `pos_image` 보존,
   manifest에 `rows_without_query_image` / `distinct_pos_text` / 중복률 추가.
   손상된 revision 고정 해제.
2. **축 적용 검증 계층** — 요청값과 실제 적용값을 쌍으로 기록하고 `purpose=timing`에서
   불일치 시 실패. 패키지 부재 축은 config 검증에서 거부.
3. **8개 축 구현 + 패키지 설치** — env/이미지 갱신 포함.
4. **오케스트레이터 견고화** — D2/D5, ledger 증분 기록, pod별 개별 deadline,
   entrypoint 자살 장치, 결과 없는 경우 fallback 레코드.
5. **`last_token_pool` 수정 + left padding 테스트** — gemma-4가 `padding_side: left`
   이면서 유일하게 probe 미실행 모델이라 사각지대가 정확히 겹친다.
6. **sweep 러너 + baseline 게이트** — pod 안에서 모델 1회 적재 후 축 sweep.
   3% 임계값은 동일 pod 5회 반복 편차를 실측해 교정한 뒤 확정.
7. **MFU / visual token 방법론** — 세 아키텍처(GDN linear attention, PLE lookup,
   sliding window)에서 표준 FLOP 공식이 전부 깨진다. tokens/s를 1차 지표로,
   MFU는 모델별 공식을 유닛 테스트로 검증한 뒤에만 제시.
8. **품질 가드레일** — Recall@k 이전에 축별 수치 등가성 검사(같은 seed로 N step 후
   baseline 대비 loss/grad norm tolerance). GradCache는 전용 등가성 테스트 필수.

## 문서 정정 대상

- `README.md`의 `uv sync --group dev` -> `--extra compose` (현재 문서대로 하면
  pytest가 hydra 부재로 실패)
- `docs/methodology.md` 신설 — `config_schema.py`와 `AGENTS.md`가 참조하는데 부재.
  deterministic on/off 비용을 실측해 기록(컨벤션 07이 요구하는 근거)
- "torch.profiler가 20~44% 부풀린다"의 출처 명기 또는 실측 (현재 4곳에 사실로 반복,
  출처 없음 — 컨벤션 16 위반)
- `PLAN.md` 저장소 구조에 적힌 미존재 파일들, 데이터 출처 표기
- support-matrix의 "env 5/5 성공"에 "`uv lock` 성공이며 설치·빌드·실행은 아님" 한정 추가
