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
| D6 | **`prepare_data`가 전 행을 디코딩된 픽셀로 누적** | 실행 중 SIGKILL, 17배 실측 |
| D7 | **push한 서브셋 revision을 읽을 수 없음** | `load_dataset(...)` → CastError |

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

### D6 상세 — D1 수정 도중 실행으로 발견 (2026-08-01)

3레인 리뷰가 아니라 `data=quality` 재생성을 실제로 돌려서 나왔다. D1을 고친 코드가
2048행(speed)에서는 통과하고 65536행(quality)에서는 24분 만에 죽었다.

- `main()`은 20개 config의 모든 행을 `rows_by_config`에 끝까지 누적한 뒤 마지막에
  한 번에 push한다.
- `datasets` 5.0.1의 `Image.decode_example`은 행을 넘길 때마다 `PIL.Image.load()`를
  호출한다(`features/image.py`). 누적되는 것은 압축 바이트가 아니라 픽셀이다.
- 실측(MSCOCO_i2t 24행): 디코딩 **270.9 KiB/행** vs 인코딩 **15.9 KiB/행**, 17.0배.
- 결과: 40377/65536행 지점에서 SIGKILL. RAM 48GB + swap 20GB 소진, traceback 없음.

**speed가 통과한 것은 결함이 없어서가 아니라 32배 작아서다.** 행 수 기반 스모크는
누적 자원 결함을 원리적으로 잡지 못한다 — `data.limit`을 줄이면 결함도 같이 줄어든다.

부수 결함: 중간 저장이 없어 24분치 스트리밍이 전부 소실됐다. 컨벤션 04가 요구하는
save/resume이 이 단계에만 없었다.

수정: `stream_rows`가 이미지 컬럼을 `Image(decode=False)`로 캐스팅해 인코딩 바이트로
들고 있고, config별 draw를 parquet shard로 캐시해 재개한다. push되는 parquet이
원본 바이트를 그대로 담게 되므로 speed/quality 양쪽 revision을 함께 재생성·재핀했다.

### D7 상세 — D6 수정을 3레인 리뷰에 걸어서 발견 (2026-08-01)

D6을 고치고 두 서브셋을 재생성·재핀한 뒤 module / architecture / critic 3레인 리뷰를
돌렸다. 리뷰가 찾은 것 중 가장 큰 것은 **그 재핀 자체가 읽을 수 없는 revision을
가리키고 있었다**는 것이다.

```
load_dataset('jinwoo-cho/mmeb-subset', revision='f436302932c0...', split='train')
CastError: ... because column names don't match
카드 선언: {qry, qry_image, pos_text, mmeb_config}      <- pos_image 없음
parquet  : {qry, qry_image, pos_text, mmeb_config, pos_image}
```

원인은 `Dataset.push_to_hub`가 이미 카드가 있는 저장소에 덮어쓸 때 기존
`dataset_info`를 유지한다는 것이다(`datasets/arrow_dataset.py`,
`info_to_dump = repo_info`). features 불일치 가드는 다른 split이 있을 때만 평가되므로
`train` 하나뿐인 덮어쓰기에서는 아예 돌지 않는다. 그 "기존 features"가 바로 **D1
시절의 4컬럼 스키마**였다 — parquet은 두 번 고쳤는데 저장소 메타데이터에서는
살아남았다. `mmeb-subset-quality`가 멀쩡했던 이유는 새 저장소에 처음 push했기
때문이다.

**이 결함이 통과한 경로가 본질이다.** push가 성공하고 revision이 나온 것을 완료로
취급했고, push된 산출물을 다시 읽지 않았다. 검사가 자기 부재를 통과로 보고하는
이 저장소의 반복 패턴이며, `data-pinned`가 그 사이 내내 PASS였다는 사실이 그 점을
정확히 보여준다 — 그 체크는 `COMMIT_SHA.fullmatch(revision)`, 즉 문자열 형식 검사라
읽을 수 없는 아티팩트에 만점을 줬다.

수정: push를 `DatasetDict`로 내보내 features를 다시 쓰게 하고(`remove_other_splits`
경로), **push 직후 그 revision을 되읽어 게이트를 재계산하는 `verify_pushed`를 추가**해
불일치 시 non-zero로 죽고 핀하지 말라고 출력한다.

### D6 수정이 만든 회귀 (같은 리뷰에서 발견)

`_image_size`가 헤더만 파싱하게 되면서 `MAX_ROWS_WITH_UNREADABLE_IMAGE = 0`의 의미가
**"픽셀이 디코딩된다"에서 "헤더가 파싱된다"로 바뀌었다.** D6 이전에는 `datasets`가 매 행
`PIL.Image.load()`를 호출해 깨진 payload가 게이트 상류에서 터졌는데, 인코딩 바이트로
들고 있게 되면서 그 보장이 대체 없이 사라졌다. IHDR이 멀쩡하고 IDAT이 깨진 이미지가
"판독 가능"으로 계산된다.

수정: `_image_size`가 `load()`까지 호출한다. 한 장씩 디코딩하고 보유하지 않으므로
누적은 돌아오지 않는다(실측: speed draw 2320장 2.2초). 이 수정을 테스트하다
`SyntaxError`(PIL이 깨진 청크에 던지는 예외)가 예외 목록을 빠져나가는 것을 추가로
발견했다.

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

### 레인 E 인계 — D6/D7 수정으로 낡은 서술 (2026-08-01)

레인 A가 데이터 쪽을 고치면서 `PLAN.md`(레인 E 소유)의 서술이 사실과 어긋났다.
소유 경계상 A가 직접 고치지 않고 여기 남긴다.

- **`jinwoo-cho/mmeb-subset-quality`가 `PLAN.md` 어디에도 없다.** 서브셋은 두 개이고
  각각 별도 저장소인데 문서는 하나만 안다.
- "`configs/data/quality.yaml`은 아직 revision 미고정" — 이제 거짓.
- "미러의 커밋을 고정한다"는 서술은 `DataConfig.source_revision`이 생기기 전까지
  코드가 하지 않는 일이었다. 지금은 참이 됐지만, 문서가 코드보다 앞서 있었다는
  사실 자체가 `plan-files`가 파일 존재만 검사하고 서술 내용은 검사하지 않기 때문에
  드러나지 않았다.
- `docs/methodology.md`에 이미지 코덱 항목이 없다. 이전 서브셋은 업스트림 JPEG를
  디코딩해 다시 JPEG로 인코딩(손실)한 것이었고, 지금은 원본 바이트를 그대로 담는다.
  DALI 축의 핵심이 하드웨어 JPEG 디코드이므로 이 차이는 dataloader 축 측정에 실제로
  영향을 준다.

---

## D8 — 체크가 참을 보고하는데 그 뜻이 안 보인다 (2026-08-02, Wave 3 게이트 리뷰)

이 저장소에서 여섯 번째로 같은 모양이다. 사실은 정확히 출력되고 있었고, 그 사실이
무엇을 뜻하는지가 어디에도 없었다.

- **`axis-wired`의 baseline note가 blocker를 가렸다.** note는 "Wave 2 (D: axes) -
  one apply site and one capture probe per axis"였다. `precision.name`과
  `train.offload`에 probe가 없다는 사실은 맞게 보고됐지만, **그 상태의 결과는
  `assert_matches`가 모든 `purpose=timing` 런을 거부해 측정이 하나도 불가능하다는
  것**이었다. note는 담당 레인만 적고 그 결과를 적지 않았고, 그 위에 Wave 3이 얹혔다.
  (해소: D 레인이 두 probe를 붙여 `axis-wired`는 이제 17/17 통과다. 남은 수리는
  note 규칙 쪽이며 `docs/CONTRACTS.md` §1에 넣었다.)
- **`axis-values`의 note는 담당 레인 자체가 틀렸다.** "F: build images so
  kernel/parallel/dataloader/precision values can be applied"로 적혀 있으나
  `axes.py`의 거부는 전부 무조건 `not implemented`이고 import 실패가 아니다.
  이미지를 빌드해도 이 수는 안 움직인다 — 담당은 D다. 같은 이유로 이 체크는 환경
  독립이며(flash-attn 없는 로컬에서도 attn 값이 applicable로 잡힌다) 게이트 수치로는
  재현 가능하지만, **"축 기계가 값을 받는다"이지 "그 값이 동작한다"가 아니다.**

## D9 — 감사 두 체크가 각각 무력화 가능했다 (2026-08-02)

- `config-consumed`의 `_strip_prose()`는 세 가지로 뚫렸다: 이름에 대입한 문자열,
  `if False:` 블록, 앵커 없는 속성 체인. 실제 방어는 Wave 2.5에서 넣은 count 추적
  이었고(줄어들면 차단), **그 말은 자기 baseline 줄을 편집하는 레인이 방어를 없앨 수
  있었다는 뜻이다.** 읽기 판정을 AST로 옮겨 체크 자체를 고쳤다.
- `plan-files`는 한 방향만 봤다. **블록은 언급을 줄이면 참으로 유지된다** — Wave 3이
  만든 `scripts/bench.py`, `trainbench/metrics/`, 테스트 6개가 전부 블록에 없는 채로
  통과했다. 반대 방향을 넣었고, 그 결과 파일을 추가한 레인이 `PLAN.md` 구조 블록에도
  한 줄을 넣어야 한다(`docs/CONTRACTS.md` §5의 새 레인 의무).

## D10 — 기록되지 않은 레인 경계 침범 (2026-08-02)

35a9a62(Wave 3 G)이 `scripts/prepare_data.py`(레인 A 소유)를 수정했다. 같은 커밋의
C->G 이관은 꼼꼼히 기록됐는데 이건 기록이 없었다. 변경 자체는 옳고(중복 정의 제거),
`percentile`이 타이밍 보고 모듈에 놓인 것에 대한 판단과 재검토 조건은
`docs/CONTRACTS.md` §5에 남겼다.

## D11 — `axis-values`가 dataloader 축에 대해 vacuous였다 (2026-08-02, 레인 D 제보)

D8/D9와 같은 계열의 일곱 번째다. 이번에는 체크가 사실을 **양방향으로** 틀리게
보고하고 있었다.

`axis-values`가 `axes.assemble(...)`을 dataset 없이 불렀고 `axes._dataloader`는
`if dataset is None`에서 packing/pretokenize에 닿기 전에 반환한다. 결과:

- **거짓 양성** — `PackedCollate.__call__`을 `raise NotImplementedError`로 갈아도
  체크 출력이 바이트 단위로 같았다. 축을 통째로 무력화해도 감사가 통과시킨다.
- **거짓 음성** — `loss/cached_mnrl`(GradCache)은 구현이 있는데 inert로 보고됐다.
  거부 사유가 "이 런의 dataset이 None"이었다. 감사가 자기 fixture를 안 준 탓에
  멀쩡한 축이 미구현으로 보였고, `axis-values`의 inert 목록은 그만큼 과장돼 있었다.

**dataset만 넘기는 것으로는 절반만 닫힌다.** collate는 배치를 뽑기 전에 호출되지
않고 packing은 전부 collate 안에 있다. 처음에 `PackedCollate` 위쪽에 `__call__`을
끼워 넣어 재현했다고 판단했는데, 실제 `__call__`이 클래스 본문 **뒤쪽**에 있어
나중 정의가 이겼고 사보타주가 애초에 동작하지 않았다 — 즉 "재현했다"는 첫 판단
자체가 틀렸다. 실제 `__call__` 본문을 갈아야 재현된다. 수리 내용과 한계는
`docs/CONTRACTS.md` §5에 있다.

### 레인 E 인계 — `docs/methodology.md`에 dataloader 축 절이 없다 (판정서 F9)

레인 D가 넘긴 것을 그대로 옮긴다. `docs/methodology.md`는 E 소유라 손대지 않았다.
결과 해석에 필요한 조건들이다.

- packing의 속도 이득은 varlen 커널의 것이지 collate의 것이 아니다. GPU가 없어
  **측정 안 함**이고, varlen 커널 없이 packing을 켜면 한 개의 긴 시퀀스에 대해
  어텐션이 quadratic이 된다.
- packed 배치의 **어텐션 격리(cross-sequence attention 차단)는 미검증**이다.
  `position_ids`는 시퀀스마다 0으로 재시작하지만, 경계 넘어 attend를 막는 것은
  커널 쪽 `cu_seqlens` 사용에 달려 있고 이 체크아웃에서 확인할 수 없다.
- `tests/test_axes.py::test_pooling_a_packed_batch_matches_pooling_the_same_rows_padded`가
  증명하는 것은 합성 `arange` 텐서 위의 **인덱스 산술 등가**뿐이다. 모델이 없으므로
  "풀링 등가성 검증됨"으로 읽으면 안 된다.
- packing x 이미지(pixel_values)는 미설계다. 현재 `PackedCollate`는 텍스트 id만 잇는다.
- `pretokenize`의 실이득도 **측정 안 함**이다. 고정된 것은 "토크나이즈가 타임드 스텝
  밖으로 나갔다"는 사실뿐이고(encode 호출 수를 창 양쪽에서 셈), 몇 %인지는 GPU 런의 몫.

`axis-values`가 이제 packing/pretokenize 경로를 실제로 통과시키지만, 그것은 축이
**적용된다**는 뜻이지 위 항목 중 어느 것도 측정했다는 뜻이 아니다.

## D12 — `doc-commands`의 초록불이 vacuous였다 (2026-08-02, optim(muon) 레인 제보)

여덟 번째. 이번에는 체크가 **자기 문구로 거짓말**을 하고 있었다.

`PASS doc-commands 5 documented command(s) install what the tests need and run as
written` — 실제로 검사한 것은 `uv sync` 줄에 `--extra compose`가 붙었는지 하나뿐이고,
근거 주석은 "but tests import hydra"였다. hydra가 테스트의 유일한 의존이 아니게 된 지
오래인데 문구는 "테스트가 필요로 하는 것을 설치한다"고 주장했다. **한 패키지를 지목한
규칙으로 전체에 대한 질문에 답할 수 없다.**

실측(같은 트리, 체크만 교체):

```
HEAD의 doc-commands:  PASS  5 documented command(s) install what the tests need and run as written
새 doc-commands:      NEW   README.md: `uv sync --extra compose` installs 136 distribution(s)
                            but the tests import 3 it does not provide:
                              datasets (datasets, imported by test_axes.py);
                              peft (peft, imported by test_applied.py);
                              transformers (transformers, imported by test_axes.py, test_probe.py)
```

제보는 `peft` 1건이었고 전수 수집하니 3건이었다.

**제보의 메커니즘 서술은 틀렸고 결론은 맞다.** "collection에서 실패한다"가 아니다 —
세 건 다 함수 안 import라 collection은 통과하고 해당 테스트만 `ImportError`로 죽는다.
`importorskip`/try 로 감싸여 있지도 않으니 skip이 아니라 error다. 문서대로 설치한
깨끗한 clone에서 스위트가 녹색이 아니라는 결론은 그대로 선다.

**남은 것은 감사 레인 몫이 아니다.** 체크는 고쳤고 지금 정직하게 실패한다. 해소는
(E) 문서화된 명령에 `--extra native`를 더하거나 (F) `native`의 해당 패키지를 `compose`로
옮기는 것이며, 전자는 실측으로 통과를 확인했다. 수리 내용과 판단 근거는
`docs/CONTRACTS.md` §5에 있다.
