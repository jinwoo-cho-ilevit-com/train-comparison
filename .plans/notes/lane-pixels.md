# lane-pixels — 머지 단계로 넘기는 것

이 워크트리에서 이번 세션에 직접 실행하거나 핀된 소스를 열어 확인한 것만 적는다.
파드에서만 답할 수 있는 것은 여기 한 문장씩만 남기고 단정하지 않는다.

## 1. `configs/data/quality.yaml`의 `text_token_ceiling`은 이 코퍼스에서 측정하지 못했다

65536행 전체 스캔이 이 호스트에서 `train-00000-of-00010.parquet` 첫 조각부터
`The read operation timed out`로 반복 실패했다 — speed.yaml의 285(2048행,
동일 upstream)를 가져다 썼다. 대역폭이 되는 호스트나 파드에서 quality 자체 코퍼스를
다시 스캔해 이 값을 갱신해야 한다.

## 2. `ms_swift`/`axolotl`/`unsloth` 세 어댑터의 `revision` 전달은 이 호스트에서 실행 검증하지 못했다

패치는 각 프레임워크의 핀된 소스(`swift/model/register.py:525,603`,
`axolotl/loaders/model.py:223-224` + `tokenizer.py`/`processor.py`,
`unsloth/models/vision.py:1103,1316,1469`)를 직접 읽고 넣은 것이지만, 이 호스트에는
세 프레임워크 다 설치되어 있지 않아 실제 로드가 그 revision을 받았는지는 파드에서만
확인된다.

## 3. KV 캐시 절감분의 실측 GPU 메모리는 파드 질문이다

`trainbench/probe/steps.py::encode`에 `use_cache=False`를 추가한 근거는 실제
config(`AutoConfig.from_pretrained`)에서 읽은 레이어 구성으로 계산한 텐서 바이트 수
(qwen3_vl_emb_2b 약 580MiB, qwen3_5_0_8b 약 61.5MiB, batch=4, 이 코퍼스의 캡 적용 후
최대 행 폭 기준)이지, CUDA 할당자의 실측 피크 메모리가 아니다 — 파편화/할당자 오버헤드는
GPU 없이는 답할 수 없다.

## 4. gemma4 관련 죽은 코드는 그대로 두었다

`model.max_tokens_per_image`와 `config_schema.py`의
`_an_image_token_cap_is_gemma4_only` 검증기는 이 레인의 새 검증기
(`_the_pixel_cap_fits_the_sequence_budget`)가 `arch not in ("qwen3_vl", "qwen3_5")`로
건드리지 않고 지나간다. gemma4 제거 레인이 두 개를 함께 지울 때 이 두 검증기가
서로 겹치지 않는다는 것만 확인하면 된다.
