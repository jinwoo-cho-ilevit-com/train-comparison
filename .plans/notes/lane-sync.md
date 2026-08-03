# lane-sync — 머지 단계로 넘기는 것

이 워크트리에서 이번 세션에 직접 실행하거나 핀된 소스를 열어 확인한 것만 적는다.
파드에서만 답할 수 있는 것은 여기 한 문장씩만 남기고 단정하지 않는다.

## 1. 제거한 두 호스트 싱크가 스텝당 실제로 얼마를 부풀렸는지는 측정 안 함

이 호스트에 CUDA가 없어 `bool((lengths > 0).all())`/`torch.equal(mask, expected)`가
스텝 시간에 실제로 얹던 비용은 재현할 수 없었다 — `docs/methodology.md`가 이미
프로파일러 오버헤드에 대해 쓰는 것과 같은 이유로, 수치를 추정해 적지 않는다.

## 2. `loss=cached_mnrl`이 이 수정으로 `mnrl`과 실제로 같은 스텝 시간을 내는지는 파드 질문이다

GradCache는 `encode`를 마이크로배치당 4회, 평범한 경로는 2회 호출하므로 싱크
횟수가 원래 달랐다는 것이 이번 finding의 핵심이었는데, 두 경로 다
`trainbench/probe/steps.py::encode` -> `last_token_pool`을 거쳐 같은
`padding_preverified()` 스킵을 타는 것은 코드로 확인했지만, 이 격차가 실제로
사라졌는지는 GPU에서 두 loss 값을 나란히 재야 답이 나온다.

## 3. (해결됨) 검증을 `trainbench/collate.py`로 옮겨 사전 전체 패스와 그 비용 질문 자체를 없앴다

`scripts/bench.py::validate_padding_before_timing`(설정당 로더를 한 번 통째로
다시 읽던 사전 패스)는 삭제했다. `Collate.__call__`과
`PretokenizedCollate.__call__`이 `attention_mask`를 만드는 바로 그 자리에서
`embedding.assert_padding_conforms`를 직접 호출하도록 옮겼다 — CPU, DataLoader
워커 프로세스, `to_device`와 타이머보다 훨씬 전. 실제로 그려지는 모든 배치가
그 배치를 만드는 김에 검증되므로, 이전에 남겨뒀던 "512개 배치 x 파드 약
45회" 추가 패스 비용 질문은 더 이상 존재하지 않는다 — 측정할 대상 자체가
없어졌다. `scripts/bench.py::train()`은 여전히 `padding_preverified()`로
`last_token_pool`의 중복 검사를 건너뛰지만, 그 스킵의 근거가 "사전에 이
로더를 한 번 다 읽었다"에서 "이 배치는 이미 collate를 통과했다"로 바뀌었을
뿐이다.

## 4. `torch.accelerator.synchronize`가 A100 파드에서 `torch.cuda.synchronize`와 타이밍 정밀도까지 동일한지는 파드 질문이다

이 노트북의 `mps` 디바이스에서 `torch.accelerator.synchronize(device)`가
`torch.cuda.synchronize`와 같은 자리에서 같은 예외 형태로 동작하는 것은 확인했지만,
CUDA 스트림 동기화 정밀도가 이 우회 경로에서 그대로인지는 실제 CUDA 디바이스에서만
검증된다.

## 5. `packed_last_token_pool`은 같은 모양의 호스트 싱크를 그대로 갖고 있고, 이번 범위 밖이었다

`trainbench/embedding.py::packed_last_token_pool`도 `bool((lengths > 0).all())`과
`int(offsets[0])`/`int(offsets[-1])`로 호스트 읽기를 한다 — 이번 finding이
`last_token_pool` 하나만 지목했고 `dataloader.packing=true`는 별도 축이라
스코프를 넓히지 않았다. `packed_last_token_pool`에 같은 처방(사전 1회 검증 +
`padding_preverified()` 스킵)을 적용할지는 그 축을 다루는 레인이 판단할 문제다.
