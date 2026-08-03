# lane V 노트 — 파드에서만 답할 수 있는 것

이 레인은 `trainbench/record.py`와 `scripts/report.py`만 고쳤고, 파드를 띄우지 않았다.
아래는 이 워크트리에서 실행해 확인할 수 없고 실 파드 결과가 있어야 답할 수 있는 것이다.

- `kernels`가 설치돼 있고 `flash-attn`이 없는 이미지에서 `attn=fa2` 요청이 실제로
  `FLASH_ATTN_KERNEL_FALLBACK`을 타 `kernels-community/flash-attn2`로 대체되는지는
  코드(`transformers/modeling_flash_attention_utils.py:65-66`)로만 확인했다 — 실제로 그 경로를
  타는 런의 `packages.kernels`/`packages.flash-attn` 조합은 파드가 있어야 본다.
- `triton` 버전이 여섯 이미지 사이에서 lockfile이 보여주는 것 이상으로 갈리는지는
  `scripts/verify_env.py`/`scripts/env_report.py`가 실제 이미지 안에서 실행돼야 안다 —
  lockfile은 해석 결과이지, 이미지에 실제로 설치된 것의 증거가 아니다.
- `fla-core`와 `flash-linear-attention`은 오늘 여섯 lockfile 모두에서 버전이 같다(래퍼가
  `==`로 코어를 고정한다). 둘이 갈라지는 날이 오는지, 오면 `kernel.name=fla` 셀의 번호가
  어느 쪽 버전에 묶이는지는 실 해석이 갈릴 때만 답이 나온다 — 지금은 기록해 둘 뿐이다.
- `_axis_confound_table`과 `cell()`의 축 거부/실패 구분은 전부 `tests/test_report.py`의
  픽스처로만 실행됐다. 이 저장소는 GPU에서 완결된 타이밍 런이 역사상 한 번도 없어서(`AGENTS.md`),
  실제 `scripts/bench.py::refusal_record`/`failure_status`가 만든 레코드를 이 렌더링이 어떻게
  다루는지는 첫 파드 결과가 올라와야 처음 본다.
