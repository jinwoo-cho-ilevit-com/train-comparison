# Lane G — 파드가 있어야만 답할 수 있는 것

- `optim=adamw_8bit`는 bitsandbytes를 그대로 쓰지만, qlora가 쓰던 `BitsAndBytesConfig`
  경로(`axes.load_kwargs`)가 이번에 코드에서 사라졌다 — 같은 패키지의 다른 코드 경로가
  CUDA 파드에서 여전히 정상 빌드되는지는 실측 안 함.
- 남은 두 모델(qwen3_vl_emb_2b, qwen3_5_0_8b) 29개 실험이 실제 A100 파드에서
  기동되는지는 `--dry-run`이 답하지 않는다 — 구성 합성과 구조만 검증했다.
- `scripts/report.py`의 `MODELS` 리스트는 `gemma4_e2b`를 의도적으로 남겨 과거 실측을
  렌더링하지만, 실제 `jinwoo-cho/trainbench-results`에서 그 결과를 내려받아
  렌더링이 여전히 맞게 나오는지는 이 세션에서 확인하지 않았다.
