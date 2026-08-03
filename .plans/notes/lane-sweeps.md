# Lane D — sweep manifest 생성 기록

## 생성한 파일 (15개, `configs/experiment/` 아래 신규)

- `phase3-attn-qwen3_vl_emb_2b.yaml`, `phase3-attn-qwen3_5_0_8b.yaml` (attn: sdpa/fa2/flex)
- `phase3-compile-qwen3_vl_emb_2b.yaml`, `phase3-compile-qwen3_5_0_8b.yaml` (compile: none/default/max_autotune)
- `phase3-dataloader-qwen3_vl_emb_2b.yaml`, `phase3-dataloader-qwen3_5_0_8b.yaml` (dataloader: torch/torch_packed/torch_pretokenized/torch_packed_pretokenized)
- `phase3-optim-qwen3_vl_emb_2b.yaml`, `phase3-optim-qwen3_5_0_8b.yaml` (optim: adamw_fused/adamw_8bit/muon)
- `phase3-peft-qwen3_vl_emb_2b.yaml`, `phase3-peft-qwen3_5_0_8b.yaml` (peft: full/lora)
- `phase3-gradient_checkpointing-qwen3_vl_emb_2b.yaml`, `phase3-gradient_checkpointing-qwen3_5_0_8b.yaml` (axis: `train.gradient_checkpointing`, 값 none/full/selective)
- `phase3-freeze-qwen3_vl_emb_2b.yaml`, `phase3-freeze-qwen3_5_0_8b.yaml` (freeze: none/vision_tower)
- `phase3-kernel-qwen3_vl_emb_2b.yaml` (kernel: none/liger, qwen3_vl 전용 — qwen3_5는 `axes.FLA_ARCHS`가 고정하므로 스윕 대상에서 제외)

기존 매니페스트는 손대지 않았고, `configs/experiment/` 밖은 건드리지 않았다.

## 총량

15개 pod, 42개 setting — 표와 정확히 일치. 모든 pod가 `baseline: canonical`, `gpu_type_id: NVIDIA A100-SXM4-80GB`를 명시하고, 7개 `qwen3_5_0_8b` 매니페스트는 전부 `overrides: [kernel=fla]`를 (다른 문자열로 쓰면 `held_constant`가 split로 읽으므로) `_baselines.yaml`/`phase2-loss-qwen3_5_0_8b.yaml`과 동일한 문자열로 박아 넣었다.

## 뺀 것과 이유

- `peft=qlora`: 지시대로 캠페인에서 제외. 파일은 남아있지만 어느 매니페스트도 참조하지 않는다.
- `attn=fa3`/`attn=fa4`: `configs/attn/`에 파일은 있지만 phase2-loss 주석이 이미 밝혔듯 Blackwell 전용 — A100 캠페인 범위 밖.
- `compile=regional`: `configs/compile/regional.yaml`은 존재하나 표에 없어 제외.
- `freeze=ple`/`freeze=vision_and_ple`: `AXIS_VALUE_COMPANIONS`가 `model=gemma4_e2b`를 요구 — gemma-4가 캠페인에서 빠졌으므로 이 두 값은 남은 두 모델 어느 쪽에도 합성되지 않는다.
- `dataloader=dali`/`dali_packed`: 표의 4값(torch 계열)만 요청받았고, DALI 백엔드는 언급이 없어 제외.
- `kernel` on `qwen3_5_0_8b`: 지시 3번대로 스윕하지 않음 — `held_constant`가 `kernel=fla` 고정을 깨는 split로 잡는다.
- gemma-4 전체: 지시대로 어떤 phase3 매니페스트도 만들지 않음.

## 검증

네 게이트 모두 이 세션에서 직접 실행:

- `uv run ruff check && uv run ruff format --check` → `All checks passed!` / `118 files already formatted`
- `infisical run --env=dev -- uv run pytest -q` → **1244 passed**, 0 failed (14 warnings, torch 내부 deprecation)
- `infisical run --env=dev -- uv run pytest tests/contract -q` → **117 passed** — 지시받은 기준선(117)과 동일, 변화 없음
- `infisical run --env=dev -- uv run python scripts/audit_plan.py` → **13/15 passing, rc 0** — `axis-values`와 `verdicts-closed`가 KNOWN(기존에 열려 있던 항목, 이 랩톱에 CUDA/triton/fla가 없어서/GPU 숫자가 필요해서 발생 — 내 변경과 무관)

`--dry-run --out`으로 전체 37개 experiment(기존 22 + 신규 15)를 합성했고, JSON을 직접 파싱해 15개 신규 매니페스트의 모든 run에 대해 (a) 축이 이름과 일치하는 값으로 해석됐는지, (b) 같은 pod의 다른 setting들 사이에서 선언한 축(과 compile=max_autotune의 `train.warmup_discard_steps` 동반값) 외에는 어떤 필드도 움직이지 않았는지를 확인했다 — 전부 일치. `check_axis_not_split`/`check_one_baseline_one_gpu`는 `load_experiments()` 내부에서 돌고, dry-run이 `ManifestError` 없이 완주했으므로 통과가 확인된다.

## 파드가 있어야만 답할 수 있는 것 (한 줄씩)

- `attn`: sdpa/fa2/flex 세 구현이 A100에서 실제로 얼마나 다른 처리량을 내는지는 이 dry-run이 답하지 않는다 — 세 커널 다 이 랩톱에 없다.
- `compile`: `max_autotune`이 `warmup_discard_steps=20`으로 충분히 워밍업을 흡수하는지, 또 실제로 `default`보다 빠른지는 GPU 측정 전엔 모른다.
- `dataloader`: packing/pretokenize가 줄이는 실제 wall-clock 오버헤드는 숫자로 나온 적이 없다.
- `optim`: `muon`이 pytorch-optimizer 경유로 이 두 모델 파라미터 분할(≥2D vs 나머지)에서 실제로 동작하고 수렴/스텝 속도가 어떤지는 미검증.
- `peft`: LoRA(r=32)가 이 두 모델에서 full finetuning 대비 실제 스텝 속도 이득이 얼마인지 미측정.
- `train.gradient_checkpointing`: `selective`가 `full`과 `none` 사이에서 실제로 의미 있는 메모리/속도 트레이드오프를 보이는지는 GPU 없이는 알 수 없다.
- `freeze`: `vision_tower` 동결이 두 모델의 vision tower 크기 차이(qwen3_vl_emb_2b vs qwen3_5_0_8b, 둘 다 `visual.*` 마커) 때문에 실제로 다른 폭의 속도 이득을 주는지 미측정.
- `kernel` (qwen3_vl only): `liger`가 qwen3_vl 아키텍처에 대해 `apply_liger_kernel_to_qwen3_vl` 엔트리포인트로 실제 패치되는지는 이 랩톱에 liger-kernel/triton이 없어 원리적으로 못 본다 — 이미지 안에서만 답이 나온다.

## 그 외 주의사항

- `train.batch_size=4`는 어느 신규 매니페스트에도 반복하지 않았다 (`configs/train/default.yaml`이 이미 4로 바뀌어 있어 상속됨). 다만 **이 값은 qwen3_5_0_8b에서만 실측된 값**이고 (2026-08-03, peak 80.7GB), `qwen3_vl_emb_2b`는 이 batch에서 어떤 pod도 돌아간 적이 없다 — 더 큰 모델이라 80GB 벽에 먼저 부딪힐 위험이 있고, 이는 파드를 켜 봐야 안다.
