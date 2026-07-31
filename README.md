# train-comparison

Qwen3-VL-Embedding-2B / Qwen3.5-0.8B / gemma-4-E2B 세 모델의 **임베딩 학습 속도**를
실측 비교하는 벤치마크. full finetuning과 LoRA를 대상으로, 최적화 기법을 축으로
ablation하여 throughput x memory x 품질의 Pareto frontier를 산출한다.

세 모델은 병목이 서로 다르다 — Qwen3-VL은 attention, Qwen3.5는 linear attention
커널(레이어의 75%가 Gated DeltaNet), gemma-4-E2B는 옵티마이저 메모리(5.1B 중 약
2.8B가 per-layer embedding). "모델 무관 최적 레시피"가 존재하지 않는다는 것이
검증하려는 가설이다.

- 연구 설계와 근거: [PLAN.md](PLAN.md)
- 환경/프레임워크 지원 현황: [docs/support-matrix.md](docs/support-matrix.md)
- 에이전트 작업 지침: [AGENTS.md](AGENTS.md)

## 실행

시크릿은 Infisical에서 주입한다.

```bash
uv sync --group dev
infisical run --env=dev -- uv run pytest
infisical run --env=dev -- uv run python scripts/env_report.py device=cpu data.limit=4
```

실험 변형은 전부 Hydra config 조합으로 표현한다. 축을 추가할 때 코드를 고치면
컨벤션 위반이다.

```bash
uv run python scripts/env_report.py model=gemma4_e2b attn=fa4 precision=mxfp8 freeze=ple
```

## 측정 규율

산출물이 속도 수치이므로, 아래를 어기면 결과가 조용히 오염된다. 일부는
`trainbench/config_schema.py`에서 실행 전에 거부된다.

- 타이밍 런과 프로파일링 런을 분리한다. `torch.profiler`는 iteration time을
  20~44% 부풀린다
- 측정 중에는 deterministic 모드를 끈다. 커널 autotuning이 측정 대상이기 때문이다
- 학습 데이터를 network volume에서 읽지 않는다. pod-local NVMe에 두지 않으면
  dataloader 축이 파이프라인이 아니라 볼륨을 측정하게 된다
- 같은 축의 설정을 여러 pod에 나누지 않는다. 모든 pod이 canonical baseline을
  돌리고, 편차 3%를 넘으면 그 pod 결과는 폐기한다
