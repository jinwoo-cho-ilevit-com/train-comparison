# Phase 0 실패 12칸을 닫는다

## Context

2026-08-03 재빌드한 이미지로 Phase 0 18칸을 A100에서 다시 쟀다. 18/18이 결과를
발행했고 6 OK / 12 FAIL이다. 실패 12건은 세 원인으로 갈리며, **원인마다 성질이
다르다** — 하나는 설정 누락, 하나는 배선 누락, 하나는 프로브 경로의 미구현이다.
그리고 네 번째로, 고칠 수 없고 고쳐서도 안 되는 것이 하나 섞여 있다.

이 계획은 그 넷을 구분해 앞의 셋을 닫고, 넷째는 사실대로 기록한다.

핀된 소스를 읽어 확정한 사실만 쓴다. 이 저장소는 "보통 그렇게 동작한다"에서
출발한 프로브가 핀된 버전에서 전부 깨진 전례를 `AGENTS.md`에 남겨두었다.

## 원인 1 — qwen3_5 6칸: 설정 누락 (재빌드 불필요)

**증상**: `kernel.name: requested 'none', applied 'fla'`. 6개 프레임워크 전부,
`model=qwen3_5_0_8b`에서만.

**사실**: 결함이 아니다. transformers 5.14.1은 `fla`가 설치돼 있고 버전이 0.2.2
이상이고 CUDA가 있으면 qwen3_5 모델링 모듈을 import하는 시점에 무조건 바인딩한다
(`transformers/utils/import_utils.py:869-872`, `models/qwen3_5/modeling_qwen3_5.py:73-78`).
끄는 환경변수도 config 플래그도 registry도 없다. `attn_implementation`은
`full_attention` 층에만 닿고 `linear_attention` 층에는 닿지 않는다.

저장소는 이미 2026-08-02에 이 사실로 결론을 내렸다 —
`configs/experiment/_baselines.yaml:24-34`가 canonical baseline을 `kernel=fla`로
선언하고 이유를 적었다. `docs/methodology.md:137-165`도 "해소됨"으로 닫았다.

**놓친 것**: `configs/experiment/phase0-*-qwen3_5_0_8b.yaml` 6개가 `kernel`을
오버라이드하지 않아 스키마 기본값 `none`을 요청한다. `docs/CONTRACTS.md:793`이
`phase2-loss-qwen3_5_0_8b.yaml`에서 같은 누락을 이미 한 번 기록했다.

**고칠 것**: 그 6개 매니페스트의 `overrides`에 `kernel=fla`를 넣는다. 왜 `none`이
아닌지는 `_baselines.yaml`에 이미 적혀 있으므로 **거기를 가리키고 복사하지 않는다**.

**재빌드 불필요.** `configs/`는 이미지에 COPY되지 않는다
(`docker/Dockerfile.framework`가 굽는 것은 `pyproject/uv.lock/README`,
`envs/${FRAMEWORK}`, `trainbench`, `scripts`, `entrypoint.sh`뿐). 오케스트레이터가
랩톱에서 합성해 `TRAINBENCH_CONFIG_JSON`으로 넘긴다.

**같이 확인할 것**: `phase2-loss-qwen3_5_0_8b.yaml`도 아직 같은 상태인지.
`configs/experiment/*.yaml` 전부를 훑어 qwen3_5를 쓰면서 `kernel`을 선언하지 않은
매니페스트가 남지 않게 한다.

## 원인 2 — axolotl 3칸: probe가 `step_context`를 열지 않는다

**증상**: `infonce_backward` → `expected mat1 and mat2 to have the same dtype,
but got: float != c10::BFloat16`.

**사실**: `scripts/bench.py:278`은 이미 `axes.step_context(config, required_context)`로
올바르게 감싼다. 그런데 **probe 경로는 `loader.load()`를 거치지 않는다** —
`trainbench/probe/axolotl.py:83`이 자기 모듈의 `load()`를 직접 부르므로 어댑터의
`required_step_context`를 보지 못한다. `grep -rn "autocast\|step_context"
trainbench/probe/` 결과가 0건이다.

죽는 matmul은 InfoNCE가 아니다. axolotl 0.18.0이 `embed_tokens`·`lm_head`·`*norm*`을
fp32로 올리고(`loaders/model.py:1025-1047`, 복원 분기 네 조건 전부 거짓) 나머지를
bf16으로 두므로, fp32 활성이 layer 0 `q_proj`의 bf16 가중치를 만난다.

unsloth와의 대조가 이 진단을 확증한다: unsloth는 **norm만** fp32로 올리고 RMSNorm은
출력을 입력 dtype으로 되돌리므로 fp32 활성이 bf16 Linear에 닿지 않는다. 그래서
같은 `mixed(bf16,fp32)`를 읽으면서도 `infonce_backward`는 통과한다.

**고칠 것**: `trainbench/probe/axolotl.py`가 `infonce_backward`를
`axes.step_context(config, required)` 안에서 돌게 한다. 요구도 설립자도 이미 있고
계약에도 맞으므로 **계약 개정 없음**. 요구를 어디서 가져올지는 두 갈래다 —
`loader.ADAPTERS`에서 읽거나 probe 모듈이 선언하거나. 어느 쪽이든 **정의가 두
군데 생기지 않게** 한다(`code-craft.md`).

상류와의 차이 둘을 주석이 아니라 노트에 남긴다: accelerate는 `model.forward`만
감싸고 손실·backward는 감싸지 않으며, `convert_outputs_to_fp32`로 한 겹 더 두른다.
우리는 forward+loss+backward를 함께 감싼다. 이 차이가 InfoNCE 로짓의 dtype을
바꾼다.

## 원인 3 — tevatron 3칸: probe가 프레임워크의 forward를 부르지 않는다

**증상**: `infonce_backward` → `EncoderModel.forward() got an unexpected keyword
argument 'input_ids'`.

**사실**: 핀된 tevatron `dd06310`의 시그니처가
`forward(self, query: Dict[str,Tensor]=None, passage: Dict[str,Tensor]=None)`이고
반환은 `EncoderOutput(q_reps, p_reps, loss, scores)`다. `last_hidden_state`가 없다.
**손실까지 forward 안에서 계산한다** — 어댑터가 `owned_axes`로 `loss.name`과
`parallel.cross_device_negatives`를 이미 프레임워크 소유로 선언해 둔 것이 그 사실과
맞는다.

`.plans/research/tevatron.md` §3.3이 **한 캠페인 전에 이 실패를 정확히 예언했다**:
"pad_token_id 만 고치면 `dense_model_load` 는 초록이 되지만 `infonce_backward` 는
`TypeError` 로 넘어간다. (…) 두 개를 한 번에 준비하지 않으면 파드 한 시간을 또 쓴다."
그 한 시간을 썼다. 이번에 반복하지 않는다.

**고칠 것**: `trainbench/probe/tevatron.py`에 자체 `_backward` 클로저를 둔다.
sentence_transformers가 이미 같은 자리에 같은 모양을 갖고 있다
(`trainbench/probe/sentence_transformers.py:107-126`) — **그것을 본으로 삼고
증거 생성은 `steps.training_step_evidence`를 그대로 재사용한다.**

배치를 절반으로 갈라 `query`/`passage` 딕트를 만들고
`model(query=..., passage=...).loss`를 쓴다. `steps.encode`도
`embedding.info_nce`도 `last_token_pool`도 타지 않는다 — pooling과 정규화가
`dense.py::_pooling`에서 이미 일어난다.

**계약 개정 없음.** sentence_transformers 건은 `Step`을 넓힌 게 아니라 거부 게이트
(`scripts/bench.py::refuse_a_forward_this_harness_cannot_call`)를 추가한 것이었고,
probe 쪽은 모듈별 클로저가 이미 허용된 모양이다.

**반드시 기록할 confound**: `DenseModel.load`가 `temperature`를 인자로 받지 않아
`self.temperature`가 항상 `1.0`이다(`encoder.py:38`, `:159-165`).
`config.loss.temperature`가 아니다. 조용히 두면 tevatron 칸만 다른 온도로 잰
숫자가 된다. 적재 후 대입하든 `DenseModel.build`를 쓰든 **택하고 그 사실을 남긴다**.

**주의**: `if query:`는 `is not None`이 아니다(`encoder.py:53-54`). 빈 딕트는 조용히
추론 경로로 빠져 `loss=None`을 돌려준다.

## 원인 4 — precision `mixed(bf16,fp32)`: 고치지 않는다

axolotl 3칸 + unsloth 3칸의 `axes_verified`가 `requested 'bf16', applied
'mixed(bf16,fp32)'`로 실패한다. **autocast를 켜도 녹지 않는다** — `torch.autocast`는
파라미터 dtype을 바꾸지 않고, `_capture_precision`은 파라미터를 읽는다.

`mixed(bf16,fp32)`는 계약이 **영구 불일치로 동결**한 값이다
(`tests/contract/test_applied_axes.py`의 `UNNAMEABLE` 표). `assert_matches`가
가장 가까운 라벨로 반올림하는 대신 거부하게 만드는 장치 자체다.

`FRAMEWORK_OWNABLE`은 `("loss.name", "parallel.cross_device_negatives")` 둘뿐이고
이것도 계약이 동결했다. precision을 넣는 것은 계약 개정이다.

**결정: 사실대로 두고 기록한다.** 이것은 버그가 아니라 "프레임워크가 요청과
다른 수치 체제로 학습한다"는 사실의 정직한 보고이고, `docs/CONTRACTS.md:195-199`이
이미 그렇게 적고 있다. 결과적으로 axolotl은 2체크 실패에서 1체크 실패가 되고
unsloth는 1체크 실패 그대로다. **최종 매트릭스에 6칸이 FAIL로 남는 것이 의도된
결과다** — 다음 사람이 이걸 미완으로 읽지 않게 `docs/support-matrix.md`에 한 문단으로
남긴다.

`FRAMEWORK_OWNABLE`에 `precision.name`을 넣는 길은 검토했고 택하지 않았다:
동결된 계약 다섯 자리를 건드려야 하고, 그 두 프레임워크에서 precision 축
ablation이 닫힌다.

## 곁다리로 닫는 것

- `docs/methodology.md` §9가 `kernel_modules` 커버리지를 `측정 안 함`으로 적는데
  이제 **18/533**이 측정됐다. 18은 24층 중 `linear_attention` 층의
  `FusedRMSNormGated`다(`configuration_qwen3_5.py:112-117`이 4층마다 1층을
  full_attention으로 둔다). 같은 파일 `:483-490`이 첫 GPU 파드에서 읽으라고
  지정한 값이 정확히 이것이다.
- **남는 한계로 적을 것**: `applied='fla'`는 fla 클래스 18개가 모델에 있다는
  증명이지 fused GDN *연산*이 돌았다는 증명이 아니다. `chunk_gated_delta_rule`과
  `causal_conv1d_fn`은 함수 속성이라 `_module_roots`에 안 잡힌다
  (`.plans/research/axis-libraries.md:511`).

## 고칠 파일

```
configs/experiment/phase0-{axolotl,ms_swift,native,sentence_transformers,tevatron,unsloth}-qwen3_5_0_8b.yaml
configs/experiment/phase2-loss-qwen3_5_0_8b.yaml     (같은 누락이면)
trainbench/probe/axolotl.py
trainbench/probe/tevatron.py
tests/test_probe*.py 또는 해당 스텁 테스트
docs/methodology.md                                  §9 커버리지
```

계약 파일(`tests/contract/*`)은 **건드리지 않는다.** 셋 다 계약 개정이 필요 없다.

## 실행 순서 — 설정 먼저, 코드 나중

원인 1은 재빌드가 필요 없고 원인 2·3은 필요하다는 비대칭을 쓴다. 설정을 먼저
띄우면 **원인 1의 진단이 맞는지 코드 변경과 섞이기 전에 독립적으로 확인된다.**

```
1  설정 6개 매니페스트 수정 + 랩톱 게이트          빌드 0분
2  qwen3_5 6칸만 파드                              ~7분      <- 원인 1 독립 확인
3  (2와 동시에) 코드 수정 + 스텁 테스트 + 게이트
4  push -> Actions                                 ~9분
5  18칸 전체 파드, --max-concurrent 12             ~7분
6  report.py 병합 + 아티팩트에서 직접 확인
```

총 파드 24대. 2단계에서 6칸이 안 닫히면 원인 1의 진단이 틀린 것이므로 **거기서
멈추고 다시 읽는다.** 코드 수정을 밀어붙이지 않는다.

## 검증

랩톱에서 죽일 수 있는 것을 파드로 보내지 않는다. 파드 왕복 1회가 약 16분이다.

1. **랩톱**: 스텁 테스트로 모양을 못박는다 — probe/tevatron이 `query`/`passage`
   딕트를 만들어 `model(query=,passage=)`를 부르는지, probe/axolotl이
   `axes.step_context`를 여는지. sentence_transformers 수정이 검증된 방식과 같다.
2. **랩톱**: 네 게이트 — `ruff check && ruff format --check`,
   `infisical run --env=dev -- uv run pytest`,
   `pytest tests/contract -q` (**122 passed 유지**, 줄면 계약을 약화한 것),
   `python scripts/audit_plan.py`.
3. **설정만 먼저 검증(재빌드 0분)**: `--experiment 'phase0-*-qwen3_5_0_8b'
   --dry-run`으로 합성이 통과하는지 본 뒤, 6칸만 실제로 띄운다. 원인 1이 맞으면
   이 6칸이 즉시 닫힌다.
4. **코드 수정 후 한 번만 재빌드**: push → Actions ~9분(코드만 바뀌면 `uv sync`
   세 층이 캐시에 남는다. 실측 `8m17s`/`8m46s`).
5. **파드**: `--experiment 'phase0-*' --tag <새 sha>`. `--max-concurrent`를 6에서
   12로 올린다 — probe는 타이밍을 재지 않으므로 호스트 편차가 결과를 오염시키지
   않고, "같은 축은 같은 파드"는 측정 런에 걸리는 규칙이다.
6. `report.py`로 매트릭스를 다시 병합하고 **각 실패가 사라진 것을 아티팩트에서
   직접 읽어 확인한다.** 스위트가 초록인 것은 증거가 아니다.

## 하지 않는 것

- 계약 파일 수정
- precision `mixed(bf16,fp32)`를 녹이는 시도 (원인 4)
- 이미지에서 `flash-linear-attention` 제거 — `kernel=fla` 축 값과 canonical
  baseline을 동시에 죽이고, axolotl 이미지에서는 axolotl을 빼야만 가능하다
- 리뷰 major 23 + minor 54 — 범위 밖
