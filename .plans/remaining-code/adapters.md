# adapters — 어댑터 레지스트리 + 빌드 지문 (wave 2)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`.
> **리서치 필독 (전부)**: `.plans/research/{unsloth,axolotl,ms-swift,sentence-transformers,
> tevatron,transformers-varlen-prompt}.md`.
> 가장 크고 설계 결정을 품는 레인이다. split·capture·kernels·probe 가 이미 머지됐다.

## 목표

`bench.py` 가 native 하나만 적재할 수 있는 상태를 끝낸다. **native 말고는 어떤 프레임워크도
숫자를 낼 수 없다** — Phase 3 가 통째로 막혀 있다.

## Owns

```
trainbench/loader.py                    ⊕ 신설. split 이 만들지 않고 남겨두었다
trainbench/probe/native.py
trainbench/probe/unsloth.py
trainbench/probe/ms_swift.py
trainbench/probe/axolotl.py
trainbench/probe/registry.py
tests/test_loader.py                    ⊕ 신설
tests/contract/test_loader_bench.py     마커 1개 제거만
```

**`scripts/bench.py` 는 건드리지 않는다.** split 이 wave 0 에서 seam 을 남겼다:
`bench.py` 가 `trainbench.loader` 를 import 해보고 없으면 native 로 떨어진다.
네가 `loader.py` 를 만들면 `bench.py` 는 **한 줄도 바뀌지 않고** 그것을 탄다.
seam 의 정확한 이름과 필드는 `.plans/notes/split.md` 에 있다.

## 근본 원인

`framework="native"` 리터럴은 증상이었고 split 이 지웠다. 원인은 그 위 세 줄이다 —
`from transformers import AutoModel, AutoProcessor` 와 두 `from_pretrained`.
그리고 **여섯 프로브 모듈 전부가 `run(config, device, report) -> None` 만 노출한다.**
적재는 `_load` 클로저 안에 있고 **재사용 가능한 `load()` 를 내주는 파일이 하나도 없다.**

## 작업 1 — 어댑터 레지스트리

`trainbench/loader.py` 에 프레임워크별 함수를 뽑아낸다.
계약(`tests/contract/test_loader_bench.py:558-567`)이 요구하는 것:

```python
set(loader.ADAPTERS) == FRAMEWORKS          # 여섯 개, FrameworkConfig 에서 유도된다
callable(loader.load)
{f.name for f in dataclass_fields(loader.AdapterOut)} == ADAPTER_OUT_FIELDS
```

`AdapterOut` 여덟 필드 (`test_loader_bench.py:38-51`):
```
framework  model  processor  step  owned_axes  required_step_context
fingerprint  documented_entry_point
```

이름은 split 이 `bench.py` 의 seam 에 이미 같은 이름으로 심어두었다. **그대로 꽂힌다.**

그 마커는 `xfail(strict=True)` 다. **스텁으로 세 단언을 만족시키면 XPASS 로 빨개진다.**
마커를 지우려면 실제로 여섯이 적재돼야 한다.

## 작업 2 — 빌드 지문 (축 G)

**프레임워크가 우리가 요청하지 않은 것을 바꾼 것이 어디에도 기록되지 않는다.**
`applied.py` 는 요청한 축의 되읽기이고, 지문은 그 여집합이다.

어댑터가 돌려줘야 할 것:
- 모듈 클래스 이름들
- 파라미터별 dtype
- 학습 가능 파라미터 이름의 집합
- **실제로 바인딩된 attention fn 의 신원과 mask fn 등록 여부**

여섯 프레임워크에 걸쳐 diff 하면 모든 차이가 교란 요인으로 드러난다.
**이번 캠페인의 실패 셋은 정확히 이 지문이 잡았을 것이다**:
unsloth 가 전 파라미터를 얼린 것, axolotl 이 모듈 둘만 fp32 로 남긴 것,
unsloth × gemma-4 가 텐서를 60개 더 만든 것.

마지막 항목(attention fn 신원 + mask fn 등록)은 **경계 `kernel-provenance`** 다.
`BUILD_FINGERPRINT_KEY` 는 **`"attention"`** 이고 `"kernel"` 이 아니다 —
`kernel.name` 은 다른 축이다. 이 payload 는 `tests/fixtures/kernel_fingerprint.sample.json` 과
`tests/fixtures/adapter_out.sample.json` 의 `fingerprint` 가 **같은 객체**임을 계약이 검증한다
(`test_the_two_fixtures_that_carry_this_payload_agree_with_it`).
kernels 레인이 wave 1 에서 만든 `trainbench/kernels.py` 를 쓴다.

## 작업 3 — axolotl autocast (결정 1)

axolotl 은 `embed_tokens`/`lm_head` 만 fp32 로 두고 나머지를 bf16 으로 적재한다.
복귀 분기는 adapter/FSDP/cut_cross_entropy 중 하나가 있어야 도는데 프로브는 셋 다 없다.
상류가 문제없는 이유는 HF Trainer 의 autocast 안에서 돌기 때문이다.
**정확한 줄과 원문은 `.plans/research/axolotl.md` 에 있다.**

**결정: autocast 로 감싸 axolotl 을 그대로 잰다.**

`docs/CONTRACTS.md §2` 의 계약("`axes.step_context` 가 precision 컨텍스트의 유일한 집이다")을
지키려면 **프레임워크가 요구하는 컨텍스트를 그 자리로 끌어와야 한다** —
어댑터가 `required_step_context` 로 그 요구를 표현하고, `step_context` 가 그것을 세운다.
`AdapterOut.required_step_context` 필드가 그것이고, 계약의
`test_established_by_resolves_to_live_code` 가 `established_by` 를 살아 있는 코드로 해석한다.

**native(순수 bf16)와 axolotl(autocast)이 다른 수치 체제에서 비교된다는 사실이 결과에 남아야
한다.** `CONTRACTS.md` 본문 개정은 **integrate** 레인의 것이다 —
`.plans/notes/adapters.md` 에 실제 배선을 적어 넘긴다.

## 작업 4 — tevatron 의 다른 시그니처 (결정 5)

`DenseModel.forward` 가 인코딩·풀링·정규화·스코어링·InfoNCE·분산 게더를 전부 자기가 한다.
우리 하네스는 그것을 `steps.encode` + `embedding.info_nce` + `axes._loss` +
`parallel.cross_device_negatives` 넷으로 나눠 갖고 있다. 정확한 줄은
`.plans/research/tevatron.md`.

**결정: 프레임워크의 학습 스텝을 그대로 잰다.** 그러므로 tevatron 칸에서 `loss` 와
`parallel.cross_device_negatives` 는 **`framework_owned`** 로 기록된다.
그 상태 자체는 **capture 레인이 wave 1 에서 만들었다.** 이 레인은 쓰기만 한다.

`steps.encode` 는 `model(**batch)` 가 `last_hidden_state` 를 주기를 기대한다. tevatron 경로에는
통하지 않는다. **어댑터별 encode 가 필요하고 그것이 이 레인의 설계 결정이다.**

`trainbench/probe/tevatron.py` 의 **적재 shim 은 probe 레인이 wave 1 에서 넣었다.**
그 파일에서 이 레인의 몫은 forward/encode 경계뿐이다 — 소유는 probe 에 남아 있으므로
필요한 변경은 `.plans/notes/adapters.md` 로 넘기거나, 새 경계를 `loader.py` 쪽에 만든다.

## 작업 5 — 문서화된 학습 진입점과의 차이

여섯 중 다섯이 "그들에게서 적재하고 우리 루프로 학습"이다:

```
native                 AutoModel.from_pretrained         기준. 일치
unsloth                FastVisionModel.from_pretrained   for_training() 미사용
ms_swift               get_model_processor               자체 trainer 미사용
sentence_transformers  SentenceTransformer(...)          자체 loss/trainer 미사용
tevatron               dense.load(...)                   forward 가 스텝 전체
axolotl                ModelLoader(cfg, tok).load()      자체 Trainer 미사용
```

**이번 캠페인의 실패 셋이 전부 이 간극에서 나왔고 셋 다 답이 핀된 소스 안에 있었다.**
필드 사례: unsloth 46,000 tok/s 에서 grad norm 0.

각 프레임워크가 문서·예제로 지목하는 학습 진입점을 **핀된 소스에서 인용**하고
(리서치 브리프 6개가 이미 그것을 담고 있다 — 확인하고 쓴다),
우리가 쓰는 것과 다르면 그 차이를 `AdapterOut.documented_entry_point` 에 기록한다.

## 완료 조건

1. 여섯 프레임워크가 공통 진입점으로 적재되고 `bench.py` 가 실제 프레임워크 이름을 넘긴다 →
   `infisical run --env=dev -- uv run pytest tests/contract/test_loader_bench.py -q`
   → **13 passed, 0 xfailed**
2. 어댑터가 빌드 지문을 돌려준다 → `uv run pytest tests/test_loader.py -k fingerprint`
3. 지문이 이번 캠페인의 세 사례를 잡는다 — 전 파라미터 동결, 모듈 둘만 fp32, 텐서 수 차이 →
   `uv run pytest tests/test_loader.py -k fingerprint_catches`
4. tevatron 칸에서 `loss`/`cross_device_negatives` 가 `framework_owned` 로 기록된다 →
   `uv run pytest tests/test_loader.py -k tevatron_owns`
5. axolotl 경로가 autocast 컨텍스트를 요구하고 그것이 `step_context` 로 적용된다 →
   `uv run pytest tests/test_loader.py -k axolotl_autocast`
6. 여섯의 문서화된 학습 진입점이 인용됐고 우리 경로와의 차이가 기록됐다 →
   `uv run pytest tests/test_loader.py -k documented_entry_point`
7. 각 검사를 되돌리면 죽는다. mutation 출력 인용. 사보타주 전에 `co_filename` 확인
8. 네 게이트. `plan-files` 가 `trainbench/loader.py`, `tests/test_loader.py` 로 빨간 것은
   실패가 아니다
9. **확인 안 함** — 이 레인이 확인할 몫이 둘 있다:
   - `Collate` 의 `processor(text=..., images=...)` 규약이 sentence_transformers 경로에서
     성립하는가
   - sentence_transformers 에 자체 loss 가 있는가

   둘 다 `.plans/research/sentence-transformers.md` 에 답이 있어야 한다. 없으면 핀된 소스를
   직접 열어 확인한다. **그래도 안 되면 "확인 안 함"으로 적는다.**

   그리고 파드 질문: 프레임워크가 실제로 적재되는가, `kernel=fla` 자동 통과가 CUDA +
   causal_conv1d 에 달려 있는가

## 하지 않는 것

- `scripts/bench.py`, `trainbench/probe/steps.py` — split / **probe** 레인
- 축 소유권 상태의 **정의** — **capture** 레인. 이 레인은 쓰기만 한다
- `trainbench/probe/tevatron.py` 의 적재 shim, `sentence_transformers.py` 의 동결 가드 —
  **probe** 레인 (wave 1). 어댑터 배선은 경계 `loader-bench` 에서 맞춘다
- `trainbench/kernels.py` — **kernels** 레인. 쓰기만 한다
- `docs/CONTRACTS.md` §2 개정 — **integrate** 레인. `.plans/notes/adapters.md` 로 넘긴다
- `envs/**` — **통합자 전용**
