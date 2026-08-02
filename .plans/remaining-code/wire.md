# wire — 레인이 소유 때문에 닫지 못한 배선 (wave 3a, 단독)

> 먼저 읽는다: `HAZARDS.md`, `PLAN.md`, 그리고 **`.plans/notes/*.md` 전부**(8개).
> 이 레인은 **혼자 돈다.** 여러 레인의 소유 경계를 가로지르는 작업만 모았고,
> 그것이 wave 로 나눌 수 없었던 이유다.

## 왜 이 레인이 있는가

wave 0~2 의 아홉 레인이 각자 자기 파일만 고쳤다. 그 결과 **한쪽 끝은 있는데 다른 쪽 끝이
없는 배선**이 남았다. 레인들은 그것을 숨기지 않고 `boundaryRequests` 와 `.plans/notes/` 로
올렸다 — 지금 그것을 잇는다.

## Owns

이 레인은 **아래 열거한 파일만** 건드린다. 그 밖은 손대지 않는다.

```
trainbench/axes.py
scripts/bench.py
trainbench/config_schema.py
trainbench/applied.py
configs/run/
scripts/audit_plan.py
docs/CONTRACTS.md          §2 호출 지점 표만
tests/                     위 변경이 요구하는 만큼
```

`tests/contract/` 는 **여전히 고치지 않는다.** 아래 작업 6 만 예외이고 그것도 근거가 필요하다.

## 작업 1 (BLOCKING) — `Built.owned_axes` 를 채우는 자가 없다

`trainbench/applied.py` 에 `Built.owned_axes` 가 있고 `applied._owned` 가 그것을 읽는다.
`trainbench/loader.py` 의 어댑터가 `owned_axes` 를 **선언한다.** 그런데 **둘을 잇는 것이 없다.**

결과: tevatron 어댑터가 선언한 `loss.name` / `parallel.cross_device_negatives` 가 레코드에서
`undetermined` 로 남고 `assert_matches` 가 그 셀의 timing 런을 **거부한다.**
결정 5(프레임워크의 학습 스텝을 그대로 잰다)가 코드에 도달하지 못한 상태다.

adapters 레인의 요청 그대로:
- `axes.assemble(..., owned_axes: Mapping[str, str] = ())` 를 받아 `Built(owned_axes=...)` 로 넘긴다
- `scripts/bench.py::build_run` 이 `owned_axes=binding.owned_axes` 를 전달한다

**소유권은 config 가 아니라 `Built` 에서 온다**는 capture 레인의 원칙을 깨지 않는다 —
어댑터가 만든 객체에서 오는 것이지 `framework=tevatron` 요청에서 오는 것이 아니다.
`applied.FRAMEWORK_OWNABLE` 밖의 축은 어댑터가 주장해도 거부돼야 하고,
`tests/test_loader.py::test_an_adapter_cannot_disclaim_an_axis_capture_will_not_let_it` 가
이미 그것을 지킨다.

## 작업 2 (BLOCKING) — axolotl 의 autocast 요구를 받을 자리가 없다

어댑터가 `required_step_context`(autocast/cuda/bfloat16)를 선언하고 `established_by` 가
살아 있는 `axes.step_context` 로 해석되는 것까지는 확인됐다. 그러나
`axes.step_context(config)` 는 **config 만 받고** bf16 에 `nullcontext` 를 돌려준다 —
어댑터의 요구를 받을 파라미터가 없다.

adapters 레인의 요청 그대로:
- `axes.step_context(config, required=None)` 가 `required.kind == "autocast"` 일 때
  해당 컨텍스트를 세운다
- `scripts/bench.py` 가 `binding.required_step_context` 를 넘긴다

**계약(`docs/CONTRACTS.md §2`)이 어댑터의 자체 `with` 를 금지하므로** 어댑터 쪽에서 닫을 수
없었다. 결정 1 은 "프레임워크가 요구하는 컨텍스트를 `step_context` 자리로 끌어온다"이고
이 작업이 그것이다. `CONTRACTS.md §2` 의 문구도 실제 배선을 읽고 고친다.

**native(순수 bf16)와 axolotl(autocast)이 다른 수치 체제에서 비교된다는 사실**이 결과에
남아야 한다.

## 작업 3 — `kernels_hub` 제거의 남은 지점

`.plans/notes/axes.md §1` 이 파일:줄로 표를 남겼다. **그 표를 그대로 적용한다.**
`config_schema.py` 의 Literal 이 사라지면 `axes.py` 의 `_patch_kernels_hub`·`KERNEL_PATCHERS`
항목·`KERNEL_MODULE_ROOTS["kernels"]` 와 짝 테스트도 같이 지워야 한다 —
`tests/test_axes.py::test_every_kernel_the_schema_offers_routes_to_a_patcher` 가 짝을 강제하므로
한쪽만 지우면 빨개진다. `tests/test_smoke_cpu.py` 와 `tests/test_pods.py` 의 세 자리는
다른 거부 사유로 바꿔야 한다(노트가 줄 번호를 적어두었다).

`docs/support-matrix.md`·`PLAN.md` 쪽 두 줄은 **integrate 레인(wave 3b)** 이 한다. 넘긴다.

## 작업 4 — trackio 제거는 한 커밋이어야 한다 (결정 3)

`config-consumed` 가 2 에서 멈춰 있고 남은 둘이 `run.trackio_project`/`run.trackio_space_id` 다.
report 레인과 measure 레인이 **각자 실측으로** 한쪽만으로는 착지 불가임을 확인하고 되돌렸다
(`.plans/notes/report.md §3`, `.plans/notes/measure.md`).

`configs/run/*.yaml` 4개와 `trainbench/config_schema.py` 를 **함께** 고친다.
`pyproject.toml` 의 `tracking` extra 도 이 레인이 함께 한다 — `uv.lock` 이 따라 움직이므로
`env-locks` 와 `doc-commands` 를 반드시 재실행해서 확인한다. 락이 흔들리면 **되돌리고
그 사실을 보고한다.**

목표: `config-consumed` 0. 그러면 그 항목이 baseline 에서 `newly fixed` 로 BLOCK 되는데
**정상이고 머지 단계가 처리한다. `docs/audit-baseline.json` 을 건드리지 않는다.**

## 작업 5 — measure 의 metrics 가 레코드에 도달하지 못한다

`.plans/notes/measure.md §3` 이 필요한 다섯 줄을 적어두었다. `trainbench/metrics/` 는 준비됐고
`scripts/bench.py` 의 호출부가 바뀌어야 `grad_norm`/`trainable_params`/`total_params`/
`profiled`/`measurement` 블록이 실제 런 레코드에 들어간다.

특히: `metrics.gradient_norm` / `metrics.parameter_counts` 는 **`optimizer.zero_grad` 직전**에
읽는다. 그 뒤에 읽으면 항상 0이고, 그것이 unsloth 3칸이 통과했던 바로 그 모양이다.

이 배선이 없으면 `record-report` 계약이 요구하는 유효성 게이트가 **fixture 위에서만 참**이다.

## 작업 6 — packing 이 올린 계약 개정 요청

`tests/fixtures/microbatch.sample.json` 의 `tensors_may_add` 에 `seq_idx` 가 없어서
packing 레인이 `arch=qwen3_5` 의 conv 쪽 격리를 닫지 못했다(어텐션 쪽은 닫혔다).

**계약을 고치는 것은 이 레인의 권한이지만 근거가 필요하다.** 핀된 소스에서
`seq_idx` 가 실제로 무엇을 하는지 확인하고, 확인되면 `tensors_may_add` 에 넣는다.
확인되지 않으면 **넣지 말고** 그 사실을 `.plans/notes/wire.md` 에 적어 파드 질문으로 넘긴다.
`.plans/research/transformers-varlen-prompt.md` 와 `.plans/notes/packing.md` 를 읽고 판단한다.

## 작업 7 — `native_binding` 은 이제 도달 불가 경로다

`scripts/bench.py::native_binding` 은 `trainbench/loader.py` 가 생긴 지금 native 적재의
**두 번째 정의**다. adapters 레인이 "지울지 남길지는 통합 단계의 판단"이라고 남겼다.

판단해서 실행한다. 남긴다면 **왜 남기는지 주석 한 줄**을 남긴다 — 죽은 코드는 규약 위반이고
(`code-craft.md`), 같은 것의 두 정의는 나중에 갈라진다.

## 작업 8 — axes 가 올린 capture 공백 셋

`.plans/notes/axes.md §2` 가 세 가지를 적었다. 전부 "축은 적용됐는데 되읽기가 못 본다" 모양이다.
`trainbench/applied.py` 를 고쳐 닫는다. 계약이 표현하지 못하는 것이 아니라 capture 구현이
아직 모르는 것이므로 계약 개정은 필요 없다.

`applied.PARALLEL_WRAPPERS` 가 FSDP1 의 `FullyShardedDataParallel` 을 찾는데 torch 2.13.0 의
`fully_shard` 는 제자리에서 감싸므로 `parallel=fsdp2` 는 구현됐어도 측정이 열리지 않는다 —
그 셋 중 하나다.

## 완료 조건

1. tevatron 셀에서 `loss.name`/`parallel.cross_device_negatives` 가 레코드에 `framework_owned`
   로 도달하고 `assert_matches` 가 그 런을 거부하지 않는다 → 테스트로 단언
2. axolotl 의 `required_step_context` 가 `axes.step_context` 로 실제 적용된다 → 테스트로 단언
3. `kernels_hub` 가 스키마·패처·감사 표에서 전부 사라졌다. 짝 테스트도 함께
4. `config-consumed` **0** → `infisical run --env=dev -- uv run python scripts/audit_plan.py`
   (`newly fixed` BLOCK 은 정상이다. baseline 을 건드리지 않는다)
5. `grad_norm`/`trainable_params` 가 실제 런 레코드에 들어간다 (fixture 가 아니라 코드 경로로)
6. `native_binding` 판단이 실행됐다
7. 각 검사를 되돌리면 죽는다. **사보타주 전에 `co_filename`/`co_firstlineno` 확인**
8. 네 게이트. `plan-files` 는 신설 파일이 없으면 초록이어야 한다
9. **확인 안 함** — 이 호스트에 deepspeed·bitsandbytes·TE·DALI·fla 가 없다.
   배선이 실제 객체에서 무엇을 읽는지는 파드가 답한다. 축별로 한 문장씩 적는다

## 하지 않는 것

- `docs/audit-baseline.json` — **머지 단계 전용**
- `docs/support-matrix.md`, `AGENTS.md`, `README.md`, 루트 `PLAN.md` — **integrate**(wave 3b)
- `docs/open-verdicts.json` — **integrate**
- `docs/methodology.md` — kernels 레인이 했다. 사실이 바뀌면 노트로 넘긴다
- 새 기능. 이 레인은 **이미 있는 두 끝을 잇는 것**이지 세 번째 끝을 만드는 것이 아니다
