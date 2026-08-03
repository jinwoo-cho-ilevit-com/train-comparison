# 리뷰 발견 — 2차 라운드 (수정 diff, 2026-08-03)

base `0e8f053a4ef3c9cc29f9158fb39d59420ba24b8e` .. head `0e600e158d7547fdb7518609c1dd58dfa2f2214c` — 1차 수정 패스가 바꾼 24개 파일 +2,047/-229.
1차와 같은 구조(micro 9 + macro 4), 발견마다 별도 에이전트가 실행으로 확정했다.

확정 28 / 반박 2 / 실행불가 0 / minor 24

**이 라운드의 존재 이유가 확인됐다** — 확정 다수가 1차 수정이 새로 넣은 코드에 있다.

## 확정 발견

### `timing-run-goes-offline-before-the-checkpoint-and-corpus-are-fetched` — blocker / correctness

- 단위: bench
- 위치: `scripts/bench.py:680`

**주장**: `build_run` 은 첫 줄에서 `HF_HUB_OFFLINE=1` 을 켠 뒤 체크포인트와 코퍼스를 Hub 에서 읽으므로, 캐시가 없는 timing/quality 파드는 모든 설정에서 결과 파일 없이 exit 1 로 죽는다.

**실패 시나리오**: purpose=timing 설정으로 갓 부팅한 파드(HF_HOME=/workspace/hf 비어 있음, orchestrate 가 볼륨을 붙이지 않고 두 Dockerfile 에 사전 다운로드 없음). close_kernel_fetch_doors 가 os.environ 과 huggingface_hub.constants.HF_HUB_OFFLINE 을 닫는다 -> load_framework 의 AutoModel.from_pretrained 가 OSError("We couldn't connect to 'https://huggingface.co' ... and couldn't find them in the cached files"), 그 뒤 load_pairs 가 ConnectionError("Couldn't reach 'jinwoo-cho/mmeb-subset' on the Hub (OfflineModeIsEnabled)"). 둘 다 refusal_types() 가 아니라 refusing() 과 main 의 except RefusedSetting 을 지나가 --out 이 기록되지 않고, entrypoint.sh 가 'exit 1' fallback 레코드를 올린다. 파드 하나가 통째로 '기동했고 아무것도 안 냈다'가 된다. tests/test_smoke_cpu.py 는 pod_setting 이 from_pretrained 와 load_pairs 를 둘 다 스텁하므로 새 fetch-door 테스트 넷이 이 상태를 볼 수 없다.

**재현**:
```text
HF_HOME=$(mktemp -d) uv run python - <<'PY'
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import huggingface_hub  # 파드 프로세스처럼 문 닫기 전에 import
from hydra import compose, initialize_config_dir
from trainbench.config import to_bench_config
from trainbench.collate import load_pairs
spec = importlib.util.spec_from_file_location('bench_entry', Path('scripts/bench.py'))
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
    cfg = compose(config_name='config', overrides=['device=cpu','model=qwen3_5_0_8b','framework=native','data.limit=4','train.batch_size=4','run.purpose=timing'])
config = to_bench_config(cfg)
bench.close_kernel_fetch_doors(config)   # build_run 의 첫 줄
try:
    load_pairs(config)                   # build_run 의 다음 Hub 읽기
    print('load_pairs OK')
except BaseException as exc:
    print('RAISED', type(exc).__name__, '|', str(exc)[:120], '| refusal type?', isinstance(exc, bench.refusal_types()))
PY
기대 출력: RAISED ConnectionError | Couldn't reach 'jinwoo-cho/mmeb-subset' on the Hub (OfflineModeIsEnabled) | refusal type? False
체크포인트 쪽 절반: HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')"  -> OSError
```

**검증** (reproduced):
```text
HF_HOME=$(mktemp -d) uv run python - <<'PY'  (리뷰어 재현 스크립트, huggingface_hub.constants 를 명시 import 하도록만 수정) / HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')"
---
[코퍼스 절반]
kernel fetch door closed: $HF_HUB_OFFLINE=None, want '1'
kernel fetch door closed: $USE_HUB_KERNELS=None, want 'NO'
kernel fetch door closed: huggingface_hub.constants.HF_HUB_OFFLINE=False, want True — cached at import, so the environment variable was set too late to reach it
env HF_HUB_OFFLINE: 1 | const: True
HF_HOME: /var/folders/7p/.../tmp.Bh1JyhROFN []
RAISED ConnectionError | Couldn't reach 'jinwoo-cho/mmeb-subset' on the Hub (OfflineModeIsEnabled) | refusal type? False

[체크포인트 절반]
RAISED OSError | We couldn't connect to 'https://huggingface.co' to load the files, and couldn't f
```

### `offline-doors-defeat-the-data-pin` — blocker / measurement-validity

- 단위: macro:measurement
- 위치: `scripts/bench.py:680`

**주장**: `close_kernel_fetch_doors` turns on HF offline mode before `load_pairs` runs, and `datasets` answers an offline `load_dataset(..., revision=…)` with whatever build is already in the local cache — measured here, that is the D1-corrupt revision `CORRUPT_DATA_REVISIONS` exists to refuse — while the run record still writes the pinned revision.

**실패 시나리오**: A `purpose=timing` (or `quality`) run composes `data=speed`, i.e. `data.revision=55aafaf9bfe171c65a8131224d3791df379f1651`. `build_run` calls `close_kernel_fetch_doors(config)` at `scripts/bench.py:680`, which calls `kernels.forbid_runtime_kernel_fetch()` and sets both `HF_HUB_OFFLINE=1` and `huggingface_hub.constants.HF_HUB_OFFLINE=True` (`trainbench/kernels.py:516-538`). 23 lines later `dataset = load_pairs(config)` (`scripts/bench.py:703`) calls `load_dataset(repo_id, revision=…, streaming=True)` (`trainbench/collate.py:136-141`), which needs the Hub. Two outcomes, both wrong, and neither raises inside a `refusing()` block:
(a) cache non-empty — measured on this host: datasets 5.0.1 prints one log line `Using the latest cached version of the dataset since jinwoo-cho/mmeb-subset couldn't be found on the Hugging Face Hub (offline mode is enabled). Found the latest cached dataset configuration 'default' at .../b750b9c3263e9ef5dce225fd50aa25d7c58f1d5f` and returns rows. `b750b9c3…` is the exact sha in `trainbench/config_schema.py:38-43` (`defect D1: pos_image was dropped, so 466 rows share one placeholder positive and 644 rows lost their query image`) that `_no_run_reads_a_corrupt_subset` refuses for every purpose. The refusal is bypassed because the *config* names the good revision; only the *load* is substituted. `build_record` then writes `config.data.revision=55aafaf9…` into the result, so the published tokens/s claims a corpus it was not measured on and nothing in the record or in `scripts/report.py` can tell.
(b) cache empty (a fresh pod — nothing in `docker/entrypoint.sh` or `scripts/verify_env.py` pre-fetches the subset, grepped) — `load_pairs` raises `OfflineModeIsEnabled`. It sits between the `refusing("binding")` and `refusing("assemble")` blocks, so it is not a `RefusedSetting`, `main`'s `except RefusedSetting` at `scripts/bench.py:993` does not catch it, no result file is written, and `docker/entrypoint.sh` publishes `no result file after the run (exit 1)` — the setting is filed as a dead pod rather than as a missing corpus.
Probe runs are unaffected (`ENFORCED_PURPOSES = ("timing", "quality")`), so the only purposes this breaks are the two whose numbers get published.

**재현**:
```text
Ordering: `sed -n '676,706p' scripts/bench.py` (doors at :680, `load_pairs` at :703).
Substitution (run from the repo root; prints the cached-revision fallback on any host that has ever built this dataset, and `OfflineModeIsEnabled` on one that has not — both are the finding):
.venv/bin/python - <<'PY'
import os
os.environ.pop("HF_HUB_OFFLINE", None)
import transformers, huggingface_hub          # what scripts/bench.py already imports
from trainbench import kernels
kernels.forbid_runtime_kernel_fetch()         # exactly close_kernel_fetch_doors()
from datasets import load_dataset             # exactly trainbench/collate.py::load_pairs
try:
    ds = load_dataset("jinwoo-cho/mmeb-subset",
                      revision="55aafaf9bfe171c65a8131224d3791df379f1651",
                      split="train", streaming=True)
    print("rows:", len(list(ds.take(4))))
except Exception as e:
    print(type(e).__name__, str(e)[:300])
PY
Observed 2026-08-03 on this checkout: `Using the latest cached version ... at /Users/jwcho/.cache/huggingface/datasets/jinwoo-cho___mmeb-subset/default/0.0.0/b750b9c3263e9ef5dce225fd50aa25d7c58f1d5f` then `rows: 4`.
Cross-check that the substituted sha is the forbidden one: `sed -n '38,43p' trainbench/config_schema.py` and `ls /Users/jwcho/.cache/huggingface/datasets/jinwoo-cho___mmeb-subset/default/0.0.0/`.
Mutation that shows the guard is what does it: comment out `close_kernel_fetch_doors(config)` at `scripts/bench.py:680` and re-run the snippet without `forbid_runtime_kernel_fetch()` — the pinned revision resolves.
```

**검증** (reproduced):
```text
.venv/bin/python - <<'PY'
import os
os.environ.pop("HF_HUB_OFFLINE", None)
import transformers, huggingface_hub
from trainbench import kernels
kernels.forbid_runtime_kernel_fetch()   # exactly close_kernel_fetch_doors()
print("HF_HUB_OFFLINE env:", os.environ.get("HF_HUB_OFFLINE"), "const:", huggingface_hub.constants.HF_HUB_OFFLINE)
from datasets import load_dataset       # exactly trainbench/collate.py::load_pairs
ds = load_dataset("jinwoo-cho/mmeb-subset",
                  revision="55aafaf9bfe171c65a8131224d3791df379f1651",
                  split="train", streaming=True)
print("rows:", len(list(ds.take(4))))
PY
---
Using the latest cached version of the dataset since jinwoo-cho/mmeb-subset couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /Users/jwcho/.cache/huggingface/datasets/jinwoo-cho___mmeb-subset/default/0.0.0/b750b9c3263e9ef5dce225fd50aa25d7c58f1d5f (last modified on Sat Aug  1 09:57:02 2026).
HF_HUB_OFFLINE env: 1 const: True
rows: 4

# 반환된 행이 실제 D1 손상 빌드임 (pos_image 없음):
keys: ['mmeb_config', 'pos_text', 'qry', 'qry_image']

# 순서 (HEAD 0e600e15):
scripts/bench.py:680:    close_kernel_fetch_doors(config)
scripts/bench.
```

### `run-record-sample-config-rejected-by-its-own-schema` — blocker / contract-split

- 단위: macro:contracts
- 위치: `tests/fixtures/run_record.sample.json:308`

**주장**: record-report 경계의 동결 페이로드가 자기 생산자의 스키마에 거부된다 — `config.run.trackio_project`/`trackio_space_id` 는 결정 3 으로 `RunConfig` 에서 제거됐고 `BenchConfig` 는 `extra="forbid"` 다.

**실패 시나리오**: 입력: `tests/fixtures/run_record.sample.json` 의 `config` 블록. 상태: 현재 트리(스키마·`configs/run/*.yaml` 양쪽에서 trackio 제거 완료, `audit_plan.py` 의 `config-consumed` PASS). 출력: `BenchConfig.model_validate(sample['config'])` 가 `ValidationError: 2 validation errors ... run.trackio_project Extra inputs are not permitted / run.trackio_space_id Extra inputs are not permitted` 로 죽는다. 즉 이 경계가 두 레인(measure/report)을 검증하는 기준 레코드는 어떤 런도 만들 수 없는 레코드다 — 계약 파일의 docstring 은 그 config 블록이 "composed by Hydra from `configs/`" 라고 적고 있다. `test_the_stored_sample_is_the_shape_the_producer_writes` 는 `set(produced) - set(payload)` 최상위 키만 보고 `config` 안쪽은 `CONSUMED_CONFIG_FIELDS` 7개의 존재만 보므로 이 드리프트를 원리적으로 볼 수 없고, `pytest tests/contract -q` 는 122 passed 로 초록이다. HAZARDS §4.3 이 세 커밋으로 수습한 "한쪽이 만들 수 없는 샘플" 과 같은 모양이다.

**재현**:
```text
infisical run --env=dev -- uv run python -c "import json;from pathlib import Path;from trainbench.config_schema import BenchConfig;BenchConfig.model_validate(json.loads(Path('tests/fixtures/run_record.sample.json').read_text())['config'])"  # ValidationError 2건
# 그 다음, 계약은 여전히 초록인 것을 확인:
infisical run --env=dev -- uv run pytest tests/contract -q   # 122 passed
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python -c "import json;from pathlib import Path;from trainbench.config_schema import BenchConfig;BenchConfig.model_validate(json.loads(Path('tests/fixtures/run_record.sample.json').read_text())['config'])" ; infisical run --env=dev -- uv run pytest tests/contract -q
---
pydantic_core._pydantic_core.ValidationError: 2 validation errors for BenchConfig
run.trackio_project
  Extra inputs are not permitted [type=extra_forbidden, input_value='train-comparison', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/extra_forbidden
run.trackio_space_id
  Extra inputs are not permitted [type=extra_forbidden, input_value=None, input_type=NoneType]
    For further information visit https://errors.pydantic.dev/2.13/v/extra_forbidden

(module actually loaded: /Users/jwcho/Codes/train-comparison/trainbench/config_schema.py)

그리고 계약 스위트는 여전히 
```

### `read-fingerprint-cannot-produce-two-of-three-frozen-samples` — blocker / correctness

- 단위: kernels
- 위치: `trainbench/kernels.py:433`

**주장**: `read_fingerprint` 는 프로덕션이 실제로 넘기는 문자열 요청에서 백본별로 다른 구현이 바인딩되면 무조건 `UnidentifiedKernel` 로 거부하므로, 경계가 동결한 세 샘플 중 둘(`fa2_hub_fallback_qwen3_vl`, `fa3_hub_kernel_mask_unregistered_qwen3_vl`)을 이 런타임은 만들어낼 수 없다.

**실패 시나리오**: `config.attn.impl` 은 `ATTN_IMPL` 표에서 항상 **평문 문자열**만 낸다(`trainbench/config_schema.py:162-176`) — dict 를 넘길 경로가 스키마에 없다. 문자열이면 `_requested_by_backbone` 이 모든 백본을 `landed` 로 돌려주고, `read_fingerprint:433` 의 `_one([backbones[name] for name in sorted(landed)], ...)` 가 서로 다른 값 두 개를 보고 거부한다. 그런데 동결된 지문 `fa2_hub_fallback_qwen3_vl` 자체가 `requested.attn_implementation == "flash_attention_2"`(문자열) 이면서 `backbones = {text_config: "kernels-community/flash-attn2", vision_config: "sdpa"}` 다. 즉 경계가 "정상 payload"로 못박은 상태를 런타임은 거부한다. 결과: `attn=fa2/fa3/fa4` x Qwen3-VL 셀은 `loader.build_fingerprint` -> `describe` -> `load` 경로에서 예외가 나고, `scripts/bench.py:689` 의 `refusing("load_kwargs")` 가 그것을 `RefusedSetting` 으로 바꿔 **수치 없는 거부 레코드**를 낸다. `docs/methodology.md:466-469` 는 같은 커밋에서 "세 모델이 전부 멀티모달이므로 텍스트 타워만 바뀐 빌드가 정상 결과"라고 적고 있어 코드와 문서가 정면으로 어긋난다. 게다가 완료 조건 3(미등록 커널 + packing 거부)이 유일하게 기대는 `fa3_hub_kernel_mask_unregistered_qwen3_vl` 도 같은 이유로 런타임이 낼 수 없는 payload여서, 그 거부는 저장된 fixture 에 대해서만 도는 검사다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && uv run python - <<'PY'
import json
from transformers import Qwen3VLConfig
from trainbench import kernels
from trainbench.config_schema import ATTN_IMPL
class M:
    def __init__(self, c): self.config = c
s = json.load(open("tests/fixtures/kernel_fingerprint.sample.json"))["samples"]["fa2_hub_fallback_qwen3_vl"]
c = Qwen3VLConfig(); c._attn_implementation = "sdpa"
c.text_config._attn_implementation_internal = s["backbones"]["text_config"]["attn_implementation"]
c.vision_config._attn_implementation_internal = s["backbones"]["vision_config"]["attn_implementation"]
print(kernels.read_fingerprint(M(c), axis="attn.name", value="fa2", requested=ATTN_IMPL["fa2"], revision_resolver=lambda r: "0"*40))
PY
# 관측: UnidentifiedKernel: the backbones the request reached do not agree on the implementation that bound: ['kernels-community/flash-attn2', 'sdpa']
# 같은 분기가 transformers 자신의 백본별 조정 경로에서도 열린다: modeling_utils.py:2239 `submodule.get_correct_attn_implementation(...)`, :2088-2093 sdpa->eager 폴백.
```

**검증** (reproduced):
```text
cd /Users/jwcho/Codes/train-comparison && uv run python - <<'PY'
import json
from transformers import Qwen3VLConfig
from trainbench import kernels
from trainbench.config_schema import ATTN_IMPL
class M:
    def __init__(self, c): self.config = c
s = json.load(open("tests/fixtures/kernel_fingerprint.sample.json"))["samples"]["fa2_hub_fallback_qwen3_vl"]
c = Qwen3VLConfig(); c._attn_implementation = "sdpa"
c.text_config._attn_implementation_internal = s["backbones"]["text_config"]["attn_implementation"]
c.vision_config._attn_implementation_internal = s["backbones"]["vision_config"]["attn_implementation"]
print(kernels.read_fingerprint(M(c), axis="attn.name", value="fa2", requested=ATTN_IMPL["fa2"], revision_resolver=lambda r: "0"*40))
PY
---
kernels file: /Users/jwcho/Codes/train-comparison/trainbench/kernels.py 410
Traceback (most recent call last):
  File "<stdin>", line 12, in <module>
  File "/Users/jwcho/Codes/train-comparison/trainbench/kernels.py", line 433, in read_fingerprint
    resolved_impl = _one(
        [backbones[name] for name in sorted(landed)], "the implementation that bound"
    )
  File "/Users/jwcho/Codes/train-comparison/trainbench/kernels.py", line 402, in _one
    raise UnidentifiedKernel(
    ...<3 lines>...
    )
trainbench.kernels.UnidentifiedKernel: the backbones the request reached do not agree on the
```

### `st-harness-step-cannot-drive-sentencetransformer-forward` — blocker / correctness

- 단위: loader-probe
- 위치: `trainbench/loader.py:471`

**주장**: `sentence_transformers` 어댑터가 `step=HARNESS_STEP` 을 선언하지만 핀된 5.6.1 의 `SentenceTransformer.forward(self, input: dict, **kwargs)` 는 위치 인자 하나를 요구하고 dict 를 돌려주므로, 하네스 루프의 `model(**batch, ...)` 는 스텝 0 에서 TypeError 로 죽는다 — 이번 커밋이 collate 앞단만 고치고 스텝은 그대로 두었다.

**실패 시나리오**: `framework=sentence_transformers purpose=timing` 파드 런. `loader.load` 가 (이제 정상적인) HF 프로세서를 돌려주고, `bench.refuse_a_step_this_harness_cannot_drive` 는 통과시킨다(owner 가 `harness` 로 일치하고 `batch_keys` 가 `('input_ids','attention_mask')` 로 collate 가 만드는 것의 부분집합이다). 체크포인트가 적재되고 타이머가 열린 뒤 `scripts/bench.py:117`/`trainbench/probe/steps.py:253` 의 `model(**batch, output_hidden_states=False)` 가 `TypeError: BaseModel.forward() missing 1 required positional argument: 'input'` 로 죽는다. TypeError 는 `refusal_types()` 에 없으므로 `refusing()` 이 잡지 않고 `--out` 이 안 써지며, `docker/entrypoint.sh` 가 `no result file after the run` 으로 fallback 을 올려 ST 칸이 '거부된 설정'이 아니라 '아무도 시도하지 않은 조합'으로 렌더된다. 위치 인자를 맞춰 `model(batch)` 로 불러도 반환은 `{'sentence_embedding': ...}` dict 라 `steps.encode` 의 `last_hidden_state`/`hidden_states`/`output[0]` 세 갈래가 모두 KeyError 로 끝난다. tevatron 은 정확히 같은 이유로 `step.owner=FRAMEWORK` 를 받았는데(`trainbench/loader.py:488-500`) ST 만 하네스 스텝으로 선언돼 있고, 같은 파일 `:373-375` 의 `aligns_padding_side=False` 주석은 이미 'ST 는 자기 모듈 안에서 pooling 한다'고 적어 자기모순이다. `.plans/notes/probe.md:86` 이 tevatron forward 벽만 등록했을 뿐 ST 는 어디에도 확인 안 함으로 선언돼 있지 않고, 핀된 휠로 이 호스트에서 답이 나온다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && curl -sLo /tmp/st.whl https://files.pythonhosted.org/packages/c1/ad/8f73f512dc7ad4031d2b64cbb67f70bdfb355756afbe0db610a5146415c1/sentence_transformers-5.6.1-py3-none-any.whl && cat > /tmp/repro_st.py <<'EOF'
import sys, zipfile, torch
sys.path.insert(0, "/Users/jwcho/Codes/train-comparison")
from trainbench.probe import steps
src = zipfile.ZipFile("/tmp/st.whl").read("sentence_transformers/base/model.py").decode().split("\n")
print("pinned:", src[495].strip())
class STLike(torch.nn.Module):
    def forward(self, input: dict, **kwargs) -> dict:
        return {"sentence_embedding": torch.zeros(input["input_ids"].shape[0], 4)}
batch = {"input_ids": torch.ones(4, 8, dtype=torch.long), "attention_mask": torch.ones(4, 8, dtype=torch.long)}
steps.encode(STLike(), batch, "left")
EOF
infisical run --env=dev --path=/ -- uv run python /tmp/repro_st.py
# 기대: pinned: def forward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]
#       TypeError: STLike.forward() missing 1 required positional argument: 'input'
# 대조: rg -n 'SentenceTransformer' /Users/jwcho/Codes/train-comparison/trainbench/loader.py 로 step=HARNESS_STEP 확인
```

**검증** (reproduced):
```text
cd /Users/jwcho/Codes/train-comparison && curl -sLo /tmp/st.whl https://files.pythonhosted.org/packages/c1/ad/8f73f512dc7ad4031d2b64cbb67f70bdfb355756afbe0db610a5146415c1/sentence_transformers-5.6.1-py3-none-any.whl && uv run python /private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/repro_st.py
---
live steps.encode: /Users/jwcho/Codes/train-comparison/trainbench/probe/steps.py 244
ST adapter step: harness None ('input_ids', 'attention_mask')
pinned: def forward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
TypeError: STLike.forward() missing 1 required positional argument: 'input'
return type: dict keys: ['sentence_embedding']
KeyError: 0
```

### `notes-order-deleting-a-deliberately-kept-kernel-module-root` — major / correctness

- 단위: axes
- 위치: `.plans/notes/axes.md:26`

**주장**: 머지 단계에 넘기는 노트가 `KERNEL_MODULE_ROOTS["kernels"]` 를 지우라고 지시하는데, `trainbench/axes.py:95-100` 이 그 행을 의도적으로 남긴다고 정반대로 적는다. 노트가 지시하는 나머지 셋(`_patch_kernels_hub`, `KERNEL_PATCHERS` 항목, `test_kernels_hub_from_the_schema_is_still_refused_with_both_reasons`)은 이미 존재하지 않는다.

**실패 시나리오**: 머지 레인이 노트의 표를 그대로 적용해 `KERNEL_MODULE_ROOTS` 에서 `"kernels": "kernels_hub"` 를 지운다. 그러면 hub dispatch 를 켠 어댑터가 만든 모델(모듈 클래스가 `kernels.*` 에서 정의됨)의 `_module_roots` 에 매핑 항목이 없어 `_capture_kernel` 이 `found == []` → `"none"` 을 돌려주고, `kernel=none` 요청과 **일치**한다. axes.py:97-100 이 그 결과를 정확히 예언한다: "dropping the row would read it back as `none` and match a `kernel=none` request." 즉 kernel 축이 조용히 오라벨된 숫자를 낸다.

**재현**:
```text
`sed -n 8,35p .plans/notes/axes.md` 와 `sed -n 93,106p trainbench/axes.py` 를 나란히 읽는다. 노트가 이미 없다고 가정한 것들을 확인: `grep -rn 'kernels_hub' trainbench/ scripts/ configs/ docs/CONTRACTS.md` (결과: `axes.py:95,104` 뿐), `grep -n 'test_kernels_hub' tests/test_axes.py` (실제 이름은 `test_kernels_hub_is_dropped_from_every_place_that_could_still_offer_it`). 변이: `trainbench/axes.py:104` 의 `"kernels": "kernels_hub",` 를 지우고 `uv run pytest tests/test_axes.py tests/test_applied.py -q` — 잡히는지 본다.
```

**검증** (reproduced):
```text
python3 -c '...delete `    "kernels": "kernels_hub",` from trainbench/axes.py...' && uv run pytest tests/test_zz_repro.py -q -s   # 임시 재현 테스트: kernels.* 에서 정의된 모듈로 만든 모델에 대해 applied._capture_kernel 을 kernel=none 요청과 대조. 이후 `uv run pytest tests/test_axes.py tests/test_applied.py -q` 로 변이 검출 여부 확인, `git checkout -- trainbench/axes.py && rm -f tests/test_zz_repro.py` 로 복구
---
변이 전(HEAD 그대로):
KERNEL_MODULE_ROOTS = {'liger_kernel': 'liger', 'fla': 'fla', 'kernels': 'kernels_hub'}
applied kernel = kernels_hub
detail = {'modules_checked': 3, 'packages': ['kernels'], 'kernel_modules': {'kernels_hub': 3}, 'superseded': 0}
requested kernel = none
MATCHES REQUEST: False

변이 후(노트 지시대로 KERNEL_MODULE_ROOTS["kernels"] 삭제):
KERNEL_MODULE_ROOTS = {'liger_kernel': 'liger', 'fla': 'fla'}
applied kernel = none
detail = {'modules_checked': 3, 'packages': ['kernels'], 'kernel_modules': {}, 'superseded': 0}
requested kernel = none
MATCHES REQUEST: True

변이 상태에서 게이트: uv run pytest test
```

### `single-cross-device-config-withholds-a-variant-on-a-dead-blocker` — major / measurement-validity

- 단위: axes
- 위치: `configs/parallel/single_cross_device.yaml:3`

**주장**: `ddp_cross_device` 변형이 없는 이유로 "`axes.assemble` still refuses every strategy but `single`" 와 "DDP is not implemented" 를 적는데 둘 다 거짓이라, `parallel.cross_device_negatives=true` 는 파일 자신이 "does not train correctly" 라고 적은 배치에서만 도달 가능한 상태로 고정된다.

**실패 시나리오**: ablation 을 돌리면 `cross_device_negatives=true` 셀은 `strategy=single` 하나뿐이다. 그 셀은 랭크마다 동기화 안 된 복제본을 돌리므로 이 파일 스스로 "speed setting, not a quality one" 이라고 적는다. 즉 이 축의 유일한 on-값이 gradient all-reduce 없는 배치이고, 리포트를 읽는 사람은 `parallel.cross_device_negatives` 행을 "cross-device negatives 의 비용" 으로 읽지만 실제로 잰 것은 "훈련이 틀린 상태에서의 비용" 이다. `ddp` 가 이미 구현됐으므로 `strategy: ddp` + `cross_device_negatives: true` 파일 하나가 그 셀을 열지만, 이 주석이 그것을 불가능하다고 적어 아무도 열지 않는다.

**재현**:
```text
1) `cat configs/parallel/single_cross_device.yaml` 로 3-13행 주장을 읽는다. 2) `uv run pytest tests/test_axes.py -k "ddp" -q` 가 통과하는 것으로 `assemble` 이 ddp 를 거부하지 않음을 확인한다. 3) 반증 실행: `printf 'strategy: ddp\ncross_device_negatives: true\n' > configs/parallel/ddp_cross_device.yaml && infisical run --env=dev -- uv run python scripts/env_report.py device=cpu model=qwen3_5_0_8b framework=native parallel=ddp_cross_device data.limit=4 train.batch_size=4` — 스키마/합성이 이 조합을 받는지 본다. 확인 후 파일을 지운다(`plan-files` 가 미선언 파일에 빨개진다).
```

**검증** (reproduced):
```text
printf 'strategy: ddp\ncross_device_negatives: true\n' > configs/parallel/ddp_cross_device.yaml && infisical run --env=dev -- uv run python scripts/env_report.py device=cpu model=qwen3_5_0_8b framework=native parallel=ddp_cross_device data.limit=4 train.batch_size=4 ; uv run pytest tests/test_axes.py -k "ddp" -q
---
$ uv run pytest tests/test_axes.py -k "ddp" -q
....                                                                     [100%]
4 passed, 230 deselected in 1.00s

$ infisical run --env=dev -- uv run python scripts/env_report.py device=cpu model=qwen3_5_0_8b framework=native parallel=ddp_cross_device data.limit=4 train.batch_size=4
2026-08-03T10:43:27+09:00 INF Injecting 27 Infisical secrets into your application process
               environment
┌───────────────┬────────────────────────┐
│ model         │ qwen3_5_0_8b (qwen3_5) │
│ framework     │ native                 │
│ purpose       │ tim
```

### `baseline-note-files-dali-as-package-absence` — major / correctness

- 단위: audit
- 위치: `docs/audit-baseline.json:4`

**주장**: 이번에 다시 쓴 note 가 `dali` 를 '패키지가 없는 축 값' 목록에 넣었지만 `axes._dataloader` 는 패키지를 한 번도 묻지 않고 `backend != "torch"` 만으로 무조건 거부한다 — 이 커밋이 `kernel=fla` 에 대해 방금 걷어낸 바로 그 오귀속이 같은 문장 안에 남아 있다.

**실패 시나리오**: note 는 '패키지가 없는 축 값(adamw_8bit / qlora / liger / dali / mxfp8 / nvfp4)은 감사가 이 호스트에서 원리적으로 못 본다' 고 적는다. 이것을 읽은 다음 레인이 DALI 가 들어간 native 이미지에서 감사를 돌리면 `dataloader` 축이 움직일 것으로 기대하지만, `trainbench/axes.py:1538` 이 `config.dataloader.backend != "torch"` 하나만 보고 `UnappliedAxis` 를 던지므로 값은 그 이미지에서도 정확히 같게 거부된다(`inspect.getsource(axes._dataloader)` 에 `find_spec`/`import_module` 이 없음 — 실측 False). 즉 `dali` 는 환경 부재가 아니라 코드 부재이고, docstring 자신도 `axes.py:1529` 에서 'the one axis value in this module still refused for missing code' 라고 적는다. note 가 이 셋을 뭉개지 말라고 스스로 선언한 문서(PLAN 완료 조건, HAZARDS §3 의 'note 가 blocker 를 가림')인데, 원인을 '이미지가 없어서' 쪽으로 옮겨 적으면 코드 공백이 파드 대기 항목으로 오독되어 아무도 열지 않는다. `kernel=fla` 에 대해 note 가 직접 쓴 문장 '아키텍처로 거부됐다 — fla 와 CUDA 가 있는 이미지에서도 셀 수 없는 값이었다' 가 `dali` 에 글자 그대로 적용된다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && infisical run --env=dev -- uv run python - <<'EOF'
import sys, inspect; sys.path[:0]=[".","scripts"]
import audit_plan as ap
from hydra import compose, initialize_config_dir
from trainbench.compose import resolve
from trainbench import axes
with initialize_config_dir(config_dir=str(ap.CONFIGS), version_base=None):
    c = resolve(compose(config_name="config", overrides=["dataloader=dali", "data.num_workers=0"]))[0]
try:
    axes._dataloader(ap._AxisValueRowsWithImages(), c)
except Exception as e:
    print(type(e).__name__, e)
src = inspect.getsource(axes._dataloader)
print("asks about the package:", ("find_spec" in src) or ("import_module" in src))
EOF
# -> UnappliedAxis ... / asks about the package: False
sed -n '1524,1546p' trainbench/axes.py   # docstring: "the one axis value in this module still refused for missing code"
```

**검증** (reproduced):
```text
cd /Users/jwcho/Codes/train-comparison && infisical run --env=dev -- uv run python - <<'EOF'
import sys, inspect; sys.path[:0]=[".","scripts"]
import audit_plan as ap
from hydra import compose, initialize_config_dir
from trainbench.compose import resolve
from trainbench import axes
print("live def:", axes._dataloader.__code__.co_filename, axes._dataloader.__code__.co_firstlineno)
with initialize_config_dir(config_dir=str(ap.CONFIGS), version_base=None):
    c = resolve(compose(config_name="config", overrides=["dataloader=dali", "data.num_workers=0"]))[0]
try:
    axes._dataloader(None, c)
except Exception as e:
    print(type(e).__name__, e)
src = inspect.getsource(axes._dataloader)
print("asks about the package:", ("find_spec" in src) or ("import_module" in src) or ("_import_or_refuse" in src))
EOF
---
live def: /Users/jwcho/Codes/train-comparison/trainbench/axes.py 1524
UnappliedAxis dataloader.backend=dali replaces the DataLoader with its own iterator, and the iterator needs a DALI pipeline over these rows that nothing here builds. Two things have to be settled first, and both need the package: how an external_source pipeline reads a datasets row, and what applied._capture_dataloader_packing reads for dali_packed — a DALI iterator has no collate_fn, which is the only place packing is visible today.
asks about the package: False
```

### `methodology-s9-liger-refusal-table-false-against-tree` — major / measurement-validity

- 단위: kernels
- 위치: `docs/methodology.md:473`

**주장**: §9 가 "liger 커버리지 질문은 오늘 아무 런에서도 발생하지 않는다"는 결론을 세 개의 거부 위에 세우는데, 그중 둘은 같은 커밋의 `trainbench/axes.py` 에 대해 거짓이다.

**실패 시나리오**: 473행은 gemma4 가 `LIGER_UNSUPPORTED` 때문에 거부된다고 적지만 `trainbench/axes.py` 에 `LIGER_UNSUPPORTED` 라는 이름은 정의로 존재하지 않는다 — axes.py:321-327 이 "그 표는 핀된 wheel 에 대해 거짓이었고 삭제했다"고 스스로 적는다. 474행은 qwen3_vl 이 "기록된 엔트리포인트 없음(`LIGER_ENTRYPOINTS`)"으로 거부된다고 적지만 `LIGER_ENTRYPOINTS` 는 axes.py:125-129 에서 `qwen3_vl` 과 `gemma4` 를 둘 다 담고 있고, `tests/test_axes.py:657` 이 그 거부를 시험하려고 `monkeypatch.delitem(axes.LIGER_ENTRYPOINTS, "qwen3_vl")` 을 해야 할 만큼 확실히 들어 있다. 두 행이 사라지면 477행의 결론("셋 다 커버리지와 무관한 이유이므로 liger 커버리지 질문은 발생하지 않는다")이 무너지고, 그 결론이 지탱하던 §9 의 "`kernel_modules` 에 임계값이 없는 것이 지금으로서는 옳다 / 측정 안 함"이 근거를 잃는다. 즉 오늘 `kernel=liger` x `qwen3_vl` 런은 실제로 돌 수 있고, 그 런의 `applied="liger"` 판정은 모듈 1개짜리 커버리지에도 참이 되는 미측정 임계값 위에 놓인다. 표는 33b1dd6 시점에는 참이었고(그때 LIGER_ENTRYPOINTS 는 qwen3_5 하나, LIGER_UNSUPPORTED 는 존재) 이후 axes 가 바뀌면서 낡았는데, §9 는 이 레인의 브리프가 명시적으로 고치라고 지정한 절이다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && grep -n 'LIGER_ENTRYPOINTS = ' -A 5 trainbench/axes.py && grep -c 'LIGER_UNSUPPORTED = ' trainbench/axes.py && sed -n '470,479p' docs/methodology.md && grep -n 'LIGER_ENTRYPOINTS' tests/test_axes.py
# 관측: axes.py:125-129 에 qwen3_vl/gemma4 둘 다 존재, `LIGER_UNSUPPORTED = ` 정의는 0건, methodology 473/474 는 둘 다 거부라고 주장.
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run pytest tests/test_zz_verify_tmp.py -q -s   # 임시 테스트: qwen3_vl/gemma4 에 kernel=liger 를 걸고 axes.patch() 실행 (실행 후 파일 삭제)
---
$ grep -c 'LIGER_UNSUPPORTED = ' trainbench/axes.py
0

$ grep -n 'LIGER_ENTRYPOINTS' trainbench/axes.py
125:LIGER_ENTRYPOINTS = {
337:    entrypoint = LIGER_ENTRYPOINTS.get(arch)
341:            f"{sorted(LIGER_ENTRYPOINTS)}. Nothing recorded is not the same as supported — "

$ sed -n '124,130p' trainbench/axes.py
LIGER_ENTRYPOINT_PREFIX = "apply_liger_kernel_to_"
LIGER_ENTRYPOINTS = {
    "qwen3_5": f"{LIGER_ENTRYPOINT_PREFIX}qwen3_5",
    "qwen3_vl": f"{LIGER_ENTRYPOINT_PREFIX}qwen3_vl",
    "gemma4": f"{LIGER_ENTRYPOINT_PREFIX}gemma4",
}

$ sed -n '471,477p' docs/methodology.md
| 아키텍처 | `ke
```

### `methodology-condemns-packing-the-collate-already-wired` — major / correctness

- 단위: macro:measurement
- 위치: `docs/methodology.md:537`

**주장**: `docs/methodology.md §10.1` states that `collate.PackedBatches.__call__` strips the pack boundaries before the model and that "오늘 Qwen3.5 + packing 런은 linear 레이어에서 격리 없이 돈다", but `trainbench/collate.py:453-454` puts them back as `cu_seq_lens_*` and `seq_idx`, so the shipped methodology condemns numbers the code makes valid.

**실패 시나리오**: `trainbench/collate.py:441-464` pops `cu_seqlens`/`seq_lengths` at :451 and immediately re-emits them at :453-454 as `varlen_kwargs(...)` (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`) and `seq_idx_kwargs(...)` (`seq_idx`) — the two names `docs/methodology.md:532-536` itself identifies as what the Gated DeltaNet chunked kernel and the causal conv read. `scripts/bench.py::pooled_embeddings` (:117) forwards the whole dict as `model(**tensors, ...)`, so both reach the model. The doc's premise is therefore half-true (it pops, then re-adds) and its conclusion is false. Consequence: a reader following §10.1 discards every valid `qwen3_5_0_8b` × `dataloader.packing=true` measurement as "다른 계산의 수치", and §10.3 ("10.1의 linear 레이어 구멍이 열려 있는 한 Qwen3.5에서 '같은 위치'는 '같은 임베딩'이 아니다") propagates the same withdrawn verdict; conversely the §10.1 table row `qwen3_5 | full + linear_attention | full만 자동. **linear은 아니다**` now tells a lane that the wiring is still open work and invites a second implementation of it. This is not an honestly-reported gap: `.plans/notes/seqidx.md:12-24` reported it correctly as `"§10.1 이 과거형이 됐다 — integrate 레인 몫"`, wave 3 merged (0e600e1) without applying it, and what ships is an active false assertion about measurement validity, not a `측정 안 함`.

**재현**:
```text
sed -n '526,541p' docs/methodology.md   # the table row and the '격리 없이 돈다' sentence
sed -n '450,455p' trainbench/collate.py  # pop at :451, varlen_kwargs at :453, seq_idx_kwargs at :454
sed -n '113,123p' scripts/bench.py       # model(**tensors, ...) forwards both
infisical run --env=dev -- uv run pytest tests/test_collate.py -k 'seq_idx' -q
# tests/test_collate.py::test_only_qwen3_5_reads_seq_idx_among_the_arches_this_study_measures passes against the pinned wheel, i.e. the name the doc says never arrives does arrive and is read.
# Handoff that was not applied: sed -n '12,25p' .plans/notes/seqidx.md
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python -c "import sys; sys.path.insert(0,'tests'); from test_collate import build_collate, _Processor, _config, ROWS; p=build_collate(_Processor(), _config(packing=True))(ROWS); print(sorted(p.tensors)); print(p.tensors['seq_idx'].tolist()); print(p.tensors['cu_seq_lens_q'].tolist())"  # plus: infisical run --env=dev -- uv run pytest tests/test_collate.py -k seq_idx -q
---
KEYS REACHING model(**tensors): ['cu_seq_lens_k', 'cu_seq_lens_q', 'input_ids', 'max_length_k', 'max_length_q', 'position_ids', 'seq_idx']
seq_idx: [[0, 0, 0, 1, 1, 2, 3, 3, 3, 3]]
cu_seq_lens_q: [0, 3, 5, 6, 10]
PackedBatches.__call__ -> /Users/jwcho/Codes/train-comparison/trainbench/collate.py 441

pytest tests/test_collate.py -k 'seq_idx' -q -> 7 passed, 12 deselected in 1.47s
```

### `companions-table-has-no-gate` — major / emptiness

- 단위: audit
- 위치: `scripts/audit_plan.py:1609`

**주장**: 새로 추가된 `AXIS_VALUE_COMPANIONS["kernel/fla"]` 는 지우거나 키가 존재하지 않는 variant 를 가리키게 되어도 감사 출력이 바이트 단위로 같고 테스트도 전부 통과한다 — 이 커밋이 한 교정을 되돌릴 때 아무 신호가 없다.

**실패 시나리오**: 누가 `configs/kernel/fla.yaml` 을 다른 이름으로 바꾸거나 머지 충돌에서 이 다섯 줄이 떨어져 나가면, `kernel/fla` 는 다시 기본 모델 `arch=qwen3_vl` 로 합성되어 `_patch_fla` 의 아키텍처 거부(`kernel=fla on arch=qwen3_vl: ... only ['qwen3_5'] import fla`)에 걸린다. 즉 fla 와 CUDA 가 있는 파드 이미지에서도 셀 수 없는 상태로 되돌아간다. 그런데 `axis-values` 는 그 상태에서도 정확히 지금과 같은 문자열 `36/52 applicable on both text-only/image-carrying data; 3 group(s) offering one usable value: kernel 1/3, precision 1/3, train.offload 1/4` 와 `count=3` 을 내므로 `0 grew, 0 shrank` 이고, `tests/test_audit.py` 111개도 그대로 통과한다(`AXIS_VALUE_COMPANIONS` 를 참조하는 테스트가 0개). 실측으로 확인한 두 변이: (A) 항목을 pop -> count=3, detail 동일. (B) 존재하지 않는 키 `"kernel/does_not_exist": ("model=nope",)` 추가 -> 예외도 경고도 없이 count=3. HAZARDS §3 이 아홉 번 겪은 '통과하면서 아무것도 보지 않는' 모양 그대로다. 고치려면 키 집합이 실제 variant 집합의 부분집합인지 확인하는 한 줄(또는 `axes.FLA_ARCHS` 에서 동반 모델을 유도)과, 동반값 없이는 거부 사유가 아키텍처가 된다는 것을 잡는 테스트가 필요하다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && grep -rn AXIS_VALUE_COMPANIONS tests/   # -> 0 hits
infisical run --env=dev -- uv run python - <<'EOF'
import sys; sys.path[:0]=[".","scripts"]
import audit_plan as ap
print("before:", ap.CHECKS["axis-values"]().detail)
ap.AXIS_VALUE_COMPANIONS.pop("kernel/fla")
print("popped:", ap.CHECKS["axis-values"]().detail)
ap.AXIS_VALUE_COMPANIONS["kernel/does_not_exist"] = ("model=nope",)
r = ap.CHECKS["axis-values"]()
print("bogus key:", r.count, r.detail)
EOF
# 세 줄이 모두 동일한 detail 과 count=3 을 낸다
infisical run --env=dev -- uv run pytest tests/test_audit.py -q   # 111 passed, 변이와 무관
```

**검증** (mutation-killed-nothing):
```text
cd /Users/jwcho/Codes/train-comparison && grep -rn AXIS_VALUE_COMPANIONS tests/ scripts/ trainbench/ ; infisical run --env=dev -- uv run python -c 'import sys; sys.path[:0]=[".","scripts"]; import audit_plan as ap; base=ap.CHECKS["axis-values"]().detail; [print(k, ap.AXIS_VALUE_COMPANIONS.pop(k) and "", "SAME" if ap.CHECKS["axis-values"]().detail==base else "CHANGED") for k in list(ap.AXIS_VALUE_COMPANIONS)]' ; infisical run --env=dev -- uv run pytest tests/test_audit.py -q
---
co_filename: /Users/jwcho/Codes/train-comparison/scripts/audit_plan.py 1657
before: 3 | 36/52 applicable on both text-only/image-carrying data; 3 group(s) offering one usable value: kernel 1/3, precision 1/3, train.offload 1/4
popped: 3 | 36/52 applicable on both text-only/image-carrying data; 3 group(s) offering one usable value: kernel 1/3, precision 1/3, train.offload 1/4
bogus : 3 | 36/52 applicable on both text-only/image-carrying data; 3 group(s) offering one usable value: kernel 1/3, precision 1/3, train.offload 1/4
identical: True True

per-key sensitivity sweep (pop one key, re-run ax
```

### `audit-check-registry-has-no-size-anchor` — major / emptiness

- 단위: macro:emptiness
- 위치: `scripts/audit_plan.py:71`

**주장**: `CHECKS` 자체에 크기·이름 앵커가 없어 체크 하나를 레지스트리에서 지우면 감사 요약·래칫·공허 방지 잠금 셋 다 조용히 줄어든다.

**실패 시나리오**: `@check("axis-wired")`, `@check("config-groups")`, `@check("axis-fields")`, `@check("evidence-committed")` 네 데코레이터를 지우면(테스트에 문자열로 이름이 안 나오는 넷이다) `tests/test_audit.py` 는 전부 초록이고 — `test_every_check_fails_on_an_empty_repository` 와 `test_every_check_returns_a_result_named_after_itself` 는 `sorted(audit_plan.CHECKS)` 로 파라미터화돼 있어 케이스 수만 4개 줄어든다 — 감사는 `8/11 passing, 1 new failure(s), 0 newly fixed, 0 grew, 0 shrank` 를 찍는다(HEAD 클린 복사본 실측; 변이 전 같은 복사본은 `11/15 passing, 2 new failure(s)`). 즉 **지금 빨간 `evidence-committed` 를 지운 것이 `newly fixed` 로도 `new failure` 로도 잡히지 않았다.** `docs/audit-baseline.json` 은 `axis-values`/`verdicts-closed` 두 이름만 들고 있으므로 `classify()` 는 나머지 13개가 사라지는 것을 볼 수단이 없다. HAZARDS §5 의 양방향 래칫이 체크 '안'의 수만 지키고 체크 '집합'은 지키지 않는다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && python - <<'PY'
import pathlib
p=pathlib.Path("scripts/audit_plan.py"); s=p.read_text()
for d in ('@check("axis-wired")','@check("config-groups")','@check("axis-fields")','@check("evidence-committed")'):
    assert d in s, d
    s=s.replace(d,'',1)
p.write_text(s)
PY
infisical run --env=dev -- uv run pytest tests/test_audit.py -q   # 전부 통과, 케이스 수만 4개 감소
infisical run --env=dev -- uv run python scripts/audit_plan.py    # 분모가 15->11, 0 newly fixed / 0 grew / 0 shrank
git checkout scripts/audit_plan.py
```

**검증** (reproduced):
```text
uv run python -c 'mutate: strip @check("axis-wired"), @check("config-groups"), @check("axis-fields"), @check("evidence-committed") from scripts/audit_plan.py' && infisical run --env=dev -- uv run pytest tests/test_audit.py -q && infisical run --env=dev -- uv run python scripts/audit_plan.py; git checkout scripts/audit_plan.py
---
HEAD(0e600e1) 기준선:
  infisical run --env=dev -- uv run python scripts/audit_plan.py
  -> "13/15 passing, 0 new failure(s), 0 newly fixed, 0 grew, 0 shrank, 0 unreadable"
  infisical run --env=dev -- uv run pytest tests/test_audit.py -q
  -> "111 passed, 14 warnings in 20.44s"

변이 후(데코레이터 4개 제거):
  uv run python -c "import sys; sys.path.insert(0,'scripts'); import audit_plan as m; print(m.__file__); print(len(m.CHECKS), sorted(m.CHECKS)); print(m.check.__code__.co_filename, m.check.__code__.co_firstlineno)"
  -> /Users/jwcho/Codes/train-comparison/scripts/audit_plan.py
  -> 11 ['assert-called',
```

### `step-regime-and-entry-point-never-reach-the-record` — major / measurement-validity

- 단위: macro:axis-pipeline
- 위치: `scripts/bench.py:1045`

**주장**: `AdapterOut.required_step_context` 와 `documented_entry_point` 는 timed loop 까지만 도달하고 결과 레코드에도 리포트에도 실리지 않아, 서로 다른 수치 체제에서 잰 axolotl 과 native 가 같은 순위표에 아무 표시 없이 나란히 선다.

**실패 시나리오**: 입력: 같은 파드에서 `framework=native` 와 `framework=axolotl`, 둘 다 `model=qwen3_5_0_8b precision=bf16 run=timing`. axolotl 어댑터는 `trainbench/loader.py:516-525` 로 `StepContext(kind=autocast, device_type=cuda, dtype=bfloat16)` 를 요구하고, `scripts/bench.py:1008` → `train()` 의 `with timer, axes.step_context(config, required_context)` 가 매 스텝을 `torch.autocast(bfloat16)` 안에서 돌린다. native 는 `required_step_context=None` 이라 `contextlib.nullcontext()` 로 순수 bf16 에서 돈다. 그런데 `build_record`(`scripts/bench.py:1045-1054`)가 싣는 최상위 키는 `applied / applied_axes / build_fingerprint / config / device / git_* / host / image / image_digest / metrics / packages / recorded_at` 14개뿐이고(동결 fixture `tests/fixtures/run_record.sample.json` 로 확인), `required_step_context` 도 `documented_entry_point` 도 없다. `grep -c 'documented_entry_point\|required_step_context' scripts/report.py trainbench/record.py` = 0. 잘못된 출력: 두 런의 `packages.torch`/`packages.transformers` 가 같으므로(`docs/support-matrix.md:90` 기준 둘 다 2.13.0 + 5.14.1) `report.stack_of` 가 같은 키를 주고 `_ranked_by_stack` 이 둘을 **같은 표 같은 순위**에 넣는데, 표에도 레코드에도 한쪽이 autocast 영역 안에서 측정됐다는 사실이 없다. `docs/CONTRACTS.md:195-199` 는 바로 그 사실을 '결과를 읽는 쪽이 프레임워크 차이로 오해하지 않도록 **결과에 실려야 한다**'고 못박고, `.plans/notes/adapters.md:89-92` 는 그 사실이 '`documented_entry_point.differs` 와 이 필드 둘 다에 남는다'고 적는다 — 남는 곳은 `build_run` 이 끝나면 사라지는 메모리 객체뿐이므로 그 서술이 틀렸다. 같은 구멍이 `documented_entry_point.differs=true` 인 다섯 프레임워크 전부에 적용된다: 여섯 중 다섯이 자기 문서화된 진입점이 아닌 경로로 측정되는데 산출물 어디에도 그 표시가 없다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && python3 -c "import json;print(sorted(json.load(open('tests/fixtures/run_record.sample.json'))))" && grep -c 'documented_entry_point\|required_step_context' scripts/report.py trainbench/record.py ; grep -rn 'documented_entry_point\|required_step_context' scripts/bench.py ; sed -n '195,199p' docs/CONTRACTS.md ; sed -n '89,92p' .plans/notes/adapters.md
# 기대: 레코드 키 14개에 두 필드가 없고, report.py/record.py 의 grep -c 는 0,
# bench.py 는 두 필드를 step_context 에 넘기기만 하고 build_record 로는 안 넘긴다.
# 실런 확인(선택): tests/test_smoke_cpu.py 에 플러그인으로 trainbench.record.write_json 을 감싸
# print(sorted(payload)) 를 찍고 axolotl 바인딩으로 main() 을 끝까지 돌린다 —
# 출력 키에 required_step_context/documented_entry_point 가 없다.
```

**검증** (reproduced):
```text
tests/test_zz_verify_tmp.py (임시, 삭제함): tests/test_smoke_cpu.py 의 pod_setting/adapter_binding 픽스처로 framework=axolotl, required_step_context=StepContext(autocast/cpu/bfloat16), documented_entry_point(differs=True) 를 선언한 바인딩으로 bench_entry.main() 을 --config/--out 끝까지 실행하고 기록된 result.json 키를 출력. 실행: `infisical run --env=dev -- uv run pytest tests/test_zz_verify_tmp.py -s -q`. 보조: `python3 -c "import json;print(sorted(json.load(open('tests/fixtures/run_record.sample.json'))))"`, `grep -c 'documented_entry_point\|required_step_context' scripts/report.py trainbench/record.py`, `grep -rn '...' scripts/bench.py`
---
EXIT 0
KEYS ["applied", "applied_axes", "build_fingerprint", "config", "device", "git_commit", "git_dirty", "git_source", "host", "image", "image_digest", "metrics", "packages", "recorded_at"]
HAS required_step_context: False
HAS documented_entry_point: False
ANY MENTION: []
SUBSTRING autocast: False | differs: False
1 passed in 1.48s

--- 보조 실행 ---
$ python3 -c "import json;print(sorted(json.load(open('tests/fixtures/run_record.sample.json'))))"
['applied', 'applied_axes', 'build_fingerprint', 'config', 'device', 'git_commit', 'git_dirty', 'git_source', 'host', 'image', 'image_digest', 'metri
```

### `use-cache-only-disabled-on-the-packed-arm` — major / measurement-validity

- 단위: bench
- 위치: `scripts/bench.py:117`

**주장**: `use_cache=False` 가 packed forward 에만 붙어, `dataloader.packing` 축의 두 팔이 타이머 안에서 KV 캐시를 만드느냐로 갈린다.

**실패 시나리오**: 같은 모델·같은 데이터로 dataloader.packing=false 와 true 를 비교하는 ablation. packing=true 는 pooled_embeddings 의 model(**tensors, use_cache=False) 로 가고, packing=false 는 steps.encode(trainbench/probe/steps.py:253) 의 model(**batch, output_hidden_states=False) 로 가서 use_cache 를 넘기지 않는다. docstring 자신이 'every model here ships config.use_cache=True' 라고 적으므로 packing=false 팔만 매 forward 마다 DynamicCache 를 할당하고 즉시 버린다. 그 비용이 packing 축의 speedup 안에 들어가는데 packing 과 무관하다. 크기는 이 호스트에서 측정 안 함(GPU 없음). 스위트가 못 보는 이유: TinyEmbedder.forward(self, input_ids, attention_mask=None, **_) 가 use_cache 를 삼키고, 두 팔의 forward kwargs 를 비교하는 테스트가 없다.

**재현**:
```text
uv run python - <<'PY'
import importlib.util, sys, torch
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path.cwd()))
spec = importlib.util.spec_from_file_location('bench_entry', Path('scripts/bench.py'))
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
seen = []
class Watch(torch.nn.Module):
    def forward(self, input_ids, **kw):
        seen.append(kw.get('use_cache', '<not passed>'))
        return SimpleNamespace(last_hidden_state=torch.zeros(input_ids.shape[0], input_ids.shape[1], 4))
m = Watch()
bench.pooled_embeddings(m, {'input_ids': torch.ones(1,7,dtype=torch.long)}, 'left', torch.tensor([0,3,7], dtype=torch.int32))
bench.pooled_embeddings(m, {'input_ids': torch.ones(2,4,dtype=torch.long), 'attention_mask': torch.ones(2,4,dtype=torch.long)}, 'left', None)
print('packing=true  -> use_cache:', seen[0])
print('packing=false -> use_cache:', seen[1])
PY
기대 출력: packing=true -> use_cache: False / packing=false -> use_cache: <not passed>
```

**검증** (reproduced):
```text
uv run python - <<'PY' (리뷰어 재현 스크립트 + 라이브 정의 확인 두 줄 추가; cwd=/Users/jwcho/Codes/train-comparison, HEAD 0e600e1)
---
live def: /Users/jwcho/Codes/train-comparison/scripts/bench.py 92
encode def: /Users/jwcho/Codes/train-comparison/trainbench/probe/steps.py 244
packing=true  -> use_cache: False
packing=false -> use_cache: <not passed>
```

### `offline-mode-silently-drops-the-pinned-corpus-revision` — major / measurement-validity

- 단위: bench
- 위치: `scripts/bench.py:680`

**주장**: 캐시가 있는 호스트에서 `HF_HUB_OFFLINE=1` 은 예외가 아니라 '최신 캐시본' 폴백이므로, timing/quality 런에서 `data.revision` 핀이 조용히 무시된다.

**실패 시나리오**: jinwoo-cho/mmeb-subset 이 이미 캐시된 호스트/파드에서 purpose=timing 런. close_kernel_fetch_doors 가 오프라인을 켜고, load_pairs 의 load_dataset(repo, revision=<핀>, streaming=True) 이 revision 을 검증하지 못한 채 'Using the latest cached version ... (offline mode is enabled)' 로그 한 줄만 남기고 캐시에 있는 아무 revision 이나 돌려준다. 존재하지 않는 revision('0'*40)으로도 로드가 성공한다. load_pairs docstring('The pin is the point')과 config_schema 의 revision 검증기가 막으려던 상태 — 이름 붙일 수 없는 코퍼스 위의 처리량 숫자 — 가 하필 ENFORCED_PURPOSES(측정 런)에서만 성립한다. blocker 항목을 'ConnectionError 를 refusal 로 잡는다'로 고치면 이 실패는 그대로 남는다.

**재현**:
```text
# 데이터셋이 캐시된 호스트에서 (없으면 먼저 온라인으로 한 번 로드)
HF_HUB_OFFLINE=1 uv run python -c "
from datasets import load_dataset
ds = load_dataset('jinwoo-cho/mmeb-subset', revision='0'*40, split='train', streaming=True)
print('LOADED WITH A BOGUS REVISION:', list(ds.features))"
기대 출력: 'Using the latest cached version ... (offline mode is enabled)' 뒤에 LOADED WITH A BOGUS REVISION: ['qry','qry_image','pos_text','mmeb_config']
```

**검증** (reproduced):
```text
uv run python - <<'PY'
import importlib.util, os
spec = importlib.util.spec_from_file_location("benchmod", "scripts/bench.py")
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
from trainbench.collate import load_pairs
class D: repo_id="jinwoo-cho/mmeb-subset"; revision="0"*40; effective_rows=4
class R: purpose="timing"
class C: data=D(); run=R()
cfg=C()
print("close_kernel_fetch_doors at", bench.close_kernel_fetch_doors.__code__.co_filename, bench.close_kernel_fetch_doors.__code__.co_firstlineno)
print("load_pairs at", load_pairs.__code__.co_filename, load_pairs.__code__.co_firstlineno)
print("HF_HUB_OFFLINE before:", os.environ.get("HF_HUB_OFFLINE"))
print("doors:", bench.close_kernel_fetch_doors(cfg))
print("HF_HUB_OFFLINE after:", os.environ.get("HF_HUB_OFFLINE"))
ds = load_pairs(cfg)
print("LOAD_PAIRS RETURNED ROWS:", len(ds), "keys:", sorted(ds[0].keys()))
PY
# and the reviewer's minimal form:
HF_HUB_OFFLINE=1 uv run python -c "
from datasets import load_dataset
ds = load_dataset('jinwoo-cho/mmeb-subset', revision='0'*40, split='train', streaming=True)
print('LOADED WITH A BOGUS REVISION:', list(ds.features))"
---
Using the latest cached version of the dataset since jinwoo-cho/mmeb-subset couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /Users/jwcho/.cache/huggingface/datasets/jinwoo-cho___mmeb-subset/default/0.0.0/b750b9c3263e9ef5dce225fd50aa25d7c58f1d5f (last modified on Sat Aug  1 09:57:02 2026).
close_kernel_fetch_doors at /Users/jwcho/Codes/train-comparison/scripts/bench.py 579
load_pairs at /Users/jwcho/Codes/train-comparison/trainbench/collate.py 126
HF_HUB_OFFLINE before: None
kernel fetch door closed: $HF_HUB_OFFLINE
```

### `framework-owned-step-refusal-closes-the-tevatron-column-and-orphans-owned-axes` — major / contract-split

- 단위: bench
- 위치: `scripts/bench.py:699`

**주장**: `refuse_a_step_this_harness_cannot_drive` 가 framework-owned step 을 무조건 거부하므로 tevatron 칸은 영구히 숫자를 못 내고, 바로 아래 `assemble(owned_axes=...)` 는 실행 경로에서 항상 빈 dict 다.

**실패 시나리오**: framework=tevatron 인 아무 설정이나 실행하면 stage='binding', kind='AdapterRefusal' 레코드가 나오고 metrics 는 절대 생기지 않는다. loader.ADAPTERS 를 전수하면 owned_axes 가 비어 있지 않은 어댑터는 tevatron 하나이고 그것이 유일한 owner=framework 이며, Adapter.__post_init__ 이 그 동치를 강제한다 -> scripts/bench.py:721 의 owned_axes=binding.owned_axes or {} 는 항상 {} 로 평가되고 applied._owned 의 면제 경로는 실런에서 한 번도 발화하지 않는다(죽은 인자). 동시에 PLAN 결정 5('프레임워크의 학습 스텝을 그대로 잰다')와 trainbench/loader.py:504 의 tevatron documented_entry_point.harness_uses('that same forward driven by the harness timer')가 거짓 단언으로 남고, 여섯 프레임워크 중 하나가 닫혔다는 사실이 .plans/notes/integfix.md 에 없다.

**재현**:
```text
uv run python -c "
from trainbench import loader
for n,a in loader.ADAPTERS.items(): print(f'{n:24} owner={a.step.owner:9} owned_axes={sorted(a.owned_axes)}')
print('owned_axes 를 들면서 framework-owned 가 아닌 어댑터:', [n for n,a in loader.ADAPTERS.items() if a.owned_axes and a.step.owner != loader.FRAMEWORK])"
기대 출력: tevatron 만 owner=framework 이고 마지막 줄은 [] -> owned_axes 를 든 바인딩은 전부 binding 단계에서 거부된다. 이어서 tevatron 설정을 pod_setting 으로 돌리면 record['refusal']['stage']=='binding', 'metrics' not in record (tests/test_smoke_cpu.py::test_a_framework_owned_step_is_refused_instead_of_measured_by_this_loop 가 그 상태를 이미 못박는다).
```

**검증** (reproduced):
```text
1) uv run python -c "from trainbench import loader; [print(f'{n:24} owner={a.step.owner:9} owned_axes={sorted(a.owned_axes)}') for n,a in loader.ADAPTERS.items()]; print('non-framework-owned with owned_axes:', [n for n,a in loader.ADAPTERS.items() if a.owned_axes and a.step.owner != loader.FRAMEWORK])"
2) tests/test_zz_verify_tevatron.py (임시, 삭제함): 실제 ADAPTERS['tevatron'] 를 그대로 쓰고 trainbench.probe.tevatron 모듈만 스텁으로 갈아끼운 뒤 pod_setting 으로 timing_config(run.purpose=probe, framework.name=tevatron) 실행 — infisical run --env=dev -- uv run pytest tests/test_zz_verify_tevatron.py -x -q -s
3) uv run python -c "loader.Adapter(... harness step + owned_axes ...)" / "... framework step + owned_axes 없음 ..."
4) infisical run --env=dev -- uv run pytest tests/test_smoke_cpu.py::test_a_framework_owned_step_is_refused_instead_of_measured_by_this_loop -x -q
5) grep -rn "tevatron" .plans/notes/integfix.md
---
(1)
native                   owner=harness   owned_axes=[]
unsloth                  owner=harness   owned_axes=[]
ms_swift                 owner=harness   owned_axes=[]
sentence_transformers    owner=harness   owned_axes=[]
tevatron                 owner=framework owned_axes=['loss.name', 'parallel.cross_device_negatives']
axolotl                  owner=harness   owned_axes=[]
non-framework-owned with owned_axes: []

(2)
EXIT 3 3
REFUSAL {'kind': 'AdapterRefusal', 'reason': "the adapter declares a framework-owned step (tevatron.retriever.modeling.DenseModel.forward), and this harness only driv
```

### `owned-axes-cannot-reach-capture-in-the-bench-path` — major / contract-split

- 단위: macro:axis-pipeline
- 위치: `scripts/bench.py:721`

**주장**: `binding.owned_axes` 는 `scripts/bench.py` 에서 언제나 빈 매핑이다 — owned_axes 를 실을 수 있는 어댑터는 정의상 framework-owned step 을 선언한 어댑터뿐인데, `refuse_a_step_this_harness_cannot_drive` 가 그 어댑터를 `assemble` 이전에 전부 거부하므로 `FRAMEWORK_OWNED` 축 상태는 런 경로가 만들어 낼 수 없다.

**실패 시나리오**: 입력: `framework=tevatron model=qwen3_vl_emb_2b run=timing`(또는 probe). `loader.ADAPTERS['tevatron']` 은 `step.owner='framework'` 와 `owned_axes={'loss.name','parallel.cross_device_negatives'}` 를 함께 든다. `build_run` 은 `scripts/bench.py:699` 에서 `refuse_a_step_this_harness_cannot_drive(binding.step)` 로 `AdapterRefusal` 을 내고 `scripts/bench.py:711` 의 `axes.assemble(..., owned_axes=binding.owned_axes)` 에 도달하지 못한다. 반대로 `trainbench/loader.py:398-403` 은 `step.owner=='harness'` 인 어댑터가 owned_axes 를 드는 것을 거부한다(실행 확인: "tevatron.step.owner is 'harness' but owned_axes claims [...]"). 두 거부가 집게처럼 맞물려 `applied._disclaimed` 은 항상 None 을 돌려주고, 어떤 결과 JSON 도 `applied.framework_owned` 를 비어 있지 않게 실을 수 없다. 잘못된 출력: (a) `scripts/report.py:1030` `render_owned_axes` 는 영원히 빈 리스트를 돌려주어 '프레임워크 소유 축' 절이 렌더되지 않는다 — 결정 5 의 대가를 보여주기 위해 존재하는 절이 발화하지 않는다. (b) 동결 fixture `tests/fixtures/axis_state.sample.json` 은 자기 `_note` 에서 '세 상태를 한 번에 담는 유일한 모양'이라며 tevatron 셀을 못박는데, 그 레코드는 런 경로가 생산할 수 없는 레코드다 — HAZARDS §4.3 이 `record-report` 로 이미 한 번 겪은 모양 그대로. (c) `docs/support-matrix.md:1132-1134` 는 `framework_owned` 가 '런을 막지 않는다'고, `docs/support-matrix.md:1171` 은 tevatron 의 `DenseModel.forward` 자체를 '하네스 타이머로 돌린다'고 현재형으로 적는데, 실제로는 tevatron 3칸 전부가 `stage=binding` 거부로 끝나 metrics 가 없다(`tests/test_smoke_cpu.py:2005-2012` 가 그 상태를 못박는다). 즉 6개 프레임워크 중 하나가 숫자를 낼 수 없게 됐는데 문서 세 곳은 반대를 말한다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && uv run python -c "
import dataclasses, importlib.util, sys
import trainbench.loader as L
print('owned_axes 를 든 어댑터:', [n for n,a in L.ADAPTERS.items() if a.owned_axes])
print('그 어댑터의 step.owner:', [a.step.owner for a in L.ADAPTERS.values() if a.owned_axes])
t = L.ADAPTERS['tevatron']
try:
    dataclasses.replace(t, step=L.HARNESS_STEP); print('harness+owned: 통과(있을 수 없음)')
except L.AdapterRefusal as e: print('harness+owned 거부:', str(e)[:90])
spec = importlib.util.spec_from_file_location('be','scripts/bench.py')
m = importlib.util.module_from_spec(spec); sys.modules['be']=m; spec.loader.exec_module(m)
try:
    m.refuse_a_step_this_harness_cannot_drive(t.step); print('framework step: 통과(있을 수 없음)')
except L.AdapterRefusal as e: print('framework step 거부:', str(e)[:90])
" ; infisical run --env=dev -- uv run pytest tests/test_smoke_cpu.py -q -k a_framework_owned_step_is_refused_instead_of_measured ; grep -n 'framework_owned' scripts/report.py ; sed -n '1127,1138p;1171p' docs/support-matrix.md
# 기대: owned_axes 를 든 어댑터는 tevatron 뿐이고 그 step.owner 는 'framework';
# harness+owned 조합은 loader 가 거부하고 framework step 은 bench 가 거부한다 →
# axes.assemble 로 가는 owned_axes 는 항상 {} 이다.
# 변이로도 확인 가능: scripts/bench.py:699 의 refuse_a_step_this_harness_cannot_drive 호출을
# 주석 처리하면 tevatron 셀이 다시 framework_owned 를 싣지만, 그것은 0c27aad 가 없앤
# '하네스가 잰 숫자를 프레임워크 소유로 적는' 상태다 — 즉 두 상태 중 하나만 존재할 수 있다.
```

**검증** (reproduced):
```text
cd /Users/jwcho/Codes/train-comparison && uv run python -c "import dataclasses, importlib.util, sys; import trainbench.loader as L; print([n for n,a in L.ADAPTERS.items() if a.owned_axes]); ..." ; infisical run --env=dev -- uv run pytest tests/test_smoke_cpu.py -q -k a_framework_owned_step_is_refused_instead_of_measured ; grep -n 'framework_owned\|render_owned_axes' scripts/report.py ; sed -n '1127,1140p;1168,1175p' docs/support-matrix.md
---
owned_axes 를 든 어댑터: ['tevatron']
그 어댑터의 step.owner: ['framework']
all adapters step.owner: {'native': 'harness', 'unsloth': 'harness', 'ms_swift': 'harness', 'sentence_transformers': 'harness', 'tevatron': 'framework', 'axolotl': 'harness'}
harness+owned 거부: tevatron.step.owner is 'harness' but owned_axes claims ['loss.name', 'parallel.cross_device_negatives']; the harness can
정의 위치: /Users/jwcho/Codes/train-comparison/scripts/bench.py 606
framework step 거부: the adapter declares a framework-owned step (tevatron.retriever.modeling.DenseModel.forward), and this harness only driv

$ uv run python
```

### `collate-metrics-does-not-gate-varlen-absence` — major / measurement-validity

- 단위: macro:contracts
- 위치: `tests/contract/test_collate_metrics.py:330`

**주장**: collate-metrics 는 varlen 넷의 *부분집합*만 막고 *전부 부재*는 막지 않는다 — packed 페이로드가 `seq_idx` 는 값까지 동결하면서 varlen 넷은 동결하지 않아, 두 그룹 중 attention 쪽에만 존재 게이트가 없다.

**실패 시나리오**: 입력: `dataloader.packing=true` 인 배치. 변이: `trainbench/collate.py:453` 의 `batch.update(varlen_kwargs(...))` 한 줄 삭제. 출력: `present=[]` 이므로 `len(present) in (0, 4)` 통과, `extra=set()` 이므로 `extra <= MAY_ADD` 통과, packed 페이로드의 동결 tensors 는 `{input_ids, position_ids, seq_idx}` 뿐이라 `set(expected['tensors']) <= set(tensors)` 도 통과 — `tests/contract/test_collate_metrics.py` 가 통째로 초록이다. 같은 변이를 `seq_idx` 쪽(454행)에 하면 `(marks is not None) is packed` 가 즉시 죽는다. 실제 결과: `arch=qwen3_5` 의 full_attention 층이 팩 전체를 한 시퀀스로 attend 하고(`modeling_flash_attention_utils.py` 의 `all(kwarg is not None ...)` 가 False), 레코드는 `dataloader.packing=True` 를 그대로 certify 한 채 평범한 throughput 숫자를 낸다. 모듈 docstring 은 "the varlen four all together for attention, `seq_idx` alone for the causal conv ... each is judged on its own gate" 라고 적지만 varlen 그룹에는 부분성 게이트만 있고 존재 게이트가 없다. (`tests/test_collate.py:105` 가 이를 덮지만 그 파일은 packing/seqidx 레인 소유이고, 계약이 따로 있는 이유가 HAZARDS §4.1 의 '레인이 자기 게이트를 다시 썼다' 다.)

**재현**:
```text
# 사보타주 전 정의 확인:
infisical run --env=dev -- uv run python -c "import trainbench.collate as m;f=m.PackedBatches.__call__;print(f.__code__.co_filename, f.__code__.co_firstlineno)"
# trainbench/collate.py:453 의 batch.update(varlen_kwargs(...)) 한 줄 삭제 후:
infisical run --env=dev -- uv run pytest tests/contract/test_collate_metrics.py -q   # 통과하면 확정
infisical run --env=dev -- uv run pytest tests/test_collate.py -k varlen -q          # 여기서만 죽는다
```

**검증** (mutation-killed-nothing):
```text
python3 - <<'EOF'
p='trainbench/collate.py'
s=open(p).read()
old='        batch.update(varlen_kwargs(boundaries["cu_seqlens"], boundaries["seq_lengths"]))\n'
assert s.count(old)==1
open(p,'w').write(s.replace(old,''))
EOF
infisical run --env=dev -- uv run pytest tests/contract/test_collate_metrics.py -q
infisical run --env=dev -- uv run pytest tests/test_collate.py -q
---
=== contract (mutated) ===
...........                                                              [100%]
11 passed in 0.41s
=== lane (mutated) ===
        that only bounds a kernel launch."""
>       assert isinstance(packed.tensors["max_length_q"], int)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       KeyError: 'max_length_q'

tests/test_collate.py:157: KeyError
=========================== short test summary info ============================
FAILED tests/test_collate.py::test_a_packed_batch_carries_the_varlen_kwargs_in_full
FAILED tests/test_collate.py::test_the_varlen_kwarg
```

### `record-report-boundary-does-not-carry-measurement-block` — major / contract-split

- 단위: macro:contracts
- 위치: `tests/fixtures/run_record.sample.json:1`

**주장**: 소비자(`scripts/report.py`)가 읽는 `metrics.measurement.baseline_tolerance` 를 record-report 경계가 싣지 않는다 — 생산자(`metrics.summarise`)는 항상 싣는데 동결 페이로드에도 `REQUIRED_METRICS` 에도 없다.

**실패 시나리오**: 입력: 캠페인 base 의 configs 로 합성한 `BenchConfig` + `metrics.summarise(...)`. 출력: summarise 가 항상 내보내는 `measurement`, `step_seconds_aggregate`, `step_seconds_stdev`, `throughput_denominator` 네 키가 `run_record.sample.json['metrics']` 에 하나도 없고, `config.measurement` 그룹도 샘플의 `config` 에 없다(실행 확인). `scripts/report.py:530-547` 의 `declared_tolerance()` 는 `artifact.metrics['measurement'][baseline_tolerance]` 와 `baseline_tolerance_status` 를 읽어 교정/미교정 판정을 가른다. 계약 디렉터리 전체에 `measurement`/`baseline_tolerance` 문자열이 없으므로(grep 0건), `test_peak_memory...`/`test_two_stacks...`/`test_an_axis_the_framework_owns...` 가 돌리는 모든 머지는 `declared` 가 비어 `BASELINE_DEVIATION_LIMIT=0.03` 상수 분기만 탄다. 결과: `metrics.summarise` 가 `measurement` 키를 개명·삭제해도 `pytest tests/contract -q` 는 122 passed 로 남고, 실제 파드 레코드에서는 첫 GPU 파드가 유도하기로 되어 있는 교정 임계값이 리포트에 영원히 도달하지 못한 채 모든 pod 판정이 미교정 3% 로 나온다. 두 레인이 갈라지는 것을 막으라고 있는 경계가 그 필드에 대해서는 아무것도 보고 있지 않다.

**재현**:
```text
infisical run --env=dev -- uv run python -c "import json,sys;from pathlib import Path;sys.path.insert(0,'.');from hydra import compose, initialize_config_dir;from trainbench.compose import resolve;from trainbench import metrics as M;\nwith initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):\n    c=resolve(compose(config_name='config', overrides=['model=qwen3_5_0_8b','framework=native','run=timing','data.limit=512','train.batch_size=16']))[0]\ns=M.summarise([0.24]*12, discard=2, config=c, rows_per_step=64.0, tokens_per_step=11264.0, padded_tokens_per_step=16384.0, peak_bytes=1)\nsample=json.loads(Path('tests/fixtures/run_record.sample.json').read_text())['metrics']\nprint('summarise-only:', sorted(set(s)-set(sample)))"
# -> ['measurement', 'step_seconds_aggregate', 'step_seconds_stdev', 'throughput_denominator']
grep -rn "measurement\|baseline_tolerance" tests/contract/   # 필드로서의 언급 0건
# 변이: trainbench/metrics/__init__.py:351 의 "measurement" 를 "measurement_block" 으로 개명 후
infisical run --env=dev -- uv run pytest tests/contract -q   # 여전히 122 passed 이면 확정
```

**검증** (mutation-killed-nothing):
```text
infisical run --env=dev -- uv run python scratchpad/repro.py  # summarise vs sample diff; then rename "measurement" -> "measurement_block" at trainbench/metrics/__init__.py:351 and run: infisical run --env=dev -- uv run pytest tests/contract -q
---
summarise-only: ['measurement', 'step_seconds_aggregate', 'step_seconds_stdev', 'throughput_denominator']
measurement block: {"declared": true, "repeats": 1, "instrument": "wall_clock", "aggregate": "mean", "trim_fraction": 0.0, "seed_policy": "fixed", "throughput_denominator": "tokens", "baseline_tolerance": 0.03, "baseline_tolerance_status": "uncalibrated"}

$ grep -rn "measurement\|baseline_tolerance" tests/contract/
tests/contract/test_record_report.py:13:Five groups are pinned, each because a measurement of the current tree showed it
tests/contract/test_record_report.py:466:def test_the_g
```

### `boundary-and-library-disagree-on-absent-total-params` — major / contract-split

- 단위: report-orchestrate
- 위치: `tests/test_report.py:790`

**주장**: `test_the_three_training_verdicts_are_one_rule` claims the three `training_verdict` implementations are one rule, but its case list omits the one input on which the frozen boundary and the library actually disagree — an absent `total_params`.

**실패 시나리오**: A record whose `metrics.total_params` is absent (or a non-int) while every other gate field is healthy: `trainbench/metrics/validity.py:215` files it False with "`total_params`=None: the peft.mode check has nothing to compare `trainable_params`=219 against", so `scripts/report.py:403` refuses it and `render_measurements` moves the run into `학습하지 않은 런` and renders no figures for it. `tests/contract/test_record_report.py:320` reaches the peft check through `elif isinstance(total, int)`, skips it, and returns True — the frozen boundary says the same record trained. Measured on this host: `{'total_params': None} | library False | boundary True`. The test that exists to stop exactly this drift passes, because none of its eight cases perturbs `total_params`.

**재현**:
```text
uv run python -c "import importlib.util,json,sys;from pathlib import Path;R=Path('.').resolve();sys.path.insert(0,str(R/'scripts'));s=importlib.util.spec_from_file_location('b',R/'tests/contract/test_record_report.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);from trainbench.metrics import validity;p=json.loads((R/'tests/fixtures/run_record.sample.json').read_text());p['metrics']['total_params']=None;print('library',validity.training_verdict(p['metrics'],peft_mode=p['config']['peft']['mode'],device=p['device'])[0],'boundary',m.training_verdict(p)[0])"  # prints: library False boundary True. Equivalent mutation: add "total-absent": (_sample(total_params=None), False) to VERDICT_CASES in tests/test_report.py and run `uv run pytest tests/test_report.py -q` — `assert theirs[0] is expected` fails.
```

**검증** (reproduced):
```text
uv run python -c "...load boundary via importlib, set metrics.total_params, compare three verdicts..."  # then: patch VERDICT_CASES with "total-absent": (_sample(total_params=None), False) and run `uv run pytest tests/test_report.py -q -k three_training_verdicts`
---
LIVE library def: /Users/jwcho/Codes/train-comparison/trainbench/metrics/validity.py 184
LIVE boundary def: /Users/jwcho/Codes/train-comparison/tests/contract/test_record_report.py 296
LIVE merge def: /Users/jwcho/Codes/train-comparison/scripts/report.py 403
None | library False | boundary True | merge False
'x' | library False | boundary True | merge False
0.5 | library False | boundary True | merge False

# unmutated test (green while the drift exists):
8 passed, 30 deselected in 0.47s

# after adding the "total-absent" case:
>       assert theirs[0] is expected, theirs[1]
E       AssertionE
```

### `precision-empty-module-scan-certifies-fp8` — major / emptiness

- 단위: capture
- 위치: `trainbench/applied.py:1149`

**주장**: `if roots and not swapped:` lets a model that listed **no** modules skip the new module scan entirely, so an fp8 recipe over an empty scan is certified `mxfp8` — and the boundary contract's own certification test passes through exactly that hole.

**실패 시나리오**: `Built(model=<object whose named_modules() yields [] and named_parameters() yields one bf16 tensor>, precision_recipe=MXFP8BlockScaling())` under `precision=mxfp8` returns `applied='mxfp8'`, `matches=True`, `detail['recipe_modules'] == []`. The identical `recipe_modules == []` is what `tests/test_applied.py:747` pins as `applied is None` — the same evidence yields opposite verdicts, decided only by whether the model listed anything at all. `tests/contract/test_applied_axes.py:506 test_precision_is_read_off_the_recipe_the_step_actually_wrapped_with` is green for this reason and no other: its `model()` helper (`tests/contract/test_applied_axes.py:142-150`) is `named_modules=lambda: list(())`. So the module-scan half of the guard is unexercised by the contract that owns this axis, and the certification path the lane added is proven only by a fake that examined nothing.

**재현**:
```text
cat > /tmp/repro_empty_scan.py <<'EOF'
import json, torch
from types import SimpleNamespace
from hydra import compose, initialize_config_dir
from trainbench.compose import resolve
from trainbench.config import to_bench_config
from trainbench.applied import Built, capture
with initialize_config_dir(config_dir="/Users/jwcho/Codes/train-comparison/configs", version_base=None):
    mapping = resolve(compose(config_name="config", overrides=["device=cpu"]))[1]
mapping = json.loads(json.dumps(mapping)); mapping["precision"]["name"] = "mxfp8"
config = to_bench_config(mapping)
model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa", sub_configs=()),
                        named_modules=lambda: [],
                        named_parameters=lambda: iter([("layer.0.weight", torch.zeros(2, dtype=torch.bfloat16))]))
st = capture(Built(model=model, precision_recipe=type("MXFP8BlockScaling", (), {})()), config)
e = next(a for a in st.axes if a.axis == "precision.name")
print("applied =", e.applied, "| matches =", e.matches, "| detail =", e.detail)
EOF
infisical run --env=dev -- uv run python /tmp/repro_empty_scan.py
# observed: applied = mxfp8 | matches = True | detail = {... 'recipe_modules': [], 'recipe': 'MXFP8BlockScaling'}
# then the mutation that shows the contract is load-bearing on the hole:
# drop `roots and ` from applied.py:1149 and run
#   infisical run --env=dev -- uv run pytest tests/contract/test_applied_axes.py -q -k precision_is_read_off_the_recipe
# it goes red, while every test in tests/test_applied.py stays green.
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python scratchpad/repro_empty_scan.py  # then: mutate applied.py:1149 `if roots and not swapped:` -> `if not swapped:` and run `infisical run --env=dev -- uv run pytest tests/contract/test_applied_axes.py -q -k precision_is_read_off_the_recipe` and `... pytest tests/test_applied.py -q`
---
applied = mxfp8 | matches = True | detail = {'base': {'bf16': 1}, 'adapter': {}, 'recipe_modules': [], 'recipe': 'MXFP8BlockScaling'}

# mutation (drop `roots and `):
tests/contract/test_applied_axes.py:521: AssertionError
FAILED tests/contract/test_applied_axes.py::test_precision_is_read_off_the_recipe_the_step_actually_wrapped_with[mxfp8]
FAILED tests/contract/test_applied_axes.py::test_precision_is_read_off_the_recipe_the_step_actually_wrapped_with[nvfp4]
2 failed, 42 deselected in 0.81s

# same mutation, unit suite:
69 passed in 3.24s
```

### `parallel-docstring-claims-fsdp2-is-unreadable` — major / correctness

- 단위: axes
- 위치: `trainbench/axes.py:1286`

**주장**: `_parallel` 독스트링이 "until the capture side reads `FSDPModule` instead, an fsdp2 run applies the axis and is then refused for being unreadable" 라고 적는데, `applied.py` 는 이미 MRO 로 읽는다.

**실패 시나리오**: 파드 캠페인을 짜는 레인이 이 문장과 그것이 가리키는 `.plans/notes/axes.md` §2.1 을 읽고 `parallel=fsdp2` 셀을 "측정 불가" 로 빼 버린다. 실제로는 `applied.FSDP2_BASE`/`_is_fsdp2` 가 `torch.distributed.fsdp.FSDPModule` 인스턴스 검사로 읽어 `applied == "fsdp2"` 를 돌려준다. 반대 방향도 있다: 이 문장을 고치라는 지시로 읽은 사람이 `PARALLEL_WRAPPERS` 에 클래스 이름 접두사 매칭을 다시 넣으면, `applied.py:867-868` 이 명시적으로 경고한 대로 사용자 클래스 `FSDPBlock` 을 fsdp2 로 오독한다.

**재현**:
```text
`uv run pytest tests/test_axes.py::test_fsdp2_shards_in_place_and_the_capture_reads_it_off_the_mro -q` — 통과하며 마지막 줄이 `axis(capture(built, config), "parallel.strategy").applied == "fsdp2"` 다. `grep -n "FSDP2_BASE\|def _is_fsdp2" -A 10 trainbench/applied.py` 로 MRO 리더를 확인하고 `sed -n 1281,1291p trainbench/axes.py` 로 반대 문장을 읽는다.
```

**검증** (reproduced):
```text
uv run pytest tests/test_axes.py::test_fsdp2_shards_in_place_and_the_capture_reads_it_off_the_mro -q ; uv run python -c "import trainbench.axes as ax, trainbench.applied as ap, torch; from torch.distributed.fsdp import FSDPModule; print(ax._parallel.__code__.co_filename, ax._parallel.__code__.co_firstlineno); print(ap._is_fsdp2.__code__.co_filename, ap._is_fsdp2.__code__.co_firstlineno); print(ap.FSDP2_BASE); print(ap.PARALLEL_WRAPPERS); m=torch.nn.Linear(4,4); m.__class__=type('FSDPLinear',(FSDPModule,type(m)),{}); print('is_fsdp2=', ap._is_fsdp2(m), 'strategy=', ap._parallel_strategy_of(m))"
---
.                                                                        [100%]
1 passed in 0.90s

/Users/jwcho/Codes/train-comparison/trainbench/axes.py 1275
/Users/jwcho/Codes/train-comparison/trainbench/applied.py 872
FSDP2_BASE= ('torch.distributed.fsdp', 'FSDPModule')
PARALLEL_WRAPPERS= {'DistributedDataParallel': 'ddp', 'FullyShardedDataParallel': 'fsdp1', 'DeepSpeedEngine': 'deepspeed'}
is_fsdp2= True strategy= fsdp2
```

### `gather-with-grad-claims-ddp-is-still-refused` — major / correctness

- 단위: axes
- 위치: `trainbench/axes.py:1954`

**주장**: `_gather_with_grad` 의 독스트링이 `parallel.strategy=ddp` 를 "still refused by `assemble`" 라고 적는데, 같은 파일 `_parallel` 이 ddp 를 구현한 지 오래다.

**실패 시나리오**: `parallel=ddp` + `cross_device_negatives=true` 로 멀티랭크 런을 계획하는 사람이 이 독스트링을 읽고 "ddp 는 아직 거부된다 → 이 조합은 못 돈다" 로 결론 낸다. 실제로는 `axes._parallel` 이 `torch.nn.parallel.DistributedDataParallel(model, device_ids=ids)` 를 돌려주고 capture 가 `applied == "ddp"` 로 읽는다. 반대 방향의 피해가 더 크다: 이 문장이 곧 "cross-device 축은 지금 gradient all-reduce 없이만 측정된다" 의 근거로 쓰이므로(아래 config 발견), 올바르게 훈련하는 조합이 이미 만들어지는데도 계속 보류된다.

**재현**:
```text
`uv run pytest tests/test_axes.py::test_ddp_wraps_the_model_and_reads_back_as_ddp -q` — 통과하며 `axis(capture(built, config), "parallel.strategy").applied == "ddp"` 를 단언한다. 그리고 `sed -n 1952,1959p trainbench/axes.py` 로 반대 문장을 읽는다. 변이 확인: `trainbench/axes.py:1296` 의 `if strategy == "ddp":` 블록을 `raise UnappliedAxis(...)` 로 바꾸면 그 테스트가 빨개진다 — 즉 지금 거부하지 않는다는 것이 실행으로 확인된다.
```

**검증** (reproduced):
```text
uv run pytest tests/test_axes.py::test_ddp_wraps_the_model_and_reads_back_as_ddp -q  # 통과. 이후 trainbench/axes.py:1296 의 `if strategy == "ddp":` 본문 첫 줄에 `raise UnappliedAxis("MUTATION: ddp refused")` 삽입 후 재실행
---
[베이스라인] .                                                                        [100%]
1 passed in 0.93s

[변이 후]
        strategy = config.parallel.strategy
        if strategy == "single" or strategy in ZERO_STAGES:
            return model, []
        _distributed_world(strategy)
        if strategy == "ddp":
>           raise UnappliedAxis("MUTATION: ddp refused")
E           trainbench.axes.UnappliedAxis: MUTATION: ddp refused

trainbench/axes.py:1296: UnappliedAxis
=========================== short test summary import ============================
FAILED tests/test_axes.py::test_ddp_wraps
```

### `pretokenize-drops-mm-token-type-ids` — major / measurement-validity

- 단위: collate-prompt
- 위치: `trainbench/collate.py:367`

**주장**: `Encode`/`PackedPairs` 는 `processor.tokenizer` 로, `Collate` 는 `processor.__call__` 로 토크나이즈하는데 실 프로세서에서 두 경로의 키 집합이 다르므로 `dataloader.pretokenize` 는 토크나이즈 위치만이 아니라 `model(**tensors)` 가 받는 텐서 집합까지 바꾼다.

**실패 시나리오**: `model=qwen3_vl_emb_2b framework=native dataloader=torch` 로 돌리면 `Collate.__call__` 이 `processor(text=..., padding=True, truncation=True, max_length=2048)` 를 불러 `input_ids`/`attention_mask`/`mm_token_type_ids` 셋을 얻고, `MicroBatch.tensors` 가 셋을 다 실어 `scripts/bench.py::to_device` 가 `(rows, seq)` int 텐서를 타임드 윈도 안에서 H2D 복사한 뒤 `steps.encode` 가 `model(**tensors)` 로 넘긴다. 같은 런을 `dataloader=torch_pretokenized` 로 돌리면 `Encode.__init__:367` 이 `processor.tokenizer` 를 잡아 `mm_token_type_ids` 가 애초에 생기지 않고 `PretokenizedCollate.__call__:572` 가 `{"input_ids","attention_mask"}` 만 만든다. 두 팔의 forward 입력이 다른데 `applied.capture` 는 양쪽 다 축 적용으로 인증하므로, 이 축의 측정 델타는 '토크나이즈가 스텝 밖으로 나갔다' 와 '토큰당 int 텐서 하나가 forward 에서 사라졌다' 를 섞은 값이 되고 결과 JSON 어디에도 그렇게 적히지 않는다. 원인은 `Qwen3VLProcessorKwargs._defaults["text_kwargs"]` 와 `Gemma4ProcessorKwargs._defaults` 의 `return_mm_token_type_ids: True` 이며 `processor.__call__` 만 `_merge_kwargs` 로 그것을 태운다. 스위트가 이것을 못 보는 이유: `tests/test_smoke_cpu.py::FakeProcessor` 에는 `.tokenizer` 속성이 없어 `getattr(processor, "tokenizer", processor)` 가 프로세서 자신으로 폴백하고, `tests/contract/test_collate_metrics.py:174` 의 `StubProcessor.__call__` 은 자기 `tokenizer` 에 그대로 위임한다 — `__call__` 에서만 나오는 키를 가진 스텁이 트리에 하나도 없어서 `microbatch.sample.json` 의 `pretokenized_padded` 와 `padded_text_only` 가 일치해 보이는 것이다.

**재현**:
```text
HF_HUB_DOWNLOAD_TIMEOUT=30 uv run python -c "
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-Embedding-2B')
t = ['<|im_start|>user\nhello there<|im_end|>\n']
print('processor:', sorted(p(text=t, return_tensors='pt', padding=True).keys()))
print('tokenizer:', sorted(p.tokenizer(t, padding=False).keys()))"
# 이 호스트 실측 결과:
#   processor: ['attention_mask', 'input_ids', 'mm_token_type_ids']
#   tokenizer: ['attention_mask', 'input_ids']
#   (input_ids 자체는 동일)
# 스텁이 눈멀었음을 확인:
#   grep -n 'self.tokenizer' tests/test_smoke_cpu.py   -> 결과 없음 (FakeProcessor 에 .tokenizer 없음)
#   sed -n '163,190p' tests/contract/test_collate_metrics.py  -> StubProcessor.__call__ 이 self.tokenizer 에 위임만 함
# 픽스처 변이: StubProcessor.__call__ 의 반환 dict 에
#   encoded['mm_token_type_ids'] = torch.zeros_like(encoded['input_ids'])
# 를 padding 갈래에만 추가하면 padded_* 페이로드만 바뀌고 pretokenized_padded 는 그대로다 — 그 비대칭이 결함이다.
```

**검증** (reproduced):
```text
HF_HUB_OFFLINE=1 uv run python scratchpad/repro.py  # 실 Qwen/Qwen3-VL-Embedding-2B 프로세서 + 실 Hydra 합성(device=cpu model=qwen3_vl_emb_2b framework=native dataloader=torch|torch_pretokenized data.limit=4 train.batch_size=4)로 build_collate 두 팔의 MicroBatch.tensors 키 비교. 보조: HF_HUB_OFFLINE=1 uv run python -c "from transformers import AutoProcessor; p=AutoProcessor.from_pretrained('Qwen/Qwen3-VL-Embedding-2B'); t=['<|im_start|>user\nhello there<|im_end|>\n']; o=p(text=t,return_tensors='pt',padding=True); print(sorted(o.keys())); print(sorted(p.tokenizer(t,padding=False).keys()))"
---
dataloader=torch pretokenize=False
  collate: Collate
  MicroBatch.tensors keys: ['attention_mask', 'input_ids', 'mm_token_type_ids']
dataloader=torch_pretokenized pretokenize=True
  collate: PretokenizedCollate
  Encode.tokenizer: Qwen2Tokenizer
  encoded row keys: ['images_dropped', 'input_ids', 'positive_input_ids']
  MicroBatch.tensors keys: ['attention_mask', 'input_ids']

보조 실행:
processor: ['attention_mask', 'input_ids', 'mm_token_type_ids']
tokenizer: ['attention_mask', 'input_ids']
ids equal: True
```

### `baseline-tolerance-refusal-premise-false` — major / contract-split

- 단위: metrics-schema
- 위치: `trainbench/config_schema.py:350`

**주장**: .plans/notes/integfix.md §1 이 이미 선언한 항목이다(중복 보고임을 밝힌다) — `_no_knob_is_declared_ahead_of_the_code_that_would_apply_it` 이 `baseline_tolerance != 0.03` / `calibrated=true` 를 거부하며 근거로 "pod validity is decided by scripts/report.py's own BASELINE_DEVIATION_LIMIT" 를 적지만, 같은 diff 범위의 dac1e85 가 넣은 `report.declared_tolerance` 가 레코드의 `metrics.measurement.baseline_tolerance` 를 읽어 판정하므로 그 전제는 HEAD 에서 거짓이다.

**실패 시나리오**: 첫 GPU pod 가 노이즈 플로어를 실측해 8.1% 를 얻고 `+measurement.baseline_tolerance=0.081 +measurement.baseline_tolerance_calibrated=true` 로 캠페인을 돌리려 하면, 런이 시작조차 못 하고 ValidationError 로 거부된다. 결과적으로 `metrics.summarise` 는 모든 레코드에 `baseline_tolerance=0.03`, `baseline_tolerance_status="uncalibrated"` 만 실을 수 있고, `scripts/report.py` 의 `TOLERANCE_CALIBRATED` 분기·`BASELINE_DEVIATION_CALIBRATED_SOURCE` 문구·서로 다른 임계값 경고는 실제 런으로는 도달 불가능해 `tests/test_report.py` 가 손으로 만든 레코드에서만 실행된다. 덤으로 `MeasurementConfig` docstring 의 "`baseline_tolerance_calibrated` 는 독자가 둘을 구분하는 수단" 이라는 문장도 거짓이 된다 — 그 필드는 False 이외의 값을 가질 수 없다.

**재현**:
```text
infisical run --env=dev -- uv run python -c "
from tests.test_config import compose_cfg
for ov in ['+measurement.baseline_tolerance=0.081','+measurement.baseline_tolerance_calibrated=true']:
    try: print('OK', ov, compose_cfg(ov).measurement.baseline_tolerance)
    except Exception as e: print('REFUSED', ov, str(e)[:160])"

둘 다 REFUSED 로 나온다. 반대편은 sed -n 514,545p scripts/report.py (declared_tolerance 가 TOLERANCE_FIELD/TOLERANCE_STATUS_FIELD 를 읽는다) 와 sed -n 99,103p scripts/report.py ("Not a second source of truth: it is the default of `measurement.baseline_tolerance`") 로 확인.
조치: `_no_knob_is_declared_ahead_of_the_code_that_would_apply_it` 의 baseline_tolerance 절과 tests/test_config.py::test_a_deviation_threshold_no_report_reads_cannot_be_declared_calibrated 의 두 pytest.raises 를 삭제하면 나머지 39개 테스트는 그대로 통과한다.
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python -c "from tests.test_config import compose_cfg; [print('REFUSED', ov) ...]"  (리뷰어 재현 스크립트 그대로) + grep -rn baseline_tolerance trainbench/
---
REFUSED +measurement.baseline_tolerance=0.081 1 validation error for BenchConfig
measurement
  Value error, measurement.baseline_tolerance=0.081 (calibrated=False) but pod validity is decided by scripts/report.py's own BASELINE_DEVIATION_LIMIT; a record claiming a c
REFUSED +measurement.baseline_tolerance_calibrated=true 1 validation error for BenchConfig
measurement
  Value error, measurement.baseline_tolerance=0.03 (calibrated=True) but pod validity is decided by scripts/report.py's own BASELINE_DEVIATION_LIMIT; a record claiming a cal
```

### `deterministic-leaks-through-quality-purpose` — major / measurement-validity

- 단위: macro:measurement
- 위치: `trainbench/config_schema.py:412`

**주장**: `_timing_runs_are_uncontaminated` guards only `purpose=="timing"`, so `run=quality train.deterministic=true` composes, `set_seed(..., deterministic=True)` disables the kernel autotuning under measurement, and the resulting step times land in `scripts/report.py`'s comparable table with nothing marking them.

**실패 시나리오**: A pod runs `run=quality train.deterministic=true` (accepted — verified by composing it). `scripts/bench.py:989` calls `set_seed(config.train.seed, deterministic=True, warn_only=True)`, which calls `torch.use_deterministic_algorithms(True)` (`trainbench/seed.py:31-35`) — the exact contamination AGENTS.md's measurement rules and `docs/methodology.md` name, and the one the timing branch of this same validator refuses in words (`it disables the kernel autotuning being measured`). The run produces a full `metrics` block with `profiled: false` (`trainbench/metrics/__init__.py:350`, `run.profiler` defaults False for quality). In `scripts/report.py::render_measurements` (:805-836), `quality` is in `MEASURING_PURPOSES` (:84), the artifact is not OOM, has metrics, passes `training_verdict`, and `profiled` is falsy — so it enters `timed` (:822) and `_ranked_by_stack` renders its `step_seconds_p50/p95/mean`, `samples_per_second` and `tokens_per_second` in the same 해석 스택 table as the `timing` runs. The only column that differs is 목적; `_figure_table` (:719-742) renders no determinism field, and `grep -rn deterministic scripts/report.py` returns nothing, so a reader comparing two rows of that table is comparing an autotuned step against a determinism-pinned one with no marker. The same record is also eligible as a pod baseline: `_baseline_value` (:491-502) filters on `profiled` only, so a deterministic quality run can become the reference `baseline_gate` judges every other pod's 3% deviation against.

**재현**:
```text
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,'.')
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from trainbench.config import to_bench_config
with initialize_config_dir(config_dir=str(Path('configs').absolute()), version_base=None):
    for ov in (["run=quality","train.deterministic=true"], ["run=timing","train.deterministic=true"]):
        try:
            c = to_bench_config(OmegaConf.to_container(compose(config_name="config", overrides=ov), resolve=True))
            print(ov, "ACCEPTED", c.run.purpose, c.train.deterministic)
        except Exception as e:
            print(ov, "REFUSED", type(e).__name__)
PY
Observed: `['run=quality', 'train.deterministic=true'] ACCEPTED quality True` / `['run=timing', ...] REFUSED ValidationError`.
Then confirm it reaches the ranked table: `sed -n '805,836p' scripts/report.py` (the `timed` bucket filters on `profiled` only) and `grep -rn 'deterministic' scripts/report.py` (no output).
Mutation: change `if self.run.purpose == "timing":` at `trainbench/config_schema.py:412` to `if self.run.purpose in ("timing", "quality"):` — the first compose above starts refusing, and `tests/test_config.py` still passes, which shows nothing currently pins the quality side.
```

**검증** (reproduced):
```text
1) compose: .venv/bin/python - <<'PY' ... to_bench_config(compose(config_name="config", overrides=["run=quality","train.deterministic=true"])) vs ["run=timing","train.deterministic=true"] PY
2) end-to-end report render: .venv/bin/python /private/tmp/.../scratchpad/repro.py (writes two result.json artifacts on podA — one purpose=timing, one purpose=quality with train.deterministic=true, both profiled:false — then calls report.load_artifacts / split_lanes / render_measurements / _baseline_value from scripts/report.py directly)
3) grep -rn 'deterministic' scripts/report.py  (rc=1, no output)
4) mutation: `if self.run.purpose == "timing":` -> `if self.run.purpose in ("timing", "quality"):` at trainbench/config_schema.py:412, then `infisical run --env=dev -- .venv/bin/python -m pytest tests/test_config.py -q`
5) restore: git checkout -- trainbench/config_schema.py
---
DEFN /Users/jwcho/Codes/train-comparison/trainbench/config_schema.py 402
['run=quality', 'train.deterministic=true'] ACCEPTED quality True profiler= False
['run=timing', 'train.deterministic=true'] REFUSED ValidationError

LIVE render_measurements: /Users/jwcho/Codes/train-comparison/scripts/report.py 805
LIVE _baseline_value: /Users/jwcho/Codes/train-comparison/scripts/report.py 491
skipped: []
measured: [('quality-det', 'quality'), ('timing-run', 'timing')]

### 측정 결과

측정 목적(`profile`/`quality`/`timing`) 런 2건 중 수치를 낸 것 2건, `지표 없음` 0건, `OOM(메모리 한계)` 0건, `학습하지 않은 런` 0건. 각 수치가 무엇을 센 것인지는 아래 '지표
```

### `deferred-frozen-graph-refusal-has-no-second-call-site` — major / emptiness

- 단위: loader-probe
- 위치: `trainbench/loader.py:330`

**주장**: `refuse_a_build_the_fingerprint_condemns` 의 docstring(그리고 커밋 본문, 그리고 `tests/test_loader.py:225` 의 'Deferred, not dropped' 주석)이 '조립 뒤에 `adapter_attaches_later=False` 로 같은 함수를 다시 부르는 호출자'를 현재형으로 단언하지만 그런 호출자는 존재하지 않는다 — 프로덕션 호출은 `describe` 하나뿐이다.

**실패 시나리오**: `peft.mode=lora`(또는 `qlora`) 로 프레임워크가 전부 얼린 빌드를 돌려주면 `loader.describe` 는 거부 없이 `AdapterOut` 을 낸다(실측: `trainable_parameter_names == []` 인 unsloth 빌드가 통과). 그 뒤 `scripts/bench.py::build_run` 은 `axes.assemble` -> `axes.step_context` -> `capture` -> `assert_matches` 를 거치는데 어느 자리도 이 함수를 다시 부르지 않고(`rg` 결과 프로덕션 호출자 1개), `applied.capture` 는 학습 가능 파라미터 수를 축으로 읽지 않으며(`_capture_peft` 는 `peft_config` 만 본다), 하네스 측정 루프에도 `trainable==0` 검사가 없다(`steps.training_step_evidence` 의 그 검사는 프로브 전용이다). 결과: 언 그래프가 파드 시간을 다 쓰고 측정까지 간 뒤 `trainbench/metrics/validity.py:208` 의 `grad_norm <= 0` 로만 사후에 걸린다 — 로드 시점 거부가 존재하는 이유(파드-시간 절약과 '거부된 설정'이라는 결과 기록)가 lora/qlora 전체에서 사라졌다. 같은 `trainable` 리스트를 공유하는 혼합 dtype 거부(`:347-353`)도 언 빌드에서는 `regimes` 가 빈 집합이 되어 함께 무력화되고, 이쪽은 사후 게이트조차 없다. 새 테스트는 함수를 기본 인자로 직접 부를 뿐이라 '연기됐고 버려지지 않았다'를 증명하지 못한다.

**재현**:
```text
cd /Users/jwcho/Codes/train-comparison && rg -n 'refuse_a_build_the_fingerprint_condemns' trainbench/ scripts/
# 기대 출력 2줄뿐: loader.py:320 (def), loader.py:559 (describe 안). assemble 뒤 호출자 0개.
# 통과 확인:
cat > /tmp/repro_defer.py <<'EOF'
import sys
sys.path.insert(0, "/Users/jwcho/Codes/train-comparison")
sys.path.insert(0, "/Users/jwcho/Codes/train-comparison/tests")
from trainbench import loader
from trainbench.config import to_bench_config
from tests.test_loader import compose_bench, _Build
lora = to_bench_config(compose_bench("peft=lora"))
frozen = _Build(); frozen.requires_grad_(False)
out = loader.describe(loader.ADAPTERS["unsloth"], frozen, object(), lora)
print("accepted:", out.framework, out.fingerprint["trainable_parameter_names"])
EOF
infisical run --env=dev --path=/ -- uv run python /tmp/repro_defer.py
# 기대: accepted: unsloth []  (거부 없음)
# 변이 대조: loader.py:341 의 ' and not adapter_attaches_later' 를 지우면 tests/test_loader.py::test_a_frozen_build_is_the_state_lora_has_not_attached_to_yet 가 죽는다 -> 유예는 살아 있고 착지점만 없다
```

**검증** (reproduced):
```text
rg -n 'refuse_a_build_the_fingerprint_condemns' trainbench/ scripts/ ; infisical run --env=dev --path=/ -- uv run python scratchpad/repro_defer.py ; # mutation: drop ' and not adapter_attaches_later' at loader.py:341, then infisical run --env=dev --path=/ -- uv run pytest tests/test_loader.py -x -q -k lora
---
$ rg -n 'refuse_a_build_the_fingerprint_condemns' trainbench/ scripts/
trainbench/loader.py:320:def refuse_a_build_the_fingerprint_condemns(
trainbench/loader.py:559:    refuse_a_build_the_fingerprint_condemns(
(프로덕션 호출자 1개 = loader.describe. scripts/ 0건. 나머지 참조는 tests/test_loader.py:228, tests/test_smoke_cpu.py:2051/2061 뿐)

$ infisical run --env=dev --path=/ -- uv run python .../repro_defer.py
live def: /Users/jwcho/Codes/train-comparison/trainbench/loader.py 320
accepted: unsloth []
regimes over trainable: []

변이 대조 (loader.py:341 의 ' and not adapter_attaches_later' 제거):
FAILED tests/test_l
```

### `norm-chunk-bound-unenforced-for-wide-rows` — major / correctness

- 단위: metrics-schema
- 위치: `trainbench/metrics/validity.py:108`

**주장**: `_squared_norm` 의 청크 계산 `rows = max(1, _NORM_CHUNK_ELEMENTS // (grad.numel() // grad.shape[0]))` 은 한 행이 `_NORM_CHUNK_ELEMENTS` 보다 넓으면 1행으로 클램프되고, 그 한 행 전체가 float64 로 승격된다 — 즉 docstring·테스트가 단언하는 `_NORM_CHUNK_ELEMENTS` 상한이 그 형상에서는 강제되지 않는다.

**실패 시나리오**: 3차원으로 쌓인 expert 가중치 형상 (8, 2048, 5632) bf16 그래디언트를 `gradient_norm` 에 넣으면, 단일 연산이 11,534,336 개 원소를 float64 로 승격한다(92 MB) — 선언된 4,194,304 원소(32 MiB) 상한의 2.75배. 형상 (1, 10_000_000) 이면 10,000,000 원소 전체가 한 번에 승격되어, 이 함수가 없애려던 '그래디언트 통째 승격'과 정확히 같아진다. 이 승격은 `scripts/bench.py` 가 `reset_peak_memory` 와 `peak_memory_bytes` 사이, 그리고 실패를 `status: oom` 으로 적는 블록 안에서 일어나므로, 할당은 스텝의 peak memory 로 보고되고 그 자신의 OOM 은 하드웨어 천장으로 발행된다 — validity.py:123-128 docstring 이 막겠다고 적은 바로 그 결과.

**재현**:
```text
infisical run --env=dev -- uv run python -c "
import torch, math
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from trainbench.metrics.validity import _NORM_CHUNK_ELEMENTS, gradient_norm
class W(TorchDispatchMode):
    def __init__(self): self.widest=0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        leaves=[l for l in (*tree_leaves((args,kwargs)), *tree_leaves(out)) if isinstance(l, torch.Tensor)]
        if any(l.dtype==torch.float64 for l in leaves): self.widest=max(self.widest, *(l.numel() for l in leaves))
        return out
class M(torch.nn.Module):
    def __init__(self, shape):
        super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(*shape, dtype=torch.bfloat16)); self.weight.grad=torch.full(shape, 0.5, dtype=torch.bfloat16)
for shape in [(4096,2048),(8,2048,5632),(1,10_000_000)]:
    m=M(shape); w=W()
    with w: gradient_norm(m)
    print(shape,'widest_f64',w.widest,'bound',_NORM_CHUNK_ELEMENTS,'OVER' if w.widest>_NORM_CHUNK_ELEMENTS else 'ok')"

실측 출력: (4096,2048)->4194304 ok / (8,2048,5632)->11534336 OVER / (1,10000000)->10000000 OVER.
tests/test_metrics.py 의 두 새 테스트는 row 폭이 2048 인 한 형상만 쓰므로 이 간극을 절대 잡지 못한다 — `_OneBigGradient` 를 (1, 10_000_000) 으로 인스턴스화하면 같은 단언이 즉시 깨진다.
```

**검증** (reproduced):
```text
infisical run --env=dev -- uv run python -c "<TorchDispatchMode float64 watcher around gradient_norm for shapes (4096,2048), (8,2048,5632), (1,10_000_000) bf16>"
---
def at /Users/jwcho/Codes/train-comparison/trainbench/metrics/validity.py 115
(4096, 2048) widest_f64 4194304 bound 4194304 ok
(8, 2048, 5632) widest_f64 11534336 bound 4194304 OVER
(1, 10000000) widest_f64 10000000 bound 4194304 OVER
```

## 반박된 발견 — 남긴다

- `[refuted]` **report-copies-the-baseline-prefix-orchestrate-writes** (major, report-orchestrate) — `BASELINE_RUN_PREFIX = "baseline:"` is a second literal copy of the name `scripts/orchestrate.py:467` writes, and the comment above it claims the two "cannot drift apart" — only the sanitisation is shared, the prefix itself is spelled out twice and nothing binds them.
  - 근거: FAILED tests/test_pods.py::test_a_plan_this_image_cannot_run_measures_nothing
FAILED tests/test_pods.py::test_one_unrunnable_setting_stops_the_settings_that_would_have_run
7 failed, 1193 passed, 14 warnings in 107.41s (0:01:47)
- `[refuted]` **report-hardcodes-the-schema-default-it-says-it-reads** (major, report-orchestrate) — `BASELINE_DEVIATION_LIMIT = 0.03` is a hardcoded copy of `trainbench/config_schema.py:28`, even though the comment directly above says "it is the default of `measurement.baseline_tolerance` (trainbench/config_schema.py)" and `tests/test_config.py:309` left the instruction "`config_schema.BASELINE_DEVIATION_LIMIT` is the one value for the report to read when it does".
  - 근거: -- 변이 직후 상수 분기 확인 --
trainbench/config_schema.py:28:BASELINE_DEVIATION_LIMIT = 0.05
scripts/report.py:102:BASELINE_DEVIATION_LIMIT = 0.03
schema 0.05 report 0.03
fallback Tolerance(value=0.03, status='미선언', notes=[])

-- 리뷰어가 "1200 passed"라고 적은 게이트 (해당 모듈만) --
>       assert tolerance.value == pytes

## minor

- `recipe-base-dtypes-admits-unreachable-fp16` (capture) `trainbench/applied.py:1068` — `RECIPE_BASE_DTYPES = ("bf16", "fp16")` admits a store nothing in this repository can load, and the comment directly above it is the argument for excluding it.
- `recipe-module-scan-accepts-a-different-recipe-package` (capture) `trainbench/applied.py:1149` — The module scan asks whether *any* package in `PRECISION_MODULE_ROOTS` is in the tree, not whether the package that defines the recipe on `Built` is — so a torchao-only model certifies a Transformer Engine recipe.
- `handoff-note-to-axes-lane-now-false` (capture) `.plans/notes/capture.md:21` — The handoff line telling the axes lane that `precision` "은 `Built.precision_recipe` 에 Transformer Engine recipe 객체를 실어야 읽힌다" is now incomplete: after this change a recipe alone never certifies the axis on any real `nn.Module`.
- `notes-fsdp2-section-names-a-test-that-does-not-exist` (axes) `.plans/notes/axes.md:52` — §2.1 이 `parallel=fsdp2` 를 "측정이 열리지 않는다" 로 적고(§5 표 182행도 "측정 불가"), 고정 근거로 `tests/test_axes.py::test_fsdp2_shards_in_place_and_the_capture_cannot_yet_see_it` 를 지목하는데 그 이름의 테스트는 없다.
- `te-stub-linear-attribute-is-never-read` (axes) `tests/test_axes.py:4643` — 이번 diff 가 추가한 `pytorch.Linear = _TELinear` 는 아무도 읽지 않는다 — `recipe_model()` 이 `_TELinear` 를 직접 쓰고, `axes.py` 는 TE 모듈에서 `autocast` 만 `getattr` 한다.
- `packedpairs-pretok-pair-order-unguarded` (collate-prompt) `trainbench/collate.py:408` — `PackedPairs.__call__` 사전토크나이즈 갈래의 쿼리-먼저 순서는 도크스트링이 '이 순서가 곧 페어링' 이라고 적는 불변식인데 전 스위트에 그것을 잡는 단언이 하나도 없다.
- `stale-scripts-bench-symbol-refs` (collate-prompt) `trainbench/axes.py:1655` — `scripts/bench.py::Collate` / `::PairDataset` / `::MicroBatch` / `::_group_by_row` 를 가리키는 참조 9곳이 남아 있는데 wave 0 이후 `scripts/bench.py` 는 그 넷 중 아무것도 정의하지 않는다.
- `loader-run-record-key-reexport-has-no-consumer` (loader-probe) `trainbench/loader.py:49` — `RUN_RECORD_KEY` 재수출의 주석은 '여기 생산자와 레코드 작성자가 한 곳에서 이름을 적는다'고 하지만 레코드 작성자는 `kernels.RUN_RECORD_KEY` 를 직접 적고, `loader.RUN_RECORD_KEY` 의 유일한 소비자는 이번에 추가된 테스트다.
- `adapter-attaching-peft-modes-is-a-second-list` (loader-probe) `trainbench/loader.py:60` — `ADAPTER_ATTACHING_PEFT_MODES = ("lora", "qlora")` 는 `PeftConfig.mode` 의 Literal 을 손으로 다시 적은 두 번째 목록이고, 같은 파일이 `FRAMEWORKS` 에 대해서는 스키마에서 유도하는 것과 어긋난다 — 둘을 묶는 테스트도 없다.
- `ast-shared-load-check-misses-inline-framework-calls-in-load` (loader-probe) `tests/test_loader.py:580` — `_plain_calls` 가 bare name 호출만 세므로, `load` 안에서 인라인으로 이루어지는 서드파티 호출(tevatron 의 `AutoProcessor.from_pretrained`)은 계약에서 빠지고 프로브의 `_tokenizer` 가 같은 호출을 두 번째로 정의한 채 초록으로 남는다.
- `revision-resolver-chain-has-no-production-caller` (kernels) `trainbench/kernels.py:417` — `revision_resolver`/`requested_ref` 배관이 `kernels` -> `loader.build_fingerprint` -> `loader.describe` -> `loader.load` 까지 이어져 있는데 프로덕션 호출자가 하나도 없어서, Hub 로 해석된 커널은 항상 거부로만 끝나고 완료 조건 1(repo+revision 이 런 레코드에 들어간다)이 어떤 런에서도 성립할 수 없다.
- `flash-attention-falls-back-to-the-hub-is-uncalled` (kernels) `trainbench/kernels.py:554` — `flash_attention_falls_back_to_the_hub` 는 `trainbench/`, `scripts/` 어디에서도 호출되지 않아 아무것도 게이팅하지 않는 죽은 서술자다.
- `gate-fields-not-wired` (metrics-schema) `trainbench/metrics/validity.py:51` — `GATE_FIELDS` 는 "Absence is a refusal rather than a pass" 라는 계약을 선언하지만 어떤 프로덕션 코드도 이 튜플을 순회하지 않는다 — 검사는 `training_verdict` 안의 손으로 쓴 필드별 분기뿐이고, 튜플에 이름을 넣는 것은 아무 검사도 만들지 않는다.
- `duplicate-lora-rank-validator` (metrics-schema) `trainbench/config_schema.py:658` — `_lora_needs_rank` 와 `_adapter_rank_is_set_when_an_adapter_is_used`(:605) 는 같은 조건을 두 번 검사한다 — `peft.mode` 의 Literal 이 full|lora|qlora 이므로 `mode != "full"` 과 `mode in ("lora","qlora")` 는 동치이고, 둘 다 `r <= 0` 에서 거부한다.
- `statistics-docstring-contradicts-schema-pin` (metrics-schema) `trainbench/metrics/statistics.py:10` — `statistics.py` 모듈 docstring("until then the schema's job is to make the answer expressible and recorded")과 `repeat_seeds` docstring(:182 "Both policies are expressible ... Making `per_repeat` producible now is what stops that change from being a schema change later")이 HEAD 의 스키마와 정면으로 어긋난다 — 새 validator 가 `seed_policy != "fixed"` 와 `repeats != 1` 을 거부하므로 config 로는 표현 불가능하고, 되돌리는 것은 정확히 schema change 다.
- `refusal-types-docstring-claims-an-openness-it-does-not-have` (bench) `scripts/bench.py:439` — `refusal_types()` docstring 은 loader 에 추가된 refusal 타입을 이 파일이 잊을 수 없다고 적지만 몸통은 네 클래스를 적은 리터럴 튜플이고, 함수인 것이 주는 것은 지연 import 뿐이다.
- `preflight-crashes-on-a-plan-item-name-containing-a-brace` (bench) `scripts/bench.py:920` — `described = f"{name}: {{}}"` 뒤의 `.format()` 이 plan 항목 이름을 포맷 템플릿으로 다시 해석해, 중괄호가 든 이름이면 preflight 가 보고 대신 트레이스백으로 죽는다.
- `report-keeps-its-own-oom-status-string` (report-orchestrate) `scripts/report.py:74` — `STATUS_OOM = "oom"` is a third copy of the record status string (`trainbench/metrics/validity.py:63` defines it, `tests/contract/test_record_report.py:137` pins it), and this diff added the `validity` import that makes the copy unnecessary.
- `axis-state-sample-packing-detail-unproducible` (macro:contracts) `tests/fixtures/axis_state.sample.json:42` — applied-axes 경계 샘플의 `dataloader.packing` 항목이 `applied: "False"` 와 `detail: {"collate": "PackedCollate"}` 를 함께 적는데, 그 조합은 capture 가 만들 수 없다.
- `harness-step-batch-keys-is-a-padded-only-declaration` (macro:contracts) `trainbench/loader.py:423` — loader-bench 가 `batch_keys` 를 "collate 가 만들어야 하는 것" 이라 못박는데 `HARNESS_STEP.batch_keys` 는 `attention_mask` 를 포함하고, collate-metrics 는 packed 배치에 `attention_mask` 가 없다고 단언한다 — 두 경계가 같은 키에 대해 반대로 적는다.
- `microbatch-frozen-at-commit-stale` (macro:contracts) `tests/fixtures/microbatch.sample.json:3` — `frozen_at_commit: "0604684"` 이 seq_idx 개정(`d2a44b2`) 이후에도 그대로라 페이로드의 출처를 잘못 가리킨다.
- `deferred-frozen-graph-refusal-has-no-caller` (macro:axis-pipeline) `trainbench/loader.py:334` — `refuse_a_build_the_fingerprint_condemns` 의 docstring 이 '`axes.assemble` 뒤에 `adapter_attaches_later=False` 로 다시 부르는 호출자가 있고 그 호출이 어댑터가 안 붙은 LoRA 런을 실제로 잡는다'고 적는데 그 호출자가 존재하지 않아, LoRA/QLoRA 런에서는 이 가드의 두 검사가 모두 빈 대상 위를 지나간다.
- `shipped-sweep-probe-cells-can-empty` (macro:emptiness) `tests/test_pods.py:2144` — `test_the_whole_shipped_sweep_keeps_every_cell_it_launched` 의 유일한 실질 단언이 `probed` 필터 결과 위를 돌고, 그 집합이 비지 않는다는 단언이 없다.
- `packing-isolation-predicate-true-on-empty-backbones` (macro:emptiness) `trainbench/kernels.py:240` — `packing_isolation_holds` 는 `backbones` 가 빈 매핑이면 `all()` 로 True 를 돌려주고, 프로덕션 호출자가 하나도 없다.
