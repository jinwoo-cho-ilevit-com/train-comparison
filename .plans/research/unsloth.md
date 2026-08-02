# unsloth — 핀된 소스 리서치 브리프

소비 레인: **adapters**
작성일: 2026-08-02
근거: 호스트 uv 캐시에 이미 풀려 있는 휠. 웹 검색·기억에서 온 문장은 이 문서에 없다.

## 0. 핀 해석 (무엇을 열었는지)

`envs/unsloth/uv.lock` 이 말하는 값만 쓴다.

| 패키지 | lock 버전 | lock 줄 | 읽은 경로 |
|---|---|---|---|
| `unsloth` | `2026.7.6` | `envs/unsloth/uv.lock:1531-1533` | `/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/` |
| `unsloth-zoo` | `2026.7.7` | `envs/unsloth/uv.lock:1598-1600` | `/Users/jwcho/.cache/uv/archive-v0/PM921ZbVZCUP68sU/unsloth_zoo/` |
| `transformers` | `5.5.0` | `envs/unsloth/uv.lock:1425-1427` | `/Users/jwcho/.cache/uv/archive-v0/plcyRhzg-LE7LDvn/transformers/` |
| `torch` | `2.11.0+cu130` | `envs/unsloth/uv.lock:1282-1284` | **열지 못함** (아래 §9) |

거부한 디코이 — 같은 캐시에 있으나 lock 이 가리키지 않는 것:

- `unsloth-2026.6.9` (`CNYs9gxUdoYppfpA`)
- `unsloth_zoo-2026.6.7` (`_j2YfL_quciSG91o`)
- `transformers` 4.57.6 / 5.3.0 / 5.9.0 / 5.10.2 / 5.11.0 / 5.12.0 / 5.12.1 / 5.13.0 / 5.13.1 / 5.14.1
- `torch` 2.8.0 / 2.10.0 / 2.12.0 / 2.12.1 / 2.13.0, 그리고 `torch-2.11.0` 이라 적힌 macOS arm64 휠 3개
  (`-4elqbiP7unMIsFk`, `2PEibOWC7OiGgOAl`, `SKrbNV4o9h-8FrcB`) — 셋 다 `Tag: cp31x-cp31x-macosx_11_0_arm64`
  이고 `+cu130` 이 아니다. 이름이 같다고 핀이 아니다.

이 lock 이 함께 고정한 스택 교란 요인 (lock 에서 그대로 읽음):

| 패키지 | 버전 | lock 줄 |
|---|---|---|
| `torch` | `2.11.0+cu130` (`download.pytorch.org/whl/cu130`) | 1282-1284 |
| `transformers` | `5.5.0` | 1425-1427 |
| `triton` | `3.6.0` | 1445-1447 |
| `trl` | `0.24.0` | 1456-1458 |
| `peft` | `0.20.0` | 891-893 |
| `sentence-transformers` | `5.6.1` | 1173-1175 |
| `cut-cross-entropy` | `25.1.1` | 246-248 |
| `bitsandbytes` | `0.50.0` | 122-124 |

---

## 1. `from_pretrained` 시그니처와 `full_finetuning` 기본값

`FastVisionModel` 과 `FastTextModel` 은 **본문이 없다**. 둘 다 `FastModel` 이고, `FastModel` 은
`FastBaseModel` 이다. 그러니까 "vision 전용 로더" 같은 것은 이 버전에 없다.

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/loader.py:2014-2019`

```python
class FastVisionModel(FastModel):
    pass


class FastTextModel(FastModel):
    pass
```

`FastModel.from_pretrained` — `full_finetuning = False` 가 기본값이고, `load_in_4bit = True` 도 기본값이다.

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/loader.py:1021`, `:1046-1056`

```python
class FastModel(FastBaseModel):
```

```python
    @staticmethod
    @_offline_aware_load
    def from_pretrained(
        model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,  # 4bit QLoRA
        load_in_8bit = False,  # 8bit  LoRA
        load_in_16bit = False,  # 16bit LoRA
        full_finetuning = False,
        token = None,
```

`FastLanguageModel` 은 `FastLlamaModel` 을 상속하며 별도 시그니처를 갖지만 기본값은 동일하다.

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/loader.py:335-347`

```python
class FastLanguageModel(FastLlamaModel):
    @staticmethod
    @_offline_aware_load
    def from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,  # 4bit QLoRA
        load_in_8bit = False,  # 8bit  LoRA
        load_in_16bit = False,  # 16bit LoRA
        full_finetuning = False,
        token = None,
        device_map = "sequential",
```

`full_finetuning=True` 를 주면 `FastLanguageModel` 은 자기 경로를 버리고 `FastModel` 로 위임한다.

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/loader.py:411-421`

```python
        # @_offline_aware_load already forced offline when needed; delegations inherit it.
        if load_in_8bit or full_finetuning or qat_scheme is not None:
            return FastModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                load_in_8bit = load_in_8bit,
                load_in_16bit = load_in_16bit,
                full_finetuning = full_finetuning,
                token = token,
```

실제 구현이 있는 곳은 `FastBaseModel.from_pretrained` 이고, 여기서도 `full_finetuning = False` 다.

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:807-819`

```python
class FastBaseModel:
    @staticmethod
    @_offline_aware_load
    def from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        load_in_8bit = False,
        load_in_16bit = False,
        full_finetuning = False,
        token = None,
        device_map = "sequential",
```

---

## 2. 기본값에서 전 파라미터 동결까지의 사슬

### 2-1. `vision.py` — 16bit LoRA 전환과 env export

4bit/8bit/full 을 전부 끄면 (우리 하네스의 `peft.mode` 가 `full` 이 아닌데 `load_in_4bit=False` 인 경우)
경고만 찍고 지나간다. **`load_in_16bit` 을 켜주지 않는다** — 문구만 "Switching to 16bit LoRA" 다.

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:1162-1187`

```python
        elif load_in_16bit:
            bnb_config = None
        elif not load_in_4bit and not load_in_8bit and not full_finetuning:
            print("Unsloth: QLoRA and full finetuning all not selected. Switching to 16bit LoRA.")

        if full_finetuning:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1"
            if dtype == torch.bfloat16:
                if float32_mixed_precision != True:
                    print(
                        f"Unsloth: Using bfloat16 full finetuning which cuts memory usage by 50%.\n"
                        f"To enable float32 training, use `float32_mixed_precision = True` during FastLanguageModel.from_pretrained"
                    )
                else:
                    print(
                        f"Unsloth: Using full float32 full finetuning. "
                        f"To enable bfloat16 training to reduce VRAM usage by 50% albeit with a slightly higher loss, do:\n"
                        "use `float32_mixed_precision = False` during FastLanguageModel.from_pretrained"
                    )
                    os.environ["UNSLOTH_BFLOAT16_MIXED_PRECISION"] = "1"
            else:
                print(
                    "Unsloth: Float16 full finetuning uses more memory since we upcast weights to float32."
                )
        else:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "0"
```

핵심: 이 export 는 **인자가 아니라 프로세스 전역 상태**다. 이후 단계는 인자를 다시 보지 않고 이 env 를 읽는다.

### 2-2. `from_pretrained` 끝에서 `post_patch_model` 호출

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:1749-1758`

```python
        model._unsloth_trust_remote_code = trust_remote_code
        # Post patches
        model = FastBaseModel.post_patch_model(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            trust_remote_code = trust_remote_code,
            model_type = model_type_arch,
            tokenizer = tokenizer,
            float32_mixed_precision = float32_mixed_precision,
        )
```

`get_peft_model` 경로도 같은 함수를 다시 호출한다 (`vision.py:2050-2056`).

### 2-3. `post_patch_model` — env 를 읽어 `prepare_model_for_training` 에 넘긴다

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:2086-2094`

```python
    def post_patch_model(
        model,
        use_gradient_checkpointing = True,
        trust_remote_code = False,
        model_type = None,
        tokenizer = None,
        float32_mixed_precision = None,
    ):
        full_finetuning = os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1"
```

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:2122-2132`

```python
        model = prepare_model_for_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant = use_reentrant,
            full_finetuning = full_finetuning,
            train_layernorms = full_finetuning,
            train_embedding = full_finetuning,
            train_lm_head = full_finetuning,
            float32_mixed_precision = float32_mixed_precision,
            patch_modules_to_save = True,
        )
```

### 2-4. `unsloth_zoo/training_utils.py` — 실제로 얼리는 곳

`/Users/jwcho/.cache/uv/archive-v0/PM921ZbVZCUP68sU/unsloth_zoo/training_utils.py:376-412`

```python
    for name, param in model.named_parameters():
        original_name = name
        upcast = False
        requires_grad = False
        _keep_param_dtype = False
        _is_norm = _is_norm_parameter(original_name, param)
        if not full_finetuning:
            if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
                upcast = True
                requires_grad = True
            elif (_is_peft_model and "bias" in name and param.requires_grad
                    and ".modules_to_save." not in name):
                # Respect PEFT's bias decision: bias="all"/"lora_only" marks biases
                # trainable; freezing them here disabled bias training (#2343).
                # _keep_param_dtype: keep the loaded dtype, since fp32 on a bf16/fp16
                # Linear breaks the matmul. modules_to_save is excluded so a saved head
                # with a frozen weight isn't partially trained via its bias (#2343 review).
                requires_grad = True
                _keep_param_dtype = True
            else:
                requires_grad = False
        else:
            # Norms need fp32 for adam writeback (~60% of bf16 norm updates round to
            # zero otherwise); a prior dangling else on train_lm_head had clobbered this.
            requires_grad = True
            upcast = False
            if (train_layernorms
                    and _is_norm
                    and id(param) not in _externally_managed_param_ids
                    and not _disable_float32_norm_upcast):
                upcast = True
        pass
        # Set training or not
        if requires_grad:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
```

**사슬 요약** (adapters 레인이 기억할 한 줄):
`full_finetuning=False` (기본) → `vision.py:1187` 이 `UNSLOTH_ENABLE_FULL_FINETUNING=0` export →
`vision.py:2094` 가 그걸 읽음 → `vision.py:2126` 이 `prepare_model_for_training(full_finetuning=False)` →
`training_utils.py:383` 의 이름 매칭(`.lora_A.` / `.lora_B.` / `.lora_magnitude_vector`)에 걸리지 않는
**모든** 파라미터가 `training_utils.py:412` 에서 `requires_grad_(False)`.
LoRA 를 붙이지 않았으면 걸리는 이름이 하나도 없고, 모델 전체가 얼어붙는다.

`_is_peft_model = hasattr(model, "peft_config")` (`training_utils.py:374`) 이므로 bias 예외 분기도
PEFT 모델이 아니면 애초에 열리지 않는다.

---

## 3. `enable_input_require_grads()` — 얼어붙은 그래프에서 backward 가 통과하는 이유

같은 함수 안, 동결 루프 **뒤에서** 호출된다. `use_reentrant` 는 `post_patch_model` 이
`not is_distributed()` 로 계산하므로 단일 GPU 프로브에서는 `True` 다 (`vision.py:2107`).

`/Users/jwcho/.cache/uv/archive-v0/PM921ZbVZCUP68sU/unsloth_zoo/training_utils.py:477-485`

```python
    # If use_reentrant = True which is the Pytorch default, we just make the input requires_grad.
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    pass
```

그 `enable_input_require_grads` 는 transformers 5.5.0 의 것이고, 하는 일은 임베딩 **출력 텐서**에
`requires_grad_(True)` 를 거는 forward hook 등록이다. 파라미터가 아니라 활성값이다.

`/Users/jwcho/.cache/uv/archive-v0/plcyRhzg-LE7LDvn/transformers/modeling_utils.py:2127-2158`

```python
    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        hooks = []
        seen_modules = set()
        found_embeddings = False

        for module in self.modules():
            if not (isinstance(module, PreTrainedModel) and hasattr(module, "get_input_embeddings")):
                continue

            try:
                input_embeddings = module.get_input_embeddings()
            except NotImplementedError:
                continue

            if input_embeddings is None or not hasattr(input_embeddings, "register_forward_hook"):
                continue

            embedding_id = id(input_embeddings)
            if embedding_id in seen_modules:
                continue

            seen_modules.add(embedding_id)
            hooks.append(input_embeddings.register_forward_hook(make_inputs_require_grads))
            found_embeddings = True
```

**왜 예외 없이 도는가.** `requires_grad=True` 인 리프 텐서가 그래프에 하나라도 있으면
autograd 는 `loss.backward()` 를 정상 실행한다. 여기서 그 리프는 임베딩 출력 활성값이다.
따라서 손실은 유한값으로 나오고 backward 는 성공하지만, 도달할 파라미터가 없으므로
`p.grad` 는 아무 데도 채워지지 않는다. 1차 캠페인의 3칸이 `params_with_grad=0,
trainable_params=0` 인데 `infonce_backward` 를 통과한 것이 정확히 이 구조다
(`AGENTS.md:162-169` 가 기록한 것과 일치).

`transformers` 는 임베딩을 찾지 못하면 예외 대신 `logger.warning_once` 만 낸다
(`modeling_utils.py:2164-2169`) — 즉 조용히 그래프가 끊긴 경우도 로드는 성공한다.

> adapters 레인이 가져갈 결론: **loss 가 유한한 것은 학습 증거가 아니다.**
> `trainbench/probe/steps.py:420-432` 의 `trainable == 0` / `with_grad == 0` 거부는 이 사슬에
> 대한 정확한 방어이며, 유지되어야 한다.

---

## 4. `padding_side` 를 무조건 덮어쓰는 지점

한 번이 아니라 **네 계층**에서 덮어쓴다. 전부 `unsloth/models/vision.py`.

### (a) 토크나이저/프로세서 로드 시점 — 인자로 `"left"` 강제

`padding_side = "left"` 가 인자로 박혀 있는 줄: `vision.py:694`, `:1507`, `:1522`, `:1533`, `:1638`, `:1674`, `:1683`.
아래는 `:1506-1513` (`ForConditionalGeneration` 계열 프로세서 로드).

```python
                    _tok = auto_processor.from_pretrained(
                        tokenizer_name,
                        padding_side = "left",
                        token = token,
                        language = whisper_language,
                        task = whisper_task,
                        trust_remote_code = trust_remote_code,
                        cache_dir = kwargs.get("cache_dir"),
                        local_files_only = lfo,
                    )
```

### (b) 로드 직후 무조건 대입 — 조건 없음

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:1715-1727`

```python
        # Save tokenizer for inference purposes
        tokenizer.padding_side = "left"  # Force inference
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.padding_side = "left"  # Force inference
        # Audio feature extractors must stay right padded: left (a text setting,
        # forwarded by from_pretrained) shifts Whisper mels and desyncs Gemma 4
        # audio token counts (crash on transformers < 5.10).
        feature_extractor = getattr(tokenizer, "feature_extractor", None)
        if (
            feature_extractor is not None
            and getattr(feature_extractor, "padding_side", None) == "left"
        ):
            feature_extractor.padding_side = "right"
```

### (c) `post_patch_model` 안에서 모델 트리를 타고 내려가며 다시 `"left"`

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:2166-2175`

```python
        # Patch tokenizer to pad to the left
        m = model
        while hasattr(m, "model"):
            if hasattr(m, "_saved_temp_tokenizer"):
                if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                    m._saved_temp_tokenizer.tokenizer.padding_side = "left"
            m = m.model
        if hasattr(m, "_saved_temp_tokenizer"):
            if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                m._saved_temp_tokenizer.tokenizer.padding_side = "left"
```

### (d) `for_inference()` / `for_training()` 이 매번 다시 뒤집는다

`vision.py:2234-2236` (for_inference → `"left"`)

```python
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"):
                m._saved_temp_tokenizer.padding_side = "left"
```

`vision.py:2295-2297` (for_training → `"right"`, 주석은 "left" 라고 적혀 있으나 코드는 `"right"`)

```python
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"):
                m._saved_temp_tokenizer.padding_side = "right"
```

**adapters 레인 함의.** `from_pretrained` 가 돌려준 processor 는 `padding_side="left"` 다.
`config.model.padding_side` 가 `right` 인 모델이면 프로브가 배치를 만들기 전에 반드시 되돌려야 한다
(`trainbench/probe/steps.py` 의 `align_padding_side` 가 이미 그 일을 한다). 그리고 (c)/(d) 는
모델에 붙은 `_saved_temp_tokenizer` 를 건드리는 것이고 (b) 는 **호출자가 받은 그 객체**를 건드리는
것이라, 같은 객체일 수도 다른 객체일 수도 있다. 어느 쪽인지는 이 호스트에서 확인 안 함 (§9).

---

## 5. 이 프레임워크가 문서화한 학습 진입점 vs 우리 하네스

unsloth 가 내놓는 학습 진입점은 셋이고, 셋 다 **HF/TRL Trainer 를 쓴다는 전제**다.

### (a) `for_training()` — 학습 모드 전환 훅

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:2279-2316`

```python
    def for_training(model, use_gradient_checkpointing = True):
        if not hasattr(model, "parameters"):
            raise TypeError(
                "Unsloth: I think you're passing a tokenizer, not the model to for_training!"
            )

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora

        def _for_training(m):
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(m, "training"):
                m.training = True
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"):
                m._saved_temp_tokenizer.padding_side = "right"
            # Set a flag for generation!
            if hasattr(m, "_flag_for_generation"):
                try:
                    # Weirdly sometimes cannot succeed so do a try except
                    del m._flag_for_generation
                except:
                    pass

        m = model
        while hasattr(m, "model"):
            _for_training(m)
            m = m.model
        _for_training(m)
        model.train()  # to turn on training on modules deeper in
```

`for_training` 은 `from_pretrained` 가 모델에 partial 로 붙여 준다 (`vision.py:2072-2078`),
그러나 **`from_pretrained` 가 자동으로 호출하지는 않는다**. 반대로 `for_inference()` 는
전역 env 두 개를 건드린다 (`vision.py:2269-2272`):

```python
        # Must disable returning hidden states in the case for GRPO
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        # Must enable returning logits
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
```

### (b) 자체 Trainer

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/trainer.py:326-330`, `:434-435`

```python
if Version(transformers_version) > Version("4.45.2"):

    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)
```

```python
class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
```

`transformers 5.5.0 > 4.45.2` 이므로 이 lock 에서 `unsloth_train` 은 **`trainer.train()` 로의 단순 위임**이다.
즉 이 버전에서 학습 루프는 TRL `SFTTrainer` 다.

### (c) 임베딩 전용 진입점 — `FastSentenceTransformer`

이 lock 에는 `sentence-transformers 5.6.1` 이 함께 고정돼 있고, unsloth 는 전용 클래스를 export 한다.

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/__init__.py:22`

```python
from .sentence_transformer import FastSentenceTransformer
```

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/sentence_transformer.py:499`, `:1440-1466`

```python
class FastSentenceTransformer(FastModel):
```

```python
    def from_pretrained(
        model_name,
        max_seq_length = None,
        dtype = None,
        load_in_4bit = False,  # Changed default: 4-bit is slow for encoders
        load_in_8bit = False,
        load_in_16bit = True,  # Changed default: 16-bit is optimal for encoders
        full_finetuning = False,
        token = None,
        device_map = "sequential",
        rope_scaling = None,
        fix_tokenizer = True,
        trust_remote_code = False,
        use_gradient_checkpointing = False,  # Changed default: conflicts with torch.compile
        resize_model_vocab = None,
        revision = None,
        use_exact_model_name = False,
        offload_embedding = False,
        random_state = 3407,
        max_lora_rank = 64,
        disable_log_stats = True,
        qat_scheme = None,
        unsloth_tiled_mlp = False,
        pooling_mode = "mean",
        for_inference = False,
        **kwargs,
    ):
```

`sentence_transformers` 가 없으면 명시적으로 거부한다 (`sentence_transformer.py:1467-1474`):

```python
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import Transformer, Pooling, Normalize
        except ImportError:
            raise ImportError(
                "Unsloth: To use `FastSentenceTransformer`, you must install `sentence-transformers`.\n"
                "Run `pip install sentence-transformers` to install it."
            )
```

기본값이 여기서만 다르다는 점이 중요하다: `load_in_16bit = True`, `use_gradient_checkpointing = False`,
`pooling_mode = "mean"`. 하지만 `full_finetuning = False` 는 그대로이고, 이 클래스도 `FastModel`
상속이므로 §2 의 동결 사슬은 동일하게 적용된다.

### (d) 우리 하네스와의 차이

우리 하네스는 unsloth 를 **적재기로만** 쓴다 (`trainbench/probe/unsloth.py:41-52` 는
`FastVisionModel.from_pretrained` 를 부르고, 학습 스텝은 `trainbench/probe/steps.py:389-438`
`infonce_backward` 가 직접 돈다). 결과적으로 아래가 전부 우회된다:

1. TRL `SFTTrainer` / `unsloth_train` — 따라서 unsloth 가 패치한 `_fast_inner_training_loop` 도,
   gradient-accumulation 수정도 우리 측정에 들어오지 않는다.
2. `for_training()` — 우리는 `model.train()` 만 부른다(`steps.py:410`). 따라서
   `use_gradient_checkpointing` 재적용도, `_flag_for_generation` 제거도 일어나지 않는다.
   단, `post_patch_model` 이 로드 시점에 이미 GC 를 켜 두었으므로 GC 자체는 살아 있다.
3. `UNSLOTH_RETURN_HIDDEN_STATES` / `UNSLOTH_RETURN_LOGITS` 를 설정하는 코드가 우리 쪽에 없다 —
   §6 이 이것이 왜 문제인지 설명한다.

**다만 우회할 수 없는 것이 하나 있다.** `post_patch_model` 은 transformers Trainer 가 패치되어
있지 않으면 로드 자체를 실패시킨다.

`/Users/jwcho/.cache/uv/archive-v0/IQlv5ILnkP_MC_J8/unsloth/models/vision.py:2157-2164`

```python
        from transformers.trainer import Trainer

        if (
            Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop"
            and trust_remote_code == False
        ):
            raise RuntimeError("Unsloth: Unsuccessfully patched inner_training_loop")
```

즉 우리가 Trainer 를 안 쓰더라도 `import unsloth` 가 transformers 보다 먼저 일어나 패치가 성립해야
`from_pretrained` 가 끝까지 간다.

---

## 6. contrastive loss(LM head 없음)에서 융합 CE 패치는 어떻게 되는가

**결론: 죽지 않는다. 조용히 우회되고, 대신 full-vocab logits 를 계산한다.**

### 6-1. 손실 매핑 패치는 `labels` 가 있어야만 발화한다

`/Users/jwcho/.cache/uv/archive-v0/PM921ZbVZCUP68sU/unsloth_zoo/loss_utils.py:137-156`

```python
    # Now patch the losses!
    import transformers.modeling_utils
    LOSS_MAPPING = transformers.loss.loss_utils.LOSS_MAPPING
    # Patch every key still aliased to the stock ForCausalLMLoss. PreTrainedModel
    # resolves loss_type by regex on the class name, so e.g.
    # Qwen3_5ForConditionalGeneration / CsmForConditionalGeneration land on keys
    # pointing at the stock loss; without this sweep they keep the un-patched
    # loss and OOM via logits.float() at large vocab sizes.
    for _key, _fn in list(LOSS_MAPPING.items()):
        if getattr(_fn, "__name__", "") == "ForCausalLMLoss":
            LOSS_MAPPING[_key] = UnslothForCausalLMLoss
```

그리고 그 `UnslothForCausalLMLoss` 자체가 `labels is None` 이면 즉시 `None` 을 돌린다
(`loss_utils.py:113-116`):

```python
    def UnslothForCausalLMLoss(
        logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
    ):
        if labels is None: return None
```

InfoNCE 는 `labels` 를 넘기지 않는다. 따라서 이 패치는 **호출 자체가 되지 않는다**. 예외도 없다.

### 6-2. 컴파일러가 rewrite 한 forward 의 분기 순서가 진짜 답이다

`unsloth_compile_transformers` 는 모델 소스의 forward 를 문자열 치환으로 다시 쓴다
(`loader.py:1674-1690`). 치환 템플릿의 분기 순서는 다음과 같다.

`/Users/jwcho/.cache/uv/archive-v0/PM921ZbVZCUP68sU/unsloth_zoo/compiler.py:1794-1795`, `:1818-1826`

```python
NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
```

```python
requires_grad_ = self.lm_head.weight.requires_grad
requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32

if RETURN_HIDDEN_STATES:
    logits = hidden_states\\1
elif labels is None:
    __DYNAMO__RECOMPILING__
    logits = self.lm_head(hidden_states\\1)
elif ((\\2) == () and (\\3) == ()) and (UNSLOTH_ENABLE_CCE) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
    loss = fused_linear_cross_entropy(
```

(같은 상수 정의가 세 벌 있다: `compiler.py:1794-1795`, `:1869-1870`, `:1983-1984` — 세 가지 패턴용.)

**읽는 법 (adapters 레인 핵심):**

- `labels is None` 분기가 융합 CE 분기들보다 **앞에** 있다. contrastive 스텝에서 융합 CE
  (`fused_linear_cross_entropy` / `unsloth_fused_ce_loss`) 는 절대 실행되지 않는다.
- 그 분기가 하는 일은 `logits = self.lm_head(hidden_states)` — **full vocab matmul 을 그대로 돈다.**
  즉 unsloth 의 대표 절감(로짓 미materialize)은 우리 축에서 0 이고, 오히려 InfoNCE 에 쓰지도 않을
  `[B, T, V]` 텐서를 매 스텝 만든다.
- 유일한 탈출구는 `UNSLOTH_RETURN_HIDDEN_STATES=1` 이다. 그러면 `logits` 필드에 hidden states 가
  담겨 나온다. unsloth 자신도 GRPO 에서 이 방식을 쓴다
  (`unsloth_zoo/rl_replacements.py:817`, `:1106`, `:1317`; `unsloth/models/rl_replacements.py:1234` 등).

`fused_linear_cross_entropy` / `unsloth_fused_ce_loss` 는 둘 다 `lm_head.weight` 를 필수 인자로 받는다
(`loss_utils.py:180-189`, `fused_losses/cross_entropy_loss.py:535-549`), 그러니 LM head 가 없는 모델에서
호출되면 죽는다 — 그러나 위 분기 순서 때문에 호출 지점에 도달하지 않는다.

### 6-3. 우리 하네스에서 지금 무슨 일이 일어나는가 — 저장소와의 불일치

`trainbench/probe/steps.py:190-195`

```python
    output = model(**batch, output_hidden_states=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(output, "hidden_states", None)
        hidden = hidden[-1] if hidden else output[0]
    return last_token_pool(hidden, batch["attention_mask"], padding_side=padding_side)
```

`*ForCausalLM` / `*ForConditionalGeneration` 출력에는 `last_hidden_state` 가 없고,
`output_hidden_states=False` 라 `hidden_states` 도 `None` 이다. 그러면 `output[0]` 이 쓰이는데,
transformers 5.5.0 의 `ModelOutput` 은 `None` 이 아닌 필드만 튜플로 만든다:

`/Users/jwcho/.cache/uv/archive-v0/plcyRhzg-LE7LDvn/transformers/utils/generic.py:461-466`, `:488-492`

```python
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]
```

```python
    def to_tuple(self) -> tuple:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
```

`labels` 를 안 넘겼으므로 `loss=None` 이고, 첫 비-None 필드는 `logits` 다. 따라서 현재 코드는
`UNSLOTH_RETURN_HIDDEN_STATES` 를 켜지 않는 한 **어휘 차원 위에서 풀링해 InfoNCE 를 계산한다**
(`[B, V]`, V ≈ 15만). `UNSLOTH_RETURN_HIDDEN_STATES=1` 이면 같은 `logits` 필드에 hidden states 가
담기므로 같은 코드가 올바른 것을 집는다.

`trainbench/probe/unsloth.py:1-9` 의 docstring 은 "contrastive loss ... bypasses the fused
cross-entropy its speedups largely come from" 이라고 이미 적고 있다. 소스가 그 문장을 확인해 준다.
다만 **어느 텐서를 풀링하는가**는 그 docstring 에도, 저장소 어디에도 적혀 있지 않다.

---

## 7. adapters 레인용 실행 요약

| 질문 | 핀된 소스가 말하는 것 | 근거 |
|---|---|---|
| `full_finetuning` 기본값 | `False`, 세 로더 전부 | `loader.py:1055`, `loader.py:345`, `vision.py:817` |
| `load_in_4bit` 기본값 | `True` (!) — 명시하지 않으면 QLoRA 로 적재된다 | `loader.py:1052`, `vision.py:814` |
| LoRA 없이 기본값으로 적재하면 | 전 파라미터 `requires_grad=False` | `training_utils.py:396, 412` |
| 그런데 backward 는 통과 | 임베딩 출력에 `requires_grad_(True)` 훅 | `training_utils.py:479-480`, `modeling_utils.py:2133-2134` |
| processor `padding_side` | 무조건 `"left"` 로 강제됨 | `vision.py:1716-1718` 외 3계층 |
| 융합 CE | contrastive 에서 우회 (죽지 않음), 대신 full lm_head matmul | `compiler.py:1821-1825` |
| 임베딩용 hidden states | `UNSLOTH_RETURN_HIDDEN_STATES=1` 이 유일한 스위치 | `compiler.py:1795, 1821-1822` |
| `for_inference()` 부작용 | 그 env 를 `0` 으로 되돌리고 `UNSLOTH_RETURN_LOGITS=1` 로 설정 | `vision.py:2269-2272` |
| 임베딩 전용 진입점 존재 | `FastSentenceTransformer` (`sentence-transformers 5.6.1` 동반 고정) | `models/__init__.py:22`, `sentence_transformer.py:499` |
| 학습 루프 | 이 lock 에서 `unsloth_train` 은 `trainer.train()` 위임 (TRL `SFTTrainer`) | `trainer.py:326-330`, `:434` |
| 로드 전제조건 | `import unsloth` 가 Trainer 패치를 성립시켜야 로드가 끝난다 | `vision.py:2157-2163` |

---

## 8. 이 브리프가 저장소 문서와 어긋나는 곳

없다. `AGENTS.md:162-169` 와 `trainbench/probe/unsloth.py:33-40` 이 기술한 내용은 핀된 소스와
줄 단위로 일치했다 (`vision.py:1164-1187`, `:2094-2129`, `training_utils.py:383` — 프로브 주석이
인용한 바로 그 줄들이다).

새로 추가되는 것은 §6-3 이다: 저장소 어디에도 "unsloth 경로에서 `encode()` 가 무엇을 풀링하는가"가
적혀 있지 않고, 소스는 그것이 `UNSLOTH_RETURN_HIDDEN_STATES` 에 달려 있다고 말한다. 이는 모순이
아니라 **빈칸**이다.

---

## 9. 이 호스트에서 확정하지 못한 것

파드/이미지가 답해야 하는 질문. 추측은 적지 않는다.

1. `torch 2.11.0+cu130` 의 소스를 열지 못했다. 이 호스트 캐시에는 macOS arm64 `torch-2.11.0` 만 있고
   `+cu130` 빌드가 없다. torch 에 대한 모든 주장은 lock 이 적은 버전 문자열 하나뿐이다.
2. `unsloth_compile_transformers` 의 문자열 rewrite 가 **Qwen3-VL-Embedding-2B / Qwen3.5-0.8B /
   gemma-4-E2B 각각의 forward 에 실제로 매칭되는지** 확인 안 함. §6-2 의 분기는 rewrite 가 성공한
   경우에만 존재한다. 매칭 실패 시 stock transformers forward 가 남는데, 그때 `output[0]` 이
   무엇인지는 모델마다 다르다.
3. `UNSLOTH_RETURN_HIDDEN_STATES=1` 로 적재한 뒤 forward 가 돌려주는 텐서의 shape 가 `[B, T, H]` 인지
   확인 안 함. 코드는 `logits = hidden_states` 라고만 적혀 있고 slice 여부(`hidden_states[:, slice_indices, :]`)는
   패턴에 따라 다르다.
4. `from_pretrained` 가 돌려주는 processor 객체와 `model._saved_temp_tokenizer` 가 **같은 객체인지**
   확인 안 함. 다르다면 `align_padding_side` 가 프로세서만 고치고 모델에 붙은 사본은 `"left"` 로 남는다.
5. `FastSentenceTransformer.from_pretrained` 가 세 체크포인트를 받는지 거부하는지 확인 안 함.
   `trainbench/probe/unsloth.py:143-155` 가 `expected_failure=True` 로 묻고 있는 바로 그 질문이며,
   소스만으로는 답이 나오지 않는다 (`_load_modules`, `_read_pooling_mode` 가 런타임 config 를 읽는다).
6. `cut_cross_entropy` import 가 이 이미지에서 성공하는지 확인 안 함. `HAS_CUT_CROSS_ENTROPY` 는
   `torch.cuda.get_device_capability()` 와 triton 버전에 걸린 런타임 판정이다
   (`unsloth_zoo/loss_utils.py:38-49`). contrastive 경로에서는 어차피 도달하지 않지만,
   비교군(LM 손실) 측정을 붙일 경우 이 값이 축이 된다.
7. `post_patch_model` 의 `use_reentrant = not is_distributed()` 가 단일 파드에서 `True` 로 계산되는지
   확인 안 함 — `True` 여야만 §3 의 `enable_input_require_grads` 훅이 실제로 걸린다.
8. `full_finetuning=True` 로 적재했을 때 `train_lm_head=True` 가 LM head 없는 임베딩 구성에서
   무엇을 하는지 확인 안 함 (`vision.py:2129`).
