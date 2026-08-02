# axolotl 0.18.0 — 핀 소스 리서치 브리프

소비 레인: **adapters**, **axes**.
작성 근거: `AGENTS.md` "Read the pinned source before asserting framework behaviour".
모든 주장은 아래 "핀 해석" 절이 확정한 경로 하위의 원문 인용으로만 뒷받침한다.

---

## 0. 핀 해석 — 무엇을 열었는가

`envs/axolotl/uv.lock` 에서 읽은 값 (기억 아님):

| 패키지 | lock 줄 | 버전 | 상태 | 경로 |
|---|---|---|---|---|
| `axolotl` | 251-253 | `0.18.0` | on-disk | `/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS` |
| `accelerate` | 21-23 | `1.13.0` | on-disk | `/Users/jwcho/.cache/uv/archive-v0/fEtyCRRqLx-Vto5O` |
| `transformers` | 3117-3119 | `5.14.1` | on-disk | `/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti` |
| `flash-linear-attention` | 860-862 | `0.4.1` | downloaded (sha 일치) | `/private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/pins/flash-linear-attention-0.4.1` (세션 스크래치, 휘발성) |
| `fla-core` | 847-849 | `0.4.1` | downloaded (sha 일치) | `/private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/pins/fla-core-0.4.1` (세션 스크래치, 휘발성) |

디코이 실측:

- `accelerate` 는 캐시에 `1.14.0`(`rGEQvWJWDPxxKQwx`) 이 함께 있다. **핀은 1.13.0.**
- `transformers` 는 캐시에 10개 버전이 함께 있다 (`4.57.6`, `5.3.0`, `5.5.0`, `5.9.0`,
  `5.10.2`, `5.11.0`, `5.12.0`, `5.12.1`, `5.13.0`, `5.13.1`). **핀은 5.14.1.**
- `axolotl` 은 캐시에 `0.18.0` 하나뿐이다.

`flash-linear-attention` / `fla-core` 는 이 macOS 호스트에 없다 (lock 마커가
`sys_platform == 'linux'`). PyPI 에서 lock 이 적은 URL 그대로 받아 sha256 을 대조했고
두 개 다 lock 의 `hash = "sha256:..."` 와 일치한다. 자세한 명령/출력은 구조화 출력의
`resolution[]` 에 있다.

---

## 1. `cli/config.py` 의 실제 호출 순서

`AGENTS.md` 가 적은 `prepare_plugins -> validate_config -> normalize_config` 는 맞다.
다만 **그 사이에 두 호출이 더 있고**, `validate_config` 는 인자를 받는다.

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/cli/config.py:303-328`

```python
    prepare_plugins(cfg)

    if cfg.use_ray:
        # Ray drivers typically have no GPU; defer capability checks to the worker.
        capabilities, env_capabilities = None, None
    else:
        capabilities, env_capabilities = gpu_capabilities()

    cfg = validate_config(
        cfg,
        capabilities=capabilities,
        env_capabilities=env_capabilities,
    )

    # NOTE(djsaunde): We start outputting to output_dir/debug.log at this point since we
    # have to wait for cfg.output to be resolved. We could call this earlier if we write
    # to a temporary file, and then move it later.
    prepare_debug_log(cfg)
    prepare_optim_env(cfg)
    normalize_config(cfg)
    normalize_cfg_datasets(cfg)
    setup_wandb_env_vars(cfg)
    setup_mlflow_env_vars(cfg)
    setup_comet_env_vars(cfg)
    setup_trackio_env_vars(cfg)
    plugin_set_cfg(cfg)
```

주의할 점 셋:

1. **`validate_config` 는 `cfg` 를 되돌려 준다** (`cfg = validate_config(...)`). 제자리
   변형이 아니다. 반환값을 버리면 검증은 아무 일도 하지 않은 것과 같다.
   `trainbench/probe/axolotl.py:73` 은 `cfg = validate_config(cfg)` 로 이미 맞다.
2. **`gpu_capabilities()` 가 `validate_config` 앞에 온다.** GPU가 없으면
   `compute_capability=None`, `bf16=False` 가 들어가고 그 값으로 검증되는 스키마가
   `AxolotlConfigWCapabilities` 로 바뀐다 (§3). 우리 프로브는 capabilities 를 넘기지
   않으므로 `AxolotlInputConfig` 쪽으로 간다 — 상류 CLI 와 다른 스키마다.
3. `prepare_optim_env(cfg)` 가 `normalize_config` **앞**에 있다. 옵티마이저 관련
   환경변수를 세우는 자리이며, 우리 프로브는 이것을 호출하지 않는다.

`load_cfg` 전체는 `@send_errors` 로 감싸져 있다 (232행) — 텔레메트리 데코레이터다.

Ray 경로만 순서가 다르다.
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/cli/train.py:168-186`

```python
    capabilities, env_capabilities = gpu_capabilities_fn()
    cfg = validate_config_fn(
        cfg,
        capabilities=capabilities,
        env_capabilities=env_capabilities,
    )

    # Derive here (not in controller normalize_config) so the worker's
    # validate_config above doesn't see both set and trip check_gas_bsz.
    cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
        cfg.batch_size // cfg.micro_batch_size
    )
    cfg.batch_size = (
        cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps
    )

    prepare_optim_env_fn(cfg)
    normalize_config_fn(cfg)
    resolve_dtype_fn(cfg)
```

---

## 2. `normalize_config` 가 이미 채워졌다고 전제하는 키들

### 2.1 `//` 가 터지는 정확한 자리

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/config/__init__.py:199-213`

```python
def normalize_config(cfg):
    # setup some derived config / hyperparams
    if not cfg.use_ray:
        cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
            cfg.batch_size // cfg.micro_batch_size
        )
        cfg.batch_size = (
            cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps
        )
    if cfg.eval_batch_size is None:
        cfg.eval_batch_size = cfg.micro_batch_size
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.eval_table_size = cfg.eval_table_size or 0
    cfg.eval_max_new_tokens = cfg.eval_max_new_tokens or 128
```

1차 캠페인의 `TypeError: unsupported operand type(s) for //: 'NoneType' and 'NoneType'`
는 202-204행이다. `or` 는 **왼쪽이 falsy 일 때만** 오른쪽을 평가하므로,
`gradient_accumulation_steps` 가 채워져 있으면 `//` 는 아예 실행되지 않는다.
검증을 건너뛴 raw `DictDefault` 는 두 키가 모두 `None` 이라 즉시 터진다.

### 2.2 검증 뒤에도 `None` 으로 남을 수 있는 키가 있다

`validate_config` 의 반환은 `model_dump(exclude_none=True)` 이다 — **기본값이 `None`
인 필드는 dict 에서 지워진다.** `DictDefault.__missing__` 이 다시 `None` 을 주므로
"검증했다 = 모든 키가 채워졌다"는 참이 아니다.

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/config/__init__.py:481-500`

```python
        return DictDefault(
            dict(
                _model_with_inherited_default_fallback(
                    AxolotlConfigWCapabilities,
                    {
                        **cfg.to_dict(),
                        "capabilities": capabilities,
                        "env_capabilities": env_capabilities,
                    },
                ).model_dump(exclude_none=True)
            )
        )

    return DictDefault(
        dict(
            _model_with_inherited_default_fallback(
                AxolotlInputConfig, cfg.to_dict()
            ).model_dump(exclude_none=True)
        )
    )
```

`batch_size` 의 스키마 기본값이 `None` (§3) 이므로 검증 뒤에도 `cfg.batch_size` 는
없다. 그럼에도 `//` 가 안 터지는 이유는 `gradient_accumulation_steps` 기본값이 `1`
이라 `or` 가 단락되기 때문이다. **`gradient_accumulation_steps: null` 을 명시하면
`exclude_none` 이 그것도 지우고 `None // 1` 로 다시 터진다.** adapters 레인은
이 키를 절대 `None`/`0` 으로 넘기면 안 된다.

`context_parallel_size` 는 다르다. 기본값은 `None` 이지만 `after` 검증기가 1을 넣는다.
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/validation.py:1634-1641`

```python
    def check_context_parallel_size(self):
        if self.sequence_parallel_degree and not self.context_parallel_size:
            LOG.warning(
                "`sequence_parallel_degree` is deprecated, use `context_parallel_size`"
            )
            self.context_parallel_size = self.sequence_parallel_degree
        if not self.context_parallel_size:
            self.context_parallel_size = 1
```

즉 `loaders/patch_manager.py:336` 의 `self.cfg.context_parallel_size > 1` 은 검증을
거친 cfg 에서만 안전하다. `trainbench/probe/axolotl.py:64-71` 의 주석이 적은 내용은
이 지점에서 정확하다.

### 2.3 `validate_config` 가 기본값을 주는 범위

pydantic 모델 인스턴스화가 곧 기본값 주입이다. 범위를 정하는 것은 두 가지다.

- `exclude_none=True` — 기본값이 `None` 인 필드는 결과 dict 에서 사라진다.
- 누락 필드 폴백. `/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/config/__init__.py:98-119`

```python
def _model_with_inherited_default_fallback(model_cls, data):
    try:
        return model_cls(**data)
    except ValidationError as exc:
        missing_fields = {
            _field_name_for_missing_loc(model_cls, err["loc"][0])
            for err in exc.errors()
            if err.get("type") == "missing" and len(err.get("loc", ())) == 1
        }
        if not missing_fields:
            raise

        data_with_defaults = dict(data)
        for field_name in missing_fields:
            if field_name in data_with_defaults:
                continue
            default = _field_default_from_mro(model_cls, field_name)
            if _is_pydantic_undefined(default):
                raise
            data_with_defaults[field_name] = default

        return _model_validate_with_field_names(model_cls, data_with_defaults)
```

**진짜로 기본값이 없는 필드(= MRO 어디에도 default 가 없는 필드)만 `raise` 로 남는다.**
그 목록이 §3 이다.

### 2.4 `normalize_config` 가 그 밖에 요구하는 것

`normalize_config` 는 **네트워크를 탄다.** 267행에서 `load_model_config(cfg)` 를 부르고,
그 반환에서 `model_type` 을 읽어 `is_multimodal` / `model_config_type` 을 정한다.
`validate_config` 만 부르는 경로와 달리 여기서부터는 체크포인트가 필요하다.

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/config/__init__.py:259-286`

```python
    if not cfg.base_model_config:
        cfg.base_model_config = cfg.base_model

    # Apply pre-config load patches (e.g., for Kimi Linear remote code patching)
    from axolotl.loaders.patch_manager import PatchManager

    PatchManager.apply_pre_config_load_patches(cfg)

    model_config = load_model_config(cfg)

    cfg.tokenizer_config = (
        cfg.tokenizer_config or cfg.base_model_config or cfg.base_model
    )

    model_support = get_model_support(getattr(model_config, "model_type", None))

    cfg.is_multimodal = (
        (model_support is not None and model_support.is_multimodal)
        or hasattr(model_config, "model_type")
        and model_config.model_type in MULTIMODAL_AUTO_MODEL_MAPPING
        or any(
            multimodal_name in cfg.base_model.lower()
            for multimodal_name in [
                "pixtral",
            ]
        )
        or cfg.is_multimodal
    )
```

그리고 마지막 줄에서 `cfg.device` 를 읽는다 (`choose_device` 가 220행에서 세운 값).
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/config/__init__.py:417`

```python
    log_gpu_memory_usage(LOG, "baseline", cfg.device)
```

gemma4 전용 분기도 여기 있다 (`utils/config/__init__.py:385-415`). **무조건이 아니라
조건부다** (2026-08-02 감사가 잡고 원문으로 재확인함):

```python
    if cfg.model_config_type in ("gemma4", "gemma4_unified"):
        if cfg.gradient_checkpointing:                                    # 391
            ...
            if cfg.gradient_checkpointing_kwargs.get("use_reentrant") is not False:
                cfg.gradient_checkpointing_kwargs["use_reentrant"] = False   # 399
        if cfg.ddp and cfg.ddp_find_unused_parameters is None:            # 400
            if cfg.activation_offloading is True:
                ...                                                        # 건너뛴다
            else:
                cfg.ddp_find_unused_parameters = True                      # 415
```

- `use_reentrant=False` 는 **`cfg.gradient_checkpointing` 이 참일 때만** 들어간다
- `ddp_find_unused_parameters=True` 는 **`cfg.ddp` 이고 그 값이 `None` 이고
  `activation_offloading` 이 `True` 가 아닐 때만** 들어간다.
  `activation_offloading=True` 면 DDP 쪽은 아예 건너뛴다

**그러므로 "gemma-4 칸에서 기울기 체크포인팅 축을 프레임워크가 덮어쓴다"는 조건부 사실이다.**
이 저장소의 프로브 경로는 `ddp` 를 세우지 않고 `gradient_checkpointing` 도 축이 켜졌을 때만
참이므로, **두 분기가 실제로 도는지는 그 런의 config 에 달려 있다.**
axes/adapters 레인은 덮어쓰기를 전제하지 말고 자기 런의 config 로 판단한다.

---

## 3. 스키마가 요구하는 최소 키 집합

`AxolotlInputConfig` 는 19개 믹스인의 합성이다.
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/config.py:189-212`

```python
class AxolotlInputConfig(
    ModelInputConfig,
    ModelOutputConfig,
    LoraConfig,
    ReLoRAConfig,
    JaggedLRConfig,
    HyperparametersConfig,
    WandbConfig,
    MLFlowConfig,
    CometConfig,
    TrackioConfig,
    OpenTelemetryConfig,
    LISAConfig,
    GradioConfig,
    RayConfig,
    MultiModalConfig,
    RemappedParameters,
    DeprecatedParameters,
    ValidationMixin,
    BaseModel,
):
    """Wrapper of all config options."""

    model_config = {"populate_by_name": True}
```

### 3.1 default 가 없어 반드시 넘겨야 하는 필드는 **둘**

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/model.py:17-21`

```python
    base_model: str = Field(
        json_schema_extra={
            "description": "This is the huggingface model that contains *.pt, *.safetensors, or *.bin files. This can also be a relative path to a model on disk"
        }
    )
```

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/training.py:116`

```python
    learning_rate: str | float
```

### 3.2 배치 관련 필드는 전부 default 가 있다

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/training.py:71-94`

```python
    gradient_accumulation_steps: int | None = Field(
        default=1,
        json_schema_extra={
            "description": "If greater than 1, backpropagation will be skipped and the gradients will be accumulated for the given number of steps."
        },
    )
    micro_batch_size: int | None = Field(
        default=1,
        json_schema_extra={
            "description": "The number of samples to include in each batch. This is the number of samples sent to each GPU. Batch size per gpu = micro_batch_size * gradient_accumulation_steps"
        },
    )
    batch_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Total batch size, we do not recommended setting this manually"
        },
    )
    eval_batch_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "per gpu micro batch size for evals, defaults to value of micro_batch_size"
        },
    )
```

**`micro_batch_size` 와 `gradient_accumulation_steps` 는 "스키마가 요구하는 키"가
아니다** — 둘 다 default 1 이다. 우리 프로브 주석 (`trainbench/probe/axolotl.py:51-57`)
이 "네 키를 axolotl 스키마가 요구한다"고 적은 것은 이 판본에서는 정확하지 않다.
넘겨야 하는 이유는 다른 데 있다: 우리 study 의 배치 폭을 axolotl 이 알아야 하기 때문이고,
그것은 여전히 옳은 선택이다.

### 3.3 `batch_size` 와 `gradient_accumulation_steps` 는 동시에 못 준다

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/validation.py:272-279`

```python
    @model_validator(mode="before")
    @classmethod
    def check_gas_bsz(cls, data):
        if data.get("gradient_accumulation_steps") and data.get("batch_size"):
            raise ValueError(
                "please set only one of gradient_accumulation_steps or batch_size"
            )
        return data
```

### 3.4 `datasets` 는 필드로는 optional, 검증기로는 필수

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/config.py:339-356`

```python
    datasets: (
        Annotated[
            list[
                SFTDataset
                | DPODataset
                | KTODataset
                | StepwiseSupervisedDataset
                | SyntheticDataset
            ],
            MinLen(1),
        ]
        | None
    ) = Field(
        default=None,
        json_schema_extra={
            "description": "A list of one or more datasets to finetune the model with"
        },
    )
```

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/validation.py:79-84`

```python
    @model_validator(mode="before")
    @classmethod
    def check_dataset_or_pretraining_dataset(cls, data):
        if data.get("datasets") is None and data.get("pretraining_dataset") is None:
            raise ValueError("either datasets or pretraining_dataset is required")
        return data
```

`MinLen(1)` 이므로 빈 리스트도 거부된다. 우리 프로브가 한 항목짜리 리스트를 주는 것은 맞다.

### 3.5 capabilities 스키마를 쓰면 두 키가 더 필수

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/config.py:1754-1758`

```python
class AxolotlConfigWCapabilities(AxolotlInputConfig):
    """Wrapper to valdiate GPU capabilities with the configured options"""

    capabilities: GPUCapabilities
    env_capabilities: EnvCapabilities
```

### 3.6 최소 cfg (이 판본 기준)

```yaml
base_model: <hf id>        # 필수, default 없음
learning_rate: <float>     # 필수, default 없음
datasets: [{...}]          # 검증기가 요구, MinLen(1)
```

나머지는 전부 기본값이 있다. `gradient_accumulation_steps` 와 `batch_size` 중
**하나만** 준다.

---

## 4. dtype — 2차 캠페인 3칸이 backward 에서 죽은 원인

### 4.1 모델 전체의 dtype 을 정하는 자리

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/config/__init__.py:191-196`

```python
    if cfg.bf16 or cfg.bfloat16:
        cfg.torch_dtype = torch.bfloat16
    elif cfg.load_in_8bit or cfg.fp16 or cfg.float16:
        cfg.torch_dtype = torch.float16
    else:
        cfg.torch_dtype = torch.float32
```

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/loaders/model.py:611-612`

```python
        self.model_kwargs["torch_dtype"] = self.cfg.torch_dtype
        self.model_kwargs["dtype"] = self.cfg.torch_dtype
```

여기까지는 우리가 요청한 `bf16: True` 대로다.

### 4.2 그 다음, 임베딩과 norm 을 **fp32 로 올린다**

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/loaders/model.py:422-477`

```python
    def _configure_embedding_dtypes(self):
        """Configure embedding module dtypes."""
        # Get embedding modules
        embedding_modules = get_linear_embedding_layers(self.cfg.model_config_type)

        # Initial dtype conversion
        if not self.is_fsdp_enabled:
            # We don't run this during FSDP because this will leave mixed and bfloat16
            # dtypes in the model which FSDP doesn't like
            if self.cfg.load_in_4bit and self.cfg.embeddings_skip_upcast:
                embedding_modules = []
            self._convert_embedding_modules_dtype(
                embedding_modules,
                dist_dtype=torch.float32,
                before_kbit_train_or_finetune=True,
            )

        # Handle DeepSpeed Zero3
        if (
            is_deepspeed_zero3_enabled()
            or os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3"
        ):
            self._set_z3_leaf_modules()

        # Apply gradient checkpointing if needed
        needs_fa2_dtype = self.cfg.adapter or self.is_fsdp_enabled
        if self.cfg.adapter in ["lora", "qlora"]:
            needs_fa2_dtype = True
            if self.cfg.gradient_checkpointing:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=self.cfg.gradient_checkpointing_kwargs
                )

        self._prepare_model_for_quantization()

        # Convert dtypes if needed
        should_convert = (
            # LlamaRMSNorm layers are in fp32 after kbit_training or full finetune, so
            # we need to convert them back to fp16/bf16 for flash-attn compatibility.
            (
                (needs_fa2_dtype or self.cfg.attn_needs_dtype_cast)
                and not self.is_qlora_and_fsdp_enabled
            )
            or (
                # CCE requires embedding layers to be in fp16/bf16 for backward pass
                self.cfg.cut_cross_entropy
            )
        )

        if should_convert:
            LOG.info("Converting modules to %s", self.cfg.torch_dtype)
            self._convert_embedding_modules_dtype(
                embedding_modules=embedding_modules,
                dist_dtype=self.cfg.torch_dtype,
                before_kbit_train_or_finetune=False,
            )
```

무엇이 올라가는지는 이 루프가 정한다.
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/loaders/model.py:1025-1047`

```python
    def _convert_embedding_modules_dtype(
        self,
        embedding_modules: list[str],
        dist_dtype: torch.dtype,
        before_kbit_train_or_finetune: bool,
    ):
        dest = {"dtype": dist_dtype}
        if self.cfg.lora_on_cpu:
            dest["device"] = "cpu"
        fp32_norm_patterns = get_fp32_norm_patterns(self.cfg)
        for name, module in self.model.named_modules():
            if fp32_norm_patterns and _matches_norm_class(module, fp32_norm_patterns):
                module.to(torch.float32)
            elif "norm" in name:
                module.to(dist_dtype)
            if before_kbit_train_or_finetune:
                if name.endswith(".gate"):
                    module.to(dist_dtype)
                if self.model_config.model_type == "btlm":
                    # don't upcast lm_head for btlm
                    continue
            if any(m in name for m in embedding_modules) and hasattr(module, "weight"):
                module.to(**dest)
```

첫 호출은 `dist_dtype=torch.float32` 이므로 **이름에 `norm` 이 든 모든 모듈**과
**임베딩 모듈**이 fp32 로 간다. 임베딩 이름은:

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/loaders/utils.py:231-239`

```python
def get_linear_embedding_layers(model_type: str) -> list[str]:
    """Returns layer names of linear embeddings needed for LoRA based on model type."""
    if model_type == "gpt_neox":
        return ["embed_in", "embed_out"]
    if model_type == "falcon":
        return ["word_embeddings", "lm_head"]
    if model_type == "nemotron_h":
        return ["embeddings", "lm_head"]
    return ["embed_tokens", "lm_head"]
```

### 4.3 되돌리는 분기의 조건 — 정확히

`should_convert` 가 참이 되는 경우는 넷이고, **full finetune + sdpa/eager 는 어느
쪽에도 해당하지 않는다.**

| 조건 | 소스 | 우리 full-FT 칸에서의 값 |
|---|---|---|
| `cfg.adapter` (truthy) | `model.py:447` | `None` — `trainbench/probe/axolotl.py:46` 이 `peft.mode == "full"` 일 때 `None` 을 넣는다 |
| `is_fsdp_enabled` | `model.py:158-160` | `False` (cfg 에 `fsdp`/`fsdp_config` 없음) |
| `cfg.attn_needs_dtype_cast` | `config.py:1500-1503` | `False` (아래) |
| `cfg.cut_cross_entropy` | `model.py:467` | `None` — 플러그인 키이며 base 스키마에 없다 |

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/loaders/model.py:157-160`

```python
    @property
    def is_fsdp_enabled(self):
        """Property that determines if FSDP is enabled."""
        return self.cfg.fsdp_config is not None or self.cfg.fsdp is not None
```

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/config.py:1498-1503`

```python
    @computed_field  # type: ignore[misc]
    @property
    def attn_needs_dtype_cast(self) -> bool:
        if self.attn_implementation is None:
            return False
        return self.attn_implementation not in ATTN_IMPLS_WITHOUT_DTYPE_CAST
```

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/enums.py:158-159`

```python
# Backends for which embeddings stay in fp32. Everything else needs fp16/bf16.
ATTN_IMPLS_WITHOUT_DTYPE_CAST = frozenset({"eager", "sdpa"})
```

`attn_implementation` 의 기본값:
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/utils/schemas/config.py:839-849`

```python
    attn_implementation: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Attention backend. Canonical values: eager, sdpa, flash_attention_2, "
                "flash_attention_3, flex_attention, xformers, sage, fp8. Hub-kernel "
                "paths (e.g. kernels-community/flash-attn3) are also accepted and passed "
                "through to transformers."
            )
        },
    )
```

**결론.** `bf16: True` + `adapter: None` + FSDP 없음 + CCE 없음 + attention 이
`None`/`sdpa`/`eager` 인 우리 프로브 cfg 에서, axolotl 은 `embed_tokens` / `lm_head` /
이름에 `norm` 이 든 모든 모듈을 **의도적으로 fp32 로 남기고** 나머지를 bf16 으로 둔다.
이것이 `RuntimeError: expected mat1 and mat2 to have the same dtype, but got:
float != c10::BFloat16` 이고, 같은 실행의 `axes_verified` 가 `mixed(bf16,fp32)` 를
기록한 것과 같은 사실의 두 면이다. `docs/support-matrix.md:955` 가
"인과는 미확인"이라 적은 부분은 이 절이 채운다.

LoRA 칸이 살아남는 이유도 같은 자리다 — `cfg.adapter` 가 truthy 이므로
`needs_fa2_dtype` 이 참이 되고 되돌리는 분기가 돈다. **2차 캠페인에서 죽은 3칸은
full finetune 3칸이라는 예측이 이 소스에서 나온다** (확인 안 함 — 실패 기록은
모델별로만 남아 있고 peft.mode 별로는 남아 있지 않다).

---

## 5. 상류가 같은 모델로 문제 없이 도는 이유 — Trainer 의 autocast

axolotl 은 이 fp32 잔재를 스스로 고치지 않는다. HF Trainer 가 forward 를 autocast 로
감싸기 때문에 fp32 가중치의 matmul 이 자동으로 bf16 으로 캐스팅된다. 사슬 전체를
핀 소스에서 확인했다.

**(1) axolotl 이 `bf16` 을 TrainingArguments 로 넘긴다.**
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/core/builders/base.py:270-278`

```python
    def _configure_precision_settings(self, training_args_kwargs: dict):
        training_args_kwargs["fp16"] = (self.cfg.fp16 and not self.cfg.bf16) or False
        training_args_kwargs["tf32"] = True if self.cfg.tf32 is True else False
        if self.cfg.bf16 == "full":
            training_args_kwargs["bf16_full_eval"] = True
        else:
            bf16 = self.cfg.bf16 or self.cfg.bfloat16
            bf16 = bf16 if bf16 is not None else False
            training_args_kwargs["bf16"] = bf16
```

**(2) transformers 가 `mixed_precision="bf16"` 로 옮긴다.**
`/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti/transformers/training_args.py:1553-1559`

```python
        # ── 5. Mixed Precision ──
        # Read from env first; DeepSpeed may override this later
        self.mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
        if self.fp16:
            self.mixed_precision = "fp16"
        elif self.bf16:
            self.mixed_precision = "bf16"
```

`/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti/transformers/trainer.py:706-711`

```python
    def _build_accelerator_args(self, **kwargs) -> dict[str, Any]:
        """Helper method to build accelerator-specific keyword arguments."""
        args = {
            "mixed_precision": self.args.mixed_precision,
            "deepspeed_plugin": self.args.deepspeed_plugin,
        }
```

**(3) accelerate 가 `native_amp` 를 켠다.**
`/Users/jwcho/.cache/uv/archive-v0/fEtyCRRqLx-Vto5O/accelerate/accelerator.py:584-595`

```python
        elif self.state.mixed_precision == "bf16" and self.distributed_type not in (
            DistributedType.DEEPSPEED,
            DistributedType.MEGATRON_LM,
        ):
            if self.device.type in ["cpu", "xpu", "hpu"]:
                self.native_amp = True
            else:
                self.native_amp = is_bf16_available(True)
            if not self.native_amp and not is_torch_xla_available():
                raise ValueError("bf16 mixed precision requires PyTorch >= 1.10 and a supported device.")
            if self.native_amp and self.device.type == "mps" and not is_torch_version(">=", "2.6.0"):
                raise ValueError("bf16 mixed precision with MPS device requires a Pytorch >= 2.6.0")
```

**(4) `prepare_model` 이 `model.forward` 자체를 autocast 로 감싼다.**
`/Users/jwcho/.cache/uv/archive-v0/fEtyCRRqLx-Vto5O/accelerate/accelerator.py:1806-1818`

```python
        if self.native_amp:
            model._original_forward = model.forward
            autocast_context = get_mixed_precision_context_manager(self.native_amp, self.autocast_handler)
            # NOTE: MS-AMP adds `__func__` already to `model.forward`, so we should always use `model.forward`
            if self.fp8_backend == FP8BackendType.MSAMP or not hasattr(model.forward, "__func__"):
                model_forward_func = model.forward
                model.forward = convert_outputs_to_fp32(autocast_context(model_forward_func))
            else:
                model_forward_func = model.forward.__func__
                new_forward = autocast_context(model_forward_func)
                model.forward = MethodType(new_forward, model)
                model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
```

**(5) 그 컨텍스트의 정체.**
`/Users/jwcho/.cache/uv/archive-v0/fEtyCRRqLx-Vto5O/accelerate/utils/modeling.py:2064-2088`

```python
    if native_amp:
        device_type = (
            "cuda"
            if (state.distributed_type == DistributedType.XLA and is_torch_xla_available(check_is_gpu=True))
            else state.device.type
        )
        if state.mixed_precision == "fp16":
            return torch.autocast(device_type=device_type, dtype=torch.float16, **autocast_kwargs)
        elif state.mixed_precision in ["bf16", "fp8"] and state.distributed_type in [
            DistributedType.NO,
            DistributedType.MULTI_CPU,
            DistributedType.MULTI_GPU,
            DistributedType.MULTI_MLU,
            DistributedType.MULTI_SDAA,
            DistributedType.MULTI_MUSA,
            DistributedType.MULTI_NPU,
            DistributedType.MULTI_XPU,
            DistributedType.MULTI_HPU,
            DistributedType.MULTI_NEURON,
            DistributedType.FSDP,
            DistributedType.XLA,
        ]:
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16, **autocast_kwargs)
        else:
            return torch.autocast(device_type=device_type, **autocast_kwargs)
```

### 5.1 우리가 감쌀 때 필요한 컨텍스트 — 정확히

```python
torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

상류와 동일하게 하려면 두 가지가 더 붙는다.

- 출력 후처리: accelerate 는 `convert_outputs_to_fp32(...)` 로 감싼다 (1813 / 1818행).
  autocast 안에서 나온 bf16 출력을 fp32 로 되돌리는 래퍼다. InfoNCE 로짓을 fp32 에서
  계산하고 싶다면 이 래핑까지 따라가야 상류와 같은 수치 경로가 된다.
- **감싸는 범위는 forward 뿐이다.** accelerate 는 `model.forward` 만 감싸고
  `loss.backward()` 는 감싸지 않는다. backward 는 forward 가 기록한 autocast 캐스팅
  그래프를 그대로 되짚으므로 별도 컨텍스트가 필요 없다. 우리 손실 계산(InfoNCE)이
  모델 밖에 있다면, 상류와 같은 모양은 **모델 forward 만** autocast 안에 넣는 것이다.
- `cache_enabled` 는 accelerate 가 `AutocastKwargs` 를 안 주면 건드리지 않는다
  (기본 `True`). 우리도 기본값을 쓰면 된다.

### 5.2 axes 레인에 대한 경고 — 저장소의 전제와 충돌한다

`trainbench/axes.py:687-700` 은 이렇게 적혀 있다.

```python
def step_context(config: BenchConfig) -> contextlib.AbstractContextManager:
    """Context wrapping one training step.

    bf16 needs none: the model is already loaded in that dtype, so an autocast
    region would be a second, different answer to the same question. The fp8
    recipes do need one, and refusing here is what keeps a bf16 step from being
    measured under their name.
    """
    if config.precision.name != "bf16":
        raise UnappliedAxis(...)
    return contextlib.nullcontext()
```

**이 전제("bf16 은 이미 그 dtype 으로 적재되어 있다")는 axolotl 0.18.0 에서 거짓이다.**
axolotl 의 `ModelLoader` 는 sdpa/eager + full finetune 에서 임베딩과 norm 을 고의로
fp32 로 남기고, 상류는 그 차이를 autocast 로 흡수한다. axolotl 칸에서 bf16 을
`nullcontext` 로 도는 한 backward 는 계속 죽는다.

더 나아가, 감싸도 `applied` 판정은 통과하지 못한다.
`trainbench/applied.py:786-843` 의 `_capture_precision` 은 **파라미터 dtype 을 읽는다.**
autocast 는 파라미터를 바꾸지 않으므로 axolotl 칸은 감싼 뒤에도 `mixed(bf16,fp32)` 로
읽힌다 (843행: `return "mixed(" + ",".join(sorted(base)) + ")", detail`). 그 함수의
독스트링 자체가 전제를 명시한다 (789-793행):

```
    Read off the parameters because nothing in `axes.py` sets them: the load dtype
    comes from `probe/steps.py::dtype_for`, which returns fp32 on anything that is
    not CUDA. `axes.step_context` is written on the premise that a bf16 run is
    already loaded in bf16 and so needs no autocast region; until this probe
    existed, nothing checked that premise.
```

즉 axolotl 에 대해서는 **선택이 두 갈래**이고, 어느 쪽이든 결정이 필요하다.

1. 프레임워크에 맞춘다 — cfg 에 `attn_implementation` 을 `flash_attention_2` 같은
   `ATTN_IMPLS_WITHOUT_DTYPE_CAST` 밖의 값으로 주면 `attn_needs_dtype_cast` 가 참이
   되어 axolotl 이 스스로 임베딩을 bf16 으로 되돌린다. 그러면 파라미터가 순수 bf16 이
   되고 `_capture_precision` 도 통과한다. 대가: attention 축이 프레임워크마다 다른
   값으로 고정된다 — 벤치마크 축을 훼손한다.
2. 상류와 같게 autocast 로 감싸고, `_capture_precision` 이 **fp32 로 남는 모듈 집합을
   프레임워크 사실로 인정**하도록 계약을 넓힌다. 대가: 계약 변경.

이 브리프는 어느 쪽도 고르지 않는다. 둘 다 소스가 뒷받침하는 선택지라는 것만 적는다.

---

## 6. 문서화된 학습 진입점과 우리 경로의 차이

콘솔 스크립트는 하나다.
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl-0.18.0.dist-info/entry_points.txt`

```
[console_scripts]
axolotl = axolotl.cli.main:main
```

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/cli/main.py:99-106`

```python
def train(
    ctx: click.Context,
    config: str,
    launcher: Literal["accelerate", "torchrun", "python"] = "accelerate",
    cloud: str | None = None,
    sweep: str | None = None,
    **kwargs,
):
```

기본 런처가 `accelerate` 다 — §5 의 autocast 는 그래서 상류에서 항상 켜진다.

경로는 `cli/main.py:train` → `cli/train.py:do_cli` → `load_cfg` → `do_train` →
`axolotl.train:train`. 모델과 트레이너를 만드는 자리:

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/train.py:69-84`

```python
    # Load tokenizer
    LOG.debug(
        f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}",
    )
    tokenizer = load_tokenizer(cfg)

    # Load processor for multimodal models if needed
    processor = None
    if cfg.is_multimodal:
        processor = load_processor(cfg, tokenizer)

    # Load the model
    LOG.debug("Loading model")

    model_loader = ModelLoader(cfg, tokenizer, processor=processor)
    model, peft_config = model_loader.load()
```

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/train.py:584-596`

```python
    # Set up trainer
    trainer = setup_trainer(
        cfg=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        total_num_steps=total_num_steps,
        model_ref=model_ref,
        peft_config=peft_config,
    )
    PLUGIN_MANAGER.post_trainer_create(cfg, trainer)
```

`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl/train.py:222-227`

```python
        # TODO: disabling for now as not compatible with FSDP2 + torchao low bit optimizers
        # if cfg.bf16:
        #     torch.set_default_dtype(torch.bfloat16)

        LOG.info("Starting trainer...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

**우리 경로와의 차이 — 목록.**

| 상류 | 우리 (`trainbench/probe/axolotl.py`) | 결과 |
|---|---|---|
| `ModelLoader(cfg, tokenizer, processor=processor)` — 멀티모달이면 processor 를 넘긴다 | `ModelLoader(cfg, tokenizer)` (76행), processor 없음 | Qwen3-VL / gemma-4 칸에서 상류와 다른 로더 인자 |
| `is_multimodal` 은 `normalize_config` 가 정한다 (`utils/config/__init__.py:275-286`) | 우리도 `normalize_config` 를 부르므로 값은 세워진다 | 하지만 processor 를 안 만든다 |
| `setup_trainer(...)` → HF Trainer 서브클래스 | 트레이너를 만들지 않고 모델을 직접 쓴다 | **autocast 가 없다** (§5) |
| `accelerate` 런처가 `Accelerator` 를 만든다 | `Accelerator` 없음 | `native_amp` 경로 전체가 빠진다 |
| `load_cfg` 가 `prepare_optim_env` 를 부른다 | 부르지 않는다 | 옵티마이저 env 미설정 (우리는 옵티마이저를 axolotl 로 안 만드니 현재는 무해) |
| `gpu_capabilities()` 를 넘겨 `AxolotlConfigWCapabilities` 로 검증 | `validate_config(cfg)` 만 — `AxolotlInputConfig` | bf16/tf32/fp8 능력 검증기가 안 돈다 (§3.5) |
| `trainer.train()` | `steps.infonce_backward(model, ...)` | 손실이 CLM 이 아니라 InfoNCE. 프레임워크가 본 적 없는 경로 |

`ModelLoader(...).load()` 를 직접 부르는 것 자체는 상류와 같다 — `train.py:83` 이
같은 호출을 한다. 어긋나는 것은 **그 뒤**다.

---

## 7. `flash-linear-attention` — 이 lock 만 0.4.1 이다

`envs/axolotl/uv.lock:859-870`

```
[[package]]
name = "flash-linear-attention"
version = "0.4.1"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "fla-core", marker = "sys_platform == 'linux'" },
    { name = "transformers", marker = "sys_platform == 'linux'" },
]
sdist = { url = "https://files.pythonhosted.org/packages/46/83/7d8ec7ffb5229080b1c9b772338ff588cbd63282ac355ede2a12a6e174a8/flash_linear_attention-0.4.1.tar.gz", hash = "sha256:127ee7273ed15ac17f72bcf4c75e1051719d8fbe0a2d1d047e59406f36d81ee2", size = 158280, upload-time = "2025-12-24T18:07:38.812Z" }
wheels = [
    { url = "https://files.pythonhosted.org/packages/63/d5/6327559a9d5b9243b10c3984f1bcef256ed2ad06d105a3bb8f7b2979659c/flash_linear_attention-0.4.1-py3-none-any.whl", hash = "sha256:d18bdfe9d1f4b424676444eac9d50fb8433b70e5d4e0e0878b20bcbcdbea57ce", size = 287415, upload-time = "2025-12-24T18:07:35.815Z" },
]
```

여섯 env 실측 (`grep -A2 '^name = "flash-linear-attention"' envs/*/uv.lock`):

| env | flash-linear-attention | fla-core |
|---|---|---|
| **axolotl** | **0.4.1** | **0.4.1** |
| ms-swift | 0.5.2 | 0.5.2 |
| native | 0.5.2 | 0.5.2 |
| sentence-transformers | 0.5.2 | 0.5.2 |
| tevatron | 0.5.2 | 0.5.2 |
| unsloth | 0.5.2 | 0.5.2 |

원인은 axolotl 자신이다.
`/Users/jwcho/.cache/uv/archive-v0/NnUfrC8dsc47HDGS/axolotl-0.18.0.dist-info/METADATA:67-68`

```
Requires-Dist: fla-core==0.4.1; platform_machine != "aarch64"
Requires-Dist: flash-linear-attention==0.4.1; platform_machine != "aarch64"
```

우리 env 선언은 하한이 없다 (`envs/axolotl/uv.lock:3102`: `{ name =
"flash-linear-attention" }`), 반면 native 는 `>=0.5` 를 건다
(`envs/native/uv.lock:1863`). 그래서 axolotl 만 갈라졌다.

### 7.1 이것이 Qwen3.5 수치에 직접 닿는다 — 소스 확인

`/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti/transformers/models/qwen3_5/modeling_qwen3_5.py:73-77`

```python
if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
```

`/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti/transformers/models/qwen3_5/modeling_qwen3_5.py:421-431`

```python
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )
```

`/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti/transformers/utils/import_utils.py:870-872`

```python
def is_flash_linear_attention_available():
    is_available, fla_version = _is_package_available("fla", return_version=True)
    return is_torch_cuda_available() and is_available and version.parse(fla_version) >= version.parse("0.2.2")
```

버전 하한이 0.2.2 이므로 0.4.1 도 0.5.2 도 "사용 가능"이다 — **즉 fallback 이 아니라
서로 다른 커널이 실제로 돈다.** 받아서 확인한 0.4.1 휠에 그 커널이 들어 있다:

`/private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/pins/fla-core-0.4.1/fla/__init__.py:4`

```python
__version__ = '0.4.1'
```

`/private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/pins/fla-core-0.4.1/fla/ops/gated_delta_rule/` 에 `chunk.py`,
`fused_recurrent.py`, `wy_fast.py` 가 있다.

**결론: Qwen3.5 의 gated-delta-rule 선형 어텐션 경로는 axolotl 이미지에서만 다른
커널 구현으로 돈다.** `docs/support-matrix.md:556-562` 가 이미 같은 결론을 적어 두었고,
이 브리프는 그 결론이 실제 dispatch 지점에서 성립함을 추가로 확인한다.

### 7.2 axolotl 이미지가 가진 다른 스택 차이 (같은 종류의 교란)

`grep -A2 '^name = "torch"$' / '^name = "transformers"$' envs/*/uv.lock`:

| env | torch | transformers |
|---|---|---|
| **axolotl** | **2.12.1+cu130** | 5.14.1 |
| ms-swift | 2.13.0+cu130 | **5.12.1** |
| native | 2.13.0+cu130 | 5.14.1 |
| sentence-transformers | 2.13.0+cu130 | 5.14.1 |
| tevatron | 2.13.0+cu130 | 5.14.1 |
| **unsloth** | **2.11.0+cu130** | **5.5.0** |

axolotl 만 torch 2.12.1 이다. AGENTS.md 의 "Record the resolved torch/framework
versions per run" 이 정확히 이 상황을 위한 규칙이다.

---

## 8. adapters 레인에 바로 쓰는 요약

1. 호출 순서는 `prepare_plugins` → (`gpu_capabilities`) → `cfg = validate_config(cfg, ...)`
   → `prepare_optim_env` → `normalize_config` → `normalize_cfg_datasets`. 반환값을 받는다.
2. 최소 cfg 는 `base_model` + `learning_rate` + (`datasets` 또는 `pretraining_dataset`).
   `micro_batch_size`/`gradient_accumulation_steps` 는 default 1 이므로 스키마 필수가 아니다.
3. `gradient_accumulation_steps` 와 `batch_size` 를 **동시에** 주면 검증이 거부한다.
4. `gradient_accumulation_steps` 를 `None`/`0` 으로 주면 `exclude_none` 이 지워
   `normalize_config` 의 `//` 가 다시 터진다.
5. full finetune 칸이 backward 에서 죽는 이유는 axolotl 이 임베딩·norm 을 고의로 fp32 로
   남기기 때문이며, 되돌리는 조건은 `adapter` / FSDP / `attn_needs_dtype_cast` /
   `cut_cross_entropy` 넷뿐이다.
6. 상류는 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 로 forward 를
   감싼다 (accelerate `prepare_model`). 우리 경로에는 그것이 없다.
7. axolotl 이미지의 `flash-linear-attention` 은 0.4.1 로 홀로 다르고, Qwen3.5 의
   `chunk_gated_delta_rule` 이 그 패키지에서 온다.

---

## 9. 이 호스트에서 확정하지 못한 것

파드/이미지가 답해야 하는 질문. 추측하지 않는다.

1. 2차 캠페인에서 dtype 으로 죽은 axolotl 3칸이 `peft.mode == "full"` 칸인지.
   `docs/support-matrix.md:1138-1144` 는 모델 이름만 남기고 peft 모드를 남기지 않았다.
   §4.3 의 예측은 full 3칸이지만, **확인 안 함**.
2. Qwen3-VL-Embedding-2B / gemma-4-E2B 의 `model_type` 에 대해
   `get_linear_embedding_layers` 가 `["embed_tokens", "lm_head"]` 를 돌려주는지.
   기본 분기가 그렇지만 세 모델 중 어느 것도 이 호스트에서 `normalize_config` 를
   돌린 적이 없어 `cfg.model_config_type` 실측값이 없다.
3. `_convert_embedding_modules_dtype` 의 `"norm" in name` 이 이 세 모델에서 정확히 몇
   개 모듈을 fp32 로 올리는지. `axes_verified` 가 `mixed(bf16,fp32)` 로 기록했다는
   사실은 있으나, fp32 텐서 수와 이름은 기록되지 않았다.
4. `attn_implementation: flash_attention_2` 를 cfg 에 주면 axolotl 이 임베딩을 bf16 으로
   되돌리고 backward 가 통과하는지. 소스상 그렇게 되어야 하지만 **한 번도 돌린 적 없다**.
5. forward 만 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 로 감쌌을 때
   `infonce_backward` 가 통과하는지, 그리고 `convert_outputs_to_fp32` 상당 처리 없이
   InfoNCE 로짓이 bf16 으로 나와도 수치가 성립하는지.
6. `ModelLoader` 에 `processor` 를 넘기지 않은 것이 Qwen3-VL / gemma-4 칸에서 실제
   차이를 만드는지. 2차 캠페인은 로더 자체는 통과했으므로 차이가 없었을 수도 있다.
7. `fla` 0.4.1 과 0.5.2 의 `chunk_gated_delta_rule` 이 **얼마나** 다른 throughput 을
   내는지. 두 버전을 나란히 놓고 측정한 적이 없다 — **측정 안 함**.
8. axolotl 의 torch 2.12.1 이 다른 다섯의 2.13.0 대비 커널 성능 차이를 만드는지.
   **측정 안 함**.
9. `cut_cross_entropy` 를 플러그인으로 켜면 임베딩이 bf16 으로 되돌아오면서 backward 가
   통과하는지, 그리고 그것이 손실 계산 축을 바꾸는지.
