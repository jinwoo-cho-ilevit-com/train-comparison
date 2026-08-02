# ms-swift 4.4.2 — 핀 원문 리서치

소비 레인: **adapters**
작성 2026-08-02. 모든 인용은 아래 "핀 해석"에서 확정한 경로의 원문 그대로다. 번역·요약 없음.

## 핀 해석 (증거)

| 패키지 | lock 핀 | lock 파일 | 상태 | 소스 루트 |
|---|---|---|---|---|
| ms-swift | `4.4.2` | `envs/ms-swift/uv.lock:1020-1021` | downloaded, sha256 일치 | `/private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/pins/ms-swift-4.4.2` |
| transformers | `5.12.1` | `envs/ms-swift/uv.lock:2126-2127` | on-disk | `/Users/jwcho/.cache/uv/archive-v0/uOhIKcY-1QKGNf7V` |
| torch | `2.13.0+cu130` | `envs/ms-swift/uv.lock:1994-1996` | **unavailable** | — |

ms-swift 는 이 호스트에 추출본이 없었다. `~/.cache/uv/wheels-v6/pypi/ms-swift/` 에는
`4.4.2-py3-none-any.msgpack` 등 메타데이터만 있었고 `~/.cache/uv/archive-v0/*/ms_swift-4.4.2.dist-info`
는 존재하지 않았다 (lock 의 의존성이 전부 `marker = "sys_platform == 'linux'"`).
`uv pip download` 는 이 uv 빌드에 없어(`error: unrecognized subcommand 'download'`)
lock 에 적힌 URL 을 그대로 받아 sha256 을 대조했다.

```
869b573de4a24129140a2b03b8c031b3c4024bba11ab768a9477f8d1dada87aa  ms_swift-4.4.2-py3-none-any.whl
```

lock 의 해당 줄 (`envs/ms-swift/uv.lock:1066`):

```
    { url = "https://files.pythonhosted.org/packages/f1/e2/cc8722b30e0874465a4ce030453e5cee9584d6976ce35859ce293acc6426/ms_swift-4.4.2-py3-none-any.whl", hash = "sha256:869b573de4a24129140a2b03b8c031b3c4024bba11ab768a9477f8d1dada87aa", size = 1239486, upload-time = "2026-07-21T15:47:06.317Z" },
```

일치. 아래의 ms-swift 인용은 전부 이 휠에서 나온 것이다.

transformers 는 디코이가 10개였다 (`5.13.0`, `5.13.1`, `5.14.1`, `5.12.0`, `5.11.0`, `5.10.2`,
`5.9.0`, `5.5.0`, `5.3.0`, `4.57.6`). 핀은 `5.12.1` → `uOhIKcY-1QKGNf7V`.

torch 는 **읽지 못했다.** 캐시에 `torch-2.13.0.dist-info` 가 네 벌 있으나
(`6T23WMnZuZp0UfUG`, `NLvfVDcPJjLEhe-a`, `jYfXy_uPyG5cyJ9v`, `lkUxu92yqFBhAdWZ`)
넷 다 `torch/version.py` 가 `__version__ = '2.13.0'` / `cuda: Optional[str] = None` 이다.
lock 이 요구하는 것은 `download.pytorch.org/whl/cu130` 의 `2.13.0+cu130` 이므로 네 개 모두 다른 물건이다.
**이 브리프에 torch 소스에 기댄 주장은 없다.**

---

## 1. `get_model_processor` 와 `from_pretrained` — 우리 docstring 이 틀렸다

`trainbench/probe/ms_swift.py:7-11` 은 이렇게 적고 있다.

> `get_model_processor` owns the `from_pretrained` call, so `axes.load_kwargs`
> (attention implementation, quantisation config) has nowhere to go here.

**앞부분은 맞고 뒷부분은 핀과 어긋난다.** `from_pretrained` 를 부르는 것은 확실히 로더지만,
`get_model_processor` 는 그 호출에 도달하는 통로를 세 개나 문서화해서 열어두고 있다.

`swift/model/register.py:516-542` (시그니처 일부):

```text
def get_model_processor(
    model_id_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Union[str, Dict[str, Any], None] = None,
    load_model: bool = True,
```

```text
    quantization_config=None,
    max_memory: Union[str, Dict[str, Any]] = None,
    attn_impl: Optional[str] = None,
    experts_impl: Optional[str] = None,
```

```text
    model_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[Optional[PreTrainedModel], Processor]:
```

docstring 이 `model_kwargs` 를 정확히 그렇게 정의한다 — `swift/model/register.py:577`:

```
        model_kwargs: Additional keyword arguments passed to the model's from_pretrained method.
```

그리고 그 dict 가 실제로 `from_pretrained` 까지 간다. 경로 전체:

`swift/model/register.py:594-596` — 보증되는 dict:

```text
    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
```

`swift/model/register.py:612-630` — 로더에 그대로 전달:

```text
    model_kwargs['device_map'] = device_map
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    if max_memory:
        model_kwargs['max_memory'] = max_memory
    loader = model_meta.loader(
        model_info,
        model_meta,
        load_model=load_model,
        attn_impl=attn_impl,
        experts_impl=experts_impl,
        rope_scaling=rope_scaling,
        max_model_len=max_model_len,
        auto_model_cls=auto_model_cls,
        return_dummy_model=return_dummy_model,
        new_special_tokens=new_special_tokens,
        model_kwargs=model_kwargs,
        **kwargs)
    return loader.load()
```

`swift/model/register.py:199-213` — 로더가 보관하고 dtype 만 덧쓴다:

```text
        self.model_kwargs = model_kwargs
        self.patch_offload = kwargs.pop('patch_offload', False)
```

```text
        if version.parse(transformers.__version__) >= version.parse('4.56'):
            model_kwargs['dtype'] = self.torch_dtype
        else:
            model_kwargs['torch_dtype'] = self.torch_dtype
```

`swift/model/register.py:463-467` — `load()` 가 그 복사본을 `get_model` 에 넘긴다:

```text
    def _get_model_processor(self, model_dir, config):
        processor = self.get_processor(model_dir, config)
        model = None
        if self.load_model:
            model = self.get_model(model_dir, config, processor, self.model_kwargs.copy())
```

`swift/model/register.py:315-317` — 종착지:

```text
            with context():
                model = auto_model_cls.from_pretrained(
                    model_dir, config=config, trust_remote_code=self.default_trust_remote_code, **model_kwargs)
```

### attention 축은 `model_kwargs` 가 아니라 config 를 지나간다

`attn_impl` 은 `from_pretrained` 인자가 되지 않는다. config 속성으로 심긴다 —
`swift/model/utils.py:26-52`:

```text
class AttnImpl:
    attn_impl_keys = ['_attn_implementation', 'attn_implementation', 'llm_attn_implementation']
    use_flash_attn_keys = ['_flash_attn_2_enabled', 'use_flash_attn', '_use_flash_attention_2']
```

```text
    @staticmethod
    def update_attn_impl(config: PretrainedConfig,
                         attn_impl: Optional[str],
                         attn_impl_keys: Optional[List[str]] = None) -> None:
        if attn_impl is None:
            return
        logger.info(f'attn_impl: {attn_impl}')
        use_flash_attn = AttnImpl.to_use_flash_attn(attn_impl)
        if use_flash_attn:
            attn_impl = 'flash_attention_2'
```

로더 생성자가 `attn_implementation` 이라는 이름의 자유 kwarg 도 같은 슬롯으로 받아준다 —
`swift/model/register.py:204-206`:

```text
        attn_impl = attn_impl or kwargs.get('attn_implementation')
        self.attn_impl = attn_impl
        self.attn_impl_keys = None
```

**adapters 레인이 가져갈 결론:** ms-swift 경로에서 `axes.load_kwargs` 는 "갈 곳이 없는" 것이 아니라
**세 갈래로 갈라져야 한다.** `quantization_config` → 동명 인자, attention → `attn_impl=`
(값 어휘는 `'flash_attn'`/`'flash_attention_2'`, `to_use_flash_attn` 기준),
나머지 `from_pretrained` kwarg → `model_kwargs=`. `dtype`/`torch_dtype` 은 로더가 덮어쓰므로
`model_kwargs` 에 넣지 말고 `torch_dtype=` 로 넘겨야 한다.

### "갈 곳이 없다"가 참인 지점은 CLI 쪽이다

CLI/arguments 경로에는 정말로 통로가 없다. `swift/arguments/base_args/model_args.py:228-249`:

```text
    def get_model_kwargs(self):
        return {
            'model_id_or_path': self.model,
            'torch_dtype': self.torch_dtype,
            'model_type': self.model_type,
            'revision': self.model_revision,
            'use_hf': self.use_hf,
            'hub_token': self.hub_token,
            'local_repo_path': self.local_repo_path,
            'device_map': self.device_map,
            'max_memory': self.max_memory,
            'quantization_config': self.get_quantization_config(),
            'attn_impl': self.attn_impl,
            'experts_impl': self.experts_impl,
            'new_special_tokens': self.new_special_tokens,
            'rope_scaling': self.rope_scaling,
            'max_model_len': self.max_model_len,
            'task_type': self.task_type,
            'num_labels': self.num_labels,
            'problem_type': self.problem_type,
            'init_strategy': self.init_strategy,
        }
```

`model_kwargs` 키가 없다. `swift/arguments/base_args/base_args.py:331-350` 이 이 dict 를 그대로
`get_model_processor(**res)` 로 흘린다. 우리 프로브는 파이썬 함수를 직접 부르므로 CLI 제약을
물려받지 않는다 — 프로브 docstring 은 CLI 사실을 라이브러리 사실로 옮겨 적은 것이다.

---

## 2. `Qwen3_5Loader` 는 `Qwen3VLLoader` 를 상속한다 — 텍스트 전용 0.8B 도 VL 경로다

`swift/model/models/qwen.py:1442-1448`:

```text
class Qwen3_5Loader(Qwen3VLLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Qwen3_5ForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen3_5ForConditionalGeneration
        _patch_qwen3_5_linear_attention_sequence_parallel()
        return Qwen2VLLoader.get_model(self, model_dir, config, processor, model_kwargs)
```

상속 사슬: `swift/model/models/qwen.py:1089` `class Qwen3VLLoader(Qwen2VLLoader):`,
`swift/model/models/qwen.py:761` `class Qwen2VLLoader(ModelLoader):`.

등록 테이블이 `Qwen/Qwen3.5-0.8B` 를 여기에 넣는다 — `swift/model/models/qwen.py:1451-1480`:

```text
register_model(
    ModelMeta(
        MLLMModelType.qwen3_5,
        [
            ModelGroup(
                [
                    Model('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B'),
                    Model('Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-2B'),
```

```text
        Qwen3_5Loader,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen3_5ForConditionalGeneration'],
        requires=['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

`MLLMModelType` 에 있으면 멀티모달로 표시된다 — `swift/model/model_meta.py:92-93`:

```text
        if self.model_type in MLLMModelType.__dict__:
            self.is_multimodal = True
```

즉 **0.8B 텍스트 모델이 `is_multimodal=True`, `model_arch=qwen2_vl` 로 적재된다.** 결과 세 가지:

**(a) `qwen_vl_utils` 가 프로세서 로드의 하드 게이트가 된다** (§3).

**(b) 로더가 `model.visual` 을 만진다.** `swift/model/models/qwen.py:761-769`:

```text
class Qwen2VLLoader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Qwen2VLForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen2VLForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        patch_get_input_embeddings(base_model.visual, 'patch_embed')
        return model
```

`.visual` 은 최상위 클래스에는 없다. 핀 transformers 5.12.1 에서
`/Users/jwcho/.cache/uv/archive-v0/uOhIKcY-1QKGNf7V/transformers/models/qwen3_5/modeling_qwen3_5.py:1709-1719`:

```text
class Qwen3_5ForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
```

`.visual` 은 안쪽에 있다 — 같은 파일 `1236-1245`:

```text
class Qwen3_5Model(Qwen3_5PreTrainedModel):
    base_model_prefix = "model"
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
```

ms-swift 가 그 간극을 transformers 5 호환 훅으로 메운다. `swift/model/register.py:329-331`
(`super().get_model` 이 돌아오기 직전, 즉 `base_model.visual` 접근보다 **먼저** 실행된다):

```text
        if transformers_5:
            self._compat_transformers5(model)
        return model
```

`swift/model/register.py:68-72`:

```text
    def _compat_transformers5(self, model):
        if self.model_meta.is_multimodal:
            for key in ['language_model', 'vision_tower', 'multi_modal_projector', 'visual', 'vision_model']:
                _set_property(model, key)
```

`swift/model/register.py:120-130`:

```text
def _set_property(model, key):
    if not hasattr(model, 'model'):
        return
    text_model = model.model
    if not hasattr(text_model, key) or hasattr(model.__class__, key):
        return

    def _value(self):
        return getattr(self.model, key)

    setattr(model.__class__, key, property(_value))
```

`_set_property` 는 `model.model` 에 그 속성이 **없으면 조용히 반환한다.** 그러면 다음 줄의
`base_model.visual` 이 `AttributeError` 로 죽는다. 이 분기는 `Qwen/Qwen3.5-0.8B` 의
`config.json` 에 `vision_config` 가 있느냐에 달려 있고 — 그건 Hub 사실이라 이 호스트에서
확정할 수 없다. 마지막 절에 남긴다.

**(c) LoRA `all-linear` 이 리스트가 아니라 정규식이 된다.** `swift/pipelines/train/tuner.py:91-110`:

```text
def get_target_modules(args, model) -> Union[str, List[str]]:
    """Replace all-linear to actual modules"""
    if isinstance(args.target_modules, str):
        return args.target_modules
    target_modules = args.target_modules.copy()
    if 'all-linear' in target_modules:
        if model.model_meta.is_multimodal:
            return get_multimodal_target_regex(
                model,
                freeze_llm=args.freeze_llm,
                freeze_vit=args.freeze_vit,
                freeze_aligner=args.freeze_aligner,
                include_embedding='all-embedding' in target_modules)
        else:
            target_modules.remove('all-linear')
            target_modules += find_all_linears(model)
```

`swift/utils/transformers_utils.py:208-225` — 기본값이 vit/aligner 를 얼린다:

```text
def get_multimodal_target_regex(
    model,
    *,
    freeze_llm: bool = False,
    freeze_vit: bool = True,
    freeze_aligner: bool = True,
    include_embedding: bool = False,
    exclude_router: bool = False,
) -> str:
    model_arch = model.model_meta.model_arch
    modules = []
    if not freeze_llm:
        modules += model_arch.language_model
    if not freeze_vit:
        modules += model_arch.vision_tower
    if not freeze_aligner:
        modules += model_arch.aligner
    assert len(modules) > 0, f'modules: {modules}'
```

`qwen2_vl` arch 의 실제 접두사 — `swift/model/model_arch.py:575-583`:

```text
if transformers_ge_4_52:
    register_model_arch(
        MultiModelKeys(
            MLLMModelArch.qwen2_vl,
            language_model=['model.language_model', 'lm_head'],
            aligner='model.visual.merger',
            vision_tower='model.visual',
        ))
```

**adapters 레인이 가져갈 결론:** Qwen3.5-0.8B 의 LoRA 타깃은 `q_proj` 류 이름 목록이 아니라
`model.language_model(?=\.).*\.(...)|lm_head(?=\.)...` 형태의 정규식이고,
`freeze_vit`/`freeze_aligner` 기본 True 때문에 vision 쪽은 애초에 빠진다.
다른 프레임워크와 "같은 LoRA" 를 주장하려면 **모듈 이름이 아니라 실제 어댑터가 붙은 파라미터 수**를
비교해야 한다. 축 하나를 이름으로 맞추면 조용히 어긋난다.

Qwen3-VL-Embedding-2B 는 별도 로더/템플릿을 갖는다 — `swift/model/models/qwen.py:2130-2153`:

```text
class Qwen3VLEmbLoader(Qwen3VLLoader):

    def _check_qwen_vl_utils(self):
        os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '1800')
        os.environ.setdefault('FPS', '1')
        os.environ.setdefault('FPS_MAX_FRAMES', '64')
        super()._check_qwen_vl_utils()
```

```text
        Qwen3VLEmbLoader,
        template=TemplateType.qwen3_vl_emb,
        model_arch=ModelArch.qwen3_vl,
        mcore_model_type='qwen3_vl',
        architectures=['Qwen3VLForConditionalGeneration'],
        requires=['transformers>=4.57', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))
```

이 ModelMeta 에 `task_type=` 이 없다. 임베딩 모델이라고 자동으로 임베딩 태스크가 되지 않는다 (§5).

gemma-4-E2B 도 등록되어 있다 — `swift/model/models/gemma.py:422-454`:

```text
class Gemma4Loader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Gemma4ForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Gemma4ForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        _patch_gemma4_forward(model.model, processor)
        return model
```

```text
            ModelGroup([
                Model('google/gemma-4-E2B', 'google/gemma-4-E2B'),
                Model('google/gemma-4-E2B-it', 'google/gemma-4-E2B-it'),
```

```text
                       template=TemplateType.gemma4_nothinking),
```

`requires` 가 없고 `_check_qwen_vl_utils` 도 없다. base 체크포인트에 `-it` 전용 템플릿을 붙인 게
아니라, **base 에 `gemma4_nothinking` 이라는 swift 자체 템플릿을 명시로 배정**했다.
AGENTS.md 가 기록한 "gemma-4-E2B 에 `chat_template.jinja` 가 없어 `apply_chat_template` 이 죽는다"는
실패 모드는 이 경로에는 구조적으로 없다 (§4).

---

## 3. 게이트인 것과 게이트가 아닌 것

### 게이트다 — `_check_qwen_vl_utils`

`swift/model/models/qwen.py:1089-1093` (Qwen3.5 가 물려받는 판본):

```text
class Qwen3VLLoader(Qwen2VLLoader):

    def _check_qwen_vl_utils(self):
        require_version('qwen_vl_utils>=0.0.14')
        compat_qwen_vl_utils(image_patch_size=16)
```

`try` 가 없다. 그리고 프로세서 로드가 반드시 이걸 지난다 —
`swift/model/models/qwen.py:782-789`:

```text
    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        self._check_qwen_vl_utils()
        from qwen_vl_utils import vision_process
        processor = super().get_processor(model_dir, config)
        global_vars = patch_qwen_vl_utils(vision_process)
        processor.global_vars = global_vars  # In order to have different hashes for the template.
        return processor
```

`from qwen_vl_utils import vision_process` 는 무조건 실행되는 최상단 임포트다.
lock 이 `qwen-vl-utils 0.0.14` 를 고정하므로 (`envs/ms-swift/uv.lock:1661-1662`) 통과하지만,
이건 lock 이 우연히 맞춰준 것이지 코드가 봐주는 게 아니다.

`require_version` 이 실제로 무엇을 던지는지 — 핀 transformers 5.12.1,
`/Users/jwcho/.cache/uv/archive-v0/uOhIKcY-1QKGNf7V/transformers/utils/versions.py:100-111`:

```text
    # check if any version is installed
    try:
        got_ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        raise importlib.metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
        )

    # check that the right version is installed if version number or a range was provided
    if want_ver is not None:
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
```

같은 파일 `36-45`:

```text
def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None or want_ver is None:
        raise ValueError(
            f"Unable to compare versions for {requirement}: need={want_ver} found={got_ver}. This is unusual. Consider"
            f" reinstalling {pkg}."
        )
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )
```

미설치는 `PackageNotFoundError`, 버전 불일치는 `ImportError`. 둘 다 `ImportError` 계열이다.

### 게이트가 아니다 — `ModelMeta.check_requires`

`swift/model/model_meta.py:106-119`:

```text
    def check_requires(self, model_info=None):
        extra_requires = []
        if model_info and model_info.quant_method:
            mapping = {'bnb': ['bitsandbytes'], 'awq': ['autoawq'], 'gptq': ['auto_gptq'], 'aqlm': ['aqlm']}
            extra_requires += mapping.get(model_info.quant_method, [])
        requires = []
        for require in self.requires + extra_requires:
            try:
                require_version(require)
            except ImportError:
                requires.append(f'"{require}"')
        if requires:
            requires = ' '.join(requires)
            logger.warning(f'Please install the package: `pip install {requires} -U`.')
```

`except ImportError` 가 위의 두 예외를 **전부 삼키고 warning 한 줄만 남긴다.**
호출 지점은 모델 적재 직전이다 — `swift/model/model_meta.py:327`:

```text
    model_meta.check_requires(model_info)
```

그래서 Qwen3.5 ModelMeta 의 `requires=['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord']`
(`swift/model/models/qwen.py:1479`) 중 **`decord` 는 아무것도 막지 않는다.**
실제로 `envs/ms-swift/uv.lock` 에 `decord` 는 없다. 즉 이 환경은 항상
`Please install the package: `pip install "decord" -U`.` 를 로그에 찍고 그대로 진행한다.

**adapters 레인이 가져갈 결론:** ms-swift 로그의 requires 경고는 진단 신호가 아니라 상수다.
프로브가 그걸 실패 신호로 잡으면 안 되고, 반대로 `qwen_vl_utils` 는 경고 없이 즉시 죽는다.
지문(fingerprint)에 넣을 실패는 후자다.

---

## 4. `get_template` 은 기본값으로 `apply_chat_template` 을 부르지 않는다

진입점 — `swift/template/register.py:55-75` (일부):

```text
def get_template(
    processor: Processor,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    *,
    template_type: Optional[str] = None,
```

```text
    # infer/deploy
    template_backend: Literal['swift', 'jinja'] = 'swift',
```

같은 기본값이 Template 에 그대로 저장된다 — `swift/template/base.py:100`, `:140`:

```text
        template_backend: Literal['swift', 'jinja'] = 'swift',
```

```text
        self.template_backend = template_backend
```

분기 — `swift/template/base.py:1485-1494`:

```text
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs.messages = deepcopy(inputs.messages)
        template_backend = self.template_backend
        if (self.template_meta.template_type == 'dummy' and self.use_chat_template and not self.is_training
                and self.task_type == 'causal_lm'):
            template_backend = 'jinja'
            logger.info_once(f'Setting template_backend: {template_backend}')
        self._swift_prepare_inputs(inputs)
        res_context_list, loss_scale_list, answer_len = (
            self._swift_encode(inputs) if template_backend == 'swift' else self._jinja_encode(inputs))
```

`apply_chat_template` 은 오직 `_jinja_encode` 안에 있다 — `swift/template/base.py:1155-1159`:

```text
        kwargs.update(self.chat_template_kwargs)
        kwargs.update(inputs.chat_template_kwargs)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
        answer_len = 1 if self.is_training else 0
```

**즉 `chat_template.jinja` 가 필요한 유일한 조건은 두 가지다:** 사용자가 `template_backend='jinja'`
를 넘겼거나, template_type 이 `'dummy'` 로 떨어졌는데(등록 테이블에 없는 모델) 추론 중이고
causal_lm 인 경우. 우리 세 모델은 모두 등록되어 있어 `'dummy'` 가 아니다 (§2).

swift 백엔드가 Qwen3.5 에서 무엇을 하는지는 템플릿이 직접 적어둔다 —
`swift/template/templates/qwen.py:598-609`:

```text
class Qwen3_5Template(Qwen3VLTemplate):
    image_token_id = 248056
    video_token_id = 248057

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.padding_free and self.sequence_parallel_size <= 1 and not self.transformers_5_9:
            raise RuntimeError('Qwen3.5 packing/padding_free with sequence_parallel_size=1 requires '
                               f'transformers>=5.9.0 (current: {self.transformers_version}). ')
        return Qwen2VLTemplate._post_encode(self, model, inputs)

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        # Normalize message content so the swift backend byte-matches Qwen3.5/Qwen3.6
        # HF `chat_template.jinja` rendering (per-role `|trim` and canonical <think> padding).
```

**adapters 레인이 가져갈 결론:** ms-swift 는 프롬프트를 자기가 조립한다. AGENTS.md 가 기록한
"세 프레임워크가 `apply_chat_template` 에서 똑같이 죽었다"는 실패는 ms-swift 기본 경로에는
해당하지 않는다. 대신 **다른 프레임워크와 토큰열이 바이트 단위로 같다는 보장이 없어진다** —
위 주석은 그걸 맞추려는 시도가 진행 중임을 자백한다. 프레임워크 간 처리량 비교에서
시퀀스 길이가 다르면 그건 커널이 아니라 템플릿 차이다. 축을 얼릴 때 토큰 수를 함께 기록해야 한다.

템플릿의 `task_type` 은 프로세서를 따라온다 — `swift/template/base.py:225-232`:

```text
    def init_processor(self, processor: Processor) -> None:
        if processor is None or self._processor_inited:
            return
        self._processor_inited = True
        self.processor = processor
        self.model_info = processor.model_info
        self.config = self.model_info.config
        self.task_type = self.model_info.task_type
```

---

## 5. 임베딩/InfoNCE 의 문서화된 진입점

ms-swift 는 **loss 를 인자로 받는 자체 Trainer 계층**을 갖는다. 우리 경로(직접 InfoNCE 를 손으로
계산)와는 다른 물건이다.

Trainer 선택 — `swift/trainers/trainer_factory.py:13-19`:

```text
class TrainerFactory:
    TRAINER_MAPPING = {
        'causal_lm': 'swift.trainers.Seq2SeqTrainer',
        'seq_cls': 'swift.trainers.Trainer',
        'embedding': 'swift.trainers.EmbeddingTrainer',
        'reranker': 'swift.trainers.RerankerTrainer',
        'generative_reranker': 'swift.trainers.RerankerTrainer',
```

loss 등록 테이블 — `swift/loss/mapping.py:6-16`:

```text
loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    # # reranker
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
}
```

주입 지점 — `swift/trainers/mixin.py:1046-1054`:

```text
    def create_loss_and_eval_metric(self, args):
        res = {}
        if args.eval_metric is not None:
            eval_metric = eval_metrics_map[args.eval_metric](args, self)
            res['compute_metrics'], res['preprocess_logits_for_metrics'] = (eval_metric.compute_metrics,
                                                                            eval_metric.preprocess_logits_for_metrics)
        if args.loss_type is not None:
            res['compute_loss_func'] = loss_map[args.loss_type](args, self)
        return res
```

CLI 필드 — `swift/trainers/arguments.py:165`:

```text
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(loss_map.keys())}'})
```

`task_type` 필드 — `swift/arguments/base_args/model_args.py:72`:

```text
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker'] = None
```

### 우리 경로와의 결정적 차이: `task_type='embedding'` 이 모델 출력을 바꾼다

`swift/model/register.py:324-326`:

```text
        if model_info.task_type == 'embedding' and auto_model_cls.__name__ != 'AutoModel':
            from swift.model.patcher import patch_output_normalizer
            patch_output_normalizer(model, model_meta=model_meta)
```

`swift/model/patcher.py:75-102`:

```text
def patch_output_normalizer(module: torch.nn.Module, model_meta):

    def lm_head_forward(self, hidden_states):
        return hidden_states

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    lm_head_model = get_lm_head_model(module, model_meta=model_meta, lm_heads=lm_heads)

    found = False
    for lm_head in lm_heads:
        if hasattr(lm_head_model, lm_head):
            getattr(lm_head_model, lm_head).forward = MethodType(lm_head_forward, getattr(lm_head_model, lm_head))
            found = True
            break

    assert found, 'Cannot find the proper lm_head name'

    def _output_embedding_hook(module, args, kwargs, output):
        attention_mask = kwargs.get('attention_mask', None)
        hidden_states = output.logits
        sequence_lengths = -1 if attention_mask is None else get_last_valid_indices(attention_mask)
        embeddings = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return {
            'last_hidden_state': embeddings.contiguous(),
        }

    lm_head_model.register_forward_hook(_output_embedding_hook, with_kwargs=True)
```

`lm_head` 를 항등함수로 만들고, 마지막 유효 토큰을 뽑아 L2 정규화한 뒤 `last_hidden_state` 키로
돌려준다. `InfonceLoss` 가 정확히 그 키를 읽는다 — `swift/loss/embedding.py:113-137`:

```text
class InfonceLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.1'))  # temperature
        # calculate CE across the batch, meaning all samples will be negative except the matching positive
        use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
        hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)  # how many negative prompts kept in one sample
```

```text
        # repeat of anchor(1)+positive(1)+negatives(n)
        sentences = outputs['last_hidden_state']
```

**하이퍼파라미터가 config 가 아니라 환경변수다.** `INFONCE_TEMPERATURE`, `INFONCE_USE_BATCH`,
`INFONCE_HARD_NEGATIVES`, `INFONCE_MASK_FAKE_NEGATIVE`, `INFONCE_FAKE_NEG_MARGIN`,
`INFONCE_INCLUDE_QQ`, `INFONCE_INCLUDE_DD` (`swift/loss/embedding.py:116-126`).
우리 `config.loss.temperature` 를 ms-swift 에 반영하려면 `os.environ` 을 통과해야 한다.

배치 구성도 우리와 다르다 — `swift/template/base.py:509-545` `_embedding_encode` 가
anchor/positive/negative 를 접두사로 나눠 인코딩하고 `labels` 를 `[1.0, 0.0, 0.0, ...]` 로 쌓으며,
`swift/template/base.py:1773-1803` `_embedding_data_collator` 가 그걸 한 축으로 편다:

```text
                indexes = ['anchor_', 'positive_']
                if max_neg is not None:
                    for i in range(0, max_neg):
                        indexes.append(f'negative{i}_')
                for prefix in indexes:
                    new_batch += self._fetch_inputs_startswith([b], prefix)
            labels.extend(b.get('labels', []))
        res = self._data_collator(new_batch, padding_to=padding_to)
        if labels:
            res['labels'] = torch.tensor(labels, dtype=torch.float32)
```

즉 `train.batch_size` 의 의미가 다르다 — ms-swift 에서 한 "샘플"은
`anchor + positive + n*negative` 행으로 펼쳐진다.

LoRA 쪽 특수 처리 — `swift/pipelines/train/tuner.py:169-175`:

```text
        elif args.tuner_backend == 'peft':
            if task_type == 'EMBEDDING':
                task_type = None
            elif task_type == 'RERANKER':
                task_type = 'SEQ_CLS'
            elif task_type == 'GENERATIVE_RERANKER':
                task_type = 'CAUSAL_LM'
```

임베딩 학습에서는 peft `LoraConfig(task_type=None)` 이 된다.

MRL 은 EmbeddingTrainer 가 loss 를 한 번 더 감싸서 구현한다 —
`swift/trainers/embedding_trainer.py:11-35`:

```text
class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        mrl_dims = self.args.mrl_dims
        if mrl_dims and self.compute_loss_func is not None:
            origin_loss_func = self.compute_loss_func
```

### 우리 프로브가 지금 하고 있는 것

`trainbench/probe/ms_swift.py:37` 은 `get_model_processor(config.model.hf_id)` 를 부른다.
`task_type` 이 없다. 기본값 결정은 `swift/model/model_meta.py:295-303`:

```text
    if task_type is None:
        if model_meta.is_reward:
            num_labels = 1
        if num_labels is None:
            task_type = 'causal_lm'
        else:
            task_type = 'seq_cls'
        if model_meta.task_type is not None:
            task_type = model_meta.task_type
```

Qwen3-VL-Embedding-2B 의 ModelMeta 에는 `task_type=` 이 없으므로(§2) **`causal_lm` 으로 떨어진다.**
따라서 `patch_output_normalizer` 가 걸리지 않고, 모델은 임베딩이 아니라 로짓을 낸다.
지금의 `infonce_backward` 는 ms-swift 의 임베딩 경로를 통과하지 않는다.

**adapters 레인이 가져갈 결론:** ms-swift 로 임베딩 학습 속도를 재려면 최소한
`get_model_processor(..., task_type='embedding')` 이 필요하고, 그 순간
`lm_head` 가 항등이 되어 forward/backward 비용 프로필 자체가 달라진다.
현재 프로브가 재는 것과 ms-swift 가 실제로 학습할 때 도는 것은 다른 그래프다.

---

## 6. lock 이 고정한 버전

`envs/ms-swift/uv.lock` 에서 그대로:

| 패키지 | 버전 | 줄 |
|---|---|---|
| ms-swift | `4.4.2` | 1020-1021 |
| transformers | `5.12.1` | 2126-2127 |
| torch | `2.13.0+cu130` (`source = { registry = "https://download.pytorch.org/whl/cu130" }`) | 1994-1996 |
| accelerate | `1.14.0` | 21-22 |
| peft | `0.19.1` | 1412-1413 |
| trl | `0.29.1` | 2164-2165 |
| qwen-vl-utils | `0.0.14` | 1661-1662 |
| decord | 없음 | — |

`Qwen3_5Loader` 의 ModelMeta 가 요구하는 `transformers>=5.0.0.dev` 는 5.12.1 로 충족된다.
핀 transformers 에 필요한 모듈이 실제로 있는지 확인했다:

```
/Users/jwcho/.cache/uv/archive-v0/uOhIKcY-1QKGNf7V/transformers/models/gemma4/
/Users/jwcho/.cache/uv/archive-v0/uOhIKcY-1QKGNf7V/transformers/models/qwen3_5/
/Users/jwcho/.cache/uv/archive-v0/uOhIKcY-1QKGNf7V/transformers/models/qwen3_5_moe/
/Users/jwcho/.cache/uv/archive-v0/uOhIKcY-1QKGNf7V/transformers/models/qwen3_vl/
```

`Qwen3_5ForConditionalGeneration` (`modeling_qwen3_5.py:1709`) 과
`Gemma4ForConditionalGeneration` (`modeling_gemma4.py:2445`) 둘 다 존재한다.
`swift/model/models/qwen.py:1445` 과 `swift/model/models/gemma.py:425` 의 임포트는 성립한다.

휠에는 `swift/` 와 dist-info 만 들어 있다 (`RECORD` 532줄). `examples/train/embedding` 은
METADATA 가 GitHub 링크로만 언급하며(`METADATA:219`, `:445`) 휠에 동봉되지 않는다.
콘솔 진입점은 둘이다 (`entry_points.txt`):

```
[console_scripts]
megatron = swift.cli._megatron.main:cli_main
swift = swift.cli.main:cli_main
```

---

## 이 호스트에서 확정하지 못한 것

파드/이미지가 답해야 하는 질문. 추측은 적지 않는다.

1. `Qwen/Qwen3.5-0.8B` 의 `config.json` 에 `vision_config` 가 있는가? 없다면
   `Qwen3_5Model.__init__` 의 `AutoModel.from_config(config.vision_config)` 이 먼저 죽고,
   설령 통과하더라도 `_set_property` 가 조용히 반환해
   `Qwen2VLLoader.get_model` 의 `base_model.visual` 이 `AttributeError` 를 낸다.
   (`swift/model/models/qwen.py:768`, `swift/model/register.py:120-125`)
2. `Qwen/Qwen3.5-0.8B` 의 아키텍처 문자열이 실제로 `Qwen3_5ForConditionalGeneration` 인가?
   등록 테이블은 그렇게 가정한다 (`swift/model/models/qwen.py:1478`). 다르면
   model_type 자동 추론이 다른 ModelMeta 로 간다.
3. `task_type='embedding'` 을 준 Qwen3-VL-Embedding-2B 에서 `patch_output_normalizer` 의
   `assert found, 'Cannot find the proper lm_head name'` 이 통과하는가?
   `get_lm_head_model` 이 `model_arch=qwen3_vl` 의 `language_model=['model.language_model', 'lm_head']`
   에서 무엇을 고르는지는 실제 모듈 트리 없이는 확정 불가다.
4. `google/gemma-4-E2B` (base) 를 `TemplateType.gemma4_nothinking` 으로 적재할 때 프로세서가
   `AutoProcessor` 로 뜨는가 `AutoTokenizer` 로 뜨는가? 분기는 체크포인트에
   `preprocessor_config.json` / `processor_config.json` 이 있느냐다
   (`swift/model/register.py:259-268`). 이게 `visual_tokens` 프로브 결과를 가른다.
5. Qwen3.5 / gemma-4 LoRA `all-linear` 이 만들어내는 정규식이 실제로 몇 개의 파라미터에
   어댑터를 붙이는가? 다른 프레임워크와 같은 수인가? 모듈 트리 없이는 셀 수 없다.
6. `torch==2.13.0+cu130` 소스는 이 호스트에 없다. torch 쪽 동작에 대한 주장은 전부 "확인 안 함".
7. ms-swift 가 `logger.warning` 으로만 넘기는 `decord` 부재가 이미지/비디오 없는 순수 텍스트
   InfoNCE 경로에서 런타임 임포트로 되살아나는가? 휠 안에서 `decord` 임포트 지점을 전수
   확인하지 않았다 — 프로브가 실제로 밟는 코드만으로 판단해야 한다.
