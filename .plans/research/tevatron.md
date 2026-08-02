# tevatron — 핀 원문 리서치

소비 레인: `probe`, `adapters`.

모든 인용은 핀이 가리키는 파일에서 그대로 복사했다. 번역·요약하지 않았다.

## 0. 핀 해석

### 0.1 tevatron (git 소스)

`envs/tevatron/uv.lock:1043-1050`

```
[[package]]
name = "tevatron"
version = "0.0.1"
source = { git = "https://github.com/texttron/tevatron.git#dd063104c81a76d6a77c845f667b46b9e5abd625" }
dependencies = [
    { name = "datasets", marker = "sys_platform == 'linux'" },
    { name = "transformers", marker = "sys_platform == 'linux'" },
]
```

해석 명령과 출력:

```
$ ls -d ~/.cache/uv/git-v0/checkouts/*/dd06310
/Users/jwcho/.cache/uv/git-v0/checkouts/af8e1386372d71f4/dd06310/

$ git -C /Users/jwcho/.cache/uv/git-v0/checkouts/af8e1386372d71f4/dd06310 rev-parse HEAD
dd063104c81a76d6a77c845f667b46b9e5abd625

$ git -C /Users/jwcho/.cache/uv/git-v0/checkouts/af8e1386372d71f4/dd06310 status --porcelain
?? .ok
```

체크아웃 HEAD 가 lock 의 sha 와 **정확히** 일치하고, 추적 파일은 하나도 수정되지 않았다
(`.ok` 는 uv 가 남기는 완료 마커다). 이후 `TEV` = `/Users/jwcho/.cache/uv/git-v0/checkouts/af8e1386372d71f4/dd06310`.

디코이: `~/.cache/uv/git-v0/checkouts/` 에는 네 개의 체크아웃이 있고 각각 짧은 sha 하나씩을
담는다 — `7579ab1d582ee7cc/5e4fb37`, `7a95f4b56e4fee6d/906f038`, `a315cf8f2d8ba162/52e6d50`.
전부 열지 않았다. `dd06310` 은 `af8e1386372d71f4` 아래에만 있다.

### 0.2 transformers 5.14.1

`envs/tevatron/uv.lock:1206-1208`

```
name = "transformers"
version = "5.14.1"
source = { registry = "https://pypi.org/simple" }
```

```
$ ls -d ~/.cache/uv/archive-v0/*/transformers-5.14.1.dist-info
/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti/transformers-5.14.1.dist-info/
```

디코이(같은 캐시에 함께 있고 경로에 버전이 적혀 있지 않다, 열지 않았다):
`70OhQvhQj042zLtn`(5.13.0), `Bh3N2cKaCNDMURw4`(5.12.0), `JV7sX-v4goOSOToR`(5.13.1),
`Oqy6RtNB3vvkLBO3`(5.10.2), `Rj8Th9Bs2T6mue_z`(4.57.6), `SMXDMezQuUQYpkT2`(5.3.0),
`nB97V5Iacpret1f5`(5.9.0), `plcyRhzg-LE7LDvn`(5.5.0), `u8fTA70ydUgg54UN`(5.11.0),
`uOhIKcY-1QKGNf7V`(5.12.1).

`TF` = `/Users/jwcho/.cache/uv/archive-v0/Kur5R2PrM3RUwEti`.

저장소 루트 `.venv` 도 transformers 5.14.1 이며, 인용한 네 파일은 캐시 아카이브와 바이트
동일이다(`diff -q` 4/4 `same`). 그래서 `.venv/bin/python` 으로 돌린 재현은 핀 위에서 돈 것이다.

### 0.3 huggingface_hub 1.26.0 (`@strict` 의 출처)

`envs/tevatron/uv.lock:407-409` 이 `1.26.0` 을 핀한다.

```
$ ls -d ~/.cache/uv/archive-v0/*/huggingface_hub-1.26.0.dist-info
/Users/jwcho/.cache/uv/archive-v0/qIyDCv3fe8kliCVk/huggingface_hub-1.26.0.dist-info/
```

디코이 16개가 같은 캐시에 있다(0.36.2 / 1.8.0 / 1.16.1 / 1.17.0 / 1.19.0 / 1.20.1 x2 /
1.21.0 x2 / 1.22.0 / 1.23.0 x2 / 1.24.0 / 1.25.1 x2). 열지 않았다.

`HH` = `/Users/jwcho/.cache/uv/archive-v0/qIyDCv3fe8kliCVk`.

---

## 1. `pad_token_id` 접근 — getattr 이 아니라 직접 접근이다

`TEV/src/tevatron/retriever/modeling/encoder.py:117-129` (`build`)

```text
    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
```

`TEV/src/tevatron/retriever/modeling/encoder.py:159-172` (`load` — 이 저장소의 프로브가
쓰는 경로)

```text
    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
```

**두 경로 모두 `getattr(..., None)` 이 아니라 `base_model.config.pad_token_id` 직접 접근이다**
(`build` 125행, `load` 167행). 그래서 속성이 없는 config 에서는 `is None` 비교에 도달하기
전에 `AttributeError` 가 난다. 상류가 `hasattr` 로 감싸지 않았으므로 의존성 버전으로
피할 수 있는 종류가 아니다.

`TRANSFORMER_CLS` 는 `encoder.py:26-27`:

```text
class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel
```

`DenseModel` 은 이를 오버라이드하지 않는다(`TEV/src/tevatron/retriever/modeling/dense.py:16`
`class DenseModel(EncoderModel):` — 클래스 본문에 `TRANSFORMER_CLS` 재정의 없음). 따라서
`DenseModel.load` 는 `AutoModel.from_pretrained` 를 탄다.

### 1.1 왜 세 모델 전부 터지는가 (transformers 5.14.1 원문)

`pad_token_id` 는 **텍스트 서브컨피그에만** 선언돼 있다.

`TF/transformers/models/qwen3_5/configuration_qwen3_5.py:104`

```text
    pad_token_id: int | None = None
```

`TF/transformers/models/qwen3_vl/configuration_qwen3_vl.py:96`

```text
    pad_token_id: int | None = None
```

`TF/transformers/models/gemma4/configuration_gemma4.py:169`

```text
    pad_token_id: int | None = 0
```

최상위 합성 config 에는 없다. `TF/transformers/models/qwen3_5/configuration_qwen3_5.py:152-181`

```text
@strict
class Qwen3_5Config(PreTrainedConfig):
    r"""
    Example:
    ...
    ```"""

    model_type = "qwen3_5"
    sub_configs = {"vision_config": Qwen3_5VisionConfig, "text_config": Qwen3_5TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False
```

`TF/transformers/models/qwen3_vl/configuration_qwen3_vl.py:107-125` 도 같은 모양이고
(`sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}` 가 125행),
`TF/transformers/models/gemma4/configuration_gemma4.py:323-341` 은

```text
    model_type = "gemma4"
    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
        "audio_config": Gemma4AudioConfig,
    }

    text_config: Gemma4TextConfig | dict[str, Any] | None = None
    vision_config: Gemma4VisionConfig | dict[str, Any] | None = None
    audio_config: Gemma4AudioConfig | dict[str, Any] | None = None
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 258_882
    image_token_id: int | None = 258_880
    video_token_id: int | None = 258_884
    boa_token_id: int | None = 256_000
    eoa_token_index: int | None = 258_883
    audio_token_id: int | None = 258_881
    initializer_range: float | None = 0.02
    tie_word_embeddings: bool = True
```

세 파일 어디에도 `attribute_map` 이 없다(`grep -n attribute_map` 세 파일 전부 0 히트).
`PreTrainedConfig` 에는 `__getattr__` 폴백이 없고 `__getattribute__` 는 `attribute_map`
치환만 한다 — `TF/transformers/configuration_utils.py:456-464`

```text
    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)
```

즉 최상위에 없는 이름은 서브컨피그로 위임되지 않고 그대로 `AttributeError` 다.

핀 위에서 재현했다(`.venv/bin/python`, transformers 5.14.1):

```
Qwen3VLConfig AttributeError: 'Qwen3VLConfig' object has no attribute 'pad_token_id'
Qwen3VLConfig setattr ok -> 0
Qwen3VLConfig text_config Qwen3VLTextConfig pad_token_id None
Qwen3_5Config AttributeError: 'Qwen3_5Config' object has no attribute 'pad_token_id'
Qwen3_5Config setattr ok -> 0
Qwen3_5Config text_config Qwen3_5TextConfig pad_token_id None
Gemma4Config AttributeError: 'Gemma4Config' object has no attribute 'pad_token_id'
Gemma4Config setattr ok -> 0
Gemma4Config text_config Gemma4TextConfig pad_token_id 0
```

2차 캠페인이 기록한 메시지와 문자열이 같다.

### 1.2 `@strict` 는 setattr 를 막지 않는다

`@strict` 는 transformers 것이 아니라 huggingface_hub 것이다
(`TF/transformers/models/qwen3_5/configuration_qwen3_5.py:20` `from huggingface_hub.dataclasses import strict`).

`HH/huggingface_hub/dataclasses.py:136-151`

```text
        # Override __setattr__ to validate fields on assignment
        original_setattr = cls.__setattr__

        def __strict_setattr__(self: Any, name: str, value: Any) -> None:
            """Custom __setattr__ method for strict dataclasses."""
            # Run all validators
            for validator in self.__validators__.get(name, []):
                try:
                    validator(value)
                except (ValueError, TypeError) as e:
                    raise StrictDataclassFieldValidationError(field=name, cause=e) from e

            # If validation passed, set the attribute
            original_setattr(self, name, value)

        cls.__setattr__ = __strict_setattr__  # type: ignore
```

`self.__validators__.get(name, [])` 는 **선언되지 않은 이름에 대해 빈 리스트**를 돌려주고
곧바로 `original_setattr` 로 넘어간다. 그래서 `config.pad_token_id = 0` 은 합성 config
에서도 통과한다(위 재현의 `setattr ok -> 0`). **읽기만 막히고 쓰기는 열려 있다** — shim 이
설 자리가 여기다.

### 1.3 `get_text_config()` 경로

`TF/transformers/configuration_utils.py:1297-1324`

```text
        return_both = decoder == encoder  # both unset or both set -> search all possible names

        decoder_possible_text_config_names = ("decoder", "generator", "text_config")
        encoder_possible_text_config_names = ("text_encoder",)
        if return_both:
            possible_text_config_names = encoder_possible_text_config_names + decoder_possible_text_config_names
        elif decoder:
            possible_text_config_names = decoder_possible_text_config_names
        else:
            possible_text_config_names = encoder_possible_text_config_names

        valid_text_config_names = []
        for text_config_name in possible_text_config_names:
            if hasattr(self, text_config_name):
                text_config = getattr(self, text_config_name, None)
                if text_config is not None:
                    valid_text_config_names += [text_config_name]

        if len(valid_text_config_names) > 1:
            raise ValueError(
                f"Multiple valid text configs were found in the model config: {valid_text_config_names}. In this "
                "case, using `get_text_config()` would be ambiguous. Please specify the desired text config directly, "
                "e.g. `text_config = config.sub_config_name`"
            )
        elif len(valid_text_config_names) == 1:
            config_to_return = getattr(self, valid_text_config_names[0])
        else:
            config_to_return = self
```

세 모델 모두 `text_config` 를 갖고 있으므로 `get_text_config()` 는 텍스트 서브컨피그를
돌려주며, 그 위에는 `pad_token_id` 가 있다(재현 출력의 마지막 줄들). tevatron 은 이 메서드를
부르지 않는다 — 소스 전체에 `get_text_config` 문자열이 없다.

### 1.4 왜 AutoModel 이 텍스트 config 로 바꿔주지 않는가

`TF/transformers/models/auto/auto_factory.py:388-396`

```text
        elif has_local_code:
            model_class = _get_model_class(config, cls._model_mapping)
            text_config_class = config.sub_configs.get("text_config", None)
            # getattr avoids AttributeError, as registered remote-code model classes may lack config_class
            if text_config_class is not None and getattr(model_class, "config_class", None) == text_config_class:
                # TODO: Validate that copying the parent quantization config to the text sub-config preserves
                # modules_to_not_convert and skip-module matching when composite-model module prefixes differ.
                parent_config = config
                config = config.get_text_config()
```

교체 조건은 `model_class.config_class == text_config_class` 다. `AutoModel` 매핑은
(`TF/transformers/models/auto/modeling_auto.py:184,417,427`)

```
        ("gemma4", "Gemma4Model"),
        ("qwen3_5", "Qwen3_5Model"),
        ("qwen3_vl", "Qwen3VLModel"),
```

이고 이 세 클래스의 `config_class` 는 핀 위에서 확인한 결과 전부 **합성** config 다.

```
Gemma4Model <class 'transformers.models.gemma4.configuration_gemma4.Gemma4Config'>
Qwen3_5Model <class 'transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5Config'>
Qwen3VLModel <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'>
```

그러므로 교체가 일어나지 않고 `base_model.config` 는 합성 config 그대로다. 세 칸이 같은
자리에서 같은 이유로 죽는 이유가 이것이다.

---

## 2. `hf_kwargs` 가 흐르는 정확한 줄 — shim 이 설 자리

| 위치 | 줄 | 무엇을 받는가 |
|---|---|---|
| `TEV/src/tevatron/retriever/modeling/encoder.py` | 124 | `cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)` (`build`) |
| `TEV/src/tevatron/retriever/modeling/encoder.py` | 166 | `cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)` (`load`) |
| `TEV/src/tevatron/retriever/modeling/encoder.py` | 131 / 170 | `LoraConfig.from_pretrained(..., **hf_kwargs)` — **같은 dict 가 peft 로도 간다** |

131행·170행이 중요하다. `hf_kwargs` 에 `dtype` / `attn_implementation` / `config` 같은
transformers 전용 키를 넣으면 LoRA 경로에서는 그 dict 가 `LoraConfig.from_pretrained` 로도
흘러간다. **LoRA 축을 켤 때 `hf_kwargs` 로 config 를 심는 방식은 이 줄과 충돌할 수 있다.**
(LoRA 경로에서 실제로 어떻게 깨지는지는 이 호스트에서 확인하지 못했다 — 아래 §7.)

`from_pretrained` 쪽에서 `config` 키가 받아들여진다는 근거는
`TF/transformers/models/auto/auto_factory.py:261-262`

```text
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], *model_args, **kwargs):
        config = kwargs.pop("config", None)
```

그리고 `TF/transformers/models/auto/auto_factory.py:324-343`

```text
        if not isinstance(config, PreTrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype") == "auto":
                _ = kwargs.pop("torch_dtype")
            if kwargs.get("dtype") == "auto":
                _ = kwargs.pop("dtype")
            # to not overwrite the quantization_config if config has a quantization_config
            if kwargs.get("quantization_config") is not None:
                _ = kwargs.pop("quantization_config")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                code_revision=code_revision,
                _commit_hash=commit_hash,
                **hub_kwargs,
                **kwargs,
            )
```

`PreTrainedConfig` 인스턴스를 넘기면 `AutoConfig.from_pretrained` 를 건너뛰고 그 객체를
그대로 쓴다.

**shim 이 설 수 있는 자리는 소스상 세 곳이다.** 어느 것을 고를지는 레인의 판단이고, 여기서는
자리와 그 자리의 원문만 기록한다.

1. `hf_kwargs` 에 `config=<미리 만든 config 인스턴스, pad_token_id 세팅됨>` 을 넣는다.
   근거: `auto_factory.py:262`, `auto_factory.py:324`. 비-LoRA 경로에서만 안전하다(위 131/170행).
2. `DenseModel.TRANSFORMER_CLS` 를 `from_pretrained` 를 감싼 클래스로 갈아끼운다.
   `encoder.py:27` 이 클래스 속성이고 `dense.py` 가 오버라이드하지 않으므로 대입만으로 바뀐다.
3. `DenseModel.load` 를 부르지 않고 `AutoModel.from_pretrained` → `config.pad_token_id = 0`
   → `DenseModel(encoder=..., pooling=..., normalize=...)` 순으로 직접 조립한다.
   `EncoderModel.__init__` 은 `encoder.py:34-50` 에 있고 `PreTrainedModel` 하나만 요구한다.

3번의 근거 원문 — `TEV/src/tevatron/retriever/modeling/encoder.py:34-50`

```text
    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
```

`load` 는 `temperature` 를 인자로도 받지 않는다(`encoder.py:159-165`) — `load` 로 만든 모델의
`self.temperature` 는 항상 기본값 `1.0` 이다. 이 저장소의 하네스는 자기 loss 를 쓰므로
지금은 무해하지만, tevatron 자체 `forward` 로 측정하려 들면 온도 축이 config 에서 오지 않는다.

---

## 3. `DenseModel.forward` 전체와 책임 분해

`DenseModel` 은 `forward` 를 정의하지 않는다. 상속받는 `EncoderModel.forward` 가 전부다.

`TEV/src/tevatron/retriever/modeling/encoder.py:52-87`

```text
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores / self.temperature, target)
            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
```

그리고 그것이 부르는 것들 — `TEV/src/tevatron/retriever/modeling/dense.py:18-46`

```text
    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
        

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
```

`TEV/src/tevatron/retriever/modeling/encoder.py:95-115`

```text
    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
```

### 3.1 줄 단위 책임표

| 책임 | tevatron 이 하는가 | 원문 위치 |
|---|---|---|
| 인코딩 (백본 forward) | 한다 | `dense.py:19-20` `self.encoder(**qry, return_dict=True)` → `.last_hidden_state` |
| 풀링 | 한다 | `dense.py:28-43` `_pooling`, `cls/first`·`mean/avg/average`·`last/eos` 셋 |
| 정규화 | 한다, **풀링 안에서 조건부로** | `dense.py:44-45` `if self.normalize: reps = ...normalize(reps, p=2, dim=-1)` |
| 스코어링 | 한다 | `encoder.py:69-70` `compute_similarity` → `torch.matmul(q, p.T)` 후 `view(q.size(0), -1)` |
| 온도 나눗셈 | 한다 | `encoder.py:75` `self.compute_loss(scores / self.temperature, target)` |
| InfoNCE | 한다, **`CrossEntropyLoss` + 대각 타깃으로** | `encoder.py:46` `nn.CrossEntropyLoss(reduction='mean')`, `encoder.py:72-73` 타깃 생성, `encoder.py:98-99` |
| 분산 게더 | 한다, **`self.training` 이고 `is_ddp` 일 때만** | `encoder.py:65-67` + `encoder.py:104-115` |
| DDP 손실 보정 | 한다 | `encoder.py:76-77` `loss = loss * self.world_size` |
| 그래디언트 체크포인팅 위임 | 한다 | `encoder.py:101-102` |

주의할 세부 셋:

- **`if query:` 는 `is not None` 이 아니다** (`encoder.py:53-54`). 빈 dict 는 falsy 라 조용히
  `None` 경로로 빠진다.
- **`is_ddp` 는 `__init__` 시점에 한 번 고정된다** (`encoder.py:47` `self.is_ddp = dist.is_initialized()`).
  모델을 만든 뒤에 프로세스 그룹을 초기화하면 게더가 영영 켜지지 않는다.
- **`last/eos` 풀링의 좌패딩 판정은 마스크에서 추론한다** (`dense.py:35`
  `left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])`). 우패딩인데 배치의
  모든 행이 최대 길이면 이 식이 참이 되어 좌패딩으로 오판한다. 선언된 padding_side 를 받지
  않는다.

### 3.2 이 저장소의 하네스와의 대응

하네스는 같은 일을 네 조각으로 나눠 갖고 있고, 그 중 셋은 tevatron 과 겹친다.

| 하네스 | 위치 | tevatron 대응 |
|---|---|---|
| `encode` (백본 forward + hidden 선택) | `/Users/jwcho/Codes/train-comparison/trainbench/probe/steps.py:181-195` | `dense.py:19-20` |
| `last_token_pool` (풀링, 선언된 side 검증) | `/Users/jwcho/Codes/train-comparison/trainbench/embedding.py:22-...` | `dense.py:34-41` (`last/eos`) |
| `info_nce` (정규화 + 스코어 + 온도 + CE) | `/Users/jwcho/Codes/train-comparison/trainbench/embedding.py:201-216` | `dense.py:44-45` + `encoder.py:69-75,98-99` |
| `infonce_backward` (backward + 학습 여부 검증) | `/Users/jwcho/Codes/train-comparison/trainbench/probe/steps.py:389-438` | tevatron 쪽 대응 없음 |

`trainbench/probe/steps.py:190-195`

```text
    output = model(**batch, output_hidden_states=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(output, "hidden_states", None)
        hidden = hidden[-1] if hidden else output[0]
    return last_token_pool(hidden, batch["attention_mask"], padding_side=padding_side)
```

`trainbench/embedding.py:212-216`

```text
    queries = F.normalize(queries, dim=-1)
    documents = F.normalize(documents, dim=-1)
    logits = queries @ documents.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
```

**분산 게더는 하네스에 없다.** tevatron 은 `forward` 안에서 하고(`encoder.py:65-67`),
하네스는 어디서도 `all_gather` 를 부르지 않는다. 다중 GPU 로 넘어갈 때 이 축이
프레임워크마다 다른 자리에 있다는 사실이 처리량 비교의 교란 요인이 된다.

### 3.3 pad_token_id 를 고쳐도 그 다음에 걸리는 것 — 시그니처 불일치

`steps.encode` 는 토크나이저 배치를 그대로 편다: `model(**batch, output_hidden_states=False)`
(`trainbench/probe/steps.py:190`). 배치 키는 `input_ids` / `attention_mask` 다
(`trainbench/probe/steps.py:198-202`). 그런데 `DenseModel` 이 상속한 forward 는

```text
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
```

(`TEV/src/tevatron/retriever/modeling/encoder.py:52`) — `input_ids` 라는 이름의 인자가 없다.

`trainbench/probe/tevatron.py:44-57` 은 `DenseModel.load` 가 돌려준 객체를 그대로
`steps.verify_axes` 에 넘기고 그 반환값을 `infonce_backward` 로 보낸다. `trainbench/` 전체에
`.encoder` 로 내부 백본을 꺼내는 코드는 없다(`grep -rn "\.encoder\b" trainbench/` 0 히트).
`verify_axes` 는 "peft, torch.compile, FSDP 가 모델을 갈아끼울 수 있어서" 모델을 되돌려줄 뿐
언랩하지 않는다(`trainbench/probe/steps.py:81-108`).

**그러므로 pad_token_id 만 고치면 `dense_model_load` 는 초록이 되지만 `infonce_backward` 는
`TypeError` 로 넘어간다.** 2차 캠페인은 `dense_model_load` 에서 멈춰서 여기까지 오지 못했다.
probe 레인은 두 가지를 한 번에 준비해야 한다. (실제 예외 메시지는 이 호스트에서 확인하지
못했다 — §7.)

---

## 4. 상류 선언 의존성 vs 실제 최상단 import

`TEV/setup.py:1-20` 전문

```text
from setuptools import setup, find_packages

setup(
    name='tevatron',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/texttron/tevatron',
    license='Apache 2.0',
    author='Luyu Gao',
    author_email='luyug@cs.cmu.edu',
    description='Tevatron: A toolkit for learning and running deep dense retrieval models.',
    python_requires='>=3.7',
    install_requires=[
        "transformers>=4.10.0",
        "datasets>=1.1.3"
    ]
)
```

`pyproject.toml` 도 `setup.cfg` 도 없다(`ls` 결과 둘 다 없음). 선언은 이 둘이 전부다.
lock 도 그대로 받아 적었다(`envs/tevatron/uv.lock:1047-1050` 의 `dependencies` 가
`datasets`, `transformers` 뿐).

`import tevatron` → `tevatron.retriever.modeling` 경로가 최상단에서 실제로 요구하는 것:

| 모듈 | 줄 | import |
|---|---|---|
| `encoder.py` | 4 | `import torch` |
| `encoder.py` | 5 | `import torch.distributed as dist` |
| `encoder.py` | 6 | `from torch import nn, Tensor` |
| `encoder.py` | 8 | `from transformers import PreTrainedModel, AutoModel` |
| `encoder.py` | **9** | `from peft import LoraConfig, TaskType, get_peft_model, PeftModel` |
| `encoder.py` | 11 | `from transformers.file_utils import ModelOutput` |
| `splade.py` | 5 | `from peft import LoraConfig, PeftModel, TaskType, get_peft_model` |
| `unicoil.py` | 4 | `from transformers import BertPreTrainedModel, BertModel` |
| `arguments.py` | 4 | `from transformers import TrainingArguments` |

`modeling/__init__.py` 는 넷을 전부 끌어온다:

```text
from .encoder import EncoderModel, EncoderOutput
from .dense import DenseModel, MultiModalDenseModel
from .unicoil import UniCoilModel
from .splade import SpladeModel, SpladeModelForCausalLM
```

**선언되지 않은 것: `torch`, `peft`.** 1차 캠페인이 죽은 자리가 `peft` 다.
`envs/tevatron/pyproject.toml:15` 이 `peft>=0.20` 을 대신 선언해서 막아둔 상태이고, lock 은
`peft 0.20.0` / `accelerate 1.14.0` / `datasets 5.0.1` / `torch 2.13.0+cu130` 을 고정한다.

패키지 전체(드라이버·평가·jax 경로 포함)로 넓히면 선언되지 않은 최상단 import 가 더 있다:
`numpy`, `tqdm`, `faiss`, `safetensors`, `yaml`, `pydantic`, `httpx`, `simple_parsing`,
`vllm`, `jax`/`flax`/`optax`/`chex`/`orbax`, `magix`, `grad_cache`, `megatron`, `PIL`.
`retriever/modeling` 만 건드리는 한 이것들은 로드되지 않지만, `tevatron.retriever.driver.*`
나 `tevatron.eval.*` 를 건드리는 순간 같은 종류의 실패가 다시 난다.

**AGENTS.md 와의 불일치 하나.** `docs/support-matrix.md:1006-1009` 는
"`encoder.py`는 8행에서 `from peft import ...`" 라고 적는다. 핀 원문에서 그 줄은 **9행**이고
8행은 `from transformers import PreTrainedModel, AutoModel` 이다. 인용 자체는 맞고 줄 번호만
하나 어긋난다.

---

## 5. `framework_version` 이 "unknown" 인 이유

기록하는 쪽 — `/Users/jwcho/Codes/train-comparison/trainbench/probe/types.py:96-105`

```text
    def add_version(self, module: Any) -> None:
        """Record the framework's own version. Each image ships a different stack,
        so this travels with the result rather than being assumed."""
        self.add(
            Check(
                name="framework_version",
                ok=True,
                detail={"version": getattr(module, "__version__", "unknown")},
            )
        )
```

부르는 쪽 — `/Users/jwcho/Codes/train-comparison/trainbench/probe/tevatron.py:29-31`

```text
    import tevatron

    report.add_version(tevatron)
```

핀 원문 — `TEV/src/tevatron/__init__.py` 는 **0 바이트, 완전히 빈 파일**이다.
그리고 소스 트리 전체에 `__version__` 이라는 문자열이 하나도 없다:

```
$ grep -rn "__version__" /Users/jwcho/.cache/uv/git-v0/checkouts/af8e1386372d71f4/dd06310/src/
(출력 없음)
```

버전은 오직 `setup.py:5` 의 `version='0.0.1'` 에만 있고, 그것은 **패키지 메타데이터**
(`.dist-info/METADATA`)로 들어갈 뿐 런타임 모듈 속성이 되지 않는다. 그래서
`getattr(tevatron, "__version__", "unknown")` 이 폴백을 돌려준다.

정정 방법은 소스가 아니라 메타데이터를 읽는 것이다 —
`importlib.metadata.version("tevatron")` 이 `0.0.1` 을 돌려준다. 다만 `0.0.1` 은 git HEAD 를
식별하지 못하므로(`docs/support-matrix.md:113` 이 이미 이 점을 지적한다), 이 프레임워크에서
결과에 실어야 할 값은 버전 문자열이 아니라 **lock 의 커밋 sha**
(`dd063104c81a76d6a77c845f667b46b9e5abd625`)다. 그 sha 는 `uv.lock` 에 있고,
`.dist-info/direct_url.json` 에도 남는지는 이 호스트에서 확인하지 못했다(§7).

---

## 6. 레인별 요약

### probe 레인

1. `dense_model_load` 실패는 `encoder.py:167` 의 직접 속성 접근이다. 상류를 고칠 수 없으니
   shim 이 필요하고, `@strict` 가 setattr 를 막지 않으므로 `config.pad_token_id = 0` 을
   미리 심는 방식이 소스상 성립한다(§1.2 재현).
2. 그것만 고치면 다음 벽은 `EncoderModel.forward(query, passage)` 시그니처다(§3.3).
   두 개를 한 번에 준비하지 않으면 파드 한 시간을 또 쓴다.
3. `framework_version` 은 `importlib.metadata` 로 바꿔도 `0.0.1` 이다. 결과에 실을 값은
   커밋 sha 다(§5).

### adapters 레인

1. `hf_kwargs` 는 `from_pretrained` 뿐 아니라 LoRA 경로에서 `LoraConfig.from_pretrained`
   로도 간다(`encoder.py:131`, `encoder.py:170`). LoRA 축에서 `hf_kwargs` 로 transformers
   전용 키를 밀어 넣는 설계는 여기서 부딪힌다.
2. `EncoderModel.load` 는 `temperature` 를 받지 않는다. `load` 로 만든 모델의 온도는 항상
   `1.0` 이다(`encoder.py:159-165`, `encoder.py:38`).
3. tevatron 은 풀링·정규화·스코어·InfoNCE·DDP 게더를 **한 `forward` 안에** 갖고 있고,
   이 저장소는 넷으로 쪼개 갖고 있다. 게더는 하네스 쪽에 대응이 없다(§3.2).
4. `last/eos` 풀링의 좌패딩 판정은 선언된 padding_side 가 아니라 마스크 추론이다
   (`dense.py:35`). 하네스의 `last_token_pool` 은 선언된 side 와 마스크가 어긋나면 예외를
   던진다 — 두 정책이 다르다.
5. `is_ddp` 는 `__init__` 시점 고정이다(`encoder.py:47`).

---

## 7. 이 호스트에서 확정하지 못한 것 — 파드/이미지가 답해야 한다

- `DenseModel.load` 에 `config=<pad_token_id 를 심은 인스턴스>` 를 넘겼을 때 실제 체크포인트
  세 종(`qwen3_vl_emb_2b`, `qwen3_5_0_8b`, `gemma4_e2b`)이 끝까지 적재되는가. 여기서는
  가중치를 받지 않아 `from_pretrained` 를 돌리지 못했다.
- 그 shim 이 통과한 뒤 `steps.encode` 의 `model(**batch)` 가 실제로 어떤 예외 문구로 죽는가,
  그리고 `axes.assemble` 이 peft/compile 로 모델을 갈아끼운 뒤에도 같은 시그니처가 남는가.
- LoRA 축을 켠 `DenseModel.build` 경로에서 `hf_kwargs` 가 `LoraConfig.from_pretrained` 로
  흘러갈 때 어떤 키가 거부되는가. peft 0.20.0 소스를 이번 조사에서 열지 않았다.
- `AutoModel.from_pretrained` 가 세 체크포인트의 `config.json` 을 읽었을 때 `text_config`
  안의 `pad_token_id` 가 실제로 무엇인가. 여기서 본 값은 **기본 생성자**의 값이다
  (qwen 둘 `None`, gemma-4 `0`) — 체크포인트가 무엇을 적어두었는지는 다르다.
- `tevatron` 의 설치된 `.dist-info/direct_url.json` 에 커밋 sha 가 남는가.
  이 호스트에는 tevatron 이 설치돼 있지 않다(`ls ~/.cache/uv/archive-v0/*/tevatron-*.dist-info`
  → no matches). 리눅스 이미지에서만 확인할 수 있다.
- `MultiModalDenseModel` 은 `Qwen2_5OmniThinkerForConditionalGeneration` 에 묶여 있고
  (`dense.py:8-13`) `self.encoder.visual` / `self.encoder.audio_tower` 를 요구한다
  (`dense.py:64-68`). 이 세 모델에 그 경로를 쓸 수 있는지는 확인하지 않았다.
