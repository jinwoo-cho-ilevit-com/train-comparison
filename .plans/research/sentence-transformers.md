# sentence-transformers 5.6.1 — 핀 원문 리서치

소비 레인: `probe`, `adapters`.

## 0. 핀 해석 (이 문서의 모든 인용이 어디서 나왔는가)

`envs/sentence-transformers/uv.lock:719-731`:

```
[[package]]
name = "sentence-transformers"
version = "5.6.1"
source = { registry = "https://pypi.org/simple" }
...
wheels = [
    { url = "https://files.pythonhosted.org/packages/c1/ad/8f73f512dc7ad4031d2b64cbb67f70bdfb355756afbe0db610a5146415c1/sentence_transformers-5.6.1-py3-none-any.whl", hash = "sha256:cefbb17b6325a982a4732c8c49fb013375392687049d1de3d435c4b04060680b", size = 596677, upload-time = "2026-07-23T14:40:40.312Z" },
]
```

**호스트 uv 캐시에는 핀이 없다.** `ls -d ~/.cache/uv/archive-v0/*/sentence_transformers-*.dist-info` 가 돌려준 것은
전부 디코이였다:

```
/Users/jwcho/.cache/uv/archive-v0/C37orLAMzH5uk-8c/sentence_transformers-5.6.0.dist-info
/Users/jwcho/.cache/uv/archive-v0/hNsNcEmpmWlRYY4w/sentence_transformers-5.5.1.dist-info
/Users/jwcho/.cache/uv/archive-v0/I22-JNvk8EZphcI2/sentence_transformers-5.6.0.dist-info
/Users/jwcho/.cache/uv/archive-v0/sBl7SsiVCyExBEal/sentence_transformers-5.2.0.dist-info
```

5.6.0 과 5.6.1 사이에 이 브리프가 답하는 질문 대부분이 걸려 있으므로(모듈 레이아웃이 5.4.0에서
통째로 재배치되었다 — 아래 `util/deprecated_import.py` 인용 참조) 5.6.0 을 대신 읽는 것은 무효다.
받아서 sha256 을 대조했다:

```
$ curl -sSL -o .../pins/sentence_transformers-5.6.1-py3-none-any.whl \
    https://files.pythonhosted.org/packages/c1/ad/.../sentence_transformers-5.6.1-py3-none-any.whl
$ shasum -a 256 .../pins/sentence_transformers-5.6.1-py3-none-any.whl
cefbb17b6325a982a4732c8c49fb013375392687049d1de3d435c4b04060680b
```

lock 의 `hash = "sha256:cefbb17b...680b"` 와 일치. 압축 해제 위치(스크래치패드, 저장소 밖):

```
/private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/pins/sentence-transformers-5.6.1/
```

이하 인용의 절대경로는 모두 이 디렉터리 하위다. `SRC` 로 줄여 쓰지 않고 전부 적는다.

> `uv pip download` 는 이 uv 빌드에 없다(`error: unrecognized subcommand 'download'`). 그래서 URL 을
> lock 에서 그대로 읽어 `curl` 로 받고 sha256 을 대조했다. 대조 대상은 동일하다.

### 이 lock 이 함께 고정한 것

`envs/sentence-transformers/uv.lock` 전체 패키지 목록(`grep -n '^name = '`)에 **`peft`, `datasets`,
`accelerate` 가 없다.** 있는 것은 `torch 2.13.0+cu130`, `transformers 5.14.1`, `tokenizers 0.22.2`,
`torchvision`, `pillow`, `scikit-learn`, `scipy` 그리고 `flash-linear-attention` / `causal-conv1d`
(`envs/sentence-transformers/pyproject.toml:8-14`).

이것이 4·6번 답의 전제다. 휠의 `METADATA` 가 그 셋을 extra 로 분리해 두었기 때문이다 —
`/private/tmp/.../sentence-transformers-5.6.1/sentence_transformers-5.6.1.dist-info/METADATA:36-38`:

```
Provides-Extra: train
Requires-Dist: datasets>=2.0.0; extra == "train"
Requires-Dist: accelerate>=0.20.3; extra == "train"
```

`envs/sentence-transformers/pyproject.toml:10` 은 `"sentence-transformers>=5.6"` 만 선언한다. extra 없음.

---

## 1. `SentenceTransformer.__init__` 이 `model_kwargs` 를 `from_pretrained` 로 넘기는 경로

여섯 프레임워크 중 유일하게 적재 시점 축(`attn_implementation`, `quantization_config`)을 존중할 수 있는
경로라는 저장소의 주장(`trainbench/probe/sentence_transformers.py:42-51`)은 **핀에서 확인된다.**
경로는 다섯 단계다.

**(1) `__init__` 는 `model_kwargs` 를 그대로 `super().__init__()` 에 넘긴다.**
`/private/tmp/claude-501/-Users-jwcho-Codes-train-comparison/528669dc-58ea-4ea9-b391-9c18fa5ed7a9/scratchpad/pins/sentence-transformers-5.6.1/sentence_transformers/sentence_transformer/model.py:148-204`

```python
    @deprecated_kwargs(tokenizer_kwargs="processor_kwargs")
    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        modules: list[nn.Module] | None = None,
        device: str | None = None,
        ...
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        ...
    ) -> None:
        ...
        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
        )
```

**(2) `BaseModel.__init__` → `_load_modules(...)`.**
`.../sentence_transformers/base/model.py:215-226`

```python
        if model_name_or_path:
            modules, self.module_kwargs = self._load_modules(
                model_name_or_path,
                token=token,
                cache_folder=cache_folder,
                revision=revision,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                model_kwargs=model_kwargs,
                processor_kwargs=processor_kwargs,
                config_kwargs=config_kwargs,
            )
```

같은 함수는 `model_kwargs["device_map"]` 를 **먼저** 읽어 `device` 인자를 무력화한다.
`.../sentence_transformers/base/model.py:184-195`

```python
        # A `device_map` in `model_kwargs` makes accelerate control placement (not `device`), so we detect it
        # here to skip the `self.to(device)` below that would otherwise pull a `device_map="cuda:1"` model to cuda:0.
        device_map = (model_kwargs or {}).get("device_map") if (backend == "torch" and model_name_or_path) else None
        device_provided = device is not None
        if device is None and device_map is None:
            device = get_device_name()
            logger.info(f"No device provided, using {device}")
        elif device_provided and device_map is not None:
            logger.warning(
                "Both `device` and `model_kwargs['device_map']` were provided. `device_map` controls "
                "device placement, so the `device` argument is ignored."
            )
```

프로브가 `device=str(device)` 와 `model_kwargs=load_kwargs` 를 함께 넘기므로(`trainbench/probe/sentence_transformers.py:46-51`),
`load_kwargs` 에 `device_map` 이 들어가는 순간 `device` 는 조용히 무시되고 경고만 남는다.
현재 `trainbench/axes.py:590-604` 의 `load_kwargs` 는 `attn_implementation` 과 `quantization_config` 만
넣으므로 지금은 충돌하지 않는다.

**(3) `_load_modules` 는 `modules.json` 유무로 갈린다.**
`.../sentence_transformers/base/model.py:957-1005`

```python
        load_kwargs = {
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "model_kwargs": model_kwargs,
            "processor_kwargs": processor_kwargs,
            "config_kwargs": config_kwargs,
        }

        # Check if this is a Sentence Transformer model
        modules_json_path = load_file_path(...)
        if modules_json_path is None:
            logger.info(f"No modules.json found for {model_name_or_path}, initializing a new {self.model_type} model.")
            return self._load_default_modules(model_name_or_path, **load_kwargs)

        model_type_being_loaded = self._get_model_type(...)
        if model_type_being_loaded == self.model_type:
            logger.info(f"Loading {self.model_type} model from {model_name_or_path}.")
            return self._load_config_modules(model_name_or_path, **load_kwargs)

        logger.info(f"Converting {model_type_being_loaded} model {model_name_or_path} to {self.model_type}.")
        return self._load_converted_modules(model_name_or_path, **load_kwargs, model_type=model_type_being_loaded)
```

두 갈래 모두 `model_kwargs` 를 들고 간다.

**(4-a) `modules.json` 이 없을 때 — 우선순위 병합 없이 그대로.**
`.../sentence_transformers/sentence_transformer/model.py:1056-1073`

```python
        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = {**shared_kwargs} if model_kwargs is None else {**shared_kwargs, **model_kwargs}
        processor_kwargs = {**shared_kwargs} if processor_kwargs is None else {**shared_kwargs, **processor_kwargs}
        config_kwargs = {**shared_kwargs} if config_kwargs is None else {**shared_kwargs, **config_kwargs}

        transformer_model = Transformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
            backend=self.backend,
        )
```

`Transformer.__init__` 시그니처에 `cache_dir` 은 **없다**. 데코레이터가 먼저 뽑아 세 dict 에 분배한다 —
`.../sentence_transformers/util/decorators.py:76-85`:

```python
        if "cache_dir" in kwargs:
            cache_dir = kwargs.pop("cache_dir")
            if cache_dir is not None:
                logger.warning(
                    "The Transformer `cache_dir` argument is deprecated. "
                    "Please pass `cache_dir` via `model_kwargs`, `processor_kwargs`, and/or `config_kwargs` instead."
                )
                for dict_name in ("model_kwargs", "processor_kwargs", "config_kwargs"):
                    kwargs.setdefault(dict_name, {})
                    kwargs[dict_name].setdefault("cache_dir", cache_dir)
```

**(4-b) `modules.json` 이 있을 때 — 사용자 `model_kwargs` 가 1순위.**
`.../sentence_transformers/base/modules/transformer.py:2078-2113`

```python
        """Build the kwargs dict for ``__init__`` by merging config file, hub kwargs, and caller overrides.

        Priority (highest to lowest): caller kwargs > hub kwargs > config file values.
        """
        ...
        # 2nd priority: hub_kwargs
        config["model_kwargs"].update(hub_kwargs)
        config["processor_kwargs"].update(hub_kwargs)
        config["config_kwargs"].update(hub_kwargs)

        # 1st priority: kwargs passed to SentenceTransformer
        if model_kwargs:
            config["model_kwargs"].update(model_kwargs)
```

같은 파일 `:2151-2163` 은 config 파일이 `trust_remote_code` 를 켤 수 없게 강제로 뽑아낸다.

**(5) 최종 착지점 — `from_pretrained`.**
`.../sentence_transformers/base/modules/transformer.py:654-656`

```python
        self.model = self._load_model(
            model_name_or_path, transformer_task, config, backend, is_peft_model, **model_kwargs
        )
```

`.../sentence_transformers/base/modules/transformer.py:1738-1751`

```python
        if backend == "torch":
            # When loading a PEFT model, we load the base model first. The revision
            # (e.g. "main") refers to the adapter checkpoint, not the base model, so
            # we must not pass it to the base model's from_pretrained.
            if is_peft_model:
                model_kwargs.pop("revision", None)

            if transformer_task == "feature-extraction":
                model = self._load_encoder_only_model(model_name_or_path, config, **model_kwargs)
                if model is not None:
                    return model

            model_cls = TRANSFORMER_TASK_TO_AUTO_MODEL[transformer_task]
            return model_cls.from_pretrained(model_name_or_path, config=config, **model_kwargs)
```

어떤 `AutoModel` 인지는 `transformer_task` 가 정한다. SentenceTransformer 기본값은 `feature-extraction`
(`transformer.py:593`) → `AutoModel`. `.../sentence_transformers/base/modules/transformer.py:147-161`:

```python
TRANSFORMER_TASK_TO_AUTO_MODEL: dict[TransformerTask, Any] = {
    "feature-extraction": AutoModel,  # Used by SentenceTransformer, also covers "image-feature-extraction"
    "sequence-classification": AutoModelForSequenceClassification,  # Used by CrossEncoder
    "text-generation": AutoModelForCausalLM,  # Used by CrossEncoder
    "fill-mask": AutoModelForMaskedLM,  # Used by SparseEncoder
}

try:
    from transformers import AutoModelForMultimodalLM

    TRANSFORMER_TASK_TO_AUTO_MODEL["any-to-any"] = (
        AutoModelForMultimodalLM  # Used by CrossEncoder, also covers "image-text-to-text"
    )
except ImportError:
    pass
```

**probe 레인에 주는 결론**: `model_kwargs` 는 중간에서 필터링되지 않는다. `attn_implementation`,
`quantization_config`, `dtype` 은 그대로 `AutoModel.from_pretrained` 에 도달한다. 단, `SentenceTransformer`
는 `AutoModelForCausalLM` 을 절대 쓰지 않으므로 생성형 VLM 도 `AutoModel` 로 적재된다.

---

## 2. module layout 이 없는 생성형 VLM 체크포인트를 받으면 무엇으로 떨어지는가

`modules.json` 이 없으면 `_load_default_modules` 다.
`.../sentence_transformers/sentence_transformer/model.py:1036-1040` (독스트링) 와 `:1074-1088`:

```python
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules, except for
        CausalLM-based models which use Last Token pooling instead.

        This is used as a fallback when no pre-trained SentenceTransformer model is found.
```

```python
        modules = [transformer_model]
        if transformer_model.module_output_name == "token_embeddings":
            config = transformer_model.config
            # If a model was originally designed for causal language modeling, then we use last token pooling,
            # except if is_causal=False, then it's still bidirectional and we default to mean pooling.
            is_causal_lm = (
                getattr(config, "architectures", None)
                and config.architectures[0].endswith("ForCausalLM")
                and getattr(config, "is_causal", True)
            )
            pooling_mode = "lasttoken" if is_causal_lm else "mean"
            modules.append(Pooling(transformer_model.get_embedding_dimension(), pooling_mode))
        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules, {}
```

확정되는 것:

- 폴백은 **`[Transformer, Pooling]` 2개 모듈**이고 Dense/Normalize 는 붙지 않는다.
- pooling mode 는 `config.architectures[0]` 의 **문자열 접미사**로 갈린다. `...ForCausalLM` 으로 끝나면
  `"lasttoken"`, 아니면 `"mean"`. `is_causal=False` 면 `ForCausalLM` 이어도 `"mean"`.
- **`Pooling` 이 아예 안 붙는 분기가 있다**: `module_output_name != "token_embeddings"` 이면 `if` 가 통째로
  건너뛰어져 모듈이 `[Transformer]` 하나뿐이 된다. 그러면 `model(features)["sentence_embedding"]` 은
  `Transformer` 가 그 키를 직접 채웠을 때만 존재한다.

`module_output_name` 은 task 기본값 또는 modality 추론이 정한다.
`.../sentence_transformers/base/modules/transformer.py:166-187`:

```python
TRANSFORMER_TASK_DEFAULTS: dict[TransformerTask, tuple[ModalityConfig, str]] = {
    "feature-extraction": (
        {"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
        "token_embeddings",
    ),
```

일반 추론 경로 — `.../sentence_transformers/base/modules/transformer.py:1846-1877`:

```python
        output_fields = self._get_method_output_fields(model.forward)
        if output_fields is None or default_method_output_name in output_fields:
            modality_config: ModalityConfig = {}
            for modality in modalities:
                entry = ModalityParams(method="forward", method_output_name=default_method_output_name)
                if modality == "message":
                    entry["format"] = self.input_formatter.message_format
                modality_config[modality] = entry
            return modality_config, default_module_output_name

        # For feature-extraction, if there's no 'last_hidden_state', we can check for modality-specific methods like get_..._features
        if self.transformer_task == "feature-extraction":
            modality_config: ModalityConfig = {}
            for modality in modalities:
                if modality == "message":
                    continue

                method_name = f"get_{modality}_features"
                if hasattr(model, method_name):
                    method = getattr(model, method_name)
                    method_output_fields = self._get_method_output_fields(method)
                    if method_output_fields and "pooler_output" in method_output_fields:
                        modality_config[modality] = {"method": method_name, "method_output_name": "pooler_output"}
                    else:
                        modality_config[modality] = {"method": method_name, "method_output_name": None}

            return modality_config, "sentence_embedding"
```

즉 `AutoModel` 로 적재된 모델의 `forward` 반환 타입 어노테이션에 `last_hidden_state` 필드가 있으면
`token_embeddings` + Pooling 이고, **없으면 `get_*_features` 를 찾아 `sentence_embedding` 을 직접 만들고
Pooling 은 붙지 않는다.** `_get_method_output_fields` 는 `get_type_hints(method)` 로 반환 어노테이션을 읽고
실패하면 `None` 을 돌려준다(`transformer.py:1966-1992`) — `None` 이면 첫 분기(Pooling 있음)로 간다.

`google/gemma-4-E2B` 계열의 하드코딩 오버라이드는 없다. `_FEATURE_EXTRACTION_EDGE_CASES`
(`transformer.py:229-333`)에 등재된 model_type 은 `blip`, `blip-2`, `sam3`, `flava`, `git`,
`visual_bert`, `kosmos-2`, `grounding-dino`, `paligemma`, `vilt`, `layoutlmv3`, `idefics` 와 오디오 계열뿐이다.
`qwen3_vl`, `gemma4`, `qwen2_vl` 은 없다. 다만 `transformer.py:566-567` 의 `unpad_inputs` 독스트링이
`qwen2_vl` 을 "unpadded inputs 를 지원하지 않는 아키텍처"로 명시적으로 언급한다:

```
            padding, which is needed for architectures that don't support unpadded inputs (e.g.
            ``qwen2_vl``). Set to ``True`` to request unpadding explicitly; a warning is logged if the
```

**probe 레인에 주는 결론**: `has_module_layout = len(list(model)) > 1`
(`trainbench/probe/sentence_transformers.py:59`)은 "ST 설정이 있었다"가 아니라
"Pooling 이 붙었다"를 재는 것이다. `modules.json` 이 없고 `module_output_name == "sentence_embedding"`
으로 떨어진 경우에도 `False` 가 되므로, 두 상황이 구분되지 않는다. 어느 쪽인지 알려면
`type(model[0]).__name__` 과 `model[0].module_output_name`, `model[0].modality_config` 를 함께 기록해야 한다.
`pooling_mode` 는 `model[1].pooling_mode` 로 읽힌다(`pooling.py:111`).

---

## 3. `MultipleNegativesRankingLoss` / `CachedMultipleNegativesRankingLoss` vs `trainbench/embedding.py::info_nce`

### MNRL 본체

`.../sentence_transformers/sentence_transformer/losses/multiple_negatives_ranking.py:232-236`

```python
    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(embeddings, labels)
```

핵심 계산 — 같은 파일 `:237-262` 및 `:311-336`:

```python
    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        if len(embeddings) < 2:
            raise ValueError(f"Expected at least 2 embeddings, got {len(embeddings)}")

        queries = embeddings[0]
        docs = embeddings[1:]
        ...
        sim_matrices = {}
        # (bs, bs * ws * (1 + nn))
        sim_matrices["query_to_doc"] = self.similarity_fct(local_queries, docs_all)
```

```python
        # Apply temperature scaling (scale = 1/temperature) and add hardness penalties.
        # Final logit = cos_sim * scale + alpha * cos_sim (penalty is not temperature-scaled).
        for key in sim_matrices:
            sim_matrices[key] = sim_matrices[key] * self.scale
        for key, pen in penalties.items():
            sim_matrices[key] = sim_matrices[key] + pen

        # Positive scores (always from query_to_doc)
        positive_scores = sim_matrices["query_to_doc"][row_indices, local_indices]

        if self.partition_mode == "joint":
            # Single softmax over all selected directions
            scores = torch.cat(list(sim_matrices.values()), dim=1)
            log_z = torch.logsumexp(scores, dim=1)

        else:
            # Separate softmax for each direction, averaged
            log_z = 0.0
            for sim_matrix in sim_matrices.values():
                log_z += torch.logsumexp(sim_matrix, dim=1)
            log_z /= len(sim_matrices)

        loss = -(positive_scores - log_z).mean()
        return loss
```

기본값(`multiple_negatives_ranking.py:18-31`):

```python
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        gather_across_devices: bool = False,
        directions: tuple[
            Literal["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"],
            ...,
        ] = ("query_to_doc",),
        partition_mode: Literal["joint", "per_direction"] = "joint",
        hardness_mode: Literal["in_batch_negatives", "hard_negatives", "all_negatives"] | None = None,
        hardness_strength: float = 0.0,
    ) -> None:
```

같은 파일 `:65-68` 이 `scale` 의 의미를 못 박는다:

```
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: ``scale = 1 / temperature``, so
                ``scale=20.0`` is equivalent to ``temperature=0.05``. A higher scale (lower temperature) puts more emphasis
                on the positive example, and values between 10 and 100 are common.
```

### 우리 `info_nce`

`/Users/jwcho/Codes/train-comparison/trainbench/embedding.py:201-216`

```python
def info_nce(
    queries: torch.Tensor,
    documents: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """In-batch-negatives contrastive loss (MultipleNegativesRankingLoss).
    ...
    """
    queries = F.normalize(queries, dim=-1)
    documents = F.normalize(documents, dim=-1)
    logits = queries @ documents.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
```

### 차이 (기본 설정 기준)

| 항목 | ST `MultipleNegativesRankingLoss` (기본값) | `trainbench.embedding.info_nce` |
|---|---|---|
| 유사도 | `util.cos_sim` (L2 정규화 후 내적) | `F.normalize` 후 내적 — 수학적으로 동일 |
| 온도 | `* scale`, `scale=20.0` (= temperature 0.05) | `/ temperature`, config 값 |
| 방향 | `("query_to_doc",)` 하나 | query→doc 하나 |
| 분모 | `logsumexp` over `docs_all` = 모든 문서(+hard negatives) | `cross_entropy` 의 softmax 분모 = 배치 내 문서 전부 |
| 감축 | `-(positive - log_z).mean()` | `F.cross_entropy` 기본 `mean` |
| hard negatives | `embeddings[1:]` 전부를 문서 축에 concat 지원 | 없음. `documents` 한 텐서만 |
| 분산 gather | `gather_across_devices` 로 `all_gather_with_grad` | 없음 |
| hardness 가중 | `hardness_mode` / `hardness_strength` | 없음 |
| 임베딩 계산 | 손실 함수가 `self.model(...)` 를 직접 호출 | 이미 pooled 된 텐서를 받음 |

**기본 설정에서 두 손실은 수치적으로 같다** — 조건 셋이 전부 성립할 때다:
`temperature = 1/scale`, `partition_mode == "joint"`(기본값, 480행 — `per_direction` 이면
분모가 방향별로 쪼개져 단일 softmax 가 아니게 된다), hard negative 없음.
다른 점은 전부 ST 쪽의 추가 기능이 꺼져 있을 때 사라진다.

손실식 자체는 `multiple_negatives_ranking.py:338` 이고(`loss = -(positive_scores - log_z).mean()`),
`scale: float = 20.0` 기본값은 21행, `scale = 1 / temperature` 문서화는 docstring 65-67행이다.
(2026-08-02 감사가 인용 색인의 줄범위 오류를 잡아 여기 정정.) 그러므로 프로브가 ST 손실 대신 공유 `info_nce` 를 쓰는 선택
(`trainbench/probe/sentence_transformers.py:76-78`)은 이 버전에서도 유효하다.

**adapters 레인에 주는 실질적 차이 하나**: ST 손실은 `sentence_features` (전처리된 dict) 를 받아
**자기가 모델 forward 를 돌린다**. 우리 `info_nce` 는 pooled 텐서를 받는다. 따라서 ST 손실을 쓰는
순간 forward 호출 지점이 손실 함수 안으로 옮겨가고, GradCache 는 그 사실에 의존한다(아래).

### CachedMNRL — GradCache 3단계

`.../sentence_transformers/sentence_transformer/losses/cached_multiple_negatives_ranking.py:569-601`

```python
    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")

        reps = []
        self.random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps)

        return loss
```

미니배치 임베딩 — 같은 파일 `:388-405`:

```python
    def embed_minibatch(
        self, ..., with_grad: bool, copy_random_state: bool, random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed a mini-batch of inputs."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = _create_minibatch(sentence_feature, begin, end)
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mini_batch_size, dim)
        return reps, random_state
```

미니배치 손실의 정규화 — 같은 파일 `:560-566`:

```python
            per_sample_loss = -(positive_scores - log_z)
            loss_mbatch = per_sample_loss.mean() * len(local_batch) / batch_size

            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)
```

**측정 관점의 함의**: forward 가 두 번 돈다(step 1 은 `no_grad`, step 3 은 grad 있음). 이것이
GradCache 오버헤드가 프레임워크마다 다르게 보고되는 이유의 코드 근거다. 저장소가 이 축을 재려는
이유는 `trainbench/probe/sentence_transformers.py:97-98` 에 이미 적혀 있다.

`RandContext` 는 dropout 등의 난수 상태를 step 1 과 step 3 사이에 재현한다
(`cached_multiple_negatives_ranking.py:22-28`). **결정성 모드가 꺼진 측정 런에서도** 이 재현은
CachedMNRL 내부에서 항상 일어난다 — AGENTS.md 의 "deterministic mode is off during measurement" 와
독립적인, 손실 함수 자체의 동작이다.

---

## 4. 자체 Trainer 의 학습 스텝 진입점

**`training_step` 오버라이드는 없다.** 휠 전체에서 `def training_step` 은 0건이다
(`grep -rn "def training_step" .../sentence_transformers/` → no match). `SentenceTransformerTrainer` 는
HF `Trainer` 를 그대로 상속한다.

`.../sentence_transformers/base/trainer.py:76`

```python
class BaseTrainer(Trainer, ABC):
```

`.../sentence_transformers/sentence_transformer/trainer.py:36`

```python
class SentenceTransformerTrainer(BaseTrainer):
```

진입점은 **`compute_loss`** 다. `.../sentence_transformers/base/trainer.py:459-509`:

```python
    def compute_loss(
        self,
        model: BaseModel,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        ...
        dataset_name = inputs.pop("dataset_name", None)
        features, labels = self.collect_features(inputs)
        loss_fn = self.loss

        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]

        # Insert the wrapped (e.g. distributed or compiled) model into the loss function,
        # if the loss stores the model. Only called once per process
        if (
            model == self.model_wrapped
            and hasattr(loss_fn, "model")  # Only if the loss stores the model
            and loss_fn.model != model  # Only if the wrapped model is not already stored
        ):
            loss_fn = self.override_model_in_loss(loss_fn, model)
        loss = loss_fn(features, labels)
```

즉 **모델 forward 는 Trainer 가 아니라 손실 함수가 부른다.** `compute_loss` 는 `inputs` 를 컬럼별
feature dict 리스트로 쪼개 손실에 넘길 뿐이다. 쪼개는 규칙 — `.../sentence_transformers/base/trainer.py:591-605`:

```python
        # All inputs ending with one of these suffixes are considered to correspond to a feature
        feature_suffixes = (
            "input_ids",  # text (Transformers)
            "sentence_embedding",  # BoW
            "pixel_values",  # image (CLIPModel, etc.)
            "input_features",  # audio (Whisper, etc.)
            "input_values",  # audio (Wav2Vec2, HuBERT, etc.)
            "pixel_values_videos",  # video
        )
```

옵티마이저 파라미터 그룹도 모델이 아니라 **손실**을 기준으로 만든다 —
`.../sentence_transformers/base/trainer.py:1206-1240`:

```python
    def get_optimizer_cls_and_kwargs(
        self, args: BaseTrainingArguments, model: BaseModel | None = None
    ) -> tuple[Any, Any]:
        """
        We have to override the optimizer_grouped_parameters because the Trainer superclass bases it on the `model`
        itself, but the BaseModel losses can have weights that should be updated as well, e.g.
        SoftmaxLoss (see #2872).
        """

        if isinstance(self.loss, dict):
            loss_model = nn.Sequential(OrderedDict(self.loss))
        else:
            loss_model = self.loss
        optimizer_cls, optimizer_kwargs = super().get_optimizer_cls_and_kwargs(args, loss_model)
```

손실 인스턴스가 `self.model` 로 SentenceTransformer 를 들고 있으므로(`multiple_negatives_ranking.py:20`)
`loss_model.named_parameters()` 는 백본 파라미터를 포함한다.

### 이 lock 에서는 Trainer 가 아예 생성되지 않는다

`.../sentence_transformers/base/trainer.py:174-180`:

```python
        if not is_training_available():
            raise RuntimeError(
                f"To train a {self.model_class.__name__} model, you need to install the `accelerate` and `datasets` modules. "
                "You can do so with the `train` extra:\n"
                'pip install -U "sentence-transformers[train]"'
            )
```

`.../sentence_transformers/util/environment.py:109-114`:

```python
def is_training_available() -> bool:
    """
    Returns True if we have the required dependencies for training Sentence
    Transformers models, i.e. Huggingface datasets and Huggingface accelerate.
    """
    return is_accelerate_available() and is_datasets_available()
```

`envs/sentence-transformers/uv.lock` 에 `accelerate` 도 `datasets` 도 없다(0절). 따라서 **이 환경에서
`SentenceTransformerTrainer(...)` 는 인스턴스화 시점에 RuntimeError 로 죽는다.** 프로브가 Trainer 를
건드리지 않고 `model.tokenize` → `model(features)` → `info_nce` → `loss.backward()` 로 직접 가는
현재 구조(`trainbench/probe/sentence_transformers.py:79-84`)는 이 제약과 일치한다. 벤치 런이 ST 의
자체 Trainer 로 측정하려 한다면 lock 을 고쳐야 한다.

`sentence_transformers/base/trainer.py:62-63` 은 `datasets` 없이도 **import** 는 통과하도록 만든다:

```python
if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value
```

즉 `import sentence_transformers` 는 성공하고(`__init__.py:26` 이 트레이너를 무조건 import 한다)
실패는 생성 시점으로 미뤄진다.

---

## 5. `processor(text=..., images=...)` 규약을 받는가

**받는다. transformers v5 규약으로 받는다.**

modality → processor 인자 이름 매핑, `.../sentence_transformers/base/modality_types.py:59-67`:

```python
ProcessorArgName: TypeAlias = Literal["text", "images", "audio", "videos", "message"]
MessageFormat: TypeAlias = Literal["auto", "structured", "flat"]
MODALITY_TO_PROCESSOR_ARG: dict[Modality, ProcessorArgName] = {
    "text": "text",
    "image": "images",
    "audio": "audio",
    "video": "videos",
    "message": "message",
}
```

실제 호출, `.../sentence_transformers/base/modules/transformer.py:1258-1290`:

```python
    def _call_multimodal_processor(
        self, modality, processor_inputs, modality_kwargs, common_kwargs,
    ) -> dict[str, Any]:
        """Call a :class:`ProcessorMixin` processor, handling both legacy and v5 calling conventions."""
        # Convert modality keys to processor argument names (e.g., "image" -> "images")
        processor_inputs = {MODALITY_TO_PROCESSOR_ARG.get(key, key): value for key, value in processor_inputs.items()}

        # Some transformers processors are still outdated, and don't accept common_kwargs, etc.
        if (
            self.config.model_type in {"clipseg", "whisper", "sam3"}
            or not _TRANSFORMERS_PROCESSOR_SUPPORTS_MODALITY_KWARGS
        ):
            ...
            return self.processor(**processor_inputs, **kwargs, **common_kwargs)

        # This is the much cleaner transformers v5 approach
        return self.processor(
            **processor_inputs,
            text_kwargs=modality_kwargs["text"],
            images_kwargs=modality_kwargs["image"],
            audio_kwargs=modality_kwargs["audio"],
            videos_kwargs=modality_kwargs["video"],
            common_kwargs=common_kwargs,
        )
```

분기 스위치, `.../sentence_transformers/base/modules/transformer.py:113`:

```python
_TRANSFORMERS_PROCESSOR_SUPPORTS_MODALITY_KWARGS = parse_version(transformers_version) > parse_version("4.56.1")
```

이 lock 은 `transformers 5.14.1` 을 고정하므로(`envs/sentence-transformers/uv.lock:928-929`) **v5 경로가
쓰인다** — `processor(text=..., images=..., text_kwargs={...}, images_kwargs={...}, common_kwargs={...})`.

### 그러나 chat_template 이 있으면 message 로 강제 변환된다

`.../sentence_transformers/base/modules/transformer.py:974-978`:

```python
        # Always convert to the message format if it's supported, since it's most flexible with e.g. defaults
        if "message" in self.modality_config and modality != "message":
            modality, processor_inputs = self.input_formatter.batch_to_message(modality, processor_inputs)
        elif modality not in self.modality_config:
            raise_unsupported_modality_error(inputs, modality, list(self.modality_config.keys()), "Transformer module")
```

`"message"` 가 modality_config 에 들어가는 조건은 하나뿐이다 — `transformer.py:1840-1842`:

```python
        modalities = self.infer_modalities_from_processor(processor)
        if hasattr(processor, "chat_template") and processor.chat_template is not None:
            modalities.append("message")
```

그리고 message 경로는 `apply_chat_template` 을 부른다(`transformer.py:1361`, `_apply_chat_template`
`:1371-1391`). **AGENTS.md 가 기록한 gemma-4-E2B `chat_template.jinja` 부재 사건이 여기서 반대 방향으로
작동한다**: 템플릿이 없으면 `"message"` 가 추가되지 않으므로 `apply_chat_template` 로 가지 않고, 대신
`text` / `("image","text")` 경로가 modality_config 에 있어야 한다. 있는지 여부는 이 호스트에서 확정 불가(7절).

modality 목록은 processor 의 하위 속성으로 정해진다 — `transformer.py:1918-1935`:

```python
        processor_attribute_mapping: dict[str, Modality] = {
            "tokenizer": "text",
            "image_processor": "image",
            "feature_extractor": "audio",
            "video_processor": "video",
        }
        if isinstance(processor, ProcessorMixin):
            processor_attributes = self._get_processor_attributes() or {}
            return [
                modality_name
                for processor_attribute, modality_name in processor_attribute_mapping.items()
                if processor_attribute in processor_attributes
            ]
```

### 학습 시 입력이 processor 에 닿는 경로

`.../sentence_transformers/base/data_collator.py:105-111`:

```python
        for column_name in column_names:
            task = router_mapping.get(column_name, None)
            prompt = self._get_prompt_for_column(prompts, column_name)
            inputs = [row[column_name] for row in features]

            preprocessed = self.preprocess_fn(inputs, prompt=prompt, task=task)
            for key, value in preprocessed.items():
                batch[f"{column_name}_{key}"] = value
```

`preprocess_fn` 은 `model.preprocess` 다 — `.../sentence_transformers/base/trainer.py:367-371`:

```python
        return self.data_collator_class(
            preprocess_fn=model.preprocess,
            router_mapping=args.router_mapping,
            prompts=args.prompts,
        )
```

`BaseModel.preprocess` 는 첫 모듈에 위임한다 — `.../sentence_transformers/base/model.py:556-568`:

```python
        try:
            preprocessed = self[0].preprocess(inputs, prompt=prompt, **kwargs)
        except TypeError:
            if prompt and modality == "text":
                inputs = [(prompt + inp[0],) + inp[1:] if isinstance(inp, tuple) else prompt + inp for inp in inputs]
            preprocessed = self[0].preprocess(inputs, **kwargs)
        except AttributeError:
            ...
            try:
                preprocessed = self[0].tokenize(inputs, **kwargs)
            except TypeError:
                preprocessed = self[0].tokenize(inputs)
```

**probe 레인 주의**: `trainbench/probe/sentence_transformers.py:79` 의 `model.tokenize(texts)` 는
5.6.1 에서 deprecated 다 — `.../sentence_transformers/base/model.py:572-579`:

```python
    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], **kwargs) -> dict[str, Tensor]:
        """
        .. deprecated::
            `tokenize` is deprecated. Use `preprocess` instead.
        """

        logger.warning_once("The `tokenize` method is deprecated, please use `preprocess` instead.")
        return self.preprocess(inputs=texts, **kwargs)
```

동작은 한다(`preprocess` 로 그대로 위임). 다만 `preprocess` 의 반환에는 `"modality"` 키가 들어간다
(`transformer.py:1022`) — 텐서가 아닌 문자열이다. 프로브의
`{k: v.to(device) if hasattr(v, "to") else v ...}` (`trainbench/probe/sentence_transformers.py:80`)는
`hasattr(v, "to")` 가드 덕분에 이를 통과시킨다. `str` 에 `.to` 는 없다. 통과.

멀티모달 입력을 담는 타입은 dict 다 — `.../sentence_transformers/base/modality_types.py:44-47`:

```python
MultimodalInput: TypeAlias = dict[
    Literal["text", "image", "audio", "video"], TextInput | ImageInput | AudioInput | VideoInput
]
SingleInput: TypeAlias = TextInput | ImageInput | AudioInput | VideoInput | MessageInput | MultimodalInput
```

---

## 6. ST 가 모델을 감싸는 방식 — `.parameters()` 가 무엇을 돌려주는가

`.../sentence_transformers/base/model.py:50`

```python
class BaseModel(nn.Sequential, PeftAdapterMixin, ABC):
```

같은 파일 `:61-63`:

```
    All models inherit from nn.Sequential and are composed of a sequence of modules
    that are called sequentially in the forward pass.
```

**따라서 `model.parameters()` 는 `nn.Sequential` 의 표준 동작 그대로다** — 백본(`Transformer.model`)과
후속 모듈의 파라미터를 전부, 재귀적으로 돌려준다. PEFT 어댑터가 `Transformer.model` 안에 주입되어도
같은 iterator 에 포함된다. 별도의 래퍼도, 필터도 없다.

`Pooling` 은 파라미터가 없다 — `.../sentence_transformers/sentence_transformer/modules/pooling.py:93-116`
의 `__init__` 은 `nn.Parameter` 나 `register_buffer` 를 하나도 만들지 않고 int/str/bool 속성만 둔다.
따라서 기본 폴백 모델(`[Transformer, Pooling]`)에서 파라미터는 전부 백본의 것이다.

기저 transformers 모델을 꺼내는 공식 통로 — `.../sentence_transformers/base/model.py:1548-1570`:

```python
    @property
    def transformers_model(self) -> PreTrainedModel | None:
        ...
        for module in self.modules():
            # The Transformer check allows for returning underlying models with backend="onnx" or "openvino"
            if isinstance(module, Transformer):
                return module.model
            if isinstance(module, PreTrainedModel):
                return module
        return None
```

### `params_with_grad` 만 세고 `trainable_params` 를 세지 않으면 무엇을 놓치는가

프로브 현재 코드 — `/Users/jwcho/Codes/train-comparison/trainbench/probe/sentence_transformers.py:85`:

```python
        with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
```

이 한 줄은 두 수를 하나로 접는다. `requires_grad=True` 인 파라미터 수(`trainable_params`)와, 그중
`.grad` 가 채워진 수(`params_with_grad`)다. `nn.Sequential` 이므로 두 수가 분리 가능하다:
`sum(p.requires_grad for p in model.parameters())` 와 위 식.

AGENTS.md 가 기록한 unsloth 사건(`FastVisionModel.from_pretrained` 가 `full_finetuning=False` 로 전부
동결 → `params_with_grad=0`, `trainable_params=0`, 그런데 backward 는 통과)의 ST 판 등가물은 다음 셋이다.
전부 이 코드에서 **`with_grad == 0` 으로만 나타나고 원인은 구분되지 않는다**:

1. `model_kwargs` 로 넘어간 `quantization_config` 가 백본을 4-bit 로 만들어 `requires_grad=False` 가 된 경우
   — 원인은 적재. `trainable_params=0`.
2. 첫 모듈이 `Pooling` 없이 `sentence_embedding` 을 직접 만들었고(2절), 그래프가 정상인데 특정 파라미터가
   forward 에 참여하지 않은 경우 — `trainable_params>0`, `params_with_grad<trainable_params`.
3. PEFT 어댑터가 붙어 백본이 동결된 정상 LoRA 상태 — `trainable_params` 는 어댑터 수만큼 작지만 0 은 아니다.

`peft` 는 이 lock 에 없다(0절). 따라서 3번은 이 환경에서 발생할 수 없고, `is_peft_available()` 로 가드된
코드 경로(`transformer.py:1139-1148` 의 prompt-learning attention mask 확장, `transformer.py:1684` 의
`PeftConfig.from_pretrained`)는 전부 죽은 분기다. **LoRA/QLoRA 축은 `envs/sentence-transformers` 에서
현재 실행 불가능하다** — `trainbench/axes.py:591-604` 의 qlora 경로가 `BitsAndBytesConfig` 를 import 하는데
`bitsandbytes` 도 이 lock 에 없다.

gradient checkpointing 은 위임된다 — `.../sentence_transformers/base/model.py:1377-1385`:

```python
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        """Enable gradient checkpointing for the model."""
        # Propagate the gradient checkpointing to the transformer model
        for module in self.modules():
            if module is not self and hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
                except TypeError:
                    module.gradient_checkpointing_enable()
```

`try/except TypeError` 로 kwargs 를 삼키므로, `gradient_checkpointing_kwargs={"use_reentrant": False}` 가
적용되었는지는 **호출 결과로 확인되지 않는다.** `applied` 읽기로 백본 쪽 플래그를 직접 봐야 한다.

---

## 7. 부수적으로 확인된 것 (레인이 밟을 것들)

### 5.4.0 대규모 재배치 — 옛 import 는 경고와 함께 살아 있다

`.../sentence_transformers/util/deprecated_import.py:45,53,67`:

```python
    "sentence_transformers.losses": "sentence_transformers.sentence_transformer.losses",
    ...
    "sentence_transformers.losses.CachedMultipleNegativesRankingLoss": "sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking",
    ...
    "sentence_transformers.losses.MultipleNegativesRankingLoss": "sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking",
```

`.../sentence_transformers/util/deprecated_import.py:225-238` 이 첫 import 때 `DeprecationWarning` 을 낸다.
`trainbench/probe/sentence_transformers.py:92` 의
`from sentence_transformers.losses import CachedMultipleNegativesRankingLoss` 는 **동작한다.**
`pytest -W error::DeprecationWarning` 을 쓰는 게이트가 있다면 거기서만 깨진다.

`.../sentence_transformers/__init__.py:3`:

```python
__version__ = "5.6.1"
```

`report.add_version(sentence_transformers)` 가 읽는 값이다.

### `get_sentence_embedding_dimension` 은 남아 있다

`.../sentence_transformers/sentence_transformer/model.py:976-977`:

```python
    def get_sentence_embedding_dimension(self) -> int | None:
        return self.get_embedding_dimension()
```

`Transformer.get_embedding_dimension` 은 config 를 훑는다 — `.../sentence_transformers/base/modules/transformer.py:1172-1207`
는 `projection_dim`(단, `module_output_name == "sentence_embedding"` 일 때만) → `hidden_size` →
`neck_hidden_sizes`/`hidden_sizes`/`embed_dims` → `hidden_dim` → `config.text_config` → `config.sub_configs`
순으로 찾고, 못 찾으면 `ValueError` 를 던진다(`:1208-1209`). **VLM 의 composite config 에서 이 순서가 어떤
값을 고르는지는 체크포인트마다 다르다** — `text_config.hidden_size` 가 우선(`:1196-1199`)이라는 것만 확정.

### ST 의 `lasttoken` pooling 은 padding side 를 묻지 않는다

`.../sentence_transformers/sentence_transformer/modules/pooling.py:224-233`:

```python
            elif mode == "lasttoken":
                bs, seq_len, hidden_dim = token_embeddings.shape
                if torch.jit.is_tracing():
                    # Avoid tracing argmax with int64: https://github.com/microsoft/onnxruntime/issues/10068
                    attention_mask = attention_mask.to(torch.int32)
                values, indices = attention_mask.flip(1).max(1)
                indices = torch.where(values == 0, seq_len - 1, indices)
                gather_indices = (seq_len - indices - 1).unsqueeze(-1).unsqueeze(1).expand(-1, 1, hidden_dim)
                mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(token_embeddings.dtype)
                output_vectors.append(torch.gather(token_embeddings * mask, 1, gather_indices).squeeze(dim=1))
```

`flip(1).max(1)` 로 **마지막 1 의 위치를 직접 찾는다.** 좌/우 패딩 어느 쪽이든 맞고, 전부 PAD 인 행은
`seq_len - 1` 로 떨어진다(예외 없음). 우리 `trainbench/embedding.py:22-77` 의 `last_token_pool` 은
정반대 설계다: `padding_side` 를 명시적으로 받고, 마스크가 그 선언과 어긋나면 `ValueError` 로 런을 세우며,
빈 행도 `ValueError` 다.

두 구현은 정상 입력에서 **같은 위치를 고른다.** 다르게 행동하는 것은 잘못된 입력에서다 — ST 는 조용히
PAD 임베딩을 돌려주고, 우리 것은 멈춘다. `trainbench/probe/sentence_transformers.py:7-11` 이 ST 에
padding-side 정렬을 하지 않는다고 적어둔 결정은 이 코드와 모순되지 않는다. 다만 그 결과로
**ST 경로에서만 PAD 오염이 검출되지 않는다**는 사실은 결과 해석에 남는다.

### `unpad_inputs` — 자동 활성화되는 성능 축

`.../sentence_transformers/base/modules/transformer.py:561-568` (독스트링):

```
        unpad_inputs (bool, optional): Controls whether text-only inputs are concatenated without
            padding for faster inference using flash attention's variable-length functions. Non-text
            inputs (images, audio, video) are always padded normally. If ``None`` (default), unpadding
            is enabled automatically when all prerequisites are met (flash attention with variable-length
            support, ``"torch"`` backend, ``"feature-extraction"`` task). Set to ``False`` to force
            padding, which is needed for architectures that don't support unpadded inputs (e.g.
            ``qwen2_vl``). Set to ``True`` to request unpadding explicitly; a warning is logged if the
            prerequisites are not met. Defaults to None.
```

`.../sentence_transformers/base/modules/transformer.py:963-972`:

```python
        # Flatten inputs to avoid padding overhead when using flash attention variable-length functions.
        # Only safe for text-only inputs, since DataCollatorWithFlattening only handles input_ids/labels.
        should_flatten = self.can_flatten_inputs and (
            modality == "text"
            or (modality == "message" and self.input_formatter.is_text_only_messages(processor_inputs["message"]))
        )
        if should_flatten:
            del common_kwargs["return_tensors"]
            modality_kwargs["text"].pop("padding", None)
            modality_kwargs["text"]["return_attention_mask"] = False
```

**측정 관점에서 중요하다**: `attn_implementation="flash_attention_2"` 를 `model_kwargs` 로 넘기면
ST 가 **자동으로** 시퀀스 패킹까지 켠다. 이것은 우리 축 정의에 없는 별개의 최적화이므로, ST 셀의
throughput 은 attention 구현만이 아니라 packing 까지 포함한 수가 된다. `model[0].unpad_inputs` 를
읽어 기록하지 않으면 축 간 비교가 오염된다. Pooling 쪽 대응 경로는
`pooling.py:132-133` (`if "cu_seq_lens_q" in features: ... _forward_flattened`) 에 있고, 이는 우리
`trainbench/embedding.py:80-139` 의 `packed_last_token_pool` 과 같은 문제를 푸는 ST 내부 구현이다.

### `torch 2.13.0+cu130` / `transformers 5.14.1`

`envs/sentence-transformers/uv.lock:798-799`, `:928-929`. AGENTS.md 의 "Record the resolved
torch/framework versions per run" 대상이다. 다른 env 와 다른 스택이므로 교란 변수로 보이게 기록해야 한다.

---

## 이 호스트에서 확정하지 못한 것

파드/이미지가 답해야 할 질문. 추측을 적지 않는다.

1. `Qwen/Qwen3-VL-Embedding-2B` 리포지토리에 `modules.json` 이 있는가. 있으면 `_load_config_modules`
   경로이고 없으면 `_load_default_modules` 폴백이다. 어느 쪽인지에 따라 `model_kwargs` 병합 우선순위와
   pooling 결정 주체가 바뀐다(1절 4-a vs 4-b, 2절).
2. `AutoModel.from_pretrained` 가 세 체크포인트 각각에 대해 어떤 클래스를 돌려주고, 그 클래스의
   `forward` 반환 어노테이션에 `last_hidden_state` 필드가 있는가. 이것이 `module_output_name` 이
   `token_embeddings`(→ Pooling 붙음)가 되는지 `sentence_embedding`(→ Pooling 안 붙음)이 되는지를 정한다.
3. 각 체크포인트의 `config.architectures[0]` 이 정확히 무엇인가. `"...ForCausalLM"` 접미사 여부가
   pooling mode 를 `lasttoken` 과 `mean` 으로 가른다. `config.is_causal` 값도 함께 필요하다.
4. `AutoProcessor.from_pretrained` 가 세 체크포인트에 대해 `ProcessorMixin` 을 돌려주는가 아니면 bare
   tokenizer 를 돌려주는가. 전자면 `_call_multimodal_processor`(v5 kwargs), 후자면
   `_call_single_modality_processor` 로 갈린다.
5. 각 processor 의 `chat_template` 이 `None` 이 아닌가. `None` 이 아니면 모든 입력이 `apply_chat_template`
   경로로 강제 변환된다(5절). `google/gemma-4-E2B` 는 AGENTS.md 기록상 base 체크포인트에 템플릿이 없다 —
   ST 에서 그 결과가 "message 없음 → text 경로 정상" 인지 "text 도 modality_config 에 없음 →
   `raise_unsupported_modality_error`" 인지는 4번의 답에 달려 있고 확인 안 함.
6. `attn_implementation="flash_attention_2"` 를 `model_kwargs` 로 넘겼을 때 `model[0].unpad_inputs` 가
   실제로 `True` 로 자동 활성화되는가, 그리고 세 아키텍처가 unpadded 입력을 받는가. 독스트링은
   `qwen2_vl` 이 못 받는다고만 적는다. `qwen3_vl`, `gemma4`, `qwen3_5` 는 미기재.
7. `Transformer.get_embedding_dimension()` 이 세 VLM composite config 에서 어떤 값을 고르는가.
   `text_config.hidden_size` 가 우선이지만 최상위 config 에 `hidden_size` 가 있으면 그것이 먼저 잡힌다.
8. `envs/sentence-transformers` 이미지가 `accelerate`/`datasets`/`peft`/`bitsandbytes` 를 lock 밖에서
   (베이스 이미지로) 들고 있는가. lock 에는 없다. 있다면 `is_training_available()` 이 True 가 되어
   Trainer 경로와 LoRA 축이 열리고, 그때 실제 버전이 무엇인지가 새 교란 변수가 된다.
9. GradCache(CachedMNRL)의 2회 forward 가 이 세 모델에서 실제로 얼마의 오버헤드를 내는가. 코드 구조는
   확정했으나 수치는 측정 안 함.
