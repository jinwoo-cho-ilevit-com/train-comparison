"""Check bodies shared by every framework probe.

Each framework loads models differently, but the questions asked of the loaded
model are the same, and asking them identically is what makes the resulting
matrix comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from trainbench import applied, axes
from trainbench.config_schema import BenchConfig
from trainbench.embedding import align_padding_side, info_nce, last_token_pool
from trainbench.probe.fixtures import PROBE_IMAGE_SIZE, PROBE_PAIRS, probe_image
from trainbench.probe.types import ProbeReport
from trainbench.prompt import format_prompt


def dtype_for(device: torch.device) -> torch.dtype:
    # bf16 is the training dtype but is not dependable on CPU/MPS, and a probe only
    # answers "does it run".
    return torch.bfloat16 if device.type == "cuda" else torch.float32


# Both currently-active archs share `Qwen2VLImageProcessor`. gemma4 uses a
# different image processor entirely and is being dropped from the campaign, so
# `pixel_budget_kwargs` never applies to it.
PIXEL_BUDGET_ARCHS = ("qwen3_vl", "qwen3_5")


def pixel_budget_kwargs(config: BenchConfig) -> dict[str, int]:
    """`max_pixels`/`min_pixels` for `AutoProcessor.from_pretrained`.

    Passing them overrides whatever `preprocessor_config.json` ships: transformers
    5.14.1 `image_processing_base.py`'s `from_dict` updates the loaded config dict
    with any kwarg matching `valid_kwargs.__annotations__` before constructing the
    image processor, and `Qwen2VLImageProcessor.__init__` (`image_processing_qwen2_vl.py`)
    then writes `min_pixels`/`max_pixels` into `size["shortest_edge"]`/`["longest_edge"]`
    ahead of whatever `size` the checkpoint declared. That override is the point —
    `data.max_pixels`/`data.min_pixels` are the budget, not the checkpoint's own range.

    Empty for any other arch: gemma4 does not use `Qwen2VLImageProcessor`, and these
    two keys reaching it would be inert at best (silently dropped as unused kwargs)
    and misleading at worst (looking like a cap that this arch never gets).
    """
    if config.model.arch not in PIXEL_BUDGET_ARCHS:
        return {}
    return {"max_pixels": config.data.max_pixels, "min_pixels": config.data.min_pixels}


def patch_axes(config: BenchConfig, report: ProbeReport) -> None:
    """Axes that have to be applied before any model exists.

    Kernel libraries replace transformers classes, so a model built first is a
    model the patch never reached (docs/CONTRACTS.md §2). Every adapter goes
    through here, not just the native one — an adapter that skips it would report
    a kernel axis it never asked for.
    """
    report.run("axes_patch", lambda: {"applied": axes.patch(config)})


def load_kwargs(config: BenchConfig, report: ProbeReport) -> dict[str, Any]:
    """`axes.load_kwargs` as a check of its own, so a refused axis is not a bad load.

    The refusal and the load are different answers. `axes.load_kwargs` raises for a
    value it cannot put into effect — `peft.mode=qlora` off CUDA is the one that
    does today — and evaluating it inside the `model_load` lambda charged that
    refusal to the checkpoint: the cell read "the model does not load" and the
    adapter's `if not ok: return` ended the probe, costing the nine checks that
    have nothing to do with the axis.

    So it is recorded under its own name and the load goes ahead without the
    kwargs, which is the shape `verify_axes` already uses for `assemble`: the axes
    these kwargs would have carried come back undetermined rather than unexamined.
    `applied.capture` then reads the built model and `axes_verified` refuses the
    mismatch, so nothing here can pass a bare load off as the requested one — this
    report has two failed checks about it and `all_ok` is False.

    Only probes do this. `scripts/bench.py` calls `axes.load_kwargs` directly and
    outside any `try`, because a measured run must die where a probe records.
    """
    resolved: dict[str, Any] = {}

    def _resolve() -> dict[str, Any]:
        resolved.update(axes.load_kwargs(config))
        # The keys rather than the values: `quantization_config` is a
        # BitsAndBytesConfig, and what this check answers is which load-time axes
        # were asked for. What came back is `applied`'s question.
        return {"requested": sorted(resolved)}

    report.run("axes_load_kwargs", _resolve)
    return resolved


def verify_axes(
    model: Any,
    config: BenchConfig,
    device: torch.device,
    framework: str,
    report: ProbeReport,
) -> Any:
    """Build the rest of the run around `model`, then read back what took effect.

    Returns the model to use afterwards: `assemble` may hand back a different
    object, because peft, `torch.compile` and FSDP all replace the model rather
    than mutate it.

    `framework` is a literal passed in by the calling adapter and never read from
    the config. The config records what was requested; this literal is the
    evidence of which code path actually ran, which is the entire reason
    applied.py exists (docs/CONTRACTS.md §2).

    A failure inside `assemble` leaves `built` holding the model alone, so the
    axes it would have covered come back undetermined rather than unexamined.
    """
    built = applied.Built(model=model)

    def _assemble() -> dict[str, Any]:
        nonlocal built
        built, names = axes.assemble(model, config, device, framework=framework)
        return {"applied": names}

    report.run("axes_assemble", _assemble)
    report.applied = applied.capture(built, config)
    # Records the verdict rather than aborting: a probe answers "does it run", and
    # purpose=probe is not enforced. A reportable purpose raises here, which is the
    # point — the same call in the measurement harness stops the run.
    report.run("axes_verified", lambda: _verified(report.applied, config))
    return built.model if built.model is not None else model


def _verified(state: applied.AppliedState, config: BenchConfig) -> dict[str, Any]:
    applied.assert_matches(state, config)
    _refuse_mismatch(state, config)
    return state.to_dict()


def _refuse_mismatch(state: applied.AppliedState, config: BenchConfig) -> None:
    """Fail this check when the built model is not the one the run asked for.

    `assert_matches` returns immediately for `purpose=probe`, so `axes_verified`
    used to be green on `all_matched: false` and a support-matrix cell read as
    clean while two mismatches sat under it (docs/support-matrix.md): `kernel.name`
    requested `none` and applied `fla` on every qwen3_5 cell, and `precision.name`
    requested `bf16` and applied `mixed(bf16,fp32)` on six others.

    The mismatch is not removed, which is the point — an uneven application is
    state to report, not noise to hide (docs/CONTRACTS.md §2). It stays on
    `report.applied`, which is what the result file carries; what changes is that
    nothing reads it as a pass.

    Undetermined axes are deliberately not refused here. A probe builds no
    dataloader, so those axes come back undetermined in every cell, and grading
    them here would paint every cell red over a question the probe never claimed
    to answer. Undetermined stops a run in `assert_matches`, for the purposes whose
    numbers get published.
    """
    mismatched = state.mismatched()
    if not mismatched:
        return
    problems = []
    for axis in mismatched:
        note = _environment_bound(axis, config)
        problems.append(
            f"{axis.axis}: requested {axis.requested!r}, applied {axis.applied!r}"
            + (f" ({note})" if note else "")
        )
    raise applied.AppliedMismatch(
        "the model that was built is not the one this run asked for: " + "; ".join(problems)
    )


def _environment_bound(state: applied.AxisState, config: BenchConfig) -> str:
    """Why this mismatch is the image rather than the run, when that can be read.

    Only `kernel.name` can be answered, and by the reader that already decides it
    at the patch site rather than by a second copy of the rule: transformers binds
    fla while it imports the modelling module, so on those architectures the
    request was unsatisfiable before any model existed.

    Every other axis returns "". Nothing reachable from here tells a framework's
    own dtype policy apart from a load that went wrong — the `mixed(bf16,fp32)`
    above is axolotl holding `embed_tokens`/`lm_head` in fp32 and peft holding
    adapter weights there, and both look identical to a bf16 request answered in
    fp32. Calling one of them environment-bound would be inventing the
    distinction rather than reading it.
    """
    if state.axis != "kernel.name" or state.applied is None:
        return ""
    bound = axes._environment_bound_kernel(config)
    if not bound or bound not in state.applied:
        return ""
    return (
        f"environment-bound: this image binds kernel={bound} on arch={config.model.arch} "
        "while transformers imports the modelling module, so no run in it can apply the "
        "requested value"
    )


# transformers' default when a checkpoint names no padding_side. It is not a
# guess: docs/model-spec.yaml records exactly this reasoning for qwen3_vl_emb_2b,
# whose tokenizer_config.json has no such key.
TRANSFORMERS_DEFAULT_PADDING_SIDE = "right"


def checkpoint_padding_side(hf_id: str, revision: str | None = None) -> dict[str, Any]:
    """Which side the checkpoint itself declares, read from the file the spec cites.

    `docs/model-spec.yaml` names `tokenizer_config.json` as the source of
    `padding_side` for all three models, so that is what the audited value has to
    be compared against. It is a cache hit in any probe: whatever loaded the model
    downloaded this file first.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(hf_id, "tokenizer_config.json", revision=revision)
    declared = json.loads(Path(path).read_text()).get("padding_side")
    if declared is None:
        return {
            "padding_side": TRANSFORMERS_DEFAULT_PADDING_SIDE,
            "source": "transformers default; tokenizer_config.json names none",
        }
    return {"padding_side": str(declared), "source": "tokenizer_config.json"}


def padding_side_alignment(
    processor: Any, padding_side: str, hf_id: str, revision: str | None = None
) -> dict[str, Any]:
    """Force the tokeniser onto the configured padding side, and say who disagreed.

    Two unrelated disagreements used to be one failure. The one this check exists
    for is the spec going stale — the checkpoint padding differently from
    `docs/model-spec.yaml`, which only a run can notice. A framework moving the
    side *after* the load is not that, and grading it as that is what filed three
    unsloth cells of the 2026-08-02 campaign as spec drift: unsloth sets
    `padding_side = "left"` unconditionally at the end of `from_pretrained`
    (unsloth 2026.7.6 models/vision.py:1716-1718), while `native` read `right` off
    the same two checkpoints, which is what the spec says.

    So the raise compares the checkpoint's own declaration, and what the loaded
    object declared is recorded next to it under `framework_forced` rather than
    graded. Alignment still runs first, and unconditionally: whatever comes after
    this check pools a real token either way, and a processor that has no
    padding_side to align is answered without reaching for the network.
    """
    detail = align_padding_side(processor, padding_side)
    checkpoint = checkpoint_padding_side(hf_id, revision)
    detail["checkpoint_padding_side"] = checkpoint["padding_side"]
    detail["checkpoint_source"] = checkpoint["source"]
    detail["framework_forced"] = sorted(
        name
        for name, value in detail["declared_before"].items()
        if value != checkpoint["padding_side"]
    )
    if checkpoint["padding_side"] != padding_side:
        raise ValueError(
            f"{hf_id} declares padding_side {checkpoint['padding_side']!r} in "
            f"{checkpoint['source']} but config.model.padding_side is {padding_side!r}; "
            "it has been forced onto the configured side, and docs/model-spec.yaml no "
            "longer matches this checkpoint."
        )
    return detail


def encode(model: Any, batch: dict[str, torch.Tensor], padding_side: str) -> torch.Tensor:
    """Pooled embedding from whatever hidden states the model exposes.

    `padding_side` is threaded through from `config.model.padding_side` rather than
    read off the processor: the config is what the audit compares against
    docs/model-spec.yaml. It is not merely a claim any more — every batch built
    here goes through `align_padding_side` first, and `last_token_pool` rejects a
    mask that disagrees with the declared side.

    `use_cache=False`: every model here ships `config.use_cache=True`, so the
    default forward allocates a KV cache none of this file's callers read back —
    an embedding forward pools one hidden state and stops, it never generates a
    second token. `scripts/bench.py`'s packed path already turns this off for the
    same reason (`pooled_embeddings`); this padded path is the one it left on.
    """
    output = model(**batch, output_hidden_states=False, use_cache=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = getattr(output, "hidden_states", None)
        hidden = hidden[-1] if hidden else output[0]
    return last_token_pool(hidden, batch["attention_mask"], padding_side=padding_side)


def text_batch(processor: Any, device: torch.device, padding_side: str) -> dict[str, torch.Tensor]:
    align_padding_side(processor, padding_side)
    texts = [q for q, _ in PROBE_PAIRS] + [d for _, d in PROBE_PAIRS]
    batch = processor(text=texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in batch.items()}


def tokenize_text(
    processor: Any, device: torch.device, into: dict[str, torch.Tensor], padding_side: str
) -> dict[str, Any]:
    """Tokenise the probe pairs into `into` and return only JSON-safe detail.

    Every adapter needs the tensors afterwards but whatever a check returns becomes
    its `detail` and is serialised, so the tensors go into the caller's dict and
    only shapes come back. Written once here because five adapters had the same
    closure, and the copies had already drifted in what they reported.
    """
    into.update(text_batch(processor, device, padding_side))
    return {
        "keys": sorted(into),
        "input_ids_shape": list(into["input_ids"].shape),
        "padding_side": padding_side,
    }


def image_batch(
    processor: Any, device: torch.device, padding_side: str, prompt_format: str
) -> dict[str, Any]:
    """Multimodal batch.

    The text must carry the model's image placeholder tokens; passing raw text
    alongside images silently produces zero image tokens against N image features,
    and the forward pass then fails on the mismatch. Which markup puts them there
    is `config.model.prompt_format` (trainbench/prompt.py) — `apply_chat_template`
    for a checkpoint that ships a template, the bare placeholder for one that does
    not. Calling `apply_chat_template` unconditionally is what failed every gemma-4
    probe of the 2026-08-02 campaign, on three frameworks at once.

    Images are grouped one sublist per row rather than passed flat, the same shape
    `scripts/bench.py::_group_by_row` builds. Measured 2026-08-02: a flat list
    reads to `Gemma4Processor` as one row carrying every image and it raises
    "Received inconsistently sized batches", while both Qwen processors return
    byte-identical tensors either way.
    """
    align_padding_side(processor, padding_side)
    image = probe_image()
    texts = [
        format_prompt(
            processor,
            q,
            with_image=True,
            prompt_format=prompt_format,
            add_generation_prompt=False,
        )
        for q, _ in PROBE_PAIRS
    ]
    batch = processor(
        text=texts, images=[[image] for _ in texts], return_tensors="pt", padding=True
    )
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


# Where the image placeholder id is declared, newest name first. transformers
# renamed `image_token_index` to `image_token_id`, and some VLM configs keep it on
# the text sub-config instead of the top level. Reading only the first name makes a
# model whose config uses another one look like it has no image tokens at all.
IMAGE_TOKEN_ID_FIELDS = ("image_token_id", "image_token_index")


def image_token_id(model: Any) -> tuple[int, str]:
    """The image placeholder token id and where it was found."""
    configs: list[tuple[str, Any]] = [("config", model.config)]
    get_text_config = getattr(model.config, "get_text_config", None)
    if callable(get_text_config):
        try:
            configs.append(("text_config", get_text_config()))
        except Exception:  # noqa: BLE001 - framework wrappers expose odd configs
            # Swallowed on purpose: the answer we owe the caller is whether an image
            # token id exists, and letting an accessor's failure surface instead
            # would report a missing id as a broken config.
            pass

    for where, config in configs:
        for field in IMAGE_TOKEN_ID_FIELDS:
            value = getattr(config, field, None)
            if value is not None:
                return int(value), f"{where}.{field}"
    raise ValueError(
        f"no image token id on this model config; looked for {list(IMAGE_TOKEN_ID_FIELDS)} "
        f"on {[where for where, _ in configs]}"
    )


def pad_token_id(processor: Any) -> int | None:
    """The processor's pad token id, wherever it keeps it. None if it has none."""
    for holder in (getattr(processor, "tokenizer", None), processor):
        value = getattr(holder, "pad_token_id", None)
        if value is not None:
            return int(value)
    return None


def visual_token_count(
    processor: Any,
    model: Any,
    device: torch.device,
    padding_side: str,
    max_tokens_per_image: int | None,
    prompt_format: str,
) -> dict[str, Any]:
    """How many tokens one fixed image costs on this model.

    All three models use patch_size 16 but differ in spatial merge and pooling, so
    the same image is not the same cost. Speed comparisons are meaningless until
    this is pinned per model.

    Every row of the batch carries the same image, so the four gates below are the
    ways a wrong id can still produce a plausible-looking number:

    * an id equal to the pad token counts padding. `0 < n < seq_len` accepts that
      happily, and the resulting count is a property of the batch shape rather
      than of the model
    * counts that differ per sample mean the id matched something the rows do not
      share; grading `per_sample[0]` alone accepted `[280, 279]`
    * a count of 0, or one filling the sequence, means the id or the prompt format
      is wrong — every format here emits text tokens around the placeholders
    * a count above `config.model.max_tokens_per_image`, where the model declares a
      cap, cannot be a count of that model's soft tokens. gemma4's processor stops
      at max_soft_tokens=280; exceeding it means the id matched more than the
      placeholders, or the processor we are measuring is not the one the spec
      describes. The Qwen models declare no cap (None), which is why this is a
      bound and not a lookup

    The bound is deliberately not an equality. It was one, against a declared 280
    read off `image_seq_length`, and it refused every real gemma-4 batch: the
    processor derives each image's count from its aspect ratio (448x448 -> 256,
    768x256 -> 252) and only reaches 280 when the ratio divides evenly. An equality
    there is a gate that fires on correct measurements, which is how it gets
    relaxed. The count that gets published is always the measured one.

    A wrong number here silently rescales every tokens/s figure that divides by it.
    """
    batch = image_batch(processor, device, padding_side, prompt_format)
    token_id, source = image_token_id(model)
    input_ids = batch["input_ids"]
    per_sample = (input_ids == token_id).sum(dim=1).tolist()
    total_seq_len = int(input_ids.shape[1])
    if not per_sample:
        raise ValueError("the probe batch is empty, so nothing was counted")
    count = per_sample[0]

    pad_id = pad_token_id(processor)
    if pad_id is not None and token_id == pad_id:
        raise ValueError(
            f"image token id {token_id} from {source} is this processor's pad token id; "
            f"the {count} tokens counted are padding, not image placeholders."
        )
    if len(set(per_sample)) != 1:
        raise ValueError(
            f"visual token counts disagree across samples: {per_sample}. Every row of this "
            f"batch carries the same image, so token id {token_id} from {source} is matching "
            "something the rows do not share."
        )
    if not 0 < count < total_seq_len:
        raise ValueError(
            f"visual token count {count} is outside 0 < n < {total_seq_len} for token id "
            f"{token_id} from {source}; the placeholder id or prompt_format={prompt_format} "
            "is wrong."
        )
    if max_tokens_per_image is not None and count > max_tokens_per_image:
        raise ValueError(
            f"measured {count} visual tokens but config.model.max_tokens_per_image caps "
            f"them at {max_tokens_per_image}; this processor cannot emit that many, so the "
            "count is of something else and every tokens/s figure divides by this number."
        )

    return {
        "image_size": list(PROBE_IMAGE_SIZE),
        "image_token_id": token_id,
        "image_token_id_source": source,
        "pad_token_id": pad_id,
        "visual_tokens_per_image": count,
        "visual_tokens_per_sample": per_sample,
        "declared_max_tokens_per_image": max_tokens_per_image,
        "total_seq_len": total_seq_len,
        # The count is only comparable across models once the prompt around it is
        # known, and the two formats wrap it in different numbers of tokens.
        "prompt_format": prompt_format,
    }


def training_step_evidence(model: Any, loss: torch.Tensor) -> dict[str, Any]:
    """What a backward that has just run actually reached, refused when it is nothing.

    A finite loss is not evidence that a step happened. Every framework here calls
    something like `enable_input_require_grads`, which puts `requires_grad` on the
    *embedding output* rather than on a parameter, so the graph stays
    differentiable and `loss.backward()` returns normally even when every
    parameter is frozen. The 2026-08-02 campaign recorded ok=True with
    `trainable_params=0` on three cells for exactly that reason.

    Shared rather than per-adapter because an adapter that computes its own loss —
    sentence_transformers pools inside its own module — otherwise reports
    `params_with_grad` alone, which reads 0 for a frozen model and 0 for a
    detached one and is the same green the campaign already published once.

    Both counts are of parameter *tensors*, not elements; what they answer is
    "did anything train", for which the number of elements is the wrong unit.
    Gradients are cleared before the refusal so a caller that records the failure
    and carries on does not accumulate into the next check.
    """
    with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    value = float(loss.detach())
    model.zero_grad(set_to_none=True)
    if trainable == 0:
        raise ValueError(
            f"the backward ran and none of this model's {total} parameter tensors were "
            f"trainable, so the step trained nothing; loss={value} is what a frozen graph "
            "returns. This cell measures a model that cannot learn, and reporting it as "
            "supported publishes a throughput figure for no training at all."
        )
    if with_grad == 0:
        raise ValueError(
            f"{trainable} of {total} parameter tensors are trainable and the backward "
            f"reached none of them (loss={value}); the pooled embedding is detached from "
            "every parameter this run would update."
        )
    return {
        "loss": value,
        "params_with_grad": with_grad,
        "trainable_params": trainable,
        "total_params": total,
    }


def infonce_backward(
    model: Any, batch: dict[str, torch.Tensor], temperature: float, padding_side: str
) -> dict:
    """One contrastive training step, refused when it trained nothing.

    This is the check that matters for framework support: patching that works for
    a language-modelling loss can still break when the loss is contrastive over
    pooled embeddings, because no LM head is involved.
    """
    model.train()
    pooled = encode(model, batch, padding_side)
    half = pooled.shape[0] // 2
    loss = info_nce(pooled[:half], pooled[half:], temperature)
    loss.backward()
    return training_step_evidence(model, loss)
