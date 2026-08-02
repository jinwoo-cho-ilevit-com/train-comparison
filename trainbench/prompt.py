"""One row of text in the model's own prompt format.

Two of the three checkpoints ship a chat template and one does not, so
`apply_chat_template` is not something every processor can be asked for. Measured
2026-08-02, transformers 5.14.1, against the real Hub repositories:

    Qwen/Qwen3-VL-Embedding-2B   chat_template.jinja present
    Qwen/Qwen3.5-0.8B            chat_template.jinja present
    google/gemma-4-E2B           absent; processor.chat_template is None

`google/gemma-4-E2B` is the pre-trained checkpoint PLAN.md pins, and a pre-trained
checkpoint has no chat format because it was never instruction-tuned — the model
card's `apply_chat_template` examples all load `google/gemma-4-E2B-it`, which does
ship one. So the format is a fact about the checkpoint, declared per model in
`configs/model/` and mirrored in `docs/model-spec.yaml`, and this module is the
only place that reads it. Branching on `model.arch` instead would hide from the
result which prompt each number was measured with.

**The two formats are not the same prompt, and that is a confound the result has
to carry.** A `chat_template` row is wrapped in role and turn markers; a `raw` row
is the image placeholder followed by the text and nothing else. Sequence lengths
are therefore not comparable across the two, which docs/model-spec.md states in
the same words.
"""

from __future__ import annotations

from typing import Any

# The processor supplies the prompt format (a chat template the checkpoint ships).
CHAT_TEMPLATE = "chat_template"
# The checkpoint ships no chat format, so the row is placeholder + text. Only a
# pre-trained checkpoint may declare this: stripping a template off a model that
# has one measures a prompt that model was never trained on.
RAW = "raw"
PROMPT_FORMATS = (CHAT_TEMPLATE, RAW)


def chat_template_of(processor: Any) -> str | None:
    """The template this processor would use, or None if it has none.

    Read from the processor first and its tokeniser second, because that is the
    order `apply_chat_template` resolves them in: a multimodal processor holds the
    template and a bare tokeniser holds its own. Measured on gemma-4 both are None,
    on both Qwen models both are set.
    """
    template = getattr(processor, "chat_template", None)
    if template is None:
        template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    return template


def verify_prompt_format(processor: Any, prompt_format: str) -> dict[str, Any]:
    """Refuse a declared format the loaded checkpoint does not match.

    Both directions are refused, for the same reason `padding_side_alignment`
    refuses both: the config is a claim about the checkpoint, and a run is the only
    place that can notice the claim has gone stale.
    """
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(
            f"model.prompt_format must be one of {list(PROMPT_FORMATS)}, got {prompt_format!r}"
        )
    present = chat_template_of(processor) is not None
    if prompt_format == CHAT_TEMPLATE and not present:
        raise ValueError(
            "model.prompt_format is 'chat_template' but this processor has none, so "
            "apply_chat_template cannot run. A pre-trained checkpoint ships no chat "
            "format; declare prompt_format: raw for it in configs/model/ and "
            "docs/model-spec.yaml, or point model.hf_id at an instruction-tuned "
            "checkpoint that ships chat_template.jinja."
        )
    if prompt_format == RAW and present:
        raise ValueError(
            "model.prompt_format is 'raw' but this processor carries a chat template. "
            "Formatting it raw would measure a prompt this checkpoint was never trained "
            "with; docs/model-spec.yaml no longer matches what model.hf_id loads."
        )
    return {"prompt_format": prompt_format, "processor_has_chat_template": present}


def image_placeholder(processor: Any) -> str:
    """The one token that stands for an image in this processor's raw text.

    The processor expands it into as many soft tokens as the image costs, so one
    occurrence per image is what its own validation counts. Measured on gemma-4:
    `<|image|>`, expanded to 256 tokens for a 448x448 image.
    """
    for holder in (processor, getattr(processor, "tokenizer", None)):
        token = getattr(holder, "image_token", None)
        if token:
            return str(token)
    raise ValueError(
        "this processor declares no image_token, so a raw prompt has no placeholder to "
        "carry an image; the pixels would be passed with nothing standing for them and "
        "the forward pass would fail on the count mismatch."
    )


def format_prompt(
    processor: Any,
    text: str,
    *,
    with_image: bool,
    prompt_format: str,
    add_generation_prompt: bool,
) -> str:
    """One side of a pair, in this model's own prompt format.

    The image placeholder comes first in both formats, so what differs between them
    is the role and turn markers around it and nothing about the pixels.
    """
    verify_prompt_format(processor, prompt_format)
    if prompt_format == RAW:
        if add_generation_prompt:
            raise ValueError(
                "model.add_generation_prompt is true under prompt_format=raw, which has no "
                "template to append a generation prompt to; with last-token pooling that "
                "would silently be a different embedding than the value asks for."
            )
        return f"{image_placeholder(processor)}{text}" if with_image else text

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if with_image:
        content.insert(0, {"type": "image"})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
