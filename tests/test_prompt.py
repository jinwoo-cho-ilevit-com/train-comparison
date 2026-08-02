"""The prompt format is a declared fact about the checkpoint, not an assumption.

Three frameworks failed the same way on gemma-4 in the 2026-08-02 campaign —
`Cannot use apply_chat_template because this processor does not have a chat
template` — because every path here called `apply_chat_template` unconditionally.
What is pinned below is that the format comes from config, that both directions of
disagreement stop the run, and that the raw format still carries an image
placeholder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trainbench.prompt import chat_template_of, format_prompt, verify_prompt_format


class _Templated:
    """A checkpoint that ships a chat template, as both Qwen repositories do."""

    chat_template = "{{ messages }}"
    image_token = "<|image_pad|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        content = messages[0]["content"]
        blocks = "".join(
            "<img>" if block["type"] == "image" else block["text"] for block in content
        )
        return f"<user>{blocks}{'<gen>' if add_generation_prompt else ''}"


class _Bare:
    """A pre-trained checkpoint: no chat template, an image token, and an
    `apply_chat_template` that raises exactly what transformers 5.14.1 raises."""

    chat_template = None
    image_token = "<|image|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        raise ValueError(
            "Cannot use apply_chat_template because this processor does not have a chat template."
        )


def test_a_missing_chat_template_is_named_before_transformers_raises():
    """The message the campaign got named neither the model nor the fix. This one
    has to say which config field decides it and what the two answers are."""
    with pytest.raises(ValueError, match="prompt_format"):
        format_prompt(
            _Bare(),
            "describe the picture",
            with_image=True,
            prompt_format="chat_template",
            add_generation_prompt=False,
        )


def test_a_raw_prompt_is_the_placeholder_then_the_text():
    """Measured against the real processor (2026-08-02, transformers 5.14.1):
    `<|image|>` in raw text expands to the image's soft tokens, 256 for the
    448x448 probe image. The placeholder leads, as it does in the chat format."""
    assert (
        format_prompt(
            _Bare(),
            "describe the picture",
            with_image=True,
            prompt_format="raw",
            add_generation_prompt=False,
        )
        == "<|image|>describe the picture"
    )


def test_a_raw_prompt_for_a_text_only_row_carries_no_placeholder():
    """Rows differ in which image columns they carry, and a placeholder with no
    image behind it is the count mismatch the processor refuses."""
    assert (
        format_prompt(
            _Bare(),
            "a square gradient test pattern",
            with_image=False,
            prompt_format="raw",
            add_generation_prompt=False,
        )
        == "a square gradient test pattern"
    )


def test_raw_is_refused_when_the_checkpoint_ships_a_template():
    """The stale-spec direction. Formatting a post-trained checkpoint raw measures a
    prompt it was never trained with, and nothing downstream could tell."""
    with pytest.raises(ValueError, match="never trained"):
        format_prompt(
            _Templated(),
            "describe the picture",
            with_image=True,
            prompt_format="raw",
            add_generation_prompt=False,
        )


def test_the_chat_template_path_passes_the_generation_prompt_through():
    """With last-token pooling `add_generation_prompt` decides which token becomes
    the embedding, so it must reach the template rather than be defaulted."""
    processor = _Templated()
    assert (
        format_prompt(
            processor,
            "q",
            with_image=True,
            prompt_format="chat_template",
            add_generation_prompt=True,
        )
        == "<user><img>q<gen>"
    )


def test_a_raw_prompt_refuses_a_generation_prompt():
    """There is no template to append one to. Ignoring the flag would report a value
    the run never applied; the schema refuses the same pair before a run starts."""
    with pytest.raises(ValueError, match="add_generation_prompt"):
        format_prompt(
            _Bare(),
            "q",
            with_image=False,
            prompt_format="raw",
            add_generation_prompt=True,
        )


def test_a_raw_prompt_needs_a_placeholder_to_stand_for_the_image():
    """Without one the pixels go in with nothing standing for them, which is the
    N-features-vs-zero-tokens mismatch the forward pass dies on."""
    processor = SimpleNamespace(chat_template=None)
    with pytest.raises(ValueError, match="no image_token"):
        format_prompt(
            processor, "q", with_image=True, prompt_format="raw", add_generation_prompt=False
        )


def test_the_template_is_read_off_the_tokenizer_when_the_processor_has_none():
    """A bare tokeniser holds its own template, and `apply_chat_template` resolves
    it the same way. Reading only the processor would call a text-only checkpoint
    templateless and format it raw."""
    processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template="{{ messages }}"))

    assert chat_template_of(processor) == "{{ messages }}"
    assert verify_prompt_format(processor, "chat_template")["processor_has_chat_template"] is True


def test_an_unknown_prompt_format_is_refused():
    """A typo in configs/model/ must not fall through to one of the two branches."""
    with pytest.raises(ValueError, match="must be one of"):
        verify_prompt_format(_Templated(), "chatml")
