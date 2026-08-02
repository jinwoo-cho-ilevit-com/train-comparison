"""The prompt format is a declared fact about the checkpoint, not an assumption.

Three frameworks failed the same way on gemma-4 in the 2026-08-02 campaign —
`Cannot use apply_chat_template because this processor does not have a chat
template` — because every path here called `apply_chat_template` unconditionally.
What is pinned below is that the format comes from config, that both directions of
disagreement stop the run, and that the raw format still carries an image
placeholder.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from trainbench.prompt import chat_template_of, format_prompt, verify_prompt_format

# The instruction `configs/model/qwen3_vl_emb_2b.yaml` declares, which is also the
# string `Qwen/Qwen3-VL-Embedding-2B`'s own template defaults to.
QWEN_INSTRUCTION = "Represent the user's input."


class _Templated:
    """A checkpoint that ships a chat template, as both Qwen repositories do.

    Every turn handed in is rendered and the roles are kept. Rendering `messages[-1]`
    alone cannot tell an empty system turn from no system turn, which is the one
    distinction `test_a_row_with_no_instruction_prompt_sends_no_system_turn` exists
    to hold.
    """

    chat_template = "{{ messages }}"
    image_token = "<|image_pad|>"

    def __init__(self) -> None:
        self.roles: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.roles = [message["role"] for message in messages]
        rendered = ""
        for message in messages:
            blocks = "".join(
                "<img>" if block["type"] == "image" else block["text"]
                for block in message["content"]
            )
            rendered += f"<{message['role']}>{blocks}"
        return f"{rendered}{'<gen>' if add_generation_prompt else ''}"


class _QwenTemplated:
    """`Qwen/Qwen3-VL-Embedding-2B`'s template, in the one behaviour that matters here.

    Read off `chat_template.jinja` on the Hub (2026-08-03): it opens with
    `set default_system_message = 'Represent the user\\'s input.'` and, when the
    first message is not a system turn, emits
    `'<|im_start|>system\\n' + default_system_message + '<|im_end|>\\n'` before the
    user turn. A row that brings its own system turn gets that content instead.

    Stubbed rather than downloaded so this test needs no network; the real
    processor produced the same two strings in the run quoted below.
    """

    chat_template = "{{ messages }}"
    image_token = "<|image_pad|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if messages[0]["role"] == "system":
            system = "".join(block["text"] for block in messages[0]["content"])
        else:
            system = QWEN_INSTRUCTION
        body = "".join(
            "<|vision_start|><|image_pad|><|vision_end|>"
            if block["type"] == "image"
            else block["text"]
            for block in messages[-1]["content"]
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{body}<|im_end|>\n"
            f"{'<|im_start|>assistant\n' if add_generation_prompt else ''}"
        )


def _tokenize(text: str) -> list[str]:
    """Words, punctuation and special tokens as separate tokens.

    The question this file has to answer is how many times the instruction's *token
    sequence* occurs, not how many times its characters do, so the count below is
    taken over tokens. A real tokenizer would need the checkpoint; this one splits
    the same boundaries for the strings under test.
    """
    return re.findall(r"<\|[^|]+\|>|\w+|[^\s\w]", text)


def _occurrences(text: str, needle: str) -> int:
    haystack, sought = _tokenize(text), _tokenize(needle)
    return sum(
        1
        for start in range(len(haystack) - len(sought) + 1)
        if haystack[start : start + len(sought)] == sought
    )


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


def test_the_query_instruction_prompt_appears_once_in_a_templated_row():
    """`qwen3-vl-query-prompt-may-go-in-twice`, decided by tokenising.

    Measured 2026-08-03, `AutoProcessor.from_pretrained('Qwen/Qwen3-VL-Embedding-2B')`,
    transformers 5.14.1, on `Collate.pair_texts` output. The instruction's token
    sequence is `[65743, 279, 1196, 594, 1946, 13]`:

        before  len=30  occurrences=2
        after   len=24  occurrences=1   delta=-6

    Two, because the collate prefixed the instruction to the template's output and
    the template inserted its own `default_system_message` — the same string — for
    a row that carried no system turn. Handing it in as that system turn is what
    makes it one, and keeps the config field the thing that decides the text.
    """
    templated = format_prompt(
        _QwenTemplated(),
        "what colour is the roof",
        with_image=False,
        prompt_format="chat_template",
        add_generation_prompt=True,
        instruction_prompt=QWEN_INSTRUCTION,
    )

    assert _occurrences(templated, QWEN_INSTRUCTION) == 1, (
        f"the instruction occurs {_occurrences(templated, QWEN_INSTRUCTION)} times in "
        f"{templated!r}. Twice is the measured pre-fix state: prefixing it to the "
        "template's output leaves the template free to insert its own identical default, "
        "and every Qwen tokens/s figure then carries a denominator six tokens too large."
    )
    assert templated.startswith(f"<|im_start|>system\n{QWEN_INSTRUCTION}<|im_end|>")


def test_the_instruction_prompt_is_what_the_config_says_and_not_the_template_default():
    """Deleting the prefix alone would pass the test above and mean nothing.

    The template supplies the same instruction on its own, so a row would still read
    correctly while `configs/model/` had stopped deciding anything. The config value
    has to be the one that lands.
    """
    templated = format_prompt(
        _QwenTemplated(),
        "q",
        with_image=False,
        prompt_format="chat_template",
        add_generation_prompt=False,
        instruction_prompt="Encode this query for retrieval.",
    )

    assert "Encode this query for retrieval." in templated
    assert QWEN_INSTRUCTION not in templated, (
        "the template's own default survived a config that asked for something else, so "
        "model.instruction_prompt is a label rather than an input"
    )


def test_a_row_with_no_instruction_prompt_sends_no_system_turn():
    """Two of the three models declare none (`instruction_prompt: null`), and an
    empty system turn is not the same row as no system turn.

    It is one more `<|im_start|>system\\n<|im_end|>\\n` on every row of
    `qwen3_5_0_8b` and `gemma4_e2b`, so the sequence length grows and the tokens/s
    denominator moves. The roles are asserted as well as the rendering because that
    is the fact this guard is about; the rendering alone was checked against a stub
    that dropped every turn but the last, and dropping the `if` in
    `trainbench/prompt.py:146` then changed nothing any test could see.
    """
    processor = _Templated()

    templated = format_prompt(
        processor,
        "q",
        with_image=False,
        prompt_format="chat_template",
        add_generation_prompt=False,
    )

    assert processor.roles == ["user"]
    assert templated == "<user>q"


def test_a_raw_row_carries_the_instruction_ahead_of_the_placeholder():
    """`raw` has no template to hold a system turn, so the instruction is the head
    of the row — and it cannot collide with a default that does not exist."""
    assert (
        format_prompt(
            _Bare(),
            "describe the picture",
            with_image=True,
            prompt_format="raw",
            add_generation_prompt=False,
            instruction_prompt="Represent the image.",
        )
        == "Represent the image.<|image|>describe the picture"
    )
