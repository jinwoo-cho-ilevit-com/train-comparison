"""A probe must always produce a result, never an exception.

Pooling and loss have their own module (tests/test_embedding.py); what is tested
here is the probe machinery around them.
"""

from __future__ import annotations

import ast
import contextlib
import json
import math
import pathlib
import sys
import types

import pytest

from trainbench.config import to_bench_config
from trainbench.device import get_device
from trainbench.probe import registry, run_probe
from trainbench.probe.fixtures import PROBE_PAIRS
from trainbench.probe.steps import image_token_id, visual_token_count
from trainbench.probe.types import ProbeReport
from trainbench.record import write_json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import verify_env  # noqa: E402

import torch  # isort: skip

FRAMEWORKS = ["unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"]


class _StubProcessor:
    """The HF processor sentence-transformers holds inside its first module.

    `AutoProcessor.from_pretrained` is what lands there (sentence-transformers
    5.6.1 base/modules/transformer.py:671), and it is republished as
    `SentenceTransformer.processor` (base/model.py:1524).
    """


def compose_probe(*overrides):
    """A resolved probe mapping, composed rather than hand-written.

    Extra overrides are for the tests whose subject is a config group — the
    environment-bound kernel only exists on one architecture, and swapping the
    field by hand would build a model config no `configs/model/` file describes.
    """
    from hydra import compose, initialize_config_dir

    from trainbench.compose import resolve

    from .conftest import CONFIG_DIR

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["run=probe", "device=cpu", *overrides])
        return resolve(cfg)[1]


@pytest.fixture
def config_mapping(tmp_path):
    return compose_probe()


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_missing_framework_is_recorded_not_raised(config_mapping, framework):
    """Frameworks are absent outside their own image. That must be a failed check,
    not a crash, or an 18-pod sweep dies on its first unavailable combination."""
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = framework

    report = run_probe(to_bench_config(mapping), get_device("cpu"))

    assert isinstance(report, ProbeReport)
    assert report.checks, "a failed probe still has to say something"
    assert not report.all_ok


def test_report_survives_unserializable_detail(tmp_path):
    """One unserializable value must not cost the whole result file."""
    report = ProbeReport(framework="native", model="m")
    report.run("returns_tensor", lambda: {"t": torch.zeros(2)})

    path = write_json(tmp_path / "r.json", {"probe": report.to_dict()})

    assert json.loads(path.read_text())["probe"]["checks"][0]["ok"] is True


def test_a_crashing_adapter_keeps_the_checks_it_already_recorded(config_mapping, monkeypatch):
    """The checks an adapter records are the expensive ones — a model load, a
    backward pass. An adapter that dies outside `report.run` used to take all of
    them with it, so an 18-pod sweep learned nothing from the pods that crashed."""
    module = types.ModuleType("trainbench.probe._crashing")

    def run(config, device, report):
        report.run("cheap_check", lambda: {"recorded": True})
        raise RuntimeError("adapter died outside report.run")

    module.run = run
    monkeypatch.setitem(sys.modules, "trainbench.probe._crashing", module)
    monkeypatch.setitem(registry._MODULES, "native", "trainbench.probe._crashing")

    report = run_probe(to_bench_config(config_mapping), get_device("cpu"))

    assert [c.name for c in report.checks] == ["cheap_check", "probe_import"]
    assert report.checks[0].ok
    assert report.checks[1].error_type == "RuntimeError"
    assert not report.all_ok


def test_what_escapes_run_probe_is_still_filed_as_a_result(config_mapping, tmp_path, monkeypatch):
    """One level above the net `report.run` casts: a pod whose probe died between
    the adapters left no artifact at all, so the combination read as never
    attempted and the traceback stayed in a log that dies with the pod."""
    config_path = write_json(tmp_path / "resolved.json", config_mapping)
    out = tmp_path / "result.json"

    def boom(config, device):
        raise MemoryError("died between the adapters")

    monkeypatch.setattr(verify_env, "run_probe", boom)
    assert verify_env.main(["--config", str(config_path), "--out", str(out)]) == 1

    record = json.loads(out.read_text())
    check = record["probe"]["checks"][0]
    assert check["name"] == "probe_process"
    assert check["error_type"] == "MemoryError"
    assert "died between the adapters" in check["error"]
    assert "MemoryError" in check["traceback"]


def test_a_deliberate_exit_code_is_not_refiled_as_a_framework_failure(
    config_mapping, tmp_path, monkeypatch
):
    """The net catches `Exception`, not `BaseException`. Swallowing `SystemExit`
    rewrote a chosen exit code as this entry point's own 1 — the same laundering
    `docker/entrypoint.sh::run_with_secrets` exists to undo — and filed the exit as
    though the framework had refused the model."""
    config_path = write_json(tmp_path / "resolved.json", config_mapping)
    out = tmp_path / "result.json"
    monkeypatch.setattr(verify_env, "run_probe", lambda config, device: sys.exit(3))

    with pytest.raises(SystemExit) as raised:
        verify_env.main(["--config", str(config_path), "--out", str(out)])
    assert raised.value.code == 3
    assert not out.exists()


class _Config:
    """Stand-in for a transformers model config.

    `get_text_config` raises when there is no sub-config, standing in for the
    framework wrappers that expose an accessor which does not survive being called.
    """

    def __init__(self, text_config=None, **fields):
        self.__dict__.update(fields)
        self._text_config = text_config

    def get_text_config(self):
        if self._text_config is None:
            raise AttributeError("no text config")
        return self._text_config


class _Model:
    def __init__(self, config):
        self.config = config


def test_image_token_id_prefers_the_current_field_name():
    model = _Model(_Config(image_token_id=7, image_token_index=9))

    assert image_token_id(model) == (7, "config.image_token_id")


def test_image_token_id_falls_back_to_the_older_field_name():
    """transformers renamed `image_token_index`; a model config that still uses it
    would otherwise read as having no image tokens at all."""
    model = _Model(_Config(image_token_index=9))

    assert image_token_id(model) == (9, "config.image_token_index")


def test_image_token_id_falls_back_to_the_text_config():
    model = _Model(_Config(text_config=_Config(image_token_id=11)))

    assert image_token_id(model) == (11, "text_config.image_token_id")


def test_image_token_id_says_where_it_looked_when_there_is_none():
    model = _Model(_Config())

    with pytest.raises(ValueError, match="image_token_id"):
        image_token_id(model)


class _Processor:
    """Emits a fixed batch so the count under test is the one written here.

    It declares a chat template because `prompt_format=chat_template` is what these
    counts are measured under, and trainbench/prompt.py refuses that format from a
    processor that has none — a fake with no template would be refused too.
    """

    chat_template = "{{ messages }}"
    image_token = "<image>"

    def __init__(self, input_ids, padding_side="right", pad_token_id=None):
        self._input_ids = input_ids
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id
        self.images_seen = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<image>text"

    def __call__(self, text=None, images=None, return_tensors=None, padding=None):
        self.images_seen = images
        return {"input_ids": self._input_ids}


def _count(processor, model, max_tokens_per_image=None, prompt_format="chat_template"):
    return visual_token_count(
        processor, model, get_device("cpu"), "right", max_tokens_per_image, prompt_format
    )


def test_visual_token_count_reports_the_placeholder_count():
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6], [5, 7, 7, 7, 6]]))
    model = _Model(_Config(image_token_id=7))

    detail = _count(processor, model)

    assert detail["visual_tokens_per_image"] == 3
    assert detail["image_token_id_source"] == "config.image_token_id"
    assert detail["total_seq_len"] == 5


class _BareProcessor(_Processor):
    """gemma-4's shape: no chat template, an image token, and an
    `apply_chat_template` that raises what transformers 5.14.1 raises."""

    chat_template = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        raise AssertionError("prompt_format=raw must not reach apply_chat_template")


def test_the_probe_counts_visual_tokens_without_a_chat_template():
    """The 2026-08-02 campaign lost `visual_tokens` on three frameworks at once
    because this call was unconditional. The count itself is unchanged: what the
    format decides is the markup around the placeholders, not the placeholders."""
    processor = _BareProcessor(torch.tensor([[5] + [7] * 256 + [6]]))
    model = _Model(_Config(image_token_id=7))

    detail = _count(processor, model, max_tokens_per_image=280, prompt_format="raw")

    assert detail["visual_tokens_per_image"] == 256
    assert detail["prompt_format"] == "raw"


def test_the_probe_hands_the_processor_one_image_list_per_row():
    """Measured 2026-08-02: `Gemma4Processor` reads a flat list as one row carrying
    every image and refuses the batch, so no gemma-4 probe could build one. Both
    Qwen processors return byte-identical tensors either way."""
    processor = _BareProcessor(torch.tensor([[5] + [7] * 256 + [6]]))
    model = _Model(_Config(image_token_id=7))

    _count(processor, model, max_tokens_per_image=280, prompt_format="raw")

    seen = processor.images_seen
    assert all(isinstance(row, list) for row in seen), f"one list per row, got {seen!r}"
    assert [len(row) for row in seen] == [1] * len(PROBE_PAIRS)


def test_visual_token_count_refuses_a_zero_count():
    """Zero means the id is wrong or the placeholders never got inserted. Reporting
    it as a measurement would rescale every tokens/s figure that divides by it."""
    processor = _Processor(torch.tensor([[5, 1, 2, 3, 6]]))
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="outside"):
        _count(processor, model)


def test_visual_token_count_refuses_an_all_image_batch():
    """apply_chat_template always wraps the placeholders in role and text tokens, so
    matching every position means the id matched something else."""
    processor = _Processor(torch.tensor([[7, 7, 7, 7]]))
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="outside"):
        _count(processor, model)


def test_visual_token_count_refuses_the_pad_token_id():
    """`0 < n < seq_len` accepts a pad id happily: padded rows are neither empty nor
    full, so the count comes back looking like a measurement of the model when it is
    a measurement of the batch shape.

    Declared with a cap, because a pad count sits under the cap and the cap check
    would wave it through. This gate is the one that has to catch it."""
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6], [5, 7, 7, 7, 6]]), pad_token_id=7)
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="pad token id"):
        _count(processor, model, max_tokens_per_image=280)


def test_visual_token_count_refuses_per_sample_disagreement():
    """Every row carries the same image, so counts that differ mean the id matched
    something the rows do not share. Grading per_sample[0] alone accepted this."""
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6], [5, 7, 7, 6, 6]]))
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="disagree"):
        _count(processor, model)


def test_visual_token_count_refuses_a_count_above_the_declared_cap():
    """The cap is what the processor can emit at most. A count above it is a count
    of something else, and it would divide every tokens/s figure."""
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6]]))
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="max_tokens_per_image"):
        _count(processor, model, max_tokens_per_image=2)


def test_visual_token_count_accepts_the_gemma4_count_the_processor_actually_emits():
    """gemma-4's declared 280 is max_soft_tokens, and the processor derives each
    image's count from its aspect ratio: 448x448 (PROBE_IMAGE_SIZE) gives 256,
    768x256 gives 252, and 960x672 gives 280 (measured, transformers 5.14.1).
    This used to be an equality against 280, so every gemma-4 probe died here on a
    correct measurement — the shape that gets a gate relaxed instead of fixed."""
    processor = _Processor(torch.tensor([[5] + [7] * 256 + [6]]))
    model = _Model(_Config(image_token_id=7))

    detail = _count(processor, model, max_tokens_per_image=280)

    assert detail["visual_tokens_per_image"] == 256
    assert detail["declared_max_tokens_per_image"] == 280


def test_every_probe_adapter_hands_visual_token_count_the_declared_cap():
    """The cap only guards anything if the adapters actually pass it.

    Read off the call site rather than run, because reaching that line needs a real
    checkpoint. Measured with in-memory mutants: replacing the argument with `None`,
    or leaving one adapter on a stale field name after a rename, passed the whole
    suite and would have surfaced as an AttributeError on a pod. An adapter that
    stops calling `visual_token_count` at all is named here too — dropping the check
    is the other way to make this pass.
    """
    expected = ("config", "model", "max_tokens_per_image")
    adapters = _adapters_calling_visual_token_count()
    wrong = {}
    for path in adapters:
        source = path.read_text()
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "visual_token_count"
        ]
        for call in calls:
            names = [
                tuple(_attribute_chain(arg)) for arg in call.args if isinstance(arg, ast.Attribute)
            ]
            if expected not in names:
                wrong[path.name] = ast.unparse(call)
        if not calls:
            wrong[path.name] = "names visual_token_count but never calls it"

    assert not wrong, wrong
    # An empty sweep would pass on nothing at all.
    assert [path.name for path in adapters] == ["ms_swift.py", "native.py", "unsloth.py"]


def _attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return reversed(parts)


def _adapters_calling_visual_token_count():
    probe = pathlib.Path(__file__).parent.parent / "trainbench" / "probe"
    return [
        path
        for path in sorted(probe.glob("*.py"))
        if path.name != "steps.py" and "visual_token_count" in path.read_text()
    ]


class _Tokenizer:
    def __init__(self, padding_side):
        self.padding_side = padding_side


def _checkpoint_declares(monkeypatch, side, source="tokenizer_config.json"):
    """Stand in for the hub read. What the checkpoint declares is the input to the
    comparison under test; downloading it here would test huggingface_hub."""
    from trainbench.probe import steps

    monkeypatch.setattr(
        steps,
        "checkpoint_padding_side",
        lambda hf_id, revision=None: {"padding_side": side, "source": source},
    )


def test_padding_side_alignment_fails_loudly_when_the_checkpoint_disagrees(monkeypatch):
    """`config.model.padding_side` used to be a claim nothing checked: with the
    processor padding the other way, both pooling branches returned a PAD
    embedding without an exception or a warning."""
    from trainbench.probe.steps import padding_side_alignment

    _checkpoint_declares(monkeypatch, "left")
    processor = _Processor(torch.tensor([[1]]), padding_side="left")

    with pytest.raises(ValueError, match="model-spec"):
        padding_side_alignment(processor, "right", "org/model")
    # Forced anyway: the check is what makes the disagreement loud, and the checks
    # that run after it still have to pool a real token.
    assert processor.padding_side == "right"


def test_padding_side_alignment_passes_when_they_agree(monkeypatch):
    from trainbench.probe.steps import padding_side_alignment

    _checkpoint_declares(monkeypatch, "left")
    processor = _Processor(torch.tensor([[1]]), padding_side="left")
    processor.tokenizer = _Tokenizer("left")

    detail = padding_side_alignment(processor, "left", "org/model")

    assert detail["disagreed"] == []
    assert detail["declared_before"] == {"tokenizer": "left", "processor": "left"}
    assert detail["framework_forced"] == []


def test_a_framework_forcing_the_padding_side_is_not_the_spec_going_stale(monkeypatch):
    """unsloth sets `padding_side = "left"` at the end of `from_pretrained`
    whatever the checkpoint says (unsloth 2026.7.6 models/vision.py:1716-1718).
    Grading the object it returned sent both Qwen cells of the 2026-08-02 campaign
    to the support matrix as "docs/model-spec.yaml no longer matches this
    checkpoint" — while `native` read `right` off the same two checkpoints, which
    is what the spec says. The framework's decision has to be recorded, not
    charged to the spec."""
    from trainbench.probe.steps import padding_side_alignment

    _checkpoint_declares(monkeypatch, "right")
    processor = _Processor(torch.tensor([[1]]), padding_side="left")
    processor.tokenizer = _Tokenizer("left")

    detail = padding_side_alignment(processor, "right", "Qwen/Qwen3-VL-Embedding-2B")

    assert detail["checkpoint_padding_side"] == "right"
    assert detail["framework_forced"] == ["processor", "tokenizer"]
    assert processor.padding_side == "right"


def test_the_padding_side_check_reads_the_file_the_spec_names(monkeypatch, tmp_path):
    """`docs/model-spec.yaml` gives `tokenizer_config.json` as the source of
    `padding_side`, and a checkpoint that names no key is `right` by transformers'
    default — which is the entry the spec carries for qwen3_vl_emb_2b."""
    import huggingface_hub

    from trainbench.probe import steps

    written = tmp_path / "tokenizer_config.json"

    def _download(hf_id, filename, revision=None):
        assert filename == "tokenizer_config.json"
        return str(written)

    # The real attribute, not a name on `steps`: the import is inside the function,
    # so rebinding it anywhere else leaves the download in place.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)

    written.write_text(json.dumps({"padding_side": "left"}))
    assert steps.checkpoint_padding_side("org/model") == {
        "padding_side": "left",
        "source": "tokenizer_config.json",
    }

    written.write_text(json.dumps({"model_max_length": 8192}))
    absent = steps.checkpoint_padding_side("org/model")
    assert absent["padding_side"] == steps.TRANSFORMERS_DEFAULT_PADDING_SIDE == "right"
    assert "names none" in absent["source"]


class _Encoder(torch.nn.Module):
    """The smallest thing `steps.encode` accepts, with real parameters.

    A counter set by the test would prove nothing about the shape under test, so
    the freeze and the backward are the real ones.
    """

    def __init__(self, vocab=8, hidden=4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.proj = torch.nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        return types.SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)))


def _frozen_batch():
    ids = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_a_step_that_trained_nothing_is_not_a_step(monkeypatch):
    """The 2026-08-02 campaign recorded `infonce_backward ok=True` with
    `params_with_grad=0, trainable_params=0` on all three unsloth cells, because a
    finite loss was the whole of the evidence.

    The shape is reproduced rather than simulated. unsloth reaches it by calling
    `FastVisionModel.from_pretrained` without `full_finetuning`, which makes
    unsloth_zoo's `prepare_model_for_training` freeze every parameter with no LoRA
    marker in its name (unsloth_zoo 2026.7.7 training_utils.py:383) — and the same
    function calls `enable_input_require_grads` (:479-484), which puts
    `requires_grad` on the embedding *output*. That is why `loss.backward()`
    returns instead of raising "does not require grad", and why the cell passed.
    """
    from trainbench.probe.steps import infonce_backward

    model = _Encoder()
    model.requires_grad_(False)
    model.embed.register_forward_hook(lambda module, args, output: output.requires_grad_(True))

    with pytest.raises(ValueError, match="trained nothing"):
        infonce_backward(model, _frozen_batch(), 0.02, "right")


def test_the_frozen_graph_it_refuses_really_does_produce_a_finite_loss():
    """The other half of the same claim: had the guard graded the loss, or had the
    backward raised on its own, there would have been nothing to catch. This is
    what the check used to record as a pass."""
    from trainbench.embedding import info_nce

    model = _Encoder()
    model.requires_grad_(False)
    model.embed.register_forward_hook(lambda module, args, output: output.requires_grad_(True))

    hidden = model(**_frozen_batch()).last_hidden_state[:, -1]
    loss = info_nce(hidden[:2], hidden[2:], 0.02)
    loss.backward()

    assert torch.isfinite(loss)
    assert sum(1 for p in model.parameters() if p.requires_grad) == 0


def test_a_step_that_did_train_still_passes():
    from trainbench.probe.steps import infonce_backward

    detail = infonce_backward(_Encoder(), _frozen_batch(), 0.02, "right")

    # embedding weight, linear weight, linear bias
    assert detail["trainable_params"] == detail["total_params"] == 3
    assert detail["params_with_grad"] == 3


def test_a_backward_that_reaches_no_trainable_parameter_is_refused():
    """The other frozen shape: parameters marked trainable that the loss is not a
    function of. `trainable_params` alone reads as a healthy run."""
    from trainbench.probe.steps import infonce_backward

    model = _Encoder()
    detached = torch.nn.Module()
    detached.forward = lambda **kwargs: types.SimpleNamespace(  # noqa: ARG005
        last_hidden_state=torch.randn(4, 3, 4, requires_grad=True)
    )
    detached.parameters = model.parameters
    detached.train = model.train
    detached.zero_grad = model.zero_grad

    with pytest.raises(ValueError, match="reached none of them"):
        infonce_backward(detached, _frozen_batch(), 0.02, "right")


def test_verify_axes_reports_the_adapter_that_ran_not_the_framework_requested(config_mapping):
    """docs/CONTRACTS.md §2: the framework literal an adapter passes is the evidence
    of which code path ran. A registry routing `framework=ms_swift` to another
    adapter must show up as a mismatch, not as a match on the request."""
    from trainbench.probe.steps import verify_axes

    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "ms_swift"
    report = ProbeReport(framework="ms_swift", model="m")

    verify_axes(
        torch.nn.Linear(2, 2), to_bench_config(mapping), get_device("cpu"), "native", report
    )

    state = {a.axis: a for a in report.applied.axes}["framework.name"]
    assert (state.requested, state.applied) == ("ms_swift", "native")
    assert not state.matches


def _unsloth_module(fast_st_accepts):
    """Stand-in for the unsloth package, which installs only inside its own image."""

    class FastVisionModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("FastVisionModel does not accept this checkpoint")

    class FastSentenceTransformer:
        @staticmethod
        def from_pretrained(hf_id, for_inference=True):
            if not fast_st_accepts:
                raise RuntimeError("encoder-only models only")
            return FastSentenceTransformer()

    module = types.ModuleType("unsloth")
    module.__version__ = "0.0.0-test"
    module.FastVisionModel = FastVisionModel
    module.FastSentenceTransformer = FastSentenceTransformer
    return module


def _unsloth_report(config_mapping, monkeypatch, fast_st_accepts):
    monkeypatch.setitem(sys.modules, "unsloth", _unsloth_module(fast_st_accepts))
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "unsloth"
    report = run_probe(to_bench_config(mapping), get_device("cpu"))
    return {c.name: c for c in report.checks}, report


def test_a_documented_refusal_is_an_answer_not_a_broken_cell(config_mapping, monkeypatch):
    """FastSentenceTransformer rejecting a VLM is the finding the check went for.
    Unmarked it rendered as a plain FAIL, which is the reading docs/CONTRACTS.md §3
    exists to prevent."""
    checks, report = _unsloth_report(config_mapping, monkeypatch, fast_st_accepts=False)
    check = checks["fast_sentence_transformer_accepts_vlm"]

    assert (check.ok, check.expected_failure) == (False, True)
    assert report.unexpected_passes == []


def test_a_documented_limitation_that_disappears_is_reported(config_mapping, monkeypatch):
    """If Unsloth starts accepting VLMs, the support matrix is wrong and this run is
    the only place that knows. `all_ok` cannot say it."""
    checks, report = _unsloth_report(config_mapping, monkeypatch, fast_st_accepts=True)

    assert checks["fast_sentence_transformer_accepts_vlm"].ok
    assert report.unexpected_passes == ["fast_sentence_transformer_accepts_vlm"]


def test_ple_report_fails_when_nothing_matches():
    """A zero match used to be reported as ok. Roughly half of gemma-4-E2B lives in
    these tables, so it means `freeze.ple` would freeze nothing and say it did."""
    from trainbench.probe.native import _ple_report

    with pytest.raises(ValueError, match="per_layer"):
        _ple_report(torch.nn.Linear(2, 2))


def _stub_transformers(monkeypatch):
    """`from_pretrained` without a checkpoint, for the checks that come after it.

    The probe's later checks are allowed to fail against these — what is under test
    is which checks get *recorded*, and a stub that makes them all pass would be a
    stub asserting itself.
    """

    def _from_pretrained(*args, **kwargs):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        model.config = types.SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
        return model

    # `from_pretrained` on the classes rather than the names on the package:
    # `transformers` is a lazy module, and rebinding `transformers.AutoModel`
    # leaves `from transformers import AutoModel` — which is how the probe reaches
    # it — still resolving to the real class. That mistake downloads a 2B
    # checkpoint into a unit test and the test still passes.
    from transformers import AutoModel, AutoProcessor

    monkeypatch.setattr(AutoModel, "from_pretrained", _from_pretrained)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", _from_pretrained)


def _native_report(config_mapping, monkeypatch, **overrides):
    mapping = json.loads(json.dumps(config_mapping))
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        mapping[section][key] = value
    _stub_transformers(monkeypatch)
    return run_probe(to_bench_config(mapping), get_device("cpu"))


def test_a_refused_load_axis_does_not_read_as_a_model_that_will_not_load(
    config_mapping, monkeypatch
):
    """`peft.mode=qlora` off CUDA is refused by `axes.load_kwargs`. Evaluating that
    refusal inside the `model_load` lambda charged it to the checkpoint — the cell
    read "the model does not load", and `if not ok: return` ended the probe three
    checks in, so the nine checks that have nothing to do with the axis were lost
    from a support matrix whose job is to decide which cells get measured.

    The refusal must still be visible; it is just not the load's fault.
    """
    report = _native_report(config_mapping, monkeypatch, **{"peft.mode": "qlora", "peft.r": 32})
    names = [c.name for c in report.checks]
    check = next(c for c in report.checks if c.name == "axes_load_kwargs")

    assert (check.ok, check.error_type) == (False, "UnappliedAxis")
    assert names.index("axes_load_kwargs") < names.index("model_load")
    # The load went ahead without the kwargs, so the probe still answers its own
    # question. Every one of these is a check the misclassification cost.
    assert {
        "model_load",
        "axes_assemble",
        "axes_verified",
        "text_tokenize",
        "infonce_backward",
        "lora_attach",
    } <= set(names), names
    # And nobody can take the bare load for the requested one. `axes_verified`
    # cannot say so — `assert_matches` returns immediately for `purpose=probe` —
    # so what carries the refusal is the state it recorded and `all_ok`.
    assert not report.all_ok
    peft = next(a for a in report.applied.axes if a.axis == "peft.mode")
    assert (peft.requested, peft.applied) == ("qlora", "full")
    assert not report.applied.to_dict()["all_matched"]


def test_the_same_refusal_does_not_read_as_a_framework_that_cannot_load(
    config_mapping, monkeypatch
):
    """The sentence_transformers adapter had the load-time axes in the same place,
    and one more way to trip over them: it is the path that imports
    `BitsAndBytesConfig`, so an image without bitsandbytes failed the load with an
    ImportError. Either way the cell read "this framework cannot load this model"
    and `encode` and `mnrl_backward` were skipped for a reason that was never true.

    The package is absent here, so it is stood in for — what is under test is the
    adapter's order of operations, which is ours, not ST's behaviour.
    """
    module = types.ModuleType("sentence_transformers")

    class _SentenceTransformer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.processor = _StubProcessor()

        def __iter__(self):
            return iter([self.linear])

        def get_sentence_embedding_dimension(self):
            return 2

    module.SentenceTransformer = _SentenceTransformer
    module.__version__ = "0.0-stub"
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "sentence_transformers"
    mapping["peft"]["mode"] = "qlora"
    mapping["peft"]["r"] = 32

    report = run_probe(to_bench_config(mapping), get_device("cpu"))
    checks = {c.name: c for c in report.checks}

    assert checks["axes_load_kwargs"].error_type == "UnappliedAxis"
    assert checks["sentence_transformer_load"].ok
    # Recorded on their own merits rather than written off with the load.
    for name in ("encode", "mnrl_backward"):
        assert checks[name].error_type != "Skipped", checks[name].error


def _fake_axolotl(monkeypatch, loaded):
    """A miniature of axolotl 0.18.0 keeping the four behaviours the probe trips on.

    Not a stub that says yes: `DictDefault` answers None for a missing key
    (utils/dict.py:12), `normalize_config` divides `batch_size //
    micro_batch_size` before anything has filled them (utils/config/__init__.py:200-203),
    `validate_config` is what puts defaults on all of them and refuses a cfg
    missing `learning_rate`, an empty `datasets` or fewer than two of the three
    batch keys (utils/schemas/training.py:71-82, utils/schemas/config.py:339-348,
    utils/schemas/validation.py:219-225), and the loader compares
    `context_parallel_size` against 1 (loaders/patch_manager.py:336), which only
    validation supplies.
    """

    class DictDefault(dict):
        def __missing__(self, key):
            return None

        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

        def to_dict(self):
            return dict(self)

    def normalize_config(cfg):
        cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
            cfg.batch_size // cfg.micro_batch_size
        )
        cfg.batch_size = cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps

    def validate_config(cfg):
        if cfg.learning_rate is None:
            raise ValueError("learning_rate: Field required")
        if not cfg.datasets:
            raise ValueError("datasets: List should have at least 1 item after validation")
        keys = ("micro_batch_size", "gradient_accumulation_steps", "batch_size")
        if sum(1 for key in keys if cfg[key]) < 2:
            raise ValueError(f"At least two of {', '.join(keys)} must be set")
        cfg.context_parallel_size = 1
        return cfg

    class ModelLoader:
        def __init__(self, cfg, tokenizer):
            self.cfg = cfg

        def load(self):
            if self.cfg.context_parallel_size > 1:
                raise RuntimeError("unreachable with a validated cfg")
            loaded["cfg"] = dict(self.cfg)
            model = torch.nn.Sequential(torch.nn.Linear(2, 2))
            model.config = types.SimpleNamespace(_attn_implementation="sdpa", sub_configs=())
            return model, None

    root = types.ModuleType("axolotl")
    root.__version__ = "0.0-stub"
    modules = {
        "axolotl": root,
        "axolotl.loaders": types.ModuleType("axolotl.loaders"),
        "axolotl.loaders.model": types.ModuleType("axolotl.loaders.model"),
        "axolotl.loaders.tokenizer": types.ModuleType("axolotl.loaders.tokenizer"),
        "axolotl.utils": types.ModuleType("axolotl.utils"),
        "axolotl.utils.config": types.ModuleType("axolotl.utils.config"),
        "axolotl.utils.dict": types.ModuleType("axolotl.utils.dict"),
    }
    modules["axolotl.loaders.model"].ModelLoader = ModelLoader
    modules["axolotl.loaders.tokenizer"].load_tokenizer = lambda cfg: _Tokenizer("right")
    modules["axolotl.utils.config"].normalize_config = normalize_config
    modules["axolotl.utils.config"].validate_config = validate_config
    modules["axolotl.utils.config"].prepare_plugins = lambda cfg: None
    modules["axolotl.utils.dict"].DictDefault = DictDefault
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_the_axolotl_probe_hands_it_a_config_it_accepts(config_mapping, monkeypatch):
    """The probe called `normalize_config` before `validate_config` and never
    called validation at all, so every axolotl cell of the 2026-08-02 campaign
    died on `None // None` before a checkpoint was touched — a support matrix
    saying axolotl cannot load these models, measured against a config axolotl was
    never given.

    Patching the two batch keys by hand is not the fix and this asserts why: only
    validation fills `context_parallel_size`, which the loader compares against 1.
    """
    loaded: dict[str, object] = {}
    _fake_axolotl(monkeypatch, loaded)
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "axolotl"

    report = run_probe(to_bench_config(mapping), get_device("cpu"))
    check = next(c for c in report.checks if c.name == "model_loader_load")

    assert check.ok, check.error
    cfg = loaded["cfg"]
    # Every required key traced to this study's config rather than invented.
    config = to_bench_config(mapping)
    assert cfg["micro_batch_size"] == config.train.batch_size
    assert cfg["gradient_accumulation_steps"] == config.train.grad_accum
    assert cfg["learning_rate"] == config.optim.lr
    assert [d["path"] for d in cfg["datasets"]] == [config.data.repo_id]


class _AxolotlTokenizer:
    """A bare tokenizer, the shape `load_tokenizer` hands back — no `.tokenizer`
    wrapper, so `align_padding_side` keeps `padding_side` on the object itself."""

    def __init__(self, padding_side="right"):
        self.padding_side = padding_side

    def __call__(self, text=None, return_tensors=None, padding=None):
        ids = torch.arange(len(text) * 3).reshape(len(text), 3) % 8
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _stub_axolotl_run(monkeypatch, model, tokenizer):
    """Stand in for the axolotl package plus this probe's own `load()`.

    The validate/normalize/load order inside `load()` is already pinned by
    `test_the_axolotl_probe_hands_it_a_config_it_accepts`; here that call is
    replaced wholesale so what is under test is only what `run()` does with the
    (model, tokenizer) it gets back.
    """
    import trainbench.probe.axolotl as axolotl_probe

    root = types.ModuleType("axolotl")
    root.__version__ = "0.0-stub"
    monkeypatch.setitem(sys.modules, "axolotl", root)
    monkeypatch.setattr(
        axolotl_probe, "load", lambda config, device, load_kwargs: (model, tokenizer)
    )


def test_the_axolotl_probe_enters_step_context_before_the_backward(config_mapping, monkeypatch):
    """The probe used to call `steps.infonce_backward` directly, so axolotl's own
    `required_step_context` (`trainbench/loader.py` `ADAPTERS["axolotl"]`) was
    declared but never entered. On a real pod this is the fp32/bf16 matmul
    (`RuntimeError: expected mat1 and mat2 to have the same dtype`, measured
    2026-08-03). This pins that `axes.step_context` is now entered, with that
    exact requirement, and that the step runs inside it rather than around it.
    """
    from trainbench import axes
    from trainbench.loader import ADAPTERS

    _stub_axolotl_run(monkeypatch, _Encoder(), _AxolotlTokenizer())
    calls = []
    entered = []

    @contextlib.contextmanager
    def _fake_ctx():
        entered.append("enter")
        yield
        entered.append("exit")

    def _spy(config, required=None):
        calls.append(required)
        return _fake_ctx()

    monkeypatch.setattr(axes, "step_context", _spy)
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "axolotl"

    report = run_probe(to_bench_config(mapping), get_device("cpu"))
    check = next(c for c in report.checks if c.name == "infonce_backward")

    assert calls == [ADAPTERS["axolotl"].required_step_context]
    assert check.ok, check.error
    # entered before the step and exited after it, not skipped and not wrapped
    # around a step that ran outside it
    assert entered == ["enter", "exit"]


def test_on_a_non_cuda_host_the_backward_is_refused_not_silently_measured(
    config_mapping, monkeypatch
):
    """`axes._autocast_step_context` raises `UnappliedAxis` when the required
    `device_type` ("cuda") does not match the resolved device. This host has no
    CUDA, so this exercises the real, unpatched `axes.step_context`: the check
    must fail loudly with that reason rather than being skipped or measured in a
    regime axolotl does not train in.
    """
    _stub_axolotl_run(monkeypatch, _Encoder(), _AxolotlTokenizer())
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "axolotl"

    report = run_probe(to_bench_config(mapping), get_device("cpu"))
    check = next(c for c in report.checks if c.name == "infonce_backward")

    assert check.error_type == "UnappliedAxis"
    assert "cuda autocast" in check.error


def _fake_unsloth(monkeypatch, seen):
    """unsloth installs only inside its own image, so what is stood in for is the
    one behaviour under test: `from_pretrained` recording what it was asked for."""
    module = types.ModuleType("unsloth")

    class _FastVisionModel:
        @staticmethod
        def from_pretrained(hf_id, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("no checkpoint here; the kwargs are what was asked")

    module.FastVisionModel = _FastVisionModel
    module.FastSentenceTransformer = _FastVisionModel
    module.__version__ = "0.0-stub"
    monkeypatch.setitem(sys.modules, "unsloth", module)


@pytest.mark.parametrize(
    ("mode", "full_finetuning"), [("full", True), ("lora", False), ("qlora", False)]
)
def test_unsloth_is_told_whether_the_run_finetunes(
    config_mapping, monkeypatch, mode, full_finetuning
):
    """`full_finetuning` defaults to False, and with 4bit, 8bit and full all off
    unsloth switches to "16bit LoRA" and exports UNSLOTH_ENABLE_FULL_FINETUNING=0
    (unsloth 2026.7.6 models/vision.py:1164-1187). `post_patch_model` reads that
    back and unsloth_zoo freezes every parameter without a LoRA marker
    (:2094-2129; unsloth_zoo 2026.7.7 training_utils.py:383) — and under
    `peft.mode=full` nothing attaches LoRA, so nothing survives.

    Passing it is not a preference: the three unsloth cells of the 2026-08-02
    campaign trained zero parameters because it was left out. `infonce_backward`'s
    guard catches the result; this is what stops it happening.
    """
    seen: dict[str, object] = {}
    _fake_unsloth(monkeypatch, seen)
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "unsloth"
    mapping["peft"]["mode"] = mode
    if mode != "full":
        mapping["peft"]["r"] = 32

    run_probe(to_bench_config(mapping), get_device("cpu"))

    assert seen.get("full_finetuning") is full_finetuning, seen
    assert seen.get("load_in_4bit") is (mode == "qlora"), seen


def test_the_load_axes_reach_from_pretrained_when_they_are_not_refused(config_mapping, monkeypatch):
    """The break for the test above. Recording the refusal is worth nothing if the
    kwargs stopped reaching the loader on the way — a `load_kwargs` helper that
    returned `{}` unconditionally would keep every assertion above green while
    each run built its model on default attention.

    `attn.name` is the load-time axis that is not refused anywhere, so it is what
    can be watched arriving.
    """
    seen: dict[str, object] = {}

    def _from_pretrained(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("no checkpoint here; the kwargs are what was asked")

    from transformers import AutoModel

    _stub_transformers(monkeypatch)
    monkeypatch.setattr(AutoModel, "from_pretrained", _from_pretrained)
    mapping = json.loads(json.dumps(config_mapping))
    # Not the default: a helper that hardcoded sdpa would pass on the default and
    # build every other run on the wrong kernel.
    mapping["attn"]["name"] = "flex"

    report = run_probe(to_bench_config(mapping), get_device("cpu"))

    assert seen.get("attn_implementation") == "flex_attention", seen
    check = next(c for c in report.checks if c.name == "axes_load_kwargs")
    assert (check.ok, check.detail) == (True, {"requested": ["attn_implementation"]})


def test_the_load_axes_reach_from_pretrained_as_the_whole_mapping(config_mapping, monkeypatch):
    """One key is not the mapping.

    The test above watches `attn_implementation` arrive, and `attn.name` is the
    only load-time axis that is never refused — so it is the only key any config
    this suite can compose puts in that mapping, and a helper that forwarded attn
    while dropping everything else passed the whole suite. `peft.mode=qlora` is
    the key that would be dropped: the base's 4-bit recipe is produced while the
    checkpoint is read, so a probe that lost it would load a full-precision base
    and file the support-matrix cell as qlora.

    Nothing here quantises. The device is the one thing forced, because
    `axes.load_kwargs` refuses off CUDA and the mapping under test only has a
    second key when it is not refused; what is compared is the request, which is
    all this helper is responsible for. Whether 4-bit weights actually arrive is a
    GPU question and `applied._capture_peft`'s — 측정 안 함 here.
    """
    from trainbench import axes
    from trainbench.probe import steps

    monkeypatch.setattr(axes, "get_device", lambda name: torch.device("cuda"))
    mapping = json.loads(json.dumps(config_mapping))
    mapping["attn"]["name"] = "flex"
    mapping["peft"]["mode"] = "qlora"
    mapping["peft"]["r"] = 32
    config = to_bench_config(mapping)

    def comparable(kwargs):
        # `BitsAndBytesConfig` has no `__eq__`, so two equal recipes are two
        # unequal objects; the dict it exports is what "the same mapping" means.
        return {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in kwargs.items()}

    wanted = axes.load_kwargs(config)
    report = ProbeReport(framework="native", model=config.model.name)
    reached = steps.load_kwargs(config, report)

    # Asserted rather than assumed: if the axis started refusing again this would
    # be one key and the comparison below would pass by having nothing to compare.
    assert set(wanted) == {"attn_implementation", "quantization_config"}
    assert comparable(reached) == comparable(wanted)
    check = next(c for c in report.checks if c.name == "axes_load_kwargs")
    assert (check.ok, check.detail) == (True, {"requested": sorted(wanted)})


def _stub_sentence_transformers(monkeypatch, frozen):
    """sentence-transformers stood in for, with the one shape the guard is about real.

    Frozen means what unsloth made real and what could happen here: every
    parameter has `requires_grad=False` while the graph stays differentiable
    through a hook on the embedding *output*, which is what
    `enable_input_require_grads` does. Counters are not simulated — the freeze,
    the forward and the backward are the real ones, so a guard that only looked at
    the loss would still see a finite number.

    `BaseModel` is an `nn.Sequential` (sentence-transformers 5.6.1
    base/model.py:50), so `.parameters()` reaching the backbone is ST's own shape;
    the package installs only inside its own image, which is why it is stood in for.
    """
    module = types.ModuleType("sentence_transformers")

    class _SentenceTransformer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4)
            self.processor = _StubProcessor()
            if frozen:
                self.requires_grad_(False)
                self.embed.register_forward_hook(
                    lambda module, args, output: output.requires_grad_(True)
                )

        def __iter__(self):
            return iter([self.embed, self.proj])

        def get_sentence_embedding_dimension(self):
            return 4

        def tokenize(self, texts):
            ids = torch.arange(len(texts) * 3).reshape(len(texts), 3) % 8
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def forward(self, features):
            return {"sentence_embedding": self.proj(self.embed(features["input_ids"]))[:, -1]}

        def encode(self, texts, convert_to_tensor=True):
            return self(self.tokenize(texts))["sentence_embedding"]

    module.SentenceTransformer = _SentenceTransformer
    module.__version__ = "0.0-stub"
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _sentence_transformers_checks(config_mapping, monkeypatch, frozen):
    _stub_sentence_transformers(monkeypatch, frozen)
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "sentence_transformers"
    report = run_probe(to_bench_config(mapping), get_device("cpu"))
    return {c.name: c for c in report.checks}


def test_the_sentence_transformers_load_hands_back_a_processor_not_the_model(
    config_mapping, monkeypatch
):
    """`AdapterOut.processor` is called as an HF processor and this adapter used to
    return the model in that slot, so every ST timing batch raised before a step.

    `trainbench/collate.py:326` does `processor(text=..., return_tensors=...)` and
    reads `chat_template` off it; `SentenceTransformer.forward(input, **kwargs)`
    (sentence-transformers 5.6.1 base/model.py:496) answers neither.
    """
    from trainbench.loader import AdapterRefusal
    from trainbench.probe import sentence_transformers as adapter

    _stub_sentence_transformers(monkeypatch, frozen=False)
    config = to_bench_config(config_mapping)

    model, processor = adapter.load(config, get_device("cpu"), {})

    assert processor is not model
    assert processor is model.processor
    # A tokeniser is the same answer for a text-only checkpoint; a model is not.
    assert adapter.processor_of(types.SimpleNamespace(tokenizer="t")) == "t"
    with pytest.raises(AdapterRefusal, match="nothing to tokenise"):
        adapter.processor_of(types.SimpleNamespace())


def test_a_sentence_transformers_frozen_model_is_refused_like_every_other(
    config_mapping, monkeypatch
):
    """This adapter computes its own loss, so `steps.infonce_backward`'s guard was
    not on it: it returned `params_with_grad` alone and never counted
    `trainable_params` or compared it to zero. The 2026-08-02 values were
    310/320/505, but a fully frozen model here would have been recorded green —
    the same reading that filed three unsloth cells as supported."""
    checks = _sentence_transformers_checks(config_mapping, monkeypatch, frozen=True)
    check = checks["mnrl_backward"]

    assert (check.ok, check.error_type) == (False, "ValueError"), check.error
    assert "trained nothing" in check.error


def test_the_sentence_transformers_frozen_graph_it_refuses_returns_a_finite_loss(
    config_mapping, monkeypatch
):
    """The other half of the same claim: had the backward raised on its own, or had
    the loss come back non-finite, there would have been nothing for the old check
    to pass on. The refused message carries the loss the old one would have
    published."""
    checks = _sentence_transformers_checks(config_mapping, monkeypatch, frozen=True)

    loss = float(checks["mnrl_backward"].error.split("loss=")[1].split(" ")[0])
    assert math.isfinite(loss)


def test_a_sentence_transformers_step_that_did_train_still_passes(config_mapping, monkeypatch):
    """The break for the test above: a guard that refused everything would satisfy
    it and close the framework."""
    checks = _sentence_transformers_checks(config_mapping, monkeypatch, frozen=False)
    check = checks["mnrl_backward"]

    assert check.ok, check.error
    # embedding weight, linear weight, linear bias
    assert check.detail["trainable_params"] == check.detail["total_params"] == 3
    assert check.detail["params_with_grad"] == 3


def _axes_verified(mapping, framework_literal):
    from trainbench.probe.steps import verify_axes

    report = ProbeReport(framework=mapping["framework"]["name"], model="m")
    verify_axes(
        torch.nn.Linear(2, 2),
        to_bench_config(mapping),
        get_device("cpu"),
        framework_literal,
        report,
    )
    return next(c for c in report.checks if c.name == "axes_verified"), report


def test_axes_verified_refuses_a_mismatch_rather_than_recording_it_green(config_mapping):
    """`assert_matches` returns immediately for `purpose=probe`, so this check was
    green on `all_matched: false` — and two mismatches were already sitting under
    that green in the 2026-08-02 matrix (docs/support-matrix.md)."""
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "ms_swift"

    check, report = _axes_verified(mapping, "native")

    assert (check.ok, check.error_type) == (False, "AppliedMismatch"), check.detail
    assert "framework.name" in check.error
    # Refused, not removed: the mismatch is what the result has to carry.
    assert not report.applied.to_dict()["all_matched"]
    assert not report.all_ok


def test_axes_verified_names_every_mismatch_rather_than_the_first(config_mapping):
    """No probe config on this host has all its determined axes matched: the load
    dtype is fp32 off CUDA (`steps.dtype_for`) and the fused AdamW kernel is
    CUDA-only, while `configs/precision/` offers no fp32 and `configs/optim/` no
    unfused AdamW. So every CPU probe cell is refused here, and whether a pod cell
    passes is a pod question — 확인 안 함. What is asserted is that the refusal
    names all of them."""
    check, report = _axes_verified(json.loads(json.dumps(config_mapping)), "native")

    named = {axis.axis for axis in report.applied.mismatched()}
    assert named, "nothing mismatched, so this test would assert nothing"
    assert not check.ok
    for axis in named:
        assert axis in check.error


def test_axes_verified_does_not_grade_an_undetermined_axis(config_mapping):
    """The break for the tests above, and the boundary this check deliberately does
    not cross. A probe builds no dataloader, so those axes come back undetermined
    in every cell; refusing them here would paint every cell red over a question
    the probe never claimed to answer. `assert_matches` is where undetermined stops
    a run, for the purposes whose numbers get published."""
    from trainbench.applied import AppliedState, AxisState
    from trainbench.probe.steps import _refuse_mismatch

    state = AppliedState(
        (
            AxisState("framework.name", "native", "native"),
            AxisState("dataloader.backend", "torch", None, {"reason": "no dataloader was built"}),
        )
    )
    assert state.undetermined() and not state.mismatched()

    assert _refuse_mismatch(state, to_bench_config(config_mapping)) is None


def test_axes_verified_names_an_environment_bound_kernel_mismatch(monkeypatch):
    """`kernel=none` is unsatisfiable on qwen3_5 in any image that ships fla:
    transformers binds it while it imports the modelling module, which is why
    `axes.patch` refuses reported purposes outright. A probe is the run that
    exists to report it, so the mismatch stays and the message says which kind it
    is. The binding is stood in for because no image on this host has fla — that
    is a property of the image, and `_fla_binding` is the reader that decides it.
    """
    from trainbench import axes
    from trainbench.applied import AxisState
    from trainbench.probe.steps import _environment_bound

    monkeypatch.setattr(axes, "_fla_binding", lambda: (True, ""))
    config = to_bench_config(compose_probe("model=qwen3_5_0_8b"))

    note = _environment_bound(AxisState("kernel.name", "none", "fla"), config)

    assert "environment-bound" in note
    assert "fla" in note


@pytest.mark.parametrize(
    ("axis", "requested", "applied_value"),
    [
        # The kernel axis on an architecture nothing binds: a real mismatch that
        # happens to be on the one axis the classifier can read.
        ("kernel.name", "liger", "none"),
        # The other measured mismatch. axolotl keeps embed_tokens/lm_head in fp32
        # and peft keeps adapter weights there; neither is distinguishable from a
        # bf16 request answered in fp32 by anything readable here.
        ("precision.name", "bf16", "mixed(bf16,fp32)"),
    ],
)
def test_a_mismatch_this_cannot_classify_is_not_called_environment_bound(
    config_mapping, axis, requested, applied_value
):
    from trainbench.applied import AxisState
    from trainbench.probe.steps import _environment_bound

    config = to_bench_config(config_mapping)

    assert _environment_bound(AxisState(axis, requested, applied_value), config) == ""


def _fake_tevatron(monkeypatch, seen):
    """A miniature of tevatron dd063104 keeping the lines the shims are for.

    `DenseModel.load` reads `base_model.config.pad_token_id` directly rather than
    through `getattr` and then fills a `None` with 0
    (retriever/modeling/encoder.py:166-169). Both are reproduced verbatim: a stub
    using `getattr` would keep passing with the shim removed, which is the whole
    thing under test. The `config` kwarg is honoured the way
    `AutoModel.from_pretrained` honours it — a `PreTrainedConfig` instance is used
    as given and `AutoConfig` is not consulted (transformers 5.14.1
    models/auto/auto_factory.py:262, :324). `self.temperature = 1.0` reproduces
    `EncoderModel.__init__`'s default (encoder.py:38), which `apply_temperature`
    overrides.
    """

    class _Loaded(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.config = config
            self.temperature = 1.0

    class DenseModel:
        @staticmethod
        def load(
            model_name_or_path, pooling="cls", normalize=False, lora_name_or_path=None, **hf_kwargs
        ):
            from transformers import AutoConfig

            seen["hf_kwargs"] = hf_kwargs
            config = hf_kwargs.get("config")
            if config is None:
                config = AutoConfig.from_pretrained(model_name_or_path)
            base_model = _Loaded(config)
            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = 0
            return base_model

    root = types.ModuleType("tevatron")
    root.__path__ = []
    root.__version__ = "0.0-stub"
    modeling = types.ModuleType("tevatron.retriever.modeling")
    modeling.DenseModel = DenseModel
    for name, module in {
        "tevatron": root,
        "tevatron.retriever": types.ModuleType("tevatron.retriever"),
        "tevatron.retriever.modeling": modeling,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


COMPOSITE_CONFIGS = {
    "qwen3_vl": "transformers.models.qwen3_vl.configuration_qwen3_vl:Qwen3VLConfig",
    "qwen3_5": "transformers.models.qwen3_5.configuration_qwen3_5:Qwen3_5Config",
    "gemma4": "transformers.models.gemma4.configuration_gemma4:Gemma4Config",
}


@pytest.mark.parametrize("arch", sorted(COMPOSITE_CONFIGS))
def test_the_tevatron_probe_plants_pad_token_id_on_every_composite_config(
    config_mapping, monkeypatch, arch
):
    """All three tevatron cells of the second campaign died on
    `'<X>Config' object has no attribute 'pad_token_id'`. The real config classes
    are built here rather than stubbed: what the shim works around is transformers
    5.14.1 declaring the field only on the text sub-config, and a stub would be
    asserting its own layout."""
    import importlib as _importlib

    from transformers import AutoConfig

    module_name, class_name = COMPOSITE_CONFIGS[arch].split(":")
    hf_config = getattr(_importlib.import_module(module_name), class_name)()
    # The premise, asserted: with the field present at the top level there would be
    # nothing for the shim to plant and this test would pass on an empty subject.
    assert not hasattr(hf_config, "pad_token_id")
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: hf_config)

    seen: dict[str, object] = {}
    _fake_tevatron(monkeypatch, seen)
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "tevatron"
    config = to_bench_config(mapping)

    report = run_probe(config, get_device("cpu"))
    check = next(c for c in report.checks if c.name == "dense_model_load")

    assert check.ok, check.error
    assert seen["hf_kwargs"]["config"] is hf_config
    assert seen["hf_kwargs"]["revision"] == config.model.revision
    assert check.detail["pad_token_id_planted"] is True
    # What the text sub-config declares, planted verbatim — including `None`, which
    # upstream's own next line then fills with 0.
    assert check.detail["pad_token_id"] == getattr(
        hf_config.get_text_config(), "pad_token_id", "missing"
    )
    assert hf_config.pad_token_id == 0


def test_the_tevatron_shim_leaves_a_config_that_already_declares_it_alone():
    """Planting over a declared value would replace the checkpoint's own answer
    with the text sub-config's."""
    from trainbench.probe.tevatron import plant_pad_token_id

    hf_config = types.SimpleNamespace(pad_token_id=7)

    assert plant_pad_token_id(hf_config)["pad_token_id_planted"] is False
    assert hf_config.pad_token_id == 7


class _FakeEncoderModel(torch.nn.Module):
    """Stand-in for tevatron's `EncoderModel`: the same `forward(query=, passage=)`
    signature and the same eval-branch truthiness trap on an empty dict (dd06310
    encoder.py:52-61), with a real `nn.Embedding` so there is a parameter for
    `training_step_evidence` to count.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = torch.nn.Embedding(8, 4)
        self.temperature = 1.0

    def _encode(self, side):
        return self.embed(side["input_ids"])[:, -1]

    def forward(self, query=None, passage=None):
        q_reps = self._encode(query) if query else None
        p_reps = self._encode(passage) if passage else None
        if q_reps is None or p_reps is None:
            return types.SimpleNamespace(loss=None, q_reps=q_reps, p_reps=p_reps)
        scores = q_reps @ p_reps.transpose(0, 1) / self.temperature
        target = torch.arange(scores.size(0))
        loss = torch.nn.functional.cross_entropy(scores, target)
        return types.SimpleNamespace(loss=loss, q_reps=q_reps, p_reps=p_reps)


class _FakeTevatronProcessor:
    """`processor(text=..., return_tensors=..., padding=...)` -> a fixed batch,
    the same shape the sentence-transformers stub's `.tokenize` uses."""

    padding_side = "right"

    def __call__(self, text=None, return_tensors=None, padding=None):
        ids = torch.arange(len(text) * 3).reshape(len(text), 3) % 8
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _fake_tevatron_encoder_model(monkeypatch, seen, frozen=False):
    """Like `_fake_tevatron`, but `DenseModel.load` hands back something that can
    run a real `forward`/`backward`, for the checks past `dense_model_load`."""
    hf_config = types.SimpleNamespace(pad_token_id=0)

    class DenseModel:
        @staticmethod
        def load(
            model_name_or_path, pooling="cls", normalize=False, lora_name_or_path=None, **hf_kwargs
        ):
            seen["hf_kwargs"] = hf_kwargs
            model = _FakeEncoderModel(hf_kwargs.get("config", hf_config))
            if frozen:
                model.requires_grad_(False)
                # The same trick real unsloth's enable_input_require_grads plays:
                # requires_grad on the embedding *output*, not on any parameter.
                model.embed.register_forward_hook(
                    lambda module, args, output: output.requires_grad_(True)
                )
            seen["model"] = model
            return model

    root = types.ModuleType("tevatron")
    root.__path__ = []
    root.__version__ = "0.0-stub"
    modeling = types.ModuleType("tevatron.retriever.modeling")
    modeling.DenseModel = DenseModel
    for name, module in {
        "tevatron": root,
        "tevatron.retriever": types.ModuleType("tevatron.retriever"),
        "tevatron.retriever.modeling": modeling,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    from transformers import AutoConfig, AutoProcessor

    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **k: hf_config)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *a, **k: _FakeTevatronProcessor())


def _tevatron_checks(config_mapping, monkeypatch, frozen=False):
    seen: dict[str, object] = {}
    _fake_tevatron_encoder_model(monkeypatch, seen, frozen=frozen)
    mapping = json.loads(json.dumps(config_mapping))
    mapping["framework"]["name"] = "tevatron"
    report = run_probe(to_bench_config(mapping), get_device("cpu"))
    return {c.name: c for c in report.checks}, seen


def test_the_tevatron_backward_builds_query_and_passage_dicts_and_reads_the_loss_off_the_encoder(
    config_mapping, monkeypatch
):
    """`EncoderModel.forward` takes `query=`/`passage=` dicts and computes the loss
    itself (dd06310 encoder.py:52-87); the probe used to call
    `model(**batch, output_hidden_states=False)`, a call this signature has no
    `input_ids` keyword for, and which failed on all three models on real pods
    (2026-08-03) with `unexpected keyword argument 'input_ids'`."""
    checks, _ = _tevatron_checks(config_mapping, monkeypatch)
    check = checks["infonce_backward"]

    assert check.ok, check.error
    assert check.detail["trainable_params"] == check.detail["total_params"] == 1
    assert check.detail["params_with_grad"] == 1


def test_the_tevatron_backward_records_the_temperature_it_actually_ran_at(
    config_mapping, monkeypatch
):
    """`DenseModel.load` takes no `temperature` kwarg, so `EncoderModel.__init__`'s
    default of 1.0 (encoder.py:38) would otherwise silently outlive
    `config.loss.temperature`, scoring one tevatron cell at a temperature no other
    framework's cell used and leaving nothing in the record to say so."""
    checks, seen = _tevatron_checks(config_mapping, monkeypatch)
    config = to_bench_config(config_mapping)

    assert seen["model"].temperature == config.loss.temperature
    assert checks["dense_model_load"].detail["temperature"] == config.loss.temperature
    assert checks["infonce_backward"].detail["temperature"] == config.loss.temperature


def test_the_tevatron_backward_refuses_a_frozen_encoder_like_every_other_framework(
    config_mapping, monkeypatch
):
    """Reuses `steps.training_step_evidence`, so the guard that caught three frozen
    unsloth cells (2026-08-02, `params_with_grad=0, trainable_params=0`) is on this
    adapter too, even though tevatron computes its own loss and never calls
    `steps.infonce_backward`."""
    checks, _ = _tevatron_checks(config_mapping, monkeypatch, frozen=True)
    check = checks["infonce_backward"]

    assert (check.ok, check.error_type) == (False, "ValueError"), check.error
    assert "trained nothing" in check.error


def test_proportional_quota_preserves_composition_and_total():
    from scripts.prepare_data import proportional_quota

    counts = {"big": 900_000, "small": 100_000}
    quota = proportional_quota(counts, 1000)

    assert sum(quota.values()) == 1000
    # Composition is preserved, which is the whole point: a subset that over-weights
    # one config would make the measured sequence-length distribution wrong.
    assert quota["big"] > quota["small"] * 5
