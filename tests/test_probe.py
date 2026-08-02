"""A probe must always produce a result, never an exception.

Pooling and loss have their own module (tests/test_embedding.py); what is tested
here is the probe machinery around them.
"""

from __future__ import annotations

import ast
import json
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

import torch  # isort: skip

FRAMEWORKS = ["unsloth", "ms_swift", "sentence_transformers", "tevatron", "axolotl"]


@pytest.fixture
def config_mapping(tmp_path):
    from hydra import compose, initialize_config_dir

    from trainbench.compose import resolve

    from .conftest import CONFIG_DIR

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["run=probe", "device=cpu"])
        return resolve(cfg)[1]


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


def test_padding_side_alignment_fails_loudly_when_the_checkpoint_disagrees():
    """`config.model.padding_side` used to be a claim nothing checked: with the
    processor padding the other way, both pooling branches returned a PAD
    embedding without an exception or a warning."""
    from trainbench.probe.steps import padding_side_alignment

    processor = _Processor(torch.tensor([[1]]), padding_side="left")

    with pytest.raises(ValueError, match="model-spec"):
        padding_side_alignment(processor, "right")
    # Forced anyway: the check is what makes the disagreement loud, and the checks
    # that run after it still have to pool a real token.
    assert processor.padding_side == "right"


def test_padding_side_alignment_passes_when_they_agree():
    from trainbench.probe.steps import padding_side_alignment

    processor = _Processor(torch.tensor([[1]]), padding_side="left")
    processor.tokenizer = _Tokenizer("left")

    detail = padding_side_alignment(processor, "left")

    assert detail["disagreed"] == []
    assert detail["declared_before"] == {"tokenizer": "left", "processor": "left"}


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


def test_proportional_quota_preserves_composition_and_total():
    from scripts.prepare_data import proportional_quota

    counts = {"big": 900_000, "small": 100_000}
    quota = proportional_quota(counts, 1000)

    assert sum(quota.values()) == 1000
    # Composition is preserved, which is the whole point: a subset that over-weights
    # one config would make the measured sequence-length distribution wrong.
    assert quota["big"] > quota["small"] * 5
