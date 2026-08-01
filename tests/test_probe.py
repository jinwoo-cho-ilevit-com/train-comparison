"""A probe must always produce a result, never an exception.

Pooling and loss have their own module (tests/test_embedding.py); what is tested
here is the probe machinery around them.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from trainbench.config import to_bench_config
from trainbench.device import get_device
from trainbench.probe import registry, run_probe
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
    """Emits a fixed batch so the count under test is the one written here."""

    def __init__(self, input_ids, padding_side="right", pad_token_id=None):
        self._input_ids = input_ids
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<image>text"

    def __call__(self, text=None, images=None, return_tensors=None, padding=None):
        return {"input_ids": self._input_ids}


def _count(processor, model, tokens_per_image=None):
    return visual_token_count(processor, model, get_device("cpu"), "right", tokens_per_image)


def test_visual_token_count_reports_the_placeholder_count():
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6], [5, 7, 7, 7, 6]]))
    model = _Model(_Config(image_token_id=7))

    detail = _count(processor, model)

    assert detail["visual_tokens_per_image"] == 3
    assert detail["image_token_id_source"] == "config.image_token_id"
    assert detail["total_seq_len"] == 5


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
    a measurement of the batch shape."""
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6], [5, 7, 7, 7, 6]]), pad_token_id=7)
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="pad token id"):
        _count(processor, model)


def test_visual_token_count_refuses_per_sample_disagreement():
    """Every row carries the same image, so counts that differ mean the id matched
    something the rows do not share. Grading per_sample[0] alone accepted this."""
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6], [5, 7, 7, 6, 6]]))
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="disagree"):
        _count(processor, model)


def test_visual_token_count_refuses_a_count_the_model_spec_contradicts():
    """gemma4 declares a fixed 280 regardless of resolution. A different measurement
    is a disagreement to resolve, not a number to divide tokens/s by."""
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6]]))
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="tokens_per_image"):
        _count(processor, model, tokens_per_image=280)


def test_visual_token_count_accepts_the_declared_count():
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6]]))
    model = _Model(_Config(image_token_id=7))

    assert _count(processor, model, tokens_per_image=3)["declared_tokens_per_image"] == 3


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


def test_proportional_quota_preserves_composition_and_total():
    from scripts.prepare_data import proportional_quota

    counts = {"big": 900_000, "small": 100_000}
    quota = proportional_quota(counts, 1000)

    assert sum(quota.values()) == 1000
    # Composition is preserved, which is the whole point: a subset that over-weights
    # one config would make the measured sequence-length distribution wrong.
    assert quota["big"] > quota["small"] * 5
