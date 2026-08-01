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

    def __init__(self, input_ids):
        self._input_ids = input_ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<image>text"

    def __call__(self, text=None, images=None, return_tensors=None, padding=None):
        return {"input_ids": self._input_ids}


def test_visual_token_count_reports_the_placeholder_count():
    processor = _Processor(torch.tensor([[5, 7, 7, 7, 6], [5, 7, 7, 7, 6]]))
    model = _Model(_Config(image_token_id=7))

    detail = visual_token_count(processor, model, get_device("cpu"))

    assert detail["visual_tokens_per_image"] == 3
    assert detail["image_token_id_source"] == "config.image_token_id"
    assert detail["total_seq_len"] == 5


def test_visual_token_count_refuses_a_zero_count():
    """Zero means the id is wrong or the placeholders never got inserted. Reporting
    it as a measurement would rescale every tokens/s figure that divides by it."""
    processor = _Processor(torch.tensor([[5, 1, 2, 3, 6]]))
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="outside"):
        visual_token_count(processor, model, get_device("cpu"))


def test_visual_token_count_refuses_an_all_image_batch():
    """apply_chat_template always wraps the placeholders in role and text tokens, so
    matching every position means the id matched something else."""
    processor = _Processor(torch.tensor([[7, 7, 7, 7]]))
    model = _Model(_Config(image_token_id=7))

    with pytest.raises(ValueError, match="outside"):
        visual_token_count(processor, model, get_device("cpu"))


def test_ple_report_fails_when_nothing_matches():
    """A zero match used to be reported as ok. Roughly half of gemma-4-E2B lives in
    these tables, so it means `freeze.ple` would freeze nothing and say it did."""
    from trainbench.probe.native import _ple_report

    with pytest.raises(ValueError, match="per_layer"):
        _ple_report(torch.nn.Linear(2, 2))


def test_proportional_quota_preserves_composition_and_total():
    from scripts.prepare_data import proportional_quota

    counts = {"big": 900_000, "small": 100_000}
    quota = proportional_quota(counts, 1000)

    assert sum(quota.values()) == 1000
    # Composition is preserved, which is the whole point: a subset that over-weights
    # one config would make the measured sequence-length distribution wrong.
    assert quota["big"] > quota["small"] * 5
