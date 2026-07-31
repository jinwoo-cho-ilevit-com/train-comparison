"""A probe must always produce a result, never an exception."""

from __future__ import annotations

import json

import pytest

from trainbench.config import to_bench_config
from trainbench.device import get_device
from trainbench.embedding import info_nce, last_token_pool
from trainbench.probe import run_probe
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


def test_last_token_pool_picks_last_attended_position():
    hidden = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]]])
    mask = torch.tensor([[1, 1, 0]])

    assert torch.equal(last_token_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))


def test_info_nce_is_lower_when_pairs_align():
    aligned = torch.eye(4)
    shuffled = torch.eye(4).flip(0)

    assert info_nce(aligned, aligned, 0.02) < info_nce(aligned, shuffled, 0.02)
