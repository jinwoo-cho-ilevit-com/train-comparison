"""Device fallback and seeding must work without a GPU (convention 03/07)."""

from __future__ import annotations

import random

import numpy as np
import torch

from trainbench.device import get_device
from trainbench.seed import dataloader_generator, set_seed


def test_override_forces_cpu():
    assert get_device("cpu").type == "cpu"


def test_no_override_resolves_some_device():
    device = get_device()
    assert device.type in ("cpu", "cuda", "mps", "xpu")


def test_set_seed_makes_draws_reproducible():
    set_seed(1234, deterministic=True)
    first = (random.random(), np.random.rand(), torch.rand(3))

    set_seed(1234, deterministic=True)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_deterministic_flag_drives_torch_state():
    set_seed(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False

    set_seed(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is True


def test_warn_only_keeps_determinism_without_raising_on_unsupported_ops():
    """Probes seed with warn_only. Without it, an op that merely lacks a
    deterministic kernel raises, and the probe records that as the framework
    refusing the model — a false entry in the support matrix, made by our seeding."""
    set_seed(0, deterministic=True, warn_only=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()

    set_seed(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_dataloader_generator_is_seeded():
    a = torch.rand(4, generator=dataloader_generator(7))
    b = torch.rand(4, generator=dataloader_generator(7))
    assert torch.equal(a, b)
