"""Single seeding helper. Every run seeds through `set_seed`."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool) -> None:
    """Seed every RNG.

    `deterministic` is False for measurement runs. Deterministic algorithms disable
    cuDNN autotuning and constrain kernel selection, which is part of what this
    project measures — leaving it on would distort every number. Tests and CPU
    smoke runs keep it True. The measured cost of the switch is recorded in
    docs/methodology.md, which is the evidence convention 07 requires.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        # Deterministic cuBLAS refuses to run without this; it is only read when the
        # CUDA context is created, so setting it later in a process is a no-op.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - DataLoader passes the id
    """DataLoader `worker_init_fn`. Reseeds numpy/random off torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling, so batch order is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
