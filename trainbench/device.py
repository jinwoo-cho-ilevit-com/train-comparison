"""Single source of device selection.

Inline `.cuda()` or `"cuda:0"` strings elsewhere in the codebase are forbidden:
they break the CPU fallback that lets GPU code paths be tested on a laptop.
"""

from __future__ import annotations

import torch


def get_device(override: str | None = None) -> torch.device:
    """Resolve the device for this run.

    `override` carries the `device=` config field. Setting it to `cpu` forces CPU
    regardless of available hardware, which is how the smoke tests exercise GPU
    code paths without a GPU.
    """
    if override:
        return torch.device(override)
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")
