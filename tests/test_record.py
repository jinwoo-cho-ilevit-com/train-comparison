"""Which package versions a run record has to carry.

AGENTS.md: framework images bring their own stacks, so version is a confound
that must be visible in results. `_TRACKED_PACKAGES` is the list that makes it
visible; each entry here is tied to an axis this study measures, not to
hardware attribution — `kernels` decides whether an `attn=fa2` request is
served locally or substituted by a Hub kernel
(`transformers/modeling_flash_attention_utils.py::FLASH_ATTN_KERNEL_FALLBACK`),
`triton` is what `compile.mode` actually compiles through, and `fla-core` is
the distribution that ships `fla.ops`/`fla.modules` — the wrapper
`flash-linear-attention` only `==`-pins it (`trainbench/axes.py`'s
`FLA_OPS_DISTRIBUTION`/`FLA_DISTRIBUTIONS`).
"""

from __future__ import annotations

from trainbench import record


def test_the_axis_critical_packages_are_tracked():
    """Absent from `_TRACKED_PACKAGES` used to mean `attn`, `compile` and
    `kernel` could silently change what a run measured with no trace in the
    record. Each name here is checked individually so a later edit that drops
    one fails on the name that is missing, not on a set-difference."""
    for name in (
        "kernels",
        "triton",
        "fla-core",
        "flash-linear-attention",
        "flash-attn",
        "causal-conv1d",
    ):
        assert name in record._TRACKED_PACKAGES, name


def test_package_versions_does_not_raise_when_none_of_them_are_installed():
    """None of `kernels`/`triton`/`fla-core` install on this host — they need a
    CUDA toolchain or a Linux wheel this dev environment does not have — so this
    is the realistic case the function has to survive rather than a synthetic one."""
    versions = record.package_versions()
    assert isinstance(versions, dict)
    for name in ("kernels", "triton", "fla-core"):
        assert name not in versions or isinstance(versions[name], str)
