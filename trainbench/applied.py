"""Requested optimisation vs what the model actually ended up running.

The failure this exists to prevent has already happened once in this project: with
`fla`/`causal-conv1d` absent, transformers silently falls back to a slow torch
implementation for Qwen3.5's Gated DeltaNet layers and says so in a single log
line — no exception, no return value (docs/support-matrix.md). The same shape of
failure applies to every axis: FlashAttention falls back to sdpa when unbuilt,
torch.compile falls back to eager on a graph break, fp8 recipes no-op on
unsupported hardware. Each produces a plausible number under a wrong label, and
the result JSON alone cannot tell them apart afterwards.

Axes that have no capture probe yet report `applied=None` (undetermined), and an
undetermined axis fails a timing run exactly like a mismatched one. That is the
fail-safe direction: an axis is only ever certified once someone writes a probe
that actually looks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from trainbench.config_schema import BenchConfig

# Purposes whose numbers get reported. A probe or profile run may proceed with
# undetermined axes; a timing run may not.
ENFORCED_PURPOSES = ("timing", "quality")


@dataclass(frozen=True)
class AxisState:
    """One axis: what was asked for, and what is actually in effect."""

    axis: str
    requested: str
    applied: str | None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def determined(self) -> bool:
        return self.applied is not None

    @property
    def matches(self) -> bool:
        return self.determined and self.applied == self.requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "requested": self.requested,
            "applied": self.applied,
            "determined": self.determined,
            "matches": self.matches,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AppliedState:
    axes: tuple[AxisState, ...]

    def mismatched(self) -> list[AxisState]:
        return [a for a in self.axes if a.determined and not a.matches]

    def undetermined(self) -> list[AxisState]:
        return [a for a in self.axes if not a.determined]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": [a.to_dict() for a in self.axes],
            "all_determined": not self.undetermined(),
            "all_matched": not self.mismatched(),
        }


class AppliedMismatch(RuntimeError):
    """A reportable run was about to measure something other than what it claims."""


# --- per-axis capture probes -------------------------------------------------
# A probe returns the applied value, or None when it cannot determine it. Adding
# an axis here is what makes that axis measurable; until then it blocks timing
# runs by design.

CaptureFn = Callable[[Any, BenchConfig], tuple[str | None, dict[str, Any]]]


def _capture_attn(model: Any, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """transformers records the implementation it settled on, which is not
    necessarily the one that was asked for."""
    applied = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if applied is None:
        return None, {"reason": "model.config._attn_implementation is absent"}
    return str(applied), {"requested_impl": config.attn.impl}


_CAPTURES: dict[str, CaptureFn] = {
    "attn": _capture_attn,
}

# What each axis asked for, read off the resolved config.
_REQUESTED: dict[str, Callable[[BenchConfig], str]] = {
    "attn": lambda c: c.attn.impl,
    "kernel": lambda c: c.kernel.name,
    "precision": lambda c: c.precision.name,
    "compile": lambda c: c.compile.mode,
    "optim": lambda c: c.optim.name,
    "freeze": lambda c: f"vision={c.freeze.vision_tower},ple={c.freeze.ple}",
    "dataloader": lambda c: f"{c.dataloader.backend},packing={c.dataloader.packing}",
    "parallel": lambda c: c.parallel.strategy,
}


def capture(model: Any, config: BenchConfig) -> AppliedState:
    """Read back the state of every axis after the model is fully constructed."""
    states = []
    for axis, requested_of in _REQUESTED.items():
        requested = requested_of(config)
        probe = _CAPTURES.get(axis)
        if probe is None:
            states.append(
                AxisState(axis, requested, None, {"reason": "no capture probe implemented"})
            )
            continue
        try:
            applied, detail = probe(model, config)
        except BaseException as exc:  # noqa: BLE001 - an unreadable axis is undetermined
            applied, detail = None, {"reason": f"{type(exc).__name__}: {exc}"[:300]}
        states.append(AxisState(axis, requested, applied, detail))
    return AppliedState(tuple(states))


def assert_matches(state: AppliedState, purpose: str) -> None:
    """Refuse to let a reportable run proceed on unverified or wrong settings.

    Undetermined is treated exactly like mismatched: "we could not check" must not
    read as "it is fine", or the whole mechanism becomes decorative.
    """
    if purpose not in ENFORCED_PURPOSES:
        return

    problems = []
    for axis in state.mismatched():
        problems.append(f"{axis.axis}: requested {axis.requested!r}, applied {axis.applied!r}")
    for axis in state.undetermined():
        reason = axis.detail.get("reason", "unknown")
        problems.append(f"{axis.axis}: requested {axis.requested!r}, undetermined ({reason})")

    if problems:
        raise AppliedMismatch(
            f"purpose={purpose} requires every axis to be verified as applied. "
            + "; ".join(problems)
        )
