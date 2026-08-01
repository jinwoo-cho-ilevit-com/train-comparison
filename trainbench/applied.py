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

The set of axes is derived from the schema, not listed here. A hand-written list
fails open — an axis missing from it is not "undetermined", it does not exist,
and deleting a line would disable a check with nothing to notice. Marking a field
`axis=True` in trainbench/config_schema.py is what puts it under this guarantee.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_args

from trainbench import axes
from trainbench.config_schema import BenchConfig, RunConfig, axis_knobs

# Purposes whose numbers get reported. A probe or profile run may proceed with
# undetermined axes; a timing run may not.
ENFORCED_PURPOSES = ("timing", "quality")
# Taken from the schema so a purpose can never be added without deciding whether
# it is enforced.
KNOWN_PURPOSES = get_args(RunConfig.model_fields["purpose"].annotation)


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

    def missing(self) -> list[str]:
        """Axis knobs the schema declares that this state says nothing about."""
        return sorted(set(axis_knobs()) - {a.axis for a in self.axes})

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": [a.to_dict() for a in self.axes],
            "all_determined": not self.undetermined(),
            "all_matched": not self.mismatched(),
            "missing": self.missing(),
        }


class AppliedMismatch(RuntimeError):
    """A reportable run was about to measure something other than what it claims."""


# --- per-axis capture probes -------------------------------------------------
# A probe returns the applied value, or None when it cannot determine it. Adding
# an axis here is what makes that axis measurable; until then it blocks timing
# runs by design. Keys are the dotted knob names from the schema.

CaptureFn = Callable[[Any, BenchConfig], tuple[str | None, dict[str, Any]]]


def _attn_per_module(model: Any) -> dict[str, str]:
    """Every place transformers records an attention implementation.

    Reading only `model.config._attn_implementation` is not enough. In
    `set_attn_implementation`, transformers updates the top-level config first and
    then walks submodules; any submodule that does not follow the AttentionInterface
    approach gets a log warning and keeps its previous implementation, and a
    submodule that cannot support the request has it downgraded individually. Vision
    towers are the common case. All three models here are multimodal, so a top-level
    'flash_attention_2' over an sdpa vision tower is the realistic outcome, and it
    would read as a clean match.
    """
    found: dict[str, str] = {}
    root = getattr(model, "config", None)
    top = getattr(root, "_attn_implementation", None)
    if top is not None:
        found["model"] = str(top)

    # Any submodule carrying its own config, rather than PreTrainedModel
    # instances specifically: what matters is that something in the tree records
    # a second answer, not what class recorded it.
    for name, module in getattr(model, "named_modules", lambda: [])():
        if module is model:
            continue
        impl = getattr(getattr(module, "config", None), "_attn_implementation", None)
        if impl is not None:
            found[name or type(module).__name__] = str(impl)

    for key in getattr(root, "sub_configs", ()) or ():
        sub = getattr(root, key, None)
        impl = getattr(sub, "_attn_implementation", None)
        if impl is not None:
            found.setdefault(f"config.{key}", str(impl))
    return found


def _capture_attn(model: Any, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    per_module = _attn_per_module(model)
    if not per_module:
        return None, {"reason": "no _attn_implementation found on the model or its subconfigs"}
    distinct = sorted(set(per_module.values()))
    if len(distinct) > 1:
        # Deliberately not equal to any requested value, so a partial application
        # is a mismatch rather than a match on the majority.
        return "mixed(" + ",".join(distinct) + ")", {"per_module": per_module}
    return distinct[0], {"per_module": per_module}


def _capture_freeze_ple(model: Any, config: BenchConfig) -> tuple[str | None, dict[str, Any]]:
    """Whether the per-layer embeddings are actually frozen.

    Counting matches is half the check. `_ple_report` used to report ok=True on a
    zero match, so a renamed parameter upstream would have read as a successful
    freeze of nothing — 2.39B parameters, 46.8% of gemma-4-E2B, quietly still
    training. Zero matches on gemma4 is undetermined, not False.
    """
    params = axes.ple_parameters(model)
    if not params and config.model.arch == "gemma4":
        return None, {
            "reason": f"no parameter name contains {axes.PLE_PARAM_MARKER!r} on a gemma4 model",
            "matched": 0,
        }
    frozen = [n for n, p in params if not p.requires_grad]
    detail = {"matched": len(params), "frozen": len(frozen)}
    if not params:
        # No per-layer embeddings exist, so none are frozen. True by absence, and
        # the schema already refuses freeze.ple on a non-gemma4 model.
        return "False", detail
    if len(frozen) == len(params):
        return "True", detail
    if not frozen:
        return "False", detail
    unfrozen = sorted({n for n, _ in params} - set(frozen))
    return "partial", {**detail, "unfrozen_sample": unfrozen[:5]}


_CAPTURES: dict[str, CaptureFn] = {
    "attn.name": _capture_attn,
    "freeze.ple": _capture_freeze_ple,
}

# Where the applied value is expressed in different words from the config value.
# transformers reports 'flash_attention_3', never 'fa3', so the request has to be
# translated before the two are compared.
_REQUESTED_OVERRIDES: dict[str, Callable[[BenchConfig], Any]] = {
    "attn.name": lambda config: config.attn.impl,
}


def capture(model: Any, config: BenchConfig) -> AppliedState:
    """Read back the state of every axis after the model is fully constructed.

    Never raises: an axis that cannot be read is undetermined, which blocks a
    reportable run just as a mismatch does.
    """
    states = []
    for axis, default_reader in axis_knobs().items():
        read_requested = _REQUESTED_OVERRIDES.get(axis, default_reader)
        try:
            requested = str(read_requested(config))
        except BaseException as exc:  # noqa: BLE001 - an unreadable request is still an axis
            reason = f"{type(exc).__name__}: {exc}"[:300]
            states.append(AxisState(axis, "unreadable", None, {"reason": reason}))
            continue
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


def assert_matches(state: AppliedState, config: BenchConfig) -> None:
    """Refuse to let a reportable run proceed on unverified or wrong settings.

    Undetermined is treated exactly like mismatched: "we could not check" must not
    read as "it is fine", or the whole mechanism becomes decorative. An empty or
    partial state is the same failure one level up — it means capture never ran
    for some axis, not that the axis was fine.

    Takes the config rather than a purpose string so the value cannot be a typo:
    `"Timing"` would otherwise pass every check silently.
    """
    purpose = config.run.purpose
    if purpose not in KNOWN_PURPOSES:
        raise ValueError(f"unknown run purpose {purpose!r}; expected one of {KNOWN_PURPOSES}")
    if purpose not in ENFORCED_PURPOSES:
        return

    problems = [f"{axis}: never captured" for axis in state.missing()]
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
