"""Probe result types."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from trainbench.applied import AppliedState

# Truncated so one exploding traceback cannot dominate a result file.
MAX_TRACEBACK_CHARS = 4000


@dataclass
class Check:
    """One verifiable claim about a framework x model combination."""

    name: str
    ok: bool
    # Some checks exist to confirm a documented limitation, e.g. whether Unsloth's
    # encoder-only embedding path rejects a VLM checkpoint. Their failure is the
    # answer, so it must not make the whole cell read as broken.
    expected_failure: bool = False
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "expected_failure": self.expected_failure,
            "detail": self.detail,
            "error": self.error,
            "error_type": self.error_type,
            "traceback": self.traceback,
        }


@dataclass
class ProbeReport:
    framework: str
    model: str
    checks: list[Check] = field(default_factory=list)
    # What the constructed model actually ended up running. Set by adapters that
    # build a model; None means no model was built, not that the axes were fine.
    applied: AppliedState | None = None

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def run(
        self, name: str, fn: Callable[[], Any], expected_failure: bool = False
    ) -> tuple[bool, Any]:
        """Run one check, converting any exception into a recorded failure.

        Returns (ok, value) so the caller can decide whether later checks are
        still meaningful — a failed model load makes a forward pass moot.

        `expected_failure` marks a check that documents a known limitation. It is
        settable here so an adapter never has to hand-build a Check just to say so.
        """
        try:
            value = fn()
        except BaseException as exc:  # noqa: BLE001 - a probe must survive anything
            self.add(
                Check(
                    name=name,
                    ok=False,
                    expected_failure=expected_failure,
                    error=str(exc)[:1000],
                    error_type=type(exc).__name__,
                    traceback=traceback.format_exc()[-MAX_TRACEBACK_CHARS:],
                )
            )
            return False, None
        detail = value if isinstance(value, dict) else {}
        self.add(Check(name=name, ok=True, expected_failure=expected_failure, detail=detail))
        return True, value

    def skip(self, name: str, reason: str, expected_failure: bool = False) -> None:
        self.add(
            Check(
                name=name,
                ok=False,
                expected_failure=expected_failure,
                error=f"skipped: {reason}",
                error_type="Skipped",
            )
        )

    def add_version(self, module: Any) -> None:
        """Record the framework's own version. Each image ships a different stack,
        so this travels with the result rather than being assumed."""
        self.add(
            Check(
                name="framework_version",
                ok=True,
                detail={"version": getattr(module, "__version__", "unknown")},
            )
        )

    @property
    def all_ok(self) -> bool:
        return all(c.ok or c.expected_failure for c in self.checks)

    @property
    def unexpected_passes(self) -> list[str]:
        """Checks marked as documented limitations that have started to succeed.

        The support matrix says these cannot work. When one passes, the matrix is
        wrong and the run is the only place that knows — `all_ok` cannot say so,
        because an expected failure that passes is still not a failure.
        """
        return [c.name for c in self.checks if c.expected_failure and c.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "model": self.model,
            "all_ok": self.all_ok,
            "unexpected_passes": self.unexpected_passes,
            "applied": self.applied.to_dict() if self.applied is not None else None,
            "checks": [c.to_dict() for c in self.checks],
        }
