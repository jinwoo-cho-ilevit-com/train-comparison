"""Phase 0 probes: can this framework load this model and take one training step.

A probe never raises. Every failure is recorded as a failed check with its
exception, because "this combination does not work" is a result we are trying to
produce, not an error to abort on.
"""

from trainbench.probe.registry import run_probe
from trainbench.probe.types import Check, ProbeReport

__all__ = ["Check", "ProbeReport", "run_probe"]
