"""Framework -> probe dispatch.

Probe modules are imported lazily: each one only exists inside its own image, and
importing unsloth from the axolotl image would fail for reasons unrelated to the
question being asked.
"""

from __future__ import annotations

import importlib
import traceback

import torch

from trainbench.config_schema import BenchConfig
from trainbench.probe.types import MAX_TRACEBACK_CHARS, Check, ProbeReport

_MODULES = {
    "native": "trainbench.probe.native",
    "unsloth": "trainbench.probe.unsloth",
    "ms_swift": "trainbench.probe.ms_swift",
    "sentence_transformers": "trainbench.probe.sentence_transformers",
    "tevatron": "trainbench.probe.tevatron",
    "axolotl": "trainbench.probe.axolotl",
}


def run_probe(config: BenchConfig, device: torch.device) -> ProbeReport:
    framework = config.framework.name
    module_path = _MODULES[framework]
    fallback = ProbeReport(framework=framework, model=config.model.name)
    try:
        module = importlib.import_module(module_path)
        return module.run(config, device)
    except BaseException as exc:  # noqa: BLE001 - any failure here is itself a result
        # Covers both an unimportable probe module and a framework that is absent
        # or explodes on import inside run(). A probe must never take the process
        # down: "this combination does not work" is the output we came for.
        fallback.add(
            Check(
                name="probe_import",
                ok=False,
                error=str(exc)[:1000],
                error_type=type(exc).__name__,
                traceback=traceback.format_exc()[-MAX_TRACEBACK_CHARS:],
            )
        )
        return fallback
