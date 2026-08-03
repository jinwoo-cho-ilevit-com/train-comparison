"""Run one framework x model probe and write the result.

Runs inside every framework image, so it takes a resolved config JSON rather than
composing with Hydra (Hydra's antlr4 pin is incompatible with axolotl). Generate
the JSON with scripts/compose_config.py.

    python scripts/verify_env.py --config resolved.json --out result.json
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from trainbench.config import load_bench_config
from trainbench.device import get_device
from trainbench.probe import Check, ProbeReport, run_probe
from trainbench.probe.types import MAX_TRACEBACK_CHARS
from trainbench.record import build_record, write_json
from trainbench.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="resolved config JSON")
    parser.add_argument("--out", required=True, type=Path, help="where to write the result")
    args = parser.parse_args(argv)

    config = load_bench_config(args.config)
    device = get_device(config.device)
    # Probes stay deterministic: they answer "does it run", and reproducing a
    # failure matters more than kernel selection here. `warn_only` keeps that from
    # inventing failures — under strict determinism an op with no deterministic
    # kernel raises, and the probe would record that as the framework refusing the
    # model rather than as our own seeding refusing the op.
    set_seed(config.train.seed, deterministic=True, warn_only=True)

    # `run_probe` catches everything inside each adapter's own `report.run()`
    # closures, but not what happens between them or after — an axolotl x
    # qwen3_5_0_8b cell (pod zql0z8hc4k8dlx, 2026-08-03) died there with no
    # result file and no traceback, costing the pod-hour and the diagnosis both.
    # This is the same net `report.run` already casts, one level up: whatever
    # escapes it still lands as a failed check `scripts/report.py` reads like
    # any other, instead of `docker/entrypoint.sh`'s generic "no result file"
    # fallback.
    #
    # The net starts here, not at the top of the function: a config this image
    # cannot parse, a device it cannot resolve or a seed it cannot set are all
    # above it and still leave no record. `bench.py --preflight` answers those
    # before the probe starts, which is why widening this would only move the
    # same failure to a worse place to read it.
    #
    # `Exception`, not `BaseException`: a `SystemExit` filed as a framework
    # failure would flatten a deliberate exit code to this function's own 1,
    # which is the laundering `run_with_secrets` exists to undo.
    try:
        report = run_probe(config, device)
        record = build_record(config, device, applied=report.applied, probe=report.to_dict())
        write_json(args.out, record)
    except Exception as exc:  # noqa: BLE001 - the last net under run_probe
        escaped = True
        report = ProbeReport(framework=config.framework.name, model=config.model.name)
        report.add(
            Check(
                name="probe_process",
                ok=False,
                error=str(exc)[:MAX_TRACEBACK_CHARS],
                error_type=type(exc).__name__,
                traceback=traceback.format_exc()[-MAX_TRACEBACK_CHARS:],
            )
        )
        record = build_record(config, device, applied=None, probe=report.to_dict())
        write_json(args.out, record)
    else:
        escaped = False

    failed = [c.name for c in report.checks if not c.ok]
    print(f"{report.framework} x {report.model}: {len(report.checks)} checks, {len(failed)} failed")
    for check in report.checks:
        mark = "OK  " if check.ok else "FAIL"
        extra = check.error_type or ""
        print(f"  {mark} {check.name} {extra}")
    print(f"wrote {args.out}")
    if escaped:
        # A genuine process-level failure, not a documented "this combination
        # refuses" — the pod log should say so even though the record itself
        # already carries the traceback.
        return 1
    # Otherwise always exit 0: a failing combination is a recorded result, not a
    # run error.
    return 0


if __name__ == "__main__":
    sys.exit(main())
