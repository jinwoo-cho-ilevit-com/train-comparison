"""Run one framework x model probe and write the result.

Runs inside every framework image, so it takes a resolved config JSON rather than
composing with Hydra (Hydra's antlr4 pin is incompatible with axolotl). Generate
the JSON with scripts/compose_config.py.

    python scripts/verify_env.py --config resolved.json --out result.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trainbench.config import load_bench_config
from trainbench.device import get_device
from trainbench.probe import run_probe
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

    report = run_probe(config, device)
    record = build_record(config, device, applied=report.applied, probe=report.to_dict())
    write_json(args.out, record)

    failed = [c.name for c in report.checks if not c.ok]
    print(f"{report.framework} x {report.model}: {len(report.checks)} checks, {len(failed)} failed")
    for check in report.checks:
        mark = "OK  " if check.ok else "FAIL"
        extra = check.error_type or ""
        print(f"  {mark} {check.name} {extra}")
    print(f"wrote {args.out}")
    # Always exit 0: a failing combination is a recorded result, not a run error.
    return 0


if __name__ == "__main__":
    sys.exit(main())
