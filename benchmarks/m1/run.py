"""Run M1 suites under the memory / time watchdog, restarting a suite after a crash.

    python -m benchmarks.m1.run [S1 S2 ...] [--max-gb 12]

A crashed job (killed by the watchdog, NPU device lost, error) leaves its current block marked
"started" in results.jsonl; the restarted job skips it, so a bad block cannot stall a suite.
"""

from __future__ import annotations

import argparse
import sys

from benchmarks.m1.blocks import SUITES
from benchmarks.m1.job import measured
from benchmarks.watchdog import run_watched

SUITE_TIMEOUT = 3 * 3600


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="*", default=[s for s in SUITES if s.startswith("S")])
    parser.add_argument("--max-gb", type=float, default=12.0)
    parser.add_argument("--max-restarts", type=int, default=20)
    args = parser.parse_args(argv)

    for suite in args.suites:
        names = [b.name for b in SUITES[suite]()]
        for attempt in range(args.max_restarts + 1):
            todo = [n for n in names if n not in measured()]
            if not todo:
                break
            print(f"== {suite}: {len(todo)} blocks to measure (attempt {attempt + 1})", flush=True)
            cmd = [sys.executable, "-u", "-W", "ignore", "-m", "benchmarks.m1.job", suite]
            w = run_watched(cmd, SUITE_TIMEOUT, args.max_gb, echo=True)
            if w.status or w.returncode:
                print(f"== {suite}: job ended with {w.status or f'exit code {w.returncode}'} "
                      f"(peak {w.peak_gb} GB, on AC: {w.on_ac}); restarting", flush=True)
        print(f"== {suite}: done", flush=True)


if __name__ == "__main__":
    main()
