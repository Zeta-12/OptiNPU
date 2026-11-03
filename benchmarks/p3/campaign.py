"""P3.5 campaign: the optimizer bench (benchmarks.p2.optimize_model) on the models still missing from the
paper, one after the other, unattended (a night).

- Windows is kept awake while it runs (the display may still turn off);
- latency on battery is throttled: before each model, it waits for mains power;
- each model gets its own log in runs/p3/campaign/, a failed model does not stop the others;
- it can be relaunched as is: models already done (their JSON in runs/p3/) are skipped;
- at the end, runs/p3/campaign/summary.md gathers the results.
--offload and --images go to every run; with --offload, logs and summary go to runs/p3/campaign/offload-<devices>/
(the results to runs/p3/offload-<devices>/).

    python -m benchmarks.p3.campaign                          # the models below
    python -m benchmarks.p3.campaign yolo26-n:fp16 ...        # or given ones
    python -m benchmarks.p3.campaign --offload gpu dfine-n:fp16 ...   # P5: the decoder may go to the iGPU
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from benchmarks.m0.baselines import ROOT
from benchmarks.p2.optimize_model import OUT, on_ac_power

MODELS = ("yolo26-n:fp16", "yolo26-s:fp16", "dfine-s:fp16", "rfdetr-n:fp16",
          "resnet50_cls224:fp32", "mobilenet_v2_cls224:fp32", "efficientnet_b0_cls224:fp32")
LOGS = OUT / "campaign"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def stamp() -> str:
    return datetime.now().strftime("%H:%M")


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while this process runs (until keep_awake(False) or exit)."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def wait_for_mains() -> None:
    warned = False
    while on_ac_power() is False:
        if not warned:
            print(f"{stamp()} on battery: waiting for mains power", flush=True)
            warned = True
        time.sleep(60)


def summary(models, out=OUT) -> str:
    """Markdown table of the results of `models` (from their JSON in `out`)."""
    lines = ["| Model | Before | After | Applied | Metric: CPU / before / after | On mains at the end |",
             "|---|---|---|---|---|---|"]
    fmt = lambda v: f"{100 * v:.1f}" if isinstance(v, float) else str(v)[:30]
    for spec in models:
        name, precision = spec.split(":")
        path = out / f"optimize_{name}_{precision}.json"
        if not path.exists():
            lines.append(f"| {spec} | not done | | | | |")
            continue
        r = json.loads(path.read_text())
        val = r.get("val_ap") or {}
        metric = " / ".join(fmt(val.get(k)) for k in ("cpu", "baseline", "final"))
        lines.append(f"| {spec} | {r['baseline_ms']:.2f} ms | {r['final_ms']:.2f} ms "
                     f"({r['baseline_ms'] / r['final_ms']:.2f}x) | {r['applied']} | {r.get('metric', 'AP')} {metric} | "
                     f"{r.get('ac_power_at_end')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="model:precision (default: the P3.5 list)")
    parser.add_argument("--offload", default="", help="devices for a heavy tail, e.g. gpu (benchmarks.p2.optimize_model)")
    parser.add_argument("--images", type=int, help="val2017 images [0, N) for the final AP (default: the bench's)")
    args = parser.parse_args()
    models = args.models or MODELS
    options = (["--offload", args.offload] if args.offload else []) + (["--images", str(args.images)] if args.images else [])
    devices = "-".join(d for d in args.offload.split(",") if d)
    logs, out = (LOGS / f"offload-{devices}", OUT / f"offload-{devices}") if devices else (LOGS, OUT)
    logs.mkdir(parents=True, exist_ok=True)
    print(f"{stamp()} campaign: {', '.join(models)} {' '.join(options)}", flush=True)
    keep_awake(True)
    try:
        for spec in models:
            wait_for_mains()
            name, precision = spec.split(":")
            log = logs / f"{name}_{precision}.log"
            print(f"{stamp()} {spec}: started (log: {log})", flush=True)
            t0 = time.time()
            with open(log, "a", encoding="utf-8") as stream:
                code = subprocess.call([sys.executable, "-u", "-m", "benchmarks.p2.optimize_model", spec, *options],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            status = "done" if code == 0 else f"FAILED (exit {code}, see the log)"
            print(f"{stamp()} {spec}: {status} in {(time.time() - t0) / 60:.0f} min", flush=True)
    finally:
        keep_awake(False)
    text = summary(models, out)
    (logs / "summary.md").write_text(text, encoding="utf-8")
    print(f"\n{text}", flush=True)


if __name__ == "__main__":
    main()
