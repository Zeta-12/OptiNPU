"""P6 campaign: the experiments asked by the review, unattended, in order of priority (PLAN.md, P6).

Night 1 (about 9 h):
  premise   latency vs weight / input values with CIs, real images, INT8 zeros, activation sparsity (~40 min)
  options   NPU_TURBO, THROUGHPUT hint, QDQ optimization, two inferences in flight (~40 min)
  igpu      the same optimizer for the iGPU, on the models where the NPU wins (~3 h)
  qwac      NNCF's accuracy-aware quantization as a baseline for mixed precision (~5 h)
Night 2 (about 14 h; resumable, it may go on the next day):
  confirm   the optimizer again, every accepted approximation confirmed on held-out images (~5 h)
  igpu-confirm  the same optimizer for the iGPU, with the confirmation: the DETRs, YOLO26-N, MobileNetV2 (~6 h)
  energy    energy again: 20 s windows, the optimized iGPU deployments, NPU_TURBO (~2 h)
  final     the latencies of the paper's tables again, paired, with 95 % CIs (~30 min)
  profile-cost  wall time of one measured profiler run vs the vendor counters and the additive model (~1 h)
Every step is resumable (what is done is skipped), waits for mains power before measuring, and keeps
Windows awake; a failed step does not stop the next ones. Logs: runs/p6/logs/<step>.log.

    python -m benchmarks.p6.campaign            # night 1
    python -m benchmarks.p6.campaign --night 2
    python -m benchmarks.p6.campaign premise qwac   # given steps
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from benchmarks.m0.baselines import ROOT
from benchmarks.p3.campaign import keep_awake, stamp

LOGS = ROOT / "runs" / "p6" / "logs"
IGPU = ["--device", "gpu", "--out", str(ROOT / "runs" / "p6" / "igpu")]
STEPS = {
    "premise": ["-m", "benchmarks.p6.premise"],
    "options": ["-m", "benchmarks.p6.options"],
    "igpu": ["-m", "benchmarks.p3.campaign", *IGPU, "yolo26-n:fp16", "yolo26-s:fp16", "resnet50_cls224:fp32",
             "mobilenet_v2_cls224:fp32", "efficientnet_b0_cls224:fp32"],
    "qwac": ["-m", "benchmarks.p6.qwac"],
    "confirm": ["-m", "benchmarks.p3.campaign", "--confirm", "--out", str(ROOT / "runs" / "p6" / "confirm"),
                "dfine-s:fp16", "rtdetrv2-r18:fp16", "rfdetr-n:fp16", "resnet50_cls224:fp32", "mobilenet_v2_cls224:fp32"],
    # with the held-out confirmation, as the NPU runs of "confirm": the five DETRs, and the two models whose
    # iGPU run of night 1 accepted an approximation (YOLO26-N's mix lost 0.64 AP on the report images)
    "igpu-confirm": ["-m", "benchmarks.p3.campaign", "--device", "gpu", "--confirm", "--out",
                     str(ROOT / "runs" / "p6" / "igpu-confirm"), "yolo26-n:fp16", "mobilenet_v2_cls224:fp32",
                     "dfine-n:fp16", "dfine-s:fp16", "deimv2-n:fp16", "rtdetrv2-r18:fp16", "rfdetr-n:fp16"],
    "energy": ["-m", "benchmarks.p5.energy", "--p6"],
    "final": ["-m", "benchmarks.p6.final"],
    "profile-cost": ["-m", "benchmarks.p6.profile_cost"],
}
NIGHTS = {1: ["premise", "options", "igpu", "qwac"], 2: ["confirm", "igpu-confirm", "energy", "final", "profile-cost"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", nargs="*", help=f"steps to run, among {', '.join(STEPS)} (default: the night's)")
    parser.add_argument("--night", type=int, default=1, choices=sorted(NIGHTS))
    args = parser.parse_args()
    unknown = [s for s in args.steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; known: {', '.join(STEPS)}")
    steps = args.steps or NIGHTS[args.night]
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"{stamp()} P6 campaign: {', '.join(steps)}", flush=True)
    keep_awake(True)
    try:
        for step in steps:
            log = LOGS / f"{step}.log"
            print(f"{stamp()} {step}: started (log: {log})", flush=True)
            t0 = time.time()
            with open(log, "a", encoding="utf-8") as stream:
                code = subprocess.call([sys.executable, "-u", *STEPS[step]], cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            status = "done" if code == 0 else f"FAILED (exit {code}, see the log)"
            print(f"{stamp()} {step}: {status} in {(time.time() - t0) / 60:.0f} min", flush=True)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
