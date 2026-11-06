"""P2 / P3: run the optimizer on bench models, then report AP on val2017 (top-1 on ImageNetV2).

Detectors: decisions use val2017 images disjoint from the reported ones (all bench models were trained
on train2017, where their AP is inflated; P3.1 still used train2017 [9670, 10000) and [0, 200)):
- accuracy guard: val2017 [4970, 5000); INT8 calibration: [4500, 4800) (as in L2 / M0);
- AP budget, which decides what is exact (the original, repairs) and what approximations may cost:
  [2500, 3500), judged on 20 folds of 50 images against the CPU FP32 original (loss within the budget
  with 95 % confidence). P3.2 and the first P3.4 batch used [2500, 2700) and a point estimate; P3.4 and
  the first P3.5 runs used [2500, 3000) in 10 folds, which left a ~0.4-point margin.
The final AP is measured on val2017 [0, N), on which nothing was decided: N = 2500 since P5 (500 before),
and [3500, 4500) stays free, e.g. to confirm a mixed-precision choice on images it was not chosen on.
Classifiers (ImageNet weights): slices of the fixed shuffle of ImageNetV2 (optinpu.data.imagenetv2_subset),
none of which the models were trained on: guard [7300, 7600) (top-1 agreement), calibration
[7000, 7300), top-1 budget [5000, 7000) in 10 folds of 200, report [0, 5000). Results go to
runs/p3/optimize_<model>_<precision>.json (the network on the NPU, only the final selection on the CPU),
or runs/p3/offload-<devices>/ with --offload (e.g. a DETR decoder on the iGPU, for comparison; batches
1 to 3 of P3.4 allowed gpu,cpu), the optimized deployment and its report to deploy/<model>_<precision>/
next to it; models already done are skipped (resumable, so the bench can be run a few models at a time).
P6 (review): --device gpu runs the same optimizer for the iGPU (a fair reference for the NPU: same rewrites,
INT8 and mixed precision); --confirm re-tests every accepted approximation once on held-out images
(val2017 [3500, 4500) in 20 folds; ImageNetV2 [7600, 9600) in 10 folds); --out sets the results folder.

    python -m benchmarks.p2.optimize_model yolo26-n:fp16 dfine-n:fp16 [--ap-budget 0.5] [--images 2500]
    python -m benchmarks.p2.optimize_model yolo26-n:fp16 --device gpu --out runs/p6/igpu
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import openvino as ov

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from optinpu.data import coco_val_subset, imagenetv2_subset
from optinpu.optimizer import optimize
from optinpu.sandbox import run_evaluation
from optinpu.tasks import ClassificationTask

OUT = ROOT / "runs" / "p3"


def on_ac_power() -> bool | None:
    """True on mains power, False on battery, None when unknown (latency on battery is throttled: the
    first P3.5 run started unplugged, 48.4 ms for RT-DETRv2 instead of 47.5)."""
    if sys.platform != "win32":
        return None

    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]

    status = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return {0: False, 1: True}.get(status.ac)


def splits(task, args) -> tuple[list, list, object, int, object, str]:
    """(guard images, calibration images, evaluation set, its folds, report set, report set name)."""
    if isinstance(task, ClassificationTask):
        return (imagenetv2_subset(300, 7300).image_paths(), imagenetv2_subset(300, 7000).image_paths(),
                imagenetv2_subset(2000, 5000) if args.eval_images else None, 10,
                imagenetv2_subset(5000) if args.images else None, "ImageNetV2 [0, 5000)")
    return (coco_val_subset(30, offset=4970).image_paths(), coco_val_subset(300, offset=4500).image_paths(),
            coco_val_subset(args.eval_images, offset=2500) if args.eval_images else None,
            max(2, args.eval_images // 50), coco_val_subset(args.images) if args.images else None,
            f"val2017 [0, {args.images})")


def confirmation_set(task) -> tuple[object, int]:
    """Held-out images for the confirmation of approximations (--confirm), disjoint from every other split."""
    if isinstance(task, ClassificationTask):
        return imagenetv2_subset(2000, 7600), 10
    return coco_val_subset(1000, offset=3500), 20


def run(name: str, precision: str, args, out) -> dict:
    entry = next(m for m in all_models() if m.name == name)
    model = ov.Core().read_model(entry.irs[precision])
    guard, calibration, evaluation, folds, report_set, report_name = splits(entry.task, args)
    confirmation, confirmation_folds = confirmation_set(entry.task) if args.confirm else (None, None)
    device = args.device
    report = optimize(model, entry.task, guard, device, None, calibration=calibration, evaluation=evaluation,
                      ap_budget=args.ap_budget / 100, folds=folds, int8=not args.no_int8, offload=offload(args),
                      confirmation=confirmation, confirmation_folds=confirmation_folds)
    print(report, flush=True)
    report.save(out / "deploy" / f"{name}_{precision}", f"{name} ({precision}) on the {device.upper()}")
    protocol = "ImageNetV2 slices" if isinstance(entry.task, ClassificationTask) else "val2017 selection"
    record = {"model": name, "precision": precision, "device": device, "confirm": args.confirm,
              "protocol": protocol, "metric": report.metric, "report_set": report_name,
              "offload": offload(args), "report": str(report),
              "applied": report.final.label(),
              "baseline_ms": report.baseline_ms, "final_ms": report.final_ms,
              "baseline_agreement": report.baseline_agreement, "final_agreement": report.final_agreement,
              "selection_ap": {"cpu": report.reference_ap, "baseline": report.baseline_ap, "final": report.final_ap},
              "trials": [vars(t) for t in report.trials], "recommendations": [vars(t) for t in report.recommendations]}
    if report_set is not None:
        val = {}
        for label, parts in (("cpu", [(model, "cpu", None)]), ("baseline", [(model, device, None)]),
                             ("final", report.final.parts(device))):
            result, job = run_evaluation(parts, entry.task, report_set)
            val[label] = result["score"] if result else f"failed: {(job.error or '')[-100:]}"
        record["val_ap"] = val
        print(f"\n{report.metric} on {report_name}: {val}", flush=True)
    record["ac_power_at_end"] = on_ac_power()
    return record


def offload(args) -> list[str]:
    return [d for d in args.offload.split(",") if d]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="model:precision of benchmarks.p0.models")
    parser.add_argument("--ap-budget", type=float, default=0.5, help="AP points an approximate rewrite may cost")
    parser.add_argument("--eval-images", type=int, default=1000,
                        help="val2017 images [2500, ...) for the AP budget, in folds of 50 (0: no AP; classifiers: on/off)")
    parser.add_argument("--images", type=int, default=2500, help="val2017 images [0, N) for the final AP")
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--offload", default="", help="devices for a heavy tail, e.g. gpu,cpu (default: none)")
    parser.add_argument("--force", action="store_true", help="redo models already done")
    parser.add_argument("--allow-battery", action="store_true", help="measure even on battery power")
    parser.add_argument("--device", default="npu", choices=("npu", "gpu"), help="device to optimize for")
    parser.add_argument("--confirm", action="store_true", help="confirm accepted approximations on held-out images")
    parser.add_argument("--out", type=Path, help="results folder (default: runs/p3, or runs/p3/offload-<devices>)")
    args = parser.parse_args()
    out = args.out or (OUT / f"offload-{'-'.join(offload(args))}" if offload(args) else OUT)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.models:
        name, precision = spec.split(":")
        path = out / f"optimize_{name}_{precision}.json"
        if path.exists() and not args.force:
            print(f"{spec}: already done ({path})", flush=True)
            continue
        if on_ac_power() is False and not args.allow_battery:
            sys.exit(f"{spec}: on battery power, latencies would be throttled; plug the PC in (or --allow-battery)")
        print(f"===== {spec}", flush=True)
        path.write_text(json.dumps(run(name, precision, args, out), indent=1, default=str))


if __name__ == "__main__":
    main()
