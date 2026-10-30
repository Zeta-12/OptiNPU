"""P2 / P3: run the optimizer on bench models, then report AP on val2017.

Decisions use val2017 images disjoint from the reported ones (all bench models were trained on
train2017, where their AP is inflated; P3.1 still used train2017 [9670, 10000) and [0, 200)):
- accuracy guard: val2017 [4970, 5000); INT8 calibration: [4500, 4800) (as in L2 / M0);
- AP budget of approximate candidates (INT8, mixed precision) and of repairs: [2500, 3000), judged on 10
  folds of 50 images against the CPU FP32 original (loss within the budget with 95 % confidence; P3.2
  and the first P3.4 batch used [2500, 2700) and a point estimate).
The final AP is measured on val2017 [0, N), on which nothing was decided. Results go to
runs/p3/optimize_<model>_<precision>.json (the network on the NPU, only the final selection on the CPU),
or runs/p3/offload-<devices>/ with --offload (e.g. a DETR decoder on the iGPU, for comparison; batches
1 to 3 of P3.4 allowed gpu,cpu); models already done are skipped (resumable, so the bench can be run a
few models at a time).

    python -m benchmarks.p2.optimize_model yolo26-n:fp16 dfine-n:fp16 [--ap-budget 0.5] [--images 500]
"""

from __future__ import annotations

import argparse
import json

import openvino as ov

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from optinpu.data import coco_val_subset
from optinpu.optimizer import optimize
from optinpu.sandbox import run_evaluation
from optinpu.tasks import DetectionTask

OUT = ROOT / "runs" / "p3"


def run(name: str, precision: str, args) -> dict:
    entry = next(m for m in all_models() if m.name == name)
    model = ov.Core().read_model(entry.irs[precision])
    detector = isinstance(entry.task, DetectionTask)
    report = optimize(model, entry.task, coco_val_subset(30, offset=4970).image_paths(), "npu", None,
                      calibration=coco_val_subset(300, offset=4500).image_paths(),
                      evaluation=coco_val_subset(args.eval_images, offset=2500) if detector and args.eval_images else None,
                      ap_budget=args.ap_budget / 100, int8=not args.no_int8, offload=offload(args))
    print(report, flush=True)
    record = {"model": name, "precision": precision, "protocol": "val2017 selection", "offload": offload(args),
              "report": str(report),
              "applied": report.final.label(),
              "baseline_ms": report.baseline_ms, "final_ms": report.final_ms,
              "baseline_agreement": report.baseline_agreement, "final_agreement": report.final_agreement,
              "selection_ap": {"cpu": report.reference_ap, "baseline": report.baseline_ap, "final": report.final_ap},
              "trials": [vars(t) for t in report.trials], "recommendations": [vars(t) for t in report.recommendations]}
    if detector and args.images:
        subset = coco_val_subset(args.images)
        val = {}
        for label, parts in (("cpu", [(model, "cpu", None)]), ("baseline", [(model, "npu", None)]),
                             ("final", report.final.parts("npu"))):
            result, job = run_evaluation(parts, entry.task, subset)
            val[label] = result["AP"] if result else f"failed: {(job.error or '')[-100:]}"
        record["val_ap"] = val
        print(f"\nAP on val2017 [0, {args.images}): {val}", flush=True)
    return record


def offload(args) -> list[str]:
    return [d for d in args.offload.split(",") if d]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="model:precision of benchmarks.p0.models")
    parser.add_argument("--ap-budget", type=float, default=0.5, help="AP points an approximate rewrite may cost")
    parser.add_argument("--eval-images", type=int, default=500, help="val2017 images [2500, ...) for the AP budget")
    parser.add_argument("--images", type=int, default=500, help="val2017 images for the final AP")
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--offload", default="", help="devices for a heavy tail, e.g. gpu,cpu (default: none)")
    parser.add_argument("--force", action="store_true", help="redo models already done")
    args = parser.parse_args()
    out = OUT / f"offload-{'-'.join(offload(args))}" if offload(args) else OUT
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.models:
        name, precision = spec.split(":")
        path = out / f"optimize_{name}_{precision}.json"
        if path.exists() and not args.force:
            print(f"{spec}: already done ({path})", flush=True)
            continue
        print(f"===== {spec}", flush=True)
        path.write_text(json.dumps(run(name, precision, args), indent=1, default=str))


if __name__ == "__main__":
    main()
