"""P2, first exact rewrite: run the final selection (TopK and everything after it) on the CPU.

P1 found that the 16-op selection tail of YOLO26-N takes ~72 % of its NPU latency. For each model:
NPU latency (2 tiles, 1 tile) with and without the tail moved to the CPU (measured side by side), then
detection agreement with CPU FP32 and AP on val2017 [0, N) of each placement.

    python -m benchmarks.p2.selection_tail [yolo26-n:fp16 ...] [--images 500]
"""

from __future__ import annotations

import argparse
import json

import openvino as ov

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from optinpu.data import coco_val_subset
from optinpu.evaluate import evaluate_coco
from optinpu.graph import split_from
from optinpu.profiler import measure
from optinpu.tasks import Deployed

SELECTION = ("TopK", "NonMaxSuppression", "MulticlassNms", "MatrixNms")
ONE_TILE = {"NPU_TILES": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["yolo26-n:fp16", "yolo26-s:fp16", "yolo26-n:int8"])
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--agreement-images", type=int, default=30)
    parser.add_argument("--placements", default="npu 2t,npu 2t + cpu tail,npu 1t + cpu tail",
                        help="placements to evaluate on images (YOLO26-S on 2 NPU tiles loses the device, M0)")
    args = parser.parse_args()
    bench = {m.name: m for m in all_models()}
    results = {}
    for spec in args.models:
        name, precision = spec.split(":")
        entry = bench[name]
        model = ov.Core().read_model(entry.irs[precision])
        first = next(op.get_friendly_name() for op in model.get_ordered_ops() if op.get_type_name() in SELECTION)
        head, tail = split_from(model, [first])
        placements = {
            "npu 2t": [(model, "npu", None)], "npu 1t": [(model, "npu", ONE_TILE)],
            "npu 2t + cpu tail": [(head, "npu", None), (tail, "cpu", None)],
            "npu 1t + cpu tail": [(head, "npu", ONE_TILE), (tail, "cpu", None)],
        }
        timing = measure(placements, "npu", rounds=7, iterations=40)
        line = {k: round(t.median_ms, 2) if t.error is None else t.error[-80:] for k, t in timing.items()}
        print(f"{name} {precision}: latency {line}", flush=True)

        reference = Deployed(model, entry.task, "cpu", cache_dir=None)
        images = coco_val_subset(args.images).image_paths()
        ref = reference.run(images[:args.agreement_images])
        record = {"latency_ms": line, "tail_ops": len(tail.get_ordered_ops()),
                  "cpu fp32": {"AP": round(evaluate_coco(reference, coco_val_subset(args.images), "cpu", "fp32").AP, 4)}}
        print(f"  {'cpu fp32 (reference)':<20} AP {record['cpu fp32']['AP']:.3f} ({args.images} images)", flush=True)
        for label in args.placements.split(","):
            parts = placements[label]
            deployed = (Deployed(parts[0][0], entry.task, "npu", parts[0][2], cache_dir=None) if len(parts) == 1
                        else Deployed(parts, entry.task, cache_dir=None))
            agreement = entry.task.agreement(ref, deployed.run(images[:args.agreement_images]))
            ap = evaluate_coco(deployed, coco_val_subset(args.images), label, precision).AP
            record[label] = {"agreement": round(agreement, 3), "AP": round(ap, 4)}
            print(f"  {label:<20} agreement {agreement:.3f}  AP {ap:.3f} ({args.images} images)", flush=True)
        results[spec] = record
    out = ROOT / "runs" / "p2" / "selection_tail.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
