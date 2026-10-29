"""P0 exit check: every bench model runs through the generic API (optinpu.tasks, optinpu.sandbox).

Before benchmarks/m0/ir_detector.py was rebuilt on optinpu.tasks, the library gave detections
identical to the M0 harness on all 9 baselines (10 images each), so the M0 AP carry over; OptiDet-N
gave its M2.5 AP (CPU 9.7, NPU 10.0 on val2017 [0, 500)). This script now checks that each model
runs (AP on a few images on CPU for detectors) and that a sandboxed NPU latency job works.

    python -m benchmarks.p0.check_tasks [--images 50]
"""

from __future__ import annotations

import argparse

from benchmarks.p0.models import all_models
from optinpu.data import coco_val_subset
from optinpu.evaluate import evaluate_coco
from optinpu.sandbox import run_job
from optinpu.tasks import Deployed, OutputTask, random_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=50)
    args = parser.parse_args()
    subset = coco_val_subset(args.images)

    for model in all_models():
        precision, ir = next(iter(model.irs.items()))
        if isinstance(model.task, OutputTask):
            import openvino as ov

            samples = random_samples(ov.Core().read_model(ir), 1)
            outputs = Deployed(ir, model.task, "cpu", cache_dir=None).run(samples)
            check = f"outputs {[o.shape for o in outputs[0]]}"
        else:
            result = evaluate_coco(Deployed(ir, model.task, "cpu", name=model.name, cache_dir=None), subset, "cpu",
                                   precision)
            check = f"AP {result.AP:.3f} on {args.images} images (CPU)"
        job = run_job({"job": "latency", "ir": ir, "device": "npu", "config": model.npu_config, "iterations": 20})
        latency = f"{job.result['latency_ms_median']:.2f} ms" if job.ok else f"FAILED: {job.error}"
        print(f"{model.name:<22} {precision:<5} {check:<40} NPU {latency} (peak {job.peak_gb:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
