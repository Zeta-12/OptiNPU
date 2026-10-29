"""P3.1: why does the optimizer's INT8 candidate of YOLO26-N not compile on the NPU, when M0's does?

Quantizes the FP16 and FP32 IRs with and without NNCF's transformer scheme (50 calibration images),
splits off the selection tail as the optimizer does, and compiles each NPU head in a sandbox.

    python -m benchmarks.p3.int8_compile [yolo26-n]
"""

from __future__ import annotations

import sys

import openvino as ov

from benchmarks.m0.baselines import BASELINES
from optinpu.data import coco_train_subset
from optinpu.graph import split_from
from optinpu.rewrites import int8_ptq, selection_tail
from optinpu.sandbox import run_job


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "yolo26-n"
    baseline = BASELINES[name]
    calibration = coco_train_subset(10000).image_paths()[-80:-30]
    for precision in ("fp16", "fp32"):
        model = ov.Core().read_model(baseline.ir_path(precision))
        for transformer in (True, False):
            quantized = int8_ptq(model, baseline.task, calibration, transformer)
            found = selection_tail(quantized)
            head = split_from(quantized, found[0])[0] if found else quantized
            for label, graph in (("whole", quantized), ("head", head)):
                job = run_job({"job": "latency", "ir": graph, "device": "npu", "iterations": 20}, timeout=900)
                result = f"{job.result['latency_ms_median']:.2f} ms" if job.ok else f"FAILED {job.error[-160:]}"
                print(f"{name} {precision} transformer={transformer} {label:<5}: {result}", flush=True)


if __name__ == "__main__":
    main()
