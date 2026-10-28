"""Deploy a trained OptiDet checkpoint on the NPU and check it end to end.

    python -m benchmarks.m2.evaluate_npu runs/optidet-n/<run>/best.pt

COCO val2017 [0, 500) AP on CPU FP32 (reference), NPU FP16 and NPU INT8 (both with the compiler
default, 2 tiles: OptiDet is designed to be exact there), NPU latency, and npu_lint judged on
detections. INT8: NNCF post-training quantization, CNN scheme, 300 train2017 images for calibration.
"""

from __future__ import annotations

import sys

from optinpu import Detector
from optinpu.data import coco_train_subset, coco_val_subset
from optinpu.evaluate import format_eval_table


def main():
    checkpoint = sys.argv[1]
    detector = Detector.from_checkpoint("optidet-n", checkpoint)
    val = coco_val_subset(500)
    calibration = coco_train_subset(10000).image_paths()[-300:]  # train images, not evaluated
    detector.export(precision="fp16")
    detector.export(precision="int8", calibration=calibration, transformer=False)

    results = [detector.evaluate(val, "cpu", "fp32")]
    results += [detector.evaluate(val, "npu", p) for p in ("fp16", "int8")]
    print(format_eval_table(results), flush=True)

    for precision in ("fp16", "int8"):
        bench = detector.benchmark(("npu",), precision=precision, iterations=100, throughput_seconds=2)[0]
        report = detector.lint(device="npu", precision=precision, images=val.image_paths()[:30])
        print(f"NPU {precision}: {bench.latency_ms_median:.2f} ms (2 tiles), detection agreement with CPU "
              f"{report.agreement:.3f}, lint {'OK' if report.ok else 'PROBLEMS'}", flush=True)


if __name__ == "__main__":
    main()
