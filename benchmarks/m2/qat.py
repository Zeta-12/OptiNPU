"""Quantization-aware fine-tuning of a trained OptiDet, then INT8 deployment on the NPU.

    python -m benchmarks.m2.qat runs/optidet-n/<run>/best.pt [--epochs 2] [--max-steps N]

Post-training INT8 loses ~1.3 AP on OptiDet, almost all in the fused RepVGG backbone (M2.5: keeping
backbone stages 2-4 in FP16 recovers it, but costs ~30% latency). QAT (Detector.train_qat) lets the
fused weights adapt to INT8 instead. M2.5 ran an earlier version of this script, which measured the
starting point with NNCF's PyTorch PTQ instead of the library's OpenVINO PTQ.
"""

from __future__ import annotations

import argparse

from optinpu import Detector
from optinpu.data import CocoDetection, coco_train_subset, coco_val_subset
from optinpu.evaluate import format_eval_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--max-steps", type=int, help="stop early (smoke test)")
    parser.add_argument("--run", default="qat")
    args = parser.parse_args()

    detector = Detector.from_checkpoint("optidet-n", args.checkpoint)
    imgsz = detector.spec.imgsz
    train_subset = coco_train_subset(10000)
    val_subset = coco_val_subset(args.val_images)
    train = CocoDetection(train_subset.annotations, train_subset.image_dir, imgsz, augment=True, zoom=True)
    val = CocoDetection(val_subset.annotations, val_subset.image_dir, imgsz)

    print("== post-training INT8 (starting point)", flush=True)
    results = [detector.evaluate(val, "cpu", "fp32"), detector.evaluate(val, "cpu", "int8")]

    print(f"== QAT: {args.epochs} epochs, lr {args.lr}", flush=True)
    detector.train_qat(train, val, run=args.run, calibration=train_subset.image_paths()[-300:],
                       epochs=args.epochs, batch=args.batch, lr=args.lr, backbone_lr=args.lr,
                       max_steps=args.max_steps, workers=4, log_every=100)
    results += [detector.evaluate(val, "cpu"), detector.evaluate(val, "npu")]
    print(format_eval_table(results), flush=True)
    bench = detector.benchmark(("npu",), iterations=100, throughput_seconds=2)[0]
    report = detector.lint(device="npu", images=val_subset.image_paths()[:30])
    print(f"QAT INT8 on NPU: {bench.latency_ms_median:.2f} ms (2 tiles), detection agreement with CPU "
          f"{report.agreement:.3f}, lint {'OK' if report.ok else 'PROBLEMS'}", flush=True)


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    main()
