"""Quantization-aware fine-tuning of a trained OptiDet, then INT8 deployment on the NPU.

    python -m benchmarks.m2.qat runs/optidet-n/<run>/best.pt [--epochs 2]

Post-training INT8 loses ~1.3 AP on OptiDet, almost all in the fused RepVGG backbone (M2.5: keeping
backbone stages 2-4 in FP16 recovers it, but costs ~30% latency). QAT lets the fused weights adapt
to INT8 instead: NNCF inserts fake quantizers into forward(), and OptiDet.loss() goes through
forward(), so the whole training path is quantized. Fine-tuned in FP32 at a low learning rate.
"""

from __future__ import annotations

import argparse
import copy

import nncf
import torch

from optinpu import Detector
from optinpu.data import CocoDetection, coco_train_subset, coco_val_subset
from optinpu.evaluate import format_eval_table
from optinpu.export import export_openvino
from optinpu.postprocess import load_image, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--run", default="m2_qat")
    args = parser.parse_args()

    detector = Detector.from_checkpoint("optidet-n", args.checkpoint)
    imgsz = detector.spec.imgsz
    train_subset = coco_train_subset(10000)
    val_subset = coco_val_subset(500)
    train = CocoDetection(train_subset.annotations, train_subset.image_dir, imgsz, augment=True, zoom=True)
    val = CocoDetection(val_subset.annotations, val_subset.image_dir, imgsz)

    model = detector.model.fuse().eval()
    calibration = [torch.from_numpy(preprocess(load_image(p), imgsz)) for p in train_subset.image_paths()[-300:]]
    model = nncf.quantize(model, nncf.Dataset(calibration), subset_size=len(calibration))
    detector.model = model

    print("== PTQ starting point", flush=True)
    detector._ir["int8"] = export_openvino(copy.deepcopy(model).deploy(), detector.workdir / args.run / "ptq_int8.xml",
                                           (1, 3, *imgsz))
    start = detector.evaluate(val, "cpu", "int8")

    print(f"== QAT: {args.epochs} epochs, lr {args.lr}", flush=True)
    detector.train(train, val, run=args.run, epochs=args.epochs, batch=args.batch, lr=args.lr,
                   backbone_lr=args.lr, weight_decay=0.0, warmup_steps=100, clip=10.0, ema_decay=0.9998,
                   amp=False, workers=4, log_every=100)

    detector._ir["int8"] = export_openvino(copy.deepcopy(detector.model).deploy(),
                                           detector.workdir / "qat_int8.xml", (1, 3, *imgsz))
    detector._compiled = {}
    results = [start, detector.evaluate(val, "cpu", "int8"), detector.evaluate(val, "npu", "int8")]
    print(format_eval_table(results), flush=True)
    bench = detector.benchmark(("npu",), precision="int8", iterations=100, throughput_seconds=2)[0]
    report = detector.lint(device="npu", precision="int8", images=val_subset.image_paths()[:30])
    print(f"QAT INT8 on NPU: {bench.latency_ms_median:.2f} ms (2 tiles), detection agreement with CPU "
          f"{report.agreement:.3f}, lint {'OK' if report.ok else 'PROBLEMS'}", flush=True)


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    main()
