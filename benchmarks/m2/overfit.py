"""Pipeline check: OptiDet must overfit a few real COCO images (train and evaluate on the same ones).

    python -m benchmarks.m2.overfit [images] [steps]

A high AP on the training images validates decoding, assignment, loss and evaluation end to end.
"""

from __future__ import annotations

import sys

from optinpu import Detector
from optinpu.data import CocoDetection, coco_train_subset


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    model = Detector("optidet-n", pretrained=True)
    subset = coco_train_subset(n)  # first n images of the 10k subset (already downloaded)
    data = CocoDetection(subset.annotations, subset.image_dir, model.spec.imgsz)
    epochs = max(1, steps * 8 // len(data))
    result = model.train(data, data, run="m2_overfit", epochs=epochs, batch=8, lr=1e-3, backbone_lr=5e-4,
                         weight_decay=0.0, warmup_steps=50, clip=10.0, ema_decay=0.99, amp=True, workers=0,
                         log_every=100)
    print(f"overfit {n} images: best AP {result.best_ap:.3f} at epoch {result.best_epoch}", flush=True)


if __name__ == "__main__":
    main()
