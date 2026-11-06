"""Compare the one-to-one head (inference) with the one-to-many head + NMS on a checkpoint.

    python -m benchmarks.m2.diagnose_heads runs/optidet-n/<run>/best.pt [images]

If one-to-many + NMS is much better, training works and the one-to-one head lags (expected early);
if both are near zero, something is wrong in decoding / assignment / evaluation.
"""

from __future__ import annotations

import sys

import torch
from torchvision.ops import batched_nms

from optinpu import Detector
from optinpu.data import coco_val_subset
from optinpu.evaluate import evaluate_coco
from optinpu.postprocess import Detections, load_image, preprocess


class Many:
    """predict() with the one-to-many head and NMS (training-time head, for diagnosis only)."""

    def __init__(self, detector):
        self.detector, self.name = detector, detector.name + "-one2many"

    @torch.no_grad()
    def predict(self, source, device=None, precision=None, conf=0.0):
        d = self.detector
        model = d.model
        image = load_image(source)
        x = torch.from_numpy(preprocess(image, d.spec.imgsz))
        cls, reg = model.flatten(model.one2many(model.features(x)))
        scores = cls.sigmoid()[0]
        boxes = (model.boxes(model.distances(reg)) / model.scale)[0]
        # multi-label (box, class) pairs as in the standard COCO protocol and the one-to-one evaluation
        anchor, labels = (scores > 0.001).nonzero(as_tuple=True)
        best, boxes = scores[anchor, labels], boxes[anchor]
        idx = batched_nms(boxes, best, labels, 0.65)[:300]
        w, h = image.size
        scale = torch.tensor([w, h, w, h])
        return Detections((boxes[idx] * scale).numpy(), best[idx].numpy(), labels[idx].numpy(), d.class_names)


def main():
    checkpoint = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    detector = Detector.from_checkpoint("optidet-n", checkpoint)
    subset = coco_val_subset(n)
    one = evaluate_coco(detector, subset, device="torch", folds=5)
    many = evaluate_coco(Many(detector), subset, device="torch", folds=5)
    for label, r in (("one-to-one (inference head)", one), ("one-to-many + NMS", many)):
        print(f"{label:<28} AP {r.AP:.3f} AP50 {r.AP50:.3f} AP75 {r.AP75:.3f} APs {r.APs:.3f} APm {r.APm:.3f} "
              f"APl {r.APl:.3f} folds {[round(a, 3) for a in r.fold_ap]}")


if __name__ == "__main__":
    main()
