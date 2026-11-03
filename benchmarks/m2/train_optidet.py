"""M2.4: train OptiDet-N on a COCO train2017 subset on the iGPU, evaluate on val2017 [0, 500).

    python -m benchmarks.m2.train_optidet --images 10000 --epochs 24 --batch 16
    python -m benchmarks.m2.train_optidet --smoke          # 30 steps: speed check

The backbone starts from ImageNet RepVGG-A0 (MIT weights via timm); neck and heads are random.
"""

from __future__ import annotations

import argparse
import time

from optinpu import Detector
from optinpu.data import CocoDetection, coco_train_subset, coco_val_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="optidet-n")
    parser.add_argument("--images", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1000, help="steps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", nargs="?", const="bf16", default=False, choices=("bf16", "fp16"),
                        help="autocast, bf16 by default; fp16 (loss scaling) on GPUs without native bf16, e.g. T4")
    parser.add_argument("--resume", metavar="LAST_PT", help="continue an interrupted run from its last.pt")
    parser.add_argument("--max-hours", type=float, help="stop after the last epoch that fits in this time")
    parser.add_argument("--run", default="m2_coco10k")
    parser.add_argument("--distill", metavar="REPO", help="PKD feature distillation from this D-FINE checkpoint, "
                                                          "e.g. ustc-community/dfine-small-coco")
    parser.add_argument("--distill-weight", type=float, default=2.0)
    parser.add_argument("--smoke", nargs="?", type=int, const=30, metavar="STEPS",
                        help="STEPS steps (default 30) without validation: speed check")
    args = parser.parse_args()

    from optinpu.models import get_model

    imgsz = get_model(args.model).imgsz
    train_subset = coco_train_subset(args.images)
    val_subset = coco_val_subset(500)
    train = CocoDetection(train_subset.annotations, train_subset.image_dir, imgsz, augment=True, zoom=True)
    val = CocoDetection(val_subset.annotations, val_subset.image_dir, imgsz)
    # built with the dataset's classes, so that train() does not rebuild the heads (and drop the adapters)
    model = Detector(args.model, pretrained=True, class_names=train.class_names)
    if args.distill:
        from optinpu.models.distill import DFineTeacher

        model.model.enable_distillation(DFineTeacher(args.distill), weight=args.distill_weight)
        print(f"distillation from {args.distill} (PKD, weight {args.distill_weight})", flush=True)
    print(f"{args.model} at {imgsz}: {len(train)} training images, {len(val)} validation images", flush=True)

    config = dict(
        epochs=1 if args.smoke else args.epochs, batch=args.batch, lr=args.lr, backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay, warmup_steps=args.warmup, clip=10.0, ema_decay=0.9998,
        amp=args.amp, workers=args.workers, log_every=10 if args.smoke else 50,
        max_steps=args.smoke, resume=args.resume, max_hours=args.max_hours,
    )
    t = time.time()
    result = model.train(train, None if args.smoke else val, run=args.run + ("_smoke" if args.smoke else ""), **config)
    print(f"done in {(time.time() - t) / 60:.1f} min; best AP {result.best_ap:.3f} at epoch {result.best_epoch}", flush=True)


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    main()
