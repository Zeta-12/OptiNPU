"""L3: fine-tune D-FINE on a 10-class COCO "animals" subset on the iGPU, then run it on the NPU.

    python examples/finetune_animals.py                 # full run (iGPU)
    python examples/finetune_animals.py --smoke cpu     # 4 steps, to check the pipeline
"""

import argparse

from optinpu import Detector
from optinpu.bench import format_table
from optinpu.data import coco_val_classes
from optinpu.evaluate import format_eval_table

ANIMALS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dfine-n")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--smoke", metavar="DEVICE", help="4 training steps on DEVICE, then export")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--backbone-device", help="run the frozen backbone on this device, e.g. npu")
    parser.add_argument("--run", default="animals")
    args = parser.parse_args()

    train = coco_val_classes(ANIMALS, 500, 3500, name="animals")  # disjoint from benchmark/calibration
    val = coco_val_classes(ANIMALS, 4000, 500, name="animals")

    model = Detector(args.model)
    if args.smoke:
        result = model.train(train, val, run="smoke", epochs=1, batch=2, max_steps=4,
                             device=args.smoke, log_every=1, workers=0)
    else:
        result = model.train(train, val, run=args.run, epochs=args.epochs, batch=args.batch,
                             freeze_backbone=args.freeze_backbone, backbone_device=args.backbone_device)
    print(f"best AP {result.best_ap:.3f} at epoch {result.best_epoch} -> {result.checkpoint}")

    results = [model.evaluate(val, device=d) for d in ("cpu", "npu")]
    print(format_eval_table(results))
    print(format_table(model.benchmark(("cpu", "npu"), iterations=30, throughput_seconds=1)))
    print(model.predict(val.image_paths()[0], device="npu"))
