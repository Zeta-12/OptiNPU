"""M3: data loading speed alone, no model: does JPEG decoding and augmentation bound training on few CPUs?

    python -m benchmarks.m3.loader_speed --workers 4 --batches 100

Same dataset and loader settings as benchmarks/m2/train_optidet.py (train2017 subset, flip, jitter, zoom).
"""

from __future__ import annotations

import argparse
import time

import torch

from optinpu.data import CocoDetection, coco_train_subset
from optinpu.data.dataset import collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="optidet-n")
    parser.add_argument("--images", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()

    from optinpu.models import get_model

    subset = coco_train_subset(args.images)
    train = CocoDetection(subset.annotations, subset.image_dir, get_model(args.model).imgsz, augment=True, zoom=True)
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         collate_fn=collate, drop_last=True)
    batches = iter(loader)
    for _ in range(min(10, args.batches)):  # worker start-up
        next(batches)
    t = time.perf_counter()
    for _ in range(args.batches):
        next(batches)
    rate = args.batches * args.batch / (time.perf_counter() - t)
    print(f"loader alone, {args.workers} workers, batch {args.batch}: {rate:.1f} img/s", flush=True)


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    main()
