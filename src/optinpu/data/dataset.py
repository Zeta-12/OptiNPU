"""Training datasets: any COCO-format folder (images + instances json)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance


class CocoDetection(torch.utils.data.Dataset):
    """COCO-format detection dataset.

    Returns (pixels (3, H, W) float in [0, 1], target) with target = {
        "class_labels": (N,) contiguous class indices,
        "boxes": (N, 4) normalized cxcywh,     # the format DETR losses expect
        "image_id": int,
    }
    Images are resized (stretched) to `imgsz`, as at inference time.
    """

    def __init__(
        self,
        annotations: str | Path,
        image_dir: str | Path,
        imgsz: tuple[int, int] = (640, 640),
        augment: bool = False,
        keep_empty: bool = False,
    ):
        data = json.loads(Path(annotations).read_text())
        self.annotations = Path(annotations)
        self.image_dir = Path(image_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.category_ids = sorted(c["id"] for c in data["categories"])
        names = {c["id"]: c["name"] for c in data["categories"]}
        self.class_names = [names[i] for i in self.category_ids]
        contiguous = {cid: i for i, cid in enumerate(self.category_ids)}

        per_image: dict[int, list] = {im["id"]: [] for im in data["images"]}
        for a in data["annotations"]:
            if a.get("iscrowd", 0) or a["bbox"][2] <= 1 or a["bbox"][3] <= 1:
                continue
            per_image[a["image_id"]].append((contiguous[a["category_id"]], a["bbox"]))
        self.images = [im for im in sorted(data["images"], key=lambda im: im["id"]) if keep_empty or per_image[im["id"]]]
        self.targets = {im["id"]: per_image[im["id"]] for im in self.images}

    @property
    def image_ids(self) -> list[int]:
        return [im["id"] for im in self.images]

    def image_path(self, image_id: int) -> Path:
        return self.image_dir / next(im["file_name"] for im in self.images if im["id"] == image_id)

    def to_subset(self):
        """CocoSubset view (for evaluation) over the same images."""
        from optinpu.data.coco import CocoSubset

        return CocoSubset(self.annotations, self.image_dir, self.image_ids, self.category_ids,
                          {im["id"]: im["file_name"] for im in self.images})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        info = self.images[index]
        image = Image.open(self.image_dir / info["file_name"]).convert("RGB")
        w, h = image.size
        anns = self.targets[info["id"]]
        labels = np.array([c for c, _ in anns], dtype=np.int64)
        boxes = np.array([b for _, b in anns], dtype=np.float32).reshape(-1, 4)  # xywh pixels

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enhancer(image).enhance(random.uniform(0.7, 1.3))

        image = image.resize((self.imgsz[1], self.imgsz[0]), Image.BILINEAR)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        cxcywh = np.stack([
            (boxes[:, 0] + boxes[:, 2] / 2) / w,
            (boxes[:, 1] + boxes[:, 3] / 2) / h,
            boxes[:, 2] / w,
            boxes[:, 3] / h,
        ], axis=1).clip(0, 1)
        target = {
            "class_labels": torch.from_numpy(labels),
            "boxes": torch.from_numpy(cxcywh.astype(np.float32)),
            "image_id": info["id"],
        }
        return pixels, target


def collate(batch):
    pixels, targets = zip(*batch)
    return torch.stack(pixels), list(targets)
