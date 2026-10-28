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
        zoom: bool = False,
    ):
        """`augment`: horizontal flip and color jitter; `zoom` adds scale augmentation (random_zoom),
        recommended when training a detector from scratch (OptiDet)."""
        data = json.loads(Path(annotations).read_text())
        self.annotations = Path(annotations)
        self.image_dir = Path(image_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.zoom = zoom
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
            if self.zoom:
                image, boxes, labels = random_zoom(image, boxes, labels)
                w, h = image.size
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


def random_zoom(image: Image.Image, boxes: np.ndarray, labels: np.ndarray):
    """Scale augmentation (boxes in xywh pixels): zoom-out with probability 0.3 (image on a canvas up to
    2x larger, filled with the dataset mean color), then zoom-in with probability 0.5 (crop of 50-100% of
    each side; boxes clipped, dropped when less than 30% of their area remains)."""
    w, h = image.size
    if random.random() < 0.3:
        ratio = random.uniform(1.0, 2.0)
        cw, ch = int(w * ratio), int(h * ratio)
        ox, oy = random.randint(0, cw - w), random.randint(0, ch - h)
        canvas = Image.new("RGB", (cw, ch), (124, 116, 104))
        canvas.paste(image, (ox, oy))
        image, w, h = canvas, cw, ch
        boxes = boxes + np.array([ox, oy, 0, 0], dtype=np.float32)
    if len(boxes) and random.random() < 0.5:
        for _ in range(10):
            cw, ch = int(w * random.uniform(0.5, 1.0)), int(h * random.uniform(0.5, 1.0))
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            x1 = np.clip(boxes[:, 0] - x0, 0, cw)
            y1 = np.clip(boxes[:, 1] - y0, 0, ch)
            x2 = np.clip(boxes[:, 0] + boxes[:, 2] - x0, 0, cw)
            y2 = np.clip(boxes[:, 1] + boxes[:, 3] - y0, 0, ch)
            keep = ((x2 - x1) * (y2 - y1) >= 0.3 * boxes[:, 2] * boxes[:, 3]) & (x2 - x1 > 2) & (y2 - y1 > 2)
            if keep.any():
                image = image.crop((x0, y0, x0 + cw, y0 + ch))
                boxes = np.stack([x1, y1, x2 - x1, y2 - y1], 1)[keep].astype(np.float32)
                labels = labels[keep]
                break
    return image, boxes, labels


def collate(batch):
    pixels, targets = zip(*batch)
    return torch.stack(pixels), list(targets)
