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

    Returns (pixels (3, H, W) uint8, target) with target = {
        "class_labels": (N,) contiguous class indices,
        "boxes": (N, 4) normalized cxcywh,     # the format DETR losses expect
        "image_id": int,
    }
    Images are resized (stretched) to `imgsz`, as at inference time. Models take float pixels in [0, 1]:
    the training engine scales the uint8 ones on its device.
    """

    def __init__(
        self,
        annotations: str | Path,
        image_dir: str | Path,
        imgsz: tuple[int, int] = (640, 640),
        augment: bool = False,
        keep_empty: bool = False,
        zoom: bool = False,
        mosaic: float = 0.0,
    ):
        """`augment`: horizontal flip and color jitter; `zoom` adds scale augmentation (random_zoom),
        recommended when training a detector from scratch (OptiDet); `mosaic`: probability that a sample is
        a mosaic of 4 images (see mosaic()) instead of one zoomed image (with augment only)."""
        data = json.loads(Path(annotations).read_text())
        self.annotations = Path(annotations)
        self.image_dir = Path(image_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.zoom = zoom
        self.mosaic = mosaic
        self.jitter = True  # color jitter here (with augment); off when the engine does it on the GPU
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

    def _load(self, index: int, size: tuple[int, int] | None = None):
        """(image, labels (N,), boxes (N, 4) xywh pixels of the image). With `size` = (width, height) the
        image is to be shown at, the JPEG is decoded at reduced size when that still covers `size` (draft
        mode, 1/2 to 1/8), boxes rescaled accordingly."""
        info = self.images[index]
        image = Image.open(self.image_dir / info["file_name"])  # reads the header only, not the pixels
        anns = self.targets[info["id"]]
        labels = np.array([c for c, _ in anns], dtype=np.int64)
        boxes = np.array([b for _, b in anns], dtype=np.float32).reshape(-1, 4)
        if "width" in info and image.size != (info["width"], info["height"]):  # a shrunk copy of the images
            boxes *= np.array([image.size[0] / info["width"], image.size[1] / info["height"]] * 2, dtype=np.float32)
        if size is not None:
            w, h = image.size
            image.draft("RGB", size)
            boxes *= np.array([image.size[0] / w, image.size[1] / h] * 2, dtype=np.float32)
        # convert() copies even an RGB image: decoded by the first operation instead
        return (image if image.mode == "RGB" else image.convert("RGB")), labels, boxes

    def __getitem__(self, index: int):
        if self.augment and self.mosaic and random.random() < self.mosaic:
            others = [random.randrange(len(self)) for _ in range(3)]
            image, boxes, labels = mosaic(self._load, [index, *others], self.imgsz)
        else:
            image, labels, boxes = self._load(index)
            if self.augment:
                if self.zoom:
                    image, boxes, labels = random_zoom(image, boxes, labels)
                if random.random() < 0.5:
                    image, boxes = hflip(image, boxes)
        w, h = image.size
        if (w, h) != (self.imgsz[1], self.imgsz[0]):
            image = image.resize((self.imgsz[1], self.imgsz[0]), Image.BILINEAR)
        if self.augment and self.jitter:
            # after the resize: the same jitter (up to rounding) on fewer pixels; it was 41 % of the loading time
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enhancer(image).enhance(random.uniform(0.7, 1.3))
        # uint8: 4x less to pass from the workers, scaled to [0, 1] on the training device (engine)
        pixels = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1)

        cxcywh = np.stack([
            (boxes[:, 0] + boxes[:, 2] / 2) / w,
            (boxes[:, 1] + boxes[:, 3] / 2) / h,
            boxes[:, 2] / w,
            boxes[:, 3] / h,
        ], axis=1).clip(0, 1)
        target = {
            "class_labels": torch.from_numpy(labels),
            "boxes": torch.from_numpy(cxcywh.astype(np.float32)),
            "image_id": self.images[index]["id"],
        }
        return pixels, target


def hflip(image: Image.Image, boxes: np.ndarray):
    """Horizontal flip (boxes in xywh pixels)."""
    boxes = boxes.copy()
    boxes[:, 0] = image.size[0] - boxes[:, 0] - boxes[:, 2]
    return image.transpose(Image.FLIP_LEFT_RIGHT), boxes


def mosaic(load, indices: list[int], imgsz: tuple[int, int], gain: tuple[float, float] = (0.5, 1.5)):
    """4 images around a random center of an imgsz canvas (YOLO mosaic, built directly at the output size).

    `load(index, size)` -> (image, labels, boxes xywh pixels), possibly decoded smaller but covering `size`.
    Each image is stretched to the canvas size times a gain drawn once per mosaic (scale augmentation;
    at gain 1 objects have their size without mosaic), flipped with probability 0.5 and placed in its
    quadrant against the center, so only part of it shows (and only that part is resized). Boxes are
    clipped to the visible part and dropped when less than 30% of their area remains or a side is under
    2 px. Returns (image, boxes, labels).
    """
    H, W = imgsz
    xc, yc = round(random.uniform(0.25, 0.75) * W), round(random.uniform(0.25, 0.75) * H)
    g = random.uniform(*gain)
    tw, th = max(1, round(W * g)), max(1, round(H * g))  # every image is stretched to this size
    canvas = Image.new("RGB", (W, H), (124, 116, 104))
    all_boxes, all_labels = [], []
    for k, index in enumerate(indices):
        image, labels, boxes = load(index, (tw, th))
        if random.random() < 0.5:
            image, boxes = hflip(image, boxes)
        # top-left corner of the stretched image on the canvas: against the center, in quadrant k
        x0 = xc - tw if k in (0, 2) else xc
        y0 = yc - th if k in (0, 1) else yc
        # visible part (canvas pixels) and the source region it comes from
        vx0, vy0, vx1, vy1 = max(x0, 0), max(y0, 0), min(x0 + tw, W), min(y0 + th, H)
        if vx1 <= vx0 or vy1 <= vy0:
            continue
        sx, sy = image.size[0] / tw, image.size[1] / th
        source = ((vx0 - x0) * sx, (vy0 - y0) * sy, (vx1 - x0) * sx, (vy1 - y0) * sy)
        canvas.paste(image.resize((vx1 - vx0, vy1 - vy0), Image.BILINEAR, box=source), (vx0, vy0))
        if not len(boxes):
            continue
        boxes = boxes / np.array([sx, sy, sx, sy], dtype=np.float32)
        x1 = np.clip(boxes[:, 0] + x0, vx0, vx1)
        y1 = np.clip(boxes[:, 1] + y0, vy0, vy1)
        x2 = np.clip(boxes[:, 0] + boxes[:, 2] + x0, vx0, vx1)
        y2 = np.clip(boxes[:, 1] + boxes[:, 3] + y0, vy0, vy1)
        keep = ((x2 - x1) * (y2 - y1) >= 0.3 * boxes[:, 2] * boxes[:, 3]) & (x2 - x1 > 2) & (y2 - y1 > 2)
        all_boxes.append(np.stack([x1, y1, x2 - x1, y2 - y1], 1)[keep])
        all_labels.append(labels[keep])
    boxes = np.concatenate(all_boxes).astype(np.float32) if all_boxes else np.zeros((0, 4), np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0, np.int64)
    return canvas, boxes, labels


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
