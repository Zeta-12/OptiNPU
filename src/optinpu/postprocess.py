"""Pre- and post-processing shared by every backend (numpy only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy in original image pixels
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) class indices
    class_names: list[str]

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def names(self) -> list[str]:
        return [self.class_names[i] for i in self.labels]

    def __repr__(self) -> str:
        items = ", ".join(f"{n} {s:.2f}" for n, s in zip(self.names, self.scores))
        return f"Detections({len(self)}: {items})"

    def draw(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        for (x0, y0, x1, y1), name, score in zip(self.boxes, self.names, self.scores):
            draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=2)
            draw.text((x0 + 3, y0 + 2), f"{name} {score:.2f}", fill=(255, 64, 64))
        return image


def load_image(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert("RGB")
    return Image.open(source).convert("RGB")


def preprocess(
    image: Image.Image,
    imgsz: tuple[int, int],
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Resize (stretch) to imgsz=(H, W), scale to [0, 1], optional normalization -> (1, 3, H, W) float32."""
    h, w = imgsz
    x = np.asarray(image.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    if mean is not None and std is not None:
        x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def decode(
    scores: np.ndarray,
    boxes: np.ndarray,
    orig_size: tuple[int, int],
    class_names: list[str],
    conf: float = 0.4,
    max_det: int = 300,
) -> Detections:
    """Top-k over (query, class) pairs, as in DETR-style NMS-free heads.

    scores: (Q, C) probabilities, boxes: (Q, 4) normalized xyxy, orig_size: (W, H).
    """
    num_classes = scores.shape[1]
    flat = scores.reshape(-1)
    k = min(max_det, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    idx = idx[flat[idx] >= conf]

    queries, labels = np.divmod(idx, num_classes)
    w, h = orig_size
    out_boxes = boxes[queries] * np.array([w, h, w, h], dtype=np.float32)
    out_boxes = out_boxes.clip(0, [w, h, w, h])
    return Detections(out_boxes, flat[idx], labels, class_names)
