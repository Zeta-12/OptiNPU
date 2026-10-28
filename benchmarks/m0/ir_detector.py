"""Run any baseline IR with its own pre/post-processing, behind the Detector.predict interface."""

from __future__ import annotations

import numpy as np
from PIL import Image

from benchmarks.m0.baselines import Baseline
from optinpu.models.coco import COCO_CLASSES
from optinpu.postprocess import Detections, decode, load_image, preprocess
from optinpu.runtime import compile_hybrid, compile_model

LETTERBOX_PAD = 114


def letterbox(image: Image.Image, imgsz: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping the aspect ratio, pad (centered) to imgsz=(H, W). Returns (x, scale, (pad_x, pad_y))."""
    h, w = imgsz
    scale = min(w / image.width, h / image.height)
    nw, nh = round(image.width * scale), round(image.height * scale)
    canvas = Image.new("RGB", (w, h), (LETTERBOX_PAD,) * 3)
    pad_x, pad_y = (w - nw) // 2, (h - nh) // 2
    canvas.paste(image.resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), scale, (pad_x, pad_y)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prepare_input(baseline: Baseline, image: Image.Image):
    """(model input, geometry needed to map boxes back) with the baseline's own pre-processing."""
    if baseline.resize == "letterbox":
        x, scale, pad = letterbox(image, baseline.imgsz)
        return x, (scale, pad)
    return preprocess(image, baseline.imgsz, baseline.mean, baseline.std), None


class IRDetector:
    """predict(source, device, precision, conf) for a Baseline IR, as optinpu.evaluate expects.

    device: "cpu", "gpu", "npu", "npu+gpu"; config: extra compile properties (e.g. {"NPU_TILES": 1});
    precision selects the IR ("fp16" or "int8").
    """

    def __init__(self, baseline: Baseline, device: str, config: dict | None = None, precision: str = "fp16"):
        self.baseline = baseline
        self.name = baseline.name
        self.class_names = list(COCO_CLASSES)
        ir = baseline.ir_path(precision)
        if "+" in device:
            head, tail = device.split("+")
            npu_cfg = config if tail.lower() == "npu" else None
            self.model = compile_hybrid(ir, baseline.split, head, tail, tail_config=npu_cfg)
        else:
            self.model = compile_model(ir, device, config=config)

    def inputs(self, image: Image.Image):
        return prepare_input(self.baseline, image)

    def predict(self, source, device=None, precision=None, conf: float = 0.4) -> Detections:
        image = load_image(source)
        x, geometry = self.inputs(image)
        outputs = self.model(x)
        return self.decode(outputs, image.size, geometry, conf)

    def decode(self, outputs, orig_size, geometry, conf: float) -> Detections:
        b = self.baseline
        if b.output == "optinpu":
            scores, boxes = outputs
            return decode(scores[0], boxes[0], orig_size, self.class_names, conf=conf)
        if b.output == "detr":
            logits, boxes = _order_detr_outputs(outputs)
            logits, boxes = logits[0], boxes[0]
            if b.class_columns is not None:
                logits = logits[:, b.class_columns]
            cx, cy, w, h = boxes.T
            xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            return decode(_sigmoid(logits), xyxy, orig_size, self.class_names, conf=conf)
        if b.output == "dets":
            dets = outputs[0][0]
            dets = dets[dets[:, 4] >= conf]
            scale, (pad_x, pad_y) = geometry
            boxes = (dets[:, :4] - np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)) / scale
            w, h = orig_size
            boxes = boxes.clip(0, [w, h, w, h])
            order = np.argsort(-dets[:, 4])
            return Detections(boxes[order], dets[order, 4], dets[order, 5].astype(np.int64), self.class_names)
        raise ValueError(f"unknown output format {b.output!r}")


def _order_detr_outputs(outputs):
    """(logits, boxes) whatever the export order: boxes are the output with 4 columns."""
    a, b = outputs[:2]
    return (b, a) if a.shape[-1] == 4 else (a, b)
