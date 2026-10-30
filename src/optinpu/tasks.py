"""Tasks: how to feed a model, read its outputs and compare two runs.

The comparison (`agreement`, 1.0 = identical) is the accuracy guard of the profiler and optimizer:
a rewrite or a compile option is acceptable when the device run still agrees with the reference
(CPU FP32 of the original model). Detection is the reference task, classification the second one
(`metric`: what an annotated evaluation set measures); `OutputTask` covers any model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar

import numpy as np
import openvino as ov
from PIL import Image

from optinpu.lint import detection_agreement
from optinpu.models.coco import COCO_CLASSES
from optinpu.postprocess import Detections, decode, load_image, preprocess
from optinpu.runtime import DEFAULT_CACHE, compile_hybrid, compile_model

LETTERBOX_PAD = 114
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox(image: Image.Image, imgsz: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
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


def _order_detr_outputs(outputs):
    """(logits, boxes) whatever the export order: boxes are the output with 4 columns."""
    a, b = outputs[:2]
    return (b, a) if a.shape[-1] == 4 else (a, b)


@dataclass
class DetectionTask:
    """Object detection, with the input / output conventions of the main detector families.

    output: "optinpu": scores (1, Q, C) probabilities, boxes (1, Q, 4) normalized xyxy (optinpu models)
            "detr": logits (1, Q, C), boxes (1, Q, 4) normalized cxcywh (RF-DETR, DEIMv2, D-FINE ONNX)
            "dets": (1, N, 6) = x1, y1, x2, y2 in input pixels, score, class (YOLO end-to-end heads)
    resize: "stretch", or "letterbox" (aspect ratio kept, centered, padded with 114 as YOLO does).
    class_columns: output column of each class (RF-DETR predicts 91 COCO ids); None = identity.
    """

    name: ClassVar[str] = "detection"
    metric: ClassVar[str] = "AP"
    imgsz: tuple[int, int]
    output: str = "optinpu"
    resize: str = "stretch"
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    class_columns: list[int] | None = None
    class_names: list[str] = field(default_factory=lambda: list(COCO_CLASSES))

    def inputs(self, source) -> tuple[list[np.ndarray], tuple]:
        """Model inputs for one image source, and what decode() needs to map boxes back."""
        image = load_image(source)
        if self.resize == "letterbox":
            x, scale, pad = letterbox(image, self.imgsz)
            return [x], (image.size, (scale, pad))
        return [preprocess(image, self.imgsz, self.mean, self.std)], (image.size, None)

    def decode(self, outputs: list[np.ndarray], meta: tuple, conf: float = 0.0) -> Detections:
        size, geometry = meta
        if self.output == "optinpu":
            scores, boxes = outputs[:2]
            return decode(scores[0], boxes[0], size, self.class_names, conf=conf)
        if self.output == "detr":
            logits, boxes = _order_detr_outputs(outputs)
            logits, boxes = logits[0], boxes[0]
            if self.class_columns is not None:
                logits = logits[:, self.class_columns]
            cx, cy, w, h = boxes.T
            xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            return decode(_sigmoid(logits), xyxy, size, self.class_names, conf=conf)
        if self.output == "dets":
            dets = outputs[0][0]
            dets = dets[dets[:, 4] >= conf]
            if geometry is None:  # stretched: input pixels -> image pixels, per axis
                sx, sy = size[0] / self.imgsz[1], size[1] / self.imgsz[0]
                boxes = dets[:, :4] * np.array([sx, sy, sx, sy], dtype=np.float32)
            else:
                scale, (pad_x, pad_y) = geometry
                boxes = (dets[:, :4] - np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)) / scale
            w, h = size
            boxes = boxes.clip(0, [w, h, w, h])
            order = np.argsort(-dets[:, 4])
            return Detections(boxes[order], dets[order, 4], dets[order, 5].astype(np.int64), self.class_names)
        raise ValueError(f"unknown detection output format {self.output!r}")

    def agreement(self, reference: list[Detections], test: list[Detections]) -> float:
        """F1 of matched detections (same class, IoU >= 0.7, score >= 0.3); calibrated on M0 (PLAN.md, M2.1)."""
        return detection_agreement(reference, test)


@dataclass
class ClassificationTask:
    """Image classification, torchvision preprocessing: the shorter side resized to `resize`, a center
    crop of `imgsz`, mean / std normalization. Predictions are [logits]; agreement is the share of
    samples whose top-1 class matches the reference (the classifier's guard, as F1 is the detector's).
    """

    name: ClassVar[str] = "classification"
    metric: ClassVar[str] = "top-1"
    imgsz: tuple[int, int] = (224, 224)
    resize: int = 256
    interpolation: str = "bilinear"  # or "bicubic" (EfficientNet)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def inputs(self, source) -> tuple[list[np.ndarray], None]:
        image = load_image(source)
        short, long = sorted(image.size)
        size = (self.resize, int(self.resize * long / short))  # as torchvision: (shorter, longer) side
        w, h = size if image.width <= image.height else size[::-1]
        resample = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[self.interpolation]
        image = image.resize((w, h), resample)
        ch, cw = self.imgsz
        left, top = int(round((w - cw) / 2.0)), int(round((h - ch) / 2.0))
        x = np.asarray(image.crop((left, top, left + cw, top + ch)), dtype=np.float32) / 255.0
        x = (x - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        return [np.ascontiguousarray(x.transpose(2, 0, 1)[None])], None

    def decode(self, outputs: list[np.ndarray], meta=None, conf: float = 0.0) -> list[np.ndarray]:
        return [np.asarray(outputs[0], dtype=np.float32).reshape(-1)]

    def agreement(self, reference: list, test: list) -> float:
        same = [int(np.argmax(r[0])) == int(np.argmax(t[0])) for r, t in zip(reference, test)]
        return float(np.mean(same)) if same else 0.0


@dataclass
class OutputTask:
    """Any model: a sample is the list of its input arrays, predictions are the raw outputs.

    agreement is the worst output's 1 - relative L2 error (0 when off by 100 %). Raw errors track
    accuracy poorly (PLAN.md, M0: harmless FP16 noise vs real detection drifts): prefer a task metric.
    """

    name: ClassVar[str] = "outputs"

    def inputs(self, sample) -> tuple[list[np.ndarray], None]:
        return [np.asarray(x, dtype=np.float32) for x in sample], None

    def decode(self, outputs: list[np.ndarray], meta=None, conf: float = 0.0) -> list[np.ndarray]:
        return [np.array(o, dtype=np.float32) for o in outputs]

    def agreement(self, reference: list, test: list) -> float:
        worst = 1.0
        for ref, out in zip(reference, test):
            for r, o in zip(ref, out):
                worst = min(worst, 1.0 - float(np.linalg.norm(o - r) / (np.linalg.norm(r) + 1e-12)))
        return max(worst, 0.0)


def task_to_dict(task) -> dict:
    """JSON-ready description of a task (to rebuild it in a sandboxed process)."""
    if isinstance(task, (DetectionTask, ClassificationTask)):
        return {"name": task.name, **{f.name: getattr(task, f.name) for f in fields(task)}}
    if isinstance(task, OutputTask):
        return {"name": "outputs"}
    raise TypeError(f"cannot serialize task {task!r}")


def task_from_dict(data: dict):
    data = dict(data)
    name = data.pop("name")
    if name == "detection":
        data["imgsz"] = tuple(data["imgsz"])
        return DetectionTask(**data)
    if name == "classification":
        return ClassificationTask(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
    return OutputTask()


def save_predictions(path, predictions: list) -> None:
    """npz of Detections (boxes, scores, labels) or of lists of arrays, per sample."""
    arrays = {}
    for i, p in enumerate(predictions):
        if isinstance(p, Detections):
            arrays.update({f"p{i}_boxes": p.boxes, f"p{i}_scores": p.scores, f"p{i}_labels": p.labels})
        else:
            arrays.update({f"p{i}_out{j}": a for j, a in enumerate(p)})
    np.savez(path, count=np.array(len(predictions)), **arrays)


def load_predictions(path, class_names: list[str] | None = None) -> list:
    with np.load(path) as data:
        result = []
        for i in range(int(data["count"])):
            if f"p{i}_boxes" in data.files:
                result.append(Detections(data[f"p{i}_boxes"], data[f"p{i}_scores"], data[f"p{i}_labels"],
                                         class_names or []))
            else:
                keys = sorted((k for k in data.files if k.startswith(f"p{i}_out")), key=lambda k: int(k[len(f"p{i}_out"):]))
                result.append([data[k] for k in keys])
        return result


def random_samples(model: ov.Model, count: int = 4, seed: int = 0) -> list[list[np.ndarray]]:
    """Uniform [0, 1) inputs matching the model's static input shapes (samples for OutputTask)."""
    rng = np.random.default_rng(seed)
    shapes = [tuple(p.get_partial_shape().to_shape()) for p in model.get_parameters()]
    return [[rng.random(s, dtype=np.float32) for s in shapes] for _ in range(count)]


class Deployed:
    """A model compiled on a device, with its task.

    predict() follows Detector.predict, so optinpu.evaluate.evaluate_coco accepts it; run() gives the
    predictions over many samples, for task.agreement(). device: "cpu" (FP32 reference precision),
    "gpu", "npu", or hybrid "npu+gpu" with `split` (op-name regex of the head, see graph.split_after).
    `config` applies to the NPU part(s). `model` can also be a list of one or two (model, device, config)
    parts (e.g. split by graph.split_from, or optimizer.load_deployment), run back to back; `device`
    and `config` are then unused.
    """

    def __init__(self, model, task, device: str = "npu", config: dict | None = None,
                 name: str = "model", split: str | None = None, cache_dir: str | Path | None = DEFAULT_CACHE):
        self.task, self.name, self.device = task, name, device
        self.class_names = getattr(task, "class_names", None)
        if isinstance(model, (list, tuple)) and len(model) == 1:
            model, device, config = model[0]
            self.device = device
        if isinstance(model, (list, tuple)):
            from optinpu.runtime import HybridModel

            (head, head_device, head_config), (tail, tail_device, tail_config) = model
            self.device = f"{head_device}+{tail_device}"
            self.model = HybridModel(compile_model(head, head_device, config=head_config, cache_dir=cache_dir),
                                     compile_model(tail, tail_device, config=tail_config, cache_dir=cache_dir))
        elif "+" in device:
            if split is None:
                raise ValueError(f"hybrid device {device!r} needs `split`")
            head, tail = device.split("+")
            self.model = compile_hybrid(model, split, head, tail, head_config=config if head == "npu" else None,
                                        tail_config=config if tail == "npu" else None)
        else:
            if device.lower() == "cpu":
                config = {"INFERENCE_PRECISION_HINT": "f32", **(config or {})}
            self.model = compile_model(model, device, config=config, cache_dir=cache_dir)

    def predict(self, source, device=None, precision=None, conf: float = 0.0):
        inputs, meta = self.task.inputs(source)
        return self.task.decode(self.model(*inputs), meta, conf)

    def run(self, samples) -> list:
        return [self.predict(s) for s in samples]
