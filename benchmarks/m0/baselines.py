"""Catalogue of M0 baselines and the conventions needed to run their IRs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "m0"
ONNX_DIR = WORK / "onnx"
IR_DIR = WORK / "ir"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Baseline:
    name: str
    family: str
    imgsz: tuple[int, int]
    # "optinpu": (scores (1,Q,C) probabilities, boxes (1,Q,4) normalized xyxy) -- models of the registry
    # "detr": (logits (1,Q,C), boxes (1,Q,4) normalized cxcywh) -- RF-DETR, DEIMv2
    # "dets": (1,N,6) = x1, y1, x2, y2 (input pixels), score, class -- YOLO26 end-to-end head
    output: str
    resize: str = "stretch"  # "stretch" or "letterbox" (YOLO, pad value 114, centered)
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    # logits column of each of the 80 COCO classes (contiguous order); None = identity
    class_columns: list[int] | None = None
    published_ap: float | None = None  # COCO val2017 AP50:95 reported by the authors
    license: str = ""
    source: str = ""  # "optinpu:<registry name>" or "onnx"
    split: str | None = None  # op-name regex of the head for hybrid npu+gpu runs
    quant_transformer: bool = True  # NNCF transformer scheme (DETRs, ViTs); False for pure CNNs
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ir(self) -> Path:
        return self.ir_path("fp16")

    def ir_path(self, precision: str = "fp16") -> Path:
        suffix = {"fp16": "", "fp32": "_fp32", "int8": "_int8"}[precision]
        return IR_DIR / f"{self.name}{suffix}.xml"

    @property
    def onnx(self) -> Path:
        return ONNX_DIR / f"{self.name}.onnx"

    @property
    def task(self):
        """Pre / post-processing of the IR, as an optinpu.tasks.DetectionTask."""
        from optinpu.tasks import DetectionTask

        return DetectionTask(self.imgsz, self.output, self.resize, self.mean, self.std, self.class_columns)


# COCO category ids of the 80 classes, in contiguous order (RF-DETR predicts 91 columns indexed by id)
COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

_DETR_SPLIT = r"\.backbone\.|encoder_input_proj|\.encoder\."
_RFDETR_SPLIT = r"^/?backbone[/.]"  # DINOv2 backbone + projector; the rest is the two-stage transformer
_DEIMV2_SPLIT = r"^/?(backbone|encoder)[/.]"

BASELINES = {b.name: b for b in [
    Baseline("dfine-n", "D-FINE", (640, 640), "optinpu", published_ap=42.8, license="Apache-2.0",
             source="optinpu:dfine-n", split=_DETR_SPLIT),
    Baseline("dfine-s", "D-FINE", (640, 640), "optinpu", published_ap=48.5, license="Apache-2.0",
             source="optinpu:dfine-s", split=_DETR_SPLIT),
    Baseline("rtdetrv2-r18", "RT-DETRv2", (640, 640), "optinpu", published_ap=47.9, license="Apache-2.0",
             source="optinpu:rtdetrv2-r18", split=_DETR_SPLIT),
    Baseline("rfdetr-n", "RF-DETR", (384, 384), "detr", mean=IMAGENET_MEAN, std=IMAGENET_STD,
             class_columns=COCO_IDS, published_ap=48.4, license="Apache-2.0", source="onnx", split=_RFDETR_SPLIT),
    Baseline("rfdetr-s", "RF-DETR", (512, 512), "detr", mean=IMAGENET_MEAN, std=IMAGENET_STD,
             class_columns=COCO_IDS, published_ap=53.0, license="Apache-2.0", source="onnx", split=_RFDETR_SPLIT),
    Baseline("deimv2-n", "DEIMv2", (640, 640), "detr", published_ap=43.0,
             license="DEIMv2 License (non-commercial)", source="onnx", split=_DEIMV2_SPLIT),
    Baseline("deimv2-s", "DEIMv2", (640, 640), "detr", mean=IMAGENET_MEAN, std=IMAGENET_STD,
             published_ap=50.9, license="DEIMv2 License (non-commercial)", source="onnx", split=_DEIMV2_SPLIT,
             notes="measured 46.9 AP on the iGPU vs 50.9 published. Pre-processing matches the official eval "
                   "config (640x640 resize, ImageNet normalization); the ONNX export of the DINOv3 backbone "
                   "(rotary position embedding computed from shapes) is the main suspect, not verified."),
    Baseline("yolo26-n", "YOLO26", (640, 640), "dets", resize="letterbox", published_ap=40.1,
             license="AGPL-3.0", source="onnx", quant_transformer=False,
             notes="published AP is the end-to-end (NMS-free) one"),
    Baseline("yolo26-s", "YOLO26", (640, 640), "dets", resize="letterbox", published_ap=47.8,
             license="AGPL-3.0", source="onnx", quant_transformer=False,
             notes="published AP is the end-to-end (NMS-free) one"),
]}
