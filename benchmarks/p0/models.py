"""P0 evaluation bench of the profiler and optimizer (PLAN.md, Enjeu 3).

- 10 detectors, judged on COCO: the 9 M0 baselines (IRs in runs/m0/ir, see benchmarks/m0) and
  OptiDet-N (M2.5 QAT, INT8). Third-party models (YOLO26: AGPL, DEIMv2: non-commercial) are only
  measured, through their IRs; no code or weights of theirs enter optinpu.
- 7 ImageNet CNNs from torchvision (feature extractors, random weights), latency only: the
  accuracy guard compares raw outputs (OutputTask). They cover blocks absent from the detectors
  (depthwise, squeeze-excitation, HardSwish, LayerNorm).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.m0.baselines import BASELINES, ROOT
from optinpu.tasks import DetectionTask, OutputTask

IR_DIR = ROOT / "runs" / "p0" / "ir"
OPTIDET_QAT = ROOT / "runs" / "optidet-n" / "m2_coco10k_e10_kd" / "m2_qat" / "qat_int8.xml"
# the same distilled checkpoint before QAT (M2.5), in float: the starting point of a PTQ / mixed precision search
OPTIDET_FLOAT = ROOT / "runs" / "optidet-n" / "m2_coco10k_e10_kd" / "optidet-n_fp32_416x416.xml"


@dataclass
class BenchModel:
    name: str
    family: str
    license: str
    task: DetectionTask | OutputTask
    irs: dict[str, Path]  # precision -> IR ("fp16", "fp32", "int8")
    split: str | None = None  # head of the hybrid npu+gpu placement
    npu_config: dict | None = None  # known-good NPU config (M0: 1 tile when 2 tiles lose AP)


def detectors() -> list[BenchModel]:
    models = [
        BenchModel(b.name, b.family, b.license, b.task, {p: b.ir_path(p) for p in ("fp16", "fp32", "int8")}, b.split)
        for b in BASELINES.values()
    ]
    models.append(BenchModel("optidet-n", "OptiDet", "Apache-2.0", DetectionTask((416, 416)), {"int8": OPTIDET_QAT}))
    models.append(BenchModel("optidet-n-float", "OptiDet", "Apache-2.0", DetectionTask((416, 416)),
                             {"fp32": OPTIDET_FLOAT}))
    return models


CNNS = ("resnet18", "resnet50", "regnet_x_400mf", "mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0",
        "convnext_tiny")


def cnn(name: str, resolution: int = 640) -> BenchModel:
    """Exported once to runs/p0/ir/<name>_<resolution>.xml (FP32 weights, random)."""
    path = IR_DIR / f"{name}_{resolution}.xml"
    if not path.exists():
        import openvino as ov
        import torch

        from benchmarks.m1.blocks import _torchvision
        from optinpu.export import load_model

        torch.manual_seed(0)
        path.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(load_model(_torchvision(name)(), (1, 3, resolution, resolution)), path, compress_to_fp16=False)
    return BenchModel(f"{name}_{resolution}", "torchvision CNN", "BSD-3-Clause", OutputTask(), {"fp32": path})


def all_models() -> list[BenchModel]:
    return detectors() + [cnn(name) for name in CNNS]
