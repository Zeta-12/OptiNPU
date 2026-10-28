"""Export third-party M0 baselines to static ONNX (batch 1).

Runs ONLY in the isolated .venv-export (rfdetr, ultralytics, DEIMv2 requirements, CPU torch):

    .venv-export\\Scripts\\python -m benchmarks.m0.export_models [names...]

DEIMv2 code is imported from a local clone in runs/third_party/DEIMv2 (not versioned, DEIMv2 License:
non-commercial, benchmarking allowed). YOLO26 comes from ultralytics (AGPL-3.0). Nothing here is
imported by optinpu; the main environment only reads the resulting .onnx files.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
ONNX_DIR = ROOT / "runs" / "m0" / "onnx"
DEIMV2_DIR = ROOT / "runs" / "third_party" / "DEIMv2"


def export_rfdetr(name: str, class_name: str, resolution: int) -> Path:
    import rfdetr

    model = getattr(rfdetr, class_name)()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(model.export(output_dir=tmp, format="onnx", shape=(resolution, resolution),
                                 batch_size=1, verbose=False))
        return Path(shutil.copy(path, ONNX_DIR / f"{name}.onnx"))


def export_yolo(name: str, weights: str) -> Path:
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        model = YOLO(str(Path(tmp) / weights))  # downloads the official COCO checkpoint
        # nms=False selects the NMS-free one-to-one head (output (1, 300, 6)); the default (None)
        # silently exports the one-to-many head (1, 84, 8400) that needs NMS
        path = Path(model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17, nms=False))
        return Path(shutil.copy(path, ONNX_DIR / f"{name}.onnx"))


def export_deimv2(name: str, repo: str, imgsz: int = 640) -> Path:
    from huggingface_hub import PyTorchModelHubMixin

    sys.path.insert(0, str(DEIMV2_DIR))
    from engine.backbone import DINOv3STAs, HGNetv2
    from engine.deim import DEIMTransformer, DFINETransformer, HybridEncoder, LiteEncoder

    class DEIMv2(nn.Module, PyTorchModelHubMixin):
        """Same assembly as DEIMv2's hf_models.ipynb, without the post-processor."""

        def __init__(self, config):
            super().__init__()
            self.backbone = DINOv3STAs(**config["DINOv3STAs"]) if "DINOv3STAs" in config else HGNetv2(**config["HGNetv2"])
            self.encoder = LiteEncoder(**config["LiteEncoder"]) if "LiteEncoder" in config else HybridEncoder(**config["HybridEncoder"])
            self.decoder = (DEIMTransformer(**config["DEIMTransformer"]) if "DEIMTransformer" in config
                            else DFINETransformer(**config["DFINETransformer"]))

        def forward(self, images):
            out = self.decoder(self.encoder(self.backbone(images)))
            return out["pred_logits"], out["pred_boxes"]

    model = DEIMv2.from_pretrained(repo).eval()
    for module in model.modules():  # same as DEIMv2's model.deploy()
        if hasattr(module, "convert_to_deploy"):
            module.convert_to_deploy()
    path = ONNX_DIR / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (torch.rand(1, 3, imgsz, imgsz),), str(path), input_names=["images"],
                          output_names=["logits", "boxes"], opset_version=17, do_constant_folding=True, dynamo=False)
    return path


EXPORTS = {
    "rfdetr-n": lambda: export_rfdetr("rfdetr-n", "RFDETRNano", 384),
    "rfdetr-s": lambda: export_rfdetr("rfdetr-s", "RFDETRSmall", 512),
    "yolo26-n": lambda: export_yolo("yolo26-n", "yolo26n.pt"),
    "yolo26-s": lambda: export_yolo("yolo26-s", "yolo26s.pt"),
    "deimv2-n": lambda: export_deimv2("deimv2-n", "Intellindust/DEIMv2_HGNetv2_N_COCO"),
    "deimv2-s": lambda: export_deimv2("deimv2-s", "Intellindust/DEIMv2_DINOv3_S_COCO"),
}

if __name__ == "__main__":
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or EXPORTS:
        try:
            print(f"{name}: {EXPORTS[name]()}", flush=True)
        except Exception as e:  # keep going: one failing export must not block the others
            print(f"{name}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:300]}", flush=True)
