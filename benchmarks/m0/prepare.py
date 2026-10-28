"""Build the FP16 OpenVINO IR of each baseline in runs/m0/ir (main environment).

Registered models are exported by optinpu (NPU patches, frozen constants); third-party ones are
converted from the ONNX files written by export_models.py, as-is apart from constant folding.
"""

from __future__ import annotations

import shutil
import sys

import openvino as ov

from benchmarks.m0.baselines import BASELINES, IR_DIR, Baseline
from optinpu.export import fold_constants


def prepare(baseline: Baseline, force: bool = False, precision: str = "fp16") -> str:
    """Write baseline.ir_path(precision) for precision "fp16" or "fp32" (the input of INT8 quantization)."""
    target = baseline.ir_path(precision)
    if target.exists() and not force:
        return "exists"
    IR_DIR.mkdir(parents=True, exist_ok=True)
    if baseline.source.startswith("optinpu:"):
        from optinpu import Detector

        xml = Detector(baseline.source.split(":", 1)[1]).export(precision=precision)
        shutil.copy(xml, target)
        shutil.copy(xml.with_suffix(".bin"), target.with_suffix(".bin"))
        return "exported by optinpu"
    if not baseline.onnx.exists():
        return f"missing {baseline.onnx.name}: run export_models.py in .venv-export"
    model = ov.convert_model(baseline.onnx)
    shapes = [p.get_partial_shape() for p in model.get_parameters()]
    if any(s.is_dynamic for s in shapes):
        return f"dynamic input {shapes}: re-export with static shapes"
    fold_constants(model)
    ov.save_model(model, target, compress_to_fp16=precision == "fp16")
    return "converted from ONNX"


def quantize(baseline: Baseline, calibration_images: int = 300) -> dict:
    """INT8 IR with NNCF post-training quantization, calibrated on COCO val2017 images
    [4500, 4800) (disjoint from the evaluation set) with the baseline's own pre-processing."""
    from benchmarks.m0.ir_detector import prepare_input
    from optinpu.data import coco_val_subset
    from optinpu.postprocess import load_image
    from optinpu.quantize import quantize_int8

    status = prepare(baseline, precision="fp32")
    if "missing" in status or "dynamic" in status:
        raise RuntimeError(status)
    images = coco_val_subset(calibration_images, offset=4500).image_paths()
    path = quantize_int8(
        baseline.ir_path("fp32"), images, lambda p: prepare_input(baseline, load_image(p))[0],
        baseline.ir_path("int8"), subset_size=calibration_images, transformer=baseline.quant_transformer,
    )
    return {"ir": str(path), "mb": round(path.with_suffix(".bin").stat().st_size / 1e6, 1)}


if __name__ == "__main__":
    for name in sys.argv[1:] or BASELINES:
        print(f"{name:<14} {prepare(BASELINES[name])}", flush=True)
