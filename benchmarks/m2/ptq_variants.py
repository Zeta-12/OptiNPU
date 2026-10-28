"""Which post-training INT8 quantization settings lose the least AP on a trained OptiDet?

    python -m benchmarks.m2.ptq_variants runs/optidet-n/<run>/best.pt

Evaluated with OpenVINO on CPU (the NPU reproduces the CPU exactly on OptiDet: detection agreement
1.000), on COCO val2017 [0, 500); calibration on train2017 images that are not evaluated.
"""

from __future__ import annotations

import sys

import nncf

from optinpu import Detector
from optinpu.data import coco_train_subset, coco_val_subset
from optinpu.evaluate import evaluate_coco
from optinpu.postprocess import load_image, preprocess
from optinpu.quantize import quantize_int8

VARIANTS = {
    "default (300 images)": dict(subset_size=300),
    "mixed preset": dict(subset_size=300, preset=nncf.QuantizationPreset.MIXED),
    "full bias correction": dict(subset_size=300, fast_bias_correction=False),
    "1000 images": dict(subset_size=1000),
}
# mixed precision: keep a part of the network in FP16 (NNCF ignored scope, op-name regexes)
MIXED = {
    "FP16 head": [r".*one2one\..*"],
    "FP16 head outputs": [r".*one2one\.levels\.\d\.(cls|reg)\.2/.*"],
    "FP16 stem": [r".*backbone\.stages\.0\..*"],
    "FP16 stem + first stage": [r".*backbone\.stages\.[01]\..*"],
    "FP16 neck": [r".*neck\..*"],
}
BACKBONE = {
    "FP16 backbone": [r".*backbone\..*"],
    "FP16 backbone stages 2-4": [r".*backbone\.stages\.[234]\..*"],
}


def main():
    detector = Detector.from_checkpoint("optidet-n", sys.argv[1])
    val = coco_val_subset(500)
    fp32 = detector.export(precision="fp32")
    calibration = coco_train_subset(10000).image_paths()[-1000:]
    transform = lambda p: preprocess(load_image(p), detector.spec.imgsz)  # noqa: E731
    print(f"FP32 (CPU): AP {evaluate_coco(detector, val, 'cpu', 'fp32').AP:.3f}", flush=True)
    which = sys.argv[2] if len(sys.argv) > 2 else "presets"
    groups = {"mixed": MIXED, "backbone": BACKBONE}
    runs = ({k: (v, []) for k, v in VARIANTS.items()} if which == "presets"
            else {k: (dict(subset_size=300), v) for k, v in groups[which].items()})
    for name, (options, ignored) in runs.items():
        n = options["subset_size"]
        out = detector.workdir / "ptq" / (name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "") + ".xml")
        ir = quantize_int8(fp32, calibration[-n:], transform, out, transformer=False, ignored_patterns=ignored, **options)
        detector._ir["int8"] = ir
        detector._compiled = {k: v for k, v in detector._compiled.items() if not k.endswith(":int8")}
        print(f"INT8 {name:<24}: AP {evaluate_coco(detector, val, 'cpu', 'int8').AP:.3f}", flush=True)


if __name__ == "__main__":
    main()
