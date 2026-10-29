"""Does the 16-op selection tail of YOLO26 alone lose the NPU on 2 tiles? (minimal repro for Intel)

The head runs on the CPU (FP32) on real images; its outputs feed the tail alone, compiled on the NPU
(2 tiles, then 1 tile), in a sandboxed process. Saves the tail IR next to the report.

    python -m benchmarks.p2.tail_repro [yolo26-s] [--images 30]
"""

from __future__ import annotations

import argparse

import openvino as ov

from benchmarks.m0.baselines import BASELINES, ROOT
from optinpu.data import coco_val_subset
from optinpu.graph import split_from
from optinpu.rewrites import selection_tail
from optinpu.sandbox import run_predictions
from optinpu.tasks import OutputTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="yolo26-s")
    parser.add_argument("--images", type=int, default=30)
    args = parser.parse_args()
    baseline = BASELINES[args.model]
    model = ov.Core().read_model(baseline.ir_path("fp16"))
    roots, n_ops, _ = selection_tail(model)
    head, tail = split_from(model, roots)
    out = ROOT / "runs" / "p2" / f"{args.model}_tail.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(tail, out, compress_to_fp16=False)
    print(f"tail: {n_ops} ops, inputs {[str(i.get_partial_shape()) for i in tail.inputs]} -> {out}", flush=True)

    compiled_head = ov.Core().compile_model(head, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    samples = []
    for path in coco_val_subset(args.images).image_paths():
        x, _ = baseline.task.inputs(path)
        result = compiled_head(x)
        samples.append([result[o] for o in compiled_head.outputs])
    for label, config in (("2 tiles", None), ("1 tile", {"NPU_TILES": 1})):
        outputs, job = run_predictions([(tail, "npu", config)], OutputTask(), samples, timeout=600)
        status = "ok" if outputs is not None else ("DEVICE LOST" if job.device_lost else f"failed: {job.error[-150:]}")
        print(f"tail alone on NPU, {label}, {len(samples)} real inputs: {status}", flush=True)


if __name__ == "__main__":
    main()
