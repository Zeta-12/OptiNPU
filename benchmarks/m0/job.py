"""One M0 measurement, run in its own process by run.py (which enforces memory and time limits).

    python -m benchmarks.m0.job '{"model": "dfine-n", "kind": "latency", "device": "npu", "precision": "int8"}'

kind: "latency" | "ap" | "lint" | "quantize". device: "cpu", "gpu", "npu" (1 tile), "npu-2t"
(compiler default: 2 tiles) or "npu+gpu" (hybrid). precision: "fp16" (default) or "int8".
Prints one line "RESULT <json>".
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from benchmarks.m0.baselines import BASELINES

DEVICES = {
    "cpu": ("cpu", None),
    "gpu": ("gpu", None),
    "npu": ("npu", {"NPU_TILES": 1}),
    "npu-2t": ("npu", None),
    "npu+gpu": ("npu+gpu", None),
}
LINT_IMAGE = "000000039769.jpg"  # two cats, two remotes, a couch


def run(spec: dict) -> dict:
    from benchmarks.m0.ir_detector import IRDetector, prepare_input

    baseline = BASELINES[spec["model"]]
    precision = spec.get("precision", "fp16")
    if spec["kind"] == "quantize":
        from benchmarks.m0.prepare import quantize

        return quantize(baseline)

    device, config = DEVICES[spec["device"]]
    ir = baseline.ir_path(precision)
    if not ir.exists():
        raise FileNotFoundError(f"{ir} missing (prepare / quantize first)")
    if spec["kind"] == "latency":
        from optinpu.bench import benchmark, benchmark_callable

        if "+" in device:
            t = time.perf_counter()
            det = IRDetector(baseline, device, config, precision)
            compile_s = time.perf_counter() - t
            inputs = [np.random.default_rng(0).random(s, dtype=np.float32) for s in det.model.input_shapes]
            r = benchmark_callable(det.model, inputs, det.model.device, iterations=100, compile_s=compile_s)
        else:
            r = benchmark(ir, device, iterations=100, throughput_seconds=3, config=config)
        return r.as_dict()
    if spec["kind"] == "ap":
        from optinpu.data import coco_val_subset
        from optinpu.evaluate import evaluate_coco

        det = IRDetector(baseline, device, config, precision)
        res = evaluate_coco(det, coco_val_subset(spec.get("images", 5000)), device=device, precision=precision)
        return res.as_dict()
    if spec["kind"] == "lint":
        from optinpu.data.coco import DEFAULT_ROOT
        from optinpu.lint import lint
        from optinpu.postprocess import load_image

        x, _ = prepare_input(baseline, load_image(DEFAULT_ROOT / "val2017" / LINT_IMAGE))
        report = lint(ir, device, [x], config=config)
        return {"ok": report.ok, "compiles": report.compiles, "rel_error": report.rel_error,
                "latency_ms": report.latency_ms, "recommended_config": report.recommended_config,
                "issues": [f"{i.severity}:{i.code}" for i in report.issues]}
    raise ValueError(f"unknown kind {spec['kind']!r}")


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    print("RESULT " + json.dumps(run(spec)), flush=True)
