"""P1.0: does NPU latency depend on weight or input values? (the premise of measured what-ifs)

For each model: original weights, random weights (same mean / std), zero weights, 50 % and 90 %
unstructured sparsity, and the original with all-zero inputs, measured side by side (round-robin, one
process). A dense accelerator gives the same latency for all; one that skips zeros does not.

    python -m benchmarks.p1.weights [model ...]
"""

from __future__ import annotations

import json
import sys

import openvino as ov

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from optinpu.profiler import measure, weight_constants, with_weights

DEFAULT = [("resnet50_640", "fp32"), ("mobilenet_v2_640", "fp32"), ("yolo26-n", "fp16"), ("yolo26-n", "int8"),
           ("dfine-n", "fp16"), ("optidet-n", "int8")]


def main():
    wanted = [tuple(a.split(":")) for a in sys.argv[1:]] or DEFAULT
    bench = {m.name: m for m in all_models()}
    results = {}
    for name, precision in wanted:
        model = ov.Core().read_model(bench[name].irs[precision])
        variants = {"original": model, "zero inputs": model}
        for mode in ("random", "zeros", "sparse50", "sparse90"):
            variants[f"{mode} weights"] = with_weights(model, mode)
        timings = measure(variants, "npu", rounds=7, iterations=40, input_modes={"zero inputs": "zeros"})
        base = timings["original"].median_ms
        line = "  ".join(f"{k} {t.median_ms:.2f} ({100 * (t.median_ms / base - 1):+.1f} %)" if t.error is None
                         else f"{k} failed" for k, t in timings.items())
        print(f"{name} {precision} ({len(weight_constants(model))} weight tensors): {line}", flush=True)
        results[f"{name}:{precision}"] = {k: vars(t) for k, t in timings.items()}
    out = ROOT / "runs" / "p1" / "weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
