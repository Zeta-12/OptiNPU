"""P6 (review): the latencies of the paper's tables with 95 % confidence intervals.

For each model, one paired session (adaptive rounds, the fixed noise input as everywhere else) of every
deployment the paper reports, against the original on the NPU (two tiles):
- npu-optimized: OptiNPU's NPU deployment (runs/p6/confirm/deploy/ when P6 re-ran the model with the held-out
  confirmation, else runs/p5/deploy/; none when the original is kept);
- npu+gpu: the deployment with the decoder on the iGPU (runs/p3/offload-gpu/deploy/, DETRs);
- gpu, gpu-optimized: the original on the iGPU, and OptiNPU's iGPU deployment (runs/p6/igpu-confirm/, else
  runs/p6/igpu/).
Each saving comes with its 95 % half-width; the speedup interval follows from it. Each model waits for mains
power; Windows is kept awake. Results: runs/p6/final.json (resumable) and runs/p6/final.md.

    python -m benchmarks.p6.final
"""

from __future__ import annotations

import json

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from benchmarks.p3.campaign import keep_awake, wait_for_mains
from benchmarks.p5.deployments import MODELS as DETECTORS
from benchmarks.p5.energy import CLASSIFIERS, deployments
from optinpu.profiler import Adaptive, measure, paired_saving

OUT = ROOT / "runs" / "p6"


def run(spec: str) -> dict:
    name, precision = spec.split(":")
    entry = next(m for m in all_models() if m.name == name)
    variants = {k: [tuple(p) for p in v] for k, v in deployments(name, precision, entry, p6=True).items()
                if k not in ("cpu", "npu-turbo")}
    timings = measure(variants, "npu", adaptive=Adaptive("npu", max_rounds=60, max_seconds=900), timeout=3600)
    base = timings["npu"].median_ms
    out = {"npu": {"median_ms": base, "rounds": len(timings["npu"].rounds), "error": timings["npu"].error}}
    for k in variants:
        if k == "npu":
            continue
        t = timings[k]
        if t.error is not None:
            out[k] = {"error": t.error[-200:]}
            continue
        saving, ci = paired_saving(timings, "npu", k)
        latency = base - saving
        out[k] = {"latency_ms": latency, "ci_ms": ci, "median_ms": t.median_ms,
                  "speedup": base / latency, "speedup_low": base / (latency + ci), "speedup_high": base / (latency - ci)}
    return out


def table(results: dict) -> str:
    labels = ["npu-optimized", "npu+gpu", "gpu", "gpu-optimized"]
    lines = ["| Model | NPU original (ms) | " + " | ".join(f"{k}: ms ± CI (speedup)" for k in labels) + " |",
             "|---|---|" + "---|" * len(labels)]
    for spec, r in results.items():
        cells = []
        for k in labels:
            v = r.get(k)
            cells.append("" if v is None else "failed" if "error" in v else
                         f"{v['latency_ms']:.2f} ± {v['ci_ms']:.2f} ({v['speedup']:.2f}×)")
        lines.append(f"| {spec} | {r['npu']['median_ms']:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "final.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    keep_awake(True)
    try:
        for spec in (*DETECTORS, *CLASSIFIERS):
            if spec in results:
                print(f"{spec}: already done", flush=True)
                continue
            wait_for_mains()
            print(f"===== {spec}", flush=True)
            results[spec] = run(spec)
            print(json.dumps(results[spec], indent=1), flush=True)
            path.write_text(json.dumps(results, indent=1))
    finally:
        keep_awake(False)
    text = table(results)
    (OUT / "final.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
