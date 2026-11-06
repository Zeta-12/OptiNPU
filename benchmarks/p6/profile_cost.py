"""P6 (review): what measuring costs, against predicting. For each model of the profiler validation (Table V),
the wall time of one measured run of all its interventions (adaptive rounds up to 120, as in the paper), of
the vendor counters (one profiled run) and of the additive model (every op compiled and timed alone), and
the number of interventions. Each model waits for mains power; Windows is kept awake.
Results: runs/p6/profile_cost.json (resumable) and runs/p6/profile_cost.md.

    python -m benchmarks.p6.profile_cost
"""

from __future__ import annotations

import json
import time

import openvino as ov

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from benchmarks.p1.validate import DEFAULT, additive_costs, interventions, measured_savings
from benchmarks.p3.campaign import keep_awake, wait_for_mains
from optinpu.profiler import reduced_outputs
from optinpu.sandbox import run_job

OUT = ROOT / "runs" / "p6"


def run(spec: str) -> dict:
    name, precision = spec.split(":")
    entry = next(m for m in all_models() if m.name == name)
    model = ov.Core().read_model(entry.irs[precision])
    t0 = time.perf_counter()
    found = interventions(model)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    measured_savings(reduced_outputs(model), {k: v for k, (_, v) in found.items()}, max_rounds=120)
    measured_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    job = run_job({"job": "vendor", "ir": model, "device": "npu"}, timeout=900)
    vendor_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    additive_costs(model)
    additive_s = time.perf_counter() - t0
    return {"interventions": len(found), "build_s": build_s, "measured_s": measured_s,
            "vendor_s": vendor_s, "vendor_ok": job.ok, "additive_s": additive_s}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "profile_cost.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    keep_awake(True)
    try:
        for spec in DEFAULT:
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
    lines = ["| Model | Interventions | Measured run | Vendor counters | Additive model |", "|---|---|---|---|---|"]
    for spec, r in results.items():
        lines.append(f"| {spec} | {r['interventions']} | {r['measured_s'] / 60:.1f} min | {r['vendor_s']:.0f} s | "
                     f"{r['additive_s'] / 60:.1f} min |")
    text = "\n".join(lines) + "\n"
    (OUT / "profile_cost.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
