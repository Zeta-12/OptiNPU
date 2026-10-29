"""P1: measured profile of a bench model, next to the vendor counters (PERF_COUNT).

    python -m benchmarks.p1.run_profile yolo26-n [--precision fp16] [--one-tile]

Writes runs/p1/<model>_<precision>_<tiles>.json and prints both views.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import openvino as ov

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from optinpu.profiler import module_regions, op_modules, profile_measured, runtime_ops

OUT = ROOT / "runs" / "p1"


def vendor_view(model: ov.Model, config: dict | None, depth: int) -> dict:
    """Vendor per-task times (sandboxed), summed per op type and per region of the measured profile."""
    from optinpu.sandbox import run_job

    job = run_job({"job": "vendor", "ir": model, "device": "npu", "config": config}, timeout=900)
    if not job.ok:
        return {"error": job.error}
    names = {op.get_friendly_name() for op in runtime_ops(model)}
    modules = op_modules(model)
    region_of = {name: r.name for r in module_regions(model, depth) for name in r.ops}
    by_type, by_region, matched, total = {}, {}, 0.0, 0.0
    for layer in job.result["layers"]:
        by_type[layer["node_type"]] = by_type.get(layer["node_type"], 0.0) + layer["ms"]
        total += layer["ms"]
        if layer["name"] in names:
            matched += layer["ms"]
            region = region_of.get(layer["name"], ".".join(modules[layer["name"]]))
            by_region[region] = by_region.get(region, 0.0) + layer["ms"]
    return {"sum_ms": total, "matched_ms": matched, "by_type": by_type, "by_region": by_region,
            "exec_types": job.result["exec_types"], "sample_names": job.result["layers"][:8]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--precision", default=None, help="IR precision (default: the first available)")
    parser.add_argument("--one-tile", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    bench = next(m for m in all_models() if m.name == args.model)
    precision = args.precision or next(iter(bench.irs))
    model = ov.Core().read_model(bench.irs[precision])
    config = {"NPU_TILES": 1} if args.one_tile else None
    report = profile_measured(model, "npu", config, rounds=args.rounds)
    print(report, flush=True)

    from optinpu.profiler import pick_depth

    vendor = vendor_view(model, config, pick_depth(model, 24))
    print(f"\nvendor counters: sum {vendor.get('sum_ms', float('nan')):.2f} ms "
          f"({vendor.get('matched_ms', float('nan')):.2f} ms on ops named as in the IR)", flush=True)
    for t, ms in sorted(vendor.get("by_type", {}).items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {t:<30} {ms:7.2f} ms")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.model}_{precision}_{'1t' if args.one_tile else '2t'}.json"
    path.write_text(json.dumps({"measured": asdict(report), "vendor": vendor}, indent=1, default=str))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
