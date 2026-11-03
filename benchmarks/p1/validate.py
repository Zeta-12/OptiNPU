"""P1 exit check: do the vendor counters, or an additive cost model, predict what a change saves?

Ground truth: the measured saving of interventions on the model, each measured twice (in reverse
order the second time) to know our own run-to-run noise:
- bypass of module regions (every depth; bypassable, 2 ops or more; distinct op sets),
- bypass of every shape-preserving op of a type,
- rewrites: activations -> ReLU, LayerNorm removed, selection tail on the CPU.
Predictions of each intervention's saving, from the set S of ops it removes:
- vendor: PERF_COUNT task times summed over S ("raw"), or rescaled by latency / sum of all tasks;
- additive: isolated costs summed over S (each op alone on the NPU, minus the cost of a graph made of
  its inputs only): the lookup-table approach of nn-Meter and of hardware-aware NAS.
Metrics: Spearman rank correlation with the measured savings, mean absolute error, top-3 overlap.
Windows is kept awake and each model waits for mains power (as benchmarks.p3.campaign); models already in
--out are skipped.

    python -m benchmarks.p1.validate [model:precision ...] [--max-rounds 40] [--out runs/p1/validate]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
from scipy.stats import spearmanr

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from benchmarks.p3.campaign import keep_awake, wait_for_mains
from optinpu.graph import split_from
from optinpu.profiler import (
    SHAPE_PRESERVING_TYPES,
    Adaptive,
    bypass,
    bypass_types,
    bypassable,
    isolated_ops,
    measure,
    module_regions,
    paired_saving,
    reduced_outputs,
    runtime_ops,
)
from optinpu.rewrites import activations_to_relu, selection_tail, without_norms
from optinpu.sandbox import run_job

OUT = ROOT / "runs" / "p1" / "validate"
DEFAULT = ["yolo26-n:fp16", "dfine-n:fp16", "resnet50_640:fp32", "mobilenet_v2_640:fp32", "efficientnet_b0_640:fp32"]
ACTIVATIONS = ("Swish", "HSwish", "Gelu", "Mish", "HSigmoid")


def interventions(model: ov.Model, max_regions: int = 40) -> dict[str, tuple[list[str], object]]:
    """name -> (removed op names, variant): an ov.Model, or [(model, device, config)] parts."""
    out: dict[str, tuple[list[str], object]] = {}
    seen: set[frozenset] = set()
    regions = []
    for depth in range(1, 9):
        for region in module_regions(model, depth):
            key = frozenset(region.ops)
            if len(region.ops) >= 2 and key not in seen:
                seen.add(key)
                regions.append(region)
    for region in sorted(regions, key=lambda r: -len(r.ops)):
        if sum(k.startswith("region:") for k in out) >= max_regions:
            break
        variant = bypass(model, region.ops)
        if variant is not None and len(region.ops) < len(runtime_ops(model)):
            out[f"region:{region.name}"] = (region.ops, reduced_outputs(variant))
    for t in SHAPE_PRESERVING_TYPES:
        names = bypassable(model, {t})
        if names:
            out[f"type:{t}"] = (names, reduced_outputs(bypass_types(model, {t})[0]))
    names = [op.get_friendly_name() for op in runtime_ops(model) if op.get_type_name() in ACTIVATIONS]
    if names:
        out["rewrite:activations-to-relu"] = (names, reduced_outputs(activations_to_relu(model, ACTIVATIONS)[0]))
    names = [op.get_friendly_name() for op in runtime_ops(model) if op.get_type_name() == "MVN"]
    if names:
        out["rewrite:layernorm-removed"] = (names, reduced_outputs(without_norms(model)[0]))
    found = selection_tail(model)
    if found is not None and found[2] == 0:
        head, tail = split_from(model, found[0])
        names = [op.get_friendly_name() for op in runtime_ops(tail)]
        out["rewrite:selection-to-cpu"] = (names, [(head, "npu", None), (reduced_outputs(tail), "cpu", None)])
    return out


def measured_savings(reference: ov.Model, variants: dict, batch: int = 6, reverse: bool = False,
                     max_rounds: int = 40) -> tuple[dict[str, float], dict[str, float], float]:
    """(saving, 95 % half-width, latency): savings paired per round with the full model, rounds repeated
    until the half-width is below 0.5 % of the latency or 0.02 ms (at most max_rounds; 15 s per round
    allowed, as the default 40 rounds in 600 s)."""
    keys = list(variants)[::-1] if reverse else list(variants)
    savings, widths, fulls = {}, {}, []
    adaptive = Adaptive("full", max_rounds=max_rounds, max_seconds=15 * max_rounds)
    for start in range(0, len(keys), batch):
        chunk = keys[start:start + batch]
        timings = measure({"full": reference, **{k: variants[k] for k in chunk}}, "npu", adaptive=adaptive,
                          timeout=adaptive.max_seconds + 1200)
        fulls.append(timings["full"].median_ms)
        for k in chunk:
            savings[k], widths[k] = paired_saving(timings, "full", k)
        rounds = len(timings["full"].rounds)
        print(f"  measured {start + len(chunk)}/{len(keys)} ({rounds} rounds)", flush=True)
    return savings, widths, float(np.nanmedian(fulls))


def additive_costs(model: ov.Model, batch: int = 12) -> dict[str, float]:
    """Isolated cost (ms) of every runtime op, by name."""
    iso = isolated_ops(model)
    graphs = {f"op:{s}": g for s, g in iso.graphs.items()}
    graphs.update({f"in:{k}": g for k, g in iso.empties.items()})
    keys, timings = list(graphs), {}
    for start in range(0, len(keys), batch):
        timings.update(measure({k: graphs[k] for k in keys[start:start + batch]}, "npu", rounds=3, iterations=30))
    print(f"  isolated {len(iso.graphs)} distinct ops", flush=True)
    cost = {}
    for name, signature in iso.signature.items():
        if signature not in iso.graphs:
            cost[name] = 0.0
            continue
        op_ms = timings[f"op:{signature}"].median_ms
        empty_ms = timings[f"in:{iso.inputs[signature]}"].median_ms
        cost[name] = max(0.0, op_ms - empty_ms) if np.isfinite(op_ms) and np.isfinite(empty_ms) else 0.0
    return cost


def metrics(truth: dict[str, float], predicted: dict[str, float]) -> dict:
    keys = [k for k in truth if np.isfinite(truth[k]) and np.isfinite(predicted.get(k, np.nan))]
    t = np.array([truth[k] for k in keys])
    p = np.array([predicted[k] for k in keys])
    top = lambda d: {k for k, _ in sorted(((k, d[k]) for k in keys), key=lambda kv: -kv[1])[:3]}
    return {"n": len(keys), "spearman": float(spearmanr(t, p).statistic) if len(keys) > 2 else float("nan"),
            "mae_ms": float(np.mean(np.abs(t - p))), "top3": len(top(truth) & top(predicted)) / 3}


def validate(name: str, precision: str, max_rounds: int = 40) -> dict:
    entry = next(m for m in all_models() if m.name == name)
    model = ov.Core().read_model(entry.irs[precision])
    reference = reduced_outputs(model)
    found = interventions(model)
    print(f"{name} {precision}: {len(found)} interventions", flush=True)
    variants = {k: v for k, (_, v) in found.items()}
    run1, ci1, latency = measured_savings(reference, variants, max_rounds=max_rounds)
    run2, ci2, _ = measured_savings(reference, variants, reverse=True, max_rounds=max_rounds)
    truth = {k: float(np.nanmean([run1[k], run2[k]])) for k in found}

    job = run_job({"job": "vendor", "ir": model, "device": "npu"}, timeout=900)
    vendor = {layer["name"]: layer["ms"] for layer in job.result["layers"]} if job.ok else {}
    vendor_total = sum(vendor.values()) or float("nan")
    cost = additive_costs(model)
    predictions = {
        "vendor raw": {k: sum(vendor.get(n, 0.0) for n in ops) for k, (ops, _) in found.items()},
        "vendor scaled": {k: sum(vendor.get(n, 0.0) for n in ops) * latency / vendor_total for k, (ops, _) in found.items()},
        "additive": {k: sum(cost.get(n, 0.0) for n in ops) for k, (ops, _) in found.items()},
        "measured (repeat)": run2,
    }
    result = {
        "model": name, "precision": precision, "max_rounds": max_rounds, "latency_ms": latency, "vendor_sum_ms": vendor_total,
        "additive_sum_ms": sum(cost.values()), "interventions": {
            k: {"ops": len(ops), "run1": run1[k], "run2": run2[k], "ci1": ci1[k], "ci2": ci2[k], "mean": truth[k],
                **{p: v[k] for p, v in predictions.items()}}
            for k, (ops, _) in found.items()},
        "metrics": {p: metrics(run1, v) for p, v in predictions.items()},
    }
    print(f"  latency {latency:.2f} ms, vendor sum {vendor_total:.2f} ms, additive sum {sum(cost.values()):.2f} ms",
          flush=True)
    for p, m in result["metrics"].items():
        print(f"  {p:<18} spearman {m['spearman']:+.2f}  MAE {m['mae_ms']:6.2f} ms  top-3 {m['top3']:.2f}  (n={m['n']})",
              flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="model:precision (default: the five of the paper)")
    parser.add_argument("--max-rounds", type=int, default=40, help="cap of the adaptive rounds")
    parser.add_argument("--out", type=Path, default=OUT, help="results folder")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    keep_awake(True)
    try:
        for spec in args.models or DEFAULT:
            name, precision = spec.split(":")
            path = args.out / f"{name}_{precision}.json"
            if path.exists():
                print(f"{spec}: already done ({path})", flush=True)
                continue
            wait_for_mains()
            path.write_text(json.dumps(validate(name, precision, args.max_rounds), indent=1))
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
