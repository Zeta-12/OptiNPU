"""Summarize runs/m0/results.jsonl into runs/m0/report.md (and stdout), around the OptiDet objective:
match SOTA AP with an NPU-only model much faster than the references' best placement (PLAN.md)."""

from __future__ import annotations

import json
from collections import defaultdict

from benchmarks.m0.baselines import BASELINES, WORK
from benchmarks.m0.run import RESULTS

# NPU-only configurations: (device, precision); "npu" = 1 tile (exact), "npu-2t" = compiler default
NPU_ONLY = [("npu", "fp16"), ("npu-2t", "fp16"), ("npu", "int8"), ("npu-2t", "int8")]
CONTEXT = [("cpu", "fp16"), ("gpu", "fp16"), ("npu+gpu", "fp16")]
TIERS = {"N": 43.0, "S": 48.5}  # AP targets (PLAN.md, Enjeu 2)
TIER_TOLERANCE = 1.0  # a model "reaches" a tier within this many AP points


def load(images: int = 5000) -> dict:
    """{model: {(kind, device, precision): result}}, last successful record of each job."""
    table: dict = defaultdict(dict)
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] != "ok" or (r["kind"] == "ap" and r.get("images") != images):
            continue
        table[r["model"]][(r["kind"], r["device"], r.get("precision", "fp16"))] = r["result"]
    return table


def pareto(points: list[tuple[str, float, float]]) -> set[str]:
    """Labels of (label, latency, ap) points not dominated by a faster-and-more-accurate one."""
    return {
        label for label, lat, ap in points
        if not any(l2 <= lat and a2 >= ap and (l2, a2) != (lat, ap) for _, l2, a2 in points)
    }


def fmt(value, spec=".1f"):
    return "—" if value is None else format(value, spec)


def _lat(rows, device, precision):
    return rows.get(("latency", device, precision), {}).get("latency_ms_median")


def _ap(rows, device, precision):
    ap = rows.get(("ap", device, precision), {}).get("AP")
    return None if ap is None else 100 * ap


def build(images: int = 5000) -> str:
    table = load(images)
    out = [
        f"# M0 — baselines on Core Ultra 5 235U (COCO val2017, {images} images, batch 1)",
        "",
        "Latency = model only, median ms. AP in %. `1t` = 1 NPU tile (exact for DETRs), `2t` = compiler "
        "default (2 tiles). INT8 = NNCF post-training quantization (300 calibration images).",
        "",
        "## NPU only (the OptiDet target)",
        "",
        "| Model | Res. | AP pub. | AP iGPU | FP16 1t AP / ms | FP16 2t AP / ms | INT8 1t AP / ms | INT8 2t ms | Lint 2t |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    npu_points, all_points, best = [], [], {}
    for name, b in BASELINES.items():
        rows = table.get(name)
        if not rows:
            continue
        ref_ap = _ap(rows, "gpu", "fp16")
        lint = rows.get(("lint", "npu-2t", "fp16"), {})
        # errors from the issue list (POOL-CEIL was an error in early runs, now a warning)
        errors = [i.split(":", 1)[1] for i in lint.get("issues", []) if i.startswith("error:") and i != "error:POOL-CEIL"]
        lint_txt = "—" if not lint else ("ok" if not errors else
                                          f"{', '.join(errors)} ({lint['rel_error']:.2f})")
        cells = []
        for device, precision in NPU_ONLY[:3]:
            ap, lat = _ap(rows, device, precision), _lat(rows, device, precision)
            cells.append(f"{fmt(ap)} / {fmt(lat)}")
            if ap is not None and lat is not None:
                npu_points.append((f"{name} {precision} {device}", lat, ap))
        out.append(f"| {name} | {b.imgsz[0]} | {fmt(b.published_ap)} | {fmt(ref_ap)} | " + " | ".join(cells)
                   + f" | {fmt(_lat(rows, 'npu-2t', 'int8'))} | {lint_txt} |")

        # best placement overall: NPU-only points with their measured AP, others with the iGPU AP
        candidates = [(lat, ap, f"{precision} {device}") for label, lat, ap in npu_points if label.startswith(name + " ")
                      for precision, device in [label.split(" ", 2)[1:]]]
        for device, precision in CONTEXT:
            lat = _lat(rows, device, precision)
            if lat is not None and ref_ap is not None:
                candidates.append((lat, ref_ap, device))
        if candidates:
            best[name] = min(candidates)
            all_points.append((f"{name} {best[name][2]}", best[name][0], best[name][1]))

    out += ["", "## Context: other placements (latency ms; AP of the iGPU run)", "",
            "| Model | CPU | iGPU | NPU+iGPU | Best placement overall | Best NPU-only | NPU-only / best |",
            "|---|---|---|---|---|---|---|"]
    for name in BASELINES:
        rows = table.get(name)
        if not rows or name not in best:
            continue
        own = [p for p in npu_points if p[0].startswith(name + " ")]
        best_npu = min(own, key=lambda p: p[1]) if own else None
        ratio = f"{best_npu[1] / best[name][0]:.1f}×" if best_npu else "—"
        out.append(
            f"| {name} | " + " | ".join(fmt(_lat(rows, d, p)) for d, p in CONTEXT)
            + f" | {best[name][2]} {best[name][0]:.1f} ms ({best[name][1]:.1f} AP)"
            + (f" | {best_npu[0].split(' ', 1)[1]} {best_npu[1]:.1f} ms ({best_npu[2]:.1f} AP)" if best_npu else " | —")
            + f" | {ratio} |"
        )

    front = pareto(npu_points)
    out += ["", "## Pareto front, NPU only (each model × configuration with measured AP)", "",
            "| Configuration | Latency (ms) | AP | Front |", "|---|---|---|---|"]
    for label, lat, ap in sorted(npu_points, key=lambda p: p[1]):
        out.append(f"| {label} | {lat:.1f} | {ap:.1f} | {'●' if label in front else ''} |")

    out += ["", "## Tiers: what OptiDet has to beat", "",
            "| Tier | AP target | Fastest reference (any placement) | Fastest reference (NPU only) | Target ÷4 | Target ÷5 |",
            "|---|---|---|---|---|---|"]
    for tier, target in TIERS.items():
        reaching = [(lat, label, ap) for label, lat, ap in all_points if ap >= target - TIER_TOLERANCE]
        npu_reaching = [(lat, label, ap) for label, lat, ap in npu_points if ap >= target - TIER_TOLERANCE]
        if not reaching:
            out.append(f"| {tier} | {target} | — | — | — | — |")
            continue
        lat, label, ap = min(reaching)
        npu_txt = "—"
        if npu_reaching:
            nlat, nlabel, nap = min(npu_reaching)
            npu_txt = f"{nlabel}: {nlat:.1f} ms, {nap:.1f} AP"
        out.append(f"| {tier} | {target} | {label}: {lat:.1f} ms, {ap:.1f} AP | {npu_txt} | {lat / 4:.1f} ms | {lat / 5:.1f} ms |")

    notes = [f"- **{name}** ({b.license}): {b.notes}" for name, b in BASELINES.items() if b.notes and name in table]
    if notes:
        out += ["", "## Notes", "", *notes]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    report = build()
    (WORK / "report.md").write_text(report, encoding="utf-8")
    print(report)
