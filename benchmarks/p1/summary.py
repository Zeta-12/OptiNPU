"""P1 validation summary over runs/p1/validate/*.json (see validate.py).

Truth is the mean of the two measured runs. "Significant" interventions save more than zero with 95 %
confidence (adaptive runs: CI of each saving, from paired rounds); for the first campaign (fixed
rounds, no CI), more than twice the run-to-run noise, the median |run1 - run2| of the model.
For each predictor: Spearman rank correlation, mean absolute error (% of latency), top-3 overlap, and
the regret of acting on its top pick: (best true saving - true saving of the predicted best) / best.
"Measured (repeat)" compares run 2 with run 1: the best our own measurement can do at that noise.

    python -m benchmarks.p1.summary [results folder, default runs/p1/validate]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from benchmarks.p1.validate import OUT

PREDICTORS = ("vendor raw", "vendor scaled", "additive", "measured (repeat)")


def scores(truth: dict, predicted: dict, latency: float) -> dict:
    keys = [k for k in truth if np.isfinite(truth[k]) and np.isfinite(predicted.get(k, np.nan))]
    if len(keys) < 3:
        return {"n": len(keys)}
    t = np.array([truth[k] for k in keys])
    p = np.array([predicted[k] for k in keys])
    best_true = max(t.max(), 1e-9)
    pick = int(np.argmax(p))
    top = lambda a: set(np.argsort(-a)[:3])
    return {"n": len(keys), "spearman": float(spearmanr(t, p).statistic),
            "mae_pct": float(100 * np.mean(np.abs(t - p)) / latency), "top3": len(top(t) & top(p)) / 3,
            "regret": float((best_true - t[pick]) / best_true)}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    rows = []
    for path in sorted(out.glob("*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text())
        items = data["interventions"]
        latency = data["latency_ms"]
        run1 = {k: v["run1"] for k, v in items.items()}
        noise = float(np.nanmedian([abs(v["run1"] - v["run2"]) for v in items.values()]))
        truth = {k: v["mean"] for k, v in items.items()}
        if all("ci1" in v for v in items.values()):  # adaptive runs: 95 % CI of the mean of the two runs
            ci = {k: float(np.hypot(v["ci1"], v["ci2"]) / 2) for k, v in items.items()}
            significant = {k: s for k, s in truth.items() if np.isfinite(s) and s - ci[k] > 0}
        else:  # first campaign (fixed 5 rounds): twice the run-to-run noise
            significant = {k: s for k, s in truth.items() if np.isfinite(s) and s > 2 * noise}
        for subset_name, subset in (("all", truth), ("significant", significant)):
            for predictor in PREDICTORS:
                if predictor == "measured (repeat)":
                    reference = {k: run1[k] for k in subset}
                    predicted = {k: items[k]["run2"] for k in subset}
                else:
                    reference = subset
                    predicted = {k: items[k][predictor] for k in subset}
                rows.append({"model": f"{data['model']} {data['precision']}", "latency": latency, "noise": noise,
                             "subset": subset_name, "predictor": predictor,
                             **scores(reference, predicted, latency)})

    print(f"{'model':<26} {'lat ms':>7} {'noise':>6} {'subset':<11} {'predictor':<18} {'n':>3} {'rho':>6} "
          f"{'MAE %':>6} {'top3':>5} {'regret':>6}")
    for r in rows:
        if "spearman" not in r:
            continue
        print(f"{r['model']:<26} {r['latency']:7.2f} {r['noise']:6.2f} {r['subset']:<11} {r['predictor']:<18} "
              f"{r['n']:3d} {r['spearman']:+6.2f} {r['mae_pct']:6.2f} {r['top3']:5.2f} {r['regret']:6.2f}")
    print("\nmean over models (significant interventions):")
    for predictor in PREDICTORS:
        sel = [r for r in rows if r["subset"] == "significant" and r["predictor"] == predictor and "spearman" in r]
        if sel:
            print(f"  {predictor:<18} rho {np.mean([r['spearman'] for r in sel]):+.2f}  "
                  f"MAE {np.mean([r['mae_pct'] for r in sel]):.2f} % of latency  "
                  f"top-3 {np.mean([r['top3'] for r in sel]):.2f}  regret {np.mean([r['regret'] for r in sel]):.2f}  "
                  f"({len(sel)} models)")
    (out / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
