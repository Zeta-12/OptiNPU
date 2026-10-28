"""OptiDet candidates latency table (results of `python -m benchmarks.m1.run M2`)."""

from __future__ import annotations

import json

from benchmarks.m1.job import RESULTS

EXACT = 0.05


def build() -> str:
    rows = []
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("suite") == "M2" and "results" in r:
            rows.append(r)
    rows.sort(key=lambda r: (r["params"]["res"], r["results"].get("int8-2t", {}).get("ms", 1e9)))
    out = ["| Candidate | Res. | GFLOPs | FP16 1t | FP16 2t | INT8 1t | INT8 2t | 2t exact (FP16/INT8) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p, res = r["params"], r["results"]
        lat = [res.get(k, {}).get("ms") for k in ("fp16-1t", "fp16-2t", "int8-1t", "int8-2t")]
        exact = []
        for k in ("fp16-2t", "int8-2t"):
            err = res.get(k, {}).get("err_vs_1t")
            exact.append("—" if err is None else ("yes" if err <= EXACT else f"NO {err:.2f}"))
        out.append(f"| {p['backbone']} n{p['neck']} | {p['res']} | {r['flops'] / 1e9:.1f} | "
                   + " | ".join("—" if t is None else f"{t:.2f}" for t in lat) + f" | {' / '.join(exact)} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(build())
