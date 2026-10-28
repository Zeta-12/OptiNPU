"""Summarize runs/m1/results.jsonl into runs/m1/report.md (and stdout)."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from benchmarks.m1.job import RESULTS, WORK

EXACT = 0.05  # max relative output error below which a configuration is considered exact


def load() -> dict:
    records = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "results" in r:
            records[r["name"]] = r
    crashed = {json.loads(line)["name"] for line in RESULTS.read_text(encoding="utf-8").splitlines()} - set(records)
    return records, crashed


def ms(r, key):
    return r["results"].get(key, {}).get("ms")


def tops(r, key):
    t = ms(r, key)
    return None if t is None or not r["flops"] else r["flops"] / (t * 1e-3) / 1e12


def fmt(v, spec=".2f"):
    return "—" if v is None else format(v, spec)


def exact_mark(r, key="fp16-2t"):
    err = r["results"].get(key, {}).get("err_vs_1t")
    return "" if err is None or err <= EXACT else f" ⚠{err:.2f}"


def s1(records) -> list[str]:
    out = ["## S1 — fixed cost per op", "",
           "Linear fit of latency vs number of chained ops: slope = cost per op, intercept = fixed cost of a graph.", "",
           "| Op | Tensor | Config | µs per op | Fixed (ms) |", "|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in records.values():
        if r["suite"] == "S1":
            p = r["params"]
            groups[(p["kind"], p["c"], p["hw"])].append(r)
    for (kind, c, hw), rs in sorted(groups.items()):
        for key in ("fp16-1t", "fp16-2t"):
            pts = [(r["params"]["n"], ms(r, key)) for r in rs if ms(r, key) is not None]
            if len(pts) < 3:
                continue
            n, t = np.array(pts, dtype=float).T
            slope, intercept = np.polyfit(n, t, 1)
            out.append(f"| {kind} | {c}×{hw}×{hw} | {key} | {1e3 * slope:.0f} | {intercept:.2f} |")
    return out


def s2(records) -> list[str]:
    out = ["## S2 — convolution throughput", "",
           "Effective TOPS (conv FLOPs / latency) and INT8 speed-up (FP16 2t latency / INT8 2t latency).", ""]
    rows = [r for r in records.values() if r["suite"] == "S2" and r["params"]["op"] != "conv3x3_align"]
    res = sorted({r["params"]["hw"] for r in rows}, reverse=True)
    for op in ("conv3x3", "conv1x1", "dwconv3x3", "conv3x3_s2"):
        sub = {(r["params"]["c"], r["params"]["hw"]): r for r in rows if r["params"]["op"] == op}
        if not sub:
            continue
        chans = sorted({c for c, _ in sub})
        out += [f"### {op}", "",
                "| C | " + " | ".join(f"{h}² FP16 2t TOPS / ms" for h in res) + " | " + " | ".join(f"{h}² INT8 ×" for h in res) + " |",
                "|---|" + "---|" * (2 * len(res))]
        for c in chans:
            cells = []
            for h in res:
                r = sub.get((c, h))
                cells.append("—" if r is None else f"{fmt(tops(r, 'fp16-2t'))} / {fmt(ms(r, 'fp16-2t'))}{exact_mark(r)}")
            speed = []
            for h in res:
                r = sub.get((c, h))
                a, b = (ms(r, "fp16-2t"), ms(r, "int8-2t")) if r else (None, None)
                speed.append("—" if not a or not b else f"{a / b:.2f}")
            out.append(f"| {c} | " + " | ".join(cells) + " | " + " | ".join(speed) + " |")
        out.append("")
    align = sorted((r for r in records.values() if r["params"].get("op") == "conv3x3_align"), key=lambda r: r["params"]["c"])
    if align:
        out += ["### Channel alignment (4 × conv3x3, 80×80)", "", "| C | FP16 2t ms | INT8 2t ms | FP16 TOPS |", "|---|---|---|---|"]
        out += [f"| {r['params']['c']} | {fmt(ms(r, 'fp16-2t'))} | {fmt(ms(r, 'int8-2t'))} | {fmt(tops(r, 'fp16-2t'))} |" for r in align]
    return out


def table(records, suite, label_key, base=None, title="") -> list[str]:
    rows = [r for r in records.values() if r["suite"] == suite]
    ref = records.get(base) if base else None
    out = [title, "", "| Variant | FP16 1t ms | FP16 2t ms | INT8 2t ms | Δ FP16 2t vs ref | 2t exact | FP16 err vs CPU |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        delta = None
        if ref and ms(r, "fp16-2t") is not None and ms(ref, "fp16-2t") is not None:
            delta = ms(r, "fp16-2t") - ms(ref, "fp16-2t")
        err2 = r["results"].get("fp16-2t", {}).get("err_vs_1t")
        out.append(f"| {r['params'][label_key]} | {fmt(ms(r, 'fp16-1t'))} | {fmt(ms(r, 'fp16-2t'))} | {fmt(ms(r, 'int8-2t'))} "
                   f"| {fmt(delta, '+.2f')} | {'—' if err2 is None else ('yes' if err2 <= EXACT else f'NO ({err2:.2f})')} "
                   f"| {fmt(r['results'].get('fp16-1t', {}).get('err_vs_cpu'), '.3f')} |")
    return out


def s5(records) -> list[str]:
    out = ["## S5 — global ops (tokens = H×W of a feature map)", "",
           "| Op | Tokens | Dim | FP16 1t ms | FP16 2t ms | INT8 2t ms | 2t exact (FP16 / INT8) | FP16 err vs CPU |",
           "|---|---|---|---|---|---|---|---|"]
    rows = sorted((r for r in records.values() if r["suite"] == "S5"),
                  key=lambda r: (r["params"]["op"], r["params"].get("tokens", r["params"].get("n")), r["params"].get("dim", 0)))
    for r in rows:
        p = r["params"]
        ex = []
        for key in ("fp16-2t", "int8-2t"):
            err = r["results"].get(key, {}).get("err_vs_1t")
            ex.append("—" if err is None else ("yes" if err <= EXACT else f"NO {err:.2f}"))
        tokens = p.get("tokens", f"N={p.get('n')}")
        dim = p.get("dim", f"k={p.get('k')}")
        out.append(f"| {p['op']} | {tokens} | {dim} | {fmt(ms(r, 'fp16-1t'))} | {fmt(ms(r, 'fp16-2t'))} | {fmt(ms(r, 'int8-2t'))} "
                   f"| {' / '.join(ex)} | {fmt(r['results'].get('fp16-1t', {}).get('err_vs_cpu'), '.3f')} |")
    return out


def s6(records) -> list[str]:
    out = ["## S6 — backbones (feature extractor only)", "",
           "| Network | Res. | GFLOPs | FP16 1t ms | FP16 2t ms | INT8 1t ms | INT8 2t ms | Best TOPS | INT8 × | 2t exact |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    rows = sorted((r for r in records.values() if r["suite"] == "S6"), key=lambda r: (r["params"]["net"], r["params"]["res"]))
    for r in rows:
        p = r["params"]
        times = [ms(r, k) for k in ("fp16-1t", "fp16-2t", "int8-1t", "int8-2t")]
        best = min((t for t in times if t), default=None)
        speed = times[1] / times[3] if times[1] and times[3] else None
        err2 = r["results"].get("fp16-2t", {}).get("err_vs_1t")
        out.append(f"| {p['net']} | {p['res']} | {r['flops'] / 1e9:.1f} | " + " | ".join(fmt(t) for t in times)
                   + f" | {fmt(best and r['flops'] / (best * 1e-3) / 1e12)} | {fmt(speed)} "
                   f"| {'—' if err2 is None else ('yes' if err2 <= EXACT else f'NO ({err2:.2f})')} |")
    return out


def s7(records) -> list[str]:
    out = ["## S7 — marginal cost of one layer inside a network", "",
           "(t(10 layers) − t(2 layers)) / 8: no input / output transfer. TOPS = layer FLOPs / marginal time.", ""]
    pairs = defaultdict(dict)
    for r in records.values():
        if r["suite"] == "S7":
            p = r["params"]
            pairs[(p["op"], p["c"], p["hw"])][p["n"]] = r
    res = sorted({k[2] for k in pairs}, reverse=True)
    for op in ("conv3x3", "conv3x3_silu", "conv1x1", "dwconv3x3", "dwsep3x3"):
        chans = sorted({c for o, c, _ in pairs if o == op})
        if not chans:
            continue
        out += [f"### {op}", "", "| C | " + " | ".join(f"{h}² µs FP16 / INT8 (TOPS FP16)" for h in res) + " |",
                "|---|" + "---|" * len(res)]
        for c in chans:
            cells = []
            for h in res:
                pair = pairs.get((op, c, h), {})
                if 2 not in pair or 10 not in pair:
                    cells.append("—")
                    continue
                per = {}
                for key in ("fp16-2t", "int8-2t"):
                    a, b = ms(pair[2], key), ms(pair[10], key)
                    per[key] = None if a is None or b is None else (b - a) / 8
                layer_flops = (pair[10]["flops"] - pair[2]["flops"]) / 8
                t16 = per["fp16-2t"]
                tops16 = layer_flops / (t16 * 1e-3) / 1e12 if t16 and t16 > 0 else None
                cells.append(f"{fmt(t16 and 1e3 * t16, '.0f')} / {fmt(per['int8-2t'] and 1e3 * per['int8-2t'], '.0f')} "
                             f"({fmt(tops16)})")
            out.append(f"| {c} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def build() -> str:
    records, crashed = load()
    out = ["# M1 — NPU building blocks (Core Ultra 5 235U, NPU 3720, batch 1)", "",
           "Median latency in ms; `1t` = 1 NPU tile, `2t` = compiler default (2 tiles). Random weights and inputs. "
           f"2t is 'exact' when its outputs differ from 1t by at most {EXACT} (max relative error).", ""]
    if crashed:
        out += [f"Blocks that crashed the job (watchdog or NPU device lost): {', '.join(sorted(crashed))}", ""]
    out += s1(records) + [""] + s2(records) + [""]
    out += table(records, "S3", "variant", "act_none", "## S3 — activations and structures (4 × conv3x3, 64 ch, 80×80; ref = no activation)")
    out += [""] + table(records, "S4", "op", "identity",
                        "## S4 — resampling, merging, layout (conv1x1 → op → conv1x1, 64 ch; ref = identity at 80×80)")
    out += [""] + s5(records) + [""] + s6(records) + [""] + s7(records)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    text = build()
    (WORK / "report.md").write_text(text, encoding="utf-8")
    print(text)
