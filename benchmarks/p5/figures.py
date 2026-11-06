"""P5: the result figure of the paper, written as pgfplots code (paper/fig_power.tex) from the results.

fig_power.tex: package power while inferring back to back, per model and deployment (runs/p5/energy.json;
the NPU deployment is the optimized one when something was applied).
With --p6: runs/p6/energy.json (20 s windows, the NPU deployment confirmed on held-out images), the iGPU
deployment optimized when the iGPU run applied something; the turbo variant is left out.

Colors: the first four slots of the dataviz categorical palette (validated for color vision
deficiencies); every series also has its own marker shape, for grayscale print.

    python -m benchmarks.p5.figures [--p6]
"""

from __future__ import annotations

import json
import sys

from benchmarks.m0.baselines import ROOT

PAPER = ROOT / "paper"
COLORS = {"npu": "2A78D6", "npu+gpu": "EB6834", "gpu": "1BAF7A", "cpu": "EDA100"}
MARKS = {"npu": "*", "npu+gpu": "diamond*", "gpu": "square*", "cpu": "triangle*"}
NAMES = {"npu": "NPU", "npu+gpu": "NPU+iGPU", "gpu": "iGPU", "cpu": "CPU"}
MODEL_NAMES = {"yolo26-n": "YOLO26-N", "yolo26-s": "YOLO26-S", "dfine-n": "D-FINE-N", "dfine-s": "D-FINE-S",
               "deimv2-n": "DEIMv2-N", "rtdetrv2-r18": "RT-DETRv2-R18", "rfdetr-n": "RF-DETR-N",
               "resnet50_cls224": "ResNet-50", "mobilenet_v2_cls224": "MobileNetV2", "efficientnet_b0_cls224": "EfficientNet-B0"}


def colors() -> str:
    return "".join(f"\\definecolor{{c{k.replace('+', '')}}}{{HTML}}{{{v}}}\n" for k, v in COLORS.items())


def power_figure(p6: bool = False) -> str:
    """Dot plot: one row per model (top to bottom in the paper's order), package power on a linear axis."""
    results = json.loads((ROOT / "runs" / ("p6" if p6 else "p5") / "energy.json").read_text())
    rows = []
    for spec, done in results.items():
        model = {}
        for label, r in done.items():
            if "error" in r or label == "npu-turbo" or (label == "npu" and "npu-optimized" in done) or \
                    (label == "gpu" and "gpu-optimized" in done):
                continue  # the NPU (iGPU) deployment is the optimized one when something was applied
            model[{"npu-optimized": "npu", "gpu-optimized": "gpu"}.get(label, label)] = r["active_w"]["package"]
        rows.append((MODEL_NAMES[spec.split(":")[0]], model))
    n = len(rows)
    y = {name: n - i for i, (name, _) in enumerate(rows)}
    labels = ",".join(name for name, _ in reversed(rows))
    xmax = 25 if p6 else 28
    out = [colors(), r"\begin{tikzpicture}",
           f"\\begin{{axis}}[scale only axis, width=0.72\\columnwidth, height=3.6cm, xmin=0, xmax={xmax},"
           f" ymin=0.4, ymax={n + 0.6}, xtick pos=bottom, xtick={{0,5,...,25}},",
           f"  ytick={{1,...,{n}}}, yticklabels={{{labels}}}, xlabel={{Package power while inferring (W)}},",
           r"  tick label style={font=\scriptsize}, label style={font=\footnotesize}, axis line style={black!60},",
           r"  tick style={black!60}, ytick style={draw=none}, xmajorgrids, major grid style={black!12},",
           r"  legend style={font=\scriptsize, draw=none, fill=none, at={(0.5,1.02)}, anchor=south, legend columns=4,",
           r"  /tikz/every even column/.append style={column sep=6pt}}, legend cell align=left]"]
    for name, model in rows:  # range of each model, recessive
        out.append(f"\\addplot[black!25, line width=0.8pt, forget plot] coordinates "
                   f"{{({min(model.values()):.3g},{y[name]}) ({max(model.values()):.3g},{y[name]})}};")
    for key in COLORS:
        c = f"c{key.replace('+', '')}"
        pts = " ".join(f"({model[key]:.3g},{y[name]})" for name, model in rows if key in model)
        out.append(f"\\addplot[only marks, mark={MARKS[key]}, mark size=2.2pt, color={c}, "
                   f"mark options={{draw=white, line width=0.3pt}}] coordinates {{{pts}}};")
        out.append(f"\\addlegendentry{{{NAMES[key]}}}")
    out += [r"\end{axis}", r"\end{tikzpicture}", ""]
    return "\n".join(out)


def main() -> None:
    (PAPER / "fig_power.tex").write_text(power_figure("--p6" in sys.argv[1:]), encoding="utf-8")
    print(f"wrote {PAPER / 'fig_power.tex'}")


if __name__ == "__main__":
    main()
