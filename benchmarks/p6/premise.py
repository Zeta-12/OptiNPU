"""P6 (review): the premise (latency does not depend on dense weight values) measured again, with paired
adaptive rounds and 95 % confidence intervals (P1.0 gave medians only), and the questions of the review:
- real images: every timing uses one fixed noise input; here the same models also run on 30 real images
  (preprocessed val2017 or ImageNetV2 images, in turn);
- YOLO26-N without its final selection (the NPU part of the optimized deployment): with no in-graph TopK,
  does the premise hold for it too;
- INT8 zeros: the share of INT8 weights that are exactly zero (every INT8 IR of the paper), and INT8 models
  whose zero weights are set to 1 ("densified"): do these zeros change latency;
- activation sparsity: models with ReLU (ResNet-50 natively, YOLO26-N's head with SiLU -> ReLU) with dense
  non-positive weights and negative biases ("dead"), so that every activation after the first layer is zero,
  against the trained and random weights: does the NPU skip zero activations, as it skips zero weights?
One paired session per model: the original on the noise input (reference), on real images, on all-zero
inputs, random dense weights, 50 % and 90 % zero weights, and the extra variants above. Each model waits
for mains power; Windows is kept awake.
Results: runs/p6/premise.json (resumable, per model) and runs/p6/premise.md.

    python -m benchmarks.p6.premise
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from benchmarks.p1.validate import ACTIVATIONS
from benchmarks.p3.campaign import keep_awake, wait_for_mains
from optinpu.data import coco_val_subset, imagenetv2_subset
from optinpu.graph import split_from
from optinpu.profiler import Adaptive, measure, paired_saving, weight_constants, with_weights
from optinpu.rewrites import activations_to_relu, selection_tail
from optinpu.tasks import ClassificationTask, OutputTask

OUT = ROOT / "runs" / "p6"
DEPLOY = ROOT / "runs" / "p5" / "deploy"
COUNT = 30  # real images per model

# label: (model name or deployment folder, precision, graph transform, extra variants)
CASES = {
    "ResNet-50 640": ("resnet50_640", "fp32", None, ("dead",)),
    "MobileNetV2 640": ("mobilenet_v2_640", "fp32", None, ()),
    "YOLO26-N FP16": ("yolo26-n", "fp16", None, ()),
    "YOLO26-N FP16, head": ("yolo26-n", "fp16", "head", ()),
    "YOLO26-N FP16, head, ReLU": ("yolo26-n", "fp16", "head-relu", ("dead",)),
    "YOLO26-N INT8": ("yolo26-n", "int8", None, ("densified",)),
    "D-FINE-N FP16": ("dfine-n", "fp16", None, ()),
    "ResNet-50 mixed INT8 (deployed)": ("deploy:resnet50_cls224_fp32", None, None, ("densified",)),
}


def bench(name: str):
    return next(m for m in all_models() if m.name == name)


def load(case: tuple) -> tuple[ov.Model, object]:
    """(graph, task giving real-image inputs)."""
    name, precision, transform, _ = case
    if name.startswith("deploy:"):
        folder = name.split(":", 1)[1]
        return ov.Core().read_model(DEPLOY / folder / "part0.xml"), bench(folder.rsplit("_", 1)[0]).task
    entry = bench(name)
    model = ov.Core().read_model(entry.irs[precision])
    task = entry.task
    if isinstance(task, OutputTask):
        size = tuple(model.inputs[0].get_partial_shape().to_shape())[-2:]  # torchvision CNN at 640 px
        task = ClassificationTask(size, size[0])
    if transform in ("head", "head-relu"):
        model = split_from(model, selection_tail(model)[0])[0]
    if transform == "head-relu":
        model = activations_to_relu(model, ACTIVATIONS)[0]
    return model, task


def real_images(task, path: Path) -> str:
    """Save COUNT preprocessed real images (guard images of the paper) as timing samples."""
    if isinstance(task, ClassificationTask):
        sources = imagenetv2_subset(COUNT, 7300).image_paths()
    else:
        sources = coco_val_subset(COUNT, offset=4970).image_paths()
    arrays = {f"s{i}_{j}": x for i, s in enumerate(sources) for j, x in enumerate(task.inputs(s)[0])}
    np.savez(path, **arrays)
    return f"file:{path}"


def densified(model: ov.Model) -> ov.Model:
    """INT8 weights that are exactly zero set to 1 (the other values kept)."""
    clone = model.clone()
    for const in weight_constants(clone):
        data = const.get_data()
        if np.issubdtype(data.dtype, np.integer):
            new = ops.constant(np.where(data == 0, 1, data).astype(data.dtype), const.get_output_element_type(0))
            for target in list(const.output(0).get_target_inputs()):
                target.replace_source_output(new.output(0))
    return ov.Model(clone.get_results(), clone.get_parameters(), "densified")


def dead(model: ov.Model) -> ov.Model:
    """Weights -|w| (dense, as many zeros as the original) and biases -|b| - 1: with non-negative inputs,
    every ReLU outputs zero from the first layer on."""
    clone = model.clone()
    replace = {}
    for const in weight_constants(clone):
        data = const.get_data()
        if np.issubdtype(data.dtype, np.floating):
            replace[const] = -np.abs(data)
    heavy = {"Convolution", "GroupConvolution", "MatMul"}
    for op in clone.get_ordered_ops():
        if op.get_type_name() != "Add":
            continue
        sources = [op.input(i).get_source_output().get_node() for i in range(2)]
        if sources[0].get_type_name() in heavy:
            node = sources[1]
            while node.get_type_name() == "Convert":
                node = node.input(0).get_source_output().get_node()
            if node.get_type_name() == "Constant" and np.issubdtype(node.get_data().dtype, np.floating):
                replace[node] = -np.abs(node.get_data()) - 1
    for const, data in replace.items():
        new = ops.constant(data.astype(const.get_data().dtype), const.get_output_element_type(0))
        for target in list(const.output(0).get_target_inputs()):
            target.replace_source_output(new.output(0))
    return ov.Model(clone.get_results(), clone.get_parameters(), "dead")


def zero_share(model: ov.Model) -> float | None:
    """Share of integer (INT8) weights equal to zero, or None without integer weights."""
    total = zeros = 0
    for const in weight_constants(model):
        data = const.get_data()
        if np.issubdtype(data.dtype, np.integer):
            total, zeros = total + data.size, zeros + int((data == 0).sum())
    return zeros / total if total else None


def run_case(case: tuple) -> dict:
    model, task = load(case)
    with tempfile.TemporaryDirectory(prefix="optinpu-premise-") as tmp:
        variants = {"original": model, "real images": model, "zero inputs": model,
                    "random weights": with_weights(model, "random"), "50 % zero weights": with_weights(model, "sparse50"),
                    "90 % zero weights": with_weights(model, "sparse90")}
        if "densified" in case[3]:
            variants["INT8 zeros set to 1"] = densified(model)
        if "dead" in case[3]:
            variants["all activations zero"] = dead(model)
        modes = {"real images": real_images(task, Path(tmp) / "images.npz"), "zero inputs": "zeros"}
        timings = measure(variants, "npu", input_modes=modes, adaptive=Adaptive("original"), timeout=3600)
    base = timings["original"].median_ms
    out = {"latency_ms": base, "rounds": len(timings["original"].rounds), "zero_share_int8": zero_share(model)}
    for name in variants:
        if name == "original":
            continue
        if timings[name].error is not None:
            out[name] = {"error": timings[name].error[-200:]}
            continue
        saving, ci = paired_saving(timings, "original", name)
        out[name] = {"change_pct": -100 * saving / base, "ci_pct": 100 * ci / base, "median_ms": timings[name].median_ms}
    return out


def table(results: dict) -> str:
    names = ["real images", "zero inputs", "random weights", "50 % zero weights", "90 % zero weights",
             "INT8 zeros set to 1", "all activations zero"]
    lines = ["| Model | ms | " + " | ".join(names) + " | INT8 zeros |", "|---|---|" + "---|" * (len(names) + 1)]
    for label, r in results.items():
        if not isinstance(r, dict) or "latency_ms" not in r:
            continue
        cells = []
        for n in names:
            v = r.get(n)
            cells.append("" if v is None else ("failed" if "error" in v else f"{v['change_pct']:+.1f} ± {v['ci_pct']:.1f} %"))
        z = r.get("zero_share_int8")
        lines.append(f"| {label} | {r['latency_ms']:.2f} | " + " | ".join(cells) + f" | {'' if z is None else f'{100 * z:.1f} %'} |")
    shares = results.get("int8 zero share of the deployments", {})
    if shares:
        lines += ["", "Share of INT8 weights exactly zero: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in shares.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "premise.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    shares = {}
    for folder in sorted(DEPLOY.glob("*/part0.xml")):
        share = zero_share(ov.Core().read_model(folder))
        if share is not None:
            shares[folder.parent.name] = share
    results["int8 zero share of the deployments"] = shares
    keep_awake(True)
    try:
        for label, case in CASES.items():
            if label in results:
                print(f"{label}: already done", flush=True)
                continue
            wait_for_mains()
            print(f"===== {label}", flush=True)
            results[label] = run_case(case)
            print(json.dumps(results[label], indent=1), flush=True)
            path.write_text(json.dumps(results, indent=1))
    finally:
        keep_awake(False)
    path.write_text(json.dumps(results, indent=1))
    text = table(results)
    (OUT / "premise.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
