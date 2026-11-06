"""P6 (review): NNCF's accuracy-aware quantization, nncf.quantize_with_accuracy_control, as a baseline for
OptiNPU's latency-guided mixed precision, on the same models, images and budget.

NNCF starts from full INT8 and returns the most harmful quantizers to float, one by one, until the metric
drop on the validation set is at most `max_drop`: a point estimate, judged on the CPU, blind to latency.
It gets what OptiNPU's search had:
- the graph the search ran on: the original, or for YOLO26 the head left once the final selection moved to
  the CPU (the tail stays on the CPU in FP32, inside the validation function, as in OptiNPU's state);
- the same 300 calibration images, decision images (val2017 [2500, 3500); ImageNetV2 [5000, 7000)) and
  budget (0.5 point, absolute), and NNCF's CNN scheme, as OptiNPU's mixes on these models.
NNCF runs in a watched child process (CPU only). Then, as for OptiNPU's deployments: latency on the NPU,
paired in one session with the state the search started from and OptiNPU's own deployment
(runs/p5/deploy/); Eq. (1) of the paper (20 or 10 folds of the decision images, against CPU FP32); and the
metric on the report images, on the NPU. Windows is kept awake and each model waits for mains power.
Results: runs/p6/qwac/<model>_<precision>.json (resumable) and runs/p6/qwac/summary.md.

    python -m benchmarks.p6.qwac [model:precision ...]
"""

from __future__ import annotations

import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from benchmarks.p2.optimize_model import splits
from benchmarks.p3.campaign import keep_awake, wait_for_mains
from optinpu.sandbox import run_watched

OUT = ROOT / "runs" / "p6" / "qwac"
DEPLOY = ROOT / "runs" / "p5" / "deploy"
MODELS = ("resnet50_cls224:fp32", "mobilenet_v2_cls224:fp32", "efficientnet_b0_cls224:fp32",
          "yolo26-n:fp16", "yolo26-s:fp16", "dfine-s:fp16")
BUDGET = 0.005  # 0.5 point of AP or top-1, as in the paper
SPLITS = Namespace(eval_images=1000, images=2500)
TIMEOUT_H = 6


def entry_of(name: str):
    return next(m for m in all_models() if m.name == name)


def child(spec: dict) -> dict:
    """Run NNCF's accuracy control; write qwac.xml (and head.xml / tail.xml for YOLO26) to spec["dir"]."""
    import nncf
    import openvino as ov

    from optinpu.data.imagenet import LabeledImages
    from optinpu.evaluate import _stats
    from optinpu.graph import split_from
    from optinpu.rewrites import selection_tail
    from optinpu.runtime import compile_model
    from optinpu.tasks import ClassificationTask

    out = Path(spec["dir"])
    entry = entry_of(spec["name"])
    task = entry.task
    model = ov.Core().read_model(entry.irs[spec["precision"]])
    tail = None
    found = selection_tail(model)
    if found is not None and found[2] == 0:  # a light selection tail: OptiNPU's search ran on the head
        head, tail = split_from(model, found[0])
        ov.save_model(head, out / "head.xml", compress_to_fp16=False)
        ov.save_model(tail, out / "tail.xml", compress_to_fp16=False)
        model = head
    guard, calibration, evaluation, folds, report_set, _ = splits(task, SPLITS)
    single = len(model.inputs) == 1
    to_input = (lambda s: task.inputs(s)[0][0]) if single else (lambda s: tuple(task.inputs(s)[0]))
    run_tail = compile_model(tail, "cpu", config={"INFERENCE_PRECISION_HINT": "f32"}, cache_dir=None) if tail else None
    calls = {"n": 0}

    def outputs(compiled, source) -> tuple[list[np.ndarray], object]:
        inputs, meta = task.inputs(source)
        result = compiled(inputs)
        values = [result[o] for o in compiled.outputs]
        return (run_tail(*values) if run_tail else values), meta

    if isinstance(task, ClassificationTask):
        assert isinstance(evaluation, LabeledImages)
        items = list(zip(evaluation.paths, evaluation.labels))
        transform = lambda item: to_input(item[0])

        def validate(compiled, data) -> tuple[float, list[float]]:
            calls["n"] += 1
            hits = [float(np.argmax(task.decode(outputs(compiled, path)[0])[0]) == label) for path, label in data]
            return float(np.mean(hits)), hits
    else:
        import contextlib
        import io

        from pycocotools.coco import COCO

        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(evaluation.annotations))
        items = list(evaluation.image_ids)
        transform = lambda image_id: to_input(evaluation.image_path(image_id))

        def validate(compiled, data) -> tuple[float, None]:
            calls["n"] += 1
            ids, results = list(data), []
            for image_id in ids:
                values, meta = outputs(compiled, evaluation.image_path(image_id))
                dets = task.decode(values, meta, 0.0)
                for (x0, y0, x1, y1), score, label in zip(dets.boxes, dets.scores, dets.labels):
                    results.append({"image_id": image_id, "category_id": evaluation.category_ids[int(label)],
                                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                                    "score": float(score)})
            return _stats(gt, results, ids)["AP"], None

    from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters

    detector = not isinstance(task, ClassificationTask)
    t0 = time.perf_counter()
    quantized = nncf.quantize_with_accuracy_control(
        model, nncf.Dataset(calibration, to_input), nncf.Dataset(items, transform), validate,
        max_drop=BUDGET, drop_type=nncf.DropType.ABSOLUTE, subset_size=len(calibration),
        # ranking on 100 images for detectors (each needs one CPU inference per quantizer group), and a
        # cap on the reverting iterations (each one validates on all 1000 images): about 6 h at most
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(
            max_num_iterations=40, ranking_subset_size=100 if detector else None))
    seconds = time.perf_counter() - t0
    ov.save_model(quantized, out / "qwac.xml", compress_to_fp16=True)
    fq = sum(op.get_type_name() == "FakeQuantize" for op in quantized.get_ordered_ops())
    return {"search_s": seconds, "validations": calls["n"], "fake_quantize": fq, "selection_to_cpu": tail is not None}


def run(spec_name: str) -> dict:
    from optinpu.optimizer import ap_within_budget, load_deployment
    from optinpu.profiler import Adaptive, measure, paired_saving
    from optinpu.sandbox import run_evaluation

    name, precision = spec_name.split(":")
    entry = entry_of(name)
    work = OUT / f"{name}_{precision}"
    work.mkdir(parents=True, exist_ok=True)
    spec_file = work / "spec.json"
    spec_file.write_text(json.dumps({"name": name, "precision": precision, "dir": str(work)}), encoding="utf-8")
    t0 = time.time()
    watched = run_watched([sys.executable, "-u", "-m", "benchmarks.p6.qwac", "--child", str(spec_file)],
                          TIMEOUT_H * 3600, cwd=ROOT, echo=True)
    result_file = work / "child.json"
    if not result_file.exists():
        tail = " / ".join(watched.lines[-5:])
        return {"model": name, "precision": precision, "error": f"{watched.status or watched.returncode}: {tail}"[-500:],
                "wall_s": time.time() - t0}
    record = {"model": name, "precision": precision, **json.loads(result_file.read_text())}
    task = entry.task
    original = entry.irs[precision]
    if record["selection_to_cpu"]:
        state = [(work / "head.xml", "npu", None), (work / "tail.xml", "cpu", None)]
        qwac = [(work / "qwac.xml", "npu", None), (work / "tail.xml", "cpu", None)]
    else:
        state = [(original, "npu", None)]
        qwac = [(work / "qwac.xml", "npu", None)]
    deploy = DEPLOY / f"{name}_{precision}"
    variants = {"state": state, "qwac": qwac}
    if (deploy / "deployment.json").exists():
        variants["optinpu"] = load_deployment(deploy)
    wait_for_mains()
    timings = measure(variants, "npu", adaptive=Adaptive("state"))
    record["latency_ms"] = {k: t.median_ms for k, t in timings.items()}
    record["saving_ms"] = {k: paired_saving(timings, "state", k) for k in variants if k != "state"}
    _, _, evaluation, folds, report_set, report_name = splits(task, SPLITS)
    reference, _ = run_evaluation([(original, "cpu", None)], task, evaluation, folds=folds)
    decided, job = run_evaluation(qwac, task, evaluation, folds=folds)
    if reference and decided:
        ok, mean, margin = ap_within_budget(decided["fold_scores"], reference["fold_scores"], BUDGET)
        record["decision"] = {"cpu": reference["score"], "qwac": decided["score"], "passes_eq1": ok,
                              "mean_points": 100 * mean, "lower_bound_points": 100 * (mean - margin)}
    else:
        record["decision"] = {"error": (job.error or "")[-300:]}
    reported, job = run_evaluation(qwac, task, report_set)
    record["report"] = {"set": report_name, "qwac_npu": reported["score"] if reported else None}
    record["wall_s"] = time.time() - t0
    return record


def summary(results: dict) -> str:
    lines = ["| Model | NNCF search | Validations | FQ left | State ms | NNCF ms | OptiNPU ms | Eq. (1) on decision "
             "images | Report metric (NPU) |", "|---|---|---|---|---|---|---|---|---|"]
    for spec, r in results.items():
        if "error" in r:
            lines.append(f"| {spec} | failed: {r['error'][:60]} | | | | | | | |")
            continue
        lat, d = r["latency_ms"], r["decision"]
        eq1 = (f"{d['mean_points']:+.2f} (bound {d['lower_bound_points']:+.2f}) {'pass' if d['passes_eq1'] else 'FAIL'}"
               if "passes_eq1" in d else "failed")
        rep = r["report"]["qwac_npu"]
        rep = f"{100 * rep:.1f}" if rep is not None else "failed"
        lines.append(f"| {spec} | {r['search_s'] / 60:.0f} min | {r['validations']} | {r['fake_quantize']} | "
                     f"{lat['state']:.2f} | {lat['qwac']:.2f} | {lat.get('optinpu', float('nan')):.2f} | {eq1} | {rep} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        result = child(spec)
        (Path(spec["dir"]) / "child.json").write_text(json.dumps(result, indent=1))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    keep_awake(True)
    try:
        for spec in sys.argv[1:] or MODELS:
            name, precision = spec.split(":")
            path = OUT / f"{name}_{precision}.json"
            if path.exists():
                print(f"{spec}: already done ({path})", flush=True)
                continue
            print(f"===== {spec}", flush=True)
            path.write_text(json.dumps(run(spec), indent=1, default=str))
    finally:
        keep_awake(False)
    results = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}
    text = summary(results)
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
