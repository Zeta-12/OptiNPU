"""P5: the NPU-only detector deployments of P3.5, rebuilt from their recorded changes, and their AP on
val2017 [0, 2500) instead of [0, 500).

The P3.5 bench kept each optimization's report (runs/p3/optimize_<model>_<precision>.json), not its
deployment. Each one is rebuilt with optinpu.optimizer.replay (same IR, same 300 calibration images
val2017 [4500, 4800), quantized in order by NNCF: same INT8 graphs), saved to
runs/p5/deploy/<model>_<precision>/ and reloaded, then checked: its AP on val2017 [0, 500) must be the
one the bench recorded. The CPU FP32 original, the original on the NPU and the deployment are then
evaluated on val2017 [0, 2500), on which nothing was decided ([3500, 4500) stays free, e.g. to confirm a
mixed-precision choice later). Latencies are those of the bench. Results go to runs/p5/report_ap.json
and report_ap.md; models already there are skipped (resumable).

    python -m benchmarks.p5.deployments                       # the 7 detectors below
    python -m benchmarks.p5.deployments dfine-s:fp16 ...      # or given ones
"""

from __future__ import annotations

import json
import sys
import time

import openvino as ov

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from benchmarks.p3.campaign import keep_awake, stamp
from optinpu.data import coco_val_subset
from optinpu.optimizer import load_deployment, replay
from optinpu.sandbox import run_evaluation

MODELS = ("yolo26-n:fp16", "yolo26-s:fp16", "dfine-n:fp16", "dfine-s:fp16", "deimv2-n:fp16",
          "rtdetrv2-r18:fp16", "rfdetr-n:fp16")
P3, OUT = ROOT / "runs" / "p3", ROOT / "runs" / "p5"
CHECK, REPORT = 500, 2500  # val2017 [0, CHECK): recorded by the bench; [0, REPORT): the new report set


def evaluate(parts, task, subset) -> float | str:
    result, job = run_evaluation(parts, task, subset)
    return result["score"] if result else f"failed: {(job.error or '')[-100:]}"


def rebuild(spec: str) -> dict:
    name, precision = spec.split(":")
    record = json.loads((P3 / f"optimize_{name}_{precision}.json").read_text())
    entry = next(m for m in all_models() if m.name == name)
    model = ov.Core().read_model(entry.irs[precision])
    changes = [] if record["applied"] == "original" else record["applied"].split(" + ")
    state = replay(model, changes, entry.task, coco_val_subset(300, offset=4500).image_paths())
    parts = load_deployment(state.save(OUT / "deploy" / f"{name}_{precision}"))
    recorded, rebuilt = record["val_ap"]["final"], evaluate(parts, entry.task, coco_val_subset(CHECK))
    same = isinstance(rebuilt, float) and abs(rebuilt - recorded) < 1e-4
    print(f"{stamp()} {spec}: {record['applied']}; AP on [0, {CHECK}) recorded {recorded:.4f}, rebuilt {rebuilt}"
          f"{'' if same else '  MISMATCH'}", flush=True)
    report_set = coco_val_subset(REPORT)
    ap = {label: evaluate(p, entry.task, report_set)
          for label, p in (("cpu", [(model, "cpu", None)]), ("baseline", [(model, "npu", None)]), ("final", parts))}
    print(f"{stamp()} {spec}: AP on [0, {REPORT}) {ap}", flush=True)
    return {"model": name, "precision": precision, "applied": record["applied"],
            "baseline_ms": record["baseline_ms"], "final_ms": record["final_ms"],
            "check": {"set": f"val2017 [0, {CHECK})", "recorded": recorded, "rebuilt": rebuilt, "same": same},
            "report_set": f"val2017 [0, {REPORT})", "ap": ap}


def summary(results: dict) -> str:
    fmt = lambda v: f"{100 * v:.1f}" if isinstance(v, float) else "failed"
    lines = [f"| Model | Before | After | Applied | Rebuilt = recorded on [0, {CHECK}) | "
             f"AP [0, {REPORT}): CPU / before / after |", "|---|---|---|---|---|---|"]
    for r in results.values():
        check = r["check"]
        lines.append(f"| {r['model']} | {r['baseline_ms']:.2f} ms | {r['final_ms']:.2f} ms | {r['applied']} | "
                     f"{'yes' if check['same'] else 'NO'} ({fmt(check['recorded'])} / {fmt(check['rebuilt'])}) | "
                     + " / ".join(fmt(r["ap"][k]) for k in ("cpu", "baseline", "final")) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "report_ap.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    keep_awake(True)
    try:
        for spec in sys.argv[1:] or MODELS:
            if spec in results:
                print(f"{spec}: already done", flush=True)
                continue
            t0 = time.time()
            results[spec] = rebuild(spec)
            path.write_text(json.dumps(results, indent=1, default=str))
            print(f"{stamp()} {spec}: done in {(time.time() - t0) / 60:.0f} min", flush=True)
    finally:
        keep_awake(False)
    text = summary(results)
    (OUT / "report_ap.md").write_text(text, encoding="utf-8")
    print(f"\n{text}", flush=True)


if __name__ == "__main__":
    main()
