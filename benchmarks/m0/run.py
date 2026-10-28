"""Run M0 jobs in isolated processes with a memory / time watchdog; results go to runs/m0/results.jsonl.

    python -m benchmarks.m0.run --models dfine-n dfine-s --kinds latency lint ap --devices cpu gpu npu

The NPU compiler runs inside the calling process and can grab tens of GB on some graphs (PLAN.md,
L5 incident), so every job is killed past --max-gb of committed memory, when the system runs low on
memory, or past its timeout. Jobs already in results.jsonl with status "ok" are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from benchmarks.m0.baselines import BASELINES, WORK
from benchmarks.watchdog import run_watched

RESULTS = WORK / "results.jsonl"
TIMEOUTS = {"latency": 900, "lint": 900, "ap": 3 * 3600, "quantize": 3600}
LINT_DEVICES = {"npu-2t"}  # lint retries with 1 tile by itself


def run_job(spec: dict, max_gb: float, min_free_gb: float) -> dict:
    cmd = [sys.executable, "-u", "-W", "ignore", "-m", "benchmarks.m0.job", json.dumps(spec)]
    w = run_watched(cmd, TIMEOUTS[spec["kind"]], max_gb, min_free_gb)
    record = {**spec, "time": datetime.now().isoformat(timespec="seconds"),
              "seconds": w.seconds, "peak_gb": w.peak_gb, "on_ac": w.on_ac}
    result = next((line[7:] for line in w.lines if line.startswith("RESULT ")), None)
    if w.status:
        record["status"] = w.status
    elif result is not None:
        record.update(status="ok", result=json.loads(result))
    else:
        tail = [line.strip() for line in w.lines if line.strip()][-3:]
        record.update(status="error", error=" | ".join(tail)[-400:])
    return record


def done_jobs() -> set[tuple]:
    if not RESULTS.exists():
        return set()
    done = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            done.add(_key(r))
    return done


def _key(spec: dict) -> tuple:
    return spec["model"], spec["kind"], spec["device"], spec.get("images"), spec.get("precision", "fp16")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(BASELINES))
    parser.add_argument("--kinds", nargs="+", default=["lint", "latency", "ap"])
    parser.add_argument("--precisions", nargs="+", default=["fp16", "int8"])
    parser.add_argument("--devices", nargs="+", default=["cpu", "gpu", "npu", "npu-2t", "npu+gpu"])
    parser.add_argument("--ap-devices", nargs="+", default=["gpu", "npu", "npu-2t"])
    parser.add_argument("--int8-devices", nargs="+", default=["npu", "npu-2t"], help="latency, INT8")
    parser.add_argument("--int8-ap-devices", nargs="+", default=["npu"])
    parser.add_argument("--images", type=int, default=5000, help="COCO val2017 images for AP")
    parser.add_argument("--max-gb", type=float, default=12.0)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--redo", nargs="*", default=[], metavar="KIND",
                        help="re-run these kinds even if done (e.g. latency on an idle machine); "
                             "the report keeps the last result")
    args = parser.parse_args(argv)

    WORK.mkdir(parents=True, exist_ok=True)
    done = {key for key in done_jobs() if key[1] not in args.redo}
    jobs = []
    for model in args.models:
        b = BASELINES[model]
        if not b.ir.exists():
            print(f"{model}: no IR (run benchmarks.m0.prepare), skipped", flush=True)
            continue
        for precision in args.precisions:
            if precision == "int8":
                # the NPU is the INT8 target; quantize first (watchdogged, NNCF can be memory hungry)
                device_sets = {"latency": args.int8_devices, "ap": args.int8_ap_devices}
                quantize = {"model": model, "kind": "quantize", "device": "cpu", "precision": "int8"}
                if not b.ir_path("int8").exists() and _key(quantize) not in done:
                    jobs.append(quantize)
            else:
                device_sets = {"lint": LINT_DEVICES, "latency": args.devices, "ap": args.ap_devices}
            for kind in args.kinds:
                for device in device_sets.get(kind, []):
                    if "+" in device and not b.split:
                        continue
                    spec = {"model": model, "kind": kind, "device": device}
                    if kind == "ap":
                        spec["images"] = args.images
                    if precision != "fp16":
                        spec["precision"] = precision
                    if _key(spec) not in done:
                        jobs.append(spec)

    print(f"{len(jobs)} jobs", flush=True)
    for i, spec in enumerate(jobs, 1):
        record = run_job(spec, args.max_gb, args.min_free_gb)
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        summary = record.get("result") or record.get("error") or ""
        if isinstance(summary, dict):
            keys = ("latency_ms_median", "AP", "rel_error", "issues")
            summary = {k: summary[k] for k in keys if k in summary}
        print(f"[{i}/{len(jobs)}] {spec['model']} {spec['kind']} {spec['device']} "
              f"{spec.get('precision', 'fp16')}: {record['status']} "
              f"({record['seconds']} s, peak {record['peak_gb']} GB) {summary}", flush=True)


if __name__ == "__main__":
    main()
