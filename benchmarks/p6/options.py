"""P6 (review): NPU options beyond NPU_TILES, and asynchronous execution, on OptiNPU's deployments
(runs/p5/deploy/, plus the original EfficientNet-B0, which OptiNPU keeps as is).

Latency, one paired session per deployment (adaptive rounds, real images): as OptiNPU compiles it (latency
hint), with NPU_TURBO, with the THROUGHPUT performance hint, and for INT8 deployments with
NPU_QDQ_OPTIMIZATION (options set on the NPU parts only).
Throughput, inferences per second over 10 s on real images, as OptiNPU compiles it: synchronous, one
inference after the other; and with two inferences in flight: for a two-part deployment (NPU network, CPU
selection tail) a pipeline, where the NPU runs the next image while the CPU finishes the previous one; for
a one-part deployment, two threads with their own infer requests.
Each deployment waits for mains power; Windows is kept awake.
Results: runs/p6/options.json (resumable) and runs/p6/options.md.

    python -m benchmarks.p6.options
"""

from __future__ import annotations

import json
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from benchmarks.p3.campaign import keep_awake, wait_for_mains
from optinpu.data import coco_val_subset, imagenetv2_subset
from optinpu.tasks import ClassificationTask

OUT = ROOT / "runs" / "p6"
DEPLOY = ROOT / "runs" / "p5" / "deploy"
COUNT, SECONDS = 30, 10.0
OPTIONS = {"turbo": {"NPU_TURBO": True}, "throughput hint": {"PERFORMANCE_HINT": "THROUGHPUT"},
           "QDQ optimization": {"NPU_QDQ_OPTIMIZATION": True}}


def deployments() -> dict[str, tuple[list, object, bool]]:
    """name -> (parts [(IR path, device, config)], task, INT8)."""
    from optinpu.optimizer import load_deployment

    bench = {m.name: m for m in all_models()}
    out = {}
    for folder in sorted(DEPLOY.glob("*/deployment.json")):
        name = folder.parent.name
        changes = json.loads(folder.read_text())["changes"]
        out[name] = (load_deployment(folder.parent), bench[name.rsplit("_", 1)[0]].task,
                     any(c.startswith("int8") for c in changes))
    entry = bench["efficientnet_b0_cls224"]
    out["efficientnet_b0_cls224_fp32"] = ([(entry.irs["fp32"], "npu", None)], entry.task, False)
    return out


def save_samples(task, path: Path) -> Path:
    sources = (imagenetv2_subset(COUNT, 7300) if isinstance(task, ClassificationTask)
               else coco_val_subset(COUNT, offset=4970)).image_paths()
    np.savez(path, **{f"s{i}_{j}": x for i, s in enumerate(sources) for j, x in enumerate(task.inputs(s)[0])})
    return path


def child(spec: dict) -> dict:
    """Inferences per second of spec["parts"] on the samples, "sync" or with two in flight."""
    from optinpu.runtime import compile_model
    from optinpu.sandbox import _timing_samples

    chain = [compile_model(p, d, config=c, cache_dir=None).compiled for p, d, c in spec["parts"]]
    samples = _timing_samples(chain[0], f"file:{spec['samples']}")
    n = len(samples)

    def infer(request, compiled, data):
        result = request.infer(data, share_inputs=True, share_outputs=True)
        return [result[o] for o in compiled.outputs]

    def sync_loop(deadline: float) -> int:
        requests = [c.create_infer_request() for c in chain]
        done = 0
        while time.perf_counter() < deadline:
            data = samples[done % n]
            for c, r in zip(chain, requests):
                data = infer(r, c, data)
            done += 1
        return done

    sync_loop(time.perf_counter() + 2.0)  # warm-up
    t0 = time.perf_counter()
    if spec["mode"] == "sync":
        done = sync_loop(t0 + SECONDS)
    elif len(chain) == 1:  # two threads, each with its own infer request
        counts = [0, 0]

        def worker(k: int) -> None:
            request = chain[0].create_infer_request()
            while time.perf_counter() < t0 + SECONDS:
                infer(request, chain[0], samples[(counts[k] * 2 + k) % n])
                counts[k] += 1

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        done = sum(counts)
    else:  # pipeline: the head of the next image runs while the tail of the previous one does
        handoff: queue.Queue = queue.Queue(maxsize=2)
        finished = [0]

        def head() -> None:
            request, i = chain[0].create_infer_request(), 0
            while time.perf_counter() < t0 + SECONDS:
                handoff.put([np.array(x) for x in infer(request, chain[0], samples[i % n])])
                i += 1
            handoff.put(None)

        def tail() -> None:
            request = chain[1].create_infer_request()
            while (data := handoff.get()) is not None:
                infer(request, chain[1], data)
                finished[0] += 1

        threads = [threading.Thread(target=head), threading.Thread(target=tail)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        done = finished[0]
    return {"fps": done / (time.perf_counter() - t0), "inferences": done}


def throughput(parts: list, samples: Path, mode: str) -> float | str:
    from optinpu.sandbox import run_watched

    with tempfile.TemporaryDirectory(prefix="optinpu-options-") as tmp:
        spec_file = Path(tmp) / "spec.json"
        spec_file.write_text(json.dumps({"parts": [[str(p), d, c] for p, d, c in parts], "samples": str(samples),
                                         "mode": mode, "out": str(Path(tmp) / "result.json")}))
        watched = run_watched([sys.executable, "-m", "benchmarks.p6.options", "--child", str(spec_file)], 900, cwd=ROOT)
        result = Path(tmp) / "result.json"
        if result.exists():
            return json.loads(result.read_text())["fps"]
        return f"failed: {watched.status or watched.returncode}: {' / '.join(watched.lines[-3:])}"[-300:]


def run(name: str, parts: list, task, int8: bool) -> dict:
    from optinpu.profiler import Adaptive, measure, paired_saving

    with tempfile.TemporaryDirectory(prefix="optinpu-options-") as tmp:
        samples = save_samples(task, Path(tmp) / "samples.npz")
        variants = {"default": parts}
        for option, config in OPTIONS.items():
            if option == "QDQ optimization" and not int8:
                continue
            variants[option] = [(p, d, {**(c or {}), **config} if d == "npu" else c) for p, d, c in parts]
        modes = {k: f"file:{samples}" for k in variants}
        timings = measure(variants, "npu", input_modes=modes, adaptive=Adaptive("default"), timeout=3600)
        base = timings["default"].median_ms
        out = {"latency_ms": base, "parts": len(parts)}
        for option in variants:
            if option == "default":
                continue
            if timings[option].error is not None:
                out[option] = {"error": timings[option].error[-200:]}
                continue
            saving, ci = paired_saving(timings, "default", option)
            out[option] = {"change_pct": -100 * saving / base, "ci_pct": 100 * ci / base}
        out["fps_sync"] = throughput(parts, samples, "sync")
        out["fps_two_in_flight"] = throughput(parts, samples, "async")
    return out


def table(results: dict) -> str:
    lines = ["| Deployment | ms | Turbo | Throughput hint | QDQ | FPS sync | FPS 2 in flight |", "|---|---|---|---|---|---|---|"]
    def cell(r: dict, k: str) -> str:
        return "" if k not in r else ("failed" if "error" in r[k] else f"{r[k]['change_pct']:+.1f} ± {r[k]['ci_pct']:.1f} %")

    def fps(r: dict, k: str) -> str:
        return f"{r[k]:.1f}" if isinstance(r.get(k), float) else str(r.get(k, ""))[:20]

    for name, r in results.items():
        lines.append(f"| {name} | {r['latency_ms']:.2f} | {cell(r, 'turbo')} | {cell(r, 'throughput hint')} | "
                     f"{cell(r, 'QDQ optimization')} | {fps(r, 'fps_sync')} | {fps(r, 'fps_two_in_flight')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        spec = json.loads(Path(sys.argv[2]).read_text())
        Path(spec["out"]).write_text(json.dumps(child(spec)))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "options.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    keep_awake(True)
    try:
        for name, (parts, task, int8) in deployments().items():
            if name in results:
                print(f"{name}: already done", flush=True)
                continue
            wait_for_mains()
            print(f"===== {name}", flush=True)
            results[name] = run(name, parts, task, int8)
            print(json.dumps(results[name], indent=1), flush=True)
            path.write_text(json.dumps(results, indent=1))
    finally:
        keep_awake(False)
    text = table(results)
    (OUT / "options.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
