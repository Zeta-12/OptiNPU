"""Run NPU work in a child process under a memory / time watchdog.

The NPU compiler runs inside the calling process and can grab tens of GB on some graphs, and a lost
NPU (ZE_RESULT_ERROR_DEVICE_LOST) stays lost for the process that hit it (PLAN.md, L5 and M0). The
profiler and the optimizer compile many untested graph variants: each job runs in a child process,
killed past `max_gb` of committed memory, when the system runs low on memory, or past its timeout.

    python -m optinpu.sandbox '{"job": "latency", "ir": "model.xml", "device": "npu"}'

prints one line "RESULT <json>".
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov
import psutil


@dataclass
class Watched:
    status: str | None  # None if the process ended by itself, else "killed: <reason>"
    returncode: int | None
    lines: list[str] = field(default_factory=list)
    seconds: float = 0.0
    peak_gb: float = 0.0
    on_ac: bool | None = None  # laptop performance depends on AC vs battery


def _family(proc: psutil.Process) -> list[psutil.Process]:
    """The process and its children; empty once it has exited (it may exit at any time)."""
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return []


def _committed_gb(proc: psutil.Process) -> float:
    total = 0
    for p in _family(proc):
        try:
            info = p.memory_info()
            total += getattr(info, "private", info.vms)  # Windows: committed private bytes
        except psutil.Error:
            pass
    return total / 1e9


def run_watched(cmd: list[str], timeout: float, max_gb: float = 12.0, min_free_gb: float = 1.0,
                echo: bool = False, cwd: str | Path | None = None) -> Watched:
    """Run `cmd`; stdout and stderr are collected (and echoed if asked). The whole process tree is
    killed on any limit, so that no compiler or DataLoader worker survives."""
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    watched = psutil.Process(proc.pid)
    peak, status = 0.0, None
    lines: list[str] = []

    def read():
        for line in proc.stdout:
            lines.append(line)
            if echo:
                print(line, end="", flush=True)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    while proc.poll() is None:
        peak = max(peak, _committed_gb(watched))
        free = psutil.virtual_memory().available / 1e9
        if peak > max_gb:
            status = f"killed: {peak:.1f} GB committed > {max_gb} GB"
        elif free < min_free_gb:
            status = f"killed: system free memory {free:.1f} GB < {min_free_gb} GB"
        elif time.time() - started > timeout:
            status = f"killed: timeout {timeout:.0f} s"
        if status:
            for p in _family(watched):
                try:
                    p.kill()
                except psutil.Error:
                    pass
            break
        time.sleep(0.2)
    proc.wait()
    reader.join(timeout=5)
    battery = psutil.sensors_battery()
    return Watched(status, proc.returncode, lines, round(time.time() - started, 1), round(peak, 2),
                   None if battery is None else battery.power_plugged)


@dataclass
class JobResult:
    ok: bool
    result: dict | None
    error: str | None  # watchdog reason, or the child's last output line
    seconds: float
    peak_gb: float

    @property
    def device_lost(self) -> bool:
        return bool(self.error) and "DEVICE_LOST" in self.error


def run_job(spec: dict, timeout: float = 600, max_gb: float = 12.0, echo: bool = False) -> JobResult:
    """Run one job (see JOBS) in a watched child process. An ov.Model under spec["ir"] is saved to a
    temporary IR first (weights kept as they are)."""
    with tempfile.TemporaryDirectory(prefix="optinpu-") as tmp:
        spec = dict(spec)
        if isinstance(spec.get("ir"), ov.Model):
            path = Path(tmp) / "model.xml"
            ov.save_model(spec["ir"], path, compress_to_fp16=False)
            spec["ir"] = str(path)
        spec = {k: str(v) if isinstance(v, Path) else v for k, v in spec.items()}
        watched = run_watched([sys.executable, "-m", "optinpu.sandbox", json.dumps(spec)], timeout, max_gb,
                              echo=echo)
    for line in reversed(watched.lines):
        if line.startswith("RESULT "):
            return JobResult(True, json.loads(line[len("RESULT "):]), None, watched.seconds, watched.peak_gb)
    last = next((line.strip() for line in reversed(watched.lines) if line.strip()), "no output")
    return JobResult(False, None, watched.status or last[-300:], watched.seconds, watched.peak_gb)


# ---------------------------------------------------------------- jobs (child side)

def _latency(spec: dict) -> dict:
    """Batch-1 latency of spec["ir"] on spec["device"] with spec.get("config")."""
    from optinpu.bench import benchmark

    result = benchmark(spec["ir"], spec.get("device", "npu"), iterations=spec.get("iterations", 100),
                       throughput_seconds=spec.get("throughput_seconds", 0), config=spec.get("config"))
    return result.as_dict()


def _outputs(spec: dict) -> dict:
    """Run spec["ir"] on the inputs saved in spec["inputs"] (npz: sample{i}_input{j}); save the outputs
    to spec["outputs"] (npz: sample{i}_output{j}). Lets the parent judge accuracy on its side."""
    from optinpu.runtime import compile_model

    device, config = spec.get("device", "npu"), spec.get("config")
    if device.lower() == "cpu":
        config = {"INFERENCE_PRECISION_HINT": "f32", **(config or {})}
    model = compile_model(spec["ir"], device, config=config, cache_dir=None)
    data = np.load(spec["inputs"])
    samples = sorted({k.split("_")[0] for k in data.files}, key=lambda s: int(s[len("sample"):]))
    out = {}
    for s in samples:
        inputs = [data[k] for k in sorted((k for k in data.files if k.startswith(s + "_")),
                                          key=lambda k: int(k.rsplit("input", 1)[1]))]
        for j, o in enumerate(model(*inputs)):
            out[f"{s}_output{j}"] = o
    np.savez(spec["outputs"], **out)
    return {"samples": len(samples)}


def _random_inputs(compiled: ov.CompiledModel, mode: str = "random") -> list[np.ndarray]:
    """Inputs for timing: uniform [0, 1) ("random") or "zeros" (to test data-dependent timing)."""
    rng = np.random.default_rng(0)
    inputs = []
    for port in compiled.inputs:
        dtype = port.get_element_type().to_dtype()
        shape = tuple(port.get_shape())
        if mode == "random" and np.issubdtype(dtype, np.floating):
            inputs.append(rng.random(shape).astype(dtype))
        else:
            inputs.append(np.zeros(shape, dtype))
    return inputs


def _chain(parts: list, mode: str):
    """Compile [path, device, config] parts and return a function running them back to back, each
    part's outputs (in order) feeding the next one's inputs without copies."""
    from optinpu.runtime import compile_model

    chain = []
    for path, device, config in parts:
        compiled = compile_model(path, device, config=config, cache_dir=None).compiled
        chain.append((compiled, compiled.create_infer_request()))
    inputs = _random_inputs(chain[0][0], mode)

    def run():
        data = inputs
        for compiled, request in chain:
            result = request.infer(data, share_inputs=True, share_outputs=True)
            data = [result[o] for o in compiled.outputs]

    return run


def _half_width(reference: list[float], other: list[float]) -> float:
    """95 % half-width of the mean per-round difference between two models."""
    d = np.asarray(reference) - np.asarray(other)
    return float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")


def _compare(spec: dict) -> dict:
    """Latency of several models compiled side by side, measured round-robin: in each round, every model
    runs spec["iterations"] inferences, in a random order, and its round median is recorded.
    spec["irs"] maps a name to an IR path, or to {"parts": [[path, device, config], ...]} for a model
    split over devices (run back to back). A model that fails to compile is reported with its error;
    the others are still measured. spec["input_modes"] (optional) maps a name to an input mode of
    _random_inputs.

    Adaptive repetitions: with spec["reference"] (a model name) and spec["max_rounds"] above
    spec["rounds"], rounds continue until the 95 % confidence half-width of every model's per-round
    difference with the reference is below max(spec["target_ms"], spec["target_rel"] x reference
    median), or until max_rounds / spec["max_seconds"]. Used by optinpu.profiler.measure."""
    device, config = spec.get("device", "npu"), spec.get("config")
    modes = spec.get("input_modes") or {}
    nan = float("nan")
    result, ready = {}, {}
    for name, ir in spec["irs"].items():
        try:
            parts = ir["parts"] if isinstance(ir, dict) else [[ir, device, config]]
            run = _chain(parts, modes.get(name, "random"))
            for _ in range(5):
                run()
            ready[name] = run
        except Exception as e:  # any compile / runtime failure is a result
            result[name] = {"median_ms": nan, "q1_ms": nan, "q3_ms": nan, "error": str(e).strip()[-300:], "rounds": []}
    min_rounds = spec.get("rounds", 5)
    max_rounds = max(min_rounds, spec.get("max_rounds") or min_rounds)
    reference = spec.get("reference")
    rng = np.random.default_rng(0)
    times: dict[str, list[float]] = {name: [] for name in ready}
    per_round: dict[str, list[float]] = {name: [] for name in ready}
    started = time.time()
    for done in range(1, max_rounds + 1):
        for name in rng.permutation(list(ready)):
            values = []
            for _ in range(spec.get("iterations", 30)):
                t = time.perf_counter()
                ready[name]()
                values.append((time.perf_counter() - t) * 1e3)
            times[name] += values
            per_round[name].append(float(np.median(values)))
        if done < min_rounds:
            continue
        if reference not in ready or time.time() - started > spec.get("max_seconds", 600):
            break
        target = max(spec.get("target_ms", 0.0), spec.get("target_rel", 0.0) * float(np.median(per_round[reference])))
        if all(_half_width(per_round[reference], per_round[n]) <= target for n in ready if n != reference):
            break
    for name, values in times.items():
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        result[name] = {"median_ms": float(median), "q1_ms": float(q1), "q3_ms": float(q3), "error": None,
                        "rounds": per_round[name]}
    return result


def _vendor(spec: dict) -> dict:
    """Vendor per-task counters (PERF_COUNT) of spec["ir"]: every executed task, plus time per
    execution unit type (on the NPU, e.g. DPU vs SHAVE vs DMA)."""
    from collections import defaultdict

    from optinpu.profile import profile

    layers = profile(spec["ir"], spec.get("device", "npu"), iterations=spec.get("iterations", 20),
                     config=spec.get("config"))
    exec_types: dict[str, float] = defaultdict(float)
    for layer in layers:
        exec_types[layer.exec_type] += layer.ms
    return {"layers": [{"name": layer.name, "node_type": layer.node_type, "exec_type": layer.exec_type,
                        "ms": layer.ms} for layer in layers], "exec_types": dict(exec_types)}


def _deployed(spec: dict):
    """optinpu.tasks.Deployed for spec["parts"] ([[path, device, config], ...], one or two parts) and
    spec["task"]."""
    from optinpu.tasks import Deployed, task_from_dict

    task = task_from_dict(spec["task"])
    parts = spec["parts"]
    if len(parts) == 1:
        path, device, config = parts[0]
        return Deployed(path, task, device, config, cache_dir=None)
    return Deployed([tuple(p) for p in parts], task, cache_dir=None)


def _evaluate(spec: dict) -> dict:
    """COCO AP of a model (spec["parts"], spec["task"]) on spec["subset"] (CocoSubset fields)."""
    from optinpu.data.coco import CocoSubset
    from optinpu.evaluate import evaluate_coco

    s = spec["subset"]
    names = {int(k): v for k, v in s["file_names"].items()} if s.get("file_names") else None
    subset = CocoSubset(Path(s["annotations"]), Path(s["image_dir"]), s["image_ids"], s["category_ids"], names)
    return evaluate_coco(_deployed(spec), subset, "sandbox", "-", folds=spec.get("folds", 10)).as_dict()


def _predict(spec: dict) -> dict:
    """Task predictions of a model (spec["parts"], spec["task"]) on spec["samples"] (image paths) or the
    samples saved in spec["samples_npz"]; saved to spec["out"]."""
    from optinpu.tasks import save_predictions

    deployed = _deployed(spec)
    if spec.get("samples_npz"):
        with np.load(spec["samples_npz"]) as data:
            count = int(data["count"])
            samples = [[data[k] for k in sorted((k for k in data.files if k.startswith(f"s{i}_")),
                                                 key=lambda k: int(k.rsplit("_", 1)[1]))] for i in range(count)]
    else:
        samples = spec["samples"]
    save_predictions(spec["out"], deployed.run(samples))
    return {"count": len(samples)}


JOBS = {"latency": _latency, "outputs": _outputs, "compare": _compare, "vendor": _vendor, "predict": _predict,
        "evaluate": _evaluate}


def save_parts(parts: list, directory: str | Path) -> list[list]:
    """[(model, device, config)] with ov.Models saved as IRs in `directory` -> JSON-ready parts."""
    saved = []
    for i, (model, device, config) in enumerate(parts):
        if isinstance(model, ov.Model):
            path = Path(directory) / f"part{i}_{id(model)}.xml"
            ov.save_model(model, path, compress_to_fp16=False)
            model = path
        saved.append([str(model), device, config])
    return saved


def run_predictions(parts: list, task, samples: list, timeout: float = 1800,
                    max_gb: float = 12.0) -> tuple[list | None, JobResult]:
    """Predictions of `task` on `samples` (image sources, or lists of arrays for OutputTask) for a model
    given as [(model, device, config)] parts (one, or two run back to back), computed in a watched
    child process: a compiler blow-up or a lost NPU cannot take the caller down."""
    from optinpu.tasks import load_predictions, task_to_dict

    with tempfile.TemporaryDirectory(prefix="optinpu-predict-") as tmp:
        spec = {"job": "predict", "parts": save_parts(parts, tmp), "task": task_to_dict(task),
                "out": str(Path(tmp) / "predictions.npz")}
        if samples and not isinstance(samples[0], (str, Path)):
            npz = Path(tmp) / "samples.npz"
            np.savez(npz, count=np.array(len(samples)),
                     **{f"s{i}_{j}": np.asarray(x) for i, s in enumerate(samples) for j, x in enumerate(s)})
            spec["samples_npz"] = str(npz)
        else:
            spec["samples"] = [str(s) for s in samples]
        job = run_job(spec, timeout, max_gb)
        if not job.ok:
            return None, job
        return load_predictions(spec["out"], getattr(task, "class_names", None)), job


def run_evaluation(parts: list, task, subset, timeout: float = 3600,
                   max_gb: float = 12.0) -> tuple[dict | None, JobResult]:
    """COCO metrics (optinpu.evaluate.EvalResult fields) of a model given as [(model, device, config)]
    parts on `subset` (an optinpu.data.CocoSubset), computed in a watched child process."""
    from optinpu.tasks import task_to_dict

    with tempfile.TemporaryDirectory(prefix="optinpu-eval-") as tmp:
        spec = {"job": "evaluate", "parts": save_parts(parts, tmp), "task": task_to_dict(task),
                "subset": {"annotations": str(subset.annotations), "image_dir": str(subset.image_dir),
                           "image_ids": list(subset.image_ids), "category_ids": list(subset.category_ids),
                           "file_names": subset.file_names}}
        job = run_job(spec, timeout, max_gb)
    return (job.result if job.ok else None), job


def run_outputs(model, samples: list[list[np.ndarray]], device: str = "npu", config: dict | None = None,
                timeout: float = 900, max_gb: float = 12.0) -> tuple[list[list[np.ndarray]] | None, JobResult]:
    """Raw outputs of `model` on `samples` (lists of input arrays), computed in a watched child process.
    Returns (outputs per sample, or None on failure; the job result)."""
    with tempfile.TemporaryDirectory(prefix="optinpu-io-") as tmp:
        inputs, outputs = Path(tmp) / "inputs.npz", Path(tmp) / "outputs.npz"
        np.savez(inputs, **{f"sample{i}_input{j}": x for i, sample in enumerate(samples) for j, x in enumerate(sample)})
        job = run_job({"job": "outputs", "ir": model, "device": device, "config": config,
                       "inputs": inputs, "outputs": outputs}, timeout, max_gb)
        if not job.ok:
            return None, job
        data = np.load(outputs)
        result = []
        for i in range(len(samples)):
            keys = sorted((k for k in data.files if k.startswith(f"sample{i}_")), key=lambda k: int(k.rsplit("output", 1)[1]))
            result.append([data[k] for k in keys])
        data.close()
        return result, job


if __name__ == "__main__":
    job_spec = json.loads(sys.argv[1])
    print("RESULT " + json.dumps(JOBS[job_spec["job"]](job_spec)), flush=True)
