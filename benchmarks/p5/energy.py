"""P5: energy per inference of the models of the paper, from the RAPL energy counters that Windows
exposes as the "Energy Meter" performance counters (no driver, no administrator rights).

Channels: the package (CPU cores, iGPU, NPU and the rest of the SoC), the CPU cores (PP0) and the iGPU
(PP1), in picowatt-hours, refreshed every ~8 ms. The NPU has no channel of its own: it is in the package.
DRAM reads 0 on this laptop.

For each model, every deployment runs in its own watched child process (as optinpu.sandbox): compiled,
warmed up for 2 s, then 3 rounds of [1 s settle, 4 s idle, 8 s of back-to-back inferences over 8
images preprocessed beforehand]. Each round gives the package power at idle and while inferring; the
energy of one inference is that power times the mean latency, in total and above idle. Deployments:
- npu, gpu, cpu: the original IR on one device (NPU on 2 tiles, CPU in FP32);
- npu-optimized: the NPU deployment of P3.5 (runs/p5/deploy/, see benchmarks.p5.deployments; the
  classifiers' are rebuilt here the same way, with optimizer.replay); skipped when nothing was applied;
- npu+gpu: the deployment with the decoder on the iGPU (runs/p3/offload-gpu/deploy/; DETRs only).
Results go to runs/p5/energy.json (resumable, per deployment) and energy.md. Windows is kept awake and
each deployment waits for mains power.
P6 (review), --p6: results in runs/p6/, 20 s inference windows, the NPU deployment confirmed on held-out
images when P6 re-ran it (runs/p6/confirm/), and two more deployments:
- gpu-optimized: the same optimizer for the iGPU (runs/p6/igpu-confirm/deploy/, else runs/p6/igpu/deploy/), when
  it applied something;
- npu-turbo: the NPU deployment (optimized, else the original) compiled with NPU_TURBO.

    python -m benchmarks.p5.energy                       # the 10 models of the paper
    python -m benchmarks.p5.energy yolo26-n:fp16 ...     # or given ones
    python -m benchmarks.p5.energy --p6
"""

from __future__ import annotations

import ctypes
import json
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import openvino as ov

from benchmarks.m0.baselines import ROOT
from benchmarks.p0.models import all_models
from benchmarks.p3.campaign import keep_awake, stamp, wait_for_mains
from benchmarks.p5.deployments import MODELS as DETECTORS
from optinpu.data import coco_val_subset, imagenetv2_subset
from optinpu.optimizer import load_deployment, replay
from optinpu.sandbox import run_watched
from optinpu.tasks import ClassificationTask, Deployed, task_from_dict, task_to_dict

CLASSIFIERS = ("resnet50_cls224:fp32", "mobilenet_v2_cls224:fp32", "efficientnet_b0_cls224:fp32")
P3, OUT = ROOT / "runs" / "p3", ROOT / "runs" / "p5"
P6 = ROOT / "runs" / "p6"
CHANNELS = {"package": "PKG", "cores": "PP0", "igpu": "PP1"}
IMAGES, ROUNDS, SETTLE, IDLE, ACTIVE = 8, 3, 1.0, 4.0, 8.0  # images, rounds, seconds


class _Raw(ctypes.Structure):  # PDH_RAW_COUNTER
    _fields_ = [("CStatus", wintypes.DWORD), ("TimeStamp", wintypes.FILETIME), ("FirstValue", ctypes.c_longlong),
                ("SecondValue", ctypes.c_longlong), ("MultiCount", wintypes.DWORD)]


class EnergyMeter:
    """The RAPL channels of the Windows "Energy Meter" counters, read through PDH."""

    def __init__(self):
        self.pdh = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        self._check(self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query)))
        self.counters = {}
        for name, channel in CHANNELS.items():
            for kind in ("Energy", "Time"):
                handle = ctypes.c_void_p()
                self._check(self.pdh.PdhAddEnglishCounterW(
                    self.query, f"\\Energy Meter(RAPL_Package0_{channel})\\{kind}", None, ctypes.byref(handle)))
                self.counters[name, kind] = handle

    @staticmethod
    def _check(code: int) -> None:
        if code != 0:
            raise OSError(f"PDH error {code & 0xFFFFFFFF:#x} (no Energy Meter counters on this machine?)")

    def _value(self, handle) -> int:
        raw, kind = _Raw(), wintypes.DWORD()
        self._check(self.pdh.PdhGetRawCounterValue(handle, ctypes.byref(kind), ctypes.byref(raw)))
        return raw.FirstValue

    def read(self) -> dict[str, tuple[float, float]]:
        """channel -> (energy in J, counter time in s)."""
        self._check(self.pdh.PdhCollectQueryData(self.query))
        return {name: (self._value(self.counters[name, "Energy"]) * 3.6e-9,  # pWh -> J
                       self._value(self.counters[name, "Time"]) / 1e3) for name in CHANNELS}  # ms -> s


def powers(start: dict, end: dict) -> dict[str, float]:
    """Mean power (W) of each channel between two reads, over the counters' own time."""
    return {k: (end[k][0] - start[k][0]) / max(end[k][1] - start[k][1], 1e-9) for k in start}


def child(spec: dict) -> dict:
    """Child side: compile spec["parts"], then the idle / inference rounds."""
    task = task_from_dict(spec["task"])
    model = Deployed([tuple(p) for p in spec["parts"]], task, cache_dir=None).model
    inputs = [task.inputs(path)[0] for path in spec["images"]]
    meter = EnergyMeter()

    def infer(seconds: float) -> tuple[int, float]:
        count, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            model(*inputs[count % len(inputs)])
            count += 1
        return count, time.perf_counter() - t0

    infer(2.0)  # warm-up
    rounds = []
    for _ in range(spec["rounds"]):
        time.sleep(spec["settle"])
        start = meter.read()
        time.sleep(spec["idle"])
        idle = powers(start, meter.read())
        start = meter.read()
        count, seconds = infer(spec["active"])
        rounds.append({"idle_w": idle, "active_w": powers(start, meter.read()), "inferences": count,
                       "seconds": seconds})
    return {"rounds": rounds}


def summarize(rounds: list[dict]) -> dict:
    latency = np.array([r["seconds"] / r["inferences"] for r in rounds])
    package = lambda r, key: r[key]["package"]
    total = np.array([package(r, "active_w") for r in rounds]) * latency
    above = np.array([package(r, "active_w") - package(r, "idle_w") for r in rounds]) * latency
    return {"latency_ms": 1e3 * float(latency.mean()),
            "idle_w": {k: float(np.mean([r["idle_w"][k] for r in rounds])) for k in CHANNELS},
            "active_w": {k: float(np.mean([r["active_w"][k] for r in rounds])) for k in CHANNELS},
            "mj": 1e3 * float(total.mean()), "mj_above_idle": 1e3 * float(above.mean()),
            "mj_above_idle_sd": 1e3 * float(above.std(ddof=1)) if len(rounds) > 1 else float("nan"),
            "rounds": rounds}


def measure(task, parts: list, images: list[str], active: float = ACTIVE) -> dict:
    """Energy of one deployment, in a watched child process (whole tree killed on a limit)."""
    spec = {"task": task_to_dict(task), "parts": parts, "images": images, "rounds": ROUNDS, "settle": SETTLE,
            "idle": IDLE, "active": active}
    with tempfile.TemporaryDirectory(prefix="optinpu-energy-") as tmp:
        spec_file = Path(tmp) / "spec.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")
        watched = run_watched([sys.executable, "-m", "benchmarks.p5.energy", "--child", str(spec_file)], 900,
                              cwd=ROOT)
    for line in reversed(watched.lines):
        if line.startswith("RESULT "):
            return summarize(json.loads(line[len("RESULT "):])["rounds"])
    last = next((line.strip() for line in reversed(watched.lines) if line.strip()), "no output")
    return {"error": (watched.status or last)[-200:]}


def optimized(name: str, precision: str, entry, p6: bool = False) -> Path | None:
    """Folder of the NPU deployment of P3.5 (rebuilt and saved if missing), None if nothing was applied.
    With p6, the deployment re-run with the held-out confirmation, when there is one (runs/p6/confirm/)."""
    confirmed = P6 / "confirm" / f"optimize_{name}_{precision}.json"
    if p6 and confirmed.exists():
        applied = json.loads(confirmed.read_text())["applied"]
        return None if applied == "original" else P6 / "confirm" / "deploy" / f"{name}_{precision}"
    record = json.loads((P3 / f"optimize_{name}_{precision}.json").read_text())
    if record["applied"] == "original":
        return None
    directory = OUT / "deploy" / f"{name}_{precision}"
    if not (directory / "deployment.json").exists():
        classifier = isinstance(entry.task, ClassificationTask)
        calibration = imagenetv2_subset(300, 7000) if classifier else coco_val_subset(300, offset=4500)
        model = ov.Core().read_model(entry.irs[precision])
        replay(model, record["applied"].split(" + "), entry.task, calibration.image_paths()).save(directory)
    return directory


def deployments(name: str, precision: str, entry, p6: bool = False) -> dict[str, list]:
    """label -> parts ([[ir, device, config]]), in measurement order (the CPU last: it heats the package)."""
    parts = lambda directory: [[str(p), d, c] for p, d, c in load_deployment(directory)]
    ir = str(entry.irs[precision])
    out = {"npu": [[ir, "npu", None]]}
    directory = optimized(name, precision, entry, p6)
    if directory is not None:
        out["npu-optimized"] = parts(directory)
    if p6:
        best = out.get("npu-optimized", out["npu"])
        out["npu-turbo"] = [[p, d, {**(c or {}), "NPU_TURBO": True} if d == "npu" else c] for p, d, c in best]
    hybrid = P3 / "offload-gpu" / "deploy" / f"{name}_{precision}"
    if hybrid.exists():
        out["npu+gpu"] = parts(hybrid)
    out["gpu"] = [[ir, "gpu", None]]
    # the iGPU run with the held-out confirmation when there is one, else the one of night 1
    folder = next((f for f in (P6 / "igpu-confirm", P6 / "igpu") if (f / f"optimize_{name}_{precision}.json").exists()), None)
    if p6 and folder and json.loads((folder / f"optimize_{name}_{precision}.json").read_text())["applied"] != "original":
        out["gpu-optimized"] = parts(folder / "deploy" / f"{name}_{precision}")
    return {**out, "cpu": [[ir, "cpu", None]]}


def line(r: dict) -> str:
    if "error" in r:
        return f"failed: {r['error']}"
    return (f"{r['latency_ms']:.2f} ms, package {r['idle_w']['package']:.2f} -> {r['active_w']['package']:.2f} W, "
            f"{r['mj']:.1f} mJ per inference ({r['mj_above_idle']:.1f} above idle)")


def summary(results: dict) -> str:
    lines = ["| Model | Deployment | Latency (ms) | Package W: idle / inferring | mJ per inference: total / above idle "
             "| Cores W | iGPU W |", "|---|---|---|---|---|---|---|"]
    for spec, done in results.items():
        for label, r in done.items():
            if "error" in r:
                lines.append(f"| {spec} | {label} | failed: {r['error'][:60]} | | | | |")
                continue
            lines.append(f"| {spec} | {label} | {r['latency_ms']:.2f} | {r['idle_w']['package']:.2f} / "
                         f"{r['active_w']['package']:.2f} | {r['mj']:.1f} / {r['mj_above_idle']:.1f} "
                         f"(sd {r['mj_above_idle_sd']:.1f}) | {r['active_w']['cores']:.2f} | {r['active_w']['igpu']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        print("RESULT " + json.dumps(child(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))), flush=True)
        return
    EnergyMeter()  # fails early without the counters
    p6 = "--p6" in sys.argv[1:]
    specs = [a for a in sys.argv[1:] if a != "--p6"]
    out = P6 if p6 else OUT
    active = 20.0 if p6 else ACTIVE
    out.mkdir(parents=True, exist_ok=True)
    path = out / "energy.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    keep_awake(True)
    try:
        for spec in specs or (*DETECTORS, *CLASSIFIERS):
            name, precision = spec.split(":")
            entry = next(m for m in all_models() if m.name == name)
            subset = imagenetv2_subset(IMAGES) if isinstance(entry.task, ClassificationTask) else coco_val_subset(IMAGES)
            images = [str(p) for p in subset.image_paths()]
            done = results.setdefault(spec, {})
            for label, parts in deployments(name, precision, entry, p6).items():
                if label in done:
                    continue
                wait_for_mains()
                done[label] = measure(entry.task, parts, images, active)
                path.write_text(json.dumps(results, indent=1))
                print(f"{stamp()} {spec} {label}: {line(done[label])}", flush=True)
    finally:
        keep_awake(False)
    text = summary(results)
    (out / "energy.md").write_text(text, encoding="utf-8")
    print(f"\n{text}", flush=True)


if __name__ == "__main__":
    main()
