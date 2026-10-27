"""Latency and throughput benchmarks on OpenVINO devices."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import openvino as ov

from optinpu.devices import core, normalize_ov_device
from optinpu.graph import stage_submodel
from optinpu.runtime import compile_model


@dataclass
class BenchResult:
    device: str
    compile_s: float
    latency_ms_median: float
    latency_ms_p90: float
    throughput_fps: float

    def as_dict(self) -> dict:
        return asdict(self)


def _random_inputs(compiled: ov.CompiledModel) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.random(tuple(i.get_shape()), dtype=np.float32) for i in compiled.inputs]


def benchmark(
    model: str | Path | ov.Model,
    device: str = "npu",
    iterations: int = 100,
    warmup: int = 10,
    throughput_seconds: float = 3.0,
    config: dict | None = None,
) -> BenchResult:
    """Measure batch-1 latency (LATENCY hint) and max throughput (THROUGHPUT hint, async queue)."""
    ov_device = normalize_ov_device(device)
    if isinstance(model, (str, Path)):
        model = core().read_model(model)

    t0 = time.perf_counter()
    lat_model = compile_model(model, ov_device, hint="LATENCY", config=config)
    compile_s = time.perf_counter() - t0

    inputs = _random_inputs(lat_model.compiled)
    request = lat_model.compiled.create_infer_request()
    for _ in range(warmup):
        request.infer(inputs)
    times = []
    for _ in range(iterations):
        t = time.perf_counter()
        request.infer(inputs)
        times.append((time.perf_counter() - t) * 1e3)

    tput_model = compile_model(model, ov_device, hint="THROUGHPUT", config=config)
    queue = ov.AsyncInferQueue(tput_model.compiled)
    done = 0
    t_end = time.perf_counter() + throughput_seconds
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        queue.start_async(inputs)
        done += 1
    queue.wait_all()
    fps = done / (time.perf_counter() - t_start)

    return BenchResult(
        device=ov_device,
        compile_s=compile_s,
        latency_ms_median=float(np.median(times)),
        latency_ms_p90=float(np.percentile(times, 90)),
        throughput_fps=fps,
    )


def benchmark_callable(fn, inputs: list[np.ndarray], device: str, iterations: int = 100, warmup: int = 10,
                       compile_s: float = 0.0) -> BenchResult:
    """Latency of any callable (e.g. a HybridModel). Throughput is sequential: 1000 / median."""
    for _ in range(warmup):
        fn(*inputs)
    times = []
    for _ in range(iterations):
        t = time.perf_counter()
        fn(*inputs)
        times.append((time.perf_counter() - t) * 1e3)
    median = float(np.median(times))
    return BenchResult(device, compile_s, median, float(np.percentile(times, 90)), 1e3 / median)


def benchmark_stages(
    model: str | Path | ov.Model,
    stages: dict[str, str],
    device: str = "npu",
    iterations: int = 50,
) -> dict[str, float]:
    """Cumulative median latency (ms) of each stage prefix, plus "full".

    `stages` maps a label to a name regex covering everything up to that stage, e.g.
    {"backbone": r"\\.backbone\\.", "+encoder": r"\\.backbone\\.|\\.encoder\\."}.
    """
    if isinstance(model, (str, Path)):
        model = core().read_model(model)
    result = {
        label: benchmark(stage_submodel(model, pattern), device, iterations, throughput_seconds=0.5).latency_ms_median
        for label, pattern in stages.items()
    }
    result["full"] = benchmark(model, device, iterations, throughput_seconds=0.5).latency_ms_median
    return result


def format_table(results: list[BenchResult]) -> str:
    header = f"{'device':<6} {'compile s':>9} {'lat med ms':>10} {'lat p90 ms':>10} {'fps':>8}"
    rows = [
        f"{r.device:<6} {r.compile_s:>9.2f} {r.latency_ms_median:>10.2f} {r.latency_ms_p90:>10.2f} {r.throughput_fps:>8.1f}"
        for r in results
    ]
    return "\n".join([header, *rows])
