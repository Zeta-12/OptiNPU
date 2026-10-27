"""Per-layer profiling of a compiled model (OpenVINO PERF_COUNT)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov

from optinpu.devices import core, normalize_ov_device


@dataclass
class LayerTime:
    name: str
    node_type: str
    exec_type: str
    ms: float


def profile(model: str | Path | ov.Model, device: str = "npu", iterations: int = 20) -> list[LayerTime]:
    """Average per-layer execution time over `iterations`, sorted slowest first.

    Names and types are those reported by the device plugin: they may be fused or renamed
    compared to the IR (e.g. the NPU reports its own task names).
    """
    ov_device = normalize_ov_device(device)
    compiled = core().compile_model(model, ov_device, {"PERF_COUNT": True, "PERFORMANCE_HINT": "LATENCY"})
    request = compiled.create_infer_request()
    rng = np.random.default_rng(0)
    inputs = [rng.random(tuple(i.get_shape()), dtype=np.float32) for i in compiled.inputs]
    request.infer(inputs)  # warmup

    totals: dict[tuple[str, str, str], float] = defaultdict(float)
    for _ in range(iterations):
        request.infer(inputs)
        for info in request.get_profiling_info():
            if info.status == info.Status.EXECUTED:
                totals[(info.node_name, info.node_type, info.exec_type)] += info.real_time.total_seconds() * 1e3
    layers = [LayerTime(n, t, e, ms / iterations) for (n, t, e), ms in totals.items()]
    return sorted(layers, key=lambda layer: -layer.ms)


def by_type(layers: list[LayerTime], key: str = "node_type") -> list[tuple[str, float, int]]:
    """Aggregate per `key` ("node_type" or "exec_type"): (key, total ms, count), slowest first."""
    agg: dict[str, list] = defaultdict(lambda: [0.0, 0])
    for layer in layers:
        agg[getattr(layer, key)][0] += layer.ms
        agg[getattr(layer, key)][1] += 1
    return sorted(((k, ms, n) for k, (ms, n) in agg.items()), key=lambda r: -r[1])


def format_profile(layers: list[LayerTime], top: int = 15) -> str:
    total = sum(layer.ms for layer in layers)
    lines = [
        f"sum of layer times {total:.2f} ms over {len(layers)} layers",
        "(the NPU overlaps tasks: compare shares, not absolute sums; use bench.benchmark_stages for real latency)",
        "",
        f"{'type':<28} {'ms':>8} {'%':>6} {'n':>5}",
    ]
    for t, ms, n in by_type(layers)[:top]:
        lines.append(f"{t:<28} {ms:>8.2f} {100 * ms / total:>6.1f} {n:>5}")
    lines += ["", f"{'slowest layers':<70} {'ms':>8}"]
    for layer in layers[:top]:
        lines.append(f"{(layer.node_type + ' ' + layer.name)[:70]:<70} {layer.ms:>8.2f}")
    return "\n".join(lines)
