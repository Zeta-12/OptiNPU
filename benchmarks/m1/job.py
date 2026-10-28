"""Measure M1 blocks on the NPU; one JSON line per block in runs/m1/results.jsonl.

    python -m benchmarks.m1.job S2 [name-substring ...]

For each block not measured yet: export (FP32 IR, the NPU computes in FP16), optional INT8
quantization (NNCF, random calibration: latency only), then on the NPU with 1 tile and with the
compiler default (2 tiles): median latency, and exactness (1 tile vs CPU FP32; 2 tiles vs 1 tile,
which isolates the tile-split defects). A "started" line is written before each block, so a block
that kills the process (memory watchdog, NPU device lost) is skipped when the job is restarted.
Exits with code 3 after an NPU device loss, so that run.py restarts a fresh process.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch

from benchmarks.m1.blocks import SUITES, Block, count_flops

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "m1"
RESULTS = WORK / "results.jsonl"
CONFIGS = {"1t": {"NPU_TILES": 1}, "2t": {}}


def measured() -> set[str]:
    """Names of blocks already done or started (a started block without result crashed the job)."""
    if not RESULTS.exists():
        return set()
    return {json.loads(line)["name"] for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()}


def append(record: dict) -> None:
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def latency_ms(request, inputs, min_iters=10, max_iters=60, budget_s=1.5) -> float:
    for _ in range(3):
        request.infer(inputs)
    times, start = [], time.perf_counter()
    while len(times) < min_iters or (len(times) < max_iters and time.perf_counter() - start < budget_s):
        t = time.perf_counter()
        request.infer(inputs)
        times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times))


def run_block(block: Block, core: ov.Core) -> dict:
    from optinpu.diagnose import _rel_error
    from optinpu.export import export_openvino
    from optinpu.quantize import quantize_int8

    torch.manual_seed(0)
    module = block.build().eval()
    record = {"suite": block.suite, "name": block.name, "params": block.params,
              "input_shape": list(block.input_shape), "flops": count_flops(module, block.input_shape)}
    irs = {"fp16": export_openvino(module, WORK / "ir" / f"{block.name}.xml", block.input_shape, fp16=False)}
    rng = np.random.default_rng(0)
    x = rng.standard_normal(block.input_shape).astype(np.float32)
    if block.int8:
        calib = [rng.standard_normal(block.input_shape).astype(np.float32) for _ in range(8)]
        irs["int8"] = quantize_int8(irs["fp16"], calib, lambda a: a, WORK / "ir" / f"{block.name}_int8.xml",
                                    subset_size=len(calib), transformer=block.transformer)

    cpu = core.compile_model(irs["fp16"], "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    reference = list(cpu.create_infer_request().infer([x]).values())
    results = {}
    for precision, ir in irs.items():
        one_tile = None
        for tiles, config in CONFIGS.items():
            key = f"{precision}-{tiles}"
            try:
                t = time.perf_counter()
                compiled = core.compile_model(ir, "NPU", {"PERFORMANCE_HINT": "LATENCY", **config})
                compile_s = time.perf_counter() - t
                request = compiled.create_infer_request()
                out = list(request.infer([x]).values())
                entry = {"ms": latency_ms(request, [x]), "compile_s": round(compile_s, 2)}
                if tiles == "1t":
                    one_tile = out
                    if precision == "fp16":
                        entry["err_vs_cpu"] = max(_rel_error(r, o) for r, o in zip(reference, out))
                elif one_tile is not None:
                    entry["err_vs_1t"] = max(_rel_error(r, o) for r, o in zip(one_tile, out))
            except RuntimeError as e:
                message = str(e).strip().splitlines()[-1][:300]
                entry = {"error": message}
                if "DEVICE_LOST" in str(e):
                    results[key] = entry
                    record["results"] = results
                    append(record)
                    print(f"{block.name}: NPU device lost", flush=True)
                    sys.exit(3)
            results[key] = entry
    record["results"] = results
    return record


def main():
    suite, filters = sys.argv[1], sys.argv[2:]
    WORK.mkdir(parents=True, exist_ok=True)
    done = measured()
    core = ov.Core()
    for block in SUITES[suite]():
        if block.name in done or (filters and not any(f in block.name for f in filters)):
            continue
        append({"name": block.name, "suite": suite, "status": "started"})
        t = time.perf_counter()
        record = run_block(block, core)
        record["seconds"] = round(time.perf_counter() - t, 1)
        append(record)
        summary = " ".join(f"{k} {v['ms']:.2f}" if "ms" in v else f"{k} ERR" for k, v in record["results"].items())
        print(f"{block.name}: {summary}", flush=True)
    print("SUITE DONE", flush=True)


if __name__ == "__main__":
    main()
