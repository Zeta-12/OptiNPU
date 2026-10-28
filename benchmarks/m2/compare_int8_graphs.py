"""Compare INT8 graphs (PTQ vs QAT): NPU latency on an idle machine and where the FakeQuantize ops sit.

    python -m benchmarks.m2.compare_int8_graphs name=path.xml [name=path.xml ...]
"""

from __future__ import annotations

import re
import sys
from collections import Counter

import openvino as ov

from optinpu.bench import benchmark


def fq_inputs(model: ov.Model) -> Counter:
    """Which op types consume FakeQuantize outputs (where the quantizers sit)."""
    where = Counter()
    for op in model.get_ordered_ops():
        if op.get_type_name() != "FakeQuantize":
            continue
        for out in op.outputs():
            for target in out.get_target_inputs():
                where[target.get_node().get_type_name()] += 1
    return where


def main():
    core = ov.Core()
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        model = core.read_model(path)
        ops = Counter(o.get_type_name() for o in model.get_ops())
        lat = benchmark(model, "npu", iterations=100, throughput_seconds=1).latency_ms_median
        consumers = ", ".join(f"{k} {v}" for k, v in fq_inputs(model).most_common(6))
        print(f"{name:<22} NPU {lat:5.2f} ms | FakeQuantize {ops.get('FakeQuantize', 0):3d} | "
              f"Convolution {ops.get('Convolution', 0)} | FQ feed: {consumers}", flush=True)
        names = [re.sub(r"^__module\.", "", o.get_friendly_name())[:50]
                 for o in model.get_ordered_ops() if o.get_type_name() == "FakeQuantize"]
        print(f"{'':<22} first FQ names: {names[:3]}", flush=True)


if __name__ == "__main__":
    main()
