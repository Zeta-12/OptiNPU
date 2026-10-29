"""Run any baseline IR with its own pre/post-processing (now optinpu.tasks.DetectionTask).

The M0 harness had its own implementation; P0 checked that the library gives identical detections
on all 9 baselines before replacing it (PLAN.md, P0).
"""

from __future__ import annotations

from PIL import Image

from benchmarks.m0.baselines import Baseline
from optinpu.tasks import Deployed


def prepare_input(baseline: Baseline, image: Image.Image):
    """(model input, geometry needed to map boxes back) with the baseline's own pre-processing."""
    inputs, (_, geometry) = baseline.task.inputs(image)
    return inputs[0], geometry


class IRDetector(Deployed):
    """predict(source, device, precision, conf) for a Baseline IR, as optinpu.evaluate expects.

    device: "cpu", "gpu", "npu", "npu+gpu"; config: extra NPU compile properties (e.g. {"NPU_TILES": 1});
    precision selects the IR ("fp16" or "int8").
    """

    def __init__(self, baseline: Baseline, device: str, config: dict | None = None, precision: str = "fp16"):
        self.baseline = baseline
        super().__init__(baseline.ir_path(precision), baseline.task, device, config, baseline.name, baseline.split)
