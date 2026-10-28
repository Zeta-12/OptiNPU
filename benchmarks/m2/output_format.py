"""How much does the detector's output format cost on the NPU? (dense scores are 1.7 MB at 512 px)

    python -m benchmarks.m2.output_format [backbone neck res]

Variants of the same deployed OptiDet:
  dense       scores (1, A, 80) sigmoid + boxes (1, A, 4)          -- current contract
  max_class   best score (1, A) + class (1, A) + boxes                -- reduction in the graph
  max_logit   best logit (1, A) + class (1, A) + boxes, no sigmoid    -- sigmoid on the host, on top-k only
  *_f16       same with FP16 output tensors (half the transfer, no FP16 -> FP32 conversion on the NPU side)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import openvino as ov
import torch
from torch import nn

from benchmarks.m2.candidates import BACKBONES, build


class Output(nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self.model, self.mode = model, mode

    def forward(self, x):
        m = self.model
        cls, reg = m.flatten(m.one2one(m.features(x)))
        boxes = m.boxes(m.distances(reg)) / m.scale
        if self.mode == "dense":
            return cls.sigmoid(), boxes
        best, label = cls.max(-1)
        if self.mode == "max_class":
            best = best.sigmoid()
        return best, label.to(torch.int32), boxes


def latency(ir_model, config, iters=60):
    core = ov.Core()
    compiled = core.compile_model(ir_model, "NPU", {"PERFORMANCE_HINT": "LATENCY", **config})
    request = compiled.create_infer_request()
    x = np.random.default_rng(0).random(tuple(compiled.input(0).get_shape()), dtype=np.float32)
    for _ in range(5):
        request.infer([x])
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        request.infer([x])
        times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times))


def main():
    name, neck, res = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else ("tiny", 64, 512)
    torch.manual_seed(0)
    model = build(*BACKBONES[name], neck, res)
    x = torch.zeros(1, 3, res, res)
    print(f"OptiDet {name} neck {neck} at {res}: {model.num_anchors} anchors")
    for mode in ("dense", "max_class", "max_logit"):
        with torch.no_grad():
            ir = ov.convert_model(Output(model, mode).eval(), example_input=x, input=[tuple(x.shape)])
        variants = {mode: ir}
        f16 = ov.preprocess.PrePostProcessor(ir.clone())
        for i, out in enumerate(ir.outputs):
            if out.get_element_type() == ov.Type.f32:
                f16.output(i).tensor().set_element_type(ov.Type.f16)
        variants[f"{mode}_f16"] = f16.build()
        for label, m in variants.items():
            row = [f"{latency(m, cfg):6.2f}" for cfg in ({"NPU_TILES": 1}, {})]
            print(f"  {label:<14} FP16 1t {row[0]} ms | 2t {row[1]} ms", flush=True)


if __name__ == "__main__":
    main()
