"""Tour of the profiler and optimizer on a toy detector (a few minutes, nothing to download).

The toy network has what makes real detectors slow or wrong on the NPU: SiLU activations, a depthwise
convolution, a LayerNorm over tokens, and an in-graph top-k selection of the best scores (as end-to-end
YOLO heads do). Its weights are random: on the NPU, dense weight values barely change latency (step 3).
Without annotations, the accuracy guard is the agreement of raw outputs with the CPU FP32 original.

    python examples/profile_and_optimize.py [cpu|npu]      # about 2 minutes on the NPU

Step by step:
1. load a PyTorch model as a static-shape OpenVINO graph;
2. check that the device computes the same thing as the CPU in FP32;
3. check how much latency depends on weight values: dense values barely matter, which makes "what-if"
   measurements of untrained variants possible; zeros (pruning) may change it, either way;
4. profile: latency saved by bypassing each module region and each op type, with a 95 % interval;
5. optimize: measured rewrites under the accuracy guard, retraining rewrites timed before training;
6. save the optimized deployment and its report, reload it and run it.
"""

import sys
from pathlib import Path

import torch
from torch import nn

from optinpu import optimize, profile_measured
from optinpu.devices import has_npu
from optinpu.export import load_model
from optinpu.optimizer import load_deployment
from optinpu.profiler import Adaptive, measure, paired_saving, with_weights
from optinpu.sandbox import run_predictions
from optinpu.tasks import Deployed, OutputTask, random_samples

OUT = Path("runs/examples/toy_detector")


class ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU())
        self.block = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, groups=32), nn.Conv2d(32, 64, 1), nn.SiLU())
        self.down = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1), nn.SiLU())
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 80)  # 80 class scores per location, as on COCO

    def forward(self, x):
        tokens = self.down(self.block(self.stem(x))).flatten(2).transpose(1, 2)  # (1, 6400, 64)
        scores = self.head(self.norm(tokens)).sigmoid()
        best, _ = scores.amax(-1).topk(300, dim=1)  # final selection of 300 detections, inside the graph
        return best, tokens.mean(1)


def step(title: str) -> None:
    print(f"\n{'=' * 8} {title}", flush=True)


device = sys.argv[1] if len(sys.argv) > 1 else ("npu" if has_npu() else "cpu")
quick = dict(rounds=3, iterations=20)

step("1. Load the model")
torch.manual_seed(0)
model = load_model(ToyDetector(), (1, 3, 320, 320))
ops = [op.get_type_name() for op in model.get_ordered_ops()]
print(f"{len(ops)} ops, of which {ops.count('Convolution') + ops.count('GroupConvolution')} convolutions "
      f"and {ops.count('TopK')} TopK; target device: {device}")

step(f"2. Does the {device.upper()} compute the right thing?")
task = OutputTask()  # raw outputs; a DetectionTask or ClassificationTask compares detections or top-1 classes
samples = random_samples(model, 30)
reference, _ = run_predictions([(model, "cpu", None)], task, samples)
predictions, job = run_predictions([(model, device, None)], task, samples)
print(f"agreement with CPU FP32: {task.agreement(reference, predictions):.3f}" if predictions
      else f"failed: {job.error}")

step("3. How much does latency depend on weight values?")
timings = measure({"trained": model, "random weights": with_weights(model, "random"),
                   "50 % zero weights": with_weights(model, "sparse50")}, device, adaptive=Adaptive("trained", max_rounds=10),
                  **quick)
for name in ("random weights", "50 % zero weights"):
    saving, ci = paired_saving(timings, "trained", name)
    print(f"{name:<18} {timings[name].median_ms:6.3f} ms, saves {saving:+.3f} +- {ci:.3f} ms "
          f"vs trained ({timings['trained'].median_ms:.3f} ms)")
print("(random dense weights: within a few %, the rest is the data-dependent top-k; zero weights: the NPU "
      "may skip them or not, which is why pruning is measured, not assumed)")

step("4. Measured profile: what does each part cost?")
print(profile_measured(model, device, adaptive=Adaptive("full", max_rounds=10), log=lambda *a: None, **quick))

step("5. Optimize under the accuracy guard")
report = optimize(model, task, samples, device, **quick)
print(report)

step("6. Save, reload and run the optimized deployment")
report.save(OUT, "Toy detector")
parts = load_deployment(OUT)
print("parts:", [(path.name, part_device, config) for path, part_device, config in parts])
reloaded = Deployed(parts, task, cache_dir=None).run(samples)
print(f"agreement of the reloaded deployment with CPU FP32: {task.agreement(reference, reloaded):.3f}")
print(f"report: {OUT / 'report.md'}")
