"""L0 smoke test: export a ResNet-50 to OpenVINO and benchmark it on CPU, GPU and NPU."""

import numpy as np
import torch
import torchvision

import optinpu
from optinpu.bench import format_table

print("Devices:", optinpu.available_devices())
print("Training device:", optinpu.torch_device())

model = torchvision.models.resnet50(weights=None).eval()
xml = optinpu.export_openvino(model, "runs/l0/resnet50.xml", input_shape=(1, 3, 224, 224))

# Numerical check: NPU (FP16) vs PyTorch (FP32)
x = np.random.default_rng(0).random((1, 3, 224, 224), dtype=np.float32)
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()
out = optinpu.compile_model(xml, "npu")(x)[0]
print(f"NPU vs torch: max abs diff {np.abs(out - ref).max():.4f}, top-1 match {out.argmax() == ref.argmax()}")

results = [optinpu.benchmark(xml, d, iterations=50) for d in optinpu.available_devices()]
print(format_table(results))
