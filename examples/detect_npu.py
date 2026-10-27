"""L1: run a pretrained D-FINE on the NPU and compare against PyTorch and CPU."""

import sys
import urllib.request
from pathlib import Path

from optinpu import Detector
from optinpu.bench import format_table

name = sys.argv[1] if len(sys.argv) > 1 else "dfine-s"
img = Path("runs/assets/coco_000000039769.jpg")
if not img.exists():
    img.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("http://images.cocodataset.org/val2017/000000039769.jpg", img)

model = Detector(name)
for device in ("torch", "cpu", "npu"):
    print(f"{device:>5}: {model.predict(img, device=device)}")

model.predict(img, device="npu").draw(__import__("PIL.Image").Image.open(img)).save(f"runs/{name}_npu.jpg")
print(format_table(model.benchmark(iterations=30, throughput_seconds=2)))
