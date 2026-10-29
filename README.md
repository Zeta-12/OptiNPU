# optinpu

Train object detectors anywhere, run them on the Intel NPU.

```python
from optinpu import Detector

model = Detector("dfine-s")
dets = model.predict("image.jpg", device="npu")   # also "cpu", "gpu", "npu+gpu", "torch"
print(dets)                                      # Detections(5: cat 0.96, cat 0.95, ...)
print(model.benchmark())                         # latency / throughput on CPU, GPU, NPU
```

```
optinpu devices
optinpu models
optinpu predict dfine-n image.jpg --device npu --save out.jpg
optinpu bench dfine-s --devices cpu,gpu,npu --stages
optinpu export dfine-n --precision int8          # calibrates on 300 COCO val images
optinpu eval dfine-n --devices cpu,npu --precisions fp16,int8 --images 500
optinpu profile dfine-n --device npu
```

## Profile and optimize your own model for the NPU

Bring any model (OpenVINO IR, ONNX or PyTorch). `optinpu` measures where its NPU latency goes, checks
that the NPU computes the right thing, and applies the changes that make it faster without losing
accuracy. Every number is a measurement on the device, never an estimate.

```
optinpu profile yolo26n.xml --measured
optinpu optimize yolo26n.xml --task dets --letterbox --images guard_images/ \
    --annotations instances.json images/ --calibration calib_images/ --out yolo26n_npu/
```

```python
from optinpu import optimize
from optinpu.data import CocoDetection
from optinpu.optimizer import load_deployment
from optinpu.tasks import Deployed, DetectionTask

task = DetectionTask((640, 640), output="dets", resize="letterbox")   # how to feed / decode the model
report = optimize("yolo26n.xml", task, guard_images, calibration=calib_images,
                  evaluation=CocoDetection("instances.json", "images/").to_subset(), ap_budget=0.005)
print(report)                       # 25.9 ms -> 8.5 ms (3.0x), AP 44.8 = CPU FP32 (YOLO26-N)
report.save("yolo26n_npu/")         # report.md, report.json, the optimized deployment (part*.xml)
model = Deployed(load_deployment("yolo26n_npu/"), task)
model.predict("image.jpg")
```

**Profiler** (`profile_measured`, `optinpu profile --measured`). For each module region and each op type,
the latency saved when it is bypassed, measured in paired, randomized rounds until the 95 % confidence
interval is narrow. The NPU's own counters overlap (they sum to 3-10x the real latency) and an additive
per-op cost model ignores fusions: on 6 models, acting on their first pick loses 26 % and 40 % of the
best possible saving, against 0 % for the measured profile.

**Optimizer** (`optimize`, `optinpu optimize`). A greedy loop over measured candidates, each checked for
accuracy against the CPU FP32 original (detection F1 on guard images, or COCO AP on an annotated set):

| Kind | Rewrites |
|---|---|
| exact | final selection (TopK / NMS and after) on the CPU; DETR decoder on the iGPU / CPU; NPU tiles; depthwise as dense conv |
| approximate (AP budget) | INT8 post-training quantization (both NNCF schemes); latency-guided mixed precision when full INT8 misses the budget |
| retraining (reported) | SiLU / GELU -> ReLU, LayerNorm -> BatchNorm, 50-75 % weight pruning: latency measured before any training |

Models that are wrong or crash on the NPU (e.g. YOLO26-S losing the device on 2 tiles) are repaired
first. Every compilation and inference runs in a watched child process (`optinpu.sandbox`): a compiler
using 50 GB or a lost NPU cannot take your process down.

Results on this machine (Core Ultra 5 235U, NPU 3720; AP on COCO val2017 [0, 500), decisions taken on
other images):

| Model | Before | After | Applied | AP CPU / before / after |
|---|---|---|---|---|
| YOLO26-N | 25.9 ms | **8.5 ms** | selection on the CPU | 44.8 / 40.7 / 44.8 |
| YOLO26-S | 33.0 ms, loses the NPU | **13.7 ms** | selection on the CPU + mixed INT8 | 50.8 / - / 50.6 |
| D-FINE-N | 25.4 ms | **15.0 ms** | decoder on the iGPU | 45.8 / 45.6 / 45.6 |

## Training

```python
model = Detector("dfine-n")
model.train(("train.json", "images/"), ("val.json", "images/"), epochs=12, batch=8)
model.predict("image.jpg", device="npu")               # trained weights, exported on demand

model = Detector.from_checkpoint("dfine-n", "runs/dfine-n/train/best.pt")
```

```
optinpu train dfine-n --train train.json images/ --val val.json images/ --epochs 12
optinpu predict dfine-n image.jpg --checkpoint runs/dfine-n/train/best.pt --device npu
```

Datasets are COCO-format (images folder + instances json). New classes re-create the class heads.
Training runs on the Intel iGPU (PyTorch XPU) when available, otherwise CUDA or CPU. The first
iGPU run JIT-compiles kernels (~17 min for D-FINE); they are cached on disk afterwards.
`examples/finetune_animals.py` fine-tunes on 10 COCO animal classes (62.6 AP).

**Frozen backbone on the NPU.** `train(..., backbone_device="npu")` freezes the backbone, runs it on
the NPU and computes the next batch's features there while the iGPU trains the rest: 2x faster
fine-tuning (10.4 vs 5.1 img/s for D-FINE-N) for the same accuracy (62.1 vs 62.6 AP on the
animals example). `freeze_backbone=True` alone gives 7.4 img/s.

Computing gradients on the NPU itself was explored and dropped (slower than the CPU on
convolutions, fragile compiler); see PLAN.md, L5.

**Quantization-aware training (INT8 without accuracy loss).** Post-training INT8 costs OptiDet-N
1.5 AP; a short QAT fine-tuning of the trained model recovers all of it (10.0 AP in INT8 on the
NPU, same as FP32, 3.4 ms):

```python
model = Detector.from_checkpoint("optidet-n", "runs/optidet-n/train/best.pt")
model.train_qat(("train.json", "images/"), ("val.json", "images/"))   # 2 epochs, lr 1e-4
model.predict("image.jpg", device="npu")    # INT8 by default from now on
```

```
optinpu qat optidet-n --checkpoint runs/optidet-n/train/best.pt --train train.json images/ --val val.json images/
```

NNCF inserts fake quantizers into the fused model, training continues through them, and the INT8
IR is exported straight from the trained quantizers. The QAT checkpoint (`<run>/qat/best.pt`)
reloads with `from_checkpoint`. Supported by OptiDet; the DETR models gain nothing from INT8 on
this NPU (PLAN.md, L2).

## Data

COCO data (annotations + only the needed images) is downloaded on demand to
`~/.cache/optinpu/coco` (override with `OPTINPU_DATA`). Evaluation uses val2017 images
`[0, N)` sorted by id; INT8 calibration uses `[4500, 4800)`, so the two never overlap.

## Install

Python 3.10 to 3.13 (3.14 is not yet supported by OpenVINO / PyTorch XPU).

```
py -3.12 -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
.venv\Scripts\pip install -e ".[dev]"
```

## How models reach the NPU

`Detector.export()` produces a static-shape OpenVINO IR that is correct on the NPU:

1. **Freeze** input-independent tensors (sin-cos position embeddings, anchors) as FP32 constants,
   clipping float32-max sentinels that would become `inf` in FP16.
2. **Patch** modules the NPU compiler mishandles with exact rewrites (`optinpu.npu_patches`).
3. **Fold** remaining constant subgraphs before FP16 weight compression.

## Hybrid NPU + GPU

`device="npu+gpu"` runs the backbone and encoder (large convolutions, what the NPU is good at) on
the NPU and the DETR decoder (hundreds of small, irregular ops) on the iGPU. Intermediate tensors
are handed over without copies. It is the fastest placement for every registered model:

| model (640×640) | CPU | GPU | NPU | **NPU+GPU** | AP (500 COCO val images) |
|---|---|---|---|---|---|
| `rtdetrv2-r18` | 227 ms | 44.0 ms | 81.0 ms | **33.9 ms** | 50.8 |
| `dfine-s` | 128 ms | 40.7 ms | 46.2 ms | **33.6 ms** | 51.5 |
| `dfine-n` | 48 ms | 16.6 ms | 25.2 ms | **15.9 ms** | 45.7 |

NPU column measured with 2 NPU tiles (D-FINE) or 1 (RT-DETRv2). With the current default of 1
tile for all DETR models, D-FINE-N on the NPU alone takes ~43 ms.

## Checking a model: npu_lint

```
optinpu lint dfine-n --image image.jpg      # registered model
optinpu lint path/to/model.xml              # any OpenVINO IR
```

Static rules (dynamic shapes, float tensors computed from shapes, FP16 range, pooling patterns the
NPU compiler mishandles, slow ops), then compilation on the NPU (locating the failing op if any),
an accuracy check against CPU FP32 and, if outputs drift, an automatic retry with `NPU_TILES=1`.

When a model misbehaves on the NPU, `optinpu.diagnose` finds the culprit:

- `locate_divergence_split(ir, inputs)`: runs ops[:k] on the NPU and the rest on CPU, bisects k.
  The prefix is compiled as a whole, so compiler fusions and tiling are reproduced.
- `locate_divergence(ir, inputs)`: first op whose NPU output differs from CPU (per-op probes;
  exposing outputs can hide fusion bugs).
- `locate_compile_failure(ir)`: first op that makes the NPU compiler fail.

## Measuring

- `Detector.evaluate()` / `optinpu eval`: COCO bbox mAP (pycocotools) on any device and precision.
- `Detector.benchmark()`: batch-1 latency and async throughput.
- `Detector.benchmark_stages()`: real latency of cumulative stages (backbone, +encoder, full),
  measured on truncated graphs.
- `optinpu.profile`: per-layer times. On the NPU, tasks overlap, so use shares, not sums.

## Models

| name | source | NPU |
|---|---|---|
| `dfine-n` | D-FINE nano, COCO | ok |
| `dfine-s` | D-FINE small, COCO | ok |
| `rtdetrv2-r18` | RT-DETRv2 R18, COCO | ok |
| `optidet-n` | OptiDet prototype, ImageNet RepVGG-A0 backbone only: train it first | ok, 2 tiles, INT8 |

On the NPU alone, all DETR models compile with `NPU_TILES=1`: splitting the decoder attention over
2 tiles is wrong in the NPU compiler (-3.6 AP on a fine-tuned D-FINE-N, unusable RT-DETRv2). One
tile is exact but slower; `npu+gpu` is both exact and the fastest.

## Tests

```
.venv\Scripts\python -m pytest       # 87 tests, ~45 s; NPU tests are skipped without an NPU
.venv\Scripts\ruff check src tests examples
```
