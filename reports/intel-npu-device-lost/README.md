# [NPU] `ZE_RESULT_ERROR_DEVICE_LOST` during inference of YOLO26-S with default multi-tile compilation (NPU 3720, Windows)

## Summary

Running a YOLO26-S detector (static shape, end-to-end NMS-free head) on the NPU with **default
compile properties** loses the device after a few inferences on **different** real images:
`zeCommandQueueExecuteCommandLists` returns `ZE_RESULT_ERROR_DEVICE_LOST` ("device hung, reset, was
removed, or driver update occurred"). The process must be restarted; the NPU itself recovers within
seconds (it is listed again by `ov.Core().available_devices`).

The failure is reproducible with a standalone script (OpenVINO + NumPy + Pillow only,
[`repro.py`](repro.py)). It does **not** occur with `NPU_TILES=1`.

## Environment

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 5 235U |
| NPU | Intel AI Boost, `DEVICE_ARCHITECTURE` 3720, `NPU_MAX_TILES` 2 |
| NPU driver | 32.0.100.4778 (2026-04-28), `NPU_DRIVER_VERSION` 1004778 |
| NPU compiler | `NPU_COMPILER_TYPE` DRIVER, `NPU_COMPILER_VERSION` 524289 |
| OpenVINO | 2026.4.1-22982-e213a147257-releases/2026/4 (pip) |
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.12.10 |
| GPU driver (not involved) | Intel Graphics 32.0.101.8860 |
| Power | AC |

## Steps to reproduce

1. Export YOLO26-S to ONNX with the end-to-end head (ultralytics 8.4.190; `nms=False` is required
   to get the NMS-free one-to-one head, output `[1, 300, 6]`):

   ```python
   from ultralytics import YOLO
   YOLO("yolo26s.pt").export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17, nms=False)
   ```

2. Get COCO val2017 images (any directory of real photos should do; the runs below use val2017 in
   file-name order).

3. Run:

   ```
   pip install openvino==2026.4.1 numpy pillow
   python repro.py yolo26s.onnx --image-dir path/to/val2017 --count 20
   ```

The script compiles the model on `NPU` with no properties, then runs one synchronous inference per
image (letterbox to 640x640, RGB, /255).

## Observed

```
compiled on NPU with default properties in 1.6 s
   0 000000000139.jpg: ...
   ...
   4 000000000776.jpg: 49.2 ms, max score 0.939
[ERROR] 22:45:20.547 [ZeHostMem] L0 zeMemFree result: ZE_RESULT_ERROR_DEVICE_LOST, code 0X70000001 - device hung, reset, was removed, or driver update occurred
RuntimeError: Exception from src\inference\src\cpp\infer_request.cpp:224:
Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:354:
L0 zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001 - device hung, reset, was removed, or driver update occurred
```

Nothing is written to the Windows System event log and no LiveKernelReport is created.

## Expected

All images are processed, as with `NPU_TILES=1` and on GPU (the same model evaluates correctly on
the 5,000 COCO val2017 images on GPU and on NPU with 1 tile: 47.7 AP).

## Characterization

Each line is a fresh process (`repro.py` with the given options).

| Input sequence | Tiles | Result |
|---|---|---|
| val2017 images from index 0 | default (2) | device lost at the 6th inference (`000000000785.jpg`), 3 runs out of 3 (FP16 and FP32 weights) |
| val2017 images from index 1 | default (2) | device lost at the 5th inference (`000000000785.jpg` again) |
| val2017 images from index 10 | default (2) | device lost at the 7th inference (another image) |
| val2017 images from index 0, 50 images | 1 | OK |
| `000000000785.jpg` alone, repeated 3 times | default (2) | OK (twice) |
| `000000000785.jpg` alone, repeated 3 times | 1 | OK |
| one image repeated 5 times; random, zeros or gray input repeated 5 times | default (2) | OK |
| 100 latency iterations on random input, `PERFORMANCE_HINT` LATENCY and THROUGHPUT | default (2) | OK (33 ms per inference) |

Variants that make no difference to the image-sequence failure: FP32 weights (ONNX read directly)
or FP16-compressed weights; with or without `PERFORMANCE_HINT=LATENCY`; with or without `CACHE_DIR`.
With a repeated single input, FP32, FP16, `PERFORMANCE_HINT=LATENCY` and `CACHE_DIR` (both the
compiling and the blob-importing run) all pass.

In short: the hang needs (a) the multi-tile compilation and (b) a few inferences on **varying**
input data; no single image triggers it alone.

The model has two attention blocks (`Softmax` in `/model.10/m/m.0/attn` and
`/model.22/m.0/m.0.1/attn`) and a data-dependent selection in the end-to-end head (`/model.23`:
2 `TopK`, 3 `GatherElements`); the rest is 94 convolutions. As a pure guess, a data-dependent path
executed across both tiles might be where the hang happens; we could not narrow it further from
the outside.

## Workaround

Compile with `{"NPU_TILES": 1}` (about 41 ms instead of 33 ms per inference for this model).

## Update (2026-10-29): the selection tail is involved

- **Moving the end-to-end selection out of the NPU graph removes the hang.** We split the IR right
  before the first `TopK` of `/model.23`: the 16 ops that depend on it (`TopK` ×2,
  `GatherElements` ×3, `Divide`, `FloorMod`, `Broadcast`, `Concat`…) run on the CPU, everything else
  on the NPU with default (2-tile) compilation. 500 consecutive COCO val2017 images run without any
  error, with detections identical to CPU FP32 (AP 50.7 vs 50.8 on those images), at 15.7 ms per
  inference instead of 32.6 ms for the full graph on 2 tiles.
- **The tail alone does not hang**: compiled alone on the NPU (2 tiles), fed with the head outputs
  of 30 real images, it runs fine. The hang needs the tail compiled together with the rest of the
  network.
- **A modified YOLO26-N also hangs**: with the head's `Sigmoid` bypassed (an experiment of our
  profiler; the graph is otherwise unchanged), YOLO26-N loses the device in the same way within the
  first inferences on random inputs, whereas the original YOLO26-N does not. The values reaching
  the selection seem to matter.
- **The same split also fixes the silent accuracy drop of YOLO26-N** (next section): AP 44.8 with
  the tail on the CPU and 2 NPU tiles, vs 40.7 with the full graph on 2 tiles and 44.8 on CPU
  (val2017 images [0, 500)). The multi-tile accuracy issue of YOLO26-N is in the selection tail too.
- Vendor counters (`PERF_COUNT`) put most of YOLO26-N's NPU time in that tail, on SHAVE: one
  `Divide` ~19.5 ms and one `GatherElements` ~15.4 ms (cumulative task times), for a 25.9 ms
  inference. That cost alone may deserve a look.

## Related observation (possibly a separate issue)

With default (multi-tile) compilation, several models also produce **degraded results without any
error**, while `NPU_TILES=1`, CPU and GPU agree. COCO val2017 AP, 5,000 images, same FP16 IR:

| Model | NPU, 1 tile | NPU, default (2 tiles) |
|---|---|---|
| YOLO26-N (end-to-end head) | 40.2 | 35.7 |
| RT-DETRv2-R18 | 48.0 | 44.3 |
| DEIMv2-N | 42.9 | 42.0 |
| D-FINE-N / D-FINE-S | 42.8 / 48.4 | 42.6 / 48.4 |

For RT-DETRv2, bisecting the graph (prefix on NPU, suffix on CPU) points to the first decoder
self-attention (`ScaledDotProductAttention`); outputs are bit-identical whether attention is left as
SDPA or decomposed into MatMul/Softmax/MatMul in the IR. We can provide the IRs and the bisection
script if useful.
