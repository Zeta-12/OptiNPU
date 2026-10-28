# M0 — baselines on Core Ultra 5 235U (COCO val2017, 5000 images, batch 1)

Latency = model only, median ms. AP in %. `1t` = 1 NPU tile (exact for DETRs), `2t` = compiler default (2 tiles). INT8 = NNCF post-training quantization (300 calibration images).

## NPU only (the OptiDet target)

| Model | Res. | AP pub. | AP iGPU | FP16 1t AP / ms | FP16 2t AP / ms | INT8 1t AP / ms | INT8 2t ms | Lint 2t |
|---|---|---|---|---|---|---|---|---|
| dfine-n | 640 | 42.8 | 42.8 | 42.8 / 43.1 | 42.6 / 25.9 | 41.5 / 61.6 | 35.8 | ok |
| dfine-s | 640 | 48.5 | 48.5 | 48.4 / 78.6 | 48.4 / 47.0 | 47.5 / 93.2 | 53.7 | ok |
| rtdetrv2-r18 | 640 | 47.9 | 48.1 | 48.0 / 80.7 | 44.3 / 47.2 | 47.9 / 70.7 | 41.4 | ACCURACY (0.42) |
| rfdetr-n | 384 | 48.4 | 48.4 | 48.4 / 106.6 | 48.4 / 83.4 | 48.1 / 100.7 | 81.9 | ACCURACY (0.06) |
| rfdetr-s | 512 | 53.0 | 52.9 | 53.0 / 168.7 | 53.0 / 125.7 | 52.8 / 159.1 | 122.8 | ACCURACY (0.12) |
| deimv2-n | 640 | 43.0 | 42.9 | 42.9 / 41.2 | 42.0 / 23.9 | 41.5 / 58.3 | 33.1 | ok |
| deimv2-s | 640 | 50.9 | 46.9 | 47.0 / 172.4 | 46.2 / 101.0 | 46.4 / 169.2 | 101.2 | ACCURACY (0.10) |
| yolo26-n | 640 | 40.1 | 40.3 | 40.2 / 29.3 | 35.7 / 26.6 | 38.5 / 29.2 | 25.9 | ACCURACY (0.05) |
| yolo26-s | 640 | 47.8 | 47.7 | 47.7 / 41.1 | — / 33.0 | 45.1 / 35.7 | 29.8 | ACCURACY (0.21) |

## Context: other placements (latency ms; AP of the iGPU run)

| Model | CPU | iGPU | NPU+iGPU | Best placement overall | Best NPU-only | NPU-only / best |
|---|---|---|---|---|---|---|
| dfine-n | 45.4 | 16.8 | 15.5 | npu+gpu 15.5 ms (42.8 AP) | fp16 npu-2t 25.9 ms (42.6 AP) | 1.7× |
| dfine-s | 137.2 | 41.2 | 34.0 | npu+gpu 34.0 ms (48.5 AP) | fp16 npu-2t 47.0 ms (48.4 AP) | 1.4× |
| rtdetrv2-r18 | 226.0 | 44.0 | 34.9 | npu+gpu 34.9 ms (48.1 AP) | fp16 npu-2t 47.2 ms (44.3 AP) | 1.4× |
| rfdetr-n | 135.4 | 23.0 | 84.5 | gpu 23.0 ms (48.4 AP) | fp16 npu-2t 83.4 ms (48.4 AP) | 3.6× |
| rfdetr-s | 235.3 | 45.2 | 125.9 | gpu 45.2 ms (52.9 AP) | fp16 npu-2t 125.7 ms (53.0 AP) | 2.8× |
| deimv2-n | 38.2 | 15.1 | 19.9 | gpu 15.1 ms (42.9 AP) | fp16 npu-2t 23.9 ms (42.0 AP) | 1.6× |
| deimv2-s | 257.4 | 54.6 | 91.8 | gpu 54.6 ms (46.9 AP) | fp16 npu-2t 101.0 ms (46.2 AP) | 1.8× |
| yolo26-n | 30.7 | 10.9 | — | gpu 10.9 ms (40.3 AP) | fp16 npu-2t 26.6 ms (35.7 AP) | 2.4× |
| yolo26-s | 106.2 | 23.8 | — | gpu 23.8 ms (47.7 AP) | int8 npu 35.7 ms (45.1 AP) | 1.5× |

## Pareto front, NPU only (each model × configuration with measured AP)

| Configuration | Latency (ms) | AP | Front |
|---|---|---|---|
| deimv2-n fp16 npu-2t | 23.9 | 42.0 | ● |
| dfine-n fp16 npu-2t | 25.9 | 42.6 | ● |
| yolo26-n fp16 npu-2t | 26.6 | 35.7 |  |
| yolo26-n int8 npu | 29.2 | 38.5 |  |
| yolo26-n fp16 npu | 29.3 | 40.2 |  |
| yolo26-s int8 npu | 35.7 | 45.1 | ● |
| yolo26-s fp16 npu | 41.1 | 47.7 | ● |
| deimv2-n fp16 npu | 41.2 | 42.9 |  |
| dfine-n fp16 npu | 43.1 | 42.8 |  |
| dfine-s fp16 npu-2t | 47.0 | 48.4 | ● |
| rtdetrv2-r18 fp16 npu-2t | 47.2 | 44.3 |  |
| deimv2-n int8 npu | 58.3 | 41.5 |  |
| dfine-n int8 npu | 61.6 | 41.5 |  |
| rtdetrv2-r18 int8 npu | 70.7 | 47.9 |  |
| dfine-s fp16 npu | 78.6 | 48.4 | ● |
| rtdetrv2-r18 fp16 npu | 80.7 | 48.0 |  |
| rfdetr-n fp16 npu-2t | 83.4 | 48.4 |  |
| dfine-s int8 npu | 93.2 | 47.5 |  |
| rfdetr-n int8 npu | 100.7 | 48.1 |  |
| deimv2-s fp16 npu-2t | 101.0 | 46.2 |  |
| rfdetr-n fp16 npu | 106.6 | 48.4 |  |
| rfdetr-s fp16 npu-2t | 125.7 | 53.0 | ● |
| rfdetr-s int8 npu | 159.1 | 52.8 |  |
| rfdetr-s fp16 npu | 168.7 | 53.0 |  |
| deimv2-s int8 npu | 169.2 | 46.4 |  |
| deimv2-s fp16 npu | 172.4 | 47.0 |  |

## Tiers: what OptiDet has to beat

| Tier | AP target | Fastest reference (any placement) | Fastest reference (NPU only) | Target ÷4 | Target ÷5 |
|---|---|---|---|---|---|
| N | 43.0 | deimv2-n gpu: 15.1 ms, 42.9 AP | deimv2-n fp16 npu-2t: 23.9 ms, 42.0 AP | 3.8 ms | 3.0 ms |
| S | 48.5 | rfdetr-n gpu: 23.0 ms, 48.4 AP | yolo26-s fp16 npu: 41.1 ms, 47.7 AP | 5.8 ms | 4.6 ms |

## Notes

- **deimv2-s** (DEIMv2 License (non-commercial)): measured 46.9 AP on the iGPU vs 50.9 published. Pre-processing matches the official eval config (640x640 resize, ImageNet normalization); the ONNX export of the DINOv3 backbone (rotary position embedding computed from shapes) is the main suspect, not verified.
- **yolo26-n** (AGPL-3.0): published AP is the end-to-end (NMS-free) one
- **yolo26-s** (AGPL-3.0): published AP is the end-to-end (NMS-free) one
