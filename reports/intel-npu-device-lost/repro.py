"""Minimal reproducer: Intel NPU (arch 3720) device lost while running YOLO26-S with the default
multi-tile compilation. Depends only on openvino, numpy and pillow.

Model: YOLO26-S, exported to ONNX with ultralytics (end-to-end NMS-free head, output [1, 300, 6]):

    from ultralytics import YOLO
    YOLO("yolo26s.pt").export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17, nms=False)

Usage:
    # fails: device lost after a few different images (default properties = both NPU tiles)
    python repro.py yolo26s.onnx --image-dir coco/val2017 --count 20
    # passes: same images on one tile
    python repro.py yolo26s.onnx --image-dir coco/val2017 --count 50 --tiles 1
    # passes: one image (or random / zeros / gray input) repeated
    python repro.py yolo26s.onnx --input image --image coco/val2017/000000000785.jpg --runs 5

--tiles default: no NPU_TILES property (the plugin uses both tiles); 1 or 2: NPU_TILES set explicitly.
--input: image (letterboxed RGB, /255, the standard YOLO pre-processing), gray (all 114/255),
         zeros, random (uniform [0, 1)).
--image-dir: run once on each image of a directory, in name order (COCO val2017 = id order).
"""

import argparse
import time

import numpy as np
import openvino as ov
from PIL import Image


def letterbox(path: str, size: int = 640) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    scale = min(size / image.width, size / image.height)
    nw, nh = round(image.width * scale), round(image.height * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def make_input(kind: str, image: str | None) -> np.ndarray:
    shape = (1, 3, 640, 640)
    if kind == "image":
        return letterbox(image)
    if kind == "gray":
        return np.full(shape, 114 / 255, dtype=np.float32)
    if kind == "zeros":
        return np.zeros(shape, dtype=np.float32)
    return np.random.default_rng(0).random(shape, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help=".onnx, or an OpenVINO .xml IR")
    parser.add_argument("--input", choices=["image", "gray", "zeros", "random"], default="image")
    parser.add_argument("--image")
    parser.add_argument("--tiles", choices=["default", "1", "2"], default="default")
    parser.add_argument("--hint", choices=["none", "LATENCY", "THROUGHPUT"], default="none")
    parser.add_argument("--cache-dir", help="enable model caching (CACHE_DIR) in this directory")
    parser.add_argument("--fp16", action="store_true", help="compress weights to FP16 first (as ov.save_model does)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--image-dir", help="run once on each of the first --count images of this directory "
                                            "(sorted by name, i.e. by id for COCO val2017)")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--start", type=int, default=0, help="index of the first image in --image-dir")
    args = parser.parse_args()

    core = ov.Core()
    print("OpenVINO", ov.__version__, "| NPU driver", core.get_property("NPU", "NPU_DRIVER_VERSION"), flush=True)
    model = core.read_model(args.model)  # static [1, 3, 640, 640]
    if args.fp16:
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp()) / "model_fp16.xml"
        ov.save_model(model, tmp, compress_to_fp16=True)
        model = core.read_model(tmp)
    config = {} if args.tiles == "default" else {"NPU_TILES": int(args.tiles)}
    if args.hint != "none":
        config["PERFORMANCE_HINT"] = args.hint
    if args.cache_dir:
        config["CACHE_DIR"] = args.cache_dir
    t = time.perf_counter()
    compiled = core.compile_model(model, "NPU", config)
    print(f"compiled on NPU with {config or 'default properties'} in {time.perf_counter() - t:.1f} s", flush=True)

    request = compiled.create_infer_request()
    if args.image_dir:
        from pathlib import Path

        paths = sorted(Path(args.image_dir).glob("*.jpg"))[args.start: args.start + args.count]
        for i, path in enumerate(paths):
            t = time.perf_counter()
            out = request.infer([letterbox(str(path))])[compiled.output(0)]
            print(f"{i:>4} {path.name}: {1e3 * (time.perf_counter() - t):.1f} ms, "
                  f"max score {out[0, :, 4].max():.3f}", flush=True)
        print("OK", flush=True)
        return
    x = make_input(args.input, args.image)
    for i in range(args.runs):
        t = time.perf_counter()
        out = request.infer([x])[compiled.output(0)]
        print(f"infer {i}: {1e3 * (time.perf_counter() - t):.1f} ms, output {out.shape}, "
              f"max score {out[0, :, 4].max():.3f}", flush=True)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
