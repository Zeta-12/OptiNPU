"""M3: COCO train2017 shrunk to 512 px on the longest side, for training on Kaggle's 4 CPUs. Runs on Kaggle
(CPU only, no GPU quota):

    kaggle kernels push -p benchmarks/m3/kaggle_resize

JPEG decoding bounds OptiDet training there; a 512 px copy has 36 % fewer pixels than COCO's 640 px and
still covers training at 416 or 512 px. Output: /kaggle/working/train2017_512.zip (stored, not compressed:
JPEGs do not compress), fed to the training kernel through its kernel_sources. Annotations are unchanged:
CocoDetection scales the boxes to the image it reads.
"""

import io
import os
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

MAX_SIDE = 512


def find_dir(name: str) -> Path:
    for top, dirs, _ in os.walk("/kaggle/input", followlinks=True):
        if name in dirs:
            return Path(top) / name
        dirs[:] = [d for d in dirs if d not in ("val2017", "test2017")]
    raise FileNotFoundError(name)


def shrink(src: Path) -> tuple[str, bytes]:
    image = Image.open(src)
    if max(image.size) > MAX_SIDE:
        image.draft("RGB", (MAX_SIDE, MAX_SIDE))
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    else:
        image = image.convert("RGB")
    out = io.BytesIO()
    image.save(out, "JPEG", quality=92)
    return src.name, out.getvalue()


def main():
    for _ in range(80):  # inputs can be mounted late
        try:
            src = find_dir("train2017")
            break
        except FileNotFoundError:
            time.sleep(15)
    files = sorted(src.iterdir())
    print(f"{len(files)} images in {src}", flush=True)
    t, total = time.time(), 0
    with zipfile.ZipFile("/kaggle/working/train2017_512.zip", "w", zipfile.ZIP_STORED) as z, Pool(4) as pool:
        for i, (name, data) in enumerate(pool.imap_unordered(shrink, files, chunksize=64)):
            z.writestr(f"train2017/{name}", data)
            total += len(data)
            if i % 10000 == 0:
                print(f"{i} images, {total / 1e9:.2f} GB, {time.time() - t:.0f} s", flush=True)
    print(f"done: {len(files)} images, {total / 1e9:.2f} GB in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
