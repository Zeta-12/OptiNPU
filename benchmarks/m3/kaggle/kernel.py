"""M3: train OptiDet on Kaggle GPUs (2 x T4). Runs on Kaggle, not locally:

    kaggle kernels push -p benchmarks/m3/kaggle --accelerator NvidiaTeslaT4 -t 43200

Inputs (kernel-metadata.json): the optinpu wheel, benchmarks/m2/train_optidet.py and benchmarks/m3/loader_speed.py,
uploaded as the private dataset julienbeloin/optinpu-code; COCO 2017 (awsaf49/coco-2017-dataset) for annotations
and validation; train2017 shrunk to 512 px (output of the kernel julienbeloin/coco-train-512, extracted to local
disk: less JPEG decoding on the 4 CPUs). A checkpoint to resume from can be added as a dataset, then
`--resume last.pt` (file names after --resume are searched under /kaggle/input). Outputs: runs/ and gpu_util.csv
in /kaggle/working, fetched with `kaggle kernels output`.
"""

import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# M3 recipe study: 15-epoch runs on all of train2017, compared at equal budget (one run per session).
# Run 1, baseline: the 72-epoch M3 recipe (29.5 AP on val2017) on a short schedule. Then the same with
# "--mosaic", "1.0", "--close-mosaic", "3" (run m3x_mosaic), then the higher-resolution variant.
# fp16, not bf16: bf16 is emulated on T4, 3.5x slower than FP32. One process per GPU (torchrun), --batch is
# global; 2 data workers per process on the 4 CPUs; color jitter on the GPUs to spare them.
GPUS = 2
COMMON = ["--images", "200000", "--epochs", "15", "--batch", "16", "--workers", "2", "--amp", "fp16",
          "--distill", "ustc-community/dfine-small-coco", "--color-jitter", "0.3", "--max-hours", "11"]
RUNS = [
    ["train_optidet.py", *COMMON, "--run", "m3x_base"],
]
SHRUNK = "train2017_512.zip"
IMAGE_DIRS = {"train2017", "val2017", "test2017"}


def find(name: str, wait_s: int = 1200) -> Path:
    """First file `name` under /kaggle/input, without walking the COCO image folders. Inputs can be symlinks
    and can be mounted late (two sessions saw coco2017/ still empty after 5 min): retry for `wait_s`."""
    deadline = time.time() + wait_s
    while True:
        for top, dirs, files in os.walk("/kaggle/input", followlinks=True):
            dirs[:] = [d for d in dirs if d not in IMAGE_DIRS]
            if name in files:
                return Path(top) / name
        if time.time() > deadline:
            break
        time.sleep(15)
    sh("find", "/kaggle/input", "-maxdepth", "6", "-not", "-name", "*.jpg", check=False)
    raise FileNotFoundError(name)


def sh(*cmd: str, check: bool = True, env: dict | None = None) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env=env).returncode


code = find("train_optidet.py").parent
wheel = next(p for p in code.iterdir() if p.suffix == ".whl")
sh(sys.executable, "-m", "pip", "install", "-q", f"{wheel}[pretrained]")
sh("nvidia-smi")
sh(sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.get_device_name(0), "
                        "torch.cuda.get_device_capability(0), 'bf16', torch.cuda.is_bf16_supported(), "
                        f"'cpus', {os.cpu_count()})")

# the Kaggle input is read-only and optinpu writes its subset jsons next to the images: link COCO into /tmp
# (not /kaggle/working, which is saved as the kernel output)
coco = find("instances_train2017.json").parent.parent
root = Path("/tmp/coco")
root.mkdir(exist_ok=True)
t = time.time()
with zipfile.ZipFile(find(SHRUNK)) as z:  # train2017/*.jpg, ~8 GB; local disk also reads faster than the input
    z.extractall("/tmp/coco512")
print(f"{SHRUNK} extracted in {time.time() - t:.0f} s", flush=True)
sources = {"annotations": coco / "annotations", "train2017": Path("/tmp/coco512/train2017"),
           "val2017": coco / "val2017"}
for name, source in sources.items():
    if not (root / name).exists():
        (root / name).symlink_to(source)
# unbuffered: otherwise the live log only arrives in 8 KB blocks (~30 min of training)
env = dict(os.environ, OPTINPU_DATA=str(root), PYTHONUNBUFFERED="1")

gpu_log = open("/kaggle/working/gpu_util.csv", "w")
monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                            "--format=csv", "-l", "5"], stdout=gpu_log)
failed = []
for script, *args in RUNS:
    if "--resume" in args:
        i = args.index("--resume") + 1
        args[i] = str(find(args[i]))
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(GPUS)]
    if sh(*(launcher if GPUS > 1 else [sys.executable]), str(code / script), *args, check=False, env=env):
        failed.append([script, *args])
monitor.terminate()
print("failed runs:", failed or "none")
