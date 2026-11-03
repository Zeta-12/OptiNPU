"""M3: train OptiDet on a Kaggle GPU (T4). Runs on Kaggle, not locally:

    kaggle kernels push -p benchmarks/m3/kaggle --accelerator NvidiaTeslaT4 -t 43200

Inputs (kernel-metadata.json): the optinpu wheel, benchmarks/m2/train_optidet.py and benchmarks/m3/loader_speed.py,
uploaded as the private dataset julienbeloin/optinpu-code, COCO 2017 (awsaf49/coco-2017-dataset) and the checkpoint
to resume from (julienbeloin/optidet-m3-ckpt, a new version after each session). Outputs: runs/ and gpu_util.csv in
/kaggle/working, fetched with `kaggle kernels output`.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# one command (script of the dataset, arguments) per entry; a failing run does not stop the next one.
# fp16, not bf16: bf16 is emulated on T4, 3.5x slower than FP32.
RUNS = [
    # M3: OptiDet-N on all of train2017 (117k annotated images), 72 epochs, M2.5 recipe (D-FINE-S distillation),
    # in 11 h sessions. --resume last.pt: the previous session's last.pt and best.pt, uploaded as the private
    # dataset julienbeloin/optidet-m3-ckpt (file names after --resume are searched under /kaggle/input)
    ["train_optidet.py", "--images", "200000", "--epochs", "72", "--batch", "16", "--workers", "4", "--amp", "fp16",
     "--distill", "ustc-community/dfine-small-coco", "--run", "m3_coco_full", "--max-hours", "11",
     "--resume", "last.pt"],
]
IMAGE_DIRS = {"train2017", "val2017", "test2017"}


def find(name: str, wait_s: int = 300) -> Path:
    """First file `name` under /kaggle/input, without walking the COCO image folders. Inputs can be symlinks
    and can be mounted late (one session did not see COCO 30 s after the start): retry for `wait_s`."""
    deadline = time.time() + wait_s
    while True:
        for top, dirs, files in os.walk("/kaggle/input", followlinks=True):
            dirs[:] = [d for d in dirs if d not in IMAGE_DIRS]
            if name in files:
                return Path(top) / name
        if time.time() > deadline:
            break
        time.sleep(15)
    sh("find", "/kaggle/input", "-maxdepth", "4", check=False)
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
for name in ("annotations", "train2017", "val2017"):
    if not (root / name).exists():
        (root / name).symlink_to(coco / name)
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
    if sh(sys.executable, str(code / script), *args, check=False, env=env):
        failed.append([script, *args])
monitor.terminate()
print("failed runs:", failed or "none")
