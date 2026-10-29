"""P3.3: image folders for an end-to-end `optinpu optimize` run, as a user would provide them
(guard: val2017 [4970, 5000), calibration: [4500, 4800); the AP budget set is the COCO subset json of
[2500, 2700) with the val2017 folder). Prints the command line.

    python -m benchmarks.p3.cli_folders
"""

from __future__ import annotations

import shutil

from benchmarks.m0.baselines import ROOT
from optinpu.data import coco_val_subset

OUT = ROOT / "runs" / "p3" / "cli"


def main():
    for name, count, offset in (("guard", 30, 4970), ("calibration", 300, 4500)):
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        for path in coco_val_subset(count, offset=offset).image_paths():
            if not (folder / path.name).exists():
                shutil.copy(path, folder / path.name)
    budget = coco_val_subset(200, offset=2500)
    print(f"optinpu optimize runs/m0/ir/yolo26-n.xml --task dets --letterbox --images {OUT / 'guard'} "
          f"--calibration {OUT / 'calibration'} --annotations {budget.annotations} {budget.image_dir} "
          f"--out {OUT / 'yolo26-n'}")


if __name__ == "__main__":
    main()
