"""Labeled image folders for classifiers: ImageNetV2 (matched frequency, 10 000 images, 10 per class).

The archive (`ARCHIVE_URL`, ~1.2 GB) is extracted once with `extract_imagenetv2`; images then live in
<root>/<folder>/<class index>/<name>.jpeg, the class folders being ImageNet-1k class indices.
Subsets are slices of one fixed shuffle of all the images (seed 0): disjoint offsets give disjoint
subsets with about the same class mix.
"""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(os.environ.get("OPTINPU_IMAGENETV2", Path.home() / ".cache" / "optinpu" / "imagenetv2"))
ARCHIVE_URL = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


@dataclass
class LabeledImages:
    """Images with their class index, for top-1 accuracy (optinpu.evaluate.evaluate_top1)."""

    paths: list[Path]
    labels: list[int]

    def image_paths(self) -> list[Path]:
        return list(self.paths)


def _class_dir(root: Path) -> Path:
    """The directory whose sub-folders are class indices: `root` or one of its children."""
    for candidate in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
        if any(p.is_dir() and p.name.isdigit() for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"no class folders (0, 1, ...) under {root}")


def labeled_folder(root: str | Path) -> LabeledImages:
    """Every image of <root>/<class index>/, in (class, file name) order."""
    base = _class_dir(Path(root))
    paths, labels = [], []
    for folder in sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        for image in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            paths.append(image)
            labels.append(int(folder.name))
    return LabeledImages(paths, labels)


def imagenetv2_subset(count: int, offset: int = 0, root: str | Path = DEFAULT_ROOT, seed: int = 0) -> LabeledImages:
    """Images [offset, offset + count) of the fixed shuffle of ImageNetV2."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ImageNetV2 not found in {root}: download {ARCHIVE_URL} and run "
                                f"optinpu.data.imagenet.extract_imagenetv2(archive)")
    every = labeled_folder(root)
    order = np.random.default_rng(seed).permutation(len(every.paths))[offset:offset + count]
    return LabeledImages([every.paths[i] for i in order], [every.labels[i] for i in order])


def extract_imagenetv2(archive: str | Path, root: str | Path = DEFAULT_ROOT) -> Path:
    """Extract the downloaded archive to `root` (tarfile "data" filter: no links or paths outside it)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return root
