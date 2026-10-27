"""COCO val2017 subsets, downloaded on demand (annotations once, then only the needed images)."""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("OPTINPU_DATA", Path.home() / ".cache" / "optinpu" / "coco"))
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"


@dataclass
class CocoSubset:
    annotations: Path  # COCO json restricted to `image_ids`
    image_dir: Path
    image_ids: list[int]
    category_ids: list[int]  # contiguous class index -> COCO category id
    file_names: dict[int, str] | None = None  # image id -> file name; None means COCO naming

    def image_path(self, image_id: int) -> Path:
        if self.file_names is not None:
            return self.image_dir / self.file_names[image_id]
        return self.image_dir / f"{image_id:012d}.jpg"

    def image_paths(self) -> list[Path]:
        return [self.image_path(i) for i in self.image_ids]


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _full_val_annotations(root: Path) -> Path:
    path = root / "annotations" / "instances_val2017.json"
    if not path.exists():
        archive = root / "annotations_trainval2017.zip"
        root.mkdir(parents=True, exist_ok=True)
        print(f"Downloading COCO annotations (~250 MB) to {archive}")
        _download(ANNOTATIONS_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extract("annotations/instances_val2017.json", root)
        archive.unlink()
    return path


def coco_val_subset(
    num_images: int = 500,
    offset: int = 0,
    root: str | Path = DEFAULT_ROOT,
    workers: int = 16,
) -> CocoSubset:
    """Images [offset, offset + num_images) of val2017 sorted by id, with their annotations.

    Disjoint ranges give disjoint sets, e.g. calibration (offset=4500) vs evaluation (offset=0).
    """
    root = Path(root)
    subset_json = root / "subsets" / f"val2017_{offset}_{num_images}.json"
    image_dir = root / "val2017"

    if subset_json.exists():
        data = json.loads(subset_json.read_text())
    else:
        full = json.loads(_full_val_annotations(root).read_text())
        images = sorted(full["images"], key=lambda im: im["id"])[offset: offset + num_images]
        ids = {im["id"] for im in images}
        data = {
            "info": full.get("info", {}),
            "licenses": full.get("licenses", []),
            "images": images,
            "annotations": [a for a in full["annotations"] if a["image_id"] in ids],
            "categories": full["categories"],
        }
        subset_json.parent.mkdir(parents=True, exist_ok=True)
        subset_json.write_text(json.dumps(data))

    image_dir.mkdir(parents=True, exist_ok=True)
    missing = [im["file_name"] for im in data["images"] if not (image_dir / im["file_name"]).exists()]
    if missing:
        print(f"Downloading {len(missing)} COCO images to {image_dir}")
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda f: _download(IMAGE_URL.format(file_name=f), image_dir / f), missing))

    return CocoSubset(
        annotations=subset_json,
        image_dir=image_dir,
        image_ids=[im["id"] for im in data["images"]],
        category_ids=sorted(c["id"] for c in data["categories"]),
    )


def coco_val_classes(
    classes: list[str],
    offset: int,
    num_images: int,
    root: str | Path = DEFAULT_ROOT,
    name: str = "custom",
) -> CocoSubset:
    """A COCO-format dataset restricted to `classes`, from val2017 images [offset, offset + num_images).

    Only images containing at least one of the classes are kept. Used to build small fine-tuning
    datasets with a new label space without downloading train2017.
    """
    root = Path(root)
    out = root / "subsets" / f"{name}_{offset}_{num_images}.json"
    image_dir = root / "val2017"
    if not out.exists():
        full = json.loads(_full_val_annotations(root).read_text())
        cats = [c for c in full["categories"] if c["name"] in classes]
        missing = set(classes) - {c["name"] for c in cats}
        if missing:
            raise ValueError(f"Not COCO classes: {sorted(missing)}")
        in_range = {im["id"] for im in sorted(full["images"], key=lambda im: im["id"])[offset: offset + num_images]}
        keep = {c["id"] for c in cats}
        anns = [a for a in full["annotations"] if a["category_id"] in keep and a["image_id"] in in_range]
        ids = {a["image_id"] for a in anns}
        data = {
            "images": [im for im in full["images"] if im["id"] in ids],
            "annotations": anns,
            "categories": cats,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data))
    data = json.loads(out.read_text())

    image_dir.mkdir(parents=True, exist_ok=True)
    missing_files = [im["file_name"] for im in data["images"] if not (image_dir / im["file_name"]).exists()]
    if missing_files:
        print(f"Downloading {len(missing_files)} COCO images to {image_dir}")
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(lambda f: _download(IMAGE_URL.format(file_name=f), image_dir / f), missing_files))
    return CocoSubset(
        annotations=out,
        image_dir=image_dir,
        image_ids=[im["id"] for im in data["images"]],
        category_ids=sorted(c["id"] for c in data["categories"]),
    )
