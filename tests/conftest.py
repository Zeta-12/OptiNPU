import json

import pytest
from PIL import Image


@pytest.fixture
def toy_coco(tmp_path):
    """6 annotated images (one 'thing' box each), a crowd box and an image without annotations."""
    images, annotations = [], []
    for i in range(1, 7):
        Image.new("RGB", (200, 100), (i * 30, 0, 0)).save(tmp_path / f"{i}.jpg")
        images.append({"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 100})
        annotations.append({"id": i, "image_id": i, "category_id": 7, "bbox": [20, 10, 60, 40], "area": 2400, "iscrowd": 0})
    annotations.append({"id": 99, "image_id": 1, "category_id": 7, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 1})
    images.append({"id": 50, "file_name": "1.jpg", "width": 200, "height": 100})  # no annotations
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"images": images, "annotations": annotations,
                               "categories": [{"id": 7, "name": "thing"}, {"id": 2, "name": "other"}]}))
    return ann, tmp_path
