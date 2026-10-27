import json
import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from optinpu.data.dataset import CocoDetection, collate
from optinpu.engine import TrainConfig, train


@pytest.fixture
def toy_coco(tmp_path):
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


def test_dataset_targets_are_normalized_cxcywh(toy_coco):
    ds = CocoDetection(*toy_coco, imgsz=(64, 64))
    assert len(ds) == 6  # image without annotations dropped, crowd ignored
    assert ds.class_names == ["other", "thing"]  # sorted by category id
    pixels, target = ds[0]
    assert pixels.shape == (3, 64, 64)
    torch.testing.assert_close(target["boxes"], torch.tensor([[0.25, 0.3, 0.3, 0.4]]))
    assert target["class_labels"].tolist() == [1]


def test_dataset_flip_mirrors_boxes(toy_coco, monkeypatch):
    ds = CocoDetection(*toy_coco, imgsz=(64, 64), augment=True)
    monkeypatch.setattr(random, "random", lambda: 0.0)  # always flip
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)  # no color jitter
    _, target = ds[0]
    torch.testing.assert_close(target["boxes"], torch.tensor([[0.75, 0.3, 0.3, 0.4]]))


class ToyDetector(nn.Module):
    """One query predicting a box and class from the mean color: enough to exercise the loop."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Conv2d(3, 8, 3, 2, 1)
        self.head = nn.Linear(8, num_classes + 4)
        self.num_classes = num_classes

    def forward(self, x):
        out = self.head(self.backbone(x).mean((2, 3)))[:, None]
        return out[..., : self.num_classes].sigmoid(), out[..., self.num_classes:].sigmoid()

    def loss(self, x, targets):
        scores, boxes = self(x)
        gt = torch.stack([t["boxes"][0] for t in targets])
        cx, cy, w, h = gt.unbind(-1)
        gt_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        loss = (boxes[:, 0] - gt_xyxy).abs().mean() + (1 - scores[:, 0, 1]).mean()
        return loss, {"loss": float(loss)}


def test_train_loop_saves_best_checkpoint(toy_coco, tmp_path):
    ds = CocoDetection(*toy_coco, imgsz=(32, 32), augment=True)
    val = CocoDetection(*toy_coco, imgsz=(32, 32))
    cfg = TrainConfig(epochs=2, batch=2, lr=1e-2, backbone_lr=1e-2, warmup_steps=1, workers=0, device="cpu", log_every=100)
    result = train(ToyDetector(), ds, val, cfg, tmp_path / "run", "toy")
    assert result.checkpoint.exists()
    assert len(result.history) == 2 and all("AP" in h for h in result.history)
    state = torch.load(result.checkpoint, weights_only=False)
    assert state["class_names"] == ["other", "thing"]


def test_collate_stacks_pixels_keeps_targets():
    batch = [(torch.zeros(3, 4, 4), {"a": 1}), (torch.ones(3, 4, 4), {"a": 2})]
    pixels, targets = collate(batch)
    assert pixels.shape == (2, 3, 4, 4) and [t["a"] for t in targets] == [1, 2]
