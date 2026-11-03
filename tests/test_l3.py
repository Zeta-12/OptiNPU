import math
import random

import pytest
import torch
from torch import nn

from optinpu.data.dataset import CocoDetection, collate
from optinpu.engine import TrainConfig, _amp_dtype, train


def test_dataset_targets_are_normalized_cxcywh(toy_coco):
    ds = CocoDetection(*toy_coco, imgsz=(64, 64))
    assert len(ds) == 6  # image without annotations dropped, crowd ignored
    assert ds.class_names == ["other", "thing"]  # sorted by category id
    pixels, target = ds[0]
    assert pixels.shape == (3, 64, 64) and pixels.dtype == torch.uint8  # scaled to [0, 1] by the engine
    torch.testing.assert_close(target["boxes"], torch.tensor([[0.25, 0.3, 0.3, 0.4]]))
    assert target["class_labels"].tolist() == [1]


def test_random_zoom_keeps_boxes_consistent():
    import numpy as np
    from PIL import Image

    from optinpu.data.dataset import random_zoom

    image = Image.new("RGB", (200, 100))
    boxes = np.array([[20, 10, 60, 40], [150, 60, 40, 30]], dtype=np.float32)  # xywh
    labels = np.array([0, 1])
    for seed in range(50):
        random.seed(seed)
        img, b, lab = random_zoom(image, boxes.copy(), labels.copy())
        w, h = img.size
        assert len(b) == len(lab) >= 1
        assert np.all(b[:, :2] >= 0) and np.all(b[:, 0] + b[:, 2] <= w + 1e-3) and np.all(b[:, 1] + b[:, 3] <= h + 1e-3)
        assert np.all(b[:, 2:] > 2)


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


def test_train_without_validation_saves_checkpoint_even_with_large_loss(toy_coco, tmp_path):
    class BigLoss(ToyDetector):
        def loss(self, x, targets):
            loss, parts = super().loss(x, targets)
            return loss + 1000.0, parts  # -loss < -1: used to never be "best", no checkpoint written

    ds = CocoDetection(*toy_coco, imgsz=(32, 32))
    cfg = TrainConfig(epochs=1, batch=2, workers=0, device="cpu", log_every=100)
    result = train(BigLoss(), ds, None, cfg, tmp_path / "run", "toy")
    assert result.checkpoint.exists() and result.best_epoch == 0


def test_resume_matches_an_uninterrupted_run(toy_coco, tmp_path):
    ds = CocoDetection(*toy_coco, imgsz=(32, 32), augment=True)
    cfg = dict(epochs=3, batch=2, lr=1e-2, backbone_lr=1e-2, warmup_steps=1, workers=0, device="cpu", log_every=100)

    def run(out, **extra):
        torch.manual_seed(0)
        random.seed(0)
        model = ToyDetector()
        return model, train(model, ds, None, TrainConfig(**{**cfg, **extra}), tmp_path / out, "toy")

    full, full_result = run("full")
    _, cut = run("cut", max_hours=1e-9)  # the time budget stops it after epoch 0
    assert len(cut.history) == 1
    last = str(tmp_path / "cut" / "last.pt")
    resumed, result = run("resumed", resume=last)  # into a new directory, as in the next Kaggle session

    assert [h["epoch"] for h in result.history] == [0, 1, 2]
    assert [h["loss"] for h in result.history] == pytest.approx([h["loss"] for h in full_result.history])
    for a, b in zip(full.state_dict().values(), resumed.state_dict().values()):
        torch.testing.assert_close(a, b)
    assert result.checkpoint.exists()
    with pytest.raises(ValueError, match="schedule"):
        run("other", resume=last, epochs=4)


def test_fp16_trains_with_loss_scaling(toy_coco, tmp_path):
    ds = CocoDetection(*toy_coco, imgsz=(32, 32))
    cfg = TrainConfig(epochs=1, batch=2, workers=0, device="cpu", amp="fp16", log_every=100)
    result = train(ToyDetector(), ds, None, cfg, tmp_path / "run", "toy")
    assert math.isfinite(result.history[0]["loss"])
    assert torch.load(tmp_path / "run" / "last.pt", weights_only=False)["scaler"]["scale"] > 0  # GradScaler on
    assert _amp_dtype(True) is torch.bfloat16 and _amp_dtype(False) is None  # amp=True still means bf16
    with pytest.raises(ValueError):
        _amp_dtype("fp8")


def test_ema_update_blends_floats_and_copies_counters():
    from optinpu.engine import EMA

    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    ema = EMA(model, decay=0.9)
    before = {k: v.clone() for k, v in ema.module.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    model[1].num_batches_tracked += 5
    ema.update(model)
    d = 0.9 * (1 - math.exp(-1 / 2000))
    for k, v in model.state_dict().items():
        expected = d * before[k] + (1 - d) * v if v.dtype.is_floating_point else v
        torch.testing.assert_close(ema.module.state_dict()[k], expected)


def test_collate_stacks_pixels_keeps_targets():
    batch = [(torch.zeros(3, 4, 4), {"a": 1}), (torch.ones(3, 4, 4), {"a": 2})]
    pixels, targets = collate(batch)
    assert pixels.shape == (2, 3, 4, 4) and [t["a"] for t in targets] == [1, 2]
