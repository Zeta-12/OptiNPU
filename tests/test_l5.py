import contextlib

import numpy as np
import torch
from torch import nn

from optinpu.data.dataset import CocoDetection
from optinpu.engine import TrainConfig, _batches, train


def test_batches_prefetch_keeps_features_aligned():
    loader = [(torch.full((1, 1), float(i)), [{"id": i}]) for i in range(5)]
    calls = []

    def backbone(pixels):
        calls.append(float(pixels[0, 0]))
        return [pixels * 10]

    out = list(_batches(loader, backbone))
    assert [t[0]["id"] for _, t, _ in out] == list(range(5))
    assert [float(f[0][0, 0]) for _, _, f in out] == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert calls == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(f is None for _, _, f in _batches(loader, None))


class _Features(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return (self.backbone(x),)


class ToyBackboneDetector(nn.Module):
    """Same interface as HFDetrWrapper for training with an external backbone."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU())
        self.head = nn.Linear(8, 2 + 4)
        self._external = None

    def backbone_features_module(self):
        return _Features(self.backbone)

    @contextlib.contextmanager
    def external_backbone(self, features):
        self._external = features[0]
        try:
            yield
        finally:
            self._external = None

    def forward(self, x):
        f = self._external if self._external is not None else self.backbone(x)
        out = self.head(f.mean((2, 3)))[:, None]
        return out[..., :2].sigmoid(), out[..., 2:].sigmoid()

    def loss(self, x, targets):
        scores, boxes = self(x)
        gt = torch.stack([t["boxes"][0] for t in targets])
        loss = (boxes[:, 0] - gt).abs().mean() + (1 - scores[:, 0, 1]).mean()
        return loss, {"loss": float(loss)}


def test_train_with_external_backbone_keeps_it_frozen(toy_coco, tmp_path):
    torch.manual_seed(0)
    ds = CocoDetection(*toy_coco, imgsz=(32, 32), augment=True)
    model = ToyBackboneDetector()
    backbone_before = [p.detach().clone() for p in model.backbone.parameters()]
    head_before = [p.detach().clone() for p in model.head.parameters()]
    cfg = TrainConfig(epochs=2, batch=2, lr=1e-2, warmup_steps=1, workers=0, device="cpu",
                      backbone_device="cpu", log_every=100)
    result = train(model, ds, None, cfg, tmp_path / "run", "toy")

    assert (tmp_path / "run" / "backbone.xml").exists()
    assert len(result.history) == 2
    for before, after in zip(backbone_before, model.backbone.parameters()):
        torch.testing.assert_close(before, after.detach())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(head_before, model.head.parameters()))


def test_freeze_backbone_only(toy_coco, tmp_path):
    ds = CocoDetection(*toy_coco, imgsz=(32, 32))
    model = ToyBackboneDetector()
    before = [p.detach().clone() for p in model.backbone.parameters()]
    cfg = TrainConfig(epochs=1, batch=2, lr=1e-2, warmup_steps=1, workers=0, device="cpu",
                      freeze_backbone=True, log_every=100)
    train(model, ds, None, cfg, tmp_path / "run", "toy")
    assert not (tmp_path / "run" / "backbone.xml").exists()
    for b, a in zip(before, model.backbone.parameters()):
        assert np.array_equal(b.numpy(), a.detach().numpy())
