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


def test_dataset_reads_a_shrunk_copy_of_the_images(toy_coco, tmp_path):
    from PIL import Image

    ann, image_dir = toy_coco
    small = tmp_path / "small"
    small.mkdir()
    for path in image_dir.glob("*.jpg"):
        Image.open(path).resize((100, 50)).save(small / path.name)  # half size; annotations unchanged
    pixels, target = CocoDetection(ann, small, imgsz=(64, 64))[0]
    torch.testing.assert_close(target["boxes"], torch.tensor([[0.25, 0.3, 0.3, 0.4]]))  # as from full size


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


def test_mosaic_boxes_land_on_their_objects(tmp_path):
    import json

    from PIL import Image, ImageDraw

    images, annotations = [], []
    for i in range(1, 5):  # white boxes on black images: a box is right if its center pixel is white
        image = Image.new("RGB", (200, 100))
        box = [10 * i, 5 * i, 50 + 10 * i, 30 + 5 * i]  # xywh
        ImageDraw.Draw(image).rectangle([box[0], box[1], box[0] + box[2] - 1, box[1] + box[3] - 1], fill="white")
        image.save(tmp_path / f"{i}.jpg", quality=95)
        images.append({"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 100})
        annotations.append({"id": i, "image_id": i, "category_id": 1, "bbox": box, "area": box[2] * box[3]})
    (tmp_path / "ann.json").write_text(json.dumps({"images": images, "annotations": annotations,
                                                   "categories": [{"id": 1, "name": "box"}]}))
    ds = CocoDetection(tmp_path / "ann.json", tmp_path, imgsz=(96, 128), augment=True, mosaic=1.0)
    n_boxes = 0
    for seed in range(40):
        random.seed(seed)
        pixels, target = ds[seed % 4]
        assert pixels.shape == (3, 96, 128)
        assert len(target["boxes"]) == len(target["class_labels"])
        for cx, cy, w, h in target["boxes"].tolist():
            assert 0 < w <= 1 and 0 < h <= 1
            # white is >= 148 after the color jitter, black stays under 60
            assert pixels[:, int(cy * 96), int(cx * 128)].min() > 100, (seed, cx, cy)
            n_boxes += 1
    assert n_boxes > 40  # several of the 4 images visible per mosaic


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


def test_device_color_jitter_matches_pil():
    import numpy as np
    from PIL import Image, ImageEnhance

    from optinpu.engine import _color_jitter

    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)) for _ in range(3)]
    x = torch.stack([torch.from_numpy(np.array(im)).permute(2, 0, 1) for im in images]).float() / 255
    torch.manual_seed(0)
    out = _color_jitter(x, 0.3)
    torch.manual_seed(0)  # the same draws: brightness, contrast, saturation factors per image
    factors = [torch.empty(3, 1, 1, 1).uniform_(0.7, 1.3).flatten().tolist() for _ in range(3)]
    for i, image in enumerate(images):
        for enhancer, f in zip((ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color), factors):
            image = enhancer(image).enhance(f[i])
        expected = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        assert (out[i] - expected).abs().max() < 4 / 255  # PIL rounds to uint8 after each step


def _ddp_worker(rank, world, port, toy, out, queue):
    import os

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank),
                      LOCAL_RANK=str(rank), USE_LIBUV="0")
    torch.manual_seed(rank)  # different initial weights: DDP must start both from process 0's
    ds = CocoDetection(*toy, imgsz=(32, 32), augment=True)
    cfg = TrainConfig(epochs=2, batch=4, lr=1e-2, backbone_lr=1e-2, warmup_steps=1, workers=0, device="cpu",
                      log_every=100, max_hours=1e-9)
    model = ToyDetector()
    result = train(model, ds, None, cfg, out, "toy")
    queue.put((rank, [v.clone() for v in model.state_dict().values()], len(result.history)))


def test_ddp_keeps_processes_in_sync(toy_coco, tmp_path):
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, port, toy_coco, tmp_path / "run", queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = dict((r, (w, n)) for r, w, n in (queue.get(timeout=120) for _ in procs))
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    for a, b in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(a, b)  # same weights after training: gradients were averaged
    # process 0 alone records, saves and decides to stop (max_hours) for both after the first epoch
    assert results[0][1] == 1 and results[1][1] == 0
    assert (tmp_path / "run" / "last.pt").exists() and (tmp_path / "run" / "best.pt").exists()


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
