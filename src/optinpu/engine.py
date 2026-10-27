"""Training loop (PyTorch, on Intel iGPU via XPU, CUDA or CPU)."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from optinpu.data.dataset import CocoDetection, collate
from optinpu.devices import torch_device
from optinpu.postprocess import decode, load_image, preprocess


@dataclass
class TrainConfig:
    epochs: int = 12
    batch: int = 4
    lr: float = 2e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    clip: float = 0.1
    ema_decay: float = 0.999
    amp: bool = False  # bf16 autocast
    workers: int = 2
    device: str | None = None
    max_steps: int | None = None  # stop early (smoke tests)
    log_every: int = 20


@dataclass
class TrainResult:
    best_ap: float
    best_epoch: int
    checkpoint: Path
    history: list[dict] = field(default_factory=list)


class EMA:
    """Exponential moving average of weights, with the usual warmup of the decay."""

    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                ema_v.copy_(v)


class TorchPredictor:
    """predict() on a PyTorch module, with the same pre/post-processing as Detector."""

    def __init__(self, model: nn.Module, name: str, imgsz, class_names, mean=None, std=None):
        self.model, self.name, self.imgsz = model, name, imgsz
        self.class_names, self.mean, self.std = class_names, mean, std

    @torch.no_grad()
    def predict(self, source, device=None, precision=None, conf: float = 0.0):
        image = load_image(source)
        x = preprocess(image, self.imgsz, self.mean, self.std)
        dev = next(self.model.parameters()).device
        scores, boxes = self.model(torch.from_numpy(x).to(dev))
        return decode(scores[0].float().cpu().numpy(), boxes[0].float().cpu().numpy(), image.size, self.class_names, conf=conf)


def _param_groups(model: nn.Module, cfg: TrainConfig) -> list[dict]:
    backbone, no_decay, rest = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".backbone." in f".{name}":
            backbone.append(p)
        elif p.ndim <= 1:  # norms and biases
            no_decay.append(p)
        else:
            rest.append(p)
    return [
        {"params": backbone, "lr": cfg.backbone_lr},
        {"params": no_decay, "lr": cfg.lr, "weight_decay": 0.0},
        {"params": rest, "lr": cfg.lr},
    ]


def train(
    model: nn.Module,
    train_set: CocoDetection,
    val_set: CocoDetection | None,
    cfg: TrainConfig,
    out_dir: str | Path,
    name: str = "model",
) -> TrainResult:
    """Train `model` (must expose .loss(pixels, targets)); keep the best EMA weights by val mAP."""
    from optinpu.evaluate import evaluate_coco

    device = torch_device(cfg.device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).train()
    ema = EMA(model, cfg.ema_decay)

    loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers,
        collate_fn=collate, drop_last=True, persistent_workers=cfg.workers > 0,
    )
    total_steps = cfg.epochs * len(loader)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    optimizer = torch.optim.AdamW(_param_groups(model, cfg), lr=cfg.lr, weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    val_subset = val_set.to_subset() if val_set is not None else None

    result = TrainResult(best_ap=-1.0, best_epoch=-1, checkpoint=out_dir / "best.pt")
    step = 0
    print(f"Training {name} on {device}: {len(train_set)} images, {len(loader)} steps/epoch, {cfg.epochs} epochs")
    for epoch in range(cfg.epochs):
        model.train()
        t0, losses = time.perf_counter(), []
        for pixels, targets in loader:
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * lr_factor(step)
            pixels = pixels.to(device, non_blocking=True)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
                loss, parts = model.loss(pixels, targets)
            if not torch.isfinite(loss):
                print(f"step {step}: non-finite loss, batch skipped")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if cfg.clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            losses.append(loss.item())
            step += 1
            if step % cfg.log_every == 0:
                rate = len(losses) * cfg.batch / (time.perf_counter() - t0)
                print(f"epoch {epoch} step {step}/{total_steps} loss {np.mean(losses[-cfg.log_every:]):.3f} "
                      f"lr {optimizer.param_groups[-1]['lr']:.2e} {rate:.1f} img/s")
            if cfg.max_steps and step >= cfg.max_steps:
                break

        record = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
                  "seconds": time.perf_counter() - t0}
        if val_subset is not None:
            predictor = TorchPredictor(ema.module, name, train_set.imgsz, train_set.class_names)
            ev = evaluate_coco(predictor, val_subset, device="torch", precision="fp32")
            record.update(AP=ev.AP, AP50=ev.AP50)
        result.history.append(record)
        print(f"epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in record.items()))

        score = record.get("AP", -record["loss"])
        if score > result.best_ap:
            result.best_ap, result.best_epoch = score, epoch
            torch.save({"state_dict": ema.module.state_dict(), "class_names": train_set.class_names,
                        "epoch": epoch, "config": asdict(cfg)}, result.checkpoint)
        (out_dir / "history.json").write_text(json.dumps(result.history, indent=2))
        if cfg.max_steps and step >= cfg.max_steps:
            break
    return result
