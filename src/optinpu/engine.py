"""Training loop (PyTorch, on Intel iGPU via XPU, CUDA or CPU)."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
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
    # autocast: True or "bf16", or "fp16" with loss scaling (GPUs without native bf16, e.g. T4: bf16 is
    # emulated there, 3.5x slower than FP32 on OptiDet-N)
    amp: bool | str = False
    workers: int = 2
    device: str | None = None
    max_steps: int | None = None  # stop early (smoke tests)
    # last.pt of an interrupted run (same epochs, batch and dataset): continue after its last epoch
    resume: str | None = None
    # stop after the last epoch that fits in this budget, e.g. under a session time limit (Kaggle: 12 h)
    max_hours: float | None = None
    # last epochs without mosaic augmentation (CocoDetection.mosaic), as YOLOv8 does: back to real images
    close_mosaic: int = 0
    # > 0: color jitter (brightness, contrast, saturation x U(1 - c, 1 + c)) on the training device instead of
    # in the data workers (CocoDetection.jitter): frees CPU when loading bounds training (Kaggle: 4 CPUs)
    color_jitter: float = 0.0
    log_every: int = 20
    freeze_backbone: bool = False
    # Run the (frozen) backbone on another OpenVINO device, e.g. "npu": features of the next batch
    # are computed there while the training device processes the current one.
    backbone_device: str | None = None


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
        floats, others = ([], []), ([], [])
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            pair = floats if ema_v.dtype.is_floating_point else others
            pair[0].append(ema_v)
            pair[1].append(v)
        # one grouped call per list instead of kernel launches per tensor (hundreds per step)
        if floats[0]:
            torch._foreach_lerp_(*floats, 1 - d)
        if others[0]:
            torch._foreach_copy_(*others)


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


def _compile_backbone(model: nn.Module, cfg: TrainConfig, imgsz, out_dir: Path):
    """Export the frozen backbone with the training batch size and compile it on cfg.backbone_device."""
    from optinpu.export import export_openvino
    from optinpu.runtime import compile_model

    module = copy.deepcopy(model.backbone_features_module()).cpu().eval()
    ir = export_openvino(module, out_dir / "backbone.xml", input_shape=(cfg.batch, 3, *imgsz))
    print(f"Frozen backbone compiled on {cfg.backbone_device.upper()} ({ir})")
    # THROUGHPUT makes the NPU compiler try (and log a failure for) batch splitting across tiles
    return compile_model(ir, cfg.backbone_device, hint="LATENCY")


def _float_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Model input in [0, 1] from CocoDetection's uint8 pixels (floats pass through)."""
    return pixels.float().div_(255) if pixels.dtype == torch.uint8 else pixels


def _batches(loader, external_backbone):
    """Yield (pixels, targets, features or None). With an external backbone, the features of the
    next batch are computed in a worker thread while the caller trains on the current one
    (OpenVINO releases the GIL during inference)."""
    if external_backbone is None:
        for pixels, targets in loader:
            yield pixels, targets, None
        return
    with ThreadPoolExecutor(1) as pool:
        it = iter(loader)
        nxt = next(it, None)
        future = pool.submit(external_backbone, _float_pixels(nxt[0]).numpy()) if nxt else None
        while nxt is not None:
            current, current_future = nxt, future
            nxt = next(it, None)
            future = pool.submit(external_backbone, _float_pixels(nxt[0]).numpy()) if nxt else None
            yield current[0], current[1], current_future.result()


def _color_jitter(x: torch.Tensor, c: float) -> torch.Tensor:
    """CocoDetection's jitter (PIL ImageEnhance Brightness, Contrast, Color, factors U(1 - c, 1 + c)) on a
    batch (B, 3, H, W) in [0, 1]: blends with black, the mean gray level, then the grayscale image."""
    def factor():
        return torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(1 - c, 1 + c)

    def gray(t):  # ITU-R 601-2 luma, as PIL's "L" mode
        return (0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3])

    x = (x * factor()).clamp_(0, 1)
    mean = gray(x).mean((2, 3), keepdim=True)
    x = ((x - mean) * factor() + mean).clamp_(0, 1)
    g = gray(x)
    return ((x - g) * factor() + g).clamp_(0, 1)


def _distributed(cfg: TrainConfig) -> tuple[int, int, torch.device]:
    """(rank, number of processes, device). Launched by torchrun with WORLD_SIZE > 1: one process per GPU,
    gradients averaged by DistributedDataParallel; `batch` stays the global batch. Otherwise (0, 1, device)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, torch_device(cfg.device)
    import torch.distributed as dist

    cuda = torch.cuda.is_available() and cfg.device != "cpu"
    local = int(os.environ["LOCAL_RANK"])
    if cuda:
        torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo")
    return dist.get_rank(), world, torch.device("cuda", local) if cuda else torch.device("cpu")


class _Loss(nn.Module):
    """forward() = model.loss(): DistributedDataParallel synchronizes gradients through forward() only."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixels, targets):
        return self.model.loss(pixels, targets)


def _all_true(flag: bool, world: int, device: torch.device) -> bool:
    """`flag` and-ed over all processes: every process must take the same branch, or they deadlock."""
    if world == 1:
        return flag
    t = torch.tensor([float(flag)], device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
    return bool(t.item())


def _amp_dtype(amp: bool | str) -> torch.dtype | None:
    """Autocast dtype of TrainConfig.amp; None: FP32."""
    if amp is False or amp is None:
        return None
    if amp is True or amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    raise ValueError(f"amp must be False, True, 'bf16' or 'fp16', got {amp!r}")


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
    checkpoint_extra: dict | None = None,
) -> TrainResult:
    """Train `model` (must expose .loss(pixels, targets)); keep the best EMA weights by val mAP.
    `checkpoint_extra` entries are saved in the checkpoint as is (e.g. the QAT quantizer config).
    The whole training state goes to last.pt after every epoch, for TrainConfig.resume.
    Under torchrun (one process per GPU) only process 0 evaluates, saves and prints."""
    from optinpu.evaluate import evaluate_coco

    rank, world, device = _distributed(cfg)
    main = rank == 0
    log = print if main else (lambda *args, **kwargs: None)
    if cfg.batch % world:
        raise ValueError(f"batch {cfg.batch} is not a multiple of the {world} processes")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    amp_dtype = _amp_dtype(cfg.amp)

    external_backbone = None
    if cfg.freeze_backbone or cfg.backbone_device:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    if cfg.backbone_device:
        external_backbone = _compile_backbone(model, cfg, train_set.imgsz, out_dir)
    model = model.to(device).train()
    ema = EMA(model, cfg.ema_decay)

    def make_loader():
        sampler, generator = None, None
        if world > 1:  # each process takes its share of every global batch
            sampler = torch.utils.data.DistributedSampler(train_set, world, rank, shuffle=True, drop_last=True)
            # workers seed their augmentations from this generator: a different stream per process
            generator = torch.Generator().manual_seed(int(torch.randint(2**62, ())) + rank)
        return torch.utils.data.DataLoader(
            train_set, batch_size=cfg.batch // world, shuffle=sampler is None, sampler=sampler,
            num_workers=cfg.workers, collate_fn=collate, drop_last=True, persistent_workers=cfg.workers > 0,
            generator=generator,
        )

    if cfg.color_jitter and hasattr(train_set, "jitter"):
        train_set.jitter = False  # done on the device below
    loader = make_loader()
    total_steps = cfg.epochs * len(loader)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    optimizer = torch.optim.AdamW(_param_groups(model, cfg), lr=cfg.lr, weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    # loss scaling keeps small fp16 gradients from flushing to zero; a pass-through otherwise
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    def lr_factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    val_subset = val_set.to_subset() if val_set is not None else None

    # -inf: without validation the score is -loss, which can be below any finite start value
    result = TrainResult(best_ap=float("-inf"), best_epoch=-1, checkpoint=out_dir / "best.pt")
    start_epoch, step = 0, 0
    if cfg.resume:
        start_epoch, step = _resume(Path(cfg.resume), cfg, len(loader), model, ema, optimizer, scaler, result)
    net = _Loss(model)
    if world > 1:  # after resuming: DDP starts every process from process 0's weights
        net = nn.parallel.DistributedDataParallel(net, device_ids=[device.index] if device.type == "cuda" else None)
    log(f"Training {name} on {device}" + (f" x {world} processes" if world > 1 else "")
        + f": {len(train_set)} images, {len(loader)} steps/epoch, {cfg.epochs} epochs"
        + (f", {amp_dtype} autocast" if amp_dtype else "")
        + (f", resuming at epoch {start_epoch}" if start_epoch else ""))
    started = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs):
        if getattr(train_set, "mosaic", 0) and cfg.close_mosaic and epoch >= cfg.epochs - cfg.close_mosaic:
            train_set.mosaic = 0.0
            loader = make_loader()  # persistent workers hold their own copy of the dataset
            log(f"epoch {epoch}: mosaic off for the last {cfg.close_mosaic} epochs")
        if world > 1:
            loader.sampler.set_epoch(epoch)
        model.train()
        t0, losses = time.perf_counter(), []
        for pixels, targets, features in _batches(loader, external_backbone):
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * lr_factor(step)
            pixels = _float_pixels(pixels.to(device, non_blocking=True))
            if cfg.color_jitter:
                pixels = _color_jitter(pixels, cfg.color_jitter)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            backbone_ctx = contextlib.nullcontext()
            if features is not None:
                backbone_ctx = model.external_backbone([torch.from_numpy(f).to(device) for f in features])
            with torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None), \
                    backbone_ctx:
                loss, parts = net(pixels, targets)
            # backward even on a non-finite loss: with DDP every process takes part in the gradient averaging
            scaler.scale(loss).backward()
            if not _all_true(bool(torch.isfinite(loss)), world, device):
                log(f"step {step}: non-finite loss, batch skipped")
                optimizer.zero_grad(set_to_none=True)
                continue
            if cfg.clip:
                scaler.unscale_(optimizer)  # clip the true gradients
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            losses.append(loss.item())
            step += 1
            if step % cfg.log_every == 0 and main:
                rate = len(losses) * cfg.batch / (time.perf_counter() - t0)
                print(f"epoch {epoch} step {step}/{total_steps} loss {np.mean(losses[-cfg.log_every:]):.3f} "
                      f"lr {optimizer.param_groups[-1]['lr']:.2e} {rate:.1f} img/s")
            if cfg.max_steps and step >= cfg.max_steps:
                break

        record = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
                  "seconds": time.perf_counter() - t0}
        if main:
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
                            "epoch": epoch, "config": asdict(cfg), **(checkpoint_extra or {})}, result.checkpoint)
            (out_dir / "history.json").write_text(json.dumps(result.history, indent=2))
            state = {"model": model.state_dict(), "ema": ema.module.state_dict(), "ema_updates": ema.updates,
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                     "step": step, "steps_per_epoch": len(loader), "best_ap": result.best_ap,
                     "best_epoch": result.best_epoch, "history": result.history,
                     "rng": (torch.get_rng_state(), random.getstate()), "config": asdict(cfg)}
            torch.save(state, out_dir / "last.pt.tmp")  # a run killed while saving keeps the previous last.pt
            (out_dir / "last.pt.tmp").replace(out_dir / "last.pt")
        if world > 1:
            torch.distributed.barrier()  # the other processes wait for process 0's validation and checkpoints
        if cfg.max_steps and step >= cfg.max_steps:
            break
        if cfg.max_hours and epoch + 1 < cfg.epochs:
            elapsed, done = time.perf_counter() - started, epoch + 1 - start_epoch
            # process 0 decides for all (the others vote "stop")
            if _all_true(elapsed * (done + 1) / done > cfg.max_hours * 3600 or not main, world, device):
                log(f"stopping after epoch {epoch}: the next one would exceed max_hours={cfg.max_hours} "
                    f"({elapsed / 3600:.2f} h so far); resume from {out_dir / 'last.pt'}")
                break
    if world > 1:
        torch.distributed.destroy_process_group()
    return result


def _resume(path: Path, cfg: TrainConfig, steps_per_epoch: int, model, ema, optimizer, scaler,
            result: TrainResult) -> tuple[int, int]:
    """Restore the training state saved in last.pt; returns (first epoch to run, global step)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved = (state["config"]["epochs"], state["steps_per_epoch"])
    if saved != (cfg.epochs, steps_per_epoch):
        raise ValueError(f"{path} was saved with {saved[0]} epochs of {saved[1]} steps, now {cfg.epochs} of "
                         f"{steps_per_epoch}: the learning rate schedule would not match")
    model.load_state_dict(state["model"])
    ema.module.load_state_dict(state["ema"])
    ema.updates = state["ema_updates"]
    optimizer.load_state_dict(state["optimizer"])
    if state["scaler"]:  # empty when saved without fp16
        scaler.load_state_dict(state["scaler"])
    # with workers=0, the same shuffling and augmentations as an uninterrupted run
    torch.set_rng_state(state["rng"][0])
    random.setstate(state["rng"][1])
    result.best_ap, result.best_epoch, result.history = state["best_ap"], state["best_epoch"], state["history"]
    best = path.with_name("best.pt")
    if best.exists() and not result.checkpoint.exists():  # resumed into a new directory (next Kaggle session)
        shutil.copy2(best, result.checkpoint)
    return state["epoch"] + 1, state["step"]
