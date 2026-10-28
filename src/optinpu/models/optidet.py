"""OptiDet: an NPU-native real-time detector (PLAN.md, Enjeu 2 and M1 design rules).

- Backbone: RepVGG-style. Training: 3x3 + 1x1 + identity branches (each with BatchNorm); deployment:
  one 3x3 conv + ReLU per layer (`deploy()`), the operation this NPU runs best (M1: 4-6 TOPS, INT8 x2).
- Neck: PAN with RepConv blocks, nearest upsampling, concat + 1x1 fusion.
- Head: dense, anchor-free, NMS-free. Two head sets as in YOLOv10 / YOLO26: one-to-one (inference)
  and one-to-many (extra supervision during training only). Boxes are distributions over `reg_max`
  bins per side (GFL / D-FINE idea: robust to FP16 / INT8 noise).
- No depthwise conv, SiLU, LayerNorm, GELU, attention, TopK: all slow or fragile on the NPU (M1).

Output contract (eval / export), like every optinpu model:
    pixels (B, 3, H, W) in [0, 1] -> scores (B, A, C) in [0, 1], boxes (B, A, 4) normalized xyxy
where A is the number of anchor points over strides 8, 16, 32.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class OptiDetConfig:
    widths: tuple[int, ...] = (32, 48, 96, 192, 384)  # stem (stride 2), stages at strides 4, 8, 16, 32
    depths: tuple[int, ...] = (1, 2, 4, 8, 1)  # layers per stage, the first one strided
    neck_width: int = 96
    neck_depth: int = 1  # RepConv layers after each fusion
    head_width: int = 96
    reg_max: int = 16  # bins per box side
    num_classes: int = 80
    imgsz: tuple[int, int] = (512, 512)
    strides: tuple[int, ...] = field(default=(8, 16, 32))


# ---------------------------------------------------------------- RepConv

def _conv_bn(cin, cout, k, s):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout))


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * bn.weight / std


class RepConv(nn.Module):
    """RepVGG block: ReLU(BN(conv3x3) + BN(conv1x1) [+ BN(x)]) in training, ReLU(conv3x3) once deployed."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.cin, self.cout, self.stride = cin, cout, stride
        self.k3 = _conv_bn(cin, cout, 3, stride)
        self.k1 = _conv_bn(cin, cout, 1, stride)
        self.identity = nn.BatchNorm2d(cin) if cin == cout and stride == 1 else None
        self.fused: nn.Conv2d | None = None

    def forward(self, x):
        if self.fused is not None:
            return F.relu(self.fused(x))
        y = self.k3(x) + self.k1(x)
        if self.identity is not None:
            y = y + self.identity(x)
        return F.relu(y)

    @torch.no_grad()
    def deploy(self) -> None:
        w, b = _fuse(self.k3[0], self.k3[1])
        w1, b1 = _fuse(self.k1[0], self.k1[1])
        w = w + F.pad(w1, (1, 1, 1, 1))
        b = b + b1
        if self.identity is not None:
            eye = torch.zeros(self.cout, self.cin, 3, 3, device=w.device)
            eye[torch.arange(self.cout), torch.arange(self.cin), 1, 1] = 1
            bn = self.identity
            std = (bn.running_var + bn.eps).sqrt()
            w = w + eye * (bn.weight / std).reshape(-1, 1, 1, 1)
            b = b + bn.bias - bn.running_mean * bn.weight / std
        self.fused = nn.Conv2d(self.cin, self.cout, 3, self.stride, 1).to(w.device)
        self.fused.weight.copy_(w)
        self.fused.bias.copy_(b)
        self.k3 = self.k1 = self.identity = None


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=1, s=1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


# ---------------------------------------------------------------- backbone, neck, head

class Backbone(nn.Module):
    def __init__(self, widths, depths):
        super().__init__()
        stages, cin = [], 3
        for w, d in zip(widths, depths):
            stages.append(nn.Sequential(RepConv(cin, w, 2), *[RepConv(w, w) for _ in range(d - 1)]))
            cin = w
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats[-3:]  # strides 8, 16, 32


class Fuse(nn.Module):
    """cat(a, b) -> 1x1 conv -> RepConv x depth."""

    def __init__(self, cin, cout, depth):
        super().__init__()
        self.merge = ConvBNReLU(cin, cout)
        self.blocks = nn.Sequential(*[RepConv(cout, cout) for _ in range(depth)])

    def forward(self, a, b):
        return self.blocks(self.merge(torch.cat([a, b], 1)))


class Neck(nn.Module):
    """PAN: top-down with nearest upsampling, bottom-up with stride-2 RepConv."""

    def __init__(self, in_widths, width, depth):
        super().__init__()
        self.lateral = nn.ModuleList(ConvBNReLU(c, width) for c in in_widths)
        self.td4, self.td3 = Fuse(2 * width, width, depth), Fuse(2 * width, width, depth)
        self.down3, self.down4 = RepConv(width, width, 2), RepConv(width, width, 2)
        self.bu4, self.bu5 = Fuse(2 * width, width, depth), Fuse(2 * width, width, depth)

    def forward(self, feats):
        c3, c4, c5 = (lat(f) for lat, f in zip(self.lateral, feats))
        p4 = self.td4(F.interpolate(c5, scale_factor=2.0, mode="nearest"), c4)
        p3 = self.td3(F.interpolate(p4, scale_factor=2.0, mode="nearest"), c3)
        n4 = self.bu4(self.down3(p3), p4)
        n5 = self.bu5(self.down4(n4), c5)
        return [p3, n4, n5]


class HeadLevel(nn.Module):
    """Decoupled classification / regression branches for one pyramid level."""

    def __init__(self, cin, width, num_classes, reg_max):
        super().__init__()
        self.cls = nn.Sequential(RepConv(cin, width), RepConv(width, width), nn.Conv2d(width, num_classes, 1))
        self.reg = nn.Sequential(RepConv(cin, width), RepConv(width, width), nn.Conv2d(width, 4 * reg_max, 1))

    def forward(self, x):
        return self.cls(x), self.reg(x)


class Head(nn.Module):
    def __init__(self, cin, width, num_classes, reg_max, strides, imgsz):
        super().__init__()
        self.levels = nn.ModuleList(HeadLevel(cin, width, num_classes, reg_max) for _ in strides)
        # Initial class probability ~ 5 objects per image spread over all cells and classes of a level:
        # with hundreds of thousands of negative (anchor, class) pairs, a flat 0.01 prior makes the
        # initial classification loss enormous (YOLOv8 / YOLOv10 use the same initialization).
        cells = imgsz[0] * imgsz[1]
        for level, stride in zip(self.levels, strides):
            p = 5 / num_classes / (cells / stride**2)
            nn.init.constant_(level.cls[-1].bias, float(torch.log(torch.tensor(p / (1 - p)))))

    def forward(self, feats):
        return [level(f) for level, f in zip(self.levels, feats)]


# ---------------------------------------------------------------- pretrained backbone

REPVGG_A0 = ((48, 48, 96, 192), (1, 2, 4, 14))  # stem + 3 stages of RepVGG-A0 (timm "repvgg_a0")


@torch.no_grad()
def load_repvgg_a0(backbone: Backbone, weights: str = "repvgg_a0.rvgg_in1k") -> list[str]:
    """Load ImageNet RepVGG-A0 weights (MIT license, via timm) into a training-form Backbone whose
    first four stages match RepVGG-A0 (widths 48-48-96-192, depths 1-2-4-14). A fifth stage gets the
    first output channels of RepVGG-A0's last block (192 -> 1280). Returns the loaded tensor names."""
    import timm

    source = timm.create_model(weights, pretrained=True).state_dict()
    blocks = [(s, j) for s, stage in enumerate(backbone.stages) for j in range(len(stage))]
    loaded = []
    for s, j in blocks:
        block: RepConv = backbone.stages[s][j]
        prefix = "stem." if s == 0 else f"stages.{s - 1}.{j}."
        mapping = {"k3.0.": "conv_kxk.conv.", "k3.1.": "conv_kxk.bn.", "k1.0.": "conv_1x1.conv.",
                   "k1.1.": "conv_1x1.bn.", "identity.": "identity."}
        for name, tensor in block.state_dict().items():
            ours = next((k for k in mapping if name.startswith(k)), None)
            key = prefix + mapping[ours] + name[len(ours):] if ours else None
            if key not in source:
                raise KeyError(f"{key} missing in {weights}: backbone does not match RepVGG-A0")
            value = source[key]
            if value.shape != tensor.shape:  # last stage: keep the first output channels
                if value.shape[1:] != tensor.shape[1:] or value.shape[0] < tensor.shape[0]:
                    raise ValueError(f"{key}: {tuple(value.shape)} cannot initialize {tuple(tensor.shape)}")
                value = value[: tensor.shape[0]]
            tensor.copy_(value)
            loaded.append(key)
    return loaded


# ---------------------------------------------------------------- detector

class OptiDet(nn.Module):
    def __init__(self, config: OptiDetConfig | None = None):
        super().__init__()
        self.config = cfg = config or OptiDetConfig()
        self.backbone = Backbone(cfg.widths, cfg.depths)
        self.neck = Neck(cfg.widths[-3:], cfg.neck_width, cfg.neck_depth)
        self.one2one = Head(cfg.neck_width, cfg.head_width, cfg.num_classes, cfg.reg_max, cfg.strides, cfg.imgsz)
        self.one2many = Head(cfg.neck_width, cfg.head_width, cfg.num_classes, cfg.reg_max, cfg.strides, cfg.imgsz)
        self.register_buffer("proj", torch.arange(cfg.reg_max, dtype=torch.float32), persistent=False)
        self._set_anchors(cfg.imgsz)

    def _set_anchors(self, imgsz):
        """Anchor points (centers of grid cells, in pixels) and their strides for a fixed image size."""
        h, w = imgsz
        points, strides = [], []
        for s in self.config.strides:
            ys, xs = torch.meshgrid(torch.arange(h // s), torch.arange(w // s), indexing="ij")
            points.append(torch.stack([xs, ys], -1).reshape(-1, 2).float().add(0.5).mul(s))
            strides.append(torch.full((points[-1].shape[0], 1), float(s)))
        self.register_buffer("anchors", torch.cat(points), persistent=False)
        self.register_buffer("anchor_strides", torch.cat(strides), persistent=False)
        self.register_buffer("scale", torch.tensor([w, h, w, h], dtype=torch.float32), persistent=False)

    @property
    def num_anchors(self) -> int:
        return self.anchors.shape[0]

    def features(self, pixels):
        return self.neck(self.backbone(pixels))

    def flatten(self, outputs):
        """Per-level (cls (B, C, h, w), reg (B, 4R, h, w)) -> logits (B, A, C), distribution logits (B, A, 4, R)."""
        cls = torch.cat([c.flatten(2).transpose(1, 2) for c, _ in outputs], 1)
        reg = torch.cat([r.flatten(2).transpose(1, 2) for _, r in outputs], 1)
        return cls, reg.reshape(reg.shape[0], reg.shape[1], 4, self.config.reg_max)

    def distances(self, reg_logits):
        """Expected left, top, right, bottom distances in pixels: softmax over bins, times stride."""
        return (reg_logits.softmax(-1) * self.proj).sum(-1) * self.anchor_strides

    def boxes(self, distances):
        """(B, A, 4) ltrb distances in pixels -> xyxy boxes in pixels."""
        return torch.cat([self.anchors - distances[..., :2], self.anchors + distances[..., 2:]], -1)

    def forward(self, pixels):
        feats = self.features(pixels)
        if getattr(self, "_raw_outputs", False):
            # training path, taken by loss(): it must go through forward() because quantization-aware
            # training (NNCF) attaches its fake quantizers to the ops executed during forward()
            return feats, self.forward_train(pixels, feats)
        cls, reg = self.flatten(self.one2one(feats))
        return cls.sigmoid(), self.boxes(self.distances(reg)) / self.scale

    def forward_train(self, pixels, feats=None):
        """Raw outputs of both head sets for the loss: {"one2one" | "one2many": (cls, reg)}."""
        feats = self.features(pixels) if feats is None else feats
        return {"one2one": self.flatten(self.one2one(feats)), "one2many": self.flatten(self.one2many(feats))}

    def enable_distillation(self, teacher, weight: float = 2.0) -> None:
        """Add PKD feature distillation from `teacher` (e.g. optinpu.models.distill.DFineTeacher), whose
        feature maps must match the neck's strides. Creates training-only 1x1 adapters (`distill.*`,
        dropped by deploy(), ignored by Detector.from_checkpoint)."""
        self.distill = nn.ModuleList(nn.Conv2d(self.config.neck_width, c, 1) for c in teacher.channels)
        self.distill.to(self.anchors.device)
        self._teacher = (teacher,)  # a tuple: not a registered submodule, kept out of the state dict
        self.distill_weight = weight

    def loss(self, pixels, targets):
        """Training loss (optinpu.engine contract). targets: [{"class_labels", "boxes" normalized cxcywh}]."""
        from optinpu.models.optidet_loss import OptiDetLoss

        if getattr(self, "_loss", None) is None:
            self._loss = OptiDetLoss(self)
        if tuple(pixels.shape[-2:]) != tuple(self.config.imgsz):
            raise ValueError(f"OptiDet built for {self.config.imgsz}, got images of {tuple(pixels.shape[-2:])}")
        self._raw_outputs = True  # under the caller's autocast (bf16) if any
        try:
            feats, outputs = self(pixels)
        finally:
            self._raw_outputs = False
        distill = getattr(self, "distill", None)
        teacher_feats = self._teacher[0](pixels) if distill is not None else None
        # The loss runs in FP32: in bf16 (3 significant digits) a 416-pixel coordinate is only known
        # to +-2 px and the assignment metric IoU^6 becomes noise.
        with torch.autocast(pixels.device.type, enabled=False):
            outputs = {k: (c.float(), r.float()) for k, (c, r) in outputs.items()}
            loss, parts = self._loss(outputs, targets)
            if distill is not None:
                from optinpu.models.distill import pkd_loss

                kd = pkd_loss([a(f.float()) for a, f in zip(distill, feats)], teacher_feats)
                loss = loss + self.distill_weight * kd
                parts["distill"] = float(kd)
            return loss, parts

    @torch.no_grad()
    def fuse(self) -> "OptiDet":
        """Fuse every RepConv into one conv, keeping both heads (still trainable: QAT). Returns self."""
        for m in self.modules():
            if isinstance(m, RepConv) and m.fused is None:
                m.deploy()
        return self

    @torch.no_grad()
    def deploy(self) -> "OptiDet":
        """Inference form: fuse every RepConv, drop the one-to-many head and distillation adapters."""
        self.fuse()
        self.one2many = None
        self.distill = None
        self._teacher = None
        return self.eval()

    def freeze_for_export(self, imgsz) -> None:
        """Hook called by Detector.export on a copy of the model: static anchors + deployed form."""
        if tuple(imgsz) != tuple(self.config.imgsz):
            self.config = dataclasses.replace(self.config, imgsz=tuple(imgsz))
            self._set_anchors(imgsz)
        self.deploy()
