"""M1 blocks: small PyTorch modules, each isolating one NPU cost. Random weights: latency only.

Suites:
  S1 overhead   chains of 1..64 identical cheap layers -> fixed cost per op
  S2 conv       conv 3x3 / 1x1 / depthwise / stride 2 over widths and resolutions -> throughput
  S3 fusion     activations, normalization, residual / concat structures around conv 3x3
  S4 resample   upsampling, downsampling, concat / split, layout changes between convs
  S5 global     MLP on tokens, LayerNorm, global and windowed attention, conv token mixer, top-k
  S6 backbone   complete torchvision backbones and a plain 3x3 network, resolution sweep
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Block:
    suite: str
    name: str
    build: Callable[[], nn.Module]
    input_shape: tuple[int, ...]
    params: dict = field(default_factory=dict)
    int8: bool = True
    transformer: bool = False  # NNCF quantization scheme


# ---------------------------------------------------------------- generic modules

class Chain(nn.Module):
    def __init__(self, make: Callable[[], nn.Module], n: int):
        super().__init__()
        self.layers = nn.ModuleList(make() for _ in range(n))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def conv(cin, cout, k=3, s=1, groups=1, act: Callable[[], nn.Module] | None = nn.ReLU):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups)]
    if act is not None:
        layers.append(act())
    return nn.Sequential(*layers)


class Between(nn.Module):
    """conv1x1 -> op -> conv1x1: puts the op in a realistic context (between two NPU convs)."""

    def __init__(self, cin, op: nn.Module, cmid, cout):
        super().__init__()
        self.pre, self.op, self.post = conv(cin, cin, 1), op, conv(cmid, cout, 1)

    def forward(self, x):
        return self.post(self.op(self.pre(x)))


def count_flops(module: nn.Module, input_shape) -> float:
    """2 x multiply-accumulates of Conv2d, ConvTranspose2d and Linear layers (other ops not counted)."""
    total = [0.0]

    def hook(m, inp, out):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            k = m.kernel_size[0] * m.kernel_size[1]
            if isinstance(m, nn.Conv2d):
                total[0] += 2 * out.numel() * (m.in_channels // m.groups) * k
            else:
                total[0] += 2 * inp[0].numel() * (m.out_channels // m.groups) * k
        elif isinstance(m, nn.Linear):
            total[0] += 2 * out.numel() * m.in_features

    handles = [m.register_forward_hook(hook) for m in module.modules()
               if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))]
    with torch.no_grad():
        module(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return total[0]


# ---------------------------------------------------------------- S1 overhead

def s1() -> list[Block]:
    blocks = []
    for c, hw in ((16, 20), (64, 80)):
        for n in (1, 2, 4, 8, 16, 32, 64):
            blocks.append(Block("S1", f"conv1x1_relu_c{c}_{hw}_n{n}", lambda c=c, n=n: Chain(lambda: conv(c, c, 1), n),
                                (1, c, hw, hw), {"kind": "conv1x1_relu", "c": c, "hw": hw, "n": n}, int8=False))
            blocks.append(Block("S1", f"silu_c{c}_{hw}_n{n}", lambda n=n: Chain(nn.SiLU, n),
                                (1, c, hw, hw), {"kind": "silu", "c": c, "hw": hw, "n": n}, int8=False))
    return blocks


# ---------------------------------------------------------------- S2 conv throughput

def s2() -> list[Block]:
    blocks = []
    for hw in (160, 80, 40, 20):
        for c in (16, 32, 64, 96, 128, 192, 256):
            specs = {
                "conv3x3": lambda c=c: Chain(lambda: conv(c, c, 3), 4),
                "conv1x1": lambda c=c: Chain(lambda: conv(c, c, 1), 4),
                "dwconv3x3": lambda c=c: Chain(lambda: conv(c, c, 3, groups=c), 4),
                "conv3x3_s2": lambda c=c: conv(c, 2 * c, 3, s=2),
            }
            for op, build in specs.items():
                blocks.append(Block("S2", f"{op}_c{c}_{hw}", build, (1, c, hw, hw), {"op": op, "c": c, "hw": hw}))
    for c in (56, 60, 63, 64, 65, 68, 72, 80):  # channel alignment at 80x80
        blocks.append(Block("S2", f"align_conv3x3_c{c}_80", lambda c=c: Chain(lambda: conv(c, c, 3), 4),
                            (1, c, 80, 80), {"op": "conv3x3_align", "c": c, "hw": 80}))
    return blocks


# ---------------------------------------------------------------- S3 activations and structures

class Residual(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        return x + self.body(x)


class ConcatMerge(nn.Module):
    """cat([x, f(x)]) -> 1x1 conv (CSP / ELAN style)."""

    def __init__(self, c):
        super().__init__()
        self.body, self.merge = conv(c, c, 3), conv(2 * c, c, 1)

    def forward(self, x):
        return self.merge(torch.cat([x, self.body(x)], 1))


class RepVGGTrain(nn.Module):
    """Training form of a RepVGG block (3x3 + 1x1 + identity); its deploy form is a single conv3x3."""

    def __init__(self, c):
        super().__init__()
        self.k3, self.k1 = nn.Conv2d(c, c, 3, 1, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        return F.relu(self.k3(x) + self.k1(x) + x)


def s3() -> list[Block]:
    c, hw = 64, 80
    acts = {
        "none": None, "relu": nn.ReLU, "relu6": nn.ReLU6, "leakyrelu": nn.LeakyReLU, "silu": nn.SiLU,
        "hardswish": nn.Hardswish, "gelu": nn.GELU, "sigmoid": nn.Sigmoid,
    }
    blocks = [Block("S3", f"act_{name}", lambda a=a: Chain(lambda: conv(c, c, 3, act=a), 4), (1, c, hw, hw),
                    {"variant": f"act {name}"}) for name, a in acts.items()]

    def conv_bn(act):
        return lambda: nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), act())

    blocks += [
        Block("S3", "bn_relu", lambda: Chain(conv_bn(nn.ReLU), 4), (1, c, hw, hw), {"variant": "conv+BN+ReLU"}),
        Block("S3", "bn_silu", lambda: Chain(conv_bn(nn.SiLU), 4), (1, c, hw, hw), {"variant": "conv+BN+SiLU"}),
        Block("S3", "residual", lambda: Chain(lambda: Residual(conv(c, c, 3)), 4), (1, c, hw, hw),
              {"variant": "residual add"}),
        Block("S3", "concat_merge", lambda: Chain(lambda: ConcatMerge(c), 4), (1, c, hw, hw),
              {"variant": "concat + 1x1 merge"}),
        Block("S3", "repvgg_train", lambda: Chain(lambda: RepVGGTrain(c), 4), (1, c, hw, hw),
              {"variant": "RepVGG training form"}),
    ]
    return blocks


# ---------------------------------------------------------------- S4 resampling and merging

def s4() -> list[Block]:
    c = 64
    up = (1, c, 40, 40)
    down = (1, c, 80, 80)

    def shuffle(x):
        b, ch, h, w = x.shape
        return x.reshape(b, 2, ch // 2, h, w).transpose(1, 2).reshape(b, ch, h, w)

    def tokens_roundtrip(x):  # NCHW -> (B, HW, C) -> NCHW, as around attention layers
        b, ch, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        return t.transpose(1, 2).reshape(b, ch, h, w)

    class GLU(nn.Module):
        def forward(self, x):
            a, g = x.chunk(2, 1)
            return a * torch.sigmoid(g)

    class TwoBranches(nn.Module):
        def __init__(self, merge):
            super().__init__()
            self.a, self.b, self.merge = conv(c, c, 1), conv(c, c, 1), merge

        def forward(self, x):
            return self.merge(self.a(x), self.b(x))

    specs = {
        # upsampling 40 -> 80
        "up_nearest": (up, lambda: Between(c, nn.Upsample(scale_factor=2, mode="nearest"), c, c)),
        "up_bilinear": (up, lambda: Between(c, nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), c, c)),
        "up_convtranspose": (up, lambda: Between(c, nn.ConvTranspose2d(c, c, 2, 2), c, c)),
        "up_pixelshuffle": (up, lambda: Between(c, nn.Sequential(nn.Conv2d(c, 4 * c, 1), nn.PixelShuffle(2)), c, c)),
        # downsampling 80 -> 40
        "down_conv3x3_s2": (down, lambda: Between(c, nn.Conv2d(c, c, 3, 2, 1), c, c)),
        "down_maxpool3x3_s2": (down, lambda: Between(c, nn.MaxPool2d(3, 2, 1), c, c)),
        "down_maxpool2x2_s2": (down, lambda: Between(c, nn.MaxPool2d(2, 2), c, c)),
        "down_avgpool2x2_s2": (down, lambda: Between(c, nn.AvgPool2d(2, 2), c, c)),
        "down_pixelunshuffle": (down, lambda: Between(c, nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * c, c, 1)), c, c)),
        # merging and layout, at 80x80
        "merge_sum": (down, lambda: Between(c, TwoBranches(lambda a, b: a + b), c, c)),
        "merge_concat": (down, lambda: Between(c, TwoBranches(lambda a, b: torch.cat([a, b], 1)), 2 * c, c)),
        "split_glu": (down, lambda: Between(c, nn.Sequential(nn.Conv2d(c, 2 * c, 1), GLU()), c, c)),
        "channel_shuffle": (down, lambda: Between(c, Lambda(shuffle), c, c)),
        "tokens_roundtrip": (down, lambda: Between(c, Lambda(tokens_roundtrip), c, c)),
        "identity": (down, lambda: Between(c, nn.Identity(), c, c)),
    }
    return [Block("S4", name, build, shape, {"op": name}) for name, (shape, build) in specs.items()]


# ---------------------------------------------------------------- S5 global ops

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TokenNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a, self.norm, self.b = nn.Linear(d, d), nn.LayerNorm(d), nn.Linear(d, d)

    def forward(self, x):
        return self.b(self.norm(self.a(x)))


class Attention(nn.Module):
    """Multi-head self-attention on (B, T, D), head dim 32."""

    def __init__(self, d):
        super().__init__()
        self.heads = d // 32
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class WindowAttention(nn.Module):
    """Attention inside non-overlapping w x w windows of a (B, D, H, W) map (Swin style, no shift)."""

    def __init__(self, d, w=8):
        super().__init__()
        self.w, self.attn = w, Attention(d)

    def forward(self, x):
        b, d, h, wd = x.shape
        w = self.w
        t = x.reshape(b, d, h // w, w, wd // w, w).permute(0, 2, 4, 3, 5, 1).reshape(-1, w * w, d)
        t = self.attn(t)
        return t.reshape(b, h // w, wd // w, w, w, d).permute(0, 5, 1, 3, 2, 4).reshape(b, d, h, wd)


class ConvMixer(nn.Module):
    """Large-kernel depthwise conv + pointwise MLP (ConvNeXt-style token mixer, no LayerNorm)."""

    def __init__(self, d, k=7):
        super().__init__()
        self.dw = nn.Conv2d(d, d, k, 1, k // 2, groups=d)
        self.pw1, self.pw2 = nn.Conv2d(d, 4 * d, 1), nn.Conv2d(4 * d, d, 1)

    def forward(self, x):
        return x + self.pw2(F.relu(self.pw1(self.dw(x))))


class TopKSelect(nn.Module):
    """DETR / YOLO26 style query selection: best class score per anchor, top-k, gather boxes."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):  # x: (1, N, 84) = 80 class logits + 4 box coordinates
        scores = x[..., :80].max(-1).values
        idx = scores.topk(self.k, dim=1).indices
        return torch.gather(x[..., 80:], 1, idx.unsqueeze(-1).expand(-1, -1, 4))


def s5() -> list[Block]:
    blocks = []
    for t_side in (20, 40, 80):
        t = t_side * t_side
        for d in (64, 128, 256):
            p = {"tokens": t, "dim": d}
            blocks.append(Block("S5", f"mlp_t{t}_d{d}", lambda d=d: MLP(d), (1, t, d), {"op": "mlp", **p}, transformer=True))
            blocks.append(Block("S5", f"layernorm_t{t}_d{d}", lambda d=d: TokenNorm(d), (1, t, d), {"op": "layernorm", **p},
                                transformer=True))
            if t <= 1600:  # global attention on 6400 tokens: 8 x 6400^2 scores, too large to be useful
                blocks.append(Block("S5", f"attention_t{t}_d{d}", lambda d=d: Attention(d), (1, t, d),
                                    {"op": "attention", **p}, transformer=True))
            if t_side >= 40:
                blocks.append(Block("S5", f"window_attention_t{t}_d{d}", lambda d=d: WindowAttention(d), (1, d, t_side, t_side),
                                    {"op": "window_attention", **p}, transformer=True))
                blocks.append(Block("S5", f"conv_mixer_t{t}_d{d}", lambda d=d: ConvMixer(d), (1, d, t_side, t_side),
                                    {"op": "conv_mixer", **p}))
    for n in (2100, 8400):
        for k in (100, 300):
            blocks.append(Block("S5", f"topk_n{n}_k{k}", lambda k=k: TopKSelect(k), (1, n, 84),
                                {"op": "topk", "n": n, "k": k}, int8=False))
    return blocks


# ---------------------------------------------------------------- S6 backbones

class PlainNet(nn.Module):
    """VGG-like stack of conv3x3 + ReLU (the deploy form of RepVGG-A0: widths 48-48-96-192-1280)."""

    def __init__(self, widths=(48, 48, 96, 192, 1280), depths=(1, 2, 4, 14, 1)):
        super().__init__()
        layers, cin = [], 3
        for w, n in zip(widths, depths):
            for i in range(n):
                layers.append(conv(cin, w, 3, s=2 if i == 0 else 1))
                cin = w
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


def _torchvision(name):
    import torchvision.models as tvm

    def build():
        m = getattr(tvm, name)(weights=None)
        if name.startswith("resnet"):
            return nn.Sequential(*list(m.children())[:-2])
        if name.startswith("regnet"):
            return nn.Sequential(m.stem, m.trunk_output)
        return m.features  # mobilenet, efficientnet, convnext

    return build


def s6() -> list[Block]:
    nets = {
        "resnet18": _torchvision("resnet18"), "resnet34": _torchvision("resnet34"), "resnet50": _torchvision("resnet50"),
        "regnet_x_400mf": _torchvision("regnet_x_400mf"), "regnet_x_800mf": _torchvision("regnet_x_800mf"),
        "regnet_x_1_6gf": _torchvision("regnet_x_1_6gf"), "mobilenet_v2": _torchvision("mobilenet_v2"),
        "mobilenet_v3_large": _torchvision("mobilenet_v3_large"), "efficientnet_b0": _torchvision("efficientnet_b0"),
        "convnext_tiny": _torchvision("convnext_tiny"), "plainnet_a0": PlainNet,
    }
    blocks = [Block("S6", f"{name}_640", build, (1, 3, 640, 640), {"net": name, "res": 640},
                    transformer=name.startswith("convnext")) for name, build in nets.items()]
    for name in ("resnet18", "regnet_x_400mf", "plainnet_a0"):
        for res in (320, 416, 512):
            blocks.append(Block("S6", f"{name}_{res}", nets[name], (1, 3, res, res), {"net": name, "res": res}))
    return blocks


# ---------------------------------------------------------------- S7 marginal cost per layer

def s7() -> list[Block]:
    """Chains of 2 and 10 identical layers: (t10 - t2) / 8 is the cost of one layer inside a network,
    without the input / output transfers that dominate single-block measurements (S2)."""
    blocks = []
    for hw in (160, 80, 40, 20):
        for c in (32, 64, 128, 256):
            specs = {
                "conv3x3": lambda c=c: conv(c, c, 3),
                "conv1x1": lambda c=c: conv(c, c, 1),
                "dwconv3x3": lambda c=c: conv(c, c, 3, groups=c),
                "dwsep3x3": lambda c=c: nn.Sequential(conv(c, c, 3, groups=c, act=None), conv(c, c, 1)),
                "conv3x3_silu": lambda c=c: conv(c, c, 3, act=nn.SiLU),
            }
            for op, make in specs.items():
                for n in (2, 10):
                    blocks.append(Block("S7", f"m_{op}_c{c}_{hw}_n{n}", lambda make=make, n=n: Chain(make, n),
                                        (1, c, hw, hw), {"op": op, "c": c, "hw": hw, "n": n}))
    return blocks


def m2() -> list[Block]:
    """OptiDet candidates (benchmarks/m2/candidates.py), measured with this machinery."""
    from benchmarks.m2.candidates import candidates

    return candidates()


SUITES = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": s6, "S7": s7, "M2": m2}


def all_blocks(suites=None) -> list[Block]:
    return [b for s in (suites or SUITES) for b in SUITES[s]()]
