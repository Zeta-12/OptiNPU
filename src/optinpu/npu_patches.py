"""Mathematically equivalent rewrites of PyTorch modules that the Intel NPU compiler mishandles.

Each patch was found with `optinpu.diagnose.locate_divergence` and is validated by tests against
the original module. `apply_npu_patches` is called on a copy of the model before export.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class MaxPool2x2s1(nn.Module):
    """MaxPool2d(kernel=2, stride=1, padding=0) as the max of four shifted slices.

    The NPU compiler (driver 32.0.100.4778, OpenVINO 2026.4) produces wrong values for this pool
    when its input is also consumed by a convolution (HGNetv2 stem in D-FINE / DEIM / RT-DETRv2).
    For stride 1, ceil_mode does not change the output, so both variants map here.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top = torch.maximum(x[..., :-1, :-1], x[..., :-1, 1:])
        bottom = torch.maximum(x[..., 1:, :-1], x[..., 1:, 1:])
        return torch.maximum(top, bottom)


class FloorWhenExactPool(nn.Module):
    """Runs a ceil_mode pool with ceil_mode=False whenever both modes give the same output.

    The NPU compiler rejects rounding_type=ceil_torch ("Bad optional access", ResNet-vd shortcuts
    in RT-DETR). With static shapes the check is resolved at trace time.
    """

    def __init__(self, pool: nn.MaxPool2d | nn.AvgPool2d):
        super().__init__()
        self.pool = pool
        self.floor_pool = copy.deepcopy(pool)
        self.floor_pool.ceil_mode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k, s, p = (_pair(getattr(self.pool, a)) for a in ("kernel_size", "stride", "padding"))
        d = _pair(getattr(self.pool, "dilation", 1))
        exact = all(
            (int(size) + 2 * p[i] - d[i] * (k[i] - 1) - 1) % s[i] == 0
            for i, size in enumerate(x.shape[-2:])
        )
        return self.floor_pool(x) if exact else self.pool(x)


def _pair(v) -> tuple:
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def _is_maxpool_2x2_s1(m: nn.Module) -> bool:
    return (
        isinstance(m, nn.MaxPool2d)
        and _pair(m.kernel_size) == (2, 2)
        and _pair(m.stride) == (1, 1)
        and _pair(m.padding) == (0, 0)
        and _pair(m.dilation) == (1, 1)
        and not m.return_indices
    )


def apply_npu_patches(model: nn.Module, inplace: bool = False) -> tuple[nn.Module, list[str]]:
    """Return (patched model, list of patched module names)."""
    if not inplace:
        model = copy.deepcopy(model)
    patched = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if _is_maxpool_2x2_s1(child):
                replacement = MaxPool2x2s1()
            elif isinstance(child, (nn.MaxPool2d, nn.AvgPool2d)) and child.ceil_mode:
                replacement = FloorWhenExactPool(child)
            else:
                continue
            setattr(module, child_name, replacement)
            patched.append(f"{name}.{child_name}" if name else child_name)
    return model, patched
