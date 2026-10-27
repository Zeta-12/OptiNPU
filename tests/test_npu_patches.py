import pytest
import torch
from torch import nn

from optinpu.npu_patches import FloorWhenExactPool, MaxPool2x2s1, apply_npu_patches


@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("size", [(8, 8), (9, 7), (321, 321)])
def test_maxpool_2x2_s1_matches_torch(ceil_mode, size):
    x = torch.randn(2, 3, *size)
    ref = nn.MaxPool2d(2, 1, ceil_mode=ceil_mode)(x)
    assert torch.equal(MaxPool2x2s1()(x), ref)


@pytest.mark.parametrize("pool_cls", [nn.MaxPool2d, nn.AvgPool2d])
@pytest.mark.parametrize("size", [(160, 160), (161, 160), (7, 9)])
def test_floor_when_exact_pool_matches_ceil_mode(pool_cls, size):
    pool = pool_cls(2, 2, ceil_mode=True)
    x = torch.randn(1, 4, *size)
    torch.testing.assert_close(FloorWhenExactPool(pool)(x), pool(x))


def test_apply_npu_patches_replaces_and_preserves_output():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.MaxPool2d(2, 1, ceil_mode=True),
        nn.AvgPool2d(2, 2, ceil_mode=True),
        nn.MaxPool2d(3, 2),  # untouched
    ).eval()
    patched, names = apply_npu_patches(model)
    assert names == ["1", "2"]
    assert isinstance(model[1], nn.MaxPool2d), "original model must not be modified"
    x = torch.randn(1, 3, 64, 64)
    torch.testing.assert_close(patched(x), model(x))
