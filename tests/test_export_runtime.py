import numpy as np
import pytest
import torch
from torch import nn

from optinpu.devices import available_devices
from optinpu.diagnose import _rel_error
from optinpu.export import export_openvino
from optinpu.runtime import compile_model


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 1, ceil_mode=True)
        self.head = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        x = nn.functional.pad(torch.relu(self.conv(x)), (0, 1, 0, 1))
        return self.head(self.pool(x)).sigmoid()


@pytest.fixture(scope="module")
def tiny_ir(tmp_path_factory):
    torch.manual_seed(0)
    model = Tiny().eval()
    path = export_openvino(model, tmp_path_factory.mktemp("ir") / "tiny.xml", input_shape=(1, 3, 32, 32))
    return model, path


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_exported_model_matches_torch(tiny_ir, device):
    if device.upper() not in available_devices():
        pytest.skip(f"{device} not available")
    model, path = tiny_ir
    x = np.random.default_rng(0).random((1, 3, 32, 32), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = compile_model(path, device)(x)[0]
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-2


def test_rel_error_is_permutation_invariant():
    a = np.arange(10, dtype=np.float32)
    assert _rel_error(a, a[::-1]) == 0
    assert _rel_error(a, a + 1) > 0.1
    assert _rel_error(a, np.full_like(a, np.inf)) == float("inf")
