import numpy as np
import openvino as ov
import pytest
import torch
from torch import nn

from optinpu.devices import available_devices, core
from optinpu.export import export_openvino
from optinpu.graph import split_after, split_model
from optinpu.lint import lint
from optinpu.runtime import compile_hybrid


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(), nn.Conv2d(8, 8, 3, 1, 1))
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f).sigmoid(), f.mean((2, 3))  # the backbone output feeds two consumers


@pytest.fixture(scope="module")
def net_ir(tmp_path_factory):
    torch.manual_seed(0)
    model = Net().eval()
    return model, export_openvino(model, tmp_path_factory.mktemp("ir") / "net.xml", (1, 3, 32, 32), fp16=False)


def _cpu(model, inputs):
    return list(core().compile_model(model, "CPU").create_infer_request().infer(inputs).values())


def test_split_model_roundtrip_matches_full(net_ir):
    _, ir = net_ir
    m = core().read_model(ir)
    x = np.random.default_rng(0).random((1, 3, 32, 32), dtype=np.float32)
    full = _cpu(m, [x])
    n_ops = len(m.get_ordered_ops())
    for cut in range(1, n_ops - 2):
        head, tail = split_model(m, cut)
        out = _cpu(tail, _cpu(head, [x]))
        for a, b in zip(full, out):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_split_after_pattern_and_hybrid_cpu(net_ir):
    _, ir = net_ir
    m = core().read_model(ir)
    head, tail = split_after(m, r"backbone")
    assert len(head.outputs) == 1  # one tensor crosses, even with two consumers
    x = np.random.default_rng(1).random((1, 3, 32, 32), dtype=np.float32)
    hybrid = compile_hybrid(ir, r"backbone", "cpu", "cpu")
    for a, b in zip(_cpu(m, [x]), hybrid(x)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        split_after(m, r"does-not-exist")


def test_lint_flags_static_problems():
    from openvino import opset13 as ops
    from openvino import opset14

    p = ops.parameter([1, 3, 9, 9], np.float32)
    ceil_pool = opset14.avg_pool(p, [2, 2], [0, 0], [0, 0], [2, 2], False, rounding_type="ceil_torch")  # ResNet-vd
    pool = ops.max_pool(ceil_pool, [1, 1], [1, 1], [0, 0], [0, 0], [2, 2])  # k2 s1
    big = ops.multiply(pool.output(0), ops.constant(np.full((1,), 1e6, np.float32)))
    report = lint(ov.Model([big], [p]), device="cpu", check_accuracy=False)
    codes = {i.code for i in report.issues}
    assert {"POOL-CEIL", "MAXPOOL-2x2-S1", "FP16-RANGE"} <= codes
    assert not report.ok


def test_lint_shape_subgraph_only_flags_float_values():
    from openvino import opset13 as ops

    p = ops.parameter([1, 4, 6, 6], np.float32)
    shape = ops.shape_of(p)
    # integer shape arithmetic for a reshape: harmless
    flat = ops.reshape(p, ops.concat([ops.gather(shape, ops.constant([0]), ops.constant(0)), ops.constant([-1])], 0), False)
    report = lint(ov.Model([flat], [p]), device="cpu", check_accuracy=False)
    assert "SHAPE-SUBGRAPH" not in {i.code for i in report.issues}
    # float positions computed from the shape (like anchors): flagged
    w = ops.convert(ops.gather(shape, ops.constant([3]), ops.constant(0)), ov.Type.f32)
    pos = ops.range(ops.constant(np.float32(0)), ops.squeeze(w, ops.constant([0])), ops.constant(np.float32(1)), ov.Type.f32)
    out = ops.add(p, pos)
    report = lint(ov.Model([out], [p]), device="cpu", check_accuracy=False)
    assert "SHAPE-SUBGRAPH" in {i.code for i in report.issues}


def test_lint_dynamic_input_is_an_error():
    from openvino import opset13 as ops

    p = ops.parameter([-1, 3, 8, 8], np.float32)
    report = lint(ov.Model([ops.relu(p)], [p]), device="cpu")
    assert [i.code for i in report.issues if i.severity == "error"] == ["DYNAMIC-INPUT"]


def test_lint_clean_model_on_npu(net_ir):
    if "NPU" not in available_devices():
        pytest.skip("no NPU")
    _, ir = net_ir
    report = lint(ir, "npu")
    assert report.compiles and report.ok, str(report)
    assert report.rel_error < 0.05
