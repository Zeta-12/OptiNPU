import numpy as np
import pytest
import torch

from optinpu.devices import available_devices
from optinpu.models.optidet import OptiDet, OptiDetConfig, RepConv

SMALL = OptiDetConfig(widths=(16, 16, 32, 32, 64), depths=(1, 1, 2, 2, 1), neck_width=32, head_width=32,
                      reg_max=8, num_classes=5, imgsz=(128, 128))


def _randomize_bn(module):
    """Non-trivial BatchNorm statistics, so that reparameterization is really tested."""
    g = torch.Generator().manual_seed(0)
    for m in module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(m.num_features, generator=g) * 0.1)


@pytest.mark.parametrize("cin,cout,stride", [(8, 8, 1), (8, 16, 1), (8, 16, 2), (8, 8, 2)])
def test_repconv_deploy_is_exact(cin, cout, stride):
    torch.manual_seed(0)
    block = RepConv(cin, cout, stride)
    _randomize_bn(block)
    block.eval()
    x = torch.randn(2, cin, 9, 10)
    with torch.no_grad():
        ref = block(x)
        block.deploy()
        out = block(x)
    assert block.k3 is None and block.fused is not None
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_optidet_shapes_and_contract():
    model = OptiDet(SMALL).eval()
    assert model.num_anchors == 16 * 16 + 8 * 8 + 4 * 4
    with torch.no_grad():
        scores, boxes = model(torch.rand(2, 3, 128, 128))
        train = model.forward_train(torch.rand(2, 3, 128, 128))
    assert scores.shape == (2, model.num_anchors, 5) and boxes.shape == (2, model.num_anchors, 4)
    assert 0 <= scores.min() and scores.max() <= 1
    assert torch.all(boxes[..., 2:] >= boxes[..., :2])  # xyxy, non-negative distances
    cls, reg = train["one2many"]
    assert cls.shape == (2, model.num_anchors, 5) and reg.shape == (2, model.num_anchors, 4, 8)
    # class prior: ~5 objects per image over each level's cells and classes
    assert scores.mean() < 0.01 and scores[:, -16:].mean() > scores[:, :256].mean()  # coarse levels higher


def test_uniform_distribution_gives_mean_bin_distance():
    model = OptiDet(SMALL)
    reg = torch.zeros(1, model.num_anchors, 4, 8)  # uniform over 8 bins -> expected 3.5 bins
    d = model.distances(reg)
    torch.testing.assert_close(d, 3.5 * model.anchor_strides.expand(-1, 4).unsqueeze(0))
    boxes = model.boxes(d)
    torch.testing.assert_close(boxes[0, :, :2], model.anchors - 3.5 * model.anchor_strides)


def test_deploy_matches_training_form():
    torch.manual_seed(0)
    model = OptiDet(SMALL)
    _randomize_bn(model)
    model.eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        ref = model(x)
        model.deploy()
        out = model(x)
    assert model.one2many is None
    for a, b in zip(ref, out):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)


def test_repvgg_a0_weights_reproduce_timm_stages():
    timm = pytest.importorskip("timm")
    from optinpu.models.optidet import Backbone, load_repvgg_a0

    try:
        reference = timm.create_model("repvgg_a0.rvgg_in1k", pretrained=True, features_only=True).eval()
    except Exception as e:  # no network and no cached weights
        pytest.skip(f"RepVGG-A0 weights unavailable: {e}")
    backbone = Backbone((48, 48, 96, 192, 384), (1, 2, 4, 14, 1))
    loaded = load_repvgg_a0(backbone)
    assert len(loaded) == len(backbone.state_dict())
    backbone.eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        ref = reference(x)  # 5 maps: strides 2, 4, 8, 16, 32
        ours, y = [], x
        for stage in backbone.stages:
            y = stage(y)
            ours.append(y)
    for i in range(4):  # identical stages
        torch.testing.assert_close(ours[i], ref[i], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ours[4], ref[4][:, :384], rtol=1e-4, atol=1e-4)  # truncated last block


def test_export_and_run(tmp_path):
    from optinpu.export import export_openvino
    from optinpu.runtime import compile_model

    torch.manual_seed(0)
    model = OptiDet(SMALL).eval()
    model.freeze_for_export((128, 128))
    ir = export_openvino(model, tmp_path / "optidet.xml", (1, 3, 128, 128))
    x = np.random.default_rng(0).random((1, 3, 128, 128), dtype=np.float32)
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(x))]
    devices = ["cpu"] + (["npu"] if "NPU" in available_devices() else [])
    for device in devices:
        out = compile_model(ir, device, cache_dir=None)(x)
        for a, b in zip(ref, out):
            assert np.abs(a - b).max() < 1e-2, device
