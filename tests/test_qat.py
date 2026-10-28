import numpy as np
import openvino as ov
import pytest
import torch

from optinpu import Detector
from optinpu.models import registry

pytest.importorskip("nncf")


@pytest.fixture
def tiny_optidet(monkeypatch):
    """A 64-pixel OptiDet registered for the test (used with pretrained=False: no ImageNet download)."""
    spec = registry.ModelSpec(name="optidet-tiny", imgsz=(64, 64), builder=registry._optidet(
        widths=(16, 16, 32, 32, 64), depths=(1, 1, 2, 2, 1), neck_width=32, head_width=32, reg_max=8,
        imgsz=(64, 64)))
    monkeypatch.setitem(registry._REGISTRY, spec.name, spec)
    return spec.name


def test_train_qat_exports_int8_and_reloads(toy_coco, tmp_path, tiny_optidet):
    torch.manual_seed(0)
    detector = Detector(tiny_optidet, pretrained=False, workdir=tmp_path, class_names=["other", "thing"])
    result = detector.train_qat(toy_coco, toy_coco, epochs=1, batch=2, workers=0, device="cpu",
                                warmup_steps=1, log_every=100)

    ir = detector.export()  # int8 by default once quantization-aware trained
    assert ir.parent == tmp_path / tiny_optidet / "qat" and ir.name.startswith(f"{tiny_optidet}_int8")
    assert [o.get_type_name() for o in ov.Core().read_model(ir).get_ops()].count("FakeQuantize") > 10
    with pytest.raises(ValueError, match="int8 only"):
        detector.export(precision="fp16")

    # the INT8 IR reproduces the simulated quantization of the PyTorch model
    x = np.random.default_rng(0).random((1, 3, 64, 64), dtype=np.float32)
    simulated = detector._torch_forward(x)
    for a, b in zip(simulated, detector.compiled("cpu")(x)):
        assert np.abs(a - b).max() < 0.05

    # the checkpoint restores the quantizers without calibration
    reloaded = Detector.from_checkpoint(tiny_optidet, result.checkpoint)
    for a, b in zip(simulated, reloaded._torch_forward(x)):
        np.testing.assert_array_equal(a, b)
    assert reloaded.export().name.startswith(f"{tiny_optidet}_int8")


def test_train_qat_rejects_other_classes(toy_coco, tmp_path, tiny_optidet):
    detector = Detector(tiny_optidet, pretrained=False, workdir=tmp_path)  # 80 COCO classes
    with pytest.raises(ValueError, match="must use its classes"):
        detector.train_qat(toy_coco, epochs=1, workers=0, device="cpu")
