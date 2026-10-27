import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from optinpu.bench import stage_submodel
from optinpu.data.coco import CocoSubset
from optinpu.devices import core
from optinpu.evaluate import evaluate_coco
from optinpu.export import export_openvino
from optinpu.postprocess import Detections
from optinpu.quantize import quantize_int8
from optinpu.runtime import compile_model


@pytest.fixture
def tiny_coco(tmp_path):
    images, annotations = [], []
    for i in range(1, 4):
        Image.new("RGB", (100, 80)).save(tmp_path / f"{i:012d}.jpg")
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": 100, "height": 80})
        annotations.append({"id": i, "image_id": i, "category_id": 3, "bbox": [10, 10, 30, 20], "area": 600, "iscrowd": 0})
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "a"}, {"id": 3, "name": "b"}],
    }))
    return CocoSubset(ann, tmp_path, [1, 2, 3], [1, 3])


class OracleDetector:
    name = "oracle"

    def __init__(self, label):
        self.label = label

    def predict(self, source, device, precision, conf):
        return Detections(np.array([[10, 10, 40, 30]], np.float32), np.array([0.9]), np.array([self.label]), ["a", "b"])


def test_evaluate_perfect_and_wrong_class(tiny_coco):
    assert evaluate_coco(OracleDetector(1), tiny_coco).AP == pytest.approx(1.0)
    assert evaluate_coco(OracleDetector(0), tiny_coco).AP == 0.0


class TwoStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU())
        self.head = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        return self.head(self.backbone(x))


@pytest.fixture(scope="module")
def two_stage_ir(tmp_path_factory):
    torch.manual_seed(0)
    model = TwoStage().eval()
    return model, export_openvino(model, tmp_path_factory.mktemp("ir") / "m.xml", (1, 3, 32, 32), fp16=False)


def test_stage_submodel_cuts_before_head(two_stage_ir):
    model, ir = two_stage_ir
    sub = stage_submodel(core().read_model(ir), r"backbone")
    assert len(sub.outputs) == 1
    assert tuple(sub.outputs[0].get_partial_shape().to_shape()) == (1, 16, 16, 16)


def test_quantize_int8_stays_close(two_stage_ir, tmp_path):
    model, ir = two_stage_ir
    rng = np.random.default_rng(0)
    calib = [rng.random((1, 3, 32, 32), dtype=np.float32) for _ in range(8)]
    q = quantize_int8(ir, calib, lambda x: x, tmp_path / "q.xml", subset_size=8, transformer=False)
    assert any(op.get_type_name() == "FakeQuantize" for op in core().read_model(q).get_ops())
    x = calib[0]
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = compile_model(q, "cpu")(x)[0]
    assert np.abs(out - ref).max() / np.abs(ref).max() < 0.1
