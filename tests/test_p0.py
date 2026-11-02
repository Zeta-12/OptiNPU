import sys

import numpy as np
import openvino as ov
import pytest
import torch
from PIL import Image

from optinpu.export import load_model
from optinpu.sandbox import run_job, run_outputs, run_watched
from optinpu.tasks import DetectionTask, OutputTask, random_samples


def test_letterbox_dets_map_back_to_image_pixels():
    task = DetectionTask((64, 64), output="dets", resize="letterbox")
    image = Image.new("RGB", (200, 100))
    (x,), meta = task.inputs(image)
    assert x.shape == (1, 3, 64, 64)
    scale, (pad_x, pad_y) = meta[1]
    assert scale == pytest.approx(0.32) and (pad_x, pad_y) == (0, 16)
    box = np.array([20, 10, 80, 50], dtype=np.float32)  # image pixels
    dets = np.array([[*(box * scale + [pad_x, pad_y, pad_x, pad_y]), 0.9, 3]], dtype=np.float32)
    out = task.decode([dets[None]], meta)
    np.testing.assert_allclose(out.boxes[0], box, atol=1e-4)
    assert out.labels.tolist() == [3] and out.scores[0] == pytest.approx(0.9)


def test_stretched_dets_scale_per_axis():
    task = DetectionTask((64, 64), output="dets")
    _, meta = task.inputs(Image.new("RGB", (128, 32)))
    out = task.decode([np.array([[[32, 32, 64, 64, 0.8, 1]]], dtype=np.float32)], meta)
    np.testing.assert_allclose(out.boxes[0], [64, 16, 128, 32])


def test_detr_logits_with_class_columns():
    task = DetectionTask((32, 32), output="detr", class_columns=[2, 0], class_names=["a", "b"])
    _, meta = task.inputs(Image.new("RGB", (100, 50)))
    logits = np.full((1, 1, 3), -10.0, dtype=np.float32)
    logits[0, 0, 0] = 4.0  # column 0 is class "b"
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32)  # cxcywh
    out = task.decode([boxes, logits], meta, conf=0.5)  # either output order
    assert out.labels.tolist() == [1] and out.scores[0] == pytest.approx(1 / (1 + np.exp(-4.0)), rel=1e-5)
    np.testing.assert_allclose(out.boxes[0], [40, 15, 60, 35], atol=1e-4)


def test_output_task_agreement():
    task = OutputTask()
    ref = [[np.ones((2, 3), np.float32)]]
    assert task.agreement(ref, ref) == 1.0
    assert task.agreement(ref, [[np.full((2, 3), 1.1, np.float32)]]) == pytest.approx(0.9, abs=1e-6)
    assert task.agreement(ref, [[np.full((2, 3), -1.0, np.float32)]]) == 0.0


def _tiny_torch():
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 4, 1))


def test_load_model_from_torch_and_ir(tmp_path):
    model = load_model(_tiny_torch(), (1, 3, 16, 16))
    assert model.inputs[0].get_partial_shape().to_shape() == ov.Shape([1, 3, 16, 16])
    ov.save_model(model, tmp_path / "m.xml")
    assert load_model(tmp_path / "m.xml").outputs[0].get_partial_shape().to_shape() == ov.Shape([1, 4, 16, 16])
    with pytest.raises(ValueError, match="input_shape"):
        load_model(_tiny_torch())


def test_load_model_fixes_a_dynamic_input(tmp_path):
    with torch.no_grad():
        dynamic = ov.convert_model(_tiny_torch().eval(), example_input=torch.zeros(1, 3, 16, 16))
    assert dynamic.inputs[0].get_partial_shape().is_dynamic
    with pytest.raises(ValueError, match="dynamic"):
        load_model(dynamic)
    assert load_model(dynamic, (1, 3, 24, 24)).outputs[0].get_partial_shape().to_shape() == ov.Shape([1, 4, 24, 24])


def test_watchdog_kills_on_timeout():
    watched = run_watched([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert watched.status.startswith("killed: timeout") and watched.seconds < 10


def test_sandboxed_jobs_match_in_process_results():
    model = load_model(_tiny_torch(), (1, 3, 16, 16))
    job = run_job({"job": "latency", "ir": model, "device": "cpu", "iterations": 5}, timeout=120)
    assert job.ok and job.result["latency_ms_median"] > 0, job.error

    samples = random_samples(model, 2)
    outputs, job = run_outputs(model, samples, "cpu", timeout=120)
    assert job.ok, job.error
    compiled = ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for sample, out in zip(samples, outputs):
        np.testing.assert_allclose(out[0], compiled(sample)[0], rtol=1e-5, atol=1e-6)


class TinyDetector(torch.nn.Module):
    """optinpu output contract: scores (1, Q, C) in [0, 1], boxes (1, Q, 4) normalized xyxy."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 8, 8)

    def forward(self, x):
        y = self.conv(x).flatten(2).transpose(1, 2)  # (1, 16, 6)
        xy = torch.sigmoid(y[..., :2]) * 0.5
        return torch.sigmoid(y[..., 4:]), torch.cat([xy, xy + 0.4], -1)


def test_sandboxed_predictions_match_in_process(tmp_path):
    from optinpu.sandbox import run_predictions
    from optinpu.tasks import Deployed

    torch.manual_seed(0)
    model = load_model(TinyDetector(), (1, 3, 32, 32))
    images = []
    for i in range(3):
        path = tmp_path / f"{i}.jpg"
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (40, 60, 3), dtype=np.uint8)).save(path)
        images.append(path)
    task = DetectionTask((32, 32), class_names=["a", "b"])
    sandboxed, job = run_predictions([(model, "cpu", None)], task, images, timeout=120)
    assert job.ok, job.error
    for a, b in zip(sandboxed, Deployed(model, task, "cpu", cache_dir=None).run(images)):
        np.testing.assert_allclose(a.boxes, b.boxes, rtol=1e-5)
        assert a.labels.tolist() == b.labels.tolist() and a.class_names == ["a", "b"]
    assert task.agreement(sandboxed, sandboxed) == 1.0

    samples = random_samples(model, 2)
    outputs, job = run_predictions([(model, "cpu", None)], OutputTask(), samples, timeout=120)
    assert job.ok and len(outputs) == 2 and outputs[0][0].shape == (1, 16, 2)


def test_sandboxed_evaluation_matches_in_process(toy_coco):
    from optinpu.data.dataset import CocoDetection
    from optinpu.evaluate import evaluate_coco
    from optinpu.sandbox import run_evaluation
    from optinpu.tasks import Deployed

    torch.manual_seed(0)
    model = load_model(TinyDetector(), (1, 3, 32, 32))
    subset = CocoDetection(*toy_coco, imgsz=(32, 32)).to_subset()
    task = DetectionTask((32, 32), class_names=["other", "thing"])
    result, job = run_evaluation([(model, "cpu", None)], task, subset, timeout=120)
    assert job.ok, job.error
    expected = evaluate_coco(Deployed(model, task, "cpu", cache_dir=None), subset, "cpu", "fp32", folds=3)
    assert result["AP"] == pytest.approx(expected.AP) and result["num_images"] == 6
    assert len(result["fold_ap"]) == 10 and len(expected.fold_ap) == 3  # the sandbox job uses 10 folds


def test_classification_inputs_match_torchvision():
    import torchvision.models as tvm

    from optinpu.tasks import ClassificationTask, task_from_dict, task_to_dict

    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (180, 300, 3), dtype=np.uint8))
    for weights, task in ((tvm.ResNet50_Weights.DEFAULT, ClassificationTask((224, 224), 232, "bilinear")),
                          (tvm.EfficientNet_B0_Weights.DEFAULT, ClassificationTask((224, 224), 256, "bicubic"))):
        (x,), _ = task.inputs(image)
        np.testing.assert_allclose(x[0], weights.transforms()(image).numpy(), atol=1e-5)
        assert task_from_dict(task_to_dict(task)) == task
    a, b = [np.array([0.1, 2.0, 0.3])], [np.array([0.1, 0.2, 3.0])]
    assert ClassificationTask().agreement([a, a], [a, b]) == 0.5  # same top-1 on one sample of two


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, 2)
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))


def test_sandboxed_top1_matches_in_process(tmp_path):
    from optinpu.data.imagenet import imagenetv2_subset, labeled_folder
    from optinpu.evaluate import evaluate_top1
    from optinpu.sandbox import run_evaluation
    from optinpu.tasks import ClassificationTask, Deployed

    rng = np.random.default_rng(0)
    for label in range(3):
        (tmp_path / "images" / str(label)).mkdir(parents=True)
        for i in range(4):
            Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(tmp_path / "images" / str(label) / f"{i}.jpeg")
    every = labeled_folder(tmp_path)
    assert len(every.paths) == 12 and every.labels[:4] == [0] * 4
    subset = imagenetv2_subset(8, 2, root=tmp_path)
    assert subset.paths == imagenetv2_subset(8, 2, root=tmp_path).paths  # one fixed shuffle
    assert not set(subset.paths) & set(imagenetv2_subset(2, 0, root=tmp_path).paths)  # disjoint slices

    torch.manual_seed(0)
    model = load_model(TinyClassifier(), (1, 3, 32, 32))
    task = ClassificationTask((32, 32), 36)
    result, job = run_evaluation([(model, "cpu", None)], task, subset, timeout=120)
    assert job.ok, job.error
    expected = evaluate_top1(Deployed(model, task, "cpu", cache_dir=None), subset, folds=4)
    assert result["score"] == pytest.approx(expected["top1"]) and result["num_images"] == 8
    assert len(result["fold_scores"]) == 10 and len(expected["fold_top1"]) == 4


def test_cli_optimizes_a_classifier_judged_by_top1(tmp_path, monkeypatch):
    import openvino as ov

    import optinpu.optimizer
    from optinpu.cli import main
    from optinpu.data.imagenet import LabeledImages
    from optinpu.tasks import ClassificationTask

    torch.manual_seed(0)
    ov.save_model(load_model(TinyClassifier(), (1, 3, 32, 32)), tmp_path / "cls.xml")
    for folder in ("guard", "labeled/0", "labeled/1"):
        (tmp_path / folder).mkdir(parents=True)
        Image.new("RGB", (40, 50)).save(tmp_path / folder / "a.jpg")
    seen = {}

    def fake_optimize(model, task, samples, **options):
        seen.update(task=task, samples=samples, **options)
        raise SystemExit(0)

    monkeypatch.setattr(optinpu.optimizer, "optimize", fake_optimize)
    with pytest.raises(SystemExit):
        main(["optimize", str(tmp_path / "cls.xml"), "--task", "classification", "--images", str(tmp_path / "guard"),
              "--labels", str(tmp_path / "labeled"), "--interpolation", "bicubic", "--folds", "4"])
    assert seen["task"] == ClassificationTask((32, 32), 37, "bicubic") and len(seen["samples"]) == 1
    assert isinstance(seen["evaluation"], LabeledImages) and seen["evaluation"].labels == [0, 1]
    assert seen["folds"] == 4


def test_failed_job_reports_the_error(tmp_path):
    job = run_job({"job": "latency", "ir": str(tmp_path / "missing.xml"), "device": "cpu"}, timeout=120)
    assert not job.ok and job.error
