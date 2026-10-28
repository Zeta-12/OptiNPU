"""High-level Detector API."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from optinpu.bench import BenchResult, benchmark, benchmark_callable, benchmark_stages
from optinpu.devices import normalize_ov_device
from optinpu.export import export_openvino
from optinpu.models import get_model
from optinpu.postprocess import Detections, decode, load_image, preprocess
from optinpu.runtime import CompiledModel, HybridModel, compile_hybrid, compile_model

if TYPE_CHECKING:
    from optinpu.engine import TrainResult

TRAINING_ONLY_PREFIXES = ("distill.",)  # checkpoint entries that inference models do not have


class Detector:
    """Object detector that trains in PyTorch and runs on CPU, GPU or NPU through OpenVINO.

    >>> model = Detector("dfine-s")
    >>> model.predict("image.jpg", device="npu")
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        workdir: str | Path = "runs",
        class_names: list[str] | None = None,
    ):
        """`class_names` different from the pretrained ones re-create the class heads (fine-tuning)."""
        self.spec = get_model(name)
        self.name = name
        if class_names is not None and list(class_names) != list(self.spec.class_names):
            self.spec = dataclasses.replace(self.spec, class_names=list(class_names))
            self.model = self.spec.builder(pretrained, class_names=list(class_names)).eval()
        else:
            self.model = self.spec.builder(pretrained).eval()
        self.workdir = Path(workdir) / name
        self._ir: dict[str, Path] = {}
        self._compiled: dict[str, CompiledModel | HybridModel] = {}

    @classmethod
    def from_checkpoint(cls, name: str, checkpoint: str | Path, workdir: str | Path | None = None) -> "Detector":
        """Load weights saved by `train()` (best.pt)."""
        import torch

        checkpoint = Path(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        detector = cls(name, pretrained=False, class_names=state["class_names"],
                       workdir=workdir or checkpoint.parent.parent)
        weights = state["state_dict"]
        expected = detector.model.state_dict().keys()
        # training-only modules (e.g. OptiDet distillation adapters) are not part of the inference model
        training_only = {k for k in weights if k not in expected and k.startswith(TRAINING_ONLY_PREFIXES)}
        detector.model.load_state_dict({k: v for k, v in weights.items() if k not in training_only})
        detector.model.eval()
        detector.workdir = Path(workdir) if workdir else checkpoint.parent
        return detector

    @property
    def class_names(self) -> list[str]:
        return self.spec.class_names

    def train(self, data, val=None, run: str = "train", **config) -> TrainResult:
        """Fine-tune on a COCO-format dataset, then keep the best EMA weights.

        data / val: CocoDetection, CocoSubset, or (annotations_json, image_dir).
        If the dataset classes differ from the model's, the class heads are re-created first.
        Extra kwargs are TrainConfig fields (epochs, batch, lr, amp, device, ...).
        Exports made afterwards use the trained weights; artifacts go to workdir/<run>.
        """
        import torch

        from optinpu.engine import TrainConfig, train

        train_set = _as_dataset(data, self.spec.imgsz, augment=True)
        val_set = _as_dataset(val, self.spec.imgsz, augment=False) if val is not None else None
        if train_set.class_names != self.class_names:
            print(f"New label space ({len(train_set.class_names)} classes): re-creating class heads")
            self.spec = dataclasses.replace(self.spec, class_names=train_set.class_names)
            self.model = self.spec.builder(True, class_names=train_set.class_names)

        self.workdir = self.workdir / run
        result = train(self.model, train_set, val_set, TrainConfig(**config), self.workdir, self.name)
        state = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
        self.model = self.model.cpu()
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self._ir, self._compiled = {}, {}
        return result

    def export(
        self,
        target: str = "npu",
        precision: str = "fp16",
        calibration=None,
        force: bool = False,
        **quantize_kwargs,
    ) -> Path:
        """Export to OpenVINO IR at the model's static input size. Returns the .xml path.

        precision: "fp32", "fp16" or "int8". For "int8", `calibration` is a list of images
        (paths, PIL images or arrays); by default 300 COCO val2017 images disjoint from the
        evaluation range are used. Extra kwargs go to `optinpu.quantize.quantize_int8`.
        An IR already on disk is reused unless `force=True`.
        `target` only selects defaults today; the same IR runs on every OpenVINO device.
        """
        if precision not in ("fp16", "fp32", "int8"):
            raise ValueError(f"precision must be fp32, fp16 or int8, got {precision!r}")
        if precision in self._ir and not force:
            return self._ir[precision]

        h, w = self.spec.imgsz
        path = self.workdir / f"{self.name}_{precision}_{h}x{w}.xml"
        if path.exists() and not force:
            self._ir[precision] = path
        elif precision == "int8":
            self._ir[precision] = self._quantize(path, calibration, force, **quantize_kwargs)
        else:
            static = copy.deepcopy(self.model)  # the training model keeps dynamic shapes
            if hasattr(static, "freeze_for_export"):
                static.freeze_for_export((h, w))
            self._ir[precision] = export_openvino(
                static, path, input_shape=(1, 3, h, w), fp16=precision == "fp16"
            )
        self._compiled = {k: v for k, v in self._compiled.items() if not k.endswith(f":{precision}")}
        return self._ir[precision]

    def _quantize(self, path: Path, calibration, force: bool, **kwargs) -> Path:
        from optinpu.quantize import quantize_int8

        if calibration is None:
            from optinpu.data import coco_val_subset

            calibration = coco_val_subset(300, offset=4500).image_paths()
        transform = lambda src: preprocess(load_image(src), self.spec.imgsz, self.spec.mean, self.spec.std)
        return quantize_int8(self.export(precision="fp32", force=force), calibration, transform, path, **kwargs)

    def _device_config(self, device: str) -> dict:
        return dict(self.spec.npu_config) if normalize_ov_device(device) == "NPU" else {}

    def compiled(self, device: str = "npu", precision: str = "fp16") -> CompiledModel | HybridModel:
        """Compiled model for `device`: "cpu", "gpu", "npu", "auto", or hybrid "npu+gpu" / "npu+cpu"
        (backbone and encoder on the first device, the rest on the second)."""
        parts = device.split("+")
        key = "+".join(normalize_ov_device(p) for p in parts) + f":{precision}"
        if key not in self._compiled:
            ir = self.export(precision=precision)
            if len(parts) == 1:
                self._compiled[key] = compile_model(ir, device, config=self._device_config(device))
            elif len(parts) == 2:
                if not self.spec.split:
                    raise ValueError(f"{self.name} defines no split point for hybrid devices")
                # npu_config targets the decoder, which is in the tail: the head keeps the defaults
                head, tail = parts
                self._compiled[key] = compile_hybrid(
                    ir, self.spec.split, head, tail, tail_config=self._device_config(tail),
                )
            else:
                raise ValueError(f"Invalid device {device!r}")
        return self._compiled[key]

    def predict(
        self,
        source,
        device: str = "npu",
        conf: float = 0.4,
        precision: str = "fp16",
    ) -> Detections:
        """Detect objects in one image (path, PIL image or HxWx3 uint8 array).

        `device`: "cpu", "gpu", "npu", hybrid "npu+gpu" / "npu+cpu", or "torch" (PyTorch model,
        reference path, no export).
        """
        image = load_image(source)
        x = preprocess(image, self.spec.imgsz, self.spec.mean, self.spec.std)
        if device == "torch":
            scores, boxes = self._torch_forward(x)
        else:
            scores, boxes = self.compiled(device, precision)(x)
        return decode(scores[0], boxes[0], image.size, self.class_names, conf=conf)

    def _torch_forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        with torch.no_grad():
            scores, boxes = self.model(torch.from_numpy(x))
        return scores.numpy(), boxes.numpy()

    def benchmark(self, devices=("cpu", "gpu", "npu"), precision: str = "fp16", **kwargs) -> list[BenchResult]:
        """Latency / throughput per device. Hybrid devices report sequential throughput."""
        ir = self.export(precision=precision)
        results = []
        for d in devices:
            if "+" in d:
                model = self.compiled(d, precision)
                inputs = [np.random.default_rng(0).random(s, dtype=np.float32) for s in model.input_shapes]
                results.append(benchmark_callable(model, inputs, model.device, kwargs.get("iterations", 100)))
            else:
                results.append(benchmark(ir, d, config=self._device_config(d), **kwargs))
        return results

    def lint(self, image=None, device: str = "npu", precision: str = "fp16", raw: bool = False,
             images=None, min_agreement: float = 0.95):
        """npu_lint report for the exported IR. `raw=True` ignores the model's known-good NPU config.

        With `images` (a list of image sources), accuracy is judged on the detections: F1 agreement
        between CPU FP32 and `device` (optinpu.lint.detection_agreement) must reach `min_agreement`;
        otherwise NPU_TILES=1 is tried and recommended when it fixes it. Without `images`, the raw
        output error on `image` (default: noise) is used, which is less reliable.
        """
        from optinpu.lint import LintIssue, detection_agreement, lint

        ir = self.export(precision=precision)
        inputs = None
        if image is not None:
            inputs = [preprocess(load_image(image), self.spec.imgsz, self.spec.mean, self.spec.std)]
        config = None if raw else self._device_config(device)
        report = lint(ir, device, inputs, config=config, check_accuracy=images is None)
        if images is None or not report.compiles:
            return report

        loaded = [load_image(src) for src in images]

        def detections(dev, cfg):
            model = compile_model(ir, dev, config=cfg, cache_dir=None)
            out = []
            for img in loaded:
                scores, boxes = model(preprocess(img, self.spec.imgsz, self.spec.mean, self.spec.std))
                out.append(decode(scores[0], boxes[0], img.size, self.class_names, conf=0.0))
            return out

        reference = detections("cpu", {"INFERENCE_PRECISION_HINT": "f32"})
        report.agreement = detection_agreement(reference, detections(device, config))
        if report.agreement < min_agreement:
            message = f"detections agree with CPU FP32 at F1 {report.agreement:.3f} (< {min_agreement})"
            if device.lower() == "npu" and not (config or {}).get("NPU_TILES"):
                one_tile = detection_agreement(reference, detections(device, {**(config or {}), "NPU_TILES": 1}))
                if one_tile >= min_agreement:
                    report.recommended_config = {"NPU_TILES": 1}
                    message += f"; with NPU_TILES=1: {one_tile:.3f}"
            report.issues.append(LintIssue("error", "ACCURACY", message))
        return report

    def benchmark_stages(self, device: str = "npu", precision: str = "fp16") -> dict[str, float]:
        """Real latency of cumulative stages (e.g. backbone, +encoder, full) in ms."""
        if not self.spec.stages:
            raise ValueError(f"{self.name} defines no stages")
        return benchmark_stages(self.export(precision=precision), self.spec.stages, device)

    def evaluate(self, data=None, device: str = "npu", precision: str = "fp16", num_images: int = 500):
        """COCO mAP on `data` (CocoSubset, CocoDetection or (json, image_dir)),
        by default the first `num_images` of COCO val2017 (downloaded on demand)."""
        from optinpu.data import coco_val_subset
        from optinpu.evaluate import evaluate_coco

        if data is None:
            subset = coco_val_subset(num_images)
        else:
            subset = _as_dataset(data, self.spec.imgsz, augment=False).to_subset()
        return evaluate_coco(self, subset, device, precision)


def _as_dataset(data, imgsz, augment: bool):
    from optinpu.data import CocoDetection, CocoSubset

    if isinstance(data, CocoDetection):
        return data
    if isinstance(data, CocoSubset):
        return CocoDetection(data.annotations, data.image_dir, imgsz, augment)
    annotations, image_dir = data
    return CocoDetection(annotations, image_dir, imgsz, augment)
