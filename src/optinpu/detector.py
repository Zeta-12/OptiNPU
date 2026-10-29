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

# Quantization-aware training defaults (TrainConfig fields): a short, low learning rate fine-tuning
# in FP32. On OptiDet-N (PLAN.md, M2.5) 2 epochs recover all the post-training INT8 loss (1.5 AP).
QAT_DEFAULTS = dict(epochs=2, lr=1e-4, backbone_lr=1e-4, weight_decay=0.0, warmup_steps=100, clip=10.0,
                    ema_decay=0.9998, amp=False)


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
        self._qat: dict | None = None  # NNCF quantizer config once the model is quantization-aware trained

    @classmethod
    def from_checkpoint(cls, name: str, checkpoint: str | Path, workdir: str | Path | None = None) -> "Detector":
        """Load weights saved by `train()` or `train_qat()` (best.pt)."""
        import torch

        checkpoint = Path(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        detector = cls(name, pretrained=False, class_names=state["class_names"],
                       workdir=workdir or checkpoint.parent.parent)
        if "qat" in state:
            detector._insert_fake_quantizers(config=state["qat"])
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

    @property
    def task(self):
        """The model's input / output conventions as an optinpu.tasks.DetectionTask (profiler, optimizer)."""
        from optinpu.tasks import DetectionTask

        return DetectionTask(self.spec.imgsz, "optinpu", "stretch", self.spec.mean, self.spec.std,
                             class_names=list(self.class_names))

    def train(self, data, val=None, run: str = "train", **config) -> TrainResult:
        """Fine-tune on a COCO-format dataset, then keep the best EMA weights.

        data / val: CocoDetection, CocoSubset, or (annotations_json, image_dir).
        If the dataset classes differ from the model's, the class heads are re-created first.
        Extra kwargs are TrainConfig fields (epochs, batch, lr, amp, device, ...).
        Exports made afterwards use the trained weights; artifacts go to workdir/<run>.
        """
        train_set, val_set = self._datasets(data, val)
        if train_set.class_names != self.class_names:
            print(f"New label space ({len(train_set.class_names)} classes): re-creating class heads")
            self.spec = dataclasses.replace(self.spec, class_names=train_set.class_names)
            self.model = self.spec.builder(True, class_names=train_set.class_names)
            self._qat = None
        return self._fit(train_set, val_set, run, config)

    def train_qat(self, data, val=None, run: str = "qat", calibration=None, **config) -> TrainResult:
        """Quantization-aware training: fine-tune a trained model through simulated INT8, so that its
        INT8 export loses no accuracy (OptiDet-N: -1.5 AP after post-training INT8, 0.0 after QAT).

        The model is fused into its inference form, NNCF inserts fake quantizers calibrated on
        `calibration` (image sources; default: 300 training images), then training resumes through
        them. Extra kwargs are TrainConfig fields and override QAT_DEFAULTS (2 epochs, lr 1e-4, FP32).
        Afterwards the model exports to INT8 only, straight from the trained quantizers (no
        post-training quantization), and "int8" becomes the default precision. The checkpoint
        (workdir/<run>/best.pt) reloads with from_checkpoint.
        Requires a model with fuse() whose loss goes through forward() (OptiDet; not the DETR models).
        """
        train_set, val_set = self._datasets(data, val)
        if train_set.class_names != self.class_names:
            raise ValueError("train_qat fine-tunes a trained model: the dataset must use its classes "
                             f"({len(self.class_names)}), got {len(train_set.class_names)}")
        if self._qat is None:
            if calibration is None:
                step = max(1, len(train_set) // 300)
                calibration = [train_set.image_dir / im["file_name"] for im in train_set.images[::step][:300]]
            self._insert_fake_quantizers(calibration=list(calibration))
        result = self._fit(train_set, val_set, run, {**QAT_DEFAULTS, **config})
        self.export(precision="int8", force=True)  # an IR left by an earlier run in workdir is stale
        return result

    def _datasets(self, data, val):
        train_set = _as_dataset(data, self.spec.imgsz, augment=True)
        val_set = _as_dataset(val, self.spec.imgsz, augment=False) if val is not None else None
        return train_set, val_set

    def _fit(self, train_set, val_set, run: str, config: dict) -> TrainResult:
        import torch

        from optinpu.engine import TrainConfig, train

        self.workdir = self.workdir / run
        extra = {"qat": self._qat} if self._qat is not None else None
        result = train(self.model, train_set, val_set, TrainConfig(**config), self.workdir, self.name,
                       checkpoint_extra=extra)
        state = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
        self.model = self.model.cpu()
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self._ir, self._compiled = {}, {}
        return result

    def _insert_fake_quantizers(self, calibration=None, config: dict | None = None) -> None:
        """Fuse the model and insert NNCF fake quantizers, calibrated on `calibration` (image sources)
        or restored from `config` (nncf.torch.get_config of a quantized model, saved in checkpoints)."""
        import nncf
        import nncf.torch
        import torch

        if not hasattr(self.model, "fuse"):
            raise ValueError(f"{self.name} does not support QAT: it needs fuse() and a loss that goes through "
                             "forward(), where NNCF attaches its fake quantizers")
        model = self.model.cpu().eval().fuse()
        if config is not None:
            self.model = nncf.torch.load_from_config(model, config)
        else:
            transform = lambda src: torch.from_numpy(
                preprocess(load_image(src), self.spec.imgsz, self.spec.mean, self.spec.std))
            self.model = nncf.quantize(model, nncf.Dataset(calibration, transform), subset_size=len(calibration))
            config = nncf.torch.get_config(self.model)
        self._qat = config
        self._ir, self._compiled = {}, {}

    def _precision(self, precision: str | None) -> str:
        """Default precision: "int8" for a quantization-aware trained model, "fp16" otherwise."""
        return precision or ("int8" if self._qat is not None else "fp16")

    def export(
        self,
        target: str = "npu",
        precision: str | None = None,
        calibration=None,
        force: bool = False,
        **quantize_kwargs,
    ) -> Path:
        """Export to OpenVINO IR at the model's static input size. Returns the .xml path.

        precision: "fp32", "fp16" or "int8" (default: "int8" after train_qat(), else "fp16").
        For "int8", `calibration` is a list of images (paths, PIL images or arrays); by default 300
        COCO val2017 images disjoint from the evaluation range are used. Extra kwargs go to
        `optinpu.quantize.quantize_int8`. After train_qat(), "int8" is the only precision and is
        exported from the trained fake quantizers (no calibration).
        An IR already on disk is reused unless `force=True`.
        `target` only selects defaults today; the same IR runs on every OpenVINO device.
        """
        precision = self._precision(precision)
        if precision not in ("fp16", "fp32", "int8"):
            raise ValueError(f"precision must be fp32, fp16 or int8, got {precision!r}")
        if self._qat is not None and precision != "int8":
            raise ValueError(f"{self.name} was quantization-aware trained: it exports to int8 only")
        if precision in self._ir and not force:
            return self._ir[precision]

        h, w = self.spec.imgsz
        path = self.workdir / f"{self.name}_{precision}_{h}x{w}.xml"
        if path.exists() and not force:
            self._ir[precision] = path
        elif precision == "int8" and self._qat is None:
            self._ir[precision] = self._quantize(path, calibration, force, **quantize_kwargs)
        else:  # float, or INT8 from QAT: the fake quantizers become FakeQuantize ops
            static = copy.deepcopy(self.model)  # the training model keeps dynamic shapes
            if hasattr(static, "freeze_for_export"):
                static.freeze_for_export((h, w))
            self._ir[precision] = export_openvino(
                static, path, input_shape=(1, 3, h, w), fp16=precision != "fp32"
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

    def compiled(self, device: str = "npu", precision: str | None = None) -> CompiledModel | HybridModel:
        """Compiled model for `device`: "cpu", "gpu", "npu", "auto", or hybrid "npu+gpu" / "npu+cpu"
        (backbone and encoder on the first device, the rest on the second)."""
        precision = self._precision(precision)
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
        precision: str | None = None,
    ) -> Detections:
        """Detect objects in one image (path, PIL image or HxWx3 uint8 array).

        `device`: "cpu", "gpu", "npu", hybrid "npu+gpu" / "npu+cpu", or "torch" (PyTorch model,
        reference path, no export; simulated INT8 after train_qat()).
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

    def benchmark(self, devices=("cpu", "gpu", "npu"), precision: str | None = None, **kwargs) -> list[BenchResult]:
        """Latency / throughput per device. Hybrid devices report sequential throughput."""
        precision = self._precision(precision)
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

    def lint(self, image=None, device: str = "npu", precision: str | None = None, raw: bool = False,
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

    def benchmark_stages(self, device: str = "npu", precision: str | None = None) -> dict[str, float]:
        """Real latency of cumulative stages (e.g. backbone, +encoder, full) in ms."""
        if not self.spec.stages:
            raise ValueError(f"{self.name} defines no stages")
        return benchmark_stages(self.export(precision=precision), self.spec.stages, device)

    def evaluate(self, data=None, device: str = "npu", precision: str | None = None, num_images: int = 500):
        """COCO mAP on `data` (CocoSubset, CocoDetection or (json, image_dir)),
        by default the first `num_images` of COCO val2017 (downloaded on demand)."""
        from optinpu.data import coco_val_subset
        from optinpu.evaluate import evaluate_coco

        precision = self._precision(precision)
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
