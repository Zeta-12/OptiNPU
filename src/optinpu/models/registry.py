"""Registry of detectors.

Every registered model is wrapped so that its forward pass is pure tensors:
    pixels (B, 3, H, W) in [0, 1]  ->  scores (B, Q, C) in [0, 1], boxes (B, Q, 4) normalized xyxy
This contract keeps decoding identical across PyTorch, CPU, GPU and NPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from optinpu.models.coco import COCO_CLASSES


@dataclass
class ModelSpec:
    name: str
    # builder(pretrained, class_names=None): class_names other than the pretrained ones re-create the class heads
    builder: Callable[..., nn.Module]
    imgsz: tuple[int, int] = (640, 640)
    class_names: list[str] = field(default_factory=lambda: list(COCO_CLASSES))
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    source: str = ""
    npu_issue: str | None = None  # known NPU problem, shown as a warning at predict time
    npu_config: dict = field(default_factory=dict)  # NPU compile properties needed for correctness
    # cumulative stage label -> op-name regex, for bench.benchmark_stages
    stages: dict[str, str] = field(default_factory=dict)
    # op-name regex of the head for hybrid devices ("npu+gpu"): head on the first device, rest on the second
    split: str | None = None


HF_DETR_STAGES = {
    "backbone": r"\.backbone\.",
    "+encoder": r"\.backbone\.|encoder_input_proj|\.encoder\.",
}
HF_DETR_SPLIT = HF_DETR_STAGES["+encoder"]


class HFDetrWrapper(nn.Module):
    """Wraps a Hugging Face RT-DETR / D-FINE model into the (scores, boxes) contract."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def freeze_for_export(self, imgsz: tuple[int, int], anchor_clip: float = 1e4) -> None:
        """Replace input-independent computations by constants for a fixed image size.

        Sin-cos position embeddings and anchors only depend on the feature map sizes. Traced as-is,
        they are recomputed in the graph, in FP16 on the NPU, and invalid anchors are float32-max
        (= inf in FP16). Here they are captured once in FP32 and stored as buffers.
        """
        detr = self.model.model
        pos_modules = [m for m in self.modules() if type(m).__name__.endswith("SinePositionEmbedding")]
        captured: dict = {"pos": {}}

        def pos_hook(module, args, kwargs, output):
            captured["pos"][module] = output.detach().clone()

        hooks = [m.register_forward_hook(pos_hook, with_kwargs=True) for m in pos_modules]
        original_anchors = detr.generate_anchors

        def capture_anchors(*args, **kwargs):
            captured["anchors"] = tuple(t.detach().clone() for t in original_anchors(*args, **kwargs))
            return captured["anchors"]

        detr.generate_anchors = capture_anchors
        try:
            with torch.no_grad():
                self(torch.zeros(1, 3, *imgsz))
        finally:
            for h in hooks:
                h.remove()
            detr.generate_anchors = original_anchors

        for module, value in captured["pos"].items():
            module.register_buffer("frozen", value, persistent=False)
            module.forward = lambda *a, _m=module, **k: _m.frozen

        anchors, valid_mask = captured["anchors"]
        detr.register_buffer("frozen_anchors", anchors.clamp(-anchor_clip, anchor_clip), persistent=False)
        detr.register_buffer("frozen_valid_mask", valid_mask, persistent=False)
        detr.generate_anchors = lambda *a, **k: (detr.frozen_anchors, detr.frozen_valid_mask)

    def loss(self, pixels: torch.Tensor, targets: list[dict]) -> tuple[torch.Tensor, dict[str, float]]:
        """Training loss. targets: [{"class_labels": (N,), "boxes": (N, 4) normalized cxcywh}]."""
        labels = [{"class_labels": t["class_labels"], "boxes": t["boxes"]} for t in targets]
        out = self.model(pixel_values=pixels, labels=labels, return_dict=True)
        return out.loss, {k: float(v) for k, v in out.loss_dict.items()}

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pixel_values=pixels, return_dict=True)
        scores = out.logits.sigmoid()
        cx, cy, w, h = out.pred_boxes.unbind(-1)
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        return scores, boxes


def _hf_detr(repo: str) -> Callable[..., nn.Module]:
    def build(pretrained: bool, class_names: list[str] | None = None) -> nn.Module:
        from transformers import AutoConfig, AutoModelForObjectDetection

        kwargs = {}
        if class_names is not None:
            kwargs = dict(
                id2label=dict(enumerate(class_names)),
                label2id={n: i for i, n in enumerate(class_names)},
            )
        if pretrained:
            hf = AutoModelForObjectDetection.from_pretrained(repo, ignore_mismatched_sizes=bool(kwargs), **kwargs)
        else:
            hf = AutoModelForObjectDetection.from_config(AutoConfig.from_pretrained(repo, **kwargs))
        return HFDetrWrapper(hf)

    return build


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    _REGISTRY[spec.name] = spec


# The RT-DETRv2 decoder self-attention is wrong when the NPU compiler splits it over 2 tiles
# (OpenVINO 2026.4, driver 32.0.100.4778). One tile is exact but slower: prefer device="npu+gpu".
_ONE_TILE = {"NPU_TILES": 1}

for _name, _repo, _npu_config in (
    ("dfine-n", "ustc-community/dfine-nano-coco", {}),
    ("dfine-s", "ustc-community/dfine-small-coco", {}),
    ("rtdetrv2-r18", "PekingU/rtdetr_v2_r18vd", _ONE_TILE),
):
    register(ModelSpec(
        name=_name, builder=_hf_detr(_repo), source=f"hf:{_repo}", npu_config=_npu_config,
        stages=HF_DETR_STAGES, split=HF_DETR_SPLIT,
    ))


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def get_model(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(list_models())}") from None
