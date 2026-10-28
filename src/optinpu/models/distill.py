"""Feature distillation for OptiDet.

PKD (Cao et al., NeurIPS 2022): the student's feature maps imitate the teacher's after both are
standardized per channel over space (Pearson correlation), which works across architectures (here a
DETR teacher, D-FINE-S, and a dense convolutional student). 1x1 adapters map the student's widths to
the teacher's; they exist only during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DFineTeacher:
    """Frozen D-FINE (Hugging Face) as a feature teacher: hybrid-encoder outputs at strides 8, 16, 32.

    Takes the same input as OptiDet (pixels in [0, 1], no normalization). Not an nn.Module on purpose:
    it must stay out of the student's parameters, state dict and optimizer.
    """

    def __init__(self, repo: str = "ustc-community/dfine-small-coco"):
        from transformers import AutoModelForObjectDetection

        self.model = AutoModelForObjectDetection.from_pretrained(repo).model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.channels = [self.model.config.encoder_hidden_dim] * len(self.model.config.feat_strides)

    @torch.no_grad()
    def __call__(self, pixels: torch.Tensor) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device
        if device != pixels.device:
            self.model.to(pixels.device)
        m = self.model
        mask = torch.ones(pixels.shape[0], *pixels.shape[2:], device=pixels.device)
        proj = [m.encoder_input_proj[i](f) for i, (f, _) in enumerate(m.backbone(pixels, mask))]
        return list(m.encoder(proj).last_hidden_state)


def _standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    mean = x.mean((2, 3), keepdim=True)
    std = x.std((2, 3), keepdim=True)
    return (x - mean) / (std + eps)


def pkd_loss(student: list[torch.Tensor], teacher: list[torch.Tensor]) -> torch.Tensor:
    """Mean over levels of MSE between per-channel standardized maps, / 2 (= 1 - Pearson correlation)."""
    losses = [F.mse_loss(_standardize(s), _standardize(t.detach())) / 2 for s, t in zip(student, teacher)]
    return torch.stack(losses).mean()
