"""OptiDet training loss, written from the papers:

- Task-aligned assignment (TOOD, Feng et al. 2021): an anchor point inside a ground-truth box is
  ranked by s^alpha * IoU^beta (s = predicted score of the box's class); the top-k per box are
  positives, an anchor claimed by several boxes goes to the one it overlaps most.
- Consistent dual assignment (YOLOv10, Wang et al. 2024): the same metric with k = 10 for the
  one-to-many head and k = 1 for the one-to-one head, the only one kept at inference (no NMS).
- Quality-aware classification targets (normalized alignment metric), GIoU box loss and
  Distribution Focal Loss on the box-side bins (GFL, Li et al. 2020).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """IoU between boxes a (..., M, 4) and b (..., A, 4), xyxy -> (..., M, A)."""
    lt = torch.maximum(a[..., :, None, :2], b[..., None, :, :2])
    rb = torch.minimum(a[..., :, None, 2:], b[..., None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_a[..., :, None] + area_b[..., None, :] - inter + eps)


def giou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Generalized IoU of matching boxes (N, 4) xyxy -> (N,)."""
    lt, rb = torch.maximum(a[:, :2], b[:, :2]), torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1) + (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1) - inter
    hull = (torch.maximum(a[:, 2:], b[:, 2:]) - torch.minimum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    iou = inter / (union + eps)
    return iou - (hull - union) / (hull + eps)


class TaskAlignedAssigner:
    def __init__(self, topk: int, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        self.topk, self.alpha, self.beta, self.eps = topk, alpha, beta, eps

    @torch.no_grad()
    def __call__(self, scores, boxes, anchors, gt_labels, gt_boxes, gt_mask):
        """
        scores (B, A, C) probabilities, boxes (B, A, 4) predicted xyxy (pixels), anchors (A, 2) pixels,
        gt_labels (B, M) long, gt_boxes (B, M, 4) xyxy pixels, gt_mask (B, M) bool (padding = False).
        Returns fg (B, A) bool, target_labels (B, A), target_boxes (B, A, 4), target_scores (B, A, C).
        """
        b, a, c = scores.shape
        m = gt_boxes.shape[1]
        if m == 0:
            return (torch.zeros(b, a, dtype=torch.bool, device=scores.device), torch.zeros(b, a, dtype=torch.long, device=scores.device),
                    torch.zeros(b, a, 4, device=scores.device), torch.zeros_like(scores))

        x, y = anchors[:, 0], anchors[:, 1]
        inside = ((x[None, None] > gt_boxes[..., 0, None]) & (x[None, None] < gt_boxes[..., 2, None])
                  & (y[None, None] > gt_boxes[..., 1, None]) & (y[None, None] < gt_boxes[..., 3, None]))
        inside &= gt_mask[..., None]  # (B, M, A)

        ious = pairwise_iou(gt_boxes, boxes)  # (B, M, A)
        cls_scores = scores.gather(2, gt_labels.clamp(min=0)[:, None, :].expand(-1, a, -1)).transpose(1, 2)  # (B, M, A)
        metric = cls_scores.pow(self.alpha) * ious.pow(self.beta) * inside

        k = min(self.topk, a)
        top = metric.topk(k, dim=-1).indices
        pos = torch.zeros_like(inside).scatter_(-1, top, True) & inside & (metric > 0)

        # an anchor claimed by several boxes goes to the one it overlaps most
        multi = pos.sum(1, keepdim=True) > 1
        best = F.one_hot(ious.masked_fill(~pos, -1).argmax(1), m).transpose(1, 2).bool()  # (B, M, A)
        pos = torch.where(multi, best & pos, pos)

        fg = pos.any(1)  # (B, A)
        gt_index = pos.float().argmax(1)  # (B, A)
        target_labels = gt_labels.gather(1, gt_index)
        target_boxes = gt_boxes.gather(1, gt_index[..., None].expand(-1, -1, 4))

        # quality target: alignment metric normalized per box, scaled by that box's best IoU
        metric = metric * pos
        max_metric = metric.amax(-1, keepdim=True)
        max_iou = (ious * pos).amax(-1, keepdim=True)
        norm = (metric * max_iou / (max_metric + self.eps)).amax(1)  # (B, A)
        target_scores = F.one_hot(target_labels.clamp(min=0), c).float() * (norm * fg)[..., None]
        return fg, target_labels, target_boxes, target_scores


class OptiDetLoss:
    # weights in the ratio of the YOLOv8 recipe (cls 0.5, box 7.5, dfl 1.5); a first run with a 6x
    # lower box weight barely learned to localize (M2, 1.4 AP after 2 epochs)
    def __init__(self, model, cls_weight: float = 0.5, box_weight: float = 7.5, dfl_weight: float = 1.5,
                 topk_many: int = 10, topk_one: int = 1):
        self.model = model
        self.weights = {"cls": cls_weight, "box": box_weight, "dfl": dfl_weight}
        self.assigners = {"one2many": TaskAlignedAssigner(topk_many), "one2one": TaskAlignedAssigner(topk_one)}

    def targets(self, targets: list[dict], device):
        """List of {"class_labels" (N,), "boxes" (N, 4) normalized cxcywh} -> padded pixel xyxy tensors."""
        m = max((len(t["class_labels"]) for t in targets), default=0)
        b = len(targets)
        labels = torch.full((b, m), -1, dtype=torch.long, device=device)
        boxes = torch.zeros(b, m, 4, device=device)
        scale = self.model.scale
        for i, t in enumerate(targets):
            n = len(t["class_labels"])
            if n:
                cx, cy, w, h = (t["boxes"].to(device) * scale).unbind(-1)
                boxes[i, :n] = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
                labels[i, :n] = t["class_labels"].to(device)
        return labels, boxes, labels >= 0

    def dfl(self, reg_logits, target_boxes, anchors, strides):
        """Distribution focal loss for positives: reg_logits (N, 4, R), target ltrb in bins."""
        r = reg_logits.shape[-1]
        ltrb = torch.cat([anchors - target_boxes[:, :2], target_boxes[:, 2:] - anchors], -1) / strides
        t = ltrb.clamp(0, r - 1 - 0.01)
        left = t.floor().long()
        w_right = t - left
        logp = reg_logits.log_softmax(-1)
        loss = -(logp.gather(-1, left[..., None]).squeeze(-1) * (1 - w_right)
                 + logp.gather(-1, (left + 1)[..., None]).squeeze(-1) * w_right)
        return loss.mean(-1)  # (N,)

    def __call__(self, outputs: dict, targets: list[dict]):
        model = self.model
        device = model.anchors.device
        gt_labels, gt_boxes, gt_mask = self.targets(targets, device)
        total, parts = 0.0, {}
        for head, (cls_logits, reg_logits) in outputs.items():
            boxes = model.boxes(model.distances(reg_logits))  # (B, A, 4) pixels
            fg, _, target_boxes, target_scores = self.assigners[head](
                cls_logits.detach().sigmoid(), boxes.detach(), model.anchors, gt_labels, gt_boxes, gt_mask)
            # normalize by the number of positives (as DETR by the number of boxes): normalizing by the
            # sum of quality targets, tiny while predicted boxes are poor, blows the initial loss up
            norm = fg.sum().clamp(min=1).float()
            loss_cls = F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="sum") / norm
            if fg.any():
                weight = target_scores.sum(-1)[fg]
                loss_box = ((1 - giou(boxes[fg], target_boxes[fg])) * weight).sum() / norm
                anchors = model.anchors.expand(fg.shape[0], -1, -1)[fg]
                strides = model.anchor_strides.expand(fg.shape[0], -1, -1)[fg]
                loss_dfl = (self.dfl(reg_logits[fg], target_boxes[fg], anchors, strides) * weight).sum() / norm
            else:
                loss_box = loss_dfl = cls_logits.sum() * 0
            for name, value in (("cls", loss_cls), ("box", loss_box), ("dfl", loss_dfl)):
                total = total + self.weights[name] * value
                parts[f"{head}_{name}"] = value.detach()  # not float(): no device sync per term and step
        return total, parts
