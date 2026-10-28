import pytest
import torch

from optinpu.models.optidet import OptiDet, OptiDetConfig
from optinpu.models.optidet_loss import TaskAlignedAssigner, giou, pairwise_iou

SMALL = OptiDetConfig(widths=(16, 16, 32, 32, 64), depths=(1, 1, 2, 2, 1), neck_width=32, head_width=32,
                      reg_max=8, num_classes=3, imgsz=(128, 128))


def test_iou_helpers():
    a = torch.tensor([[0.0, 0, 10, 10]])
    b = torch.tensor([[0.0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    torch.testing.assert_close(pairwise_iou(a, b), torch.tensor([[1.0, 25 / 175, 0.0]]))
    g = giou(a.expand(3, 4), b)
    assert g[0] == 1.0 and g[2] < 0  # disjoint boxes: negative GIoU


def test_assigner_topk_and_targets():
    anchors = torch.tensor([[x + 0.5, 5.5] for x in range(20)], dtype=torch.float32)  # a row of 20 points
    gt_boxes = torch.tensor([[[2.0, 0, 12, 11], [0, 0, 0, 0]]])  # one real box + padding
    gt_labels, gt_mask = torch.tensor([[1, -1]]), torch.tensor([[True, False]])
    scores = torch.full((1, 20, 3), 0.5)
    boxes = gt_boxes[:, :1].expand(1, 20, 4).clone()  # every prediction equals the box: IoU 1
    for k, expected in ((10, 10), (1, 1)):
        fg, labels, tboxes, tscores = TaskAlignedAssigner(k)(scores, boxes, anchors, gt_labels, gt_boxes, gt_mask)
        assert fg.sum() == expected
        assert torch.all((anchors[fg[0], 0] > 2) & (anchors[fg[0], 0] < 12))  # only anchors inside the box
        assert torch.all(labels[fg] == 1) and torch.allclose(tboxes[fg], gt_boxes[0, 0])
        assert tscores[fg].argmax(-1).eq(1).all() and torch.allclose(tscores[fg].amax(-1), torch.ones(expected))
        assert tscores[~fg].sum() == 0


def test_assigner_without_boxes():
    fg, _, _, tscores = TaskAlignedAssigner(10)(torch.rand(2, 20, 3), torch.rand(2, 20, 4), torch.rand(20, 2),
                                                torch.zeros(2, 0, dtype=torch.long), torch.zeros(2, 0, 4),
                                                torch.zeros(2, 0, dtype=torch.bool))
    assert not fg.any() and tscores.sum() == 0


class FakeTeacher:
    """Frozen random convs giving maps at the neck's strides (8, 16, 32) with 24 channels."""

    channels = [24, 24, 24]

    def __init__(self):
        torch.manual_seed(1)
        self.convs = [torch.nn.Conv2d(3, 24, 3, s, 1) for s in (8, 16, 32)]

    @torch.no_grad()
    def __call__(self, pixels):
        return [c(pixels) for c in self.convs]


def test_distillation_loss_and_checkpoint_roundtrip(tmp_path):
    from optinpu.models.distill import pkd_loss

    a = torch.randn(2, 4, 8, 8)
    assert float(pkd_loss([a], [3 * a + 1])) < 1e-4  # Pearson: invariant to scale and offset
    assert float(pkd_loss([a], [-a])) > 1.9

    torch.manual_seed(0)
    model = OptiDet(SMALL).train()
    model.enable_distillation(FakeTeacher(), weight=2.0)
    image = torch.rand(2, 3, 128, 128)
    targets = [{"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])}] * 2
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    first = None
    for _ in range(30):
        loss, parts = model.loss(image, targets)
        first = parts["distill"] if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert parts["distill"] < first  # the student's neck learns to correlate with the teacher
    assert any(k.startswith("distill.") for k in model.state_dict())
    assert not any("teacher" in k for k in model.state_dict())

    import copy

    deployed = copy.deepcopy(model).deploy()
    assert deployed.distill is None and not any(k.startswith("distill.") for k in deployed.state_dict())


def test_qat_fake_quantizers_act_in_the_loss_path():
    """NNCF attaches fake quantizers to forward(); loss() must go through forward(), otherwise QAT
    silently trains the float model (calling features() directly bypasses them)."""
    import copy

    nncf = pytest.importorskip("nncf")
    torch.manual_seed(0)
    model = OptiDet(SMALL).eval().fuse()
    q = nncf.quantize(copy.deepcopy(model), nncf.Dataset([torch.rand(1, 3, 128, 128) for _ in range(4)]), subset_size=4)
    x = torch.rand(1, 3, 128, 128)
    targets = [{"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])}]
    with torch.no_grad():
        float_loss, _ = model.loss(x, targets)
        quant_loss, _ = q.loss(x, targets)
    assert float(float_loss) != float(quant_loss)


def test_training_reduces_loss_and_learns_the_box():
    """Overfit one synthetic image: a bright square, class 2."""
    torch.manual_seed(0)
    model = OptiDet(SMALL).train()
    image = torch.zeros(1, 3, 128, 128)
    image[:, :, 32:96, 40:104] = 1.0
    target = [{"class_labels": torch.tensor([2]), "boxes": torch.tensor([[72 / 128, 64 / 128, 64 / 128, 64 / 128]])}]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(80):
        loss, parts = model.loss(image, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.3 * losses[0]
    model.eval()
    with torch.no_grad():
        scores, boxes = model(image)
    best = scores[0, :, 2].argmax()
    assert scores[0, best, 2] > 0.5
    assert pairwise_iou(boxes[0, best:best + 1] * 128, torch.tensor([[40.0, 32, 104, 96]]))[0, 0] > 0.6
