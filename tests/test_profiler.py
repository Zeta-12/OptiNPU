import numpy as np
import openvino as ov
import torch
from torch import nn

from optinpu.export import load_model
from optinpu.profiler import (
    bypass,
    bypass_types,
    module_regions,
    module_tokens,
    prefix,
    profile_measured,
    runtime_ops,
)


def test_module_tokens_from_both_frontends():
    assert module_tokens("__module.backbone.stages.1/aten::conv2d/Convolution") == ("backbone", "stages", "1")
    assert module_tokens("/model.8/m.0/cv2/conv/Conv/WithoutBiases") == ("model", "8", "m", "0", "cv2", "conv")
    assert module_tokens("/backbone/backbone.0/projector/stages.0/stages.0.0/m.0/cv2/act/Mul") == (
        "backbone", "0", "projector", "stages", "0", "0", "m", "0", "cv2", "act")
    assert module_tokens("Multiply_13759") is None
    assert module_tokens("__module.__nncf_hooks.post_hooks.pixels__0.0/prim::PythonOp/FakeQuantize") is None


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.conv(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 2, 1)
        self.blocks = nn.Sequential(Block(16), Block(16))
        self.head = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        return self.head(self.blocks(torch.relu(self.stem(x))))


def _net() -> ov.Model:
    torch.manual_seed(0)
    return load_model(Net(), (1, 3, 32, 32))


def _run(model: ov.Model, x: np.ndarray) -> list[np.ndarray]:
    return list(ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(x).values())


def test_regions_follow_modules():
    model = _net()
    names = [r.name for r in module_regions(model, 1)]
    assert names[:3] == ["stem", "blocks", "head"] or names[0] == "stem"
    blocks = {r.name: r for r in module_regions(model, 2)}
    assert "blocks.0" in blocks and blocks["blocks.0"].types["Convolution"] == 1


def test_bypassing_a_residual_block_is_identity_on_that_block():
    model, torch_model = _net(), None
    region = {r.name: r for r in module_regions(model, 2)}["blocks.1"]
    variant = bypass(model, region.ops)
    assert variant is not None and len(runtime_ops(variant)) < len(runtime_ops(model))
    # expected: the network without blocks.1
    torch.manual_seed(0)
    torch_model = Net().eval()
    x = np.random.default_rng(0).random((1, 3, 32, 32), dtype=np.float32)
    with torch.no_grad():
        t = torch.from_numpy(x)
        expected = torch_model.head(torch_model.blocks[0](torch.relu(torch_model.stem(t)))).numpy()
    np.testing.assert_allclose(_run(variant, x)[0], expected, rtol=1e-4, atol=1e-4)
    assert len(runtime_ops(model)) == len(runtime_ops(_net()))  # the original is untouched


def test_bypass_never_makes_a_loop():
    """Region {a, b} with an outside op c between them (a -> c -> b): a's output cannot be replaced by
    c's output, which depends on it."""
    from openvino import opset13 as ops

    x = ops.parameter([1, 4], np.float32, name="x")
    a = ops.relu(x)
    a.set_friendly_name("a")
    c = ops.sigmoid(a)
    c.set_friendly_name("c")
    b = ops.add(c, x)
    b.set_friendly_name("b")
    model = ov.Model([ops.multiply(a, b)], [x])
    variant = bypass(model, ["a", "b"])
    assert variant is None or len(variant.get_ordered_ops()) > 0  # building the model would raise on a loop


def test_bypass_refuses_shape_changing_regions():
    model = _net()
    stem = {r.name: r for r in module_regions(model, 1)}["stem"]
    assert bypass(model, stem.ops) is None


def test_bypass_types_and_prefix():
    model = _net()
    variant, done, total = bypass_types(model, {"Swish"})
    assert done == total == 2
    assert not any(op.get_type_name() == "Swish" for op in variant.get_ordered_ops())
    cut = max(i for i, op in enumerate(model.get_ordered_ops()) if "blocks.0" in op.get_friendly_name())
    head = prefix(model, cut)
    assert all(out.get_partial_shape().rank.get_length() == 0 for out in head.outputs)
    _run(head, np.zeros((1, 3, 32, 32), np.float32))


def test_weight_variants_change_values_only():
    from optinpu.profiler import weight_constants, with_weights

    model = _net()
    assert len(weight_constants(model)) == 4  # stem, 2 blocks, head
    x = np.random.default_rng(0).random((1, 3, 32, 32), dtype=np.float32)
    zeros = with_weights(model, "zeros")
    assert len(runtime_ops(zeros)) == len(runtime_ops(model))
    out = _run(zeros, x)[0]
    assert np.allclose(out, out.reshape(4, -1)[:, :1].reshape(1, 4, 1, 1))  # constant per channel: bias only
    sparse = with_weights(model, "sparse90")
    w = [c.get_data() for c in weight_constants(sparse)]
    assert 0.8 < np.mean([np.mean(a == 0) for a in w]) < 0.97
    assert not np.allclose(_run(with_weights(model, "random"), x)[0], _run(model, x)[0])


class TopKHead(nn.Module):
    """A dense head followed by a selection tail (as in end-to-end YOLO heads), plus a side output."""

    def __init__(self):
        super().__init__()
        self.box = nn.Conv2d(3, 4, 3, padding=1)
        self.cls = nn.Conv2d(3, 2, 3, padding=1)

    def forward(self, x):
        boxes = self.box(x).flatten(2).transpose(1, 2)  # (1, 256, 4)
        scores = self.cls(x).flatten(2).transpose(1, 2).sigmoid()
        best, idx = scores.amax(-1).topk(10, dim=1)
        return torch.gather(boxes, 1, idx[..., None].expand(-1, -1, 4)), best, boxes.mean()


def test_split_from_moves_the_selection_tail():
    from optinpu.graph import split_from

    torch.manual_seed(0)
    model = load_model(TopKHead(), (1, 3, 16, 16))
    topk = [op.get_friendly_name() for op in model.get_ordered_ops() if op.get_type_name() == "TopK"]
    head, tail = split_from(model, topk)
    assert not any(op.get_type_name() == "TopK" for op in head.get_ordered_ops())
    assert len(head.outputs) == len(tail.inputs)
    x = np.random.default_rng(0).random((1, 3, 16, 16), dtype=np.float32)
    middle = _run(head, x)
    out = list(ov.Core().compile_model(tail, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(middle).values())
    for a, b in zip(_run(model, x), out):  # same outputs, same order (the side output passes through)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_isolated_ops_build_runnable_single_op_graphs():
    from optinpu.profiler import isolated_ops

    model = _net()
    iso = isolated_ops(model)
    assert set(iso.signature) == {op.get_friendly_name() for op in runtime_ops(model)}
    # the two residual blocks are identical: their ops share signatures
    assert len(iso.graphs) < len(iso.signature)
    for graph in list(iso.graphs.values()) + list(iso.empties.values()):
        inputs = [np.zeros(tuple(p.get_partial_shape().to_shape()), np.float32) for p in graph.get_parameters()]
        out = list(ov.Core().compile_model(graph, "CPU")(inputs).values())
        assert all(o.shape == () for o in out)


def test_validation_interventions_on_a_small_model():
    from benchmarks.p1.validate import interventions

    found = interventions(_net())
    assert {"region:blocks.0", "region:blocks.1", "type:Swish", "rewrite:activations-to-relu"} <= set(found)
    ops_, variant = found["type:Swish"]
    assert len(ops_) == 2 and isinstance(variant, ov.Model)


def test_adaptive_rounds_give_paired_savings_with_confidence():
    from optinpu.profiler import Adaptive, measure, paired_saving

    model = _net()
    lighter = bypass_types(model, {"Swish"})[0]
    timings = measure({"full": model, "lighter": lighter, "same": model}, "cpu", rounds=3, iterations=5,
                      adaptive=Adaptive("full", target_ms=0.0, target_rel=0.0, max_rounds=8))
    assert len(timings["full"].rounds) == 8  # an unreachable target runs every allowed round
    saving, ci = paired_saving(timings, "full", "same")
    assert np.isfinite(saving) and 0 < ci < np.inf
    loose = measure({"full": model, "same": model}, "cpu", rounds=3, iterations=5,
                    adaptive=Adaptive("full", target_ms=1e3, max_rounds=8))
    assert len(loose["full"].rounds) == 3  # target met as soon as the minimum rounds are done


def test_profile_measured_runs_end_to_end_on_cpu():
    report = profile_measured(_net(), "cpu", depth=1, region_depth=2, rounds=2, iterations=5, batch=8,
                              log=lambda *a: None)
    assert report.latency_ms > 0
    assert [e.name for e in report.depth][0].startswith("(input")
    assert {e.name for e in report.regions} >= {"blocks.0", "blocks.1"}
    assert {e.name for e in report.op_types} >= {"Swish"}
    assert all(e.error is None for e in report.depth + report.regions + report.op_types)
    assert "measured profile on cpu" in str(report)
