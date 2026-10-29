import numpy as np
import openvino as ov
import torch
from torch import nn

from optinpu.export import load_model
from optinpu.rewrites import activations_to_relu, depthwise_to_dense, placements, tiles, without_norms


def _run(model: ov.Model, x: np.ndarray) -> list[np.ndarray]:
    return list(ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(x).values())


class Mobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, 1, 1)
        self.dw = nn.Conv2d(8, 16, 3, 2, 1, groups=8)  # depth multiplier 2, stride 2
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(1, 16)
        self.ln = nn.LayerNorm(16)

    def forward(self, x):
        y = self.act(self.dw(self.stem(x)))
        return self.ln(y.permute(0, 2, 3, 1))


def _mobile(seed=0) -> ov.Model:
    torch.manual_seed(seed)
    return load_model(Mobile(), (1, 3, 16, 16))


def test_depthwise_to_dense_is_exact():
    model = _mobile()
    dense, n = depthwise_to_dense(model)
    assert n == 1
    assert not any(op.get_type_name() == "GroupConvolution" for op in dense.get_ordered_ops())
    x = np.random.default_rng(0).random((1, 3, 16, 16), dtype=np.float32)
    np.testing.assert_allclose(_run(dense, x)[0], _run(model, x)[0], rtol=1e-4, atol=1e-5)


def test_retrain_tier_rewrites_keep_shapes():
    model = _mobile()
    relu, n = activations_to_relu(model)
    assert n == 1 and not any(op.get_type_name() == "Swish" for op in relu.get_ordered_ops())
    plain, m = without_norms(model)
    assert m == 1 and not any(op.get_type_name() == "MVN" for op in plain.get_ordered_ops())
    x = np.zeros((1, 3, 16, 16), np.float32)
    assert _run(relu, x)[0].shape == _run(plain, x)[0].shape == _run(model, x)[0].shape


def _optimized():
    from optinpu.optimizer import optimize
    from optinpu.tasks import OutputTask, random_samples

    model = _mobile()
    return optimize(model, OutputTask(), random_samples(model, 2), device="cpu", rounds=2, iterations=5,
                    log=lambda *a: None)


def test_optimize_loop_on_cpu():
    report = _optimized()
    assert report.baseline_agreement > 0.999 and report.final_agreement > 0.999
    # what gets accepted depends on CPU timing, so trials are matched by their last change
    last = {t.label.split(" + ")[-1] for t in report.trials}
    assert {"depthwise-to-dense", "int8", "int8-transformer"} <= last  # LayerNorm present: both INT8 schemes
    assert all(t.note.startswith("does not compile") for t in report.trials if t.label.endswith("tiles=1"))
    assert {t.label for t in report.recommendations} >= {"activations-to-relu", "layernorm-to-batchnorm", "prune-50"}
    assert "optimized for cpu" in str(report)


def test_report_saves_a_deployment_that_reloads(tmp_path):
    from optinpu.optimizer import load_deployment
    from optinpu.tasks import Deployed, OutputTask, random_samples

    report = _optimized()
    out = report.save(tmp_path / "opt", "tiny model")
    assert {"report.md", "report.json", "deployment.json", "part0.xml"} <= {p.name for p in out.iterdir()}
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert markdown.startswith("# tiny model") and "## Candidates tried" in markdown
    parts = load_deployment(out)
    assert parts[0][1] == "cpu" and parts[0][0].exists()
    sample = random_samples(_mobile(), 1)
    reloaded = Deployed(parts, OutputTask(), cache_dir=None).run(sample)
    original = Deployed(report.final.parts("cpu"), OutputTask(), cache_dir=None).run(sample)
    assert OutputTask().agreement(original, reloaded) > 0.99  # FP16 weights in the saved IRs


def test_choose_float_groups_by_ap_per_ms():
    from optinpu.rewrites import choose_float_groups

    gains = {"stem": (0.004, 0.1), "body": (0.006, 1.0), "head": (0.001, 0.05), "neck": (-0.001, 0.2),
             "broken": (float("nan"), 0.1)}
    assert choose_float_groups(gains, 0.003) == ["stem"]  # 0.04 AP/ms beats body (0.006) and head (0.02)
    assert choose_float_groups(gains, 0.005) == ["stem", "head"]
    assert choose_float_groups(gains, 1.0) == ["stem", "head", "body"]  # useless / failed groups never chosen
    assert choose_float_groups(gains, 0.0) == []


def test_mixed_precision_keeps_a_group_in_float():
    from optinpu.rewrites import int8_ptq, quantizable_groups
    from optinpu.tasks import OutputTask, random_samples

    model = _mobile()
    groups = quantizable_groups(model, 3)
    assert groups and all(names for names in groups.values())
    name, ops_ = next(iter(groups.items()))
    mixed = int8_ptq(model, OutputTask(), random_samples(model, 4), keep_float=ops_)
    full = int8_ptq(model, OutputTask(), random_samples(model, 4))
    count = lambda m: sum(op.get_type_name() == "FakeQuantize" for op in m.get_ordered_ops())
    assert count(mixed) < count(full)


def test_split_group_into_sub_modules():
    from optinpu.profiler import module_regions
    from optinpu.rewrites import split_group
    from tests.test_profiler import _net

    model = _net()
    blocks = {r.name: r for r in module_regions(model, 1)}["blocks"]
    subs = split_group(model, "blocks", blocks.ops)
    assert set(subs) == {"blocks.0", "blocks.1"} and sorted(sum(subs.values(), [])) == sorted(
        n for n in blocks.ops if any(n in s for s in subs.values()))
    head = {r.name: r for r in module_regions(model, 1)}["head"]
    assert split_group(model, "head", head.ops) == {}  # a single conv: nothing to split


def test_int8_ptq_quantizes_the_graph():
    from optinpu.rewrites import int8_ptq
    from optinpu.tasks import OutputTask, random_samples

    model = _mobile()
    quantized = int8_ptq(model, OutputTask(), random_samples(model, 4))
    assert any(op.get_type_name() == "FakeQuantize" for op in quantized.get_ordered_ops())
    assert not any(op.get_type_name() == "FakeQuantize" for op in model.get_ordered_ops())  # original untouched


def test_placement_candidates():
    from tests.test_profiler import TopKHead

    torch.manual_seed(0)
    model = load_model(TopKHead(), (1, 3, 16, 16))
    (candidate,) = placements(model)
    assert candidate.name == "selection-to-cpu" and candidate.kind == "exact"
    assert [device for _, device, _ in candidate.parts] == ["npu", "cpu"]
    assert placements(_mobile()) == []  # nothing to move without a TopK / NMS
    assert tiles(None)[0].name == "tiles=1" and tiles({"NPU_TILES": 1})[0].name == "tiles=2"
