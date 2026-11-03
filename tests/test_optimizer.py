"""Decision rules of the optimizer, on a fake device: each graph kind has a latency, an agreement with the
CPU FP32 original and an AP loss over the evaluation folds, so that every decision is deterministic."""

import numpy as np
import pytest

from optinpu import optimizer, sandbox
from optinpu.profiler import Timing
from optinpu.tasks import OutputTask, random_samples
from tests.test_rewrites import _mobile

FOLDS = 20
Z = np.array([1.0, -1.0] * (FOLDS // 2))  # fold pattern of the AP differences: mean 0, sd ~1


class FakeDevice:
    """kinds: "fp16", "int8", "int8-transformer", "mixed", "tiles=1" -> (ms, agreement, AP loss, its spread)."""

    def __init__(self, monkeypatch, kinds):
        self.kinds, self.tags, self.kept_float = kinds, {}, []
        monkeypatch.setattr(optimizer, "measure", self.measure)
        monkeypatch.setattr(optimizer, "int8_ptq", self.int8_ptq)
        monkeypatch.setattr(sandbox, "run_predictions", self.predictions)
        monkeypatch.setattr(sandbox, "run_evaluation", self.evaluation)

    def kind(self, parts):
        model, _, config = parts[0]
        return "tiles=1" if (config or {}).get("NPU_TILES") == 1 else self.tags.get(id(model), "fp16")

    def int8_ptq(self, graph, task, calibration, transformer=None, keep_float=None):
        clone = graph.clone()
        self.tags[id(clone)] = "mixed" if keep_float else "int8-transformer" if transformer else "int8"
        self.kept_float.append((sorted(keep_float or []), bool(transformer)))
        return clone

    def measure(self, variants, device="npu", config=None, rounds=5, iterations=30, adaptive=None, **_):
        timings = {}
        for name, variant in variants.items():
            ms = self.kinds[self.kind(variant if isinstance(variant, list) else [(variant, device, config)])][0]
            timings[name] = Timing(ms, ms, ms, rounds=[ms] * 3)
        return timings

    def predictions(self, parts, task, samples, **_):
        if parts[0][1] == "cpu":
            return [[np.ones(4, np.float32)] for _ in samples], None
        agreement = self.kinds[self.kind(parts)][1]  # OutputTask: 1 - relative L2 error
        return [[np.full(4, 2 - agreement, np.float32)] for _ in samples], None

    def evaluation(self, parts, task, subset, folds=10, **_):
        scores = 0.5 + 0.01 * np.arange(folds) / folds
        if parts[0][1] != "cpu":
            _, _, loss, spread = self.kinds[self.kind(parts)]
            scores = scores + loss + spread * Z[:folds]
        return {"score": float(scores.mean()), "fold_scores": [float(s) for s in scores]}, None


def _optimize(evaluation=True, log=None, model=None):
    model = model or _mobile()
    return optimizer.optimize(model, OutputTask(), random_samples(model, 2), "npu", evaluation=object() if evaluation else None,
                              folds=FOLDS, recommend=False, rounds=3, iterations=5, log=log or (lambda *a: None))


def test_int8_can_repair_an_inexact_original(monkeypatch):
    # RT-DETRv2: the original loses 2.4 AP on 2 tiles; INT8 (transformer scheme) repairs it and is faster
    FakeDevice(monkeypatch, {"fp16": (47.0, 0.72, -0.024, 0.002), "int8": (40.0, 0.65, -0.026, 0.002),
                             "int8-transformer": (40.5, 0.92, -0.001, 0.002), "mixed": (41.0, 0.93, -0.001, 0.001),
                             "tiles=1": (80.0, 0.995, 0.0, 0.001)})
    report = _optimize()
    assert report.final.changes == ["int8-transformer"] and report.final_ms == pytest.approx(40.5)


def test_mixed_precision_closes_a_near_miss_on_an_inexact_state(monkeypatch):
    # INT8 repairs the original but misses the budget by its confidence bound only (mean -0.3 point):
    # a mix may close the gap, so the search runs although the state is inexact
    FakeDevice(monkeypatch, {"fp16": (47.0, 0.72, -0.024, 0.002), "int8": (40.0, 0.65, -0.026, 0.002),
                             "int8-transformer": (40.5, 0.92, -0.003, 0.01), "mixed": (41.0, 0.93, -0.001, 0.001),
                             "tiles=1": (80.0, 0.995, 0.0, 0.001)})
    report = _optimize()
    assert report.final.changes[-1].startswith("int8-transformer-mixed") and report.final_ms == pytest.approx(41.0)


def test_the_metric_decides_exactness_when_annotations_exist(monkeypatch):
    # D-FINE-N: agreement 0.92 (< 0.95) but only 0.18 AP lost, within the budget
    kinds = {"fp16": (26.0, 0.92, -0.0018, 0.003), "int8": (26.5, 0.65, -0.058, 0.003),
             "int8-transformer": (36.0, 0.9, -0.01, 0.003), "mixed": (26.4, 0.9, -0.03, 0.003),
             "tiles=1": (43.0, 0.995, 0.0, 0.001)}
    FakeDevice(monkeypatch, kinds)
    lines = []
    report = _optimize(log=lambda *a: lines.append(" ".join(map(str, a))))
    assert report.final.changes == [] and not any("inexact" in line for line in lines)  # no repair to pay for
    FakeDevice(monkeypatch, kinds)
    assert _optimize(evaluation=False).final.changes == ["tiles=1"]  # without annotations, the agreement decides


def test_replay_rebuilds_the_optimized_state(monkeypatch):
    # the campaign kept only the applied changes: replaying them must give the same graph and placement
    device = FakeDevice(monkeypatch, {"fp16": (47.0, 0.72, -0.024, 0.002), "int8": (40.0, 0.65, -0.026, 0.002),
                                      "int8-transformer": (40.5, 0.92, -0.003, 0.01),
                                      "mixed": (41.0, 0.93, -0.001, 0.001), "tiles=1": (80.0, 0.995, 0.0, 0.001)})
    model = _mobile()
    final = _optimize(model=model).final
    used = device.kept_float[-1]
    state = optimizer.replay(model, final.changes, OutputTask(), random_samples(model, 2))
    assert state.label() == final.label() and device.kept_float[-1] == used and used[0] and used[1]
    assert (state.config, state.offload, state.offload_config) == (final.config, final.offload, final.offload_config)
    state = optimizer.replay(model, ["selection-to-cpu", "tiles=1"], OutputTask(), [])
    assert (state.config, state.offload, state.offload_config) == ({"NPU_TILES": 1}, "cpu", None)
    state = optimizer.replay(model, ["decoder-on-1-tile"], OutputTask(), [])
    assert (state.config, state.offload, state.offload_config) == (None, "npu", {"NPU_TILES": 1})
    with pytest.raises(ValueError):
        optimizer.replay(model, ["int8-mixed (float: no.such.group)"], OutputTask(), [])
