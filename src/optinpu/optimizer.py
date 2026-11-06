"""Measured optimizer: apply the rewrites that cut latency on the device without hurting accuracy.

Greedy loop over a deployment state (graph, compile options, placement of the selection tail):
1. the reference is the original model on the CPU in FP32; the state's accuracy is the task agreement
   with it on the given samples (detections F1 for detectors, calibrated in PLAN.md M2.1);
2. every candidate derived from the state is timed side by side with it (one sandboxed process); its
   saving is paired per round with the state, repeated until its 95 % confidence interval is narrow;
3. candidates are checked for accuracy, largest saving first (sandboxed: a lost NPU is a rejection);
   the first that keeps the agreement above `min_agreement` and saves at least 1 % of the latency,
   above zero with 95 % confidence, is applied;
4. repeat until nothing helps. A state that is inexact on the device (e.g. YOLO26-N on 2 NPU tiles,
   or YOLO26-S losing the NPU) accepts any candidate that repairs it, even a slower one: correctness
   first. With an annotated `evaluation` set, "exact" means that the task metric (AP, top-1) stays
   within `ap_budget` of the CPU FP32 original; the agreement only decides without annotations.
Approximate candidates (INT8 post-training quantization) change the function on purpose: with an
annotated `evaluation` set, they are judged by their AP against the CPU FP32 reference (`ap_budget`);
without one, by the same agreement threshold as exact candidates. They can repair an inexact state
(INT8 removes the 2-tile defect of RT-DETRv2); on an inexact state, mixed precision is only searched
when full INT8 already keeps the mean AP within the budget and misses only its confidence bound.
The network stays on the device: only a light selection tail (TopK / NMS and the gathers after it, no
convolution or matmul) may move to the CPU, as host post-processing. Heavier parts (a DETR decoder,
everything after an encoder) move only to the devices listed in `offload`, for comparison: the NPU is
worth using because it leaves the CPU and iGPU free at low power (PLAN.md P3.4).
Retrain-tier rewrites are then timed on the final state and reported as recommendations.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov

from optinpu.devices import core
from optinpu.graph import split_from, split_model
from optinpu.profiler import Adaptive, measure, paired_saving
from optinpu.rewrites import (
    choose_float_groups,
    depthwise_to_dense,
    graph_rewrites,
    int8_ptq,
    quantizable_groups,
    selection_tail,
    split_group,
)


@dataclass
class State:
    """A deployment of the model: graph, compile options, and where the selection tail runs."""

    model: ov.Model
    config: dict | None = None
    offload: str | None = None  # device of the tail compiled apart, or None
    # tail = the first TopK / NMS and what depends on it (None), or every op from this op on, in
    # topological order (a module boundary, e.g. the end of a DETR encoder)
    offload_from: str | None = None
    offload_config: dict | None = None  # compile options of the tail (a DETR decoder on 1 NPU tile)
    changes: list[str] = field(default_factory=list)

    def parts(self, device: str = "npu") -> list:
        if self.offload is None:
            return [(self.model, device, self.config)]
        if self.offload_from is None:
            roots, _, _ = selection_tail(self.model)
            head, tail = split_from(self.model, roots)
        else:
            index = next(i for i, op in enumerate(self.model.get_ordered_ops())
                         if op.get_friendly_name() == self.offload_from)
            head, tail = split_model(self.model, index - 1)
        return [(head, device, self.config), (tail, self.offload, self.offload_config)]

    def label(self) -> str:
        return " + ".join(self.changes) or "original"

    def save(self, directory: str | Path, device: str = "npu") -> Path:
        """Write the deployment: one IR per part (part0.xml on the device, part1.xml for an offloaded
        tail) and deployment.json (devices, compile options, changes). Load it with load_deployment."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        parts = []
        for i, (model, part_device, config) in enumerate(self.parts(device)):
            ov.save_model(model, directory / f"part{i}.xml", compress_to_fp16=True)
            parts.append({"ir": f"part{i}.xml", "device": part_device, "config": config})
        (directory / "deployment.json").write_text(json.dumps({"changes": self.changes, "parts": parts}, indent=1))
        return directory


def load_deployment(directory: str | Path) -> list[tuple[Path, str, dict | None]]:
    """Parts [(IR path, device, config)] of a deployment written by OptimizationReport.save: pass them to
    optinpu.tasks.Deployed (predictions) or optinpu.profiler.measure (latency)."""
    directory = Path(directory)
    spec = json.loads((directory / "deployment.json").read_text())
    return [(directory / p["ir"], p["device"], p["config"]) for p in spec["parts"]]


def replay(model, changes: list[str], task, calibration: list, device: str = "npu",
           config: dict | None = None) -> State:
    """The state reached by applying recorded `changes` (State.changes, "applied" in report.json) to `model`
    again, without measuring anything: e.g. to save a deployment that was not saved. INT8 must use the
    calibration samples of the optimization (NNCF quantizes them in order: same samples, same graph).
    "int8-mixed" is replayed with the CNN scheme: older reports also labelled transformer-scheme mixes so."""
    if not isinstance(model, ov.Model):
        model = core().read_model(model)
    state = State(model, config)
    for change in changes:
        if change in ("int8", "int8-transformer"):
            state = dataclasses.replace(state, model=int8_ptq(state.model, task, calibration, change == "int8-transformer"))
        elif change.startswith(("int8-mixed (float: ", "int8-transformer-mixed (float: ")):
            chosen = change[change.index("(float: ") + len("(float: "):-1].split(", ")
            groups = _float_groups(state.model, chosen)
            names = [n for g in chosen for n in groups[g]]
            transformer = change.startswith("int8-transformer")
            state = dataclasses.replace(state, model=int8_ptq(state.model, task, calibration, transformer, names))
        elif change.startswith("tiles="):
            state = dataclasses.replace(state, config={**(state.config or {}), "NPU_TILES": int(change[len("tiles="):])})
        elif change == "decoder-on-1-tile":
            state = dataclasses.replace(state, offload=device, offload_config={"NPU_TILES": 1})
        elif change == "depthwise-to-dense":
            state = dataclasses.replace(state, model=depthwise_to_dense(state.model)[0])
        elif change.startswith("from-") and "-to-" in change:  # a heavy tail cut at a module boundary
            region, target = change[len("from-"):].rsplit("-to-", 1)
            roots, _, _ = selection_tail(state.model)
            first = dict(_boundaries(state.model, roots[0]))[region]
            state = dataclasses.replace(state, offload=target, offload_from=first)
        elif change.split("-to-")[0] in ("selection", "decoder"):
            state = dataclasses.replace(state, offload=change.split("-to-")[1])
        else:
            raise ValueError(f"cannot replay the change {change!r}")
        state = dataclasses.replace(state, changes=[*state.changes, change])
    return state


def _float_groups(model: ov.Model, chosen: list[str]) -> dict[str, list[str]]:
    """Module groups of a mixed-precision search, with the sub-groups it split (up to three times)."""
    groups = quantizable_groups(model)
    for _ in range(3):
        missing = [g for g in chosen if g not in groups]
        parents = {p for m in missing for p in groups if p != "(rest)" and m.startswith(p + ".")}
        if not parents:
            break
        for p in parents:
            groups.update(split_group(model, p, groups.pop(p)))
    missing = [g for g in chosen if g not in groups]
    if missing:
        raise ValueError(f"unknown module groups {missing}")
    return groups


@dataclass
class Trial:
    label: str
    kind: str
    latency_ms: float
    agreement: float | None = None
    accepted: bool = False
    note: str = ""
    ap: float | None = None  # task metric (AP, top-1) on the evaluation set, judged candidates only
    ci_ms: float | None = None  # 95 % half-width of the paired saving behind latency_ms


@dataclass
class OptimizationReport:
    device: str
    baseline_ms: float
    baseline_agreement: float
    final: State
    final_ms: float
    final_agreement: float
    trials: list[Trial] = field(default_factory=list)
    recommendations: list[Trial] = field(default_factory=list)
    reference_ap: float | None = None  # CPU FP32 original, on the evaluation set
    baseline_ap: float | None = None  # original on the device
    final_ap: float | None = None
    metric: str = "AP"  # what the *_ap fields measure (the task's metric: AP, top-1)

    def __str__(self) -> str:
        speedup = self.baseline_ms / self.final_ms if self.final_ms else float("nan")
        lines = [f"optimized for {self.device}: {self.baseline_ms:.2f} ms -> {self.final_ms:.2f} ms ({speedup:.2f}x), "
                 f"agreement with CPU FP32 {self.baseline_agreement:.3f} -> {self.final_agreement:.3f}"]
        if self.reference_ap is not None:
            fmt = lambda ap: "failed" if ap is None else f"{ap:.3f}"
            lines.append(f"{self.metric} on the evaluation set: CPU FP32 {fmt(self.reference_ap)}, original on "
                         f"{self.device} {fmt(self.baseline_ap)}, optimized {fmt(self.final_ap)}")
        lines += [f"applied: {self.final.label()}", "", "trials:"]
        for t in self.trials:
            agreement = "  -  " if t.agreement is None else f"{t.agreement:.3f}"
            ap = "" if t.ap is None else f"  {self.metric} {t.ap:.3f}"
            mark = "ACCEPTED" if t.accepted else ""
            lines.append(f"  {t.label[:56]:<56} {t.latency_ms:7.2f} ms  agreement {agreement}{ap}  {mark} {t.note}")
        if self.recommendations:
            lines += ["", "with retraining (latency measured, accuracy to recover by fine-tuning):"]
            for t in self.recommendations:
                lines.append(f"  {t.label[:56]:<56} {t.latency_ms:7.2f} ms ({100 * (t.latency_ms / self.final_ms - 1):+.1f} %)"
                             f"  {t.note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"device": self.device, "applied": self.final.changes, "baseline_ms": self.baseline_ms,
                "final_ms": self.final_ms, "speedup": self.baseline_ms / self.final_ms,
                "baseline_agreement": self.baseline_agreement, "final_agreement": self.final_agreement,
                "metric": self.metric,
                "ap": {"cpu_fp32": self.reference_ap, "baseline": self.baseline_ap, "final": self.final_ap},
                "trials": [dataclasses.asdict(t) for t in self.trials],
                "recommendations": [dataclasses.asdict(t) for t in self.recommendations]}

    def to_markdown(self, title: str = "Optimization report") -> str:
        fmt = lambda v, spec=".3f": "—" if v is None else format(v, spec)
        speedup = self.baseline_ms / self.final_ms
        md = [f"# {title}", "",
              f"**{self.baseline_ms:.2f} ms → {self.final_ms:.2f} ms on the {self.device.upper()} ({speedup:.2f}×)**, "
              f"applied: {', '.join(f'`{c}`' for c in self.final.changes) or 'nothing'}.", "",
              f"| | Latency | Agreement with CPU FP32 | {self.metric} (evaluation set) |", "|---|---|---|---|",
              f"| CPU FP32 reference | — | 1.000 | {fmt(self.reference_ap)} |",
              f"| Original on the {self.device.upper()} | {self.baseline_ms:.2f} ms | {self.baseline_agreement:.3f} | "
              f"{fmt(self.baseline_ap)} |",
              f"| **Optimized** | **{self.final_ms:.2f} ms** | **{self.final_agreement:.3f}** | **{fmt(self.final_ap)}** |",
              "", "## Candidates tried", "",
              f"| Candidate | Kind | Latency | Agreement | {self.metric} | Decision |", "|---|---|---|---|---|---|"]
        for t in self.trials:
            decision = "**accepted**" if t.accepted else t.note.replace("|", "/")
            latency = "—" if not np.isfinite(t.latency_ms) else f"{t.latency_ms:.2f} ms"
            md.append(f"| {t.label} | {t.kind} | {latency} | {fmt(t.agreement)} | {fmt(t.ap)} | {decision} |")
        if self.recommendations:
            md += ["", "## With retraining", "",
                   "Latency measured on the optimized deployment; accuracy must be recovered by fine-tuning.", "",
                   "| Change | Latency | vs optimized | Details |", "|---|---|---|---|"]
            for t in self.recommendations:
                md.append(f"| {t.label} | {t.latency_ms:.2f} ms | {100 * (t.latency_ms / self.final_ms - 1):+.1f} % | "
                          f"{t.note.replace('|', '/')} |")
        md += ["", "Latencies are medians of paired, randomized rounds (95 % confidence intervals in the details); "
                   "agreement is the task's agreement with the CPU FP32 original on the guard samples."]
        return "\n".join(md) + "\n"

    def save(self, directory: str | Path, title: str = "Optimization report") -> Path:
        """report.md, report.json and the optimized deployment (deployment.json, part*.xml) in `directory`."""
        directory = Path(directory)
        self.final.save(directory, self.device)
        (directory / "report.md").write_text(self.to_markdown(title), encoding="utf-8")
        (directory / "report.json").write_text(json.dumps(self.to_dict(), indent=1, default=str))
        return directory


def ap_within_budget(folds: list[float], reference: list[float], budget: float) -> tuple[bool, float, float]:
    """Non-inferiority test of a candidate's AP against the reference on the same folds of images: (loss
    within `budget` at the 5 % level, mean AP difference, margin of its one-sided 95 % lower bound).
    One-sided paired t test of H0 "the candidate loses more than `budget`", rejected when
    mean - t(0.95, n - 1) x sd / sqrt(n) >= -budget. On 200 images a point estimate was not enough: a mix
    within 0.5 AP there lost 1 AP on other images (PLAN.md P3.4)."""
    from scipy.stats import t

    d = np.asarray(folds, dtype=float) - np.asarray(reference, dtype=float)
    if len(d) < 2 or not np.all(np.isfinite(d)):
        return False, float("nan"), float("nan")
    margin = float(t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean()) - margin >= -budget, float(d.mean()), margin


TRANSFORMER_OPS = {"Softmax", "MVN", "ScaledDotProductAttention"}


def _boundaries(model: ov.Model, selection: str, count: int = 3) -> list[tuple[str, str]]:
    """(module region, its first op) for the last `count` top-level module boundaries before the
    `selection` op, past the first 40 % of the graph (the backbone stays on the NPU)."""
    from optinpu.profiler import module_regions, pick_depth

    index = {op.get_friendly_name(): i for i, op in enumerate(model.get_ordered_ops())}
    limit = index[selection]
    found = []
    for region in module_regions(model, pick_depth(model, 8)):
        first = min(region.ops, key=index.__getitem__)
        if 0.4 * len(index) < index[first] < limit:
            found.append((index[first], region.name, first))
    return [(name, first) for _, name, first in sorted(found)[-count:]]


def _neighbours(state: State, int8=None, offload: tuple[str, ...] = (),
                device: str = "npu") -> list[tuple[State, str, str]]:
    """(state, kind, description) one change away from `state`. int8: function (graph, transformer) ->
    INT8 graph (None: no quantization candidate). Both NNCF schemes are tried on graphs with attention or
    LayerNorm: the measurements decide (on YOLO26 the transformer scheme makes a FakeQuantize the NPU
    compiler rejects, PLAN.md P3.1). offload: devices that may run a heavy tail (none: a light selection
    tail goes to the CPU, nothing else leaves the device)."""
    out = []
    if int8 is not None and not any(c.startswith("int8") for c in state.changes):
        kinds = {op.get_type_name() for op in state.model.get_ordered_ops()}
        for transformer in ((False, True) if kinds & TRANSFORMER_OPS else (False,)):
            name = "int8-transformer" if transformer else "int8"
            scheme = "transformer scheme" if transformer else "CNN scheme"
            out.append((dataclasses.replace(state, model=int8(state.model, transformer),
                                            changes=[*state.changes, name]),
                        "approx", f"INT8 post-training quantization (NNCF, {scheme})"))
    if device == "npu":  # a compile option of the NPU plugin only
        tiles = 2 if (state.config or {}).get("NPU_TILES") == 1 else 1
        out.append((dataclasses.replace(state, config={**(state.config or {}), "NPU_TILES": tiles},
                                        changes=[*state.changes, f"tiles={tiles}"]), "exact", f"NPU_TILES={tiles}"))
    if state.offload is None:
        found = selection_tail(state.model)
        if found is not None:
            roots, n_ops, heavy = found
            for target in (["cpu"] if heavy == 0 else offload):
                what = "selection" if heavy == 0 else "decoder"
                out.append((dataclasses.replace(state, offload=target, changes=[*state.changes, f"{what}-to-{target}"]),
                            "exact", f"{n_ops} ops after the first TopK / NMS on the {target.upper()}"))
            # DETR: also cut at the module boundaries before the query selection (RT-DETRv2 in L4)
            for target in (offload if heavy else ()):
                for region, first in _boundaries(state.model, roots[0]):
                    out.append((dataclasses.replace(state, offload=target, offload_from=first,
                                                    changes=[*state.changes, f"from-{region}-to-{target}"]),
                                "exact", f"everything from {region} on (topological order) on the {target.upper()}"))
            if heavy and device == "npu" and (state.config or {}).get("NPU_TILES") != 1:
                # the 2-tile defect sits in the decoder's self-attention (RT-DETRv2, L4): only the decoder on
                # 1 tile, the backbone and encoder keep 2; nothing leaves the NPU
                out.append((dataclasses.replace(state, offload=device, offload_config={"NPU_TILES": 1},
                                                changes=[*state.changes, "decoder-on-1-tile"]),
                            "exact", f"{n_ops} ops after the first TopK / NMS compiled apart on 1 NPU tile"))
    dense, n = depthwise_to_dense(state.model)
    if n:
        out.append((dataclasses.replace(state, model=dense, changes=[*state.changes, "depthwise-to-dense"]),
                    "exact", f"{n} depthwise convolutions as dense"))
    return out


def optimize(
    model,
    task,
    samples: list,
    device: str = "npu",
    config: dict | None = None,
    min_agreement: float = 0.95,
    calibration: list | None = None,
    evaluation=None,
    ap_budget: float = 0.005,
    folds: int = 10,
    int8: bool = True,
    sensitivity_samples: int = 64,
    offload: tuple[str, ...] = (),
    max_steps: int = 5,
    recommend: bool = True,
    rounds: int = 5,
    iterations: int = 30,
    confirmation=None,
    confirmation_folds: int | None = None,
    log=print,
) -> OptimizationReport:
    """Optimize `model` (ov.Model or IR path) for `device`; see the module docstring.

    samples: image sources (detection, classification) or lists of input arrays (OutputTask), 30 or more.
    calibration: samples for INT8 quantization (default: `samples`; ideally 300, disjoint from them).
    evaluation: an annotated optinpu.data.CocoSubset (detectors) or optinpu.data.LabeledImages
    (classifiers): approximate candidates must keep the task metric (AP, top-1) within `ap_budget`
    (0.005 = 0.5 point) of the CPU FP32 original on it, and the original or a repair must stay within it
    to count as exact. Use images that are not those you report results on. folds: interleaved folds of
    it for the paired test (about 50 images each: 500 images left a ~0.4-point margin, PLAN.md P3.5).
    int8: try INT8 post-training quantization; sensitivity_samples: calibration samples used to rank the
    module groups of a mixed precision search (the final mix uses all of `calibration`).
    offload: devices ("gpu", "cpu") that may run more than a light selection tail, e.g. a DETR decoder;
    default none: the network runs on `device` and only host post-processing leaves it.
    confirmation: another annotated set, disjoint from `evaluation` (in `confirmation_folds` folds, default
    `folds`): an approximation that passes the test on `evaluation` (full INT8, or the first mix of a mixed
    precision search) must pass it once more on these images. Keeping the first of several noisy tests that
    passes favours an underestimated loss (ResNet-50's mix: no loss on the decision images, -0.64 points on
    the report ones, PLAN.md P5); one more test on fresh images is not biased by that selection.
    """
    from optinpu.sandbox import run_evaluation, run_predictions

    if not isinstance(model, ov.Model):
        model = core().read_model(model)
    reference, job = run_predictions([(model, "cpu", None)], task, samples)
    if reference is None:
        raise RuntimeError(f"reference run on the CPU failed: {job.error}")

    def accuracy(state: State) -> tuple[float | None, str]:
        predictions, job = run_predictions(state.parts(device), task, samples)
        if predictions is None:
            return None, ("NPU lost" if job.device_lost else "failed") + f": {(job.error or '')[-120:]}"
        return task.agreement(reference, predictions), ""

    metric = getattr(task, "metric", "AP")  # AP (detection), top-1 (classification)

    def ap_of(parts) -> tuple[float | None, list[float] | None, str]:
        """(metric, metric per fold, error) on the evaluation set."""
        result, job = run_evaluation(parts, task, evaluation, folds=folds)
        if result is None:
            return None, None, ("NPU lost" if job.device_lost else "failed") + f": {(job.error or '')[-120:]}"
        return result["score"], result["fold_scores"], ""

    reference_folds: list[float] = []
    confirmation_reference: list[float] = []

    def confirm(parts) -> tuple[bool, str]:
        """(the loss stays within the budget on the confirmation images too, verdict); always true without
        a confirmation set."""
        if confirmation is None:
            return True, ""
        n = confirmation_folds or folds
        if not confirmation_reference:
            result, job = run_evaluation([(model, "cpu", None)], task, confirmation, folds=n)
            if result is None:
                raise RuntimeError(f"evaluation of the original on the confirmation set failed: {job.error}")
            confirmation_reference[:] = result["fold_scores"]
        result, job = run_evaluation(parts, task, confirmation, folds=n)
        if result is None:
            return False, " (confirmation failed: " + ("NPU lost" if job.device_lost else (job.error or "")[-100:]) + ")"
        ok, mean, margin = ap_within_budget(result["fold_scores"], confirmation_reference, ap_budget)
        word = "confirmed" if ok else "NOT confirmed"
        return ok, (f" ({word} on held-out images: {metric} {100 * mean:+.2f} points, 95 % lower bound "
                    f"{100 * (mean - margin):+.2f})")

    def judge(parts) -> tuple[bool, float | None, str]:
        """(metric loss vs the CPU FP32 original within the budget with 95 % confidence, metric, verdict)."""
        ap, scores, error = ap_of(parts)
        if ap is None:
            return False, None, error
        ok, mean, margin = ap_within_budget(scores, reference_folds, ap_budget)
        return ok, ap, (f"{metric} {100 * mean:+.2f} points vs CPU, 95 % lower bound {100 * (mean - margin):+.2f}, "
                        f"budget -{100 * ap_budget:.1f}")

    quantized: dict[tuple[int, bool], ov.Model] = {}  # (graph id, scheme) -> INT8 graph (quantized once)

    def quantize(graph: ov.Model, transformer: bool) -> ov.Model:
        key = (id(graph), transformer)
        if key not in quantized:
            log(f"quantizing to INT8 ({'transformer' if transformer else 'CNN'} scheme, "
                f"{len(calibration or samples)} calibration samples)")
            quantized[key] = int8_ptq(graph, task, calibration or samples, transformer)
        return quantized[key]

    state = State(model, config)
    current_ms = measure({"current": state.parts(device)}, device, rounds=rounds, iterations=iterations)["current"].median_ms
    current_agreement, note = accuracy(state)
    if current_agreement is None:
        # e.g. YOLO26-S on 2 NPU tiles loses the device on real images: an inexact state to repair
        log(f"the original model fails on {device} ({note}): looking for a placement that works")
        current_agreement = 0.0
    log(f"original: {current_ms:.2f} ms, agreement {current_agreement:.3f}")
    report = OptimizationReport(device, current_ms, current_agreement, state, current_ms, current_agreement,
                                metric=metric)
    baseline_ok = True
    if evaluation is not None:
        report.reference_ap, scores, note = ap_of([(model, "cpu", None)])
        if report.reference_ap is None:
            raise RuntimeError(f"evaluation of the original on the CPU failed: {note}")
        reference_folds[:] = scores
        baseline_ok, report.baseline_ap, note = judge(state.parts(device)) if current_agreement > 0 else (False, None, "")
        log(f"{metric}: CPU FP32 {report.reference_ap:.3f}, original on {device} {report.baseline_ap} ({note})")
    # exact: with annotations, the metric decides (agreement 0.966 on 30 images hid a 4-point AP loss of
    # YOLO26-N on 2 NPU tiles, PLAN.md P3.1, while D-FINE-N's 0.923 hides a 0.2-point loss only, P3.5);
    # without annotations, the agreement with the CPU FP32 original does
    exact_state = baseline_ok if evaluation is not None else current_agreement >= min_agreement
    if not exact_state:
        log(f"the original is inexact on {device}: looking for exact placements first")
    adaptive = Adaptive("current")
    searched: set[tuple[int, bool]] = set()  # (graph id, scheme) already searched for mixed precision

    def mixed_precision(base: State, full: State, full_ap: float, full_ms: float,
                        transformer: bool) -> tuple[State, Trial] | None:
        """Full INT8 is faster but outside the AP budget: keep the module groups in float that recover
        the most AP per ms of latency (each measured alone against full INT8), then verify the mix.
        The group recovering the most AP is split into its sub-modules (and measured again) while it
        costs more than a quarter of what INT8 saves, so that only its sensitive part stays in float."""

        def assess(groups: dict[str, list[str]]) -> dict[str, tuple[float, float]]:
            # sensitivity only ranks the groups: a smaller calibration set (5x faster on the CPU) is enough;
            # the chosen mix is then quantized with the full set and verified
            pool = calibration or samples
            quick = pool[:: max(1, len(pool) // sensitivity_samples)][:sensitivity_samples]
            alone = {g: dataclasses.replace(base, model=int8_ptq(base.model, task, quick, transformer, names))
                     for g, names in groups.items()}
            timings = measure({"int8": full.parts(device), **{g: s.parts(device) for g, s in alone.items()}}, device,
                              rounds=rounds, iterations=iterations, adaptive=Adaptive("int8"))
            gains = {}
            for g, s in alone.items():
                saving, _ = paired_saving(timings, "int8", g)
                ap = ap_of(s.parts(device))[0]  # point estimates: they only rank the groups
                gains[g] = (ap - full_ap if ap is not None else float("nan"), -saving)
                log(f"  float {g}: {metric} {gains[g][0]:+.4f}, latency {gains[g][1]:+.2f} ms")
            return gains

        groups = quantizable_groups(base.model)
        log(f"mixed precision: {len(groups)} groups, each kept in float alone")
        gains = assess(groups)
        for _ in range(3):
            # the most useful of the costly groups that can be split: (rest) cannot, and stopping at it
            # left D-FINE-N's backbone whole (PLAN.md P3.5)
            costly = sorted((g for g in gains if g != "(rest)" and np.isfinite(gains[g][0]) and gains[g][0] > 0
                             and gains[g][1] > 0.25 * (current_ms - full_ms)), key=lambda g: -gains[g][0])
            best, subs = next(((g, s) for g in costly if (s := split_group(base.model, g, groups[g]))), (None, {}))
            if best is None:
                break
            log(f"mixed precision: splitting {best} ({gains[best][1]:+.2f} ms) into {len(subs)} parts")
            del groups[best], gains[best]
            groups.update(subs)
            gains.update(assess(subs))
        floor = report.reference_ap - ap_budget
        order = choose_float_groups(gains, float("inf"))  # every useful group, best AP per ms first
        for k in range(max(1, len(choose_float_groups(gains, floor - full_ap))), len(order) + 1):
            chosen = order[:k]
            names = [n for g in chosen for n in groups[g]]
            scheme = "int8-transformer" if transformer else "int8"  # in the label, so that replay() can rebuild it
            mixed = dataclasses.replace(base, model=int8_ptq(base.model, task, calibration or samples, transformer, names),
                                        changes=[*base.changes, f"{scheme}-mixed (float: {', '.join(chosen)})"])
            t = measure({"current": base.parts(device), "mixed": mixed.parts(device)}, device, rounds=rounds,
                        iterations=iterations, adaptive=adaptive)
            saving, ci = paired_saving(t, "current", "mixed")
            trial = Trial(mixed.label(), "approx", current_ms - saving, ci_ms=ci,
                          note=f"{len(chosen)} of {len(groups)} module groups kept in float")
            report.trials.append(trial)
            if not (saving - ci > 0 and saving > 0.01 * current_ms):
                trial.note += f" (no significant gain: {saving:+.2f} +- {ci:.2f} ms; more float only slows down)"
                return None
            ok, trial.ap, verdict = judge(mixed.parts(device))
            if ok:
                trial.note += f" ({verdict})"
                confirmed, note = confirm(mixed.parts(device))
                trial.note += note
                if not confirmed:  # one confirmation per search: trying the next mix would select again
                    return None
                trial.agreement, _ = accuracy(mixed)
                return mixed, trial
            trial.note += f" ({verdict})"
        return None

    for _ in range(max_steps):
        neighbours = _neighbours(state, quantize if int8 else None, tuple(offload), device)
        if not neighbours:
            break
        variants = {"current": state.parts(device)}
        variants.update({f"n{i}": s.parts(device) for i, (s, _, _) in enumerate(neighbours)})
        timings = measure(variants, device, rounds=rounds, iterations=iterations, adaptive=adaptive)
        savings = {i: paired_saving(timings, "current", f"n{i}") for i in range(len(neighbours))}
        order = sorted(savings, key=lambda i: -np.nan_to_num(savings[i][0], nan=-np.inf))
        # once an approximation is accepted, or when the metric judges exactness, the state is the
        # reference: exact candidates on top of it must keep its agreement (within 0.02) instead of
        # reaching min_agreement
        approximated = any(c.startswith("int8") for c in state.changes)
        exact_now = exact_state
        exact_floor = current_agreement - 0.02 if approximated or evaluation is not None else min_agreement
        applied = False
        for i in order:
            candidate, kind, description = neighbours[i]
            t, (saving, ci) = timings[f"n{i}"], savings[i]
            trial = Trial(candidate.label(), kind, current_ms - saving, note=description, ci_ms=ci)
            repaired = False
            report.trials.append(trial)
            if t.error is not None:
                trial.note = f"does not compile: {t.error[-100:]}"
                continue
            # a gain counts when it is above zero with 95 % confidence and worth 1 % of the latency;
            # an inexact state takes any repair, even a slower one
            faster = saving - ci > 0 and saving > 0.01 * current_ms
            if exact_now and not faster:
                trial.note += f" (no significant gain: {saving:+.2f} +- {ci:.2f} ms)"
                continue
            trial.agreement, error = accuracy(candidate)
            if trial.agreement is None:
                trial.note = error
                continue
            if kind == "approx" and evaluation is not None:
                ok, trial.ap, verdict = judge(candidate.parts(device))
                trial.note += f" ({verdict})"
                if ok:
                    confirmed, note = confirm(candidate.parts(device))
                    trial.note += note
                    if not confirmed:
                        continue
                if not ok:
                    transformer = candidate.changes[-1] == "int8-transformer"
                    key = (id(state.model), transformer)
                    # on an inexact state, keeping groups in float can bring its own error back, so the
                    # search only runs when INT8 already repairs it and misses the budget by its confidence
                    # bound alone (RT-DETRv2, INT8 transformer on 2 tiles: -0.29 points, bound -0.59; the
                    # slower 1-tile repair was taken instead, PLAN.md P3.5)
                    close = trial.ap is not None and trial.ap - report.reference_ap > -ap_budget
                    if trial.ap is None or key in searched or not (exact_now or close):
                        continue
                    searched.add(key)
                    found = mixed_precision(state, candidate, trial.ap, trial.latency_ms, transformer)
                    if found is None:
                        continue
                    candidate, trial = found
            elif kind == "approx":  # no annotations: an approximation must agree like an exact candidate
                if trial.agreement < min_agreement:
                    trial.note += f" (agreement < {min_agreement}; pass an evaluation set to judge by {metric})"
                    continue
            elif exact_now:  # the candidate must stay exact
                if trial.agreement < exact_floor:
                    trial.note += f" (agreement < {exact_floor:.3f})"
                    continue
                if evaluation is not None:
                    ok, trial.ap, verdict = judge(candidate.parts(device))
                    trial.note += f" ({verdict})"
                    if not ok:
                        continue
            else:  # repairing an inexact state: correctness first, judged by the metric when available
                if evaluation is None:
                    repaired = trial.agreement >= min_agreement
                elif trial.agreement > current_agreement:  # a repair agrees better: only those are judged
                    repaired, trial.ap, verdict = judge(candidate.parts(device))
                    trial.note += f" ({verdict})"
                if not repaired and not (faster and trial.agreement > current_agreement):
                    trial.note += f" (still inexact: agreement {trial.agreement:.3f})"
                    continue
            if trial.agreement is None:
                trial.note += " (accuracy check failed)"
                continue
            trial.accepted, applied = True, True
            log(f"accepted {candidate.label()}: {trial.latency_ms:.2f} ms, agreement {trial.agreement:.3f}")
            state, current_ms, current_agreement = candidate, trial.latency_ms, trial.agreement
            exact_state = exact_now or kind == "approx" or repaired
            break
        if not applied:
            break

    report.final, report.final_ms, report.final_agreement = state, current_ms, current_agreement
    if evaluation is not None:
        report.final_ap = report.baseline_ap if not state.changes else ap_of(state.parts(device))[0]
    if recommend:
        what_ifs = [c for c in graph_rewrites(state.model, state.config) if c.kind == "retrain"]
        if what_ifs:
            variants = {"current": state.parts(device)}
            for c in what_ifs:
                variant = dataclasses.replace(state, model=c.parts[0][0])
                variants[c.name] = variant.parts(device)
            timings = measure(variants, device, rounds=rounds, iterations=iterations, adaptive=adaptive)
            for c in what_ifs:
                saving, ci = paired_saving(timings, "current", c.name)
                note = c.description if timings[c.name].error is None else f"does not compile: {timings[c.name].error[-80:]}"
                report.recommendations.append(Trial(c.name, c.kind, current_ms - saving, note=f"{note} (+-{ci:.2f} ms)"))
            report.recommendations.sort(key=lambda t: t.latency_ms)
    return report
