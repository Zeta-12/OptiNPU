"""npu_lint: static and dynamic checks of an OpenVINO IR before deploying it on the Intel NPU.

Every rule comes from a problem met in practice (see PLAN.md, "Journal technique NPU").
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov

from optinpu.devices import core

FP16_MAX = 65504.0
_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class LintIssue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    ops: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        where = ""
        if self.ops:
            shown = ", ".join(op.split("/")[0][-60:] for op in self.ops[:3])
            more = f" (+{len(self.ops) - 3})" if len(self.ops) > 3 else ""
            where = f"\n      at {shown}{more}"
        return f"[{self.severity.upper():<7}] {self.code}: {self.message}{where}"


@dataclass
class LintReport:
    device: str
    issues: list[LintIssue] = field(default_factory=list)
    compiles: bool | None = None
    rel_error: float | None = None  # max relative error of outputs vs CPU FP32
    latency_ms: float | None = None
    recommended_config: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def __str__(self) -> str:
        lines = [f"npu_lint on {self.device}: {'OK' if self.ok else 'PROBLEMS FOUND'}"]
        if self.compiles is not None:
            lines.append(f"  compiles: {self.compiles}")
        if self.rel_error is not None:
            lines.append(f"  output error vs CPU FP32: {self.rel_error:.4f}")
        if self.latency_ms is not None:
            lines.append(f"  latency: {self.latency_ms:.1f} ms")
        if self.recommended_config:
            lines.append(f"  recommended config: {self.recommended_config}")
        for issue in sorted(self.issues, key=lambda i: _ORDER[i.severity]):
            lines.append("  " + str(issue))
        return "\n".join(lines)


def _static_checks(model: ov.Model) -> list[LintIssue]:
    issues = []
    ops = model.get_ordered_ops()
    by_type: dict[str, list[str]] = {}
    for op in ops:
        by_type.setdefault(op.get_type_name(), []).append(op.get_friendly_name())

    dynamic = [p.get_friendly_name() for p in model.get_parameters() if p.get_partial_shape().is_dynamic]
    if dynamic:
        issues.append(LintIssue("error", "DYNAMIC-INPUT", "the NPU needs static input shapes; export with a fixed shape", dynamic))

    # Float values computed only from shapes (anchors, sin-cos embeddings): recomputed in FP16 on
    # the NPU. Integer shape arithmetic feeding Reshape & co. is harmless and not reported.
    shape_derived: set[str] = set()
    float_from_shapes = []
    for op in ops:
        name, kind = op.get_friendly_name(), op.get_type_name()
        sources = [i.get_source_output().get_node() for i in op.inputs()]
        if kind == "ShapeOf" or (
            sources
            and any(s.get_friendly_name() in shape_derived for s in sources)
            and all(s.get_friendly_name() in shape_derived or s.get_type_name() == "Constant" for s in sources)
        ):
            shape_derived.add(name)
            if op.get_output_size() and op.get_output_element_type(0).is_real():
                float_from_shapes.append(name)
    if float_from_shapes:
        issues.append(LintIssue(
            "warning", "SHAPE-SUBGRAPH",
            "float tensors computed from shapes at runtime (anchors / position embeddings?): the NPU "
            "recomputes them in FP16; freeze them as constants before export", float_from_shapes))

    big = []
    for op in ops:
        if op.get_type_name() != "Constant" or not op.get_output_element_type(0).is_real():
            continue
        data = op.get_data()
        if data.size and np.isfinite(data).all() and np.abs(data).max() > FP16_MAX:
            big.append(op.get_friendly_name())
    if big:
        issues.append(LintIssue("warning", "FP16-RANGE", f"constants beyond FP16 range (|x| > {FP16_MAX:.0f}) become inf on the NPU", big))

    ceil_pools, maxpool_2x2s1 = [], []
    for op in ops:
        if op.get_type_name() not in ("MaxPool", "AvgPool"):
            continue
        attrs = op.get_attributes()
        if "ceil" in str(attrs.get("rounding_type", "")):
            ceil_pools.append(op.get_friendly_name())
        if op.get_type_name() == "MaxPool" and list(attrs.get("kernel", [])) == [2, 2] and list(attrs.get("strides", [])) == [1, 1]:
            maxpool_2x2s1.append(op.get_friendly_name())
    if ceil_pools:
        issues.append(LintIssue("error", "POOL-CEIL", "ceil rounding in pooling makes the NPU compiler fail (npu_patches.FloorWhenExactPool)", ceil_pools))
    if maxpool_2x2s1:
        issues.append(LintIssue("warning", "MAXPOOL-2x2-S1", "MaxPool k2 s1 is mis-compiled when its input feeds a conv (npu_patches.MaxPool2x2s1)", maxpool_2x2s1))

    if "GridSample" in by_type:
        issues.append(LintIssue("info", "GRID-SAMPLE", "deformable attention (GridSample): runs on the NPU but slowly", by_type["GridSample"]))
    attention = by_type.get("ScaledDotProductAttention", []) + by_type.get("Softmax", [])
    if attention:
        issues.append(LintIssue(
            "info", "ATTENTION-TILES",
            "attention can be mis-split across NPU tiles; if outputs drift try NPU_TILES=1 or a hybrid device", attention))
    if "TopK" in by_type:
        issues.append(LintIssue("info", "TOPK", "TopK: near-ties may order differently than on CPU (harmless for DETR heads)", by_type["TopK"]))

    counts = Counter(op.get_type_name() for op in ops if op.get_type_name() not in ("Constant", "Parameter", "Result", "Convert"))
    compute = sum(counts.values())
    heavy = counts.get("Convolution", 0) + counts.get("GroupConvolution", 0) + counts.get("MatMul", 0)
    if compute and heavy / compute < 0.25:
        issues.append(LintIssue(
            "info", "SMALL-OPS",
            f"only {heavy}/{compute} ops are conv/matmul: the NPU is likely bound by per-op overhead, INT8 will not help much"))
    return issues


def _run(model: ov.Model, device: str, inputs, config=None):
    cfg = {"INFERENCE_PRECISION_HINT": "f32"} if device == "CPU" else dict(config or {})
    request = core().compile_model(model, device, cfg).create_infer_request()
    return list(request.infer(inputs).values()), request


def lint(
    model: str | Path | ov.Model,
    device: str = "npu",
    inputs: list[np.ndarray] | None = None,
    config: dict | None = None,
    check_accuracy: bool = True,
    threshold: float = 0.05,
) -> LintReport:
    """Static checks, then compile on `device`, compare outputs with CPU FP32 and time it.

    `inputs` default to uniform noise; a real image gives a more meaningful accuracy check.
    If outputs drift beyond `threshold`, NPU_TILES=1 is tried and recommended when it helps.
    """
    from optinpu.diagnose import _rel_error, locate_compile_failure

    if isinstance(model, (str, Path)):
        model = core().read_model(model)
    device = device.upper()
    report = LintReport(device, _static_checks(model))
    if any(i.code == "DYNAMIC-INPUT" for i in report.issues):
        return report
    if inputs is None:
        rng = np.random.default_rng(0)
        inputs = [rng.random(tuple(p.get_partial_shape().to_shape()), dtype=np.float32) for p in model.get_parameters()]

    try:
        out, request = _run(model, device, inputs, config)
        report.compiles = True
    except RuntimeError as e:
        report.compiles = False
        culprit = locate_compile_failure(model, device)
        ops = [f"#{culprit[0]} {culprit[1]} {culprit[2]}"] if culprit else []
        report.issues.append(LintIssue("error", "COMPILE-FAIL", str(e).strip().splitlines()[-1][:200], ops))
        return report

    t = time.perf_counter()
    for _ in range(10):
        request.infer(inputs)
    report.latency_ms = (time.perf_counter() - t) / 10 * 1e3

    if check_accuracy and device != "CPU":
        reference, _ = _run(model, "CPU", inputs)
        report.rel_error = max(_rel_error(r, o) for r, o in zip(reference, out))
        if report.rel_error > threshold:
            message = f"outputs differ from CPU FP32 by {report.rel_error:.3f} (threshold {threshold})"
            if device == "NPU" and not (config or {}).get("NPU_TILES"):
                one_tile, _ = _run(model, device, inputs, {**(config or {}), "NPU_TILES": 1})
                err = max(_rel_error(r, o) for r, o in zip(reference, one_tile))
                if err <= threshold:
                    report.recommended_config = {"NPU_TILES": 1}
                    message += f"; with NPU_TILES=1 the error drops to {err:.4f} (slower; or use a hybrid device)"
                else:
                    message += "; NPU_TILES=1 does not fix it: run diagnose.locate_divergence_split"
            report.issues.append(LintIssue("error", "ACCURACY", message))
    return report
