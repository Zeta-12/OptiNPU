"""Find where a device (typically the NPU) numerically diverges from a reference device.

Intermediate tensors are exposed as extra model outputs and compared between the two devices.
Probes are refined by bisection until the first diverging op is isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov

from optinpu.devices import core, normalize_ov_device
from optinpu.graph import split_model

_SKIP = {"Constant", "Parameter", "Result", "ShapeOf", "Convert"}


@dataclass
class Probe:
    index: int  # position in topological order
    name: str
    type: str
    rel_error: float

    def __repr__(self) -> str:
        return f"#{self.index} {self.type} {self.name!r}: rel_err={self.rel_error:.3g}"


def _rel_error(ref: np.ndarray, out: np.ndarray) -> float:
    """max|ref - out| / max|ref|, or the same on sorted values if smaller.

    The sorted variant makes the metric permutation-invariant: after a TopK, near-tied scores can
    be ordered differently by each device, which is harmless but breaks positional comparison.
    """
    ref = ref.astype(np.float64)
    out = out.astype(np.float64)
    if not np.isfinite(out).all() and np.isfinite(ref).all():
        return float("inf")
    finite = np.isfinite(ref) & np.isfinite(out)
    if not finite.any():
        return 0.0
    scale = np.abs(ref[finite]).max() + 1e-6
    direct = np.abs(ref[finite] - out[finite]).max()
    permuted = np.abs(np.sort(ref[finite]) - np.sort(out[finite])).max()
    return float(min(direct, permuted) / scale)


def _probe(model: ov.Model, ops: list, indices: list[int], inputs, ref: str, target: str) -> list[Probe]:
    m = model.clone()
    by_name = {op.get_friendly_name(): op for op in m.get_ordered_ops()}
    outputs = [by_name[ops[i].get_friendly_name()].output(0) for i in indices]
    m.add_outputs(outputs)
    n_orig = len(model.outputs)

    results = {}
    for dev in (ref, target):
        compiled = core().compile_model(m, dev, {"INFERENCE_PRECISION_HINT": "f32"} if dev == "CPU" else {})
        r = compiled.create_infer_request().infer(inputs)
        results[dev] = [r[o] for o in compiled.outputs][n_orig:]

    return [
        Probe(i, ops[i].get_friendly_name(), ops[i].get_type_name(), _rel_error(a, b))
        for i, a, b in zip(indices, results[ref], results[target])
    ]


def _compiles(model: ov.Model, op, device: str) -> bool:
    sub = ov.Model([op.output(0)], model.get_parameters(), "probe")
    try:
        core().compile_model(sub, device)
        return True
    except RuntimeError:
        return False


def locate_compile_failure(model: str | Path | ov.Model, device: str = "npu") -> tuple[int, str, str] | None:
    """Return (index, type, name) of the first op whose truncated graph fails to compile, or None.

    Assumes failures are monotonic: once an op breaks compilation, every graph containing it does.
    """
    if isinstance(model, (str, Path)):
        model = core().read_model(model)
    device = normalize_ov_device(device)
    ops = [op for op in model.get_ordered_ops() if op.get_type_name() not in _SKIP and op.get_output_size() > 0]
    if _compiles(model, ops[-1], device):
        return None
    lo, hi = -1, len(ops) - 1  # ops[lo] compiles (or lo = -1), ops[hi] fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _compiles(model, ops[mid], device):
            lo = mid
        else:
            hi = mid
    op = ops[hi]
    return hi, op.get_type_name(), op.get_friendly_name()


def locate_divergence_split(
    model: str | Path | ov.Model,
    inputs: list[np.ndarray],
    target: str = "npu",
    ref: str = "cpu",
    threshold: float = 0.05,
    verbose: bool = False,
) -> tuple[int, str, str, float] | None:
    """Find the first op that, once executed on `target`, degrades the final outputs.

    The prefix ops[:cut + 1] is compiled as one graph on `target` (so compiler fusions are kept,
    unlike per-op probes) and the rest runs on `ref` in FP32. Bisects `cut`.
    Returns (index, type, name, error) or None if the full model on `target` is fine.
    """
    if isinstance(model, (str, Path)):
        model = core().read_model(model)
    target, ref = normalize_ov_device(target), normalize_ov_device(ref)
    ref_cfg = {"INFERENCE_PRECISION_HINT": "f32"} if ref == "CPU" else {}
    reference = list(core().compile_model(model, ref, ref_cfg).create_infer_request().infer(inputs).values())

    def final_error(device_model: ov.Model, device: str) -> float:
        out = list(core().compile_model(device_model, device).create_infer_request().infer(inputs).values())
        return max(_rel_error(r, o) for r, o in zip(reference, out))

    def error_at(cut: int) -> float:
        head, tail = split_model(model, cut)
        mids = list(core().compile_model(head, target).create_infer_request().infer(inputs).values())
        out = list(core().compile_model(tail, ref, ref_cfg).create_infer_request().infer(mids).values())
        return max(_rel_error(r, o) for r, o in zip(reference, out))

    full = final_error(model, target)
    if verbose:
        print(f"full model on {target}: error {full:.3g}")
    if full <= threshold:
        return None

    ordered = model.get_ordered_ops()
    candidates = [i for i, op in enumerate(ordered) if op.get_type_name() not in ("Constant", "Parameter", "Result")]
    lo, hi = -1, len(candidates) - 1  # error(candidates[lo]) ok, error(candidates[hi]) bad (assumed)
    errors = {}
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            err = error_at(candidates[mid])
        except RuntimeError:  # prefix does not compile: treat as bad
            err = float("inf")
        errors[mid] = err
        if verbose:
            op = ordered[candidates[mid]]
            print(f"cut #{candidates[mid]} {op.get_type_name()} {op.get_friendly_name()[:70]}: error {err:.3g}")
        if err > threshold:
            hi = mid
        else:
            lo = mid
    op = ordered[candidates[hi]]
    return candidates[hi], op.get_type_name(), op.get_friendly_name(), errors.get(hi, full)


def locate_divergence(
    model: str | Path | ov.Model,
    inputs: list[np.ndarray],
    target: str = "npu",
    ref: str = "cpu",
    threshold: float = 0.05,
    probes_per_round: int = 24,
) -> tuple[Probe | None, list[Probe]]:
    """Return (first diverging op or None, all probes measured).

    An op "diverges" when max|ref - target| / max|ref| exceeds `threshold`.
    """
    if isinstance(model, (str, Path)):
        model = core().read_model(model)
    ref, target = normalize_ov_device(ref), normalize_ov_device(target)
    ops = [
        op for op in model.get_ordered_ops()
        if op.get_type_name() not in _SKIP
        and op.get_output_size() > 0
        and op.get_output_element_type(0).is_real()
    ]

    lo, hi = 0, len(ops) - 1  # invariant: divergence (if any) is in (lo, hi]
    history: list[Probe] = []
    first_bad: Probe | None = None
    while hi - lo > 0:
        step = max(1, (hi - lo) // probes_per_round)
        indices = sorted(set(range(lo + step, hi + 1, step)) | {hi})
        probes = _probe(model, ops, indices, inputs, ref, target)
        history += probes
        bad = next((p for p in probes if p.rel_error > threshold), None)
        if bad is None:
            break
        first_bad = bad
        prev_good = max((p.index for p in probes if p.index < bad.index), default=lo)
        if bad.index - prev_good <= 1:
            break
        lo, hi = prev_good, bad.index - 1
        if lo >= hi:
            break
    return first_bad, history
