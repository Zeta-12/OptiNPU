"""Measured profiler: where the latency of a model goes on a device, as measured latency deltas.

Vendor counters (optinpu.profile) report per-task times that overlap on the NPU and ignore fusions,
so they do not tell what a change would save. Here every number is the measured latency of a graph
variant, compared with the model in the same process, round-robin (thermal drift cancels):

- depth profile: latency of growing prefixes of the graph (cumulative cost along the network);
- region bypass: latency saved when a module (residual block, transformer layer...) is replaced
  by identity, which bounds what any rewrite of that module can save, fusions included;
- op-type bypass: same for every shape-preserving op of a type (activations, norms, softmax).

Latency does not depend on weight values on dense hardware, so variants keep the original weights
and their outputs are meaningless: only their timing is used.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

from optinpu.devices import core
from optinpu.graph import constant_ops, split_model

STRUCTURAL = ("Parameter", "Result", "Constant")


# ---------------------------------------------------------------- op -> source module

def module_tokens(name: str) -> tuple[str, ...] | None:
    """Source module of an op from its friendly name, as path tokens; None for ops created by
    OpenVINO transformations ("Multiply_123") or by NNCF hooks.

    PyTorch frontend: "__module.backbone.stages.1/aten::conv2d/Convolution" -> (backbone, stages, 1)
    ONNX: "/model.8/m.0/cv2/conv/Conv/WithoutBiases" -> (model, 8, m, 0, cv2, conv); op names and
    OpenVINO suffixes start with an uppercase letter, module names do not.
    """
    if name.startswith("__module."):
        path = name[len("__module."):].split("/", 1)[0]
        return None if path.startswith("__nncf") else tuple(path.split("."))
    if name.startswith("/"):
        segments = [s for s in name.split("/") if s]
        while segments and segments[-1][:1].isupper():
            segments.pop()
        tokens, previous = [], ""
        for seg in segments:
            # exporters repeat the parent's name: /stages.0/stages.0.0/ -> stages, 0, 0
            own = seg[len(previous) + 1:] if previous and seg.startswith(previous + ".") else seg
            tokens += own.split(".")
            previous = seg
        return tuple(tokens) or None
    return None


def runtime_ops(model: ov.Model) -> list[ov.Node]:
    """Ops executed at inference time, in topological order."""
    const = constant_ops(model)
    return [op for op in model.get_ordered_ops()
            if op.get_type_name() not in STRUCTURAL and op.get_friendly_name() not in const]


def op_modules(model: ov.Model) -> dict[str, tuple[str, ...]]:
    """Runtime op name -> module tokens. Ops without a module name take their producer's module,
    else their consumer's (unattributed ops get ("?",))."""
    ordered = runtime_ops(model)
    names = {op.get_friendly_name() for op in ordered}
    modules = {op.get_friendly_name(): module_tokens(op.get_friendly_name()) for op in ordered}
    for _ in range(3):  # propagate through chains of unnamed ops
        for op in ordered:
            name = op.get_friendly_name()
            if modules[name] is not None:
                continue
            producers = [i.get_source_output().get_node().get_friendly_name() for i in op.inputs()]
            consumers = [t.get_node().get_friendly_name() for o in op.outputs() for t in o.get_target_inputs()]
            for other in producers + consumers:
                if other in names and modules.get(other) is not None:
                    modules[name] = modules[other]
                    break
    return {k: v or ("?",) for k, v in modules.items()}


# ---------------------------------------------------------------- regions

@dataclass
class Region:
    name: str  # dotted module path, or "type:<OpType>"
    ops: list[str]  # runtime op names
    first: int  # topological index of its first and last ops
    last: int
    types: Counter = field(default_factory=Counter)

    def summary(self, top: int = 3) -> str:
        return ", ".join(f"{t} {n}" for t, n in self.types.most_common(top))


def common_prefix(model: ov.Model) -> tuple[str, ...]:
    """Module tokens shared by every attributed op (e.g. ("model", "model") for wrapped HF models)."""
    paths = [m for m in op_modules(model).values() if m != ("?",)]
    prefix_tokens: tuple[str, ...] = paths[0] if paths else ()
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix_tokens), len(path)) and prefix_tokens[n] == path[n]:
            n += 1
        prefix_tokens = prefix_tokens[:n]
    return prefix_tokens


def module_regions(model: ov.Model, depth: int) -> list[Region]:
    """Runtime ops grouped by their module path truncated to `depth` tokens after the common prefix,
    ordered by first op."""
    ordered = runtime_ops(model)
    modules = op_modules(model)
    skip = len(common_prefix(model))
    groups: dict[tuple, Region] = {}
    for index, op in enumerate(ordered):
        path = modules[op.get_friendly_name()]
        key = path[skip:skip + depth] if path != ("?",) else path
        region = groups.setdefault(key, Region(".".join(key) or "(root)", [], index, index))
        region.ops.append(op.get_friendly_name())
        region.last = index
        region.types[op.get_type_name()] += 1
    return sorted(groups.values(), key=lambda r: r.first)


def pick_depth(model: ov.Model, min_regions: int, max_depth: int = 12) -> int:
    """Smallest depth giving at least `min_regions` module regions (or `max_depth`)."""
    for depth in range(1, max_depth + 1):
        if len(module_regions(model, depth)) >= min_regions:
            return depth
    return max_depth


def _by_name(model: ov.Model) -> dict[str, ov.Node]:
    return {op.get_friendly_name(): op for op in model.get_ordered_ops()}


def bypass(model: ov.Model, names: set[str] | list[str]) -> ov.Model | None:
    """Copy of `model` where the ops `names` are skipped: each tensor they hand to the rest of the
    graph is replaced by the one tensor they receive with the same shape and type. None when that
    mapping is ambiguous or impossible (the region changes shapes). Inputs computed from the region's
    own outputs (a region that is not contiguous) cannot stand in for them: that would make a loop."""
    names = set(names)
    clone = model.clone()
    nodes = _by_name(clone)
    const = constant_ops(clone)
    inputs, outputs = [], []
    for name in names:
        op = nodes[name]
        for port in op.inputs():
            source = port.get_source_output()
            if source.get_node().get_friendly_name() not in names | const and source not in inputs:
                inputs.append(source)
        for out in op.outputs():
            if any(t.get_node().get_friendly_name() not in names for t in out.get_target_inputs()):
                outputs.append(out)
    if not outputs:
        return None
    downstream, stack = set(), [t.get_node() for o in outputs for t in o.get_target_inputs()]
    while stack:
        node = stack.pop()
        name = node.get_friendly_name()
        if name in downstream or name in names:
            continue
        downstream.add(name)
        stack += [t.get_node() for o in node.outputs() for t in o.get_target_inputs()]
    inputs = [i for i in inputs if i.get_node().get_friendly_name() not in downstream]
    signature = lambda o: (str(o.get_partial_shape()), o.get_element_type().get_type_name())
    for out in outputs:
        matches = [i for i in inputs if signature(i) == signature(out)]
        if len(matches) != 1:
            return None
        for target in list(out.get_target_inputs()):
            if target.get_node().get_friendly_name() not in names:
                target.replace_source_output(matches[0])
    return ov.Model(clone.get_results(), clone.get_parameters(), f"{model.get_friendly_name()}-bypass")


def bypass_types(model: ov.Model, types: set[str] | list[str]) -> tuple[ov.Model, int, int]:
    """Copy of `model` with every op of `types` whose first input has its output's shape replaced by
    that input (identity). Returns (model, bypassed count, total count of those types)."""
    clone = model.clone()
    const = constant_ops(clone)
    total = bypassed = 0
    for op in clone.get_ordered_ops():
        if op.get_type_name() not in types or op.get_friendly_name() in const:
            continue
        total += 1
        source = op.input(0).get_source_output()
        if op.get_output_size() != 1 or source.get_partial_shape() != op.output(0).get_partial_shape():
            continue
        if source.get_element_type() != op.output(0).get_element_type():
            continue
        for target in list(op.output(0).get_target_inputs()):
            target.replace_source_output(source)
        bypassed += 1
    return ov.Model(clone.get_results(), clone.get_parameters(), f"{model.get_friendly_name()}-types"), bypassed, total


def bypassable(model: ov.Model, types: set[str] | list[str]) -> list[str]:
    """Names of the ops that bypass_types(model, types) replaces by identity."""
    const = constant_ops(model)
    names = []
    for op in model.get_ordered_ops():
        if op.get_type_name() not in types or op.get_friendly_name() in const or op.get_output_size() != 1:
            continue
        source = op.input(0).get_source_output()
        if (source.get_partial_shape() == op.output(0).get_partial_shape()
                and source.get_element_type() == op.output(0).get_element_type()):
            names.append(op.get_friendly_name())
    return names


def _signature(op: ov.Node, const: set[str]) -> str:
    """Ops with the same signature cost the same in isolation (type, attributes, input shapes, and
    which inputs are constants)."""
    inputs = []
    for port in op.inputs():
        source = port.get_source_output()
        is_const = source.get_node().get_friendly_name() in const
        inputs.append(f"{'c' if is_const else 'x'}{source.get_partial_shape()}{source.get_element_type().get_type_name()}")
    return f"{op.get_type_name()}|{sorted(op.get_attributes().items())}|{inputs}"


@dataclass
class IsolatedOps:
    """Single-op graphs for an additive cost model (nn-Meter style: the baseline the measured profile is
    compared with). Each runtime op alone, its non-constant inputs as parameters and its outputs reduced
    to scalars; and the same inputs alone, whose latency (the fixed cost of a graph) is subtracted."""

    signature: dict[str, str]  # op name -> signature (ops of equal signature cost the same)
    graphs: dict[str, ov.Model]  # signature -> single-op graph
    inputs: dict[str, str]  # signature -> key of its inputs-only graph
    empties: dict[str, ov.Model]  # inputs key -> inputs-only graph


def isolated_ops(model: ov.Model) -> IsolatedOps:
    const = constant_ops(model)
    result = IsolatedOps({}, {}, {}, {})
    for op in runtime_ops(model):
        signature = _signature(op, const)
        result.signature[op.get_friendly_name()] = signature
        if signature in result.graphs:
            continue
        clone = model.clone()
        node = _by_name(clone)[op.get_friendly_name()]
        params = []
        for port in node.inputs():
            source = port.get_source_output()
            if source.get_node().get_friendly_name() in const:
                continue
            p = ops.parameter(source.get_partial_shape(), source.get_element_type())
            port.replace_source_output(p.output(0))
            params.append(p)
        if not params:
            continue
        result.graphs[signature] = reduced_outputs(ov.Model(list(node.outputs()), params, "op"))
        key = str([(str(p.get_partial_shape()), p.get_element_type().get_type_name()) for p in params])
        result.inputs[signature] = key
        if key not in result.empties:
            result.empties[key] = reduced_outputs(ov.Model([p.output(0) for p in params], params, "inputs"))
    return result


def reduced_outputs(model: ov.Model) -> ov.Model:
    """Same graph with each output replaced by its mean (a scalar): output transfers then cost the same
    for every prefix of the depth profile, while no op can be pruned."""
    outs = []
    for result in model.get_results():
        source = result.input(0).get_source_output()
        if not source.get_element_type().is_real():  # indices, masks: reduce them as floats
            source = ops.convert(source, ov.Type.f32).output(0)
        rank = source.get_partial_shape().rank.get_length()
        outs.append(ops.reduce_mean(source, ops.constant(np.arange(rank, dtype=np.int64)), keep_dims=False).output(0)
                    if rank else source)
    return ov.Model(outs, model.get_parameters(), model.get_friendly_name())


def prefix(model: ov.Model, cut: int) -> ov.Model:
    """Ops up to topological position `cut` (of model.get_ordered_ops()), outputs reduced to scalars."""
    head, _ = split_model(model, cut)
    return reduced_outputs(head)


WEIGHT_CONSUMERS = ("Convolution", "GroupConvolution", "MatMul", "ConvolutionBackpropData")


def weight_constants(model: ov.Model) -> list[ov.Node]:
    """Constants holding conv / matmul weights: found from each weight input, back through
    decompression (Convert, Multiply / Subtract by scales, Reshape) and FakeQuantize (data input)."""
    found: dict[str, ov.Node] = {}
    for op in model.get_ordered_ops():
        if op.get_type_name() not in WEIGHT_CONSUMERS:
            continue
        node = op.input(1).get_source_output().get_node()
        for _ in range(8):
            kind = node.get_type_name()
            if kind == "Constant":
                if len(node.get_output_shape(0)) >= 2:
                    found[node.get_friendly_name()] = node
                break
            if kind in ("Convert", "Reshape", "Multiply", "Subtract", "FakeQuantize", "Transpose", "Unsqueeze"):
                node = node.input(0).get_source_output().get_node()
            else:
                break
    return list(found.values())


def with_weights(model: ov.Model, mode: str, seed: int = 0) -> ov.Model:
    """Copy of `model` whose conv / matmul weights are replaced, to test that latency does not depend
    on weight values. mode: "random" (normal, same mean and std per tensor; integers uniform over the
    tensor's range), "zeros", or "sparseNN" (NN % of the original weights set to zero)."""
    clone = model.clone()
    rng = np.random.default_rng(seed)
    for const in weight_constants(clone):
        data = const.get_data()
        if mode == "random":
            if np.issubdtype(data.dtype, np.floating):
                new = rng.normal(float(data.mean()), float(data.std()) + 1e-6, data.shape)
            else:
                new = rng.integers(int(data.min()), int(data.max()) + 1, data.shape)
        elif mode == "zeros":
            new = np.zeros_like(data)
        elif mode.startswith("sparse"):
            new = np.where(rng.random(data.shape) < int(mode[len("sparse"):]) / 100, 0, data)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        replacement = ops.constant(np.asarray(new).astype(data.dtype), const.get_output_element_type(0))
        for target in list(const.output(0).get_target_inputs()):
            target.replace_source_output(replacement.output(0))
    return ov.Model(clone.get_results(), clone.get_parameters(), f"{model.get_friendly_name()}-{mode}")


# ---------------------------------------------------------------- measurement

@dataclass
class Timing:
    median_ms: float
    q1_ms: float
    q3_ms: float
    error: str | None = None
    rounds: list[float] = field(default_factory=list)  # median of each round, in time order


def paired_saving(timings: dict[str, Timing], reference: str, name: str) -> tuple[float, float]:
    """(mean per-round saving of `name` vs `reference` in ms, its 95 % confidence half-width).
    Rounds pair the two models in time, which cancels thermal and power drift."""
    a, b = timings[reference].rounds, timings[name].rounds
    n = min(len(a), len(b))
    if n == 0 or timings[name].error is not None:
        return float("nan"), float("nan")
    d = np.asarray(a[:n]) - np.asarray(b[:n])
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")


@dataclass
class Adaptive:
    """Repeat rounds until every saving's 95 % half-width is below max(target_ms, target_rel x
    reference latency), within max_rounds and max_seconds per job."""

    reference: str
    target_ms: float = 0.02
    target_rel: float = 0.005
    max_rounds: int = 40
    max_seconds: float = 600


def measure(variants: dict, device: str = "npu", config: dict | None = None,
            rounds: int = 5, iterations: int = 30, timeout: float = 1200, max_gb: float = 12.0,
            input_modes: dict[str, str] | None = None, adaptive: Adaptive | None = None) -> dict[str, Timing]:
    """Latency of each variant, measured in one sandboxed process, round-robin over `rounds` rounds of
    `iterations` inferences each (pairs the variants against the same thermal / power state); with
    `adaptive`, more rounds until the savings against adaptive.reference are known precisely enough.

    A variant is an ov.Model or IR path (run on `device` with `config`), or a list of
    (model, device, config) parts run back to back (e.g. NPU head, CPU tail).
    input_modes: variant name -> "random" (default), "zeros", or "file:<npz>" (samples s<i>_<j> used in turn,
    e.g. real preprocessed images).
    When the whole job fails (one variant losing the NPU kills the process), each variant is measured
    again in its own process (with the reference, if any), so that only the culprit is reported as failed."""
    timings = _measure_once(variants, device, config, rounds, iterations, timeout, max_gb, input_modes, adaptive)
    failed = all(t.error is not None for t in timings.values())
    if failed and len(variants) > 1:
        timings = {}
        reference = adaptive.reference if adaptive else None
        for name, model in variants.items():
            if name == reference:
                continue
            group = {reference: variants[reference], name: model} if reference else {name: model}
            modes = {k: v for k, v in (input_modes or {}).items() if k in group} or None
            result = _measure_once(group, device, config, rounds, iterations, timeout, max_gb, modes, adaptive)
            timings[name] = result[name]
            if reference and (reference not in timings or timings[reference].error):
                timings[reference] = result[reference]
    return timings


def _measure_once(variants, device, config, rounds, iterations, timeout, max_gb, input_modes,
                  adaptive=None) -> dict[str, Timing]:
    import tempfile

    from optinpu.sandbox import run_job

    with tempfile.TemporaryDirectory(prefix="optinpu-variants-") as tmp:
        counter = iter(range(10**6))

        def saved(model) -> str:
            if isinstance(model, ov.Model):
                path = Path(tmp) / f"v{next(counter)}.xml"
                ov.save_model(model, path, compress_to_fp16=False)
                return str(path)
            return str(model)

        paths = {}
        for name, model in variants.items():
            if isinstance(model, list):
                paths[name] = {"parts": [[saved(m), d, c] for m, d, c in model]}
            else:
                paths[name] = saved(model)
        spec = {"job": "compare", "irs": paths, "device": device, "config": config, "rounds": rounds,
                "iterations": iterations, "input_modes": input_modes}
        if adaptive is not None:
            spec.update(reference=adaptive.reference, target_ms=adaptive.target_ms, target_rel=adaptive.target_rel,
                        max_rounds=adaptive.max_rounds, max_seconds=adaptive.max_seconds)
        job = run_job(spec, timeout, max_gb)
    if not job.ok:
        return {name: Timing(float("nan"), float("nan"), float("nan"), job.error) for name in variants}
    return {name: Timing(**t) for name, t in job.result.items()}


# ---------------------------------------------------------------- report

@dataclass
class Entry:
    name: str
    ms: float  # measured saving (bypass) or segment cost (depth); NaN if the variant failed
    share: float  # of the full latency
    detail: str = ""
    error: str | None = None
    ci_ms: float = float("nan")  # 95 % confidence half-width of a saving (bypass entries)

    @property
    def significant(self) -> bool:
        """The saving is above zero with 95 % confidence."""
        return self.error is None and np.isfinite(self.ci_ms) and self.ms - self.ci_ms > 0


@dataclass
class ProfileReport:
    device: str
    latency_ms: float
    noise_ms: float  # interquartile half-range of the full model's latency
    depth: list[Entry] = field(default_factory=list)
    regions: list[Entry] = field(default_factory=list)
    op_types: list[Entry] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"measured profile on {self.device}: {self.latency_ms:.2f} ms (+-{self.noise_ms:.2f})"]
        for title, entries in (("depth profile (segment cost; indicative: prefixes compile differently)", self.depth),
                               ("region bypass (saving +- 95 % CI; * = significant)", self.regions),
                               ("op-type bypass (saving +- 95 % CI; * = significant)", self.op_types)):
            if not entries:
                continue
            lines += ["", f"{title}:"]
            for e in entries:
                if e.error is not None:
                    value = f"failed: {e.error[:60]}"
                elif np.isfinite(e.ci_ms):
                    value = f"{e.ms:7.2f} +-{e.ci_ms:5.2f} ms {100 * e.share:5.1f} % {'*' if e.significant else ' '}"
                else:
                    value = f"{e.ms:7.2f} ms {100 * e.share:5.1f} %"
                lines.append(f"  {e.name[:48]:<48} {value}  {e.detail}")
        return "\n".join(lines)


def _depth_cuts(model: ov.Model, regions: list[Region]) -> list[tuple[str, int]]:
    """(label, cut index in model.get_ordered_ops()) after the last op of each region, in order."""
    index = {op.get_friendly_name(): i for i, op in enumerate(model.get_ordered_ops())}
    cuts, seen = [], set()
    for region in sorted(regions, key=lambda r: r.last):
        cut = max(index[name] for name in region.ops)
        if cut not in seen:
            cuts.append((region.name, cut))
            seen.add(cut)
    return cuts


SHAPE_PRESERVING_TYPES = ("Swish", "HSwish", "Gelu", "Mish", "Sigmoid", "HSigmoid", "Relu", "Clamp", "PRelu",
                          "Elu", "Tanh", "MVN", "Softmax", "NormalizeL2")


def profile_measured(
    model: str | Path | ov.Model,
    device: str = "npu",
    config: dict | None = None,
    depth: int | None = None,
    region_depth: int | None = None,
    op_types: tuple[str, ...] = SHAPE_PRESERVING_TYPES,
    rounds: int = 5,
    iterations: int = 30,
    batch: int = 6,
    adaptive: Adaptive | None = None,
    log=print,
) -> ProfileReport:
    """Measured profile of `model` on `device` (see module docstring).

    depth: module-path depth (after the common prefix) of the depth-profile segments; default: the
    smallest giving 8 segments. region_depth: depth of the bypassed regions; default: the smallest
    giving 24 regions. op_types: op types to bypass.
    Variants are measured `batch` at a time with the full model, in sandboxed processes. Savings are
    paired per round with the full model of the same batch, with a 95 % confidence interval; rounds are
    repeated until it is narrow enough (`adaptive`, default Adaptive("full"): 0.5 % of the latency or
    0.02 ms, at most 40 rounds), so small models get as many repetitions as they need.
    """
    adaptive = adaptive or Adaptive("full")
    if isinstance(model, (str, Path)):
        model = core().read_model(model)
    depth = depth or pick_depth(model, 8)
    region_depth = region_depth or pick_depth(model, 24)
    reference = reduced_outputs(model)

    variants: dict[str, tuple[str, ov.Model, str]] = {}  # key -> (kind, model, detail)
    params = model.get_parameters()
    variants["depth:(input transfer, fixed cost)"] = (
        "depth", reduced_outputs(ov.Model([p.output(0) for p in params], params, "input")), "")
    for label, cut in _depth_cuts(model, module_regions(model, depth)):
        variants[f"depth:{label}"] = ("depth", prefix(model, cut), "")
    for region in module_regions(model, region_depth):
        if len(region.ops) < 2:
            continue
        variant = bypass(model, region.ops)
        if variant is not None:
            variants[f"region:{region.name}"] = ("region", reduced_outputs(variant),
                                                 f"{len(region.ops)} ops: {region.summary()}")
    present = Counter(op.get_type_name() for op in runtime_ops(model))
    for t in op_types:
        if present[t]:
            variant, done, total = bypass_types(model, {t})
            if done:
                variants[f"type:{t}"] = ("type", reduced_outputs(variant), f"{done}/{total} bypassed")

    keys = list(variants)
    timings: dict[str, Timing] = {}
    savings: dict[str, tuple[float, float]] = {}
    full: list[Timing] = []
    for start in range(0, len(keys), batch):
        chunk = keys[start:start + batch]
        log(f"measuring {start + len(chunk)}/{len(keys)} variants on {device}")
        result = measure({"full": reference, **{k: variants[k][1] for k in chunk}}, device, config, rounds, iterations,
                         adaptive=adaptive)
        for k in chunk:
            savings[k] = paired_saving(result, "full", k)
        full.append(result.pop("full"))
        timings.update(result)
    latency = float(np.nanmedian([t.median_ms for t in full]))
    report = ProfileReport(device, latency, float(np.nanmedian([(t.q3_ms - t.q1_ms) / 2 for t in full])))
    # rescale each variant by the full model measured in the same batch (drift between batches)
    scale = {k: latency / full[i // batch].median_ms for i, k in enumerate(keys)}

    previous = 0.0
    for key in keys:
        kind, _, detail = variants[key]
        t = timings[key]
        name = key.split(":", 1)[1]
        ci = float("nan")
        if kind == "depth":
            value = t.median_ms * scale[key]
            ms = value - previous
            if t.error is None:
                previous = value
        else:
            ms, ci = savings[key]
        entry = Entry(name, ms, ms / latency, detail, t.error, ci)
        {"depth": report.depth, "region": report.regions, "type": report.op_types}[kind].append(entry)
    report.regions.sort(key=lambda e: -np.nan_to_num(e.ms, nan=-1e9))
    report.op_types.sort(key=lambda e: -np.nan_to_num(e.ms, nan=-1e9))
    return report


def vendor_by_region(model: ov.Model, layers, depth: int) -> dict[str, float]:
    """Vendor counter times (optinpu.profile.profile) summed per module region, for comparison.
    Device task names that match no IR op are reported under "?"."""
    modules = op_modules(model)
    totals: dict[str, float] = defaultdict(float)
    for layer in layers:
        base = re.split(r"[?]", layer.name)[0]
        key = modules.get(base) or module_tokens(base) or ("?",)
        totals[".".join(key[:depth])] += layer.ms
    return dict(totals)
