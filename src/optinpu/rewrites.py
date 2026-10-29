"""Rewrites the optimizer tries on a model, each producing a deployable variant (a Candidate).

Three kinds (PLAN.md, Enjeu 3):
- "exact": same function (up to floating point): placement, compile options, equivalent graphs;
- "approx": a slightly different function, judged by the accuracy guard (e.g. INT8 PTQ);
- "retrain": needs training to recover accuracy. Applied to the IR, the variant has the exact latency
  of the retrained model (latency does not depend on dense weight values, P1.0) but meaningless
  outputs: only its latency is measured, as a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import openvino as ov
from openvino import opset13 as ops

from optinpu.graph import constant_ops, split_from

SELECTION = ("TopK", "NonMaxSuppression", "MulticlassNms", "MatrixNms", "NMSRotated")
HEAVY = ("Convolution", "GroupConvolution", "MatMul", "ConvolutionBackpropData", "GridSample")
ONE_TILE = {"NPU_TILES": 1}


@dataclass
class Candidate:
    name: str
    kind: str  # "exact" | "approx" | "retrain"
    parts: list  # [(ov.Model, device, config)], run back to back
    description: str
    changes: list[str] = field(default_factory=list)  # rewrites applied, in order

    @property
    def latency_only(self) -> bool:
        return self.kind == "retrain"


def _npu(model: ov.Model, config: dict | None) -> list:
    return [(model, "npu", config)]


# ---------------------------------------------------------------- placement and compile options

def selection_tail(model: ov.Model) -> tuple[list[str], int, int] | None:
    """(first selection op, ops after it, heavy ops after it) for the first TopK / NMS, or None."""
    first = next((op.get_friendly_name() for op in model.get_ordered_ops() if op.get_type_name() in SELECTION), None)
    if first is None:
        return None
    head, tail = split_from(model, [first])
    kinds = [op.get_type_name() for op in tail.get_ordered_ops() if op.get_type_name() not in ("Parameter", "Result")]
    const = constant_ops(tail)
    runtime = [k for op, k in zip(tail.get_ordered_ops(), kinds) if op.get_friendly_name() not in const]
    return [first], len(runtime), sum(k in HEAVY for k in runtime)


def placements(model: ov.Model, config: dict | None = None) -> list[Candidate]:
    """Move what follows the first TopK / NMS off the NPU: to the CPU when it is a light selection
    tail (end-to-end detector heads); to the iGPU or CPU when it is a heavy decoder (DETR)."""
    found = selection_tail(model)
    if found is None:
        return []
    roots, n_ops, heavy = found
    head, tail = split_from(model, roots)
    if heavy == 0:
        return [Candidate("selection-to-cpu", "exact", [(head, "npu", config), (tail, "cpu", None)],
                          f"run the final selection ({n_ops} ops from {roots[0]}) on the CPU")]
    return [Candidate(f"decoder-to-{device}", "exact", [(head, "npu", config), (tail, device, None)],
                      f"run the {n_ops} ops after {roots[0]} ({heavy} matmul / conv) on the {device.upper()}")
            for device in ("gpu", "cpu")]


def tiles(config: dict | None) -> list[Candidate]:
    """The other NPU tile setting (the compiler default is 2 tiles on this NPU)."""
    current = (config or {}).get("NPU_TILES")
    other = {**(config or {}), "NPU_TILES": 2 if current == 1 else 1}
    return [Candidate(f"tiles={other['NPU_TILES']}", "exact", [], f"compile with NPU_TILES={other['NPU_TILES']}")]


# ---------------------------------------------------------------- graph rewrites

def _replace(node: ov.Node, new: ov.Output) -> None:
    for target in list(node.output(0).get_target_inputs()):
        target.replace_source_output(new)


def activations_to_relu(model: ov.Model, types=("Swish", "HSwish", "Gelu", "Mish", "HSigmoid")) -> tuple[ov.Model, int]:
    """Every activation of `types` replaced by ReLU (retrain tier: SiLU -> ReLU and the like)."""
    clone = model.clone()
    done = 0
    for op in clone.get_ordered_ops():
        if op.get_type_name() in types:
            _replace(op, ops.relu(op.input(0).get_source_output()).output(0))
            done += 1
    return ov.Model(clone.get_results(), clone.get_parameters(), model.get_friendly_name()), done


def _constant_data(output: ov.Output) -> tuple[np.ndarray, ov.Type] | None:
    """Value of a weight input given as Constant or Convert(Constant) (FP16 IRs); None otherwise."""
    node = output.get_node()
    if node.get_type_name() == "Convert":
        source = node.input(0).get_source_output().get_node()
        if source.get_type_name() == "Constant":
            return source.get_data().astype(np.float32), node.get_output_element_type(0)
        return None
    if node.get_type_name() == "Constant":
        return node.get_data().astype(np.float32), node.get_output_element_type(0)
    return None


def depthwise_to_dense(model: ov.Model, max_channels: int = 512) -> tuple[ov.Model, int]:
    """Depthwise convolutions rewritten as dense convolutions with block-diagonal weights (exact).
    The NPU runs depthwise at 0.1-0.3 TOPS and refuses it in INT8 (M1); a dense conv with zeros
    computes more but may run faster, and its zeros are weight sparsity, which the NPU exploits (P1.0)."""
    clone = model.clone()
    done = 0
    for op in clone.get_ordered_ops():
        if op.get_type_name() != "GroupConvolution":
            continue
        weights = _constant_data(op.input(1).get_source_output())
        if weights is None:
            continue
        w, _ = weights
        groups, per_out, per_in, kh, kw = w.shape
        if per_in != 1 or groups > max_channels:
            continue
        dense = np.zeros((groups * per_out, groups, kh, kw), np.float32)
        for g in range(groups):
            dense[g * per_out:(g + 1) * per_out, g] = w[g, :, 0]
        attrs = op.get_attributes()
        weight = ops.convert(ops.constant(dense.astype(np.float16)), ov.Type.f32) \
            if op.input(1).get_source_output().get_node().get_type_name() == "Convert" else ops.constant(dense)
        conv = ops.convolution(op.input(0).get_source_output(), weight, attrs["strides"], attrs["pads_begin"],
                               attrs["pads_end"], attrs["dilations"], attrs["auto_pad"])
        _replace(op, conv.output(0))
        done += 1
    return ov.Model(clone.get_results(), clone.get_parameters(), model.get_friendly_name()), done


def without_norms(model: ov.Model) -> tuple[ov.Model, int]:
    """MVN (LayerNorm's core) replaced by identity: latency of the model once LayerNorm is replaced by a
    BatchNorm, which folds into the neighbouring ops (retrain tier)."""
    clone = model.clone()
    const = constant_ops(clone)
    done = 0
    for op in clone.get_ordered_ops():
        if op.get_type_name() == "MVN" and op.get_friendly_name() not in const:
            _replace(op, op.input(0).get_source_output())
            done += 1
    return ov.Model(clone.get_results(), clone.get_parameters(), model.get_friendly_name()), done


def int8_ptq(model: ov.Model, task, calibration: list, transformer: bool | None = None,
             keep_float: list[str] | None = None) -> ov.Model:
    """Post-training INT8 of the graph (NNCF; approx tier), calibrated on `calibration` samples turned
    into inputs by task.inputs. transformer: NNCF's transformer scheme (LayerNorm / Softmax inputs kept
    in float); default: when the graph has attention or LayerNorm. keep_float: names of ops left in
    float (mixed precision)."""
    from optinpu.quantize import quantize_model

    if transformer is None:
        kinds = {op.get_type_name() for op in model.get_ordered_ops()}
        transformer = bool(kinds & {"Softmax", "MVN", "ScaledDotProductAttention"})
    if len(model.inputs) == 1:
        transform = lambda sample: task.inputs(sample)[0][0]
    else:
        transform = lambda sample: tuple(task.inputs(sample)[0])
    return quantize_model(model, calibration, transform, len(calibration), transformer,
                          ignored_names=keep_float or ())


def quantizable_groups(model: ov.Model, count: int = 10) -> dict[str, list[str]]:
    """Module regions holding the convolutions / matmuls (what INT8 accelerates), for mixed precision:
    region name -> names of its runtime ops. About `count` groups, each with at least one heavy op."""
    from optinpu.profiler import module_regions, pick_depth

    regions = module_regions(model, pick_depth(model, count))
    groups = {r.name: r.ops for r in regions if any(t in HEAVY for t in r.types)}
    return groups


def split_group(model: ov.Model, name: str, ops_: list[str], max_depth: int = 12) -> dict[str, list[str]]:
    """Sub-groups of a group (its module regions one or more levels deeper), each with a heavy op;
    {} when the group cannot be split into two or more such sub-groups."""
    from optinpu.profiler import module_regions

    inside = set(ops_)
    kinds = {op.get_friendly_name(): op.get_type_name() for op in model.get_ordered_ops()}
    for depth in range(name.count(".") + 2, max_depth + 1):
        subs = {}
        for region in module_regions(model, depth):
            members = [n for n in region.ops if n in inside]
            if members and region.name.startswith(name) and any(kinds[n] in HEAVY for n in members):
                subs[region.name] = members
        if len(subs) >= 2:
            return subs
    return {}


def choose_float_groups(gains: dict[str, tuple[float, float]], deficit: float) -> list[str]:
    """Groups to keep in float to recover `deficit` AP at the least latency: each group's (AP recovered,
    latency added in ms) when kept in float alone, assumed additive. Greedy by AP per ms (groups with no
    measurable cost first); stops as soon as the deficit is covered. All useful groups if it cannot be."""
    useful = {g: (ap, ms) for g, (ap, ms) in gains.items() if np.isfinite(ap) and ap > 0}
    order = sorted(useful, key=lambda g: -useful[g][0] / max(useful[g][1], 1e-3))
    chosen, recovered = [], 0.0
    for g in order:
        if recovered >= deficit:
            break
        chosen.append(g)
        recovered += useful[g][0]
    return chosen


def graph_rewrites(model: ov.Model, config: dict | None = None) -> list[Candidate]:
    """Graph-level candidates: exact rewrites first, then retrain-tier what-ifs."""
    from optinpu.profiler import weight_constants, with_weights

    candidates = []
    variant, n = depthwise_to_dense(model)
    if n:
        candidates.append(Candidate("depthwise-to-dense", "exact", _npu(variant, config),
                                    f"{n} depthwise convolutions as dense block-diagonal convolutions"))
    variant, n = activations_to_relu(model)
    if n:
        candidates.append(Candidate("activations-to-relu", "retrain", _npu(variant, config),
                                    f"{n} SiLU / HardSwish / GELU / Mish replaced by ReLU (fine-tune to recover AP)"))
    variant, n = without_norms(model)
    if n:
        candidates.append(Candidate("layernorm-to-batchnorm", "retrain", _npu(variant, config),
                                    f"{n} LayerNorm replaced by a foldable BatchNorm (fine-tune to recover AP)"))
    for sparsity in (50, 75) if weight_constants(model) else ():
        candidates.append(Candidate(f"prune-{sparsity}", "retrain", _npu(with_weights(model, f"sparse{sparsity}"), config),
                                    f"{sparsity} % unstructured weight sparsity (pruning + fine-tune)"))
    return candidates
