"""OpenVINO graph surgery: sub-models and two-way splits."""

from __future__ import annotations

import re

import openvino as ov


def stage_submodel(model: ov.Model, pattern: str) -> ov.Model:
    """Sub-model made of the ops whose name matches `pattern` (and their inputs).

    Its outputs are every tensor produced by a matching op and consumed outside the stage.
    """
    regex = re.compile(pattern)
    outputs = []
    for op in model.get_ordered_ops():
        if not regex.search(op.get_friendly_name()):
            continue
        for out in op.outputs():
            if any(not regex.search(t.get_node().get_friendly_name()) for t in out.get_target_inputs()):
                outputs.append(out)
    if not outputs:
        raise ValueError(f"No op matches {pattern!r}")
    return ov.Model(outputs, model.get_parameters(), pattern)


def split_model(model: ov.Model, cut: int) -> tuple[ov.Model, ov.Model]:
    """Split at topological position `cut`: (prefix, suffix).

    prefix: original inputs -> every tensor produced by ops[:cut + 1] and consumed after the cut.
    suffix: those tensors (as new parameters, in the same order) -> original outputs.
    """
    from openvino import opset13 as ops

    def crossing(m: ov.Model):
        ordered = m.get_ordered_ops()
        # Results always stay in the suffix, even when ordered before the cut
        prefix = {op.get_friendly_name() for op in ordered[: cut + 1] if op.get_type_name() != "Result"}
        outs = []
        for op in ordered[: cut + 1]:
            if op.get_type_name() in ("Constant", "Result"):
                continue
            for out in op.outputs():
                if any(t.get_node().get_friendly_name() not in prefix for t in out.get_target_inputs()):
                    outs.append(out)
        return outs, prefix

    # Work on clones: building a Model attaches Result nodes to the outputs it uses, which would
    # add phantom consumers to `model` and corrupt later splits.
    head_model = model.clone()
    head_outs, _ = crossing(head_model)
    head = ov.Model(head_outs, head_model.get_parameters(), "prefix")

    tail_model = model.clone()
    tail_outs, prefix = crossing(tail_model)
    params = []
    for out in tail_outs:
        p = ops.parameter(out.get_partial_shape(), out.get_element_type())
        params.append(p)
        for target in list(out.get_target_inputs()):
            if target.get_node().get_friendly_name() not in prefix:
                target.replace_source_output(p.output(0))
    tail = ov.Model(tail_model.get_results(), params, "suffix")
    return head, tail


def constant_ops(model: ov.Model) -> set[str]:
    """Names of ops computed from constants only (weight decompression...), folded at compile time."""
    const: set[str] = set()
    for op in model.get_ordered_ops():
        sources = [i.get_source_output().get_node() for i in op.inputs()]
        if op.get_type_name() == "Constant" or (sources and all(s.get_friendly_name() in const for s in sources)):
            const.add(op.get_friendly_name())
    return const


def split_from(model: ov.Model, roots: list[str]) -> tuple[ov.Model, ov.Model]:
    """Split so that the ops `roots` and everything that depends on them form the tail (e.g. the final
    TopK selection of a detector, to run it on the CPU).

    head: original inputs -> the tensors the tail reads from the rest of the graph (constants stay in
    the tail), then the original outputs that do not depend on `roots`.
    tail: those tensors, as parameters in the same order -> the original outputs, in their order.
    """
    from openvino import opset13 as ops

    def analyse(m: ov.Model):
        nodes = {op.get_friendly_name(): op for op in m.get_ordered_ops()}
        tail, stack = set(), [nodes[name] for name in roots]
        while stack:
            op = stack.pop()
            if op.get_friendly_name() in tail or op.get_type_name() == "Result":
                continue
            tail.add(op.get_friendly_name())
            stack += [t.get_node() for out in op.outputs() for t in out.get_target_inputs()]
        const = constant_ops(m)
        crossing = []
        for op in m.get_ordered_ops():
            if op.get_friendly_name() in tail | const or op.get_type_name() == "Result":
                continue
            for out in op.outputs():
                if any(t.get_node().get_friendly_name() in tail for t in out.get_target_inputs()):
                    crossing.append(out)
        key = lambda o: (o.get_node().get_friendly_name(), o.get_index())
        known = {key(o) for o in crossing}
        for result in m.get_results():
            source = result.input(0).get_source_output()
            if source.get_node().get_friendly_name() not in tail and key(source) not in known:
                crossing.append(source)
                known.add(key(source))
        return tail, crossing

    head_model = model.clone()
    _, head_outs = analyse(head_model)
    head = ov.Model(head_outs, head_model.get_parameters(), "head")

    tail_model = model.clone()
    tail_ops, crossing = analyse(tail_model)
    params = []
    for out in crossing:
        p = ops.parameter(out.get_partial_shape(), out.get_element_type())
        params.append(p)
        for target in list(out.get_target_inputs()):
            node = target.get_node()
            if node.get_friendly_name() in tail_ops or node.get_type_name() == "Result":
                target.replace_source_output(p.output(0))
    return head, ov.Model(tail_model.get_results(), params, "tail")


def split_after(model: ov.Model, pattern: str) -> tuple[ov.Model, ov.Model]:
    """Split right after the last op (in topological order) whose name matches `pattern`."""
    regex = re.compile(pattern)
    indices = [i for i, op in enumerate(model.get_ordered_ops()) if regex.search(op.get_friendly_name())]
    if not indices:
        raise ValueError(f"No op matches {pattern!r}")
    return split_model(model, max(indices))
