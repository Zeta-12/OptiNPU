"""PyTorch -> OpenVINO IR export.

The NPU compiles static graphs: every export fixes the input shape.
"""

from __future__ import annotations

from pathlib import Path

import openvino as ov
from openvino import passes


def export_openvino(
    model,
    output: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    fp16: bool = True,
    npu_patches: bool = True,
) -> Path:
    """Convert a torch.nn.Module to an OpenVINO IR (.xml + .bin) with a static input shape.

    `fp16=True` stores weights in FP16, which is what the NPU executes natively.
    `npu_patches=True` rewrites modules the NPU compiler mishandles (see optinpu.npu_patches);
    the rewrites are exact, so the IR stays correct on every device.
    Returns the path of the .xml file.
    """
    import torch

    from optinpu.npu_patches import apply_npu_patches

    model = model.eval().cpu()
    if npu_patches:
        model, _ = apply_npu_patches(model)
    example = torch.zeros(input_shape)
    with torch.no_grad():
        ov_model = ov.convert_model(model, example_input=example, input=[input_shape])
    fold_constants(ov_model)

    output = Path(output).with_suffix(".xml")
    output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, output, compress_to_fp16=fp16)
    return output


def load_model(model, input_shape: tuple[int, ...] | None = None) -> ov.Model:
    """Any model as a static-shape ov.Model: an ov.Model, an IR (.xml) or ONNX path, or a
    torch.nn.Module (then `input_shape` is required). `input_shape` also fixes a dynamic input of a
    single-input IR / ONNX model. Constant subgraphs are folded; nothing else is changed (the NPU
    patches are exact rewrites that the optimizer applies and measures).
    """
    from optinpu.devices import core

    if isinstance(model, ov.Model):
        ov_model = model
    elif isinstance(model, (str, Path)):
        path = Path(model)
        ov_model = core().read_model(path) if path.suffix == ".xml" else ov.convert_model(path)
    else:
        import torch

        if input_shape is None:
            raise ValueError("a PyTorch model needs input_shape, e.g. (1, 3, 640, 640)")
        with torch.no_grad():
            ov_model = ov.convert_model(model.eval().cpu(), example_input=torch.zeros(input_shape),
                                        input=[tuple(input_shape)])
    if input_shape is not None and ov_model.inputs[0].get_partial_shape().is_dynamic:
        ov_model.reshape({ov_model.inputs[0]: ov.PartialShape(list(input_shape))})
    dynamic = [p.get_friendly_name() for p in ov_model.get_parameters() if p.get_partial_shape().is_dynamic]
    if dynamic:
        raise ValueError(f"dynamic inputs {dynamic}: the NPU needs static shapes, pass input_shape")
    return fold_constants(ov_model)


def fold_constants(ov_model: ov.Model) -> ov.Model:
    """Precompute in FP32 every subgraph that does not depend on the input.

    With a static input shape, anchors and sin-cos position embeddings are constants. Left in the
    graph, the NPU recomputes them in FP16 (e.g. 10000 ** x) and gets them wrong.
    """
    manager = passes.Manager()
    manager.register_pass(passes.ConstantFolding())
    manager.run_passes(ov_model)
    return ov_model
