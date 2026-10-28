"""Post-training INT8 quantization with NNCF."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import openvino as ov

from optinpu.devices import core


def quantize_int8(
    ir: str | Path,
    calibration: Iterable,
    transform: Callable[[object], np.ndarray],
    output: str | Path,
    subset_size: int = 300,
    transformer: bool = True,
    ignored_types: Iterable[str] = (),
    ignored_patterns: Iterable[str] = (),
    **nncf_options,
) -> Path:
    """Quantize an FP32/FP16 IR to INT8 and save it (activations INT8, weights INT8).

    `calibration` items (e.g. image paths) are turned into model inputs by `transform`.
    `transformer=True` uses NNCF's transformer scheme (keeps LayerNorm/Softmax inputs in float).
    `ignored_types` / `ignored_patterns` keep matching ops (OpenVINO op types / name regexes) in float.
    `nncf_options` go to nncf.quantize (e.g. preset=nncf.QuantizationPreset.MIXED, fast_bias_correction=False).
    """
    import nncf

    model = core().read_model(ir)
    ignored_types, ignored_patterns = list(ignored_types), list(ignored_patterns)
    ignored = None
    if ignored_types or ignored_patterns:
        ignored = nncf.IgnoredScope(types=ignored_types, patterns=ignored_patterns, validate=False)

    quantized = nncf.quantize(
        model,
        nncf.Dataset(list(calibration), transform),
        subset_size=subset_size,
        model_type=nncf.ModelType.TRANSFORMER if transformer else None,
        ignored_scope=ignored,
        **nncf_options,
    )
    output = Path(output).with_suffix(".xml")
    output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, output, compress_to_fp16=True)
    return output
